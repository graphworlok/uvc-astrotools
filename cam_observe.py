#!/usr/bin/env python3
"""
cam_observe.py -- standalone GUI observing tool for UVC cameras, built on the
cam_characterise.py measurement stack.

Two operating modes, switched explicitly by the user (the tool cannot know if
the lens is capped, so the human says so):

  DARK  -- the lens is capped. Acquire a deep master dark (streaming per-pixel
           mean) and derive a hot-pixel defect map from it, in exactly the
           per-device format cam_manager.py plans and PHD2 auto-loads:
           {vid+pid[_serial]}/master_WxH.npy (uint16 x257) and defects_WxH.txt.
           Existing files for the device+resolution are loaded automatically.

  LIGHT -- the camera can see photons. Frames are continuously captured,
           dark-subtracted, defect-repaired, optionally aligned by phase
           correlation, and stacked. Stars are extracted from the stack
           (robust background + connected components -> centroid/flux/FWHM)
           and the stack can be written to FITS and plate-solved with a local
           astrometry.net installation (solve-field).

Display: live frame and stack side by side, auto-stretched, with extracted
stars overlaid on the stack.

Requires: Linux, v4l2-ctl, numpy, tkinter. Plate solving needs solve-field
(astrometry.net) plus index files on the local machine.

Usage:
  python3 cam_observe.py [--device /dev/video0] [--data-dir .]
"""

import argparse
import glob
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

import numpy as np
import tkinter as tk
from tkinter import ttk

import cam_characterise as cc
from cam_manager import usb_device_tag, query_formats


# ----------------------------------------------------------------------------
# Image processing
# ----------------------------------------------------------------------------

def stretch_to_u8(img, lo_pct=0.5, hi_pct=99.7):
    """Percentile auto-stretch to displayable uint8."""
    lo, hi = np.percentile(img, (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    out = (img - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def pgm_bytes(img8):
    h, w = img8.shape
    return b"P5\n%d %d\n255\n" % (w, h) + img8.tobytes()


def downsample_to_fit(img, max_w, max_h):
    """Integer-stride downsample so the frame fits the canvas. Returns
    (small_image, stride)."""
    h, w = img.shape
    f = max(1, (w + max_w - 1) // max_w, (h + max_h - 1) // max_h)
    return img[::f, ::f], f


def write_fits_u16(path, img):
    """Minimal single-HDU 16-bit FITS writer (BZERO=32768 unsigned trick).
    Keeps the data linear -- what a plate solver wants."""
    data = np.asarray(img, dtype=np.float64)
    mx = data.max()
    if mx <= 0:
        mx = 1.0
    u16 = np.clip(np.round(data * (65535.0 / mx)), 0, 65535).astype(np.uint16)
    h, w = u16.shape
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    2",
        "NAXIS1  = %20d" % w,
        "NAXIS2  = %20d" % h,
        "BZERO   =              32768.0",
        "BSCALE  =                  1.0",
        "COMMENT  written by cam_observe.py",
        "END",
    ]
    hdr = "".join(c.ljust(80) for c in cards)
    hdr = hdr.ljust(((len(hdr) + 2879) // 2880) * 2880).encode("ascii")
    body = (u16.astype(np.int32) - 32768).astype(">i2").tobytes()
    pad = (-len(body)) % 2880
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(body)
        fh.write(b"\x00" * pad)


def phase_shift(ref_small, img_small):
    """Integer (dy, dx) translation of img relative to ref via phase
    correlation on downsampled frames. Cheap and rotation-free -- right for
    short-session drift, not for field rotation."""
    f1 = np.fft.rfft2(ref_small)
    f2 = np.fft.rfft2(img_small)
    cps = f1 * np.conj(f2)
    mag = np.abs(cps)
    mag[mag < 1e-12] = 1e-12
    corr = np.fft.irfft2(cps / mag, s=ref_small.shape)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    if dy > ref_small.shape[0] // 2:
        dy -= ref_small.shape[0]
    if dx > ref_small.shape[1] // 2:
        dx -= ref_small.shape[1]
    return dy, dx


def repair_defects(img, yx):
    """Replace each defect pixel with the median of its in-bounds 8-neighbours
    (the same correction PHD2's defect map applies). img modified in place."""
    if yx is None or len(yx) == 0:
        return img
    h, w = img.shape
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)]
    ys = yx[:, 0]
    xs = yx[:, 1]
    neigh = np.empty((len(offs), len(ys)), dtype=np.float64)
    for k, (dy, dx) in enumerate(offs):
        ny = np.clip(ys + dy, 0, h - 1)
        nx = np.clip(xs + dx, 0, w - 1)
        neigh[k] = img[ny, nx]
    img[ys, xs] = np.median(neigh, axis=0)
    return img


def extract_stars(img, nsigma=5.0, min_pix=2, max_pix=500, max_stars=100):
    """Detect stars on a (dark-subtracted) image: robust background via
    median/MAD, threshold at nsigma, 8-connected components by flood fill on
    the sparse mask, flux-weighted centroids, flux, and a moment-based FWHM.
    Returns a list of dicts sorted by flux descending."""
    bg = float(np.median(img))
    mad = float(np.median(np.abs(img - bg)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(img.std()) or 1e-6
    th = bg + nsigma * sigma
    mask = img > th
    n_above = int(mask.sum())
    if n_above == 0 or n_above > img.size // 10:
        # nothing above threshold, or the "stars" are a third of the frame
        # (clouds / stray light / wrong threshold) -- either way, no star list
        return [], bg, sigma
    visited = np.zeros_like(mask, dtype=bool)
    h, w = img.shape
    stars = []
    cand = np.argwhere(mask)
    for cy, cx in cand:
        if visited[cy, cx]:
            continue
        stack_px = [(cy, cx)]
        visited[cy, cx] = True
        comp = []
        while stack_px and len(comp) <= max_pix:
            y, x = stack_px.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                            and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack_px.append((ny, nx))
        if not (min_pix <= len(comp) <= max_pix):
            continue
        ys = np.array([p[0] for p in comp], dtype=np.float64)
        xs = np.array([p[1] for p in comp], dtype=np.float64)
        fluxes = img[ys.astype(int), xs.astype(int)] - bg
        ftot = float(fluxes.sum())
        if ftot <= 0:
            continue
        cyf = float((ys * fluxes).sum() / ftot)
        cxf = float((xs * fluxes).sum() / ftot)
        var = float(((ys - cyf) ** 2 * fluxes).sum() / ftot
                    + ((xs - cxf) ** 2 * fluxes).sum() / ftot) / 2.0
        fwhm = 2.355 * np.sqrt(max(var, 0.05))
        stars.append({"x": cxf, "y": cyf, "flux": ftot,
                      "npix": len(comp), "fwhm": round(fwhm, 2)})
    stars.sort(key=lambda s: -s["flux"])
    return stars[:max_stars], bg, sigma


# ----------------------------------------------------------------------------
# Plate solving (local astrometry.net)
# ----------------------------------------------------------------------------

def solve_field(stack, solve_path, scale_low, scale_high, timeout=180):
    """Write the stack to FITS in a temp dir and run solve-field on it.
    Returns (ok, message, info) where info carries parsed ra/dec/rotation
    in degrees when available. Blocking -- call from a worker thread."""
    exe = shutil.which(solve_path)
    if not exe:
        return False, f"solve-field not found at '{solve_path}' -- install " \
                      "astrometry.net (and index files) or fix the path", {}
    tmp = tempfile.mkdtemp(prefix="camobs_solve_")
    fits = os.path.join(tmp, "stack.fits")
    try:
        write_fits_u16(fits, stack)
        cmd = [exe, fits, "--overwrite", "--no-plots",
               "--cpulimit", str(timeout - 20), "--downsample", "2",
               "--dir", tmp]
        if scale_low > 0 and scale_high > scale_low:
            cmd += ["--scale-units", "arcsecperpix",
                    "--scale-low", str(scale_low),
                    "--scale-high", str(scale_high)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        solved = os.path.exists(os.path.join(tmp, "stack.solved"))
        info = {}
        m = re.search(r"Field center: \(RA,Dec\) = \(([-0-9.]+), ([-0-9.]+)\)"
                      r" deg", out)
        if m:
            info["ra"] = float(m.group(1))
            info["dec"] = float(m.group(2))
        m = re.search(r"Field rotation angle: up is ([-0-9.]+) degrees"
                      r" ([EW]) of N", out)
        if m:
            info["rot"] = float(m.group(1)) * (1 if m.group(2) == "E" else -1)
        lines = []
        for pat in (r"Field center: \(RA,Dec\) = \([^)]*\) deg",
                    r"Field center: \(RA H:M:S, Dec D:M:S\) = \([^)]*\)",
                    r"Field size: [^\n]*",
                    r"Field rotation angle: [^\n]*"):
            m = re.search(pat, out)
            if m:
                lines.append(m.group(0))
        if solved and lines:
            return True, "\n".join(lines), info
        if solved:
            return True, "solved (no centre line found in output?)", info
        return False, "did not solve -- need more stars, better scale " \
                      "hints, or matching index files", {}
    except subprocess.TimeoutExpired:
        return False, f"solve-field timed out after {timeout}s", {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pole_offset_text(ra, dec):
    """Angular separation of the solved field centre from the celestial pole
    of its hemisphere. With the camera mounted in / coaxial to the RA axis,
    this is the live 'how far is my axis from the pole' polar-alignment
    number (the mechanical-axis refinement -- solving at several RA-axis
    rotations and fitting the rotation centre -- can sit on top of this)."""
    pole = "NCP" if dec >= 0 else "SCP"
    sep_deg = 90.0 - abs(dec)
    if sep_deg < 1.0:
        return f"{sep_deg * 60.0:.1f} arcmin from {pole}"
    return f"{sep_deg:.3f} deg from {pole}"


# ----------------------------------------------------------------------------
# GUI application
# ----------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 480, 360

DARK_BANNER = ("DARK MODE — lens capped: build / hold the master dark",
               "#1c1c1c", "#e0e0e0")
LIGHT_BANNER = ("LIGHT MODE — capturing photons: stacking and extracting",
                "#143a63", "#ffffff")


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        root.title("cam_observe — UVC dark/defect + live stack + plate solve")

        self.q = queue.Queue()
        self.stop_evt = threading.Event()
        self.worker = None          # the active capture thread (dark or light)

        # camera state (set by Probe)
        self.fourcc = None
        self.sizes = []             # [(w, h), ...]
        self.exp_rng = {}
        self.dev_tag = ""

        # calibration state
        self.master_dark = None     # float64 luma 0-255 at current resolution
        self.master_exp = None      # exposure units the master was built at
        self.defects = None         # np.ndarray [[y, x], ...]

        # stack state (touched only by the light worker + reset on UI thread
        # while no worker is running)
        self.stack_sum = None
        self.stack_n = 0
        self.align_ref = None
        self.stars = []

        self._photo_live = None     # keep references or Tk drops the images
        self._photo_stack = None

        self.solving = False
        self.last_solve_t = 0.0

        self._build_ui()
        self._set_mode_banner()
        root.after(100, self._poll)

    # ---------------- UI construction ----------------

    def _build_ui(self):
        self.banner = tk.Label(self.root, font=("TkDefaultFont", 12, "bold"),
                               pady=6)
        self.banner.pack(fill="x")

        disp = tk.Frame(self.root)
        disp.pack(fill="both", expand=True, padx=4, pady=4)
        lf = tk.LabelFrame(disp, text="Live frame")
        lf.pack(side="left", padx=2)
        self.canvas_live = tk.Canvas(lf, width=CANVAS_W, height=CANVAS_H,
                                     bg="black", highlightthickness=0)
        self.canvas_live.pack()
        sf = tk.LabelFrame(disp, text="Stack / master dark")
        sf.pack(side="left", padx=2)
        self.canvas_stack = tk.Canvas(sf, width=CANVAS_W, height=CANVAS_H,
                                      bg="black", highlightthickness=0)
        self.canvas_stack.pack()

        ctl = tk.Frame(self.root)
        ctl.pack(fill="x", padx=4)

        # row 0: device / probe / resolution / exposure
        r0 = tk.Frame(ctl); r0.pack(fill="x", pady=2)
        tk.Label(r0, text="Device:").pack(side="left")
        devs = sorted(glob.glob("/dev/video*")) or [self.args.device]
        self.var_dev = tk.StringVar(value=self.args.device if
                                    self.args.device in devs else devs[0])
        self.cmb_dev = ttk.Combobox(r0, textvariable=self.var_dev,
                                    values=devs, width=14)
        self.cmb_dev.pack(side="left", padx=2)
        self.btn_probe = tk.Button(r0, text="Probe", command=self.on_probe)
        self.btn_probe.pack(side="left", padx=4)
        tk.Label(r0, text="Resolution:").pack(side="left")
        self.var_res = tk.StringVar()
        self.cmb_res = ttk.Combobox(r0, textvariable=self.var_res, width=11,
                                    state="readonly")
        self.cmb_res.pack(side="left", padx=2)
        self.cmb_res.bind("<<ComboboxSelected>>", lambda e: self.on_res_change())
        tk.Label(r0, text="Exposure (100µs units):").pack(side="left")
        self.var_exp = tk.IntVar(value=1024)
        self.spn_exp = tk.Spinbox(r0, from_=1, to=65535, width=7,
                                  textvariable=self.var_exp)
        self.spn_exp.pack(side="left", padx=2)
        self.lbl_exp_rng = tk.Label(r0, text="")
        self.lbl_exp_rng.pack(side="left")

        # row 1: mode + dark controls
        r1 = tk.Frame(ctl); r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Mode:").pack(side="left")
        self.var_mode = tk.StringVar(value="dark")
        tk.Radiobutton(r1, text="DARK (lens capped)", value="dark",
                       variable=self.var_mode,
                       command=self.on_mode_change).pack(side="left")
        tk.Radiobutton(r1, text="LIGHT (capturing photons)", value="light",
                       variable=self.var_mode,
                       command=self.on_mode_change).pack(side="left", padx=4)
        tk.Label(r1, text="   Dark frames:").pack(side="left")
        self.var_dark_n = tk.IntVar(value=64)
        tk.Spinbox(r1, from_=8, to=1024, width=5,
                   textvariable=self.var_dark_n).pack(side="left")
        self.btn_dark = tk.Button(r1, text="Acquire master dark",
                                  command=self.on_acquire_dark)
        self.btn_dark.pack(side="left", padx=4)
        self.prg = ttk.Progressbar(r1, length=140, mode="determinate")
        self.prg.pack(side="left", padx=4)
        self.lbl_dark = tk.Label(r1, text="no master dark")
        self.lbl_dark.pack(side="left")

        # row 2: light controls
        r2 = tk.Frame(ctl); r2.pack(fill="x", pady=2)
        self.btn_start = tk.Button(r2, text="Start capture",
                                   command=self.on_start_light,
                                   state="disabled")
        self.btn_start.pack(side="left")
        self.btn_stop = tk.Button(r2, text="Stop", command=self.on_stop,
                                  state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.var_sub = tk.BooleanVar(value=True)
        self.var_fix = tk.BooleanVar(value=True)
        self.var_align = tk.BooleanVar(value=True)
        tk.Checkbutton(r2, text="subtract dark",
                       variable=self.var_sub).pack(side="left")
        tk.Checkbutton(r2, text="repair defects",
                       variable=self.var_fix).pack(side="left")
        tk.Checkbutton(r2, text="align (phase corr.)",
                       variable=self.var_align).pack(side="left")
        self.btn_reset = tk.Button(r2, text="Reset stack",
                                   command=self.on_reset_stack)
        self.btn_reset.pack(side="left", padx=8)
        self.lbl_stack = tk.Label(r2, text="stack: 0 frames")
        self.lbl_stack.pack(side="left", padx=4)
        self.lbl_stars = tk.Label(r2, text="stars: -")
        self.lbl_stars.pack(side="left", padx=4)

        # row 3: solving
        r3 = tk.Frame(ctl); r3.pack(fill="x", pady=2)
        tk.Label(r3, text="solve-field:").pack(side="left")
        self.var_solver = tk.StringVar(value=self.args.solver)
        tk.Entry(r3, textvariable=self.var_solver, width=18).pack(side="left")
        tk.Label(r3, text="scale arcsec/px low/high:").pack(side="left")
        self.var_sc_lo = tk.DoubleVar(value=0.0)
        self.var_sc_hi = tk.DoubleVar(value=0.0)
        tk.Entry(r3, textvariable=self.var_sc_lo, width=6).pack(side="left")
        tk.Entry(r3, textvariable=self.var_sc_hi, width=6).pack(side="left")
        self.btn_solve = tk.Button(r3, text="Plate solve stack",
                                   command=self.on_solve, state="disabled")
        self.btn_solve.pack(side="left", padx=6)
        self.var_autosolve = tk.BooleanVar(value=False)
        tk.Checkbutton(r3, text="auto-solve every",
                       variable=self.var_autosolve).pack(side="left")
        self.var_solve_interval = tk.IntVar(value=20)
        tk.Spinbox(r3, from_=5, to=600, width=4,
                   textvariable=self.var_solve_interval).pack(side="left")
        tk.Label(r3, text="s, min stars:").pack(side="left")
        self.var_solve_minstars = tk.IntVar(value=10)
        tk.Spinbox(r3, from_=3, to=200, width=4,
                   textvariable=self.var_solve_minstars).pack(side="left")
        self.btn_fits = tk.Button(r3, text="Save stack (FITS + npy)",
                                  command=self.on_save_fits, state="disabled")
        self.btn_fits.pack(side="left", padx=6)

        # row 4: sky position / polar alignment readout
        r4 = tk.Frame(ctl); r4.pack(fill="x", pady=2)
        self.lbl_solve = tk.Label(r4, text="sky position: (not solved)",
                                  font=("TkDefaultFont", 10, "bold"),
                                  anchor="w")
        self.lbl_solve.pack(side="left", fill="x")

        # log
        self.txt = tk.Text(self.root, height=8, state="disabled",
                           font=("TkFixedFont", 9))
        self.txt.pack(fill="x", padx=4, pady=4)

    # ---------------- helpers ----------------

    def log(self, msg):
        self.txt.configure(state="normal")
        self.txt.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _set_mode_banner(self):
        text, bg, fg = DARK_BANNER if self.var_mode.get() == "dark" \
            else LIGHT_BANNER
        self.banner.configure(text=text, bg=bg, fg=fg)

    def current_size(self):
        m = re.match(r"(\d+)x(\d+)", self.var_res.get() or "")
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    def data_dir(self):
        d = self.args.data_dir
        return os.path.join(d, self.dev_tag) if self.dev_tag else d

    def make_capture(self):
        w, h = self.current_size()
        cap = cc.Capture(self.var_dev.get(), w, h, self.fourcc)
        cap.exposure = int(self.var_exp.get())
        return cap

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _show(self, canvas, img8, which, stars=None, stride=1):
        photo = tk.PhotoImage(data=pgm_bytes(img8))
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor="nw")
        if stars:
            for s in stars[:50]:
                x, y = s["x"] / stride, s["y"] / stride
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6,
                                   outline="#00ff60")
        if which == "live":
            self._photo_live = photo
        else:
            self._photo_stack = photo

    # ---------------- device probing / calibration files ----------------

    def on_probe(self):
        dev = self.var_dev.get()
        try:
            ranked = cc.rank_formats(cc.enum_formats(dev))
        except OSError as e:
            self.log(f"cannot open {dev}: {e}")
            return
        best = cc.best_measurement_format(ranked)
        if best is None:
            self.log(f"{dev}: no uncompressed format -- unusable for "
                     "measurement (only lossy codecs offered)")
            return
        self.fourcc = best["fourcc"].strip()
        fmts = query_formats(dev)
        sizes = []
        for f in fmts:
            if f["fourcc"] == self.fourcc:
                sizes = [(s["w"], s["h"]) for s in f["sizes"]]
        if not sizes:
            w, h = 640, 480
            sizes = [(w, h)]
        self.sizes = sorted(set(sizes))
        self.cmb_res["values"] = [f"{w}x{h}" for w, h in self.sizes]
        self.var_res.set(f"{self.sizes[0][0]}x{self.sizes[0][1]}")
        self.exp_rng = cc.get_ctrl_range(dev, "exposure_time_absolute")
        if self.exp_rng:
            self.lbl_exp_rng.configure(
                text=f"[{self.exp_rng.get('min', '?')}.."
                     f"{self.exp_rng.get('max', '?')}]")
            if self.exp_rng.get("max"):
                self.var_exp.set(self.exp_rng["max"])
        self.dev_tag = usb_device_tag(dev)
        self.log(f"{dev}: format {self.fourcc}, "
                 f"{len(self.sizes)} resolutions, "
                 f"device tag {self.dev_tag or '(none)'}")
        self.on_res_change()
        self.btn_start.configure(state="normal")

    def on_res_change(self):
        self.master_dark = None
        self.master_exp = None
        self.defects = None
        self.on_reset_stack()
        w, h = self.current_size()
        if w is None:
            return
        d = self.data_dir()
        mp = os.path.join(d, f"master_{w}x{h}.npy")
        dp = os.path.join(d, f"defects_{w}x{h}.txt")
        jp = os.path.join(d, f"dark_meta_{w}x{h}.json")
        if os.path.exists(mp):
            try:
                m = np.load(mp)
                if m.shape == (h, w):
                    self.master_dark = m.astype(np.float64) / 257.0
                    self.log(f"loaded master dark {mp}")
                else:
                    self.log(f"{mp} shape {m.shape} does not match {w}x{h}; "
                             "ignored")
            except Exception as e:
                self.log(f"could not load {mp}: {e}")
        if os.path.exists(dp):
            coords = []
            with open(dp) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        coords.append((int(parts[1]), int(parts[0])))  # x y -> y x
            self.defects = np.array(coords, dtype=int) if coords else None
            self.log(f"loaded {0 if self.defects is None else len(self.defects)}"
                     f" defects from {dp}")
        if os.path.exists(jp):
            try:
                meta = json.load(open(jp))
                self.master_exp = meta.get("exposure_units")
            except Exception:
                pass
        self._update_dark_label()
        if self.master_dark is not None:
            small, _ = downsample_to_fit(self.master_dark, CANVAS_W, CANVAS_H)
            self._show(self.canvas_stack, stretch_to_u8(small), "stack")

    def _update_dark_label(self):
        if self.master_dark is None:
            self.lbl_dark.configure(text="no master dark")
        else:
            nd = 0 if self.defects is None else len(self.defects)
            exp = f" @ exp {self.master_exp}" if self.master_exp else ""
            self.lbl_dark.configure(
                text=f"master dark ready{exp}, {nd} defects")

    # ---------------- mode / actions ----------------

    def on_mode_change(self):
        if self.var_mode.get() == "dark" and self.busy():
            self.on_stop()
        self._set_mode_banner()

    def on_reset_stack(self):
        if self.busy():
            return
        self.stack_sum = None
        self.stack_n = 0
        self.align_ref = None
        self.stars = []
        self.lbl_stack.configure(text="stack: 0 frames")
        self.lbl_stars.configure(text="stars: -")
        self.btn_solve.configure(state="disabled")
        self.btn_fits.configure(state="disabled")

    def on_acquire_dark(self):
        if self.busy() or self.fourcc is None:
            self.log("probe a device first / wait for the current task")
            return
        if self.var_mode.get() != "dark":
            self.log("switch to DARK mode (and cap the lens) first")
            return
        n = int(self.var_dark_n.get())
        self.stop_evt.clear()
        self.btn_dark.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.worker = threading.Thread(target=self._dark_worker, args=(n,),
                                       daemon=True)
        self.worker.start()

    def on_start_light(self):
        if self.busy() or self.fourcc is None:
            self.log("probe a device first / wait for the current task")
            return
        if self.var_mode.get() != "light":
            self.var_mode.set("light")
            self._set_mode_banner()
        self.on_reset_stack()
        exp = int(self.var_exp.get())
        if self.master_dark is not None and self.master_exp \
                and exp != self.master_exp and self.var_sub.get():
            self.log(f"NOTE: master dark was built at exposure "
                     f"{self.master_exp}, capturing at {exp} -- "
                     "subtraction will be biased")
        self.stop_evt.clear()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.worker = threading.Thread(target=self._light_worker, daemon=True)
        self.worker.start()

    def on_stop(self):
        self.stop_evt.set()

    def on_save_fits(self):
        if self.stack_sum is None or self.stack_n == 0:
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        w, h = self.current_size()
        os.makedirs(self.data_dir(), exist_ok=True)
        base = os.path.join(self.data_dir(), f"stack_{w}x{h}_{ts}")
        stack = self.stack_sum / self.stack_n
        write_fits_u16(base + ".fits", stack)
        # float64 mean stack, unquantised, for offline noise-floor analysis
        np.save(base + ".npy", stack)
        self.log(f"stack written to {base}.fits and .npy "
                 f"({self.stack_n} frames)")

    def on_solve(self):
        self._trigger_solve(auto=False)

    def _trigger_solve(self, auto):
        if self.stack_sum is None or self.stack_n == 0 or self.solving:
            return
        stack = self.stack_sum / self.stack_n
        if self.master_dark is None and not auto:
            self.log("solving an unsubtracted stack -- hot pixels may "
                     "masquerade as stars")
        self.solving = True
        self.last_solve_t = time.monotonic()
        self.btn_solve.configure(state="disabled")
        self.log("plate solving..." + (" (auto)" if auto else ""))
        t = threading.Thread(
            target=lambda: self.q.put(
                ("solved", solve_field(stack, self.var_solver.get(),
                                       float(self.var_sc_lo.get()),
                                       float(self.var_sc_hi.get())))),
            daemon=True)
        t.start()

    # ---------------- workers (no Tk calls in here) ----------------

    def _dark_worker(self, total):
        dev = self.var_dev.get()
        exp = int(self.var_exp.get())
        cap = self.make_capture()
        try:
            cc.set_ctrl(dev, "auto_exposure", 1)
            cc.set_ctrl(dev, "exposure_time_absolute", exp)
            mean = None
            count = 0
            while count < total and not self.stop_evt.is_set():
                burst = min(16, total - count)
                path, _, _ = cap.capture_run(burst, discard=1,
                                             timeout=30.0, verbose=False)
                for fr in cap.iter_luma(path, burst, discard=1):
                    count += 1
                    if mean is None:
                        mean = np.zeros_like(fr)
                    mean += (fr - mean) / count
                try:
                    os.remove(path)
                except OSError:
                    pass
                # downsampled COPY: the UI must not read the array this loop
                # keeps mutating, and a full-res frame is queue bloat anyway
                small, _ = downsample_to_fit(mean, CANVAS_W, CANVAS_H)
                self.q.put(("progress", count, total, small.copy()))
            if mean is None:
                self.q.put(("log", "dark acquisition produced no frames"))
                return
            hot = cc.hot_pixel_mask(mean)
            coords = np.argwhere(hot)
            self.q.put(("dark_done", mean, coords, exp, count))
        except Exception as e:
            self.q.put(("log", f"dark acquisition failed: {e}"))
        finally:
            cap.close()
            self.q.put(("worker_done",))

    def _light_worker(self):
        dev = self.var_dev.get()
        exp = int(self.var_exp.get())
        sub = self.var_sub.get() and self.master_dark is not None
        fix = self.var_fix.get() and self.defects is not None \
            and len(self.defects) > 0
        align = self.var_align.get()
        master = self.master_dark
        defects = self.defects
        cap = self.make_capture()
        burst = 6
        try:
            cc.set_ctrl(dev, "auto_exposure", 1)
            cc.set_ctrl(dev, "exposure_time_absolute", exp)
            while not self.stop_evt.is_set():
                path, _, _ = cap.capture_run(burst, discard=1,
                                             timeout=60.0, verbose=False)
                last = None
                for fr in cap.iter_luma(path, burst, discard=1):
                    if self.stop_evt.is_set():
                        break
                    proc = fr
                    if sub:
                        proc = np.clip(fr - master, 0.0, None)
                    if fix:
                        proc = repair_defects(proc.copy(), defects)
                    if align:
                        small, f = downsample_to_fit(proc, 256, 256)
                        if self.align_ref is None:
                            self.align_ref = small
                        else:
                            dy, dx = phase_shift(self.align_ref, small)
                            if dy or dx:
                                proc = np.roll(proc, (dy * f, dx * f),
                                               axis=(0, 1))
                    if self.stack_sum is None:
                        self.stack_sum = proc.astype(np.float64)
                        self.stack_n = 1
                    else:
                        self.stack_sum += proc
                        self.stack_n += 1
                    last = fr
                try:
                    os.remove(path)
                except OSError:
                    pass
                if last is None:
                    continue
                stack = self.stack_sum / self.stack_n
                stars, bg, sigma = extract_stars(stack)
                self.stars = stars
                self.q.put(("light_update", last, stack, stars,
                            self.stack_n, cap.last_fps, bg, sigma))
        except Exception as e:
            self.q.put(("log", f"capture failed: {e}"))
        finally:
            cap.close()
            self.q.put(("worker_done",))

    # ---------------- UI-thread event pump ----------------

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "log":
            self.log(msg[1])
        elif kind == "progress":
            _, count, total, small = msg
            self.prg["maximum"] = total
            self.prg["value"] = count
            self._show(self.canvas_stack, stretch_to_u8(small), "stack")
        elif kind == "dark_done":
            _, mean, coords, exp, count = msg
            self.master_dark = mean
            self.master_exp = exp
            self.defects = coords if len(coords) else None
            self._save_dark(mean, coords, exp, count)
            self._update_dark_label()
            self.log(f"master dark: {count} frames, "
                     f"{len(coords)} hot pixels")
        elif kind == "light_update":
            _, live, stack, stars, n, fps, bg, sigma = msg
            small, _ = downsample_to_fit(live, CANVAS_W, CANVAS_H)
            self._show(self.canvas_live, stretch_to_u8(small), "live")
            ssmall, stride = downsample_to_fit(stack, CANVAS_W, CANVAS_H)
            self._show(self.canvas_stack, stretch_to_u8(ssmall), "stack",
                       stars=stars, stride=stride)
            # noise-floor readout: measured single-frame sigma vs stack sigma
            # against the theoretical sqrt(N) gain (computed on the
            # downsampled frame -- plenty for a gain estimate)
            b1 = float(np.median(small))
            s1 = 1.4826 * float(np.median(np.abs(small - b1)))
            gain = (s1 / sigma) if sigma > 0 else 0.0
            self.lbl_stack.configure(
                text=f"stack: {n} frames @ {fps:.1f} fps | noise "
                     f"×{gain:.1f} down (theory ×{np.sqrt(n):.1f})")
            if stars:
                s0 = stars[0]
                self.lbl_stars.configure(
                    text=f"stars: {len(stars)} (brightest flux "
                         f"{s0['flux']:.0f}, FWHM {s0['fwhm']}px) "
                         f"bg {bg:.1f} σ {sigma:.2f}")
            else:
                self.lbl_stars.configure(
                    text=f"stars: 0  bg {bg:.1f} σ {sigma:.2f}")
            self.btn_solve.configure(
                state="disabled" if self.solving else "normal")
            self.btn_fits.configure(state="normal")
            if (self.var_autosolve.get() and not self.solving and stars
                    and len(stars) >= int(self.var_solve_minstars.get())
                    and time.monotonic() - self.last_solve_t
                        > float(self.var_solve_interval.get())):
                self._trigger_solve(auto=True)
        elif kind == "solved":
            ok, text, info = msg[1]
            self.solving = False
            self.log(("SOLVED:\n" if ok else "solve failed: ") + text)
            if ok and "ra" in info:
                pos = (f"sky position: RA {info['ra']:.4f}° "
                       f"Dec {info['dec']:+.4f}°")
                if "rot" in info:
                    pos += f"  rot {info['rot']:+.1f}°"
                pos += "  |  " + pole_offset_text(info["ra"], info["dec"])
                self.lbl_solve.configure(text=pos, fg="#006000")
            elif not ok:
                self.lbl_solve.configure(
                    text="sky position: " + text.splitlines()[0],
                    fg="#802000")
            self.btn_solve.configure(state="normal")
        elif kind == "worker_done":
            self.btn_dark.configure(state="normal")
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.prg["value"] = 0

    def _save_dark(self, mean, coords, exp, count):
        w, h = self.current_size()
        d = self.data_dir()
        os.makedirs(d, exist_ok=True)
        mp = os.path.join(d, f"master_{w}x{h}.npy")
        u16 = np.clip(np.round(mean * 257.0), 0, 65535).astype(np.uint16)
        np.save(mp, u16)
        dp = os.path.join(d, f"defects_{w}x{h}.txt")
        with open(dp, "w") as fh:
            fh.write("# PHD2 Defect Map v1\n")
            fh.write(f"# cam_observe.py {self.var_dev.get()} {w}x{h}\n")
            fh.write(f"# Exposure: {exp} units\n")
            fh.write(f"# Defect count: {len(coords)}\n")
            for y, x in coords:
                fh.write(f"{x} {y}\n")
        jp = os.path.join(d, f"dark_meta_{w}x{h}.json")
        with open(jp, "w") as fh:
            json.dump({"exposure_units": exp, "frames": count,
                       "hot_pixels": int(len(coords)),
                       "timestamp_utc":
                           datetime.now(timezone.utc).isoformat()}, fh,
                      indent=2)
        self.log(f"saved {mp}, {dp}, {jp}")


def main():
    ap = argparse.ArgumentParser(
        description="GUI: UVC dark/defect map + live stacking + plate solve")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--data-dir", default=".",
                    help="base directory for per-device calibration files "
                         "(default: current directory, same layout as "
                         "cam_manager.py)")
    ap.add_argument("--solver", default="solve-field",
                    help="path to astrometry.net's solve-field")
    args = ap.parse_args()

    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
