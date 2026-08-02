#!/usr/bin/env python3
"""
cam_characterise.py  --  noise / codec quality characterisation for an
arbitrary (unspecified) UVC camera.

Goals
-----
1. Enumerate the formats the camera *actually* delivers and rank them by how
   little they touch the data (raw/uncompressed > subsampled > lossy). The
   "best quality codec" for measurement is the top uncompressed format that
   streams without dropping frames -- which is VERIFIED here, not trusted from
   the descriptor (a device may advertise bandwidth it physically cannot meet).

2. Photon-transfer curve (PTC) on flat-field pairs -> system gain (e-/ADU),
   read noise (e-), full-well. Measured in the bright mid-range where it is
   well-conditioned; these are exposure-independent constants.

3. Dark-noise characterisation, weighted HARD to the longest exposure the
   camera offers, because that is the operating point that matters:
     - master dark (deep stack)  -> the FIXED-PATTERN part you can subtract
     - residual after subtraction -> the TEMPORAL dark noise you CANNOT remove
     - dark-current slope vs exposure, with a linearity check that flags
       firmware faking long exposure (gain-boost / frame-sum instead of real
       integration).

It captures via V4L2 (uncompressed formats only -- compressed formats are
refused for measurement because the codec destroys exactly the noise you are
trying to quantify). Capture shells out to v4l2-ctl --stream-mmap (see the
Capture class for why); no OpenCV required.

The ONE exception, and it is not a measurement: the framerate-vs-exposure test
(--fps-probe, and the probe --dark runs by default) uses MJPEG deliberately.
That test reads frame TIMESTAMPS, never pixel values, so a lossy codec cannot
corrupt it -- while the smaller payload lifts the USB bandwidth ceiling far
enough that the frame period is set by integration rather than by the bus,
which is the only condition under which the test says anything at all. In an
uncompressed format at any useful resolution the framerate is pinned flat and
the verdict is forever "indeterminate (bandwidth-limited)". See FpsProbe: it
has no decode path, structurally, so nothing it captures can reach a mean, a
noise figure, a master dark or a defect map.

USAGE (typical, run as root for v4l2 control access):
  # 1. just list + rank formats and pick the measurement format
  sudo ./cam_characterise.py --device /dev/video0 --list

  # 2. PTC: vary the LIGHT level by hand (lamp/screen brightness), the script
  #    prompts you between levels. Run at a fixed MODERATE exposure.
  sudo ./cam_characterise.py --device /dev/video0 --ptc \
       --width 1280 --height 720 --exposure 200

  # 3. Dark characterisation (CAP THE LENS). Spends most frames at max exposure.
  sudo ./cam_characterise.py --device /dev/video0 --dark \
       --width 1280 --height 720 \
       --dark-frames 128 --exposure-max 8192

  # 4. everything, writing a json report
  sudo ./cam_characterise.py --device /dev/video0 --ptc --dark \
       --report report.json

Notes specific to long exposure:
  * UVC frame interval (fps) is decoupled from integration time. Exposure is
    set via V4L2 'exposure_time_absolute' (units of 100us on standard UVC).
    --exposure / --exposure-max are in those raw control units.
  * Auto-exposure, AGC and any in-firmware denoise MUST be off or the PTC/dark
    maths (which assume a stationary process) are invalid. The script disables
    what it can and warns about what it can't.
  * Dark current ~doubles every 6-8 C. A master dark is only valid at the
    temperature it was taken. The script timestamps stacks and warns; if your
    rig has a temperature readout, log it alongside.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

# fcntl is Linux-only and only the format-enumeration ioctl needs it. Importing
# it at module scope made the whole file unimportable elsewhere -- including the
# pure-analysis layer (--compare, the fits, the guards), which touches no
# hardware and is explicitly meant to run on a machine with no camera attached.
try:
    import fcntl
except ImportError:                     # not Linux: capture is unavailable,
    fcntl = None                        # analysis is not

# ----------------------------------------------------------------------------
# Minimal V4L2 ioctl layer (avoids depending on python-v4l2 packaging).
# ----------------------------------------------------------------------------

# Linux _IOC encoding. Hardcoding the request codes is fragile: the size
# field must exactly match sizeof(struct), or the ioctl marshals the wrong
# number of bytes and corrupts the readback (symptom: garbage fourcc). So we
# compute every code from the actual ctypes struct size at runtime.
_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS, _IOC_DIRBITS = 8, 8, 14, 2
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2  # ARM/x86 generic encoding

def _IOC(d, t, nr, size):
    return ((d << _IOC_DIRSHIFT) | (ord(t) << _IOC_TYPESHIFT) |
            (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT))

def _IOWR(t, nr, structtype): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, ctypes.sizeof(structtype))

# Only format enumeration is done via ioctl; capture itself shells out to
# v4l2-ctl (see Capture), so no other request codes or structs are needed.

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1

def fourcc(a, b, c, d):
    return (ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24))

def fourcc_str(v):
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))

# Compression / fidelity ranking. Lower rank number == less processing == better
# for noise measurement. Anything not listed is treated as unknown/lossy.
FORMAT_FIDELITY = {
    # raw bayer variants (best, rarely on UVC gadgets)
    "BA81": (0, "raw bayer 8"),  "pBAA": (0, "raw bayer packed"),
    "RG10": (0, "raw bayer 10"), "RG12": (0, "raw bayer 12"),
    "BG10": (0, "raw bayer 10"), "BYR2": (0, "raw bayer"),
    # true mono: purer than YUV (no chroma path at all) -- the ideal
    # measurement format on mono guide-camera gadgets
    "GREY": (0, "uncompressed 8-bit mono"),
    "Y800": (0, "uncompressed 8-bit mono"),
    "Y8  ": (0, "uncompressed 8-bit mono"),
    "Y16 ": (0, "uncompressed 16-bit mono"),
    # uncompressed YUV: full-res luma, this is what we want
    "YUYV": (1, "uncompressed 4:2:2, 16bpp"),
    "UYVY": (1, "uncompressed 4:2:2, 16bpp"),
    "YUY2": (1, "uncompressed 4:2:2, 16bpp"),
    "NV12": (2, "uncompressed 4:2:0, 12bpp (luma full-res)"),
    "NV21": (2, "uncompressed 4:2:0, 12bpp (luma full-res)"),
    "I420": (2, "uncompressed 4:2:0, 12bpp (luma full-res)"),
    # lossy: refused for measurement
    "MJPG": (9, "LOSSY motion-jpeg -- unusable for noise"),
    "JPEG": (9, "LOSSY jpeg -- unusable for noise"),
    "H264": (9, "LOSSY h.264 -- unusable for noise"),
    "HEVC": (9, "LOSSY hevc -- unusable for noise"),
    "H265": (9, "LOSSY hevc -- unusable for noise"),
}

class v4l2_fmtdesc(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("description", ctypes.c_char * 32),
                ("pixelformat", ctypes.c_uint32), ("mbus_code", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 3)]


# Computed from the actual ctypes struct size so the encoded size always
# matches sizeof(struct) on THIS platform/kernel.
VIDIOC_ENUM_FMT = _IOWR('V', 2, v4l2_fmtdesc)


def xioctl(fd, req, arg):
    """Issue an ioctl, operating in place for ctypes structs.

    fcntl.ioctl with a ctypes Structure and the default path can pass a
    non-writable / copied buffer; enumeration needs the kernel to write
    fields (pixelformat, description...) back into OUR struct. So we wrap
    the struct's own memory in a writable buffer and pass it with
    mutate_flag=True so the kernel's writes land in `arg` directly.
    """
    if fcntl is None:
        raise RuntimeError(
            "V4L2 capture requires Linux (the fcntl module is unavailable on "
            "this platform). Analysis modes such as --compare still work.")
    buf = (ctypes.c_char * ctypes.sizeof(arg)).from_address(ctypes.addressof(arg))
    while True:
        try:
            fcntl.ioctl(fd, req, buf, True)
            return 0
        except OSError as e:
            if e.errno == 4:  # EINTR
                continue
            raise


# ----------------------------------------------------------------------------
# Control handling via v4l2-ctl (simplest portable path for named controls)
# ----------------------------------------------------------------------------

def v4l2ctl(device, args):
    return subprocess.run(["v4l2-ctl", "-d", device, *args],
                          capture_output=True, text=True)

def list_controls(device):
    r = v4l2ctl(device, ["--list-ctrls"])
    return r.stdout

def set_ctrl(device, name, value):
    r = v4l2ctl(device, ["-c", f"{name}={value}"])
    return r.returncode == 0, r.stderr.strip()

def get_ctrl(device, name):
    r = v4l2ctl(device, ["-C", name])
    # output like "exposure_time_absolute: 200"
    if ":" in r.stdout:
        try:
            return int(r.stdout.split(":")[1].strip())
        except ValueError:
            return None
    return None

def get_ctrl_range(device, name):
    """Return {min,max,step,default} for a control by parsing --list-ctrls,
    or {} if not found. Lets verdicts scale to the actual device range rather
    than any hard-coded assumption."""
    txt = list_controls(device)
    for line in txt.splitlines():
        if re.match(rf"\s*{re.escape(name)}\s+0x", line):
            d = {}
            for k in ("min", "max", "step", "default"):
                m = re.search(rf"{k}=(-?\d+)", line)
                if m:
                    d[k] = int(m.group(1))
            return d
    return {}

def prepare_manual(device, exposure=None, gain=None, gamma=None,
                   brightness=None, contrast=None, sharpness=None,
                   saturation=None, verbose=True):
    """Force the camera out of every adaptive mode we can reach, so the noise
    process is stationary. Warn about anything that won't move.

    gamma/brightness/contrast/sharpness/saturation: if given, set them. For
    clean noise measurement you want a linear, unsharpened, un-stretched path
    (gamma=100, sharpness=min, contrast/brightness neutral); pass those values
    explicitly. None = leave the control untouched.
    """
    notes = []
    ctrls = list_controls(device)

    # auto-exposure: on UVC, '1' = Manual Mode, '3' = Aperture Priority(auto)
    if "auto_exposure" in ctrls:
        ok, err = set_ctrl(device, "auto_exposure", 1)
        notes.append(f"auto_exposure->manual: {'ok' if ok else 'FAILED '+err}")
    # white balance auto off
    for c in ("white_balance_temperature_auto", "white_balance_automatic"):
        if c in ctrls:
            set_ctrl(device, c, 0)
            notes.append(f"{c}->0")
    # focus auto off (irrelevant for fixed lens but harmless)
    if "focus_automatic_continuous" in ctrls:
        set_ctrl(device, "focus_automatic_continuous", 0)
    # power line freq / backlight comp can inject periodic structure; neutralise
    if "power_line_frequency" in ctrls:
        set_ctrl(device, "power_line_frequency", 0)
        notes.append("power_line_frequency->disabled")

    # processing-path controls that distort the noise measurement. Each only
    # fires if the user asked AND the control exists; we report misses.
    for name, val in (("gamma", gamma), ("brightness", brightness),
                      ("contrast", contrast), ("sharpness", sharpness),
                      ("saturation", saturation)):
        if val is None:
            continue
        if name in ctrls:
            ok, err = set_ctrl(device, name, val)
            rb = get_ctrl(device, name)
            notes.append(f"{name}={val} (readback={rb}): "
                         f"{'ok' if ok else 'FAILED '+err}")
        else:
            notes.append(f"{name}={val} requested but control absent on device")

    if gain is not None and "gain" in ctrls:
        ok, err = set_ctrl(device, "gain", gain)
        rb = get_ctrl(device, "gain")
        notes.append(f"gain={gain} (readback={rb}): {'ok' if ok else 'FAILED '+err}")
    elif gain is not None:
        notes.append("gain requested but control absent on device")
    if exposure is not None and "exposure_time_absolute" in ctrls:
        ok, err = set_ctrl(device, "exposure_time_absolute", exposure)
        rb = get_ctrl(device, "exposure_time_absolute")
        notes.append(f"exposure_time_absolute={exposure} (readback={rb}): "
                     f"{'ok' if ok else 'FAILED '+err}")

    # things we CANNOT control are the real risk -- flag them loudly
    if "exposure_dynamic_framerate" in ctrls:
        notes.append("WARNING: exposure_dynamic_framerate present -- if it is "
                     "read-only/on, the firmware owns frame timing and may "
                     "perturb integration. Long-exposure linearity check below "
                     "will catch synthetic exposure.")
    if verbose:
        for n in notes:
            print("   ", n)
    return notes


# ----------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------

class Capture:
    """Frame source backed by v4l2-ctl --stream-to.

    We do NOT issue V4L2 ioctls ourselves. On this device (and minimal UVC
    gadgets generally) direct MMAP QUERYBUF from our own fd returns ENOTTY,
    while v4l2-ctl streams perfectly in its single-process path. So we drive
    v4l2-ctl as the capture engine and do all analysis on the raw frames it
    writes out. Returns the LUMA plane as a 2-D float64 array (luma only;
    chroma is subsampled and irrelevant to the sensor noise floor).

    Critically, format + exposure + capture happen in ONE v4l2-ctl invocation,
    so there is never a second opener racing the streaming fd.
    """

    def __init__(self, device, width, height, pixfmt_str, nbuf=4):
        self.device = device
        self.w, self.h = width, height
        # fourcc may carry trailing spaces ("Y16 "); strip once so comparisons
        # and the v4l2-ctl pixelformat argument are uniform
        self.pixfmt_str = pixfmt_str.strip()
        if self.pixfmt_str in ("YUYV", "YUY2", "UYVY"):
            self.frame_bytes = width * height * 2
            self.bytesperline = width * 2
        elif self.pixfmt_str in ("NV12", "NV21", "I420"):
            self.frame_bytes = width * height * 3 // 2
            self.bytesperline = width
        elif self.pixfmt_str in ("GREY", "Y800", "Y8"):
            self.frame_bytes = width * height
            self.bytesperline = width
        elif self.pixfmt_str == "Y16":
            self.frame_bytes = width * height * 2
            self.bytesperline = width * 2
        else:
            raise RuntimeError(f"{pixfmt_str} is not an uncompressed format; "
                               "refusing to measure noise on it")
        self.sizeimage = self.frame_bytes
        self.exposure = None  # callers may set; folded into each capture
        self._tmp = tempfile.mkdtemp(prefix="camchar_")

    # start()/stop() are no-ops in the subprocess model (kept for interface
    # compatibility with the PTC/dark callers).
    def start(self): pass
    def stop(self): pass

    def _luma_from_frame(self, raw):
        if self.pixfmt_str in ("YUYV", "YUY2", "UYVY"):
            arr = np.frombuffer(raw[: self.w * self.h * 2], dtype=np.uint8)
            y = arr[1::2] if self.pixfmt_str == "UYVY" else arr[0::2]
            return y.reshape(self.h, self.w).astype(np.float64)
        elif self.pixfmt_str == "Y16":
            # normalise 16-bit mono into the same 0-255 float domain as the
            # 8-bit paths (65535/257 = 255 exactly), so every downstream
            # threshold/fit/export works unchanged at sub-ADU precision
            y = np.frombuffer(raw[: self.w * self.h * 2], dtype="<u2")
            return y.reshape(self.h, self.w).astype(np.float64) / 257.0
        else:  # NV12 / NV21 / I420 / GREY — luma plane is first w*h bytes
            y = np.frombuffer(raw[: self.w * self.h], dtype=np.uint8)
            return y.reshape(self.h, self.w).astype(np.float64)

    def _capture_to_file(self, count, timeout):
        """Run v4l2-ctl once: set fmt (+ exposure if set) and stream `count`
        frames to a temp file. Returns (path, stderr). Raises on failure.
        --verbose makes v4l2-ctl print a dqbuf line per frame with the buffer
        timestamp, which is the only overhead-free framerate measurement."""
        out = os.path.join(self._tmp, f"cap_{count}.raw")
        cmd = ["v4l2-ctl", "-d", self.device,
               f"--set-fmt-video=width={self.w},height={self.h},"
               f"pixelformat={self.pixfmt_str}"]
        if self.exposure is not None:
            # set exposure in the SAME invocation -> single open, no race
            cmd += ["--set-ctrl", f"exposure_time_absolute={self.exposure}"]
        cmd += ["--stream-mmap", f"--stream-count={count}",
                f"--stream-to={out}", "--verbose"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"v4l2-ctl capture timed out after {timeout}s "
                               "(bandwidth/firmware stall?)")
        if not os.path.exists(out) or os.path.getsize(out) == 0:
            raise RuntimeError(f"v4l2-ctl produced no data. stderr:\n{r.stderr}")
        # dqbuf lines land on stderr in current v4l2-ctl, but join both streams
        # so a version that logs to stdout still yields timestamps
        return out, (r.stderr or "") + (r.stdout or "")

    def capture_run(self, n, discard=3, timeout=20.0, verbose=True):
        """Capture discard+n frames in one v4l2-ctl run. Returns
        (path, total, full_frames). Does NOT decode — caller iterates."""
        total = n + discard
        # Timeout must scale with frames x integration time, or any device
        # whose exposure exceeds ~1.5s/frame has its deep stack guaranteed to
        # time out. 2x margin on the commanded exposure, 1.5s/frame floor for
        # the bandwidth-limited regime, plus fixed startup overhead.
        per_frame_s = max(1.5, (self.exposure or 0) * 1e-4 * 2.0)
        t0 = time.monotonic()
        path, errtxt = self._capture_to_file(
            total, timeout=max(timeout, total * per_frame_s + 10))
        t_cap = time.monotonic() - t0
        size = os.path.getsize(path)
        full = size // self.frame_bytes
        if size % self.frame_bytes and full > 0:
            print(f"      !! capture size {size} is not a multiple of "
                  f"{self.frame_bytes}B/frame -- driver stride padding? "
                  "frame parsing may be misaligned")
        self.last_capture_s = t_cap
        # fps from the kernel's per-buffer timestamps (frame intervals only;
        # excludes process spawn / S_FMT / STREAMON overhead, which otherwise
        # biases short captures low and long captures less -- distorting any
        # fps<->exposure calibration built from mixed-length runs).
        ts = [float(m) for m in re.findall(r"\bts:\s*([0-9]+\.[0-9]+)", errtxt)]
        if len(ts) >= 2 and ts[-1] > ts[0]:
            self.last_fps = (len(ts) - 1) / (ts[-1] - ts[0])
        else:
            self.last_fps = (total / t_cap) if t_cap > 0 else 0.0
        self.last_whole_frames = full
        if verbose:
            print(f"      [capture {total} frames in {t_cap:.2f}s "
                  f"({self.last_fps:.1f} fps eff), "
                  f"{full} whole frames in {size//1024}KiB]")
        return path, total, full

    def iter_luma(self, path, n, discard=3):
        """Yield up to n luma frames (2-D float64) from a capture file, one at
        a time, after skipping `discard`. Memory stays at ~one frame."""
        with open(path, "rb") as fh:
            fh.seek(discard * self.frame_bytes)
            for i in range(n):
                raw = fh.read(self.frame_bytes)
                if len(raw) < self.frame_bytes:
                    # a truncated final frame is not data; zero-padding it
                    # would drag the mean down and inflate the variance
                    return
                yield self._luma_from_frame(raw)

    def grab_n(self, n, discard=3, timeout=20.0, verbose=True):
        """Capture and return (list_of_luma, drop_count). Used by the PTC path
        where n is small. For large stacks use capture_run + iter_luma + a
        streaming accumulator to avoid holding every frame in RAM."""
        path, total, full = self.capture_run(n, discard, timeout, verbose)
        out = list(self.iter_luma(path, n, discard))
        drops = max(0, n - len(out))
        try:
            os.remove(path)
        except OSError:
            pass
        if not out:
            raise RuntimeError("no complete frames captured")
        return out, drops

    def close(self):
        try:
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Framerate probe: MJPEG as a CLOCK, never as a measurement
#
# The fps-vs-exposure test is the instrument that separates real integration
# from synthetic (gain-boost / frame-sum) exposure. It only works where the
# frame PERIOD is set by integration rather than by bus bandwidth -- and in an
# uncompressed format at any useful resolution, USB pins the frame rate flat, so
# the test returns "indeterminate (bandwidth-limited)" and tells you nothing.
#
# But that test never looks at a single pixel value. It reads frame TIMESTAMPS.
# A lossy codec destroys the noise this tool exists to measure, which is why
# compressed formats are refused everywhere else -- and it does not perturb when
# frames arrive. So for timing alone, MJPEG is the right instrument: it cuts the
# per-frame payload by an order of magnitude, lifting the bandwidth ceiling well
# above the integration knee, and the knee becomes directly observable.
#
# The hard rule, enforced by this class having no decode path at all: frames
# captured here are COUNTED, never READ. Nothing from an MJPEG probe reaches a
# mean, a noise figure, a master dark or a defect map.
# ----------------------------------------------------------------------------

class FpsProbe:
    """Measures effective framerate vs exposure in a low-bandwidth format.

    Deliberately has no _luma_from_frame and never writes frames to disk: it
    runs v4l2-ctl --stream-mmap WITHOUT --stream-to and reads the per-buffer
    timestamps off --verbose output. Frame payload is discarded by the kernel;
    only the clock survives."""

    def __init__(self, device, width, height, pixfmt_str):
        self.device = device
        self.w, self.h = width, height
        self.pixfmt_str = pixfmt_str.strip()

    def measure(self, exposure, frames=30, discard=3, timeout=None):
        """Set exposure + format and stream `frames` frames, returning
        (fps, n_timestamps). fps comes from the kernel's buffer timestamps, so
        it excludes process spawn / S_FMT / STREAMON overhead."""
        total = frames + discard
        if timeout is None:
            per_frame_s = max(0.2, (exposure or 0) * 1e-4 * 2.0)
            timeout = max(20.0, total * per_frame_s + 10)
        cmd = ["v4l2-ctl", "-d", self.device,
               f"--set-fmt-video=width={self.w},height={self.h},"
               f"pixelformat={self.pixfmt_str}",
               "--set-ctrl", f"exposure_time_absolute={exposure}",
               "--stream-mmap", f"--stream-count={total}", "--verbose"]
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0, 0
        elapsed = time.monotonic() - t0
        txt = (r.stderr or "") + (r.stdout or "")
        ts = [float(m) for m in re.findall(r"\bts:\s*([0-9]+\.[0-9]+)", txt)]
        # drop the settle frames before timing: the first buffers after
        # STREAMON carry pipeline spin-up, not the steady-state period
        ts = ts[discard:] if len(ts) > discard + 1 else ts
        if len(ts) >= 2 and ts[-1] > ts[0]:
            return (len(ts) - 1) / (ts[-1] - ts[0]), len(ts)
        return ((total / elapsed) if elapsed > 0 else 0.0), len(ts)


def pick_timing_format(ranked, advertised, width, height):
    """Choose the lowest-bandwidth format available at this resolution.

    Prefers MJPEG: it is the most widely advertised compressed format on UVC
    bridges and gives the largest bandwidth headroom. Falls back through any
    other compressed format, then gives up (returning None) so the caller can
    say plainly that the timing probe cannot help at this resolution rather
    than silently running it at full bandwidth and reporting a flat curve."""
    def _advertised_here(fcc):
        if not advertised:
            return None          # no mode list -- unknown, allow the attempt
        return any(m["fourcc"].strip() == fcc
                   and m["width"] == width and m["height"] == height
                   for m in advertised)

    have = {f["fourcc"].strip() for f in ranked}
    for fcc in ("MJPG", "JPEG", "H264", "HEVC", "H265"):
        if fcc not in have:
            continue
        adv = _advertised_here(fcc)
        if adv is False:
            continue
        return {"fourcc": fcc, "advertised_at_size": adv}
    return None


def run_fps_probe(device, width, height, fourcc, ladder, frames=30,
                  discard=3, verbose=True):
    """Sweep exposure and measure framerate only, in a low-bandwidth format.

    Returns records shaped exactly like the dark ladder's (exposure_units,
    eff_fps) so derive_knee and exposure_fidelity_verdict consume them
    unchanged -- but carrying no mean_adu, because there is none to carry."""
    probe = FpsProbe(device, width, height, fourcc)
    if verbose:
        print(f"\n=== FRAMERATE PROBE ({fourcc} @ {width}x{height}) ===")
        print("Timing only: frames are counted, never read. A lossy codec "
              "cannot perturb\nwhen frames arrive, and the reduced payload "
              "lifts the bandwidth ceiling so the\nintegration knee becomes "
              "visible. No pixel value from this pass is used.\n")
    out = []
    for exp in ladder:
        fps, nts = probe.measure(exp, frames=frames, discard=discard)
        rb = get_ctrl(device, "exposure_time_absolute")
        rec = {"exposure_units": exp, "exposure_readback": rb,
               "eff_fps": round(fps, 3), "timestamps": nts,
               "timing_format": fourcc}
        out.append(rec)
        if verbose:
            print(f"   exp={exp:6d} (rb={rb})  {fps:7.2f} fps  "
                  f"({nts} timestamps)")
    return out


# ----------------------------------------------------------------------------
# Format enumeration + ranking
# ----------------------------------------------------------------------------

def enum_formats(device):
    fd = os.open(device, os.O_RDWR)
    formats = []
    try:
        i = 0
        while True:
            d = v4l2_fmtdesc()
            d.index = i
            d.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            try:
                xioctl(fd, VIDIOC_ENUM_FMT, d)
            except OSError:
                break
            fs = fourcc_str(d.pixelformat)
            formats.append({
                "fourcc": fs,
                "description": d.description.decode(errors="replace"),
                "compressed": bool(d.flags & 0x0001),
            })
            i += 1
    finally:
        os.close(fd)
    return formats

def rank_formats(formats):
    ranked = []
    for f in formats:
        rank, why = FORMAT_FIDELITY.get(f["fourcc"], (8, "unknown -- treat as lossy"))
        ranked.append({**f, "fidelity_rank": rank, "fidelity_note": why})
    ranked.sort(key=lambda x: (x["fidelity_rank"], x["fourcc"]))
    return ranked

# Formats Capture._luma_from_frame can actually decode. Raw Bayer variants
# rank 0 in FORMAT_FIDELITY but have no decode path here (demosaic is out of
# scope for the UVC bridges this targets), so the auto-pick must skip them or
# Capture.__init__ raises on the "best" format.
CAPTURE_DECODABLE = {"YUYV", "YUY2", "UYVY", "NV12", "NV21", "I420",
                     "GREY", "Y800", "Y8", "Y16"}

def best_measurement_format(ranked):
    for f in ranked:
        if (f["fidelity_rank"] <= 2  # raw mono or uncompressed YUV
                and f["fourcc"].strip() in CAPTURE_DECODABLE):
            return f
    return None


# ----------------------------------------------------------------------------
# Statistics: PTC and dark
# ----------------------------------------------------------------------------

def frame_pair_stats(a, b):
    """Mean signal and TEMPORAL variance from a frame pair.
    Variance is taken from the difference (cancels fixed-pattern non-uniformity)
    and halved. This is the standard PTC pair method."""
    mean = 0.5 * (a.mean() + b.mean())
    diff = a - b
    var = diff.var() / 2.0
    return mean, var

def run_ptc(cap, device, levels_prompt=True, pairs_per_level=1):
    """Walk illumination levels (user adjusts the light), capture pairs, build
    variance-vs-mean. Returns list of (mean, var) and the fit."""
    print("\n=== PHOTON TRANSFER CURVE ===")
    print("Point the camera at an evenly lit, defocused, uniform field.")
    print("You will adjust the LIGHT level between steps (lamp dimmer, screen")
    print("brightness, ND, etc). Keep EXPOSURE and GAIN fixed throughout.")
    print("Aim for ~8-12 levels spanning near-black to near-saturation.")
    print("Enter blank line when you've set a level; type 'done' to finish.\n")

    cap.start()
    points = []
    try:
        idx = 0
        while True:
            if levels_prompt:
                s = input(f"[level {idx}] set light then ENTER (or 'done'): ").strip()
                if s.lower() == "done":
                    break
            frames, drops = cap.grab_n(2 * pairs_per_level, discard=4)
            if drops:
                print(f"   !! {drops} short/dropped frames -- bandwidth limited; "
                      "result suspect at this size")
            # pair only the frames that actually arrived: drops can leave
            # fewer than 2*pairs_per_level and indexing past the end crashes
            npairs = min(pairs_per_level, len(frames) // 2)
            if npairs == 0:
                print("   !! fewer than two whole frames captured -- "
                      "level skipped")
                if not levels_prompt:
                    break
                continue
            means, varis = [], []
            for k in range(npairs):
                m, v = frame_pair_stats(frames[2*k], frames[2*k+1])
                means.append(m); varis.append(v)
            mean = float(np.mean(means)); var = float(np.mean(varis))
            sat = mean >= 250.0  # 8-bit luma near clip
            print(f"   mean={mean:8.2f} ADU   var={var:8.2f}"
                  + ("   <-- near saturation" if sat else ""))
            points.append({"mean": mean, "var": var})
            idx += 1
            if not levels_prompt:
                break
    finally:
        cap.stop()

    fit = fit_ptc(points)
    return points, fit

def fit_ptc(points):
    """Linear fit of var = read_var + (1/gain)*mean over the linear region
    (exclude saturation rollover and the very bottom). gain in e-/ADU is the
    inverse slope; read noise (ADU) = sqrt(intercept)."""
    if len(points) < 3:
        return {"ok": False, "reason": "need >=3 levels"}
    m = np.array([p["mean"] for p in points])
    v = np.array([p["var"] for p in points])
    order = np.argsort(m)
    m, v = m[order], v[order]
    # linear region: drop top points where variance turns over (saturation)
    keep = np.ones(len(m), bool)
    peak = np.argmax(v)
    keep[peak+1:] = False           # everything past variance peak is rollover
    keep &= m < 0.95 * m.max() + 1  # belt and braces
    if keep.sum() < 3:
        keep = m < m.max()          # fallback
    A = np.vstack([m[keep], np.ones(keep.sum())]).T
    slope, intercept = np.linalg.lstsq(A, v[keep], rcond=None)[0]
    gain_e_per_adu = 1.0 / slope if slope > 0 else float("nan")
    read_noise_adu = float(np.sqrt(intercept)) if intercept > 0 else float("nan")
    read_noise_e = read_noise_adu / gain_e_per_adu if gain_e_per_adu == gain_e_per_adu else float("nan")
    full_well_e = float(m[peak] * gain_e_per_adu)
    return {
        "ok": True,
        "gain_e_per_adu": float(gain_e_per_adu),
        "read_noise_adu": read_noise_adu,
        "read_noise_e": float(read_noise_e),
        "full_well_e_approx": full_well_e,
        "n_points_used": int(keep.sum()),
        "saturation_mean_adu": float(m[peak]),
    }


def radial_profile(master, n_bins=12):
    """Mean ADU vs distance from frame centre, to expose shading structure.
    True optical vignetting -> centre-bright, falling outward (ratio<1).
    Reverse vignetting (LSC/processing over-correction) -> corners brighter
    than centre, rising outward (ratio>1). Returns dict with per-annulus means
    and the corner/centre ratio."""
    h, w = master.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()
    edges = np.linspace(0, r_max, n_bins + 1)
    means = []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1] if i < n_bins - 1 else r <= edges[i + 1])
        means.append(float(master[m].mean()) if m.any() else float("nan"))
    centre = means[0]
    corner = means[-1]
    ratio = (corner / centre) if centre else float("nan")
    if ratio > 1.02:
        shape = "REVERSE vignetting (corners brighter -> LSC/processing over-correction)"
    elif ratio < 0.98:
        shape = "normal vignetting (centre brighter)"
    else:
        shape = "flat (no significant radial structure)"
    return {"n_bins": n_bins,
            "annulus_mean_adu": [round(x, 3) for x in means],
            "centre_adu": round(centre, 3), "corner_adu": round(corner, 3),
            "corner_centre_ratio": round(ratio, 4), "shape": shape}


def _med3(a, b, c):
    """Elementwise median of three arrays without stacking (memory-friendly)."""
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def _robust_scale(a):
    """(median, 1.4826*MAD) of an array, with a fallback when the MAD is zero
    (exactly flat data -- synthetic, clamped, or quantiser-collapsed)."""
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:
        sigma = float(a.std()) or 1e-6
    return med, sigma


def local_residual(master):
    """Structure-removed residual of a master frame, plus its robust scale.

    Thresholding the RAW master at mean+6*std fails at native resolution: the
    std includes the shading/FPN structure (the LSC gradient), which inflates
    the threshold and hides genuine hot pixels -- or, with a tilted pedestal,
    flags whole corners. So: remove structure with a separable median-of-3
    smooth (kills single-pixel spikes, follows gradients), then threshold the
    residual against a robust MAD sigma (so the hot pixels themselves cannot
    inflate the noise estimate they are tested against).

    Returns (resid, smooth, med, sigma). The smoothed frame is handed back so
    the defect classifier can test other statistics (a per-pixel maximum, say)
    against the same neighbourhood baseline instead of recomputing it."""
    m = master.astype(np.float32, copy=False)
    pl = np.pad(m, ((0, 0), (1, 1)), mode="edge")
    hsm = _med3(pl[:, :-2], pl[:, 1:-1], pl[:, 2:])
    pv = np.pad(hsm, ((1, 1), (0, 0)), mode="edge")
    smooth = _med3(pv[:-2, :], pv[1:-1, :], pv[2:, :])
    resid = m - smooth
    med, sigma = _robust_scale(resid)
    return resid, smooth, med, sigma


def hot_pixel_mask(master, nsigma=6.0):
    """Single-pixel outliers above a locally-smoothed master."""
    resid, _smooth, med, sigma = local_residual(master)
    return resid > (med + nsigma * sigma)


# ----------------------------------------------------------------------------
# Defect classification
#
# "1057 hot pixels" is not actionable, because the pixels in that list fail in
# three different ways with three different remedies:
#
#   stuck / hot        consistently wrong in every frame. A STUCK pixel has
#                      zero temporal variance -- it is not responding at all --
#                      while a HOT one still responds, just from an elevated
#                      floor. Both are stable, so a master-dark subtraction
#                      genuinely removes them, until they saturate.
#
#   intermittent hot   wrong in SOME frames. This is the one that matters most
#                      and the one a single-threshold defect map cannot see:
#                      the pixel's average looks unremarkable, so subtraction
#                      leaves a residual that flickers frame to frame. It
#                      cannot be calibrated out -- it has to be interpolated
#                      over. Random telegraph signal behaves exactly this way.
#
#   dark current       fine at short exposure, elevated at long. Not a defect
#                      so much as a steep pixel: it scales with integration
#                      time, so a SCALED dark subtraction handles it, and it
#                      does not belong in an interpolation list at all.
#
# Separating them tells you which pixels to interpolate (the first two) and
# which the dark frame already handles (the last two).
# ----------------------------------------------------------------------------

def classify_defects(master, temporal_std, max_px=None, bias=None,
                     exposure_max=None, exposure_bias=None, nsigma=6.0,
                     unstable_sigma=6.0, excursion_sigma=6.0):
    """Sort defective pixels into stuck / hot / intermittent / dark-current.

    master        per-pixel mean of the deep stack
    temporal_std  per-pixel temporal std of the deep stack (sqrt(M2/(n-1)))
    max_px        per-pixel maximum over the deep stack. Without it the
                  "hot in some frames but not the average" test cannot run and
                  intermittents are found by instability alone.
    bias          per-pixel mean at the SHORTEST ladder exposure. Without it
                  there is no exposure lever arm and the dark-current category
                  is not attempted.

    Categories are mutually exclusive, assigned in this precedence: stuck,
    then intermittent, then hot, then dark current. Instability outranks
    elevation deliberately -- a pixel that is hot in 60% of frames reads as
    "hot" on the average, but subtraction will not fix it, so it must land in
    the intermittent bucket where it gets interpolated instead."""
    resid, smooth, med, sigma = local_residual(master)
    hot_thresh = med + nsigma * sigma
    elevated = resid > hot_thresh

    t = np.asarray(temporal_std, dtype=np.float32)
    t_med, t_sig = _robust_scale(t)

    # STUCK: never moved across the whole stack. Definitive -- zero variance
    # over N frames is not something a live pixel does.
    frozen = (t <= 0)
    out = {"nsigma": nsigma, "n_pixels": int(master.size)}

    # A collapsed output makes EVERY pixel frozen. That is the quantiser guard's
    # business, not a defect finding, and mislabelling it as millions of stuck
    # pixels would be worse than saying nothing.
    frozen_frac = float(frozen.mean())
    if frozen_frac > 0.10:
        out.update({
            "ok": False,
            "reason": (f"{frozen_frac*100:.1f}% of pixels have zero temporal "
                       "variance -- the output is quantiser-collapsed, not "
                       "covered in stuck pixels. Classification refused; fix "
                       "the processing path and re-run."),
            "frozen_fraction": round(frozen_frac, 4)})
        return out

    stuck = frozen & (elevated | (master <= 0.5) | (master >= 254.5))

    # INTERMITTENT: unstable in time. Two independent detectors --
    #  (a) temporal std far above the population (switching, not shot noise);
    #  (b) a maximum that reaches hot while the MEAN never does, which is the
    #      signature a threshold-on-the-average defect map walks straight past.
    unstable = t > (t_med + unstable_sigma * t_sig)
    spike = np.zeros_like(unstable)
    if max_px is not None:
        # excursion measured in units of the POPULATION's temporal noise, so a
        # pixel is only "spiking" if it exceeds what N samples of ordinary
        # noise would produce
        excursion = (np.asarray(max_px, dtype=np.float32) - master.astype(
            np.float32, copy=False))
        spike = excursion > (excursion_sigma * max(t_med, 1e-6))
        spike &= (np.asarray(max_px, dtype=np.float32) - smooth) > hot_thresh
    intermittent = (~stuck) & (unstable | spike)

    hot = elevated & (~stuck) & (~intermittent)

    # DARK CURRENT: needs the exposure lever arm. The two endpoints of the
    # ladder give the longest baseline available and therefore the best-
    # conditioned per-pixel slope.
    dark_current = np.zeros_like(hot)
    dc_stats = None
    if bias is not None and exposure_max and exposure_bias is not None \
            and exposure_max > exposure_bias:
        dark_signal = master.astype(np.float32, copy=False) - np.asarray(
            bias, dtype=np.float32)
        d_med, d_sig = _robust_scale(dark_signal)
        span = float(exposure_max - exposure_bias)
        steep = dark_signal > (d_med + nsigma * d_sig)
        dark_current = steep & (~stuck) & (~intermittent) & (~hot)
        rate = dark_signal / span
        dc_stats = {
            "exposure_span_units": span,
            "median_dark_signal_adu": round(d_med, 5),
            "robust_sigma_adu": round(d_sig, 5),
            "median_rate_adu_per_unit": float(np.median(rate)),
            "p99_9_rate_adu_per_unit": float(np.percentile(rate, 99.9)),
            "max_rate_adu_per_unit": float(rate.max()),
            "threshold_adu": round(d_med + nsigma * d_sig, 5),
        }
    else:
        dc_stats = {"skipped": (
            "no bias frame or no exposure lever arm: the dark-current category "
            "needs a short-exposure reference to compare the deep stack "
            "against. Run a ladder with at least one point well below "
            "--exposure-max.")}

    def _coords(mask):
        return np.argwhere(mask).tolist()     # [[y, x], ...]

    cats = {"stuck": stuck, "intermittent_hot": intermittent,
            "hot": hot, "dark_current": dark_current}
    out.update({
        "ok": True,
        "counts": {k: int(v.sum()) for k, v in cats.items()},
        "coords": {k: _coords(v) for k, v in cats.items()},
        "thresholds": {
            "hot_residual_adu": round(hot_thresh, 5),
            "residual_robust_sigma_adu": round(sigma, 6),
            "temporal_median_adu": round(t_med, 5),
            "temporal_robust_sigma_adu": round(t_sig, 6),
            "unstable_above_adu": round(t_med + unstable_sigma * t_sig, 5),
            "excursion_sigma": excursion_sigma,
            "max_px_available": max_px is not None,
        },
        "dark_current_stats": dc_stats,
        "frozen_fraction": round(frozen_frac, 6),
    })
    out["counts"]["total"] = int(sum(out["counts"].values()))
    # what to do about them, which is the point of splitting them up
    out["remedy"] = {
        "stuck": "interpolate -- the pixel carries no signal to correct",
        "intermittent_hot": ("interpolate -- inconsistent frame to frame, so "
                             "no master dark can subtract it"),
        "hot": ("dark subtraction removes it while it stays below "
                "saturation; interpolate only the extreme ones"),
        "dark_current": ("scaled dark subtraction handles it; does not belong "
                         "in an interpolation list"),
    }
    return out


# ----------------------------------------------------------------------------
# Quantiser-limited detection
#
# The clip guard catches the rails (0 and 255). It does NOT catch the other way
# a processing path destroys a measurement: collapsing the output onto so few
# codes that the sensor's noise no longer reaches the least significant bit. A
# run can sit at mean 18 -- nowhere near a rail -- and still be meaningless.
#
# The tell is temporal noise of exactly 0.000 across a deep stack, with the mean
# identical to three decimals at every ladder point. A real 8-bit path CANNOT do
# that: read noise alone guarantees a pixel sitting between two codes lands on
# either side from frame to frame, so any genuine measurement dithers its own
# LSB. Zero temporal noise is the quantiser's floor being reported as if it were
# the sensor's.
#
# This matters more than it sounds: with the output collapsed, hot_pixel_mask
# sees a nearly flat residual and finds 12 defects where the true count is 1057.
# PHD2 auto-loads that defect map at every camera connect, so exporting it is
# strictly worse than exporting nothing.
# ----------------------------------------------------------------------------

# Below this, a deep stack is reporting quantisation, not sensor noise.
QUANTISER_NOISE_FLOOR_ADU = 0.05
# A master dark holding this few distinct values has no gradient left to
# threshold against; hot-pixel detection on it is meaningless.
QUANTISER_MAX_DISTINCT_CODES = 2


def code_step_adu(native_bit_depth):
    """Size of one output code in the report's 0-255 ADU domain.

    Every format is normalised into that domain (Y16 is divided by 257), so an
    8-bit path steps by 1.0 and a 16-bit path by 1/257."""
    return 1.0 if (native_bit_depth or 8) <= 8 else 1.0 / 257.0


def quantiser_check(distinct_master, temporal, code_step,
                    distinct_stack=None):
    """Decide whether a stack is quantiser-limited, and say why.

    Two independent detectors, either one sufficient:
      * the master dark holds <= 2 distinct values -- there is no structure
        left to measure, whatever the noise number says;
      * temporal noise is below the dither floor -- the path is not resolving
        its own least significant bit.
    """
    reasons = []
    if distinct_master is not None and distinct_master <= QUANTISER_MAX_DISTINCT_CODES:
        reasons.append(
            f"master dark holds only {distinct_master} distinct value(s) "
            f"(<= {QUANTISER_MAX_DISTINCT_CODES})")
    if temporal == temporal and temporal < QUANTISER_NOISE_FLOOR_ADU:
        reasons.append(
            f"temporal noise {temporal:.4f} ADU is below the "
            f"{QUANTISER_NOISE_FLOOR_ADU} ADU dither floor -- an 8-bit path "
            "cannot have zero temporal noise")
    out = {
        "quantiser_limited": bool(reasons),
        "distinct_codes_master": distinct_master,
        "distinct_codes_stack": distinct_stack,
        "code_step_adu": code_step,
        "reasons": reasons,
    }
    if reasons:
        out["note"] = (
            "QUANTISER-LIMITED: the output is collapsed onto too few codes for "
            "the sensor's noise to reach the LSB. FPN, hot-pixel count and "
            "dark current from this run describe the processing path, not the "
            "sensor. Reduce gamma/contrast or raise gain until the deep stack "
            "shows temporal dither, then re-run.")
    return out


def noise_upper_bound(temporal, code_step):
    """Turn a floored noise number into an honest upper bound.

    "IRREDUCIBLE temporal dark noise: 0.000 ADU" reads as a result -- a
    remarkably quiet sensor. It is not one. When the measurement has hit the
    quantiser floor, all that is known is that the true noise is smaller than
    the step that would have been needed to register: half a code."""
    ub = 0.5 * code_step
    return {
        "upper_bound_adu": ub,
        "measured_adu": (float(temporal) if temporal == temporal else None),
        "statement": (f"< {ub:.3g} ADU, unmeasurable at this bit depth and "
                      "processing setting"),
    }


def minimum_detectable_slope(xs, ys, code_step, quantiser_limited,
                             resid_std=None):
    """The smallest dark-current slope this ladder could have resolved.

    "0.00000 ADU/unit (r2=0.206)" is not a measurement of zero dark current; it
    is a measurement that found nothing, reported as if it had found zero. What
    PHD2's dark-scaling decision actually needs is the detection limit: how big
    would the slope have had to be before this ladder could have seen it?

    Two regimes:
      * quantiser-limited -- with no temporal dither, nothing below one whole
        code ever flips a pixel, so the level literally cannot register a change
        smaller than the code step, no matter how many frames are averaged;
      * normally noisy -- the limit is the scatter of the ladder means about the
        fit, taken at 3 sigma.
    Divide that smallest detectable level change by the exposure span."""
    if len(xs) < 2:
        return None
    span = float(np.max(xs) - np.min(xs))
    if span <= 0:
        return None
    if quantiser_limited or not resid_std or resid_std <= 0:
        delta = float(code_step)
        basis = ("one quantiser code -- with no temporal dither, a level change "
                 "smaller than a full code flips no pixels and is invisible "
                 "however many frames are stacked")
    else:
        delta = 3.0 * float(resid_std)
        basis = ("3x the residual scatter of the ladder means about the fit")
    slope = delta / span
    return {
        "min_detectable_slope_adu_per_unit": slope,
        "min_detectable_slope_adu_per_second": slope / 100e-6,
        "min_detectable_level_change_adu": delta,
        "exposure_span_units": span,
        "basis": basis,
        "statement": (f"< {slope:.3g} ADU/unit "
                      f"(no level change above {delta:.3g} ADU was detectable "
                      f"across a {span:.0f}-unit exposure span)"),
    }


def derive_knee(slope_data, tol=0.05):
    """Find the exposure where integration takes over from bus bandwidth.

    Below the knee the frame period is pinned by USB/pipeline bandwidth and the
    commanded exposure is clamped inside it, so the level is a bias pedestal
    carrying no dark-current information. Fitting those points is what drags the
    dark-current regression down.

    The knee does not need to be supplied -- the ladder already measures it. It
    is the first exposure at which fps departs from its pinned maximum. Taking
    it as an argument means inheriting a value from a different firmware: --knee
    256 was right for a 2047-unit range and wrong for 9411, where fps stayed
    flat through 256 and only broke at 384.

    Returns None when fps never departs (everything is bandwidth-limited)."""
    pts = sorted((d["exposure_units"], d["eff_fps"]) for d in slope_data
                 if d.get("eff_fps", 0) > 0)
    if len(pts) < 3:
        return None
    fps_max = max(f for _, f in pts)
    limit = (1.0 - tol) * fps_max
    for i, (e, f) in enumerate(pts):
        if f >= limit:
            continue
        # One point below the line can be a capture hiccup rather than the
        # knee. Require the departure to persist: every later point must stay
        # down too (the last point is allowed to stand alone).
        later = [ff for _, ff in pts[i + 1:]]
        if later and not all(ff < fps_max for ff in later):
            continue
        return {
            "knee_units": int(e),
            "fps_pinned_max": round(fps_max, 2),
            "fps_at_knee": round(f, 2),
            "departure_tolerance": tol,
            "last_pinned_exposure": (int(pts[i - 1][0]) if i else None),
            "detail": (f"fps held ~{fps_max:.1f} up to "
                       f"{pts[i-1][0] if i else '(first point)'} and fell to "
                       f"{f:.1f} at {e} -- integration takes over there"),
        }
    return None


def build_exposure_ladder(exposure_max, ladder=None, knee=None,
                          ladder_points=7):
    """The dark run's exposure ladder: two low points for the bias intercept,
    then ladder_points spaced across the integrating region."""
    if ladder is None:
        lo = [max(1, int(exposure_max * f)) for f in (0.05, 0.15)]
        # The ladder has to be laid out BEFORE the knee can be measured (the
        # measurement is the ladder), so span a fixed fraction of the range
        # here and let derive_knee sort out which points were pedestal-only
        # once the fps data exists.
        start = (int(exposure_max * 0.30) if knee is None
                 else max(knee, int(exposure_max * 0.30)))
        n = max(2, ladder_points)
        hi = [int(start + (exposure_max - start) * i / (n - 1))
              for i in range(n)]
        out = sorted(set(lo + hi))
    else:
        out = sorted(set(int(x) for x in ladder))
    if exposure_max not in out:
        out.append(exposure_max)
        out = sorted(set(out))
    return out


def build_timing_ladder(exp_min, exp_max, n=14):
    """Exposure sweep for the framerate probe: GEOMETRIC, not linear.

    The knee sits low -- 384 units out of a 9411 range in one case -- and the
    dark ladder's two sub-knee points (5% and 15% of max) would step straight
    over it. The probe is cheap (no decode, no disk), so it can afford a wide
    sweep, and geometric spacing puts the resolution where the knee actually
    lives instead of spreading it evenly across a range that is mostly flat."""
    lo = max(1, int(exp_min or 1))
    hi = int(exp_max)
    if hi <= lo:
        return [hi]
    vals = np.unique(np.round(np.geomspace(lo, hi, max(3, n))).astype(np.int64))
    return [int(v) for v in vals]


def build_fps_calibration(ladder_data, fourcc=None, width=None, height=None):
    """From a dark run's per-exposure (exposure, eff_fps) pairs, build a
    monotonic fps->exposure inverse over the INTEGRATION-LIMITED region only.

    At low exposure the framerate is pinned by USB/pipeline bandwidth (flat ~max
    fps) and carries no exposure info. We keep only the region where fps is
    actually FALLING with exposure -- there, frame period is set by integration
    and fps maps 1:1 to effective exposure. Returns a dict with the usable
    (fps, exposure) anchor points, sorted by fps ascending for interpolation.
    """
    pts = [(d["exposure_units"], d.get("eff_fps", 0.0))
           for d in ladder_data if d.get("eff_fps", 0) > 0]
    pts.sort()
    if len(pts) < 3:
        return None
    fps_max = max(f for _, f in pts)
    thresh = 0.90 * fps_max
    integ = [(e, f) for e, f in pts if f < thresh]
    mono = []
    last_f = None
    for e, f in integ:
        if last_f is None or f < last_f:
            mono.append((e, f)); last_f = f
    if len(mono) < 3:
        return None
    anchors = sorted(((f, e) for e, f in mono))
    # The fps<->exposure mapping is a property of THIS format at THIS size --
    # the bandwidth ceiling moves with both. Stamp them so a calibration built
    # in one mode is never silently applied in another (and in particular so a
    # compressed timing probe's numbers can never be mistaken for these).
    return {"fps_max_bandwidth": round(fps_max, 2),
            "integration_limited_below_fps": round(thresh, 2),
            "anchors_fps": [round(f, 3) for f, _ in anchors],
            "anchors_exposure": [e for _, e in anchors],
            "valid_fps_range": [round(anchors[0][0], 2), round(anchors[-1][0], 2)],
            "fourcc": fourcc, "width": width, "height": height}


def fps_to_effective_exposure(calib, fps):
    """Invert measured fps to effective exposure via the calibration. Returns
    (exposure_estimate_or_None, flag)."""
    if not calib:
        return None, "no-calibration"
    af = calib["anchors_fps"]; ae = calib["anchors_exposure"]
    if fps >= calib["integration_limited_below_fps"]:
        return None, "bandwidth-limited (fps too high to imply exposure)"
    lo, hi = calib["valid_fps_range"]
    if fps < lo:
        return ae[0], "extrapolated-long (below calibrated fps)"
    if fps > hi:
        return ae[-1], "extrapolated-short (above calibrated fps)"
    est = float(np.interp(fps, af, ae))
    return est, "interpolated"


def run_auto_dark(cap, device, iterations=20, frames=30, settle_s=1.0,
                  calib=None, verbose=True):
    """Auto-exposure in the dark: leave AE enabled, cap the lens, and watch
    what the firmware does. Each iteration captures `frames` frames (AE running
    free), reads back the AE-CHOSEN exposure_time_absolute and gain, measures
    the real framerate from the capture, and records mean/noise. Reveals:
      - where AE parks exposure in the dark (does it rail to max? which max?)
      - whether AE is stable or HUNTS (oscillating exposure/gain frame-to-frame)
      - the framerate at the AE-chosen integration time.
    No exposure is commanded (cap.exposure stays None) so AE keeps control.
    """
    print("\n=== AUTO-EXPOSURE DARK BEHAVIOUR ===")
    print("CAP THE LENS. Auto-exposure is LEFT ON; we observe what it does.")
    print(f"{iterations} iterations x {frames} frames. Reading back AE choices.\n")

    # UVC controls are STICKY across processes: if a dark/manual run came
    # before this one (the cam_manager plan always sequences it that way),
    # auto_exposure is still 1 (manual) and we would "observe" a frozen manual
    # exposure and call it a stable AE. Explicitly hand control back to AE.
    # UVC menu: 0=Auto, 1=Manual, 2=Shutter Priority, 3=Aperture Priority;
    # webcam firmwares implement 3 (and sometimes only 1/3).
    ae_note = "auto_exposure control absent"
    if "auto_exposure" in list_controls(device):
        for mode in (3, 0, 2):
            ok, err = set_ctrl(device, "auto_exposure", mode)
            if ok:
                break
        rb = get_ctrl(device, "auto_exposure")
        ae_note = (f"auto_exposure->{mode}: {'ok' if ok else 'FAILED ' + err} "
                   f"(readback={rb})")
        if rb == 1:
            ae_note += " -- WARNING: device still reads back MANUAL; the " \
                       "series below observes manual mode, not AE"
        print(f"   {ae_note}")

    # real device exposure range, so the verdict scales to THIS camera rather
    # than any hard-coded assumption (exposure max varies: 2047, 8192, ...).
    exp_rng = get_ctrl_range(device, "exposure_time_absolute")
    exp_lo = exp_rng.get("min", 1)
    exp_hi = exp_rng.get("max", None)

    cap.exposure = None  # critical: do NOT command exposure; let AE drive
    series = []
    t_start = time.monotonic()
    for i in range(iterations):
        # read AE state BEFORE and AFTER the capture to catch hunting
        exp_before = get_ctrl(device, "exposure_time_absolute")
        gain_before = get_ctrl(device, "gain")
        try:
            path, total, full = cap.capture_run(frames, discard=2,
                                                timeout=60.0, verbose=False)
        except (TimeoutError, RuntimeError) as e:
            print(f"   iter {i:2d}: capture failed: {e}")
            continue
        exp_after = get_ctrl(device, "exposure_time_absolute")
        gain_after = get_ctrl(device, "gain")

        # streaming mean/noise over the frames
        count = 0; mean_px = None; m2_px = None
        for fr in cap.iter_luma(path, frames, discard=2):
            count += 1
            if mean_px is None:
                mean_px = np.zeros_like(fr); m2_px = np.zeros_like(fr)
            d = fr - mean_px; mean_px += d / count; m2_px += d * (fr - mean_px)
            del fr, d
        try:
            os.remove(path)
        except OSError:
            pass
        if count == 0:
            print(f"   iter {i:2d}: no frames"); continue
        mean_level = float(mean_px.mean())
        temporal = float(np.sqrt(m2_px / (count - 1)).mean()) if count > 1 else float("nan")
        fps = cap.last_fps
        elapsed = time.monotonic() - t_start

        # fps-implied effective exposure (independent of the possibly-lying
        # exposure_time_absolute readback)
        fps_exp, fps_flag = fps_to_effective_exposure(calib, fps)

        rec = {"iter": i, "t_s": round(elapsed, 1),
               "exp_before": exp_before, "exp_after": exp_after,
               "gain_before": gain_before, "gain_after": gain_after,
               "fps": round(fps, 2), "mean_adu": round(mean_level, 3),
               "temporal_noise_adu": round(temporal, 4), "frames": count,
               "fps_implied_exposure": (round(fps_exp) if fps_exp is not None else None),
               "fps_implied_flag": fps_flag}
        series.append(rec)
        if verbose:
            hunt = "" if exp_before == exp_after else f" ->{exp_after}"
            imp = ""
            if fps_exp is not None:
                imp = f"  fps_implies_exp~{round(fps_exp)}"
            elif calib:
                imp = f"  ({fps_flag})"
            print(f"   iter {i:2d}  t={elapsed:6.1f}s  AE_exp={exp_before}{hunt}  "
                  f"gain={gain_before}  fps={fps:5.2f}  mean={mean_level:7.3f}  "
                  f"noise={temporal:.4f}{imp}")
        time.sleep(settle_s)

    # summary: where did AE park, did it hunt, what framerate
    out = {"iterations": len(series), "series": series,
           "auto_exposure_restore": ae_note}
    if series:
        exps = [s["exp_after"] for s in series if s["exp_after"] is not None]
        gains = [s["gain_after"] for s in series if s["gain_after"] is not None]
        fpss = [s["fps"] for s in series]
        if exps:
            out["exposure_chosen_min"] = min(exps)
            out["exposure_chosen_max"] = max(exps)
            out["exposure_hunting"] = (max(exps) != min(exps))
        if fpss:
            out["fps_mean"] = round(sum(fpss) / len(fpss), 2)
            out["fps_min"] = round(min(fpss), 2)
            out["fps_max"] = round(max(fpss), 2)
        if verbose:
            print("\n--- AUTO-EXPOSURE DARK SUMMARY ---")
            if exps:
                if out["exposure_hunting"]:
                    print(f"  AE exposure HUNTS: {min(exps)}..{max(exps)} "
                          "(oscillating, not settling)")
                else:
                    print(f"  AE exposure parked at: {exps[-1]} (stable)")
                # where AE parked, as a fraction of THIS device's real range,
                # and in seconds (UVC exposure_time_absolute is in 100us units).
                pk = exps[-1]
                if pk is not None:
                    secs = pk * 100e-6  # 100us per unit
                    out["exposure_parked_units"] = pk
                    out["exposure_parked_seconds"] = round(secs, 4)
                    if exp_hi:
                        frac = pk / exp_hi
                        out["exposure_parked_fraction_of_max"] = round(frac, 3)
                        out["exposure_max_units"] = exp_hi
                        if frac >= 0.97:
                            where = (f"RAILED at max ({exp_hi} = {exp_hi*100e-6:.3f}s) "
                                     "-- AE drove exposure to the ceiling (expected "
                                     "in the dark: no light to reach its target)")
                        elif frac <= 0.05:
                            where = (f"at min ({pk}) -- AE chose near-minimum exposure")
                        else:
                            where = (f"{frac*100:.0f}% of max ({pk}/{exp_hi}, "
                                     f"{secs:.3f}s)")
                    else:
                        where = f"{pk} units ({secs:.3f}s); device max unknown"
                    print(f"  -> AE parked: {where}")
            if gains:
                print(f"  gain: {min(gains)}..{max(gains)}")
            print(f"  framerate: mean {out.get('fps_mean')} fps "
                  f"(range {out.get('fps_min')}-{out.get('fps_max')})")

            # the key verdict: does exposure_time_absolute readback agree with
            # what the framerate implies AE actually did?
            imp = [s["fps_implied_exposure"] for s in series
                   if s["fps_implied_exposure"] is not None]
            if calib and imp:
                imp_lo, imp_hi = min(imp), max(imp)
                out["fps_implied_exposure_min"] = imp_lo
                out["fps_implied_exposure_max"] = imp_hi
                print(f"  fps-implied effective exposure: {imp_lo}..{imp_hi}")
                if exps:
                    rb = exps[-1]
                    # compare last readback to last fps-implied
                    last_imp = next((s["fps_implied_exposure"] for s in reversed(series)
                                     if s["fps_implied_exposure"] is not None), None)
                    if rb is not None and last_imp is not None:
                        ratio = (rb / last_imp) if last_imp else float("inf")
                        if 0.7 <= ratio <= 1.43:
                            verdict = ("control readback AGREES with fps-implied "
                                       "exposure -> exposure_time_absolute is honest in AE")
                        else:
                            verdict = (f"control readback ({rb}) DISAGREES with "
                                       f"fps-implied ({last_imp}) -> the readback does "
                                       "NOT reflect AE's real integration; use fps")
                        out["control_vs_fps_verdict"] = verdict
                        print(f"  -> {verdict}")
            elif not calib:
                print("  (no --calib supplied: cannot cross-check the exposure "
                      "readback against framerate. Run --dark with --save-calib "
                      "first, then pass --calib here.)")
    return out


def exposure_fidelity_verdict(slope_data, knee):
    """Decide real-vs-synthetic exposure from the dark ladder's (exposure,
    eff_fps) pairs. Returns fields to merge into the exposure_fidelity report.

    The fps-vs-exposure test only works where the frame PERIOD is set by
    INTEGRATION, not by bus bandwidth. At high resolution each frame is large
    and USB caps the frame rate, so the frame period is pinned long regardless
    of exposure. If the longest exposure still fits inside that bandwidth-
    limited frame period, fps physically CANNOT fall, and flat fps says nothing
    about real-vs-synthetic -- calling it "synthetic" there is wrong. So gate on
    the bandwidth ceiling: only points whose integration time exceeds the
    bandwidth-limited frame period (1 / fastest-observed-fps) are genuinely
    integration-limited. Points below the knee (clamped sub-frame pedestal) are
    excluded too."""
    out = {}
    fps_ceiling = max((d["eff_fps"] for d in slope_data if d["eff_fps"] > 0),
                      default=0.0)
    # exposure units are 100us each; frame period (s) -> units = period / 100e-6
    bw_period_units = ((1.0 / fps_ceiling) / 100e-6
                       if fps_ceiling > 0 else float("inf"))
    out["bandwidth_ceiling_fps"] = round(fps_ceiling, 2)
    out["bandwidth_frame_period_units"] = (
        round(bw_period_units) if bw_period_units != float("inf") else None)
    integ = [d for d in slope_data
             if d["exposure_units"] >= knee
             and d["exposure_units"] >= bw_period_units
             and d["eff_fps"] > 0]
    max_exp = max((d["exposure_units"] for d in slope_data), default=0)
    if len(integ) >= 2:
        f0, f1 = integ[0]["eff_fps"], integ[-1]["eff_fps"]
        e0, e1 = integ[0]["exposure_units"], integ[-1]["exposure_units"]
        fps_drop = f0 / f1 if f1 else float("nan")
        expo_rise = e1 / e0 if e0 else float("nan")
        out["integrating_region"] = {
            "exposure_from": e0, "exposure_to": e1,
            "fps_drop": round(fps_drop, 2), "expo_rise": round(expo_rise, 2)}
        # Real integration: fps falls roughly in proportion to exposure once
        # past the bandwidth-limited frame period. Synthetic: fps stays flat
        # while exposure climbs through that integration-limited region.
        if fps_drop >= 0.6 * expo_rise:
            out["verdict"] = (
                f"in integration-limited region fps fell {fps_drop:.1f}x as "
                f"exposure rose {expo_rise:.1f}x -> REAL integration")
        elif fps_drop < 1.3 and expo_rise > 2:
            out["verdict"] = (
                f"in integration-limited region fps held ~flat while exposure "
                f"rose {expo_rise:.1f}x -> synthetic exposure (gain/sum)")
        else:
            out["verdict"] = (
                f"partial: fps fell {fps_drop:.1f}x vs exposure {expo_rise:.1f}x "
                "-> integration real but sub-proportional (clamped/quantised)")
    else:
        # The common high-res case: the longest exposure never exceeds the
        # bandwidth-limited frame period, so the test is simply untestable.
        if bw_period_units != float("inf") and max_exp < bw_period_units:
            out["verdict"] = (
                f"indeterminate (bandwidth-limited): max exposure {max_exp} "
                f"units (~{max_exp * 0.1:.0f}ms) stays within the bandwidth-"
                f"limited frame period (~{bw_period_units * 0.1:.0f}ms at the "
                f"{fps_ceiling:.1f}fps ceiling), so fps cannot respond to "
                "exposure here. Real-vs-synthetic is UNTESTABLE at this "
                "resolution -- use a smaller resolution (higher fps ceiling) "
                "or a longer exposure to push past the frame period.")
        else:
            out["verdict"] = (
                "too few integration-limited points for a verdict; add "
                "exposures above the bandwidth-limited frame period "
                f"(~{bw_period_units:.0f} units)")
    return out


def run_dark(cap, device, dark_frames, exposure_max, slope_points,
             ladder=None, discard=2, knee=None, ladder_frames=16,
             ladder_points=7, timing_ladder=None, verbose=True):
    """Dark characterisation, weighted to the longest exposure.

    Most frames are spent at exposure_max to build the master dark and measure
    the irreducible temporal dark noise. Additional exposures in the INTEGRATING
    region (above the frame-period knee) fix the dark-current slope. Exposures
    below the knee are clamped to a sub-frame period and only establish the
    bias/offset, so we sample just a couple there and concentrate the rest high.

    ladder:        explicit list of exposure values; overrides the default sweep.
    ladder_points: number of points to auto-space across the integrating region
                   (knee..exposure_max) when ladder is None. Default 7.
    ladder_frames: frames captured at each NON-deep ladder point. Default 16.
    dark_frames:   frames in the deep stack at exposure_max. Default 128.
    discard:       frames to skip at the start of each capture (settle). Def 2.
    knee:          exposure above which real integration begins. None (the
                   default) DERIVES it from fps measurements after capture --
                   see derive_knee. Pass a number only to override a bad
                   detection; a knee carried over from a different firmware's
                   exposure range is worse than no knee.
    timing_ladder: optional (exposure, eff_fps) records from a low-bandwidth
                   FpsProbe pass. When present these drive the knee and the
                   real-vs-synthetic verdict INSTEAD of this run's own fps,
                   because an uncompressed stream at any useful resolution is
                   bandwidth-pinned and its flat fps curve says nothing. Pixel
                   data still comes only from the uncompressed capture here.
    """
    print("\n=== DARK NOISE CHARACTERISATION ===")
    print("CAP THE LENS / full darkness. Keep temperature stable across the run.")
    print(f"Deep stack: {dark_frames} frames at exposure={exposure_max} (max).")
    print(f"Ladder points: {ladder_frames} frames each. "
          f"Discarding first {discard} frame(s) per exposure (settle).\n")

    # Two low points for the bias intercept, then ladder_points evenly spaced
    # across the integrating region, so the dark-current slope is well
    # determined.
    ladder = build_exposure_ladder(exposure_max, ladder, knee, ladder_points)

    cap.start()
    started = datetime.now(timezone.utc).isoformat()
    slope_data = []
    master = None
    residual_std = None
    deep_drops = 0
    deep_clipped = False
    deep_quant = None
    radial = None
    fpn = hot = None
    hot_coords = None
    defects = None
    bias_px = None          # per-pixel mean at the shortest ladder exposure
    bias_exposure = None
    # one output code, in the 0-255 ADU domain every format is normalised into
    native_depth = 16 if cap.pixfmt_str == "Y16" else 8
    code_step = code_step_adu(native_depth)

    # Warm-up: the FIRST v4l2-ctl capture of a run pays one-time costs (gadget
    # streaming spin-up, ISP pipeline init, cold-start) that drag its effective
    # fps down and make point one untrustworthy. Fire a small throwaway capture
    # first so every real ladder point measures a warm pipeline.
    try:
        cap.exposure = ladder[0]
        set_ctrl(device, "exposure_time_absolute", ladder[0])
        wpath, _, _ = cap.capture_run(6, discard=0, timeout=20.0, verbose=False)
        try:
            os.remove(wpath)
        except OSError:
            pass
        if verbose:
            print(f"   (warm-up capture done in {cap.last_capture_s:.2f}s; "
                  "discarded)")
    except Exception as e:
        if verbose:
            print(f"   (warm-up skipped: {e})")

    try:
        for exp in ladder:
            # set exposure on the capture object; it is applied in the SAME
            # v4l2-ctl invocation as the frame grab (single open, no race).
            cap.exposure = exp
            ok, err = set_ctrl(device, "exposure_time_absolute", exp)
            rb = get_ctrl(device, "exposure_time_absolute")
            is_deep = (exp == exposure_max)
            # the shortest ladder point is the bias reference the dark-current
            # classifier compares the deep stack against; the ladder is sorted
            # ascending so it is captured before the deep point needs it
            is_bias = (exp == ladder[0]) and not is_deep
            nframes = dark_frames if is_deep else max(4, ladder_frames)
            path, total, full = cap.capture_run(
                nframes, discard=discard,
                timeout=max(20.0, exp * 1e-4 * 4 + 10))

            # Streaming Welford over frames: per-pixel running mean + M2.
            # Holds only mean and M2 (one frame each) -> ~15MB, not GBs.
            count = 0
            mean_px = None   # per-pixel running mean (the master dark / FPN)
            m2_px = None     # per-pixel sum of squared deviations
            # Presence histogram over the whole stack, in native code units
            # (x257 puts both the 8-bit and the Y16/257 domains on the same
            # integer grid). Counting distinct codes is what catches a
            # collapsed output that the mean and the rails both look fine at.
            code_seen = np.zeros(65536, dtype=bool)
            # per-pixel maximum, deep stack only: a pixel that reaches hot in a
            # handful of frames but averages out is invisible to any threshold
            # on the mean, and that is exactly the intermittent case
            max_px = None
            for fr in cap.iter_luma(path, nframes, discard=discard):
                count += 1
                if mean_px is None:
                    mean_px = np.zeros_like(fr)
                    m2_px = np.zeros_like(fr)
                    if is_deep:
                        max_px = fr.copy()
                delta = fr - mean_px
                mean_px += delta / count
                m2_px += delta * (fr - mean_px)
                if max_px is not None:
                    np.maximum(max_px, fr, out=max_px)
                codes = np.rint(fr * 257.0).astype(np.int32)
                np.clip(codes, 0, 65535, out=codes)
                code_seen[codes.ravel()] = True
                del fr, delta, codes
            try:
                os.remove(path)
            except OSError:
                pass
            drops = max(0, nframes - count)
            if count == 0:
                raise RuntimeError(f"no frames at exp={exp}")

            mean_level = float(mean_px.mean())
            # per-pixel temporal variance = M2/(count-1); temporal noise =
            # mean over pixels of sqrt(variance). This is read+dark-shot.
            if count > 1:
                var_px = m2_px / (count - 1)
                std_px = np.sqrt(var_px)
                temporal = float(std_px.mean())
            else:
                std_px = None
                temporal = float("nan")
            if is_bias:
                # float32: this is a reference level, not an accumulator, and
                # it has to survive until the deep point at the end
                bias_px = mean_px.astype(np.float32)
                bias_exposure = exp

            # clip guard: if the processing path (brightness/contrast/gamma) has
            # driven the output against the floor or ceiling, the frame is not a
            # measurement. Detect pixels pinned at 0 / 255 with no variance.
            floor_frac = float((mean_px <= 0.5).mean())
            ceil_frac = float((mean_px >= 254.5).mean())
            clipped = (floor_frac > 0.5 or ceil_frac > 0.5)
            clip_note = ""
            if clipped:
                where = "floor(0)" if floor_frac > ceil_frac else "ceiling(255)"
                clip_note = (f"  !! CLIPPED at {where}: "
                             f"{max(floor_frac, ceil_frac)*100:.0f}% of pixels "
                             "pinned -- measurement INVALID (processing path "
                             "saturated; reduce brightness/contrast/gamma)")

            # quantiser guard: the other way the processing path invalidates a
            # run, and the one the clip guard cannot see (a collapsed output
            # sitting at mean 18 is nowhere near a rail).
            n_stack_codes = int(code_seen.sum())
            n_master_codes = int(np.unique(mean_px).size)
            qcheck = quantiser_check(n_master_codes, temporal, code_step,
                                     distinct_stack=n_stack_codes)
            quant_note = ""
            if qcheck["quantiser_limited"]:
                quant_note = ("  !! QUANTISER-LIMITED: " +
                              "; ".join(qcheck["reasons"]))

            slope_data.append({"exposure_units": exp, "exposure_readback": rb,
                               "mean_adu": mean_level,
                               "temporal_noise_adu": temporal,
                               "frames": count, "drops": drops,
                               "capture_s": round(cap.last_capture_s, 3),
                               "eff_fps": round(cap.last_fps, 2),
                               "clipped": clipped,
                               "clip_floor_frac": round(floor_frac, 4),
                               "clip_ceil_frac": round(ceil_frac, 4),
                               "distinct_codes_stack": n_stack_codes,
                               "distinct_codes_master": n_master_codes,
                               "quantiser_limited":
                                   qcheck["quantiser_limited"]})
            if verbose:
                print(f"   exp={exp:5d} (rb={rb})  mean={mean_level:7.3f} ADU  "
                      f"temporal_noise={temporal:6.3f} ADU  "
                      f"{cap.last_fps:5.1f}fps  "
                      f"codes={n_stack_codes:3d}/{n_master_codes:<3d}  "
                      f"n={count}" + (f"  drops={drops}" if drops else "")
                      + clip_note + quant_note)
            if is_deep:
                deep_drops = drops
                deep_clipped = clipped
                deep_quant = qcheck
                master = mean_px                 # FIXED-PATTERN part (per-pixel)
                # irreducible temporal noise after master-dark subtraction is
                # exactly the temporal std we already computed per-pixel:
                residual_std = temporal
                fpn = float(master.std())        # spatial FPN spread
                _hot_mask = hot_pixel_mask(master)
                hot = int(_hot_mask.sum())
                _yx = np.argwhere(_hot_mask)
                hot_coords = _yx.tolist()        # [[y, x], ...] numpy row-major convention
                # A flat count of defects is not actionable: the pixels in it
                # fail in three different ways needing three different
                # remedies. Split them.
                if std_px is not None:
                    defects = classify_defects(
                        master, std_px, max_px=max_px, bias=bias_px,
                        exposure_max=exp, exposure_bias=bias_exposure)
                    if verbose:
                        if defects.get("ok"):
                            c = defects["counts"]
                            print(f"   defects: {c['total']} total  "
                                  f"stuck={c['stuck']}  "
                                  f"intermittent={c['intermittent_hot']}  "
                                  f"hot={c['hot']}  "
                                  f"dark-current={c['dark_current']}")
                            if not defects["thresholds"]["max_px_available"]:
                                print("     (no per-pixel maximum: "
                                      "intermittents found by instability "
                                      "alone)")
                            if "skipped" in (defects["dark_current_stats"] or {}):
                                print("     dark-current category skipped: "
                                      "no bias reference in this ladder")
                        else:
                            print(f"   defects: not classified -- "
                                  f"{defects['reason']}")
                del max_px
                max_px = None
                if qcheck["quantiser_limited"] and verbose:
                    print(f"   !! deep stack QUANTISER-LIMITED "
                          f"({n_stack_codes} distinct codes across {count} "
                          f"frames, {n_master_codes} in the master). The "
                          f"hot-pixel count below ({hot}) is what survives a "
                          "collapsed output, not the sensor's defect count.")
                if clipped or qcheck["quantiser_limited"]:
                    why = "clipped" if clipped else "quantiser-limited"
                    radial = {"shape": f"N/A - output {why}", "clipped": clipped,
                              "quantiser_limited": qcheck["quantiser_limited"]}
                    if verbose:
                        print(f"   radial: skipped (deep stack {why.upper()} -- "
                              "FPN/noise/hot-pixel numbers below are meaningless)")
                else:
                    radial = radial_profile(master)
                    if verbose:
                        prof = "  ".join(f"{v:.2f}" for v in radial["annulus_mean_adu"])
                        print(f"   radial (centre->corner): {prof}")
                        print(f"   corner/centre ratio = {radial['corner_centre_ratio']} "
                              f"-> {radial['shape']}")
    finally:
        cap.stop()
    finished = datetime.now(timezone.utc).isoformat()

    # --- exposure reality check, from numbers we actually have ---
    # Real long integration forces the framerate down once exposure exceeds the
    # frame period; faked exposure (gain-boost / frame-sum) holds fps flat.
    # And in darkness mean should track exposure if dark current integrates.
    # --- the knee, derived from fps measurements ---
    # Prefer the low-bandwidth timing probe when there is one: the uncompressed
    # stream's fps is pinned by USB, so its knee is the BANDWIDTH ceiling, not
    # the integration knee, and using it over-excludes perfectly good points
    # (below a bandwidth-pinned frame period the integration still tracks the
    # commanded exposure -- the frame merely waits).
    if timing_ladder:
        knee_src = timing_ladder
        knee_src_name = (f"{timing_ladder[0].get('timing_format', '?')} timing "
                         "probe")
    else:
        knee_src = slope_data
        knee_src_name = f"{cap.pixfmt_str} dark ladder"
    knee_auto = derive_knee(knee_src)
    if knee is not None:
        knee_used = int(knee)
        knee_info = {"knee_units": knee_used, "source": "user override (--knee)",
                     "auto_detected": knee_auto}
        if knee_auto and knee_auto["knee_units"] != knee_used:
            knee_info["disagreement"] = (
                f"--knee {knee_used} overrides the measured knee of "
                f"{knee_auto['knee_units']} ({knee_auto['detail']}). A knee "
                "inherited from a different exposure range excludes good "
                "points or admits pedestal-only ones.")
    elif knee_auto:
        knee_used = knee_auto["knee_units"]
        knee_info = dict(knee_auto,
                         source=f"auto-detected from fps departure "
                                f"({knee_src_name})")
    else:
        # fps never left its pinned maximum: nothing here is integration-
        # limited, so there is no integrating region to restrict the fit to.
        knee_used = 0
        knee_info = {
            "knee_units": 0,
            "source": (f"not detectable -- fps never departed its pinned "
                       f"maximum ({knee_src_name})"),
            "auto_detected": None,
            "note": ("every ladder point is bandwidth-limited, so the dark-"
                     "current fit runs over the whole ladder and is not an "
                     "integration-limited measurement. Extend --exposure-max "
                     "or drop to a smaller resolution to push past the "
                     "bandwidth-limited frame period."
                     + ("" if timing_ladder else
                        " A low-bandwidth timing probe (MJPEG) would lift the "
                        "ceiling and expose the knee -- it is on by default "
                        "with --dark unless --no-timing-probe was passed."))}
    knee_info["measured_from"] = knee_src_name
    if verbose:
        print(f"\n   knee: {knee_used} units ({knee_info['source']})")
        if knee_auto:
            print(f"     {knee_auto['detail']}")
        if "disagreement" in knee_info:
            print(f"     !! {knee_info['disagreement']}")
        if "note" in knee_info:
            print(f"     {knee_info['note']}")

    expo = [d["exposure_units"] for d in slope_data]
    fpss = [d["eff_fps"] for d in slope_data]
    expfid = {"by_step": []}
    if len(slope_data) >= 2:
        for a, b in zip(slope_data[:-1], slope_data[1:]):
            er = b["exposure_units"] / a["exposure_units"] if a["exposure_units"] else float("nan")
            fr = (a["eff_fps"] / b["eff_fps"]) if b["eff_fps"] else float("nan")  # fps should DROP as expo rises
            mr = (b["mean_adu"] / a["mean_adu"]) if a["mean_adu"] else float("nan")
            expfid["by_step"].append({
                "from": a["exposure_units"], "to": b["exposure_units"],
                "expo_ratio": round(er, 3),
                "fps_drop_ratio": round(fr, 3),
                "mean_ratio": round(mr, 3)})
        if verbose:
            print("\n   exposure fidelity (per step):")
            print("     expo_x  fps_drop_x  mean_x   (ideal: fps_drop≈expo once past frame period)")
            for s in expfid["by_step"]:
                print(f"     {s['expo_ratio']:6.2f}  {s['fps_drop_ratio']:9.2f}  "
                      f"{s['mean_ratio']:7.2f}")
        # The verdict comes from the timing probe when there is one. In an
        # uncompressed stream the frame rate is pinned by USB, so fps cannot
        # respond to exposure and the honest answer there is always
        # "indeterminate (bandwidth-limited)" -- which is a statement about the
        # bus, not about the firmware's exposure handling.
        expfid["verdict_source"] = knee_src_name
        if timing_ladder:
            expfid["timing_probe"] = {
                "format": timing_ladder[0].get("timing_format"),
                "points": timing_ladder,
                "why": ("fps measured in a low-bandwidth format so the frame "
                        "period is set by integration rather than by the bus. "
                        "Timing only -- no pixel value from this pass is used "
                        "in any noise, FPN or defect figure.")}
        expfid.update(exposure_fidelity_verdict(knee_src, knee_used))
        if verbose:
            print(f"   -> [{knee_src_name}] {expfid['verdict']}")

    # dark current slope: fit ONLY the integrating region (exposure >= knee).
    # Below the knee the level is the clamped pedestal and carries no dark-
    # current information; including it corrupts the slope (the r2=0.82 problem).
    integ_pts = [d for d in slope_data if d["exposure_units"] >= knee_used]
    if len(integ_pts) >= 2:
        xs = np.array([d["exposure_units"] for d in integ_pts], float)
        ys = np.array([d["mean_adu"] for d in integ_pts], float)
        fit_region = f"integrating (>= {knee_used})"
    else:
        xs = np.array([d["exposure_units"] for d in slope_data], float)
        ys = np.array([d["mean_adu"] for d in slope_data], float)
        fit_region = "all points (too few above knee)"
    linfit = {}
    if len(xs) >= 2:
        A = np.vstack([xs, np.ones(len(xs))]).T
        slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
        # linearity check: how well does the line actually fit?
        pred = A @ np.array([slope, intercept])
        ss_res = float(np.sum((ys - pred)**2))
        ss_tot = float(np.sum((ys - ys.mean())**2)) or 1e-12
        r2 = 1.0 - ss_res / ss_tot
        linfit = {
            "dark_current_adu_per_unit": float(slope),
            "bias_offset_adu": float(intercept),
            "r2": r2,
            "linear": bool(r2 > 0.97),
            "fit_region": fit_region,
            "n_points": int(len(xs)),
        }
        # A slope of 0.00000 with r2=0.206 is not a dark current of zero, it is
        # a null result. Report the detection limit alongside it: the smallest
        # slope this ladder could have resolved. That is the number PHD2's
        # dark-scaling decision actually needs.
        deep_quant_limited = bool(deep_quant and deep_quant["quantiser_limited"])
        mds = minimum_detectable_slope(
            xs, ys, code_step, deep_quant_limited,
            resid_std=float(np.sqrt(ss_res / max(1, len(xs) - 2))))
        if mds:
            linfit["detection_limit"] = mds
            # the ladder is "flat" when the whole span moved less than the
            # smallest change that could have registered
            span_change = float(abs(slope) * (np.max(xs) - np.min(xs)))
            linfit["flat_within_detection_limit"] = bool(
                span_change < mds["min_detectable_level_change_adu"])
            if linfit["flat_within_detection_limit"]:
                linfit["dark_current_statement"] = (
                    "NOT DETECTED: " + mds["statement"])
            else:
                linfit["dark_current_statement"] = (
                    f"{slope:.5g} ADU/unit (detection limit "
                    f"{mds['min_detectable_slope_adu_per_unit']:.3g})")
        if not linfit["linear"]:
            print(f"   !! note: dark level vs exposure fit is weak (r2={r2:.3f}, "
                  f"{len(xs)} pts, {fit_region}). Either too few integrating-region "
                  "points to fit a slope, or thermal drift across the run. The "
                  "exposure-fidelity verdict above (from fps vs exposure) is the "
                  "reliable real-vs-synthetic indicator, not this fit.")
        if linfit.get("flat_within_detection_limit"):
            print(f"   !! dark level is FLAT across the whole ladder to within "
                  f"the detection limit -- reporting an upper bound, not a "
                  f"slope: {mds['statement']}")

    result = {
        "exposure_max_units": exposure_max,
        "exposure_max_ms": round(exposure_max * 0.1, 1),
        # pixel_depth is the domain of every ADU value in this report -- the
        # 0-255 luma float domain all formats are normalised into (Y16 /257).
        # PHD2's dark-model import keys on it; the capture path's native
        # depth is recorded separately below.
        "pixel_depth": 8,
        "capture_fourcc": cap.pixfmt_str,
        "native_bit_depth": 16 if cap.pixfmt_str == "Y16" else 8,
        "deep_stack_frames": dark_frames,
        "deep_stack_drops": deep_drops,
        "deep_stack_clipped": deep_clipped,
        "deep_stack_quantiser_limited": bool(
            deep_quant and deep_quant["quantiser_limited"]),
        "quantiser_check": deep_quant,
        "knee": knee_info,
        "master_dark_fpn_adu": fpn if master is not None else None,
        "hot_pixels_6sigma": hot if master is not None else None,
        "hot_pixel_coords": hot_coords,  # [[y, x], ...] — x y when writing PHD2 defect file
        # the same pixels, sorted by HOW they fail: stuck / intermittent hot /
        # hot / dark current. See classify_defects for why that matters.
        "defect_classes": defects,
        "bias_reference_exposure_units": bias_exposure,
        "irreducible_temporal_dark_noise_adu": residual_std,
        "radial_profile": radial if master is not None else None,
        "dark_current_fit": linfit,
        "exposure_fidelity": expfid,
        "ladder": slope_data,
        "stack_started_utc": started,
        "stack_finished_utc": finished,
    }
    # When the noise measurement has hit the quantiser floor, the measured
    # number is a floor, not a value. Carry the bound alongside it so nothing
    # downstream can read 0.000 as "remarkably quiet sensor".
    if deep_quant and deep_quant["quantiser_limited"]:
        result["irreducible_temporal_dark_noise_bound"] = noise_upper_bound(
            residual_std, code_step)
    return result, (master if master is not None else None)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def sanitize_tag_component(s, maxlen=32):
    s = re.sub(r'[^A-Za-z0-9._-]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s[:maxlen]


def _ctrl_capabilities_str(device):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"],
                           capture_output=True, text=True, timeout=10)
        entries = []
        for line in r.stdout.splitlines():
            m = re.match(r'\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)', line)
            if not m:
                continue
            name, ctype, rest = m.group(1), m.group(2), m.group(3)
            parts = [f"{name}({ctype})"]
            for k in ('min', 'max', 'step', 'default'):
                mm = re.search(rf'{k}=(-?\d+)', rest)
                if mm:
                    parts.append(f"{k}={mm.group(1)}")
            entries.append(':'.join(parts))
        return '|'.join(sorted(entries))
    except Exception:
        return ''


def _enum_advertised_modes(device):
    try:
        r = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"],
                           capture_output=True, text=True, timeout=10)
        modes, cur_fourcc, cur_w, cur_h = [], None, None, None
        for line in r.stdout.splitlines():
            # fourccs may be space-padded ('Y16 '); \w+ would miss them and
            # silently drop every mode of that format. Capture to the quote
            # and strip, so mono 16-bit devices enumerate correctly.
            m = re.search(r"\[\d+\]:\s+'([^']+)'", line)
            if m:
                cur_fourcc = m.group(1).strip(); cur_w = cur_h = None; continue
            m = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
            if m and cur_fourcc:
                cur_w, cur_h = int(m.group(1)), int(m.group(2)); continue
            m = re.search(r"Interval:\s+Discrete\s+[\d.]+s\s+\(([\d.]+)\s*fps\)", line)
            if m and cur_fourcc and cur_w is not None:
                fps = float(m.group(1))
                hit = next((x for x in modes if x['fourcc'] == cur_fourcc
                            and x['width'] == cur_w and x['height'] == cur_h), None)
                if hit:
                    if fps > hit['max_fps']:
                        hit['max_fps'] = fps
                else:
                    modes.append({'fourcc': cur_fourcc, 'width': cur_w,
                                  'height': cur_h, 'max_fps': fps})
        return modes
    except Exception:
        return []


def device_capability_hash(device, adv=None):
    """Stable 12-char fingerprint of the device's ADVERTISED capabilities: the
    mode list (fourcc + resolution + max fps) and the control ranges
    (name/type/min/max/step/default -- never current values). Independent of
    VID:PID and descriptor strings, so it tells apart two products that share a
    generic bridge ID: a 4K sensor and a 1080p sensor advertise different
    resolutions, so their mode lists -- and therefore this hash -- differ.

    MUST stay byte-identical to cam_manager.py's copy, or the device tags the
    two tools build will diverge and their output paths won't line up."""
    if adv is None:
        adv = _enum_advertised_modes(device)
    modes_str = '|'.join(
        f"{m['fourcc']}:{m['width']}x{m['height']}@{m['max_fps']}"
        for m in sorted(adv, key=lambda m: (m['fourcc'], m['width'], m['height']))
    )
    return hashlib.sha1(
        f"{modes_str};{_ctrl_capabilities_str(device)}".encode()
    ).hexdigest()[:12]


def get_usb_device_info(device):
    name = os.path.basename(device)
    try:
        iface = os.path.realpath(f"/sys/class/video4linux/{name}/device")
        usb_dev = os.path.dirname(iface)

        def _rd(fname):
            try:
                v = open(os.path.join(usb_dev, fname)).read().strip()
                return v if v else None
            except OSError:
                return None

        vid = _rd('idVendor')
        pid = _rd('idProduct')
        if not vid or not pid:
            return None

        bus_raw = _rd('busnum')
        dev_raw = _rd('devnum')

        driver = None
        drv = os.path.join(iface, 'driver')
        if os.path.islink(drv):
            driver = os.path.basename(os.path.realpath(drv))

        adv = _enum_advertised_modes(device)
        cap_hash = device_capability_hash(device, adv)

        return {
            'usb_vendor_id':    vid.lower(),
            'usb_product_id':   pid.lower(),
            'usb_manufacturer': _rd('manufacturer'),
            'usb_product':      _rd('product'),
            'usb_serial':       _rd('serial'),
            'usb_bcd_device':   _rd('bcdDevice'),
            'usb_bus_num':      bus_raw.zfill(3) if bus_raw else None,
            'usb_dev_num':      dev_raw.zfill(3) if dev_raw else None,
            'kernel_driver':    driver,
            'advertised_modes': adv,
            'capability_hash':  cap_hash,
        }
    except Exception:
        return None


def build_device_tag(usb_info):
    """Filename-safe device tag: {vid}{pid}_{serial}_{capability_hash}.

    The capability hash is appended because the bridge serial alone collides:
    1bcf:28c4 reports serial '01.00.00' (a firmware string, not a per-unit id)
    on both an IMX678 and an IMX385 module. The capability hash carries the
    advertised mode list (incl. resolution), so the 4K and 1080p sensors get
    distinct tags and their output dirs/reports never overwrite each other."""
    if not usb_info:
        return None
    vid = usb_info.get('usb_vendor_id') or ''
    pid = usb_info.get('usb_product_id') or ''
    serial = usb_info.get('usb_serial')
    if serial:
        suffix = sanitize_tag_component(serial)
    else:
        mfr = usb_info.get('usb_manufacturer') or ''
        prd = usb_info.get('usb_product') or ''
        bcd = usb_info.get('usb_bcd_device') or ''
        h = hashlib.sha1((mfr + prd + bcd).encode()).hexdigest()[:8]
        suffix = f"NOSERIAL-{h}"
    caphash = usb_info.get('capability_hash')
    if caphash:
        suffix = f"{suffix}_{caphash}"
    return f"{vid}{pid}_{suffix}"


_NAMES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_names.json")


def load_device_names():
    try:
        with open(_NAMES_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_device_name(tag, name):
    names = load_device_names()
    names[tag] = name
    with open(_NAMES_FILE, "w") as f:
        json.dump(names, f, indent=2)


def _ensure_dir(path):
    """Create the parent directory of an output file. Save targets may carry a
    device-tag prefix (046d0825_SN123/master_640x480.npy); failing on a missing
    directory AFTER an hours-long capture run loses the whole stack."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# Processing path provenance
#
# Every one of these reshapes the output, and the defect map moved 12 -> 1057 on
# gamma alone. A master dark or defect map built at one setting and auto-loaded
# at another is silently wrong, so the settings travel with the data.
# ----------------------------------------------------------------------------

PROCESSING_CTRLS = ("gamma", "brightness", "contrast", "sharpness",
                    "saturation", "hue", "gain", "backlight_compensation",
                    "auto_exposure", "exposure_time_absolute")


def read_processing_path(device):
    """Current value of every control that reshapes the output, read back from
    the device. Absent controls are simply omitted; the key set is therefore
    also a record of what this camera exposes.

    Shared with cam_observe (which stamps it into dark_meta_WxH.json) so both
    tools record provenance in exactly the same vocabulary."""
    out = {}
    try:
        txt = list_controls(device)
    except Exception:
        return out
    for line in txt.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\(\w+\)\s*:\s*(.*)", line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        if name not in PROCESSING_CTRLS:
            continue
        mm = re.search(r"value=(-?\d+)", rest)
        if mm:
            out[name] = int(mm.group(1))
    return out


# ----------------------------------------------------------------------------
# Report comparison (--compare)
#
# Reflashing firmware is routine here, so before/after diffing is the most-used
# operation there is. The gamma result sat in two reports and took a manual read
# to spot; that is exactly the comparison this automates.
# ----------------------------------------------------------------------------

def _report_processing(report):
    """Processing path of a report, preferring the recorded readback and
    falling back to parsing argv for reports written before it was stamped."""
    proc = dict(report.get("processing_path") or {})
    if proc:
        return proc, "readback"
    argv = report.get("argv") or []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            name = a[2:]
            if "=" in name:
                name, val = name.split("=", 1)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                val = argv[i + 1]
                i += 1
            else:
                val = "(set)"
            key = name.replace("-", "_")
            if key in PROCESSING_CTRLS or key in (
                    "gamma", "brightness", "contrast", "sharpness",
                    "saturation", "hue", "gain"):
                proc[key] = val
        i += 1
    return proc, "argv"


def _fmt_delta(a, b, width=9, prec=3):
    """b - a, rendered so a zero delta is visibly zero and a missing side is
    visibly missing."""
    if a is None or b is None:
        return " " * (width - 1) + "-"
    d = b - a
    return f"{d:+{width}.{prec}f}"


def compare_reports(path_a, path_b, verbose=True):
    """Per-point deltas between two characterisation reports.

    Ladder points are matched on exposure_units; anything present in only one
    report is listed separately rather than silently dropped, because a changed
    ladder is itself a difference worth seeing."""
    with open(path_a) as fh:
        ra = json.load(fh)
    with open(path_b) as fh:
        rb = json.load(fh)

    da = ra.get("dark") or {}
    db = rb.get("dark") or {}
    la = {d["exposure_units"]: d for d in (da.get("ladder") or [])}
    lb = {d["exposure_units"]: d for d in (db.get("ladder") or [])}
    common = sorted(set(la) & set(lb))
    only_a = sorted(set(la) - set(lb))
    only_b = sorted(set(lb) - set(la))

    pa, sa = _report_processing(ra)
    pb, sb = _report_processing(rb)
    proc_keys = sorted(set(pa) | set(pb))
    proc_diff = {k: [pa.get(k), pb.get(k)] for k in proc_keys
                 if str(pa.get(k)) != str(pb.get(k))}

    def _summary(r, d):
        dcf = d.get("dark_current_fit") or {}
        return {
            "label": os.path.basename(r.get("_path", "")),
            "timestamp_utc": r.get("timestamp_utc"),
            "device_name": r.get("device_name"),
            "resolution": f"{r.get('width')}x{r.get('height')}",
            "format": r.get("measurement_format"),
            "hot_pixels": d.get("hot_pixels_6sigma"),
            "fpn_adu": d.get("master_dark_fpn_adu"),
            "temporal_noise_adu": d.get("irreducible_temporal_dark_noise_adu"),
            "dark_current_adu_per_unit": dcf.get("dark_current_adu_per_unit"),
            "r2": dcf.get("r2"),
            "knee_units": (d.get("knee") or {}).get("knee_units"),
            "clipped": d.get("deep_stack_clipped"),
            "quantiser_limited": d.get("deep_stack_quantiser_limited"),
            "distinct_codes_master":
                (d.get("quantiser_check") or {}).get("distinct_codes_master"),
        }

    ra["_path"], rb["_path"] = path_a, path_b
    sum_a, sum_b = _summary(ra, da), _summary(rb, db)

    per_point = []
    for e in common:
        A, B = la[e], lb[e]
        per_point.append({
            "exposure_units": e,
            "mean_adu": [A.get("mean_adu"), B.get("mean_adu")],
            "temporal_noise_adu": [A.get("temporal_noise_adu"),
                                   B.get("temporal_noise_adu")],
            "eff_fps": [A.get("eff_fps"), B.get("eff_fps")],
            "distinct_codes_stack": [A.get("distinct_codes_stack"),
                                     B.get("distinct_codes_stack")],
        })

    result = {"a": sum_a, "b": sum_b,
              "processing_path": {"a": pa, "b": pb, "a_source": sa,
                                  "b_source": sb, "differs": proc_diff},
              "per_point": per_point,
              "exposures_only_in_a": only_a,
              "exposures_only_in_b": only_b}

    if not verbose:
        return result

    print("=== COMPARE ===")
    print(f"  A: {path_a}")
    print(f"     {sum_a['timestamp_utc']}  {sum_a['device_name'] or '(unnamed)'}"
          f"  {sum_a['resolution']} {sum_a['format']}")
    print(f"  B: {path_b}")
    print(f"     {sum_b['timestamp_utc']}  {sum_b['device_name'] or '(unnamed)'}"
          f"  {sum_b['resolution']} {sum_b['format']}")
    if sum_a["resolution"] != sum_b["resolution"]:
        print("  !! different resolutions -- per-pixel numbers are not "
              "comparable across a resolution change")

    print(f"\n--- processing path (A from {sa}, B from {sb}) ---")
    if proc_diff:
        for k, (va, vb) in sorted(proc_diff.items()):
            print(f"  {k:24s} {str(va):>10s} -> {str(vb):>10s}   <== CHANGED")
    else:
        print("  identical")
    for k in proc_keys:
        if k not in proc_diff:
            print(f"  {k:24s} {str(pa.get(k)):>10s}     (same)")

    print("\n--- per ladder point (B - A) ---")
    print("   exposure      mean_A    mean_B     d_mean   "
          "noise_A  noise_B    d_noise    fps_A   fps_B     d_fps  codes A/B")
    for p in per_point:
        ma, mb = p["mean_adu"]
        na, nbv = p["temporal_noise_adu"]
        fa, fb = p["eff_fps"]
        ca, cb = p["distinct_codes_stack"]
        print(f"   {p['exposure_units']:8d}  {ma:8.3f}  {mb:8.3f}  "
              f"{_fmt_delta(ma, mb)}  {na:7.4f}  {nbv:7.4f}  "
              f"{_fmt_delta(na, nbv, 9, 4)}  {fa:6.1f}  {fb:6.1f}  "
              f"{_fmt_delta(fa, fb, 8, 2)}  "
              f"{'-' if ca is None else ca}/{'-' if cb is None else cb}")
    if only_a:
        print(f"   exposures only in A: {only_a}")
    if only_b:
        print(f"   exposures only in B: {only_b}")

    print("\n--- deep-stack summary (B - A) ---")
    rows = [("hot pixels (>6sigma)", "hot_pixels", 0),
            ("master FPN (ADU)", "fpn_adu", 3),
            ("temporal noise (ADU)", "temporal_noise_adu", 4),
            ("dark current (ADU/unit)", "dark_current_adu_per_unit", 6),
            ("dark-current fit r2", "r2", 3),
            ("integration knee (units)", "knee_units", 0),
            ("distinct codes in master", "distinct_codes_master", 0)]
    for label, key, prec in rows:
        va, vb = sum_a.get(key), sum_b.get(key)
        if va is None and vb is None:
            continue
        sva = "-" if va is None else f"{va:.{prec}f}"
        svb = "-" if vb is None else f"{vb:.{prec}f}"
        print(f"  {label:26s} {sva:>12s} -> {svb:>12s}   "
              f"{_fmt_delta(va, vb, 12, prec)}")
    for label, key in (("deep stack CLIPPED", "clipped"),
                       ("QUANTISER-LIMITED", "quantiser_limited")):
        va, vb = bool(sum_a.get(key)), bool(sum_b.get(key))
        if va or vb:
            print(f"  {label:26s} {str(va):>12s} -> {str(vb):>12s}"
                  + ("   <== CHANGED" if va != vb else ""))

    # The headline: a defect-count swing with a processing change behind it is
    # the whole reason this mode exists.
    ha, hb = sum_a["hot_pixels"], sum_b["hot_pixels"]
    if ha is not None and hb is not None and ha != hb and proc_diff:
        print(f"\n  -> hot-pixel count moved {ha} -> {hb} across a processing "
              f"change ({', '.join(sorted(proc_diff))}). The defect map is a "
              "function of the processing path, not just the sensor: a map "
              "built at one setting is wrong at the other.")
    return result


# ----------------------------------------------------------------------------
# Lit transfer curve (--transfer)
# ----------------------------------------------------------------------------

def _stack_stats(cap, path, n, discard):
    """Streaming per-pixel mean + temporal noise + distinct-code count over a
    capture file. Same Welford accumulation the dark ladder uses, so transfer
    points and dark points are measured identically."""
    count = 0
    mean_px = m2_px = None
    code_seen = np.zeros(65536, dtype=bool)
    for fr in cap.iter_luma(path, n, discard=discard):
        count += 1
        if mean_px is None:
            mean_px = np.zeros_like(fr)
            m2_px = np.zeros_like(fr)
        delta = fr - mean_px
        mean_px += delta / count
        m2_px += delta * (fr - mean_px)
        codes = np.rint(fr * 257.0).astype(np.int32)
        np.clip(codes, 0, 65535, out=codes)
        code_seen[codes.ravel()] = True
        del fr, delta, codes
    if count == 0:
        return None
    temporal = (float(np.sqrt(m2_px / (count - 1)).mean()) if count > 1
                else float("nan"))
    return {"count": count, "mean_px": mean_px, "temporal": temporal,
            "distinct_codes": int(code_seen.sum())}


def fit_transfer_curve(points, code_step=1.0):
    """Classify the ISP transfer function from a panel-level sweep.

    Two competing descriptions of what a gamma control does: it either SCALES
    the output (a straight line through the pedestal, slope changes) or it
    RESHAPES the tone curve (a power law, curvature changes). Fit both over the
    unclipped region and let the residuals decide.

    The exponent is the product of the display's transfer and the ISP's, since
    the stimulus is an 8-bit grey on an uncalibrated screen. That confound
    CANCELS when two sweeps taken on the same display are compared, which is
    what --compare is for -- so the absolute exponent is indicative and the
    difference between two settings is the measurement."""
    usable = [p for p in points
              if not p.get("clipped") and p["mean_adu"] < 250.0]
    if len(usable) < 4:
        return {"ok": False, "reason": "need >=4 unclipped sweep points"}
    lv = np.array([p["level"] for p in usable], float)
    my = np.array([p["mean_adu"] for p in usable], float)

    def _r2(y, pred):
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
        return 1.0 - ss_res / ss_tot

    A = np.vstack([lv, np.ones(len(lv))]).T
    slope, intercept = np.linalg.lstsq(A, my, rcond=None)[0]
    r2_lin = _r2(my, A @ np.array([slope, intercept]))

    # pedestal: the output with the panel black, which is the offset the power
    # law sits on top of
    black = float(min(p["mean_adu"] for p in points
                      if p["level"] == min(p["level"] for p in points)))
    m = (lv > 0) & ((my - black) > 0)
    fit_pow = {"ok": False, "reason": "not enough positive points above the "
                                      "pedestal for a log-log fit"}
    if m.sum() >= 4:
        lx = np.log(lv[m])
        ly = np.log(my[m] - black)
        Ap = np.vstack([lx, np.ones(len(lx))]).T
        expo, cpow = np.linalg.lstsq(Ap, ly, rcond=None)[0]
        r2_pow = _r2(ly, Ap @ np.array([expo, cpow]))
        fit_pow = {"ok": True, "exponent": float(expo),
                   "log_intercept": float(cpow), "r2_loglog": float(r2_pow),
                   "n_points": int(m.sum())}

    out = {"ok": True,
           "n_points_used": len(usable),
           "pedestal_adu": black,
           "linear_fit": {"slope_adu_per_level": float(slope),
                          "intercept_adu": float(intercept),
                          "r2": float(r2_lin)},
           "power_fit": fit_pow,
           "display_gamma_caveat": (
               "the panel is an 8-bit grey on an uncalibrated display, so the "
               "exponent below is (display transfer) x (ISP transfer). Compare "
               "two sweeps taken on the SAME display with --compare to cancel "
               "the display term; the difference is the ISP's contribution.")}

    if fit_pow.get("ok"):
        e = fit_pow["exponent"]
        if abs(e - 1.0) < 0.08 and r2_lin > 0.985:
            out["verdict"] = (
                f"LINEAR within measurement: exponent {e:.3f}, straight-line "
                f"r2 {r2_lin:.4f}. At this setting the path SCALES the signal; "
                "it does not reshape the tone curve.")
        elif fit_pow["r2_loglog"] > r2_lin:
            out["verdict"] = (
                f"POWER LAW: exponent {e:.3f} (log-log r2 "
                f"{fit_pow['r2_loglog']:.4f} beats straight-line r2 "
                f"{r2_lin:.4f}). The path RESHAPES the tone curve -- this is a "
                "LUT, not a gain.")
        else:
            out["verdict"] = (
                f"mixed: exponent {e:.3f} but the straight line fits at least "
                f"as well (r2 {r2_lin:.4f} vs log-log "
                f"{fit_pow['r2_loglog']:.4f}). Sweep more levels, especially "
                "in the bottom quarter where a LUT bends hardest.")
    return out


def run_transfer(cap, device, levels, frames=8, settle_s=0.8, port=8088,
                 discard=2, wait_client=120.0, verbose=True):
    """Sweep the software flat panel and record mean output at each level.

    This is the measurement that settles what a processing control actually
    does. The PTC needs a controllable source and the flat panel already is
    one, so stepping its level and reading the sensor gives the ISP transfer
    function directly -- and it is the only way to put a measured curve next to
    a firmware LUT and compare entry for entry.

    The panel is served over HTTP; point any browser at the printed URL. Each
    level waits for the browser to ACK that it is painted before capturing, so
    a slow display cannot leave the previous level in the frame."""
    import cam_panel

    print("\n=== LIT TRANSFER CURVE ===")
    print("Aim the camera at the panel display, slightly defocused, filling "
          "the frame.")
    print("Keep EXPOSURE and GAIN fixed; only the panel level moves.\n")

    holder = {"level": int(levels[0])}
    panel = cam_panel.PanelServer(lambda: holder["level"], None, port)
    points = []
    try:
        for u in cam_panel.panel_urls(port):
            print(f"   open the panel at: {u}")
        print(f"   waiting up to {wait_client:.0f}s for a browser to connect...")
        if not panel.wait_for_client(timeout=wait_client):
            raise RuntimeError(
                f"no browser connected to the panel on port {port}. Open one "
                "of the URLs above on the display you want to use as the "
                "source, then re-run.")
        print("   panel client connected.\n")

        for lv in levels:
            lv = int(max(0, min(255, lv)))
            holder["level"] = lv
            acked = panel.wait_shown(lv, timeout=10.0)
            time.sleep(settle_s)
            path, total, full = cap.capture_run(
                frames, discard=discard, timeout=30.0, verbose=False)
            st = _stack_stats(cap, path, frames, discard)
            try:
                os.remove(path)
            except OSError:
                pass
            if st is None:
                print(f"   level {lv:3d}: no frames -- skipped")
                continue
            mean_px = st["mean_px"]
            mean_level = float(mean_px.mean())
            floor_frac = float((mean_px <= 0.5).mean())
            ceil_frac = float((mean_px >= 254.5).mean())
            clipped = (floor_frac > 0.5 or ceil_frac > 0.5)
            rec = {"level": lv,
                   "mean_adu": mean_level,
                   "temporal_noise_adu": st["temporal"],
                   "distinct_codes": st["distinct_codes"],
                   "frames": st["count"],
                   "eff_fps": round(cap.last_fps, 2),
                   "clipped": clipped,
                   "clip_floor_frac": round(floor_frac, 4),
                   "clip_ceil_frac": round(ceil_frac, 4),
                   "panel_ack": bool(acked)}
            points.append(rec)
            if verbose:
                flag = ""
                if clipped:
                    flag = ("  !! CLIPPED at "
                            + ("floor" if floor_frac > ceil_frac else "ceiling"))
                if not acked:
                    flag += "  (no panel ACK -- level may not have been painted)"
                print(f"   level={lv:3d}  mean={mean_level:7.3f} ADU  "
                      f"noise={st['temporal']:6.3f}  "
                      f"codes={st['distinct_codes']:3d}{flag}")
    finally:
        panel.stop()

    fit = fit_transfer_curve(points)
    return {"points": points, "fit": fit,
            "panel_port": port, "frames_per_level": frames,
            "settle_s": settle_s}


def main():
    ap = argparse.ArgumentParser(description="UVC camera noise/codec characteriser")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--name", default=None,
                    help="human-readable camera name; saved against the device tag "
                    "and auto-loaded on future runs for the same device")
    ap.add_argument("--identify", action="store_true",
                    help="print full USB identity + capability hash + device tag "
                    "for this device, then exit. Run on each unit to compare two "
                    "cameras that share a VID:PID (different capability_hash = "
                    "different sensor/firmware).")
    ap.add_argument("--list", action="store_true",
                    help="enumerate + rank formats and pick measurement format")
    ap.add_argument("--ptc", action="store_true", help="run photon transfer curve")
    ap.add_argument("--dark", action="store_true", help="run dark characterisation")
    ap.add_argument("--fps-probe", action="store_true",
                    help="framerate-vs-exposure sweep only, in a low-bandwidth "
                    "format (MJPEG by default). Timing is all this measures, "
                    "so the lossy codec is harmless and its reduced payload "
                    "lifts the bandwidth ceiling until the integration knee "
                    "becomes visible. No pixel data is read.")
    ap.add_argument("--timing-format", default=None,
                    help="fourcc to use for the framerate probe (default: "
                    "MJPG if the device offers it, else the most compressed "
                    "format available). Never used for pixel measurements.")
    ap.add_argument("--no-timing-probe", action="store_true",
                    help="do NOT run the MJPEG timing probe before --dark; "
                    "derive the knee and the real-vs-synthetic verdict from "
                    "the uncompressed ladder's own fps instead (usually "
                    "bandwidth-pinned and therefore indeterminate)")
    ap.add_argument("--timing-frames", type=int, default=30,
                    help="frames per exposure in the framerate probe "
                    "(default 30; more = tighter fps estimate)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    default=None,
                    help="diff two reports: per-ladder-point deltas in mean, "
                    "temporal noise, fps and distinct codes, plus the "
                    "deep-stack summary and the processing path behind it. "
                    "Touches no hardware.")
    ap.add_argument("--transfer", action="store_true",
                    help="lit transfer curve: sweep the software flat panel's "
                    "level and record mean output, giving the ISP transfer "
                    "function. Serves the panel over HTTP -- open the printed "
                    "URL on the display you are using as the source.")
    ap.add_argument("--transfer-levels", default=None,
                    help="comma-separated 8-bit panel levels to sweep "
                    "(default: 17 points, denser at the bottom where a LUT "
                    "bends hardest)")
    ap.add_argument("--transfer-frames", type=int, default=8,
                    help="frames captured at each panel level (default 8)")
    ap.add_argument("--transfer-settle", type=float, default=0.8,
                    help="extra settle seconds after the panel ACKs a level, "
                    "for the camera's own pipeline (default 0.8)")
    ap.add_argument("--panel-port", type=int, default=8088,
                    help="port for the HTTP flat panel used by --transfer "
                    "(default 8088)")
    ap.add_argument("--auto", action="store_true",
                    help="auto-exposure observation: leave AE on, cap the lens, "
                    "log AE-chosen exposure/gain and framerate over time")
    ap.add_argument("--auto-iters", type=int, default=20,
                    help="number of AE observation iterations (default 20)")
    ap.add_argument("--auto-frames", type=int, default=30,
                    help="frames per AE iteration (default 30; more = better "
                    "framerate estimate)")
    ap.add_argument("--save-calib", default=None,
                    help="(with --dark) write an fps<->exposure calibration JSON "
                    "from the integrating region, for later --auto fps inversion")
    ap.add_argument("--calib", default=None,
                    help="(with --auto) load an fps<->exposure calibration to "
                    "convert AE framerate into effective exposure")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--format", default=None,
                    help="force fourcc (e.g. YUYV). Default: best uncompressed.")
    ap.add_argument("--exposure", type=int, default=200,
                    help="PTC exposure in raw control units (100us each on UVC)")
    ap.add_argument("--gain", type=int, default=None)
    ap.add_argument("--gamma", type=int, default=None,
                    help="set gamma (e.g. 100 for linear); default leaves it alone")
    ap.add_argument("--brightness", type=int, default=None,
                    help="set brightness (neutral is usually 0); default untouched")
    ap.add_argument("--contrast", type=int, default=None,
                    help="set contrast (neutral is usually 0); default untouched")
    ap.add_argument("--sharpness", type=int, default=None,
                    help="set sharpness (use min to avoid edge processing); "
                    "default untouched")
    ap.add_argument("--saturation", type=int, default=None,
                    help="set saturation (irrelevant on mono); default untouched")
    ap.add_argument("--exposure-max", type=int, default=8192,
                    help="max exposure (raw units) for the dark deep stack")
    ap.add_argument("--dark-frames", type=int, default=128,
                    help="frames in the deep (max-exposure) dark stack")
    ap.add_argument("--ladder-frames", type=int, default=16,
                    help="frames captured at each non-deep ladder point "
                    "(default 16)")
    ap.add_argument("--ladder-points", type=int, default=7,
                    help="number of points auto-spaced across the integrating "
                    "region for the default ladder (ignored if --ladder given). "
                    "Default 7.")
    ap.add_argument("--slope-points", type=int, default=5,
                    help="(legacy) number of shorter exposures; ignored when "
                    "--ladder is given")
    ap.add_argument("--ladder", default=None,
                    help="explicit comma-separated exposure values (raw units), "
                    "e.g. 410,2458,3277,4096,4915,5734,6554,7373,8192. "
                    "Overrides the default sweep.")
    ap.add_argument("--discard", type=int, default=2,
                    help="frames to skip at the start of each exposure (settle). "
                    "Default 2.")
    ap.add_argument("--knee", type=int, default=None,
                    help="OVERRIDE the exposure (raw units) above which real "
                    "integration begins; points below only set the bias and "
                    "are excluded from the dark-current slope fit. Default is "
                    "to DERIVE it from the ladder's own fps measurements (the "
                    "knee is where fps first departs its pinned maximum). Only "
                    "pass this to override a bad detection -- a knee carried "
                    "over from a different firmware's exposure range excludes "
                    "good points or admits pedestal-only ones.")
    ap.add_argument("--save-master", default=None,
                    help="write master dark as uint16 .npy (scaled 0-65535 for PHD2)")
    ap.add_argument("--save-defects", default=None,
                    help="write hot pixel coordinates in PHD2 defect map format (x y per line)")
    ap.add_argument("--defect-classes", default="stuck,intermittent_hot,hot",
                    help="which defect classes go into the PHD2 defect map. "
                    "Default writes the pixels that need INTERPOLATION and "
                    "omits dark_current, which a scaled dark subtraction "
                    "already handles. Known classes: stuck, intermittent_hot, "
                    "hot, dark_current. The full per-class breakdown is always "
                    "written alongside as <file>.classes.json.")
    ap.add_argument("--save-dark-model", default=None,
                    help="write dark current model JSON for PHD2 auto-import at camera connect")
    ap.add_argument("--report", default=None, help="write json report here")
    ap.add_argument("--report-dir", default=None,
                    help="directory to auto-write a timestamped JSON report into "
                    "(filename encodes mode/resolution/timestamp)")
    args = ap.parse_args()

    # --compare reads two files and touches no hardware, so it runs before any
    # device probing (which would fail on a machine that has no camera attached
    # -- exactly where you want to diff two reports).
    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
        return

    report = {"device": args.device,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "argv": sys.argv[1:],
              "width": args.width, "height": args.height}

    usb_info = get_usb_device_info(args.device)
    report["usb_device"] = usb_info
    _usb_warn = (None if usb_info is not None
                 else f"WARNING: could not resolve USB device identity via sysfs for {args.device}")
    _dev_tag = build_device_tag(usb_info)

    _names = load_device_names()
    _device_name = args.name
    if _device_name and _dev_tag:
        save_device_name(_dev_tag, _device_name)
        print(f"   device name '{_device_name}' saved for {_dev_tag}")
    elif not _device_name and _dev_tag and _dev_tag in _names:
        _device_name = _names[_dev_tag]
        print(f"   device name: '{_device_name}'")
    report["device_name"] = _device_name

    def _write_report():
        """Write the report to --report (explicit path) and/or --report-dir
        (timestamped auto-name). Safe to call at any exit point."""
        if args.report:
            _ensure_dir(args.report)
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)
            print(f"\nreport written to {args.report}")
        if args.report_dir:
            os.makedirs(args.report_dir, exist_ok=True)
            mode = ("auto" if args.auto else "transfer" if args.transfer
                    else "ptc" if args.ptc
                    else "dark" if args.dark else "list")
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            tag_part = f"{_dev_tag}_" if _dev_tag else ""
            fn = f"camchar_{tag_part}{mode}_{args.width}x{args.height}_{ts}.json"
            path = os.path.join(args.report_dir, fn)
            json.dump(report, open(path, "w"), indent=2)
            print(f"\nreport written to {path}")

    # --- identify: print identity + capability hash and exit ---
    if args.identify:
        print(f"\n=== IDENTITY for {args.device} ===")
        print(f"  device_name : {_device_name or '(unnamed)'}")
        print(f"  device_tag  : {_dev_tag or '(unresolved)'}")
        if usb_info is None:
            print(f"  {_usb_warn}")
        else:
            for k in ("usb_vendor_id", "usb_product_id", "usb_manufacturer",
                      "usb_product", "usb_serial", "usb_bcd_device",
                      "kernel_driver", "capability_hash"):
                print(f"  {k:16s}: {usb_info.get(k)}")
            modes = usb_info.get("advertised_modes") or []
            print(f"  advertised_modes: {len(modes)}")
            for m in modes:
                print(f"      {m['fourcc']:5s} {m['width']}x{m['height']} "
                      f"@ {m['max_fps']}fps")
        _write_report()
        return

    # --- formats ---
    formats = enum_formats(args.device)
    ranked = rank_formats(formats)
    report["formats_ranked"] = ranked
    best = best_measurement_format(ranked)

    print(f"=== FORMATS on {args.device} (ranked best-for-measurement first) ===")
    for f in ranked:
        tag = "  <== MEASURE WITH THIS" if f is best else ""
        print(f"  [{f['fidelity_rank']}] {f['fourcc']}  "
              f"{f['fidelity_note']:42s} ({f['description']}){tag}")
    if best is None:
        print("\n!! No uncompressed format available. This camera only offers "
              "lossy codecs; sensor noise CANNOT be measured meaningfully. "
              "Stop here.")
        _write_report()
        return

    meas_fmt = args.format or best["fourcc"]
    report["measurement_format"] = meas_fmt
    _adv = (usb_info or {}).get('advertised_modes') or []
    report["is_native_advertised_mode"] = (
        any(m['fourcc'] == meas_fmt.strip() and m['width'] == args.width
            and m['height'] == args.height for m in _adv)
        if usb_info is not None else None
    )
    # A resolution the device does not advertise is produced by the bridge
    # scaling/cropping a different native readout. The interpolation seam shows
    # up as edge-column "hot pixels" and inflated FPN, so the defect map and
    # FPN from such a run are artefacts, not the sensor. Warn loudly.
    _nonnative_warn = None
    if report["is_native_advertised_mode"] is False:
        native = ", ".join(sorted({f"{m['width']}x{m['height']}"
                                   for m in _adv if m['fourcc'] == meas_fmt.strip()})) \
                 or "(none advertised for this format)"
        _nonnative_warn = (
            f"WARNING: {meas_fmt} {args.width}x{args.height} is NOT an advertised "
            f"native mode (advertised: {native}). The bridge is scaling/cropping; "
            "defect map and FPN will be rescaling artefacts, not the sensor. "
            "Re-run at an advertised native resolution.")
        print(f"\n!! {_nonnative_warn}")

    if args.list and not (args.ptc or args.dark or args.auto or args.transfer
                          or args.fps_probe):
        _write_report()
        return

    def _timing_ladder(exp_ladder):
        """Run the low-bandwidth framerate probe over `exp_ladder`, or explain
        why it could not run. Returns (records_or_None, note)."""
        if args.timing_format:
            pick = {"fourcc": args.timing_format.strip(),
                    "advertised_at_size": None}
        else:
            pick = pick_timing_format(ranked, _adv, args.width, args.height)
        if not pick:
            note = ("no compressed format available at "
                    f"{args.width}x{args.height}: the framerate probe cannot "
                    "lift the bandwidth ceiling here, so the exposure-fidelity "
                    "verdict falls back to the uncompressed ladder and will "
                    "likely be indeterminate. Try a smaller resolution.")
            print(f"\n!! {note}")
            return None, note
        fcc = pick["fourcc"]
        if pick.get("advertised_at_size") is False:
            print(f"   note: {fcc} is not advertised at {args.width}x"
                  f"{args.height}; attempting anyway")
        try:
            recs = run_fps_probe(args.device, args.width, args.height, fcc,
                                 exp_ladder, frames=args.timing_frames,
                                 discard=args.discard)
        except Exception as e:
            note = f"framerate probe failed ({fcc}): {e}"
            print(f"\n!! {note}")
            return None, note
        if not any(r["eff_fps"] > 0 for r in recs):
            note = (f"framerate probe in {fcc} returned no usable timestamps; "
                    "falling back to the uncompressed ladder's fps")
            print(f"\n!! {note}")
            return None, note
        return recs, f"framerate measured in {fcc} (timing only)"

    # --- auto-exposure observation mode (do NOT force manual) ---
    if args.auto:
        print("\n=== Auto-exposure mode: leaving AE ENABLED (not forcing manual) ===")
        calib = None
        if args.calib:
            try:
                calib = json.load(open(args.calib))
                print(f"   loaded fps<->exposure calibration from {args.calib} "
                      f"(valid fps {calib.get('valid_fps_range')})")
                # A calibration is only valid in the mode it was built in: the
                # bandwidth ceiling moves with format and frame size, so
                # inverting AE's fps through the wrong one yields a confident
                # wrong exposure.
                cf, cw, ch = (calib.get("fourcc"), calib.get("width"),
                              calib.get("height"))
                if cf and (cf.strip() != meas_fmt.strip()
                           or cw != args.width or ch != args.height):
                    print(f"   !! calibration was built for {cf} {cw}x{ch} but "
                          f"this run is {meas_fmt} {args.width}x{args.height}. "
                          "The fps<->exposure mapping is format- and size-"
                          "specific; rebuild it in this mode or the implied "
                          "exposures below are wrong.")
                elif not cf:
                    print("   note: calibration predates format stamping -- "
                          "confirm it was built in this same mode")
            except Exception as e:
                print(f"   could not load calibration {args.calib}: {e}")
        cap = Capture(args.device, args.width, args.height, meas_fmt)
        try:
            ae = run_auto_dark(cap, args.device,
                               iterations=args.auto_iters,
                               frames=args.auto_frames,
                               calib=calib)
        finally:
            cap.close()
        # AE control readback is unreliable across every bridge tested: the
        # exposure_time_absolute readback disagrees with the fps-implied
        # exposure. Make the recommendation explicit and machine-readable so
        # downstream tooling keys off fps, not the readback.
        verdict = ae.get("control_vs_fps_verdict", "")
        ae["exposure_source_recommendation"] = (
            "use fps_implied_exposure as ground truth; do NOT trust "
            "exposure_time_absolute readback"
            if "DISAGREE" in verdict or not verdict
            else "exposure_time_absolute readback agrees with fps; either may be used")
        report["auto_dark"] = ae
        _write_report()
        return

    # --- manual mode ---
    print(f"\n=== Forcing manual / disabling adaptive processing ===")
    _manual_notes = prepare_manual(args.device,
                                   exposure=args.exposure,
                                   gain=args.gain,
                                   gamma=args.gamma,
                                   brightness=args.brightness,
                                   contrast=args.contrast,
                                   sharpness=args.sharpness,
                                   saturation=args.saturation)
    if _usb_warn:
        print(f"   {_usb_warn}")
        _manual_notes.insert(0, _usb_warn)
    if _nonnative_warn:
        _manual_notes.insert(0, _nonnative_warn)
    report["manual_notes"] = _manual_notes
    # Read the processing path BACK from the device rather than trusting what
    # was asked for: a control that silently refused the write would otherwise
    # be recorded as if it had taken. This is what --compare diffs.
    report["processing_path"] = read_processing_path(args.device)
    if report["processing_path"]:
        print("   processing path: " + ", ".join(
            f"{k}={v}" for k, v in sorted(report["processing_path"].items())))

    # --- PTC ---
    if args.ptc:
        cap = Capture(args.device, args.width, args.height, meas_fmt)
        cap.exposure = args.exposure   # applied in each v4l2-ctl capture run
        print(f"\n(capturing {cap.w}x{cap.h} {meas_fmt}, "
              f"bytesperline={cap.bytesperline}, sizeimage={cap.sizeimage})")
        try:
            points, fit = run_ptc(cap, args.device)
        finally:
            cap.close()
        report["ptc"] = {"points": points, "fit": fit}
        if fit.get("ok"):
            print("\n--- PTC RESULT ---")
            print(f"  system gain      : {fit['gain_e_per_adu']:.4f} e-/ADU")
            print(f"  read noise       : {fit['read_noise_adu']:.3f} ADU "
                  f"= {fit['read_noise_e']:.2f} e-")
            print(f"  full well (approx): {fit['full_well_e_approx']:.0f} e-")

    # --- framerate probe (standalone) ---
    if args.fps_probe:
        _rng = get_ctrl_range(args.device, "exposure_time_absolute")
        tl = build_timing_ladder(_rng.get("min", 1),
                                 _rng.get("max", args.exposure_max))
        recs, note = _timing_ladder(tl)
        probe = {"note": note, "exposure_range": _rng, "points": recs or []}
        if recs:
            kn = derive_knee(recs)
            probe["knee"] = kn
            probe["exposure_fidelity"] = exposure_fidelity_verdict(
                recs, kn["knee_units"] if kn else 0)
            print("\n--- FRAMERATE PROBE RESULT ---")
            if kn:
                print(f"  integration knee : {kn['knee_units']} units "
                      f"({kn['knee_units'] * 0.1:.1f}ms)")
                print(f"    {kn['detail']}")
            else:
                print("  integration knee : not detectable -- fps never "
                      "departed its pinned maximum even in a compressed "
                      "format. Either exposure is synthetic, or the range "
                      "tested never exceeds the frame period.")
            print(f"  -> {probe['exposure_fidelity']['verdict']}")
        report["fps_probe"] = probe
        _write_report()
        return

    # --- lit transfer curve ---
    if args.transfer:
        if args.transfer_levels:
            levels = [int(x) for x in
                      args.transfer_levels.replace(" ", "").split(",") if x]
        else:
            # denser at the bottom: a tone-curve LUT bends hardest in the
            # first quarter, and a uniform sweep walks straight past it
            levels = [0, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128,
                      160, 192, 224, 255]
        cap = Capture(args.device, args.width, args.height, meas_fmt)
        cap.exposure = args.exposure
        try:
            tr = run_transfer(cap, args.device, levels,
                              frames=args.transfer_frames,
                              settle_s=args.transfer_settle,
                              port=args.panel_port,
                              discard=args.discard)
        except RuntimeError as e:
            print(f"\n!! transfer sweep aborted: {e}")
            _write_report()
            return
        finally:
            cap.close()
        tr["exposure_units"] = args.exposure
        tr["processing_path"] = report.get("processing_path")
        report["transfer"] = tr
        fit = tr["fit"]
        print("\n--- TRANSFER CURVE RESULT ---")
        if fit.get("ok"):
            lf = fit["linear_fit"]
            print(f"  pedestal (panel black)     : {fit['pedestal_adu']:.3f} ADU")
            print(f"  straight line              : "
                  f"{lf['slope_adu_per_level']:.4f} ADU/level, "
                  f"intercept {lf['intercept_adu']:.3f}, r2 {lf['r2']:.4f}")
            pf = fit["power_fit"]
            if pf.get("ok"):
                print(f"  power law                  : exponent "
                      f"{pf['exponent']:.3f}, log-log r2 {pf['r2_loglog']:.4f}")
            print(f"  -> {fit['verdict']}")
            print(f"\n  NOTE: {fit['display_gamma_caveat']}")
        else:
            print(f"  no fit: {fit.get('reason')}")
        _write_report()
        return

    # --- dark ---
    if args.dark:
        ladder = None
        if args.ladder:
            ladder = [int(x) for x in args.ladder.replace(" ", "").split(",") if x]

        # Framerate probe FIRST, in a low-bandwidth format. It is the only way
        # to see the integration knee at a useful resolution, and it must run
        # before the uncompressed capture so the knee is available to filter
        # the dark-current fit. Timing only -- nothing it captures is read.
        timing = None
        if not args.no_timing_probe:
            _rng = get_ctrl_range(args.device, "exposure_time_absolute")
            tl = build_timing_ladder(_rng.get("min", 1),
                                     _rng.get("max", args.exposure_max))
            timing, _tnote = _timing_ladder(tl)
            report["timing_probe_note"] = _tnote
            # the probe leaves the device in a compressed format and at the
            # last swept exposure; put the processing path back before the
            # real capture starts
            prepare_manual(args.device, exposure=args.exposure,
                           gain=args.gain, gamma=args.gamma,
                           brightness=args.brightness, contrast=args.contrast,
                           sharpness=args.sharpness,
                           saturation=args.saturation, verbose=False)

        cap = Capture(args.device, args.width, args.height, meas_fmt)
        try:
            dark, master = run_dark(cap, args.device,
                                    dark_frames=args.dark_frames,
                                    exposure_max=args.exposure_max,
                                    slope_points=args.slope_points,
                                    ladder=ladder,
                                    discard=args.discard,
                                    knee=args.knee,
                                    ladder_frames=args.ladder_frames,
                                    ladder_points=args.ladder_points,
                                    timing_ladder=timing)
        finally:
            cap.close()
        report["dark"] = dark
        # A clipped deep stack is not a measurement: the master, hot-pixel list
        # and dark-current fit are all meaningless, and writing them anyway
        # means PHD2 silently auto-loads garbage at every camera connect.
        deep_clipped = bool(dark.get("deep_stack_clipped"))
        deep_quantised = bool(dark.get("deep_stack_quantiser_limited"))
        # Either failure mode invalidates the same three exports. Writing them
        # anyway means PHD2 silently auto-loads garbage at every camera
        # connect -- and a 12-pixel defect map where the truth is 1057 is worse
        # than no defect map at all, because it looks like it worked.
        refuse_export = deep_clipped or deep_quantised
        _want_export = (args.save_master or args.save_defects
                        or args.save_dark_model)
        if deep_clipped and _want_export:
            print("   !! deep stack CLIPPED -- refusing to write master/defects/"
                  "dark-model (fix the processing path: reduce brightness/"
                  "contrast/gamma, then rerun)")
        if deep_quantised and _want_export:
            qc = dark.get("quantiser_check") or {}
            print("   !! deep stack QUANTISER-LIMITED -- refusing to write "
                  "master/defects/dark-model:")
            for r in qc.get("reasons", []):
                print(f"        - {r}")
            print("      The output is collapsed onto too few codes for the "
                  "sensor's noise to reach the LSB, so the defect map and FPN "
                  "describe the processing path, not the sensor. Reduce "
                  "gamma/contrast (or raise gain) until the deep stack shows "
                  "temporal dither, then rerun.")
        # Non-native resolution: still write if asked (the user may want it for a
        # matching guide resolution), but stamp the artefact warning so the files
        # are never mistaken for a true sensor defect map.
        if _nonnative_warn and (args.save_master or args.save_defects
                                or args.save_dark_model):
            print(f"   !! writing calibration from a NON-NATIVE mode -- "
                  f"{args.width}x{args.height} is bridge-scaled; these are "
                  "rescaling artefacts, not the sensor's true defects/FPN")
        if args.save_master and master is not None and not refuse_export:
            # Scale float64 luma (0-255) to uint16 (0-65535) for direct PHD2 import.
            # 257 * 255 = 65535 exactly; np.round preserves sub-ADU precision.
            master_u16 = np.clip(np.round(master * 257.0), 0, 65535).astype(np.uint16)
            _ensure_dir(args.save_master)
            np.save(args.save_master, master_u16)
            print(f"   master dark (uint16) written to {args.save_master}")
        if args.save_defects and master is not None and not refuse_export:
            dcls = dark.get("defect_classes") or {}
            want = [c.strip() for c in args.defect_classes.split(",") if c.strip()]
            coords = []
            per_class = {}
            if dcls.get("ok"):
                for c in want:
                    got = dcls["coords"].get(c)
                    if got is None:
                        print(f"   !! unknown defect class '{c}' -- known: "
                              + ", ".join(sorted(dcls["coords"])))
                        continue
                    per_class[c] = got
                    coords.extend(got)
                coords = sorted({tuple(p) for p in coords})
                coords = [list(p) for p in coords]
            else:
                # classification unavailable (single-frame stack, or refused):
                # fall back to the flat 6-sigma list rather than writing nothing
                coords = dark.get("hot_pixel_coords") or []
            _ensure_dir(args.save_defects)
            with open(args.save_defects, "w") as fh:
                fh.write("# PHD2 Defect Map v1\n")
                fh.write(f"# cam_characterise.py {args.device} "
                         f"{args.width}x{args.height}\n")
                fh.write(f"# Exposure: {args.exposure_max} units "
                         f"({dark['exposure_max_ms']}ms)\n")
                fh.write(f"# Native advertised mode: "
                         f"{report['is_native_advertised_mode']}\n")
                if _nonnative_warn:
                    fh.write("# WARNING: non-native (bridge-scaled) mode -- "
                             "these defects are rescaling artefacts\n")
                if dcls.get("ok"):
                    fh.write(f"# Classes included: {','.join(want)}\n")
                    for c in sorted(dcls["counts"]):
                        if c == "total":
                            continue
                        mark = "*" if c in per_class else " "
                        fh.write(f"#  {mark} {c}: {dcls['counts'][c]}\n")
                    fh.write("#  (* = written to this map; unmarked classes "
                             "are handled by dark subtraction)\n")
                else:
                    fh.write("# Classes: unavailable -- flat 6-sigma list\n")
                fh.write(f"# Defect count: {len(coords)}\n")
                for y, x in coords:  # PHD2 format is x y (screen coords)
                    fh.write(f"{x} {y}\n")
            print(f"   defect map written to {args.save_defects} "
                  f"({len(coords)} pixels"
                  + (f", classes: {','.join(want)}" if dcls.get("ok") else "")
                  + ")")
            # full per-class detail alongside, for anything that wants to treat
            # the classes differently (the map itself must stay x-y only, since
            # PHD2 parses it)
            if dcls.get("ok"):
                side = args.save_defects + ".classes.json"
                with open(side, "w") as fh:
                    json.dump({"counts": dcls["counts"],
                               "thresholds": dcls["thresholds"],
                               "remedy": dcls["remedy"],
                               "dark_current_stats": dcls["dark_current_stats"],
                               "written_to_map": want,
                               "coords": dcls["coords"]}, fh, indent=2)
                print(f"   defect classes written to {side}")
        if args.save_dark_model and master is not None and not refuse_export:
            dcf = dark.get("dark_current_fit") or {}
            model = {
                "exposure_max_units": args.exposure_max,
                "exposure_max_ms": dark["exposure_max_ms"],
                "pixel_depth": 8,  # value domain (see run_dark result note)
                "capture_fourcc": dark["capture_fourcc"],
                "native_bit_depth": dark["native_bit_depth"],
                "dark_current_fit": dcf,
                "exposure_fidelity_verdict": (
                    dark.get("exposure_fidelity", {}).get("verdict", "")),
            }
            _ensure_dir(args.save_dark_model)
            with open(args.save_dark_model, "w") as fh:
                json.dump(model, fh, indent=2)
            print(f"   dark current model written to {args.save_dark_model}")
        if args.save_calib:
            calib = build_fps_calibration(dark.get("ladder", []), meas_fmt,
                                          args.width, args.height)
            if calib:
                _ensure_dir(args.save_calib)
                with open(args.save_calib, "w") as fh:
                    json.dump(calib, fh, indent=2)
                print(f"   fps<->exposure calibration written to {args.save_calib} "
                      f"(valid fps {calib['valid_fps_range']}, "
                      f"{len(calib['anchors_fps'])} anchors)")
            else:
                print("   could not build fps calibration (need >=3 points in the "
                      "integration-limited region; use a ladder that spans the "
                      "fps-falling range, e.g. up to the switch at ~4096)")

        print("\n--- DARK RESULT (at max exposure) ---")
        if refuse_export:
            why = "CLIPPED" if deep_clipped else "QUANTISER-LIMITED"
            print(f"  *** deep stack {why}: every number below describes the "
                  "processing path, not the sensor ***")
        qc = dark.get("quantiser_check") or {}
        print(f"  distinct output codes (stack/master): "
              f"{qc.get('distinct_codes_stack')}/"
              f"{qc.get('distinct_codes_master')}")
        print(f"  fixed-pattern (subtractable) FPN : "
              f"{dark['master_dark_fpn_adu']:.3f} ADU")
        print(f"  hot pixels (>6sigma)             : {dark['hot_pixels_6sigma']}"
              + ("  <-- collapsed output: NOT the sensor's defect count"
                 if deep_quantised else ""))
        _dc = dark.get("defect_classes") or {}
        if _dc.get("ok"):
            _c = _dc["counts"]
            print(f"  defects by failure mode          : {_c['total']} total")
            print(f"    stuck (no response)            : {_c['stuck']:6d}   "
                  "-> interpolate")
            print(f"    intermittent hot               : "
                  f"{_c['intermittent_hot']:6d}   -> interpolate "
                  "(subtraction cannot fix it)")
            print(f"    hot (stable, elevated)         : {_c['hot']:6d}   "
                  "-> dark subtraction handles it")
            print(f"    dark current (exposure-scaled) : "
                  f"{_c['dark_current']:6d}   -> scaled dark subtraction")
            _ds = _dc.get("dark_current_stats") or {}
            if "median_rate_adu_per_unit" in _ds:
                print(f"    per-pixel dark current: median "
                      f"{_ds['median_rate_adu_per_unit']:.3g}, p99.9 "
                      f"{_ds['p99_9_rate_adu_per_unit']:.3g}, max "
                      f"{_ds['max_rate_adu_per_unit']:.3g} ADU/unit "
                      f"(vs bias at exp "
                      f"{dark.get('bias_reference_exposure_units')})")
        elif _dc:
            print(f"  defects by failure mode          : not classified -- "
                  f"{_dc.get('reason')}")
        # 0.000 ADU is not a quiet sensor, it is an unmeasurable one. Say so.
        nb = dark.get("irreducible_temporal_dark_noise_bound")
        if nb:
            print(f"  IRREDUCIBLE temporal dark noise  : {nb['statement']}"
                  f"  (measured {dark['irreducible_temporal_dark_noise_adu']:.3f}, "
                  "which is the quantiser's floor, not the sensor's)")
        else:
            print(f"  IRREDUCIBLE temporal dark noise  : "
                  f"{dark['irreducible_temporal_dark_noise_adu']:.3f} ADU")
        kn = dark.get("knee") or {}
        print(f"  integration knee                 : {kn.get('knee_units')} units "
              f"({kn.get('source')})")
        dcf = dark["dark_current_fit"]
        if dcf:
            if dcf.get("flat_within_detection_limit"):
                print(f"  dark current                     : NOT DETECTED -- "
                      f"{dcf['detection_limit']['statement']}")
                print(f"                                     (r2={dcf['r2']:.3f} "
                      "is meaningless on a flat ladder; use the bound above for "
                      "PHD2 dark scaling)")
            else:
                print(f"  dark current                     : "
                      f"{dcf['dark_current_adu_per_unit']:.5f} ADU/unit "
                      f"(linear: {dcf['linear']}, r2={dcf['r2']:.3f})")
                if dcf.get("detection_limit"):
                    print(f"    detection limit                : "
                          f"{dcf['detection_limit']['min_detectable_slope_adu_per_unit']:.3g}"
                          " ADU/unit")

        # combined floor at the operating point, if we also have gain
        if report.get("ptc", {}).get("fit", {}).get("ok"):
            g = report["ptc"]["fit"]["gain_e_per_adu"]
            rn_e = report["ptc"]["fit"]["read_noise_e"]
            # a quantiser-limited dark term is an upper bound, so the combined
            # floor built from it is an upper bound too -- never a measurement
            dn_adu = (nb["upper_bound_adu"] if nb
                      else dark["irreducible_temporal_dark_noise_adu"])
            dn_e = dn_adu * g
            floor = float(np.sqrt(rn_e**2 + dn_e**2))
            report["operating_point_floor_e"] = floor
            report["operating_point_floor_is_upper_bound"] = bool(nb)
            print(f"\n  COMBINED single-frame floor @ max exposure : "
                  f"{'< ' if nb else ''}{floor:.2f} e-  (read {rn_e:.2f} + "
                  f"dark-shot {'<' if nb else ''}{dn_e:.2f} in quadrature)")

    _write_report()


if __name__ == "__main__":
    main()
