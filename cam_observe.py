#!/usr/bin/env python3
"""
cam_observe.py -- standalone GUI observing tool for UVC cameras, built on the
cam_characterise.py measurement stack.

Two operating modes, switched explicitly by the user (the tool cannot know if
the lens is capped, so the human says so):

  DARK  -- the lens is capped. The raw stream is shown beside the
           dark-corrected residual (frame minus master dark at a FIXED
           stretch -- ideally complete blackness, and the residual stats say
           how close to ideal it is). Acquire a deep master dark (streaming
           per-pixel mean) and derive a hot-pixel defect map, in exactly the
           per-device format cam_manager.py plans and PHD2 auto-loads:
           {vid+pid[_serial]}/master_WxH.npy (uint16 x257) and defects_WxH.txt.
           Existing files for the device+resolution are loaded automatically.

  LIGHT -- the camera can see photons. The raw stream is shown beside the
           integrated stack: frames are dark-subtracted, flat-corrected,
           defect-repaired, optionally aligned by phase correlation, and
           integrated over a
           ROLLING window of N frames (N visible and changeable). Stars are
           extracted from the stack (robust background + connected
           components -> centroid/flux/FWHM) at a threshold that adapts
           toward the user's declared "visible stars" count, and every
           candidate must be CONFIRMED by a temporal tracker before it is
           reported: on a static camera real stars persist burst to burst
           and drift slowly and predictably (alpha-beta filtered tracks
           with velocity prediction), while noise doesn't survive
           consecutive bursts -- so the detector can dig fainter without
           false stars reaching the display, the solver gate, or the
           pause reference. The stack can be written to FITS / npy and
           plate-solved with a local astrometry.net installation
           (solve-field), manually or on an auto-solve timer. The raw and
           stack views each pop out into a dedicated resizable window
           (click / double-click), whose size feeds the capture
           downsampling so a bigger window shows real resolution.

           A master FLAT (camera aimed at an evenly illuminated field:
           twilight sky, light panel, defocused white surface) can be
           acquired here too: streamed per-pixel mean, dark-subtracted,
           normalised to a median-1 gain map and saved per device as
           flat_WxH.npy (+ metadata sidecar), auto-loaded per resolution
           like the dark. Correction is applied per frame either by
           division (true gain correction -- right on a linear path) or by
           subtracting the flat's structure scaled to the frame's own
           background (robust on gamma'd / black-clamped paths, at the
           cost of not being photometric). "View flat" analyses the map:
           radial gain profile, per-pixel noise the divide injects, and a
           corner-vs-centre verdict separating optical falloff from
           upstream LSC over-correction; during integration the stack's
           corner/centre background ratio is reported live, so toggling
           the correction on an evenly lit field shows directly whether
           the frame flattens. Acquisition pre-flights the signal level
           and holds -- telling the user which way to adjust the light --
           until the median is in the usable band; an opt-in auto-exposure
           search instead bisects the exposure to mid-scale (flagged,
           since a flat not shot at the observing exposure may not fully
           cancel a nonlinear ISP's shading). A built-in software flat
           panel (a uniform grey window on any attached screen) makes the
           illumination itself a knob the tool can turn: with auto level,
           acquisition servos the panel brightness into the band -- fully
           automatic flats with no camera setting touched.

           "Pause integration" freezes stacking (capture and display keep
           running) for when the sensor is about to move -- e.g. adjusting an
           equatorial mount's altitude/azimuth. While paused, stars detected
           in the live frames are matched against their pre-pause positions
           and drawn as motion arrows with a median displacement readout:
           live feedback on where the field is going during polar alignment.

The V4L2/UVC processing controls the device actually exposes (gain, gamma,
brightness, contrast, sharpness, saturation) are presented with their real
ranges and applied live.

A "Deep-dive dump" button answers "that looks WRONG -- why?": one press
writes the next capture burst in full (every raw frame, every processed
frame, the calibration data in use, all parameters, per-frame statistics)
to a self-describing directory whose manifest.json documents each file --
press the button, hand over the directory. Idle, it dumps the session's
in-memory state instead.

Requires: Linux, v4l2-ctl, numpy, tkinter. Plate solving needs solve-field
(astrometry.net) plus index files on the local machine.

Usage:
  python3 cam_observe.py [--device /dev/video0] [--data-dir .]
"""

import argparse
import glob
import json
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone

import numpy as np
import tkinter as tk
from tkinter import ttk

import cam_characterise as cc
from cam_manager import usb_device_tag, query_formats, query_controls, identify

TOOL_VERSION = "0.14"


# ----------------------------------------------------------------------------
# Image processing
# ----------------------------------------------------------------------------

def stretch_to_u8(img, lo_pct=0.5, hi_pct=99.7, return_bounds=False):
    """Percentile auto-stretch to displayable uint8. With return_bounds, also
    returns the (lo, hi) ADU values mapped to black/white -- a small span
    means the display is heavily amplifying a near-flat frame."""
    lo, hi = np.percentile(img, (lo_pct, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    out = (img - lo) * (255.0 / (hi - lo))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return (out, float(lo), float(hi)) if return_bounds else out


def residual_to_u8(resid, gain=8.0):
    """FIXED-stretch display for the dark-corrected residual: 0 ADU is black,
    255/gain ADU is white. Deliberately NOT auto-stretched -- a perfect
    correction should LOOK black, not have its nothingness amplified."""
    return np.clip(resid * gain, 0, 255).astype(np.uint8)


def pgm_bytes(img8):
    h, w = img8.shape
    return b"P5\n%d %d\n255\n" % (w, h) + img8.tobytes()


def downsample_to_fit(img, max_w, max_h):
    """Integer-stride downsample so the frame fits the canvas. Returns
    (small_image, stride)."""
    h, w = img.shape
    f = max(1, (w + max_w - 1) // max_w, (h + max_h - 1) // max_h)
    return img[::f, ::f], f


def aspect_str(w, h):
    """Human aspect ratio: exact when the reduced form is sane (4:3, 16:9),
    nearest common ratio for marketing pixel counts (1366x768 -> ~16:9),
    decimal otherwise."""
    g = math.gcd(w, h)
    a, b = w // g, h // g
    if (a, b) == (8, 5):
        a, b = 16, 10
    if max(a, b) <= 21:
        return f"{a}:{b}"
    r = w / h
    common = [(4, 3), (16, 9), (3, 2), (16, 10), (5, 4), (1, 1), (21, 9)]
    best = min(common, key=lambda ab: abs(ab[0] / ab[1] - r))
    if abs(best[0] / best[1] - r) / r < 0.02:
        return f"~{best[0]}:{best[1]}"
    return f"{r:.2f}:1"


def frame_stats(img):
    """Compact, self-describing statistics of a frame for the debug log.
    Values are in the 8-bit luma float domain (0-255 ADU)."""
    med = float(np.median(img))
    return {"min": round(float(img.min()), 2),
            "max": round(float(img.max()), 2),
            "mean": round(float(img.mean()), 3),
            "median": round(med, 3),
            "mad_sigma": round(1.4826 * float(np.median(np.abs(img - med))), 4)}


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


# Phase-correlation peak prominence (peak height over the surface's robust
# sigma) below which the estimated shift is treated as untrustworthy: a
# featureless field (vignette + noise, no stars) gives a flat surface whose
# argmax is random, and aligning on that smears the stack. Tunable; logged
# per burst as align_conf so it can be calibrated against real fields.
ALIGN_MIN_CONF = 6.0

# Thumbnail box the phase correlation runs on. Bigger thumbnail -> smaller
# downsample stride -> finer shift. At 1920x1080 a 256 box gave stride 8, so
# any drift under 8 full-res px rounded to a zero shift and the stack smeared
# on exactly the sub-pixel drift it was meant to correct. 512 gives stride ~4,
# and phase_shift's parabolic refinement adds sub-pixel within that.
ALIGN_THUMB = 512


def phase_shift(ref_small, img_small):
    """Sub-pixel (dy, dx, conf) translation of img relative to ref via phase
    correlation on downsampled frames. The integer peak is refined by
    parabolic interpolation on its neighbours, so the estimate is NOT
    quantized to the downsample stride. conf is the correlation peak's
    prominence over the surface (peak-minus-median / robust sigma); a low
    conf means there is no trustworthy translation signal and the caller
    should not align. Cheap and rotation-free -- right for short-session
    drift, not for field rotation."""
    a = ref_small.astype(np.float64)
    b = img_small.astype(np.float64)
    # mean-subtract + separable Hann window: removes the DC term and the
    # FFT edge-wrap cross that otherwise plant a spurious peak when the frame
    # carries no real translation signal
    a -= a.mean()
    b -= b.mean()
    win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
    a *= win
    b *= win
    f1 = np.fft.rfft2(a)
    f2 = np.fft.rfft2(b)
    cps = f1 * np.conj(f2)
    mag = np.abs(cps)
    mag[mag < 1e-12] = 1e-12
    corr = np.fft.irfft2(cps / mag, s=ref_small.shape)
    h, w = corr.shape
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    med = float(np.median(corr))
    sigma = 1.4826 * float(np.median(np.abs(corr - med))) or 1e-9
    conf = (float(corr[py, px]) - med) / sigma

    # parabolic sub-pixel refinement along each axis from the peak's immediate
    # neighbours. The correlation surface is circular, so index mod size. The
    # offset of a true quadratic peak falls in [-0.5, 0.5]; clamp to [-1, 1]
    # against a near-flat denominator on a featureless field.
    def _sub(c_minus, c0, c_plus):
        denom = c_minus - 2.0 * c0 + c_plus
        if denom == 0.0:
            return 0.0
        return max(-1.0, min(1.0, 0.5 * (c_minus - c_plus) / denom))

    c0 = float(corr[py, px])
    fy = _sub(float(corr[(py - 1) % h, px]), c0, float(corr[(py + 1) % h, px]))
    fx = _sub(float(corr[py, (px - 1) % w]), c0, float(corr[py, (px + 1) % w]))
    dy = py + fy
    dx = px + fx
    if dy > h / 2.0:
        dy -= h
    if dx > w / 2.0:
        dx -= w
    return dy, dx, conf


def shift_fill(img, dy, dx):
    """Translate by (dy, dx) -- fractional allowed, bilinear -- WITHOUT
    wraparound. Exposed edges are filled with the frame's own background
    (median), so a drift correction never folds the bright opposite edge back
    into the stack (np.roll's wrap was the source of the mirrored 'butterfly'
    ghosts on featureless fields)."""
    if dy == 0 and dx == 0:
        return img
    h, w = img.shape
    fill = float(np.median(img))
    if abs(dy) >= h or abs(dx) >= w:        # shifted entirely out of frame
        return np.full_like(img, fill)

    def _int_shift(a, sy, sx):
        out = np.full_like(a, fill)
        ah, aw = a.shape
        dy0, dy1 = max(0, -sy), min(ah, ah - sy)
        dx0, dx1 = max(0, -sx), min(aw, aw - sx)
        if dy1 > dy0 and dx1 > dx0:
            out[max(0, sy):max(0, sy) + (dy1 - dy0),
                max(0, sx):max(0, sx) + (dx1 - dx0)] = a[dy0:dy1, dx0:dx1]
        return out

    iy, fy = int(np.floor(dy)), float(dy - np.floor(dy))
    ix, fx = int(np.floor(dx)), float(dx - np.floor(dx))
    if fy == 0.0 and fx == 0.0:
        return _int_shift(img, iy, ix)
    # bilinear blend of the four integer-shifted corners: out[y] samples img
    # between rows (y-iy-1) and (y-iy), weighted by the fractional part.
    c00 = _int_shift(img, iy, ix)
    c01 = _int_shift(img, iy, ix + 1)
    c10 = _int_shift(img, iy + 1, ix)
    c11 = _int_shift(img, iy + 1, ix + 1)
    return ((1.0 - fy) * ((1.0 - fx) * c00 + fx * c01)
            + fy * ((1.0 - fx) * c10 + fx * c11))


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


# Master-flat guards and clamps. A flat with a meaningful fraction of pixels
# at the ceiling is saturated (the gain information there is destroyed); a
# flat whose dark-subtracted median is tiny is mostly shot noise and would
# imprint that noise on every corrected frame. The gain map is clamped so a
# near-dead region (deep vignette corner, dust shadow at low signal) can
# never turn a divide into a x5+ noise amplifier.
FLAT_SAT_FRAC = 0.02        # reject: fraction of pixels >= 254.5 ADU
# Flat signal guidance bands (median ADU of the raw flat frame). OK_* is the
# gate the pre-flight loop waits for before accumulating -- outside it the
# tool tells the user which way to adjust the illumination rather than
# building a flat that would only be rejected afterwards. TARGET_* is the
# tighter mid-scale band the (opt-in) auto-exposure search aims for.
FLAT_OK_LO, FLAT_OK_HI = 40.0, 220.0
FLAT_TARGET_LO, FLAT_TARGET_HI = 100.0, 160.0
FLAT_MIN_MEDIAN = 5.0       # reject: dark-subtracted median below this (ADU)
FLAT_LOW_MEDIAN = 30.0      # warn: usable but noisy below this (ADU)
FLAT_GAIN_MIN, FLAT_GAIN_MAX = 0.2, 5.0


def build_flat(mean, dark=None):
    """Normalise an averaged flat exposure into a median-1 gain map:
    dark-subtract when a master dark is supplied, reject saturated or
    signal-starved flats, clamp the gain into [FLAT_GAIN_MIN, FLAT_GAIN_MAX].
    Returns (gain, reject_reason, info); gain is None when rejected, info
    always carries the statistics."""
    ceil_frac = float((mean >= 254.5).mean())
    sig = mean - dark if dark is not None \
        else np.asarray(mean, dtype=np.float64)
    med = float(np.median(sig))
    info = {"median_adu": round(med, 3),
            "ceil_frac": round(ceil_frac, 4),
            "dark_subtracted": dark is not None}
    if ceil_frac > FLAT_SAT_FRAC:
        return None, (f"{ceil_frac * 100:.1f}% of pixels saturated -- "
                      "reduce exposure or illumination"), info
    if med < FLAT_MIN_MEDIAN:
        return None, (f"median signal {med:.1f} ADU is too low to "
                      "normalise -- brighten the illumination or raise "
                      "the exposure"), info
    gain = sig / med
    lo_frac = float((gain < FLAT_GAIN_MIN).mean())
    gain = np.clip(gain, FLAT_GAIN_MIN, FLAT_GAIN_MAX)
    info.update(clamped_low_frac=round(lo_frac, 4),
                gain_p5=round(float(np.percentile(gain, 5)), 4),
                gain_p95=round(float(np.percentile(gain, 95)), 4))
    return gain, None, info


def apply_flat(img, gain, mode):
    """Remove the fixed illumination/gain pattern (vignetting, dust shadows)
    using a normalised master flat.

    divide   -- img / gain: true gain correction, photometrically right when
                the processing path is linear.
    subtract -- img - bg*(gain - 1): removes the flat's STRUCTURE scaled to
                this frame's own background level, leaving the background
                flat at bg. Star amplitudes are untouched, so it is not
                photometric -- but it cannot over/under-correct on the
                nonlinear (gamma'd, black-clamped) paths these cameras
                usually run, where division assumes a linearity the data
                does not have."""
    if mode == "subtract":
        bg = float(np.median(img))
        return np.clip(img - bg * (gain - 1.0), 0.0, None)
    return img / gain


def radial_profile(img, nbins=48):
    """Mean value in equal-width radial bins from the frame centre, radius
    normalised so 1.0 is the corner distance. Returns (r_centres, means)
    with empty bins dropped -- the vignette/LSC signature of a gain map."""
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx) / math.hypot(cy, cx)
    idx = np.minimum((r * nbins).astype(int), nbins - 1)
    sums = np.bincount(idx.ravel(), weights=img.ravel(), minlength=nbins)
    cnts = np.bincount(idx.ravel(), minlength=nbins)
    keep = cnts > 0
    centres = (np.arange(nbins) + 0.5) / nbins
    return centres[keep], sums[keep] / cnts[keep]


def flat_noise_sigma(gain):
    """Per-pixel noise of the gain map, from horizontal first differences:
    smooth structure (vignette) cancels between neighbours while noise adds
    in quadrature, hence the /sqrt(2). This is the fractional noise the
    divide injects into every corrected frame."""
    d = np.diff(gain, axis=1)
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) \
        / math.sqrt(2.0)


def field_flatness(img):
    """Corner-vs-centre background of a frame: robust medians of the four
    corner boxes and the central box (each frame/6 on a side). Returns
    (centre, corner_mean, ratio); ratio is None when the centre background
    is too near zero for the quotient to mean anything (a well-subtracted
    night frame). Pointed at an evenly lit field, this ratio moving to ~1.0
    when 'flat correct' is enabled IS the closed-loop flat verification."""
    h, w = img.shape
    bh, bw = max(2, h // 6), max(2, w // 6)
    centre = float(np.median(img[(h - bh) // 2:(h + bh) // 2,
                                 (w - bw) // 2:(w + bw) // 2]))
    corner = float(np.mean([np.median(img[:bh, :bw]),
                            np.median(img[:bh, -bw:]),
                            np.median(img[-bh:, :bw]),
                            np.median(img[-bh:, -bw:])]))
    ratio = corner / centre if centre > 0.5 else None
    return centre, corner, ratio


def extract_stars(img, nsigma=5.0, min_pix=2, max_pix=500, max_stars=100,
                  min_fwhm=1.3):
    """Detect stars on a (dark-subtracted) image: robust background via
    median/MAD, threshold at nsigma, 8-connected components by flood fill on
    the sparse mask, flux-weighted centroids, flux, and a moment-based FWHM.
    Returns a list of dicts sorted by flux descending.

    min_fwhm rejects components tighter than a real (even undersampled) star:
    single hot pixels and 2-3px hot-pixel clusters have FWHM well under ~1.3,
    and on bridges with an unstable dark floor they are the dominant source of
    phantom 'stars'. A genuine point source on these optics spreads wider."""
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
        size = 0
        # walk the WHOLE component even past max_pix (only storing up to the
        # cap): stopping the fill early leaves the rest of an oversized blob
        # unvisited, and its remainder chunks re-seed as small components
        # that pass the size filter -- phantom stars on the rim of any
        # saturated region. The n_above guard bounds total work.
        while stack_px:
            y, x = stack_px.pop()
            size += 1
            if size <= max_pix:
                comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                            and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack_px.append((ny, nx))
        if not (min_pix <= size <= max_pix):
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
        if fwhm < min_fwhm:
            continue            # hot pixel / tight cluster, not a star
        stars.append({"x": cxf, "y": cyf, "flux": ftot,
                      "npix": len(comp), "fwhm": round(fwhm, 2)})
    stars.sort(key=lambda s: -s["flux"])
    return stars[:max_stars], bg, sigma


class StarTracker:
    """Temporal confirmation of star detections for a static (or slowly
    drifting) camera. On a fixed mount stars persist burst after burst and
    move slowly and predictably (sidereal drift, or ~zero when alignment
    cancels it); noise spikes and residual-pattern flickers do not. Each
    detection therefore feeds an alpha-beta-filtered track, and only tracks
    seen in `confirm`+ bursts are reported -- so the detector can run at a
    LOWER threshold (finding faint real stars) without single-burst noise
    ever reaching the display, the solver gate, or the pause reference.
    The per-track velocity estimate makes next-burst matching predictive,
    and `miss_max` coasting tolerates clouds and scintillation dropouts."""

    ALPHA = 0.5     # position correction gain
    BETA = 0.15     # velocity correction gain (px/s per px of residual)

    def __init__(self, confirm=3, gate=8.0, miss_max=4, max_tracks=400):
        self.confirm = confirm
        self.gate = gate
        self.miss_max = miss_max
        self.max_tracks = max_tracks
        self.tracks = []
        self.t_last = None

    def update(self, dets, t):
        """Feed one burst's raw detections; returns the confirmed star
        list (dicts with x/y/flux/fwhm/npix plus hits/speed), brightest
        first. Tracks not matched this burst coast on their prediction."""
        dt = 0.0 if self.t_last is None else max(t - self.t_last, 1e-3)
        self.t_last = t
        for tr in self.tracks:
            tr["px"] = tr["x"] + tr["vx"] * dt
            tr["py"] = tr["y"] + tr["vy"] * dt
        used = set()
        # brightest tracks claim their detections first: a faint new
        # neighbour can't steal a bright star's measurement
        for tr in sorted(self.tracks, key=lambda d: -d["flux"]):
            best_j = None
            best_d = self.gate
            for j, s in enumerate(dets):
                if j in used:
                    continue
                d = math.hypot(s["x"] - tr["px"], s["y"] - tr["py"])
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is None:
                tr["misses"] += 1
                tr["x"], tr["y"] = tr["px"], tr["py"]   # coast
                continue
            used.add(best_j)
            s = dets[best_j]
            rx = s["x"] - tr["px"]
            ry = s["y"] - tr["py"]
            tr["x"] = tr["px"] + self.ALPHA * rx
            tr["y"] = tr["py"] + self.ALPHA * ry
            if dt > 0:
                tr["vx"] += self.BETA * rx / dt
                tr["vy"] += self.BETA * ry / dt
            tr["flux"] = 0.7 * tr["flux"] + 0.3 * s["flux"]
            tr["fwhm"] = s["fwhm"]
            tr["npix"] = s.get("npix", tr.get("npix", 0))
            tr["hits"] += 1
            tr["misses"] = 0
        self.tracks = [tr for tr in self.tracks
                       if tr["misses"] <= self.miss_max]
        for j, s in enumerate(dets):
            if j in used:
                continue
            if len(self.tracks) >= self.max_tracks:
                break
            self.tracks.append(dict(x=s["x"], y=s["y"], vx=0.0, vy=0.0,
                                    flux=s["flux"], fwhm=s["fwhm"],
                                    npix=s.get("npix", 0),
                                    hits=1, misses=0))
        conf = [dict(x=tr["x"], y=tr["y"], flux=tr["flux"],
                     fwhm=round(tr["fwhm"], 2), npix=tr["npix"],
                     hits=tr["hits"],
                     speed=round(math.hypot(tr["vx"], tr["vy"]), 3))
                for tr in self.tracks
                if tr["hits"] >= self.confirm and tr["misses"] == 0]
        conf.sort(key=lambda s: -s["flux"])
        return conf


def match_stars(ref_stars, cur_stars, max_dist=80.0):
    """Greedy nearest-neighbour matching of current star detections to a
    reference list. Returns [(rx, ry, cx, cy), ...] for matched pairs --
    the motion vectors drawn while integration is paused."""
    pairs = []
    used = set()
    for r in ref_stars:
        best_j = None
        best_d = max_dist
        for j, c in enumerate(cur_stars):
            if j in used:
                continue
            d = math.hypot(c["x"] - r["x"], c["y"] - r["y"])
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            used.add(best_j)
            c = cur_stars[best_j]
            pairs.append((r["x"], r["y"], c["x"], c["y"]))
    return pairs


def motion_summary(pairs):
    """Median displacement of matched star pairs: (dx, dy, magnitude,
    direction in degrees with 0=right(+x), 90=up)."""
    if not pairs:
        return None
    dxs = np.array([p[2] - p[0] for p in pairs])
    dys = np.array([p[3] - p[1] for p in pairs])
    mdx = float(np.median(dxs))
    mdy = float(np.median(dys))
    mag = math.hypot(mdx, mdy)
    ang = math.degrees(math.atan2(-mdy, mdx)) % 360.0  # screen y is down
    return mdx, mdy, mag, ang


def hist_counts(img, lo, hi, bins=128):
    """Histogram counts for the display panel."""
    return np.histogram(img, bins=bins, range=(lo, hi))[0]


# ----------------------------------------------------------------------------
# Plate solving (local astrometry.net)
# ----------------------------------------------------------------------------

def _parse_wcs_cards(raw):
    """Parse the numeric cards of a FITS header blob (solve-field's .wcs)."""
    cards = {}
    text = raw.decode("ascii", "replace")
    for i in range(0, len(text) - 79, 80):
        card = text[i:i + 80]
        if card.startswith("END"):
            break
        if card[8:10] != "= ":
            continue
        key = card[:8].strip()
        val = card[10:].split("/")[0].strip()
        try:
            cards[key] = float(val)
        except ValueError:
            pass
    return cards


def pole_pixel_from_wcs(raw):
    """Pixel position (0-based x, y) of the celestial pole of the field's
    hemisphere, from a TAN WCS. The great circle toward the pole is 'north'
    by definition, and the CD matrix encodes rotation AND parity, so this is
    exact -- no mirror-flip ambiguity. Returns None if not computable.

    Gnomonic: for dec=+/-90 the tangent-plane coords relative to a tangent
    point at dec0 are xi=0, eta=cot(dec0) (radians; same expression for the
    same-hemisphere pole in either hemisphere)."""
    c = _parse_wcs_cards(raw)
    try:
        dec0 = math.radians(c["CRVAL2"])
        cd = ((c["CD1_1"], c["CD1_2"]), (c["CD2_1"], c["CD2_2"]))
        crpix = (c["CRPIX1"], c["CRPIX2"])
    except KeyError:
        return None
    if abs(math.sin(dec0)) < 1e-6:
        return None  # pointing at the equator: pole at infinity in TAN
    xi = 0.0
    eta = math.degrees(1.0 / math.tan(dec0))
    det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0]
    if abs(det) < 1e-20:
        return None
    dx = (cd[1][1] * xi - cd[0][1] * eta) / det
    dy = (-cd[1][0] * xi + cd[0][0] * eta) / det
    return (crpix[0] - 1.0 + dx, crpix[1] - 1.0 + dy)

def solve_field(stack, solve_path, scale_low, scale_high, timeout=180,
                dbg=None):
    """Write the stack to FITS in a temp dir and run solve-field on it.
    Returns (ok, message, info) where info carries parsed ra/dec/rotation
    in degrees when available. Blocking -- call from a worker thread."""
    if dbg is None:
        dbg = NullDebug()
    exe = shutil.which(solve_path)
    if not exe:
        dbg.log("solve_end", ok=False, reason="solve-field not on PATH",
                solver=solve_path)
        return False, f"solve-field not found at '{solve_path}' -- install " \
                      "astrometry.net (and index files) or fix the path", {}
    tmp = tempfile.mkdtemp(prefix="camobs_solve_")
    fits = os.path.join(tmp, "stack.fits")
    t0 = time.monotonic()
    try:
        write_fits_u16(fits, stack)
        cmd = [exe, fits, "--overwrite", "--no-plots",
               "--cpulimit", str(timeout - 20), "--downsample", "2",
               "--dir", tmp]
        if scale_low > 0 and scale_high > scale_low:
            cmd += ["--scale-units", "arcsecperpix",
                    "--scale-low", str(scale_low),
                    "--scale-high", str(scale_high)]
        dbg.log("solve_start", cmd=cmd, stack=frame_stats(stack),
                shape=list(stack.shape))
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
        m = re.search(r"pixel scale ([0-9.]+) arcsec/pix", out)
        if m:
            info["arcsec_per_px"] = float(m.group(1))
        wcs_path = os.path.join(tmp, "stack.wcs")
        if solved and os.path.exists(wcs_path):
            try:
                with open(wcs_path, "rb") as fh:
                    pp = pole_pixel_from_wcs(fh.read())
                if pp:
                    info["pole_xy"] = [round(pp[0], 1), round(pp[1], 1)]
            except Exception:
                pass
        lines = []
        for pat in (r"Field center: \(RA,Dec\) = \([^)]*\) deg",
                    r"Field center: \(RA H:M:S, Dec D:M:S\) = \([^)]*\)",
                    r"Field size: [^\n]*",
                    r"Field rotation angle: [^\n]*"):
            m = re.search(pat, out)
            if m:
                lines.append(m.group(0))
        dur = round(time.monotonic() - t0, 2)
        if solved and lines:
            dbg.log("solve_end", ok=True, duration_s=dur, parsed=info,
                    rc=r.returncode)
            return True, "\n".join(lines), info
        if solved:
            dbg.log("solve_end", ok=True, duration_s=dur, parsed=info,
                    rc=r.returncode, note="no centre line parsed")
            return True, "solved (no centre line found in output?)", info
        dbg.log("solve_end", ok=False, duration_s=dur, rc=r.returncode,
                output_tail=out[-1500:])
        return False, "did not solve -- need more stars, better scale " \
                      "hints, or matching index files", {}
    except subprocess.TimeoutExpired:
        dbg.log("solve_end", ok=False, duration_s=round(
            time.monotonic() - t0, 2), reason=f"timeout after {timeout}s")
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
# Debug / diagnostics (--debug): JSONL aimed at LLM consumption
# ----------------------------------------------------------------------------

DEBUG_SCHEMA = {
    "session_start": "environment, versions, argv, and this schema",
    "probe": "device identity, ranked formats, chosen fourcc, resolutions, "
             "exposure range, USB device tag",
    "controls": "full V4L2 control dump (name -> min/max/step/default/value) "
                "at probe time",
    "calibration_load": "master dark / master flat / defect map / metadata "
                        "found (or not) for the selected resolution",
    "ctrl_set": "a V4L2 control write and whether the device accepted it",
    "worker_start": "capture worker launched: kind and all processing "
                    "parameters in effect",
    "worker_end": "capture worker exited",
    "burst": "one capture burst: timing, fps, frame statistics before/after "
             "calibration, alignment shift, integration depth, raw vs "
             "track-confirmed star counts with the adaptive detection "
             "threshold, and the stack's corner/centre background ratio "
             "(the live flat-field verification on an evenly lit field)",
    "dark_progress": "master-dark acquisition progress with accumulating "
                     "master statistics",
    "dark_done": "master dark finished: statistics and hot-pixel count",
    "dark_saved": "calibration files written to disk",
    "flat_progress": "master-flat acquisition progress with accumulating "
                     "mean statistics and live signal level",
    "flat_wait": "pre-flight illumination guidance: signal out of the "
                 "usable band, waiting for the user to adjust the light",
    "flat_autoexp": "one iteration of the opt-in auto-exposure search for "
                    "a mid-scale flat signal",
    "flat_panel_level": "one iteration of the software flat panel's "
                        "auto-level servo",
    "flat_done": "master flat finished: normalisation statistics, or the "
                 "rejection reason",
    "flat_saved": "flat calibration files written to disk",
    "pause": "integration paused/resumed, and star-motion tracking summaries "
             "while paused",
    "solve_start": "plate solve launched: stack depth and full command line",
    "solve_end": "plate solve result: parsed RA/Dec/rotation, or failure "
                 "with the tail of solve-field's output",
    "error": "exception with full traceback",
    "ui": "a user action in the GUI",
    "deepdive": "a full per-frame diagnostic dump was written to disk "
                "(directory and file list in the event)",
}

# What each deep-dive dump contains, written into its manifest.json so the
# directory is self-describing: hand it (or its manifest plus a few arrays)
# to a person or a model and the processing state is fully reconstructable.
DEEPDIVE_README = {
    "light": "One LIGHT-mode integration burst in full. raw_NN.npy = luma "
             "frames as decoded (0-255 float); proc_NN.npy = the same "
             "frame after the enabled calibration steps in processing "
             "order (dark subtraction, flat correction, defect repair, "
             "alignment shift); stack.npy = the rolling mean after this "
             "burst; master_dark/master_flat/defects = calibration data "
             "in use. per_frame carries raw/proc statistics per frame; "
             "align_burst is the burst's alignment decision. Arrays are "
             "float32 (converted from float64 processing).",
    "dark_preview": "One DARK-mode preview burst. raw_NN.npy = luma frames "
                    "as decoded; residual.npy = last frame minus master "
                    "dark after defect repair (what the right-hand panel "
                    "shows); master_dark/defects = calibration in use.",
    "dark_acquire": "One burst from master-dark acquisition. raw_NN.npy = "
                    "the burst's frames; mean_so_far.npy = the streaming "
                    "per-pixel mean including this burst.",
    "flat_acquire": "One burst from master-flat acquisition. raw_NN.npy = "
                    "the burst's frames; mean_so_far.npy = the streaming "
                    "per-pixel mean including this burst.",
    "idle": "No capture was running: the session's in-memory state "
            "(current stack, calibration masters, settings) at the "
            "moment the button was pressed.",
}


class NullDebug:
    """No-op stand-in so call sites never need a conditional."""
    enabled = False

    def log(self, ev, **fields):
        pass

    def exc(self, ev, **fields):
        pass

    def close(self):
        pass


class DebugLog:
    """JSONL diagnostics designed for LLM consumption: one self-describing
    event per line, schema documented in the first line, monotonic seq
    across threads, numpy types coerced to JSON. Paste the file (or its
    tail) into a model session and it can reconstruct what happened
    without the source code."""
    enabled = True

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8")
        self._seq = 0

    @staticmethod
    def _default(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def log(self, ev, **fields):
        with self._lock:
            self._seq += 1
            rec = {"seq": self._seq,
                   "t": datetime.now(timezone.utc).isoformat(
                       timespec="milliseconds"),
                   "ev": ev}
            rec.update(fields)
            self._f.write(json.dumps(rec, default=self._default) + "\n")
            self._f.flush()

    def exc(self, ev, **fields):
        self.log(ev, traceback=traceback.format_exc(), **fields)

    def close(self):
        with self._lock:
            self._f.close()


def start_debug(args):
    """Open the debug log and write the session header."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.debug_file or os.path.join(
        args.data_dir, f"camobs_debug_{ts}.jsonl")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    dbg = DebugLog(path)
    try:
        v4l2ver = subprocess.run(["v4l2-ctl", "--version"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout.strip()
    except Exception as e:
        v4l2ver = f"unavailable: {e}"
    dbg.log("session_start",
            tool="cam_observe.py", tool_version=TOOL_VERSION,
            argv=sys.argv[1:],
            python=sys.version.split()[0], numpy=np.__version__,
            tk=str(tk.TkVersion), platform=platform.platform(),
            v4l2_ctl=v4l2ver,
            schema=DEBUG_SCHEMA,
            notes="One JSON object per line, chronological; 'seq' is "
                  "monotonic across threads. Pixel statistics are in the "
                  "8-bit luma float domain (0-255 ADU) regardless of the "
                  "sensor's native depth (Y16 is normalised /257). "
                  "Exposure values are UVC exposure_time_absolute units "
                  "of 100 microseconds. 'burst' events are the heartbeat: "
                  "one per v4l2-ctl capture invocation. A healthy dark "
                  "preview has residual mad_sigma near the sensor's "
                  "temporal noise and mean near 0; a healthy light burst "
                  "shows raw mad_sigma > proc mad_sigma after calibration "
                  "and ring growing to maxn.")
    return dbg


# ----------------------------------------------------------------------------
# GUI application
# ----------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 480, 360
AUX_W = 220  # width of the bullseye / noise / histogram panel column

DARK_BANNER = ("DARK MODE — lens capped: build / verify the master dark",
               "#1c1c1c", "#e0e0e0")
LIGHT_BANNER = ("LIGHT MODE — capturing photons: integrating and extracting",
                "#143a63", "#ffffff")

# UVC processing controls surfaced in the settings panel (when the device
# actually has them); exposure has its own dedicated control
UVC_PANEL_CTRLS = ("gain", "gamma", "brightness", "contrast",
                   "sharpness", "saturation")


class App:
    def __init__(self, root, args, dbg=None):
        self.root = root
        self.args = args
        self.dbg = dbg or NullDebug()
        root.title("cam_observe — UVC dark/defect + live stack + plate solve")

        self.q = queue.Queue()
        self.stop_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.deepdive_evt = threading.Event()  # dump the next burst in full
        self.worker = None          # the active capture thread (any kind)

        # camera state (set by Probe)
        self.fourcc = None
        self.sizes = []             # [(w, h), ...]
        self.exp_rng = {}
        self.dev_tag = ""
        self.exp_value = 1024      # mirrored from the Tk var on the UI thread
        self.stack_max_value = 32  # so workers never touch Tk variables
        self.star_target_value = 30  # ~how many stars the user says are visible
        self.reset_on_resume = True

        # calibration state
        self.master_dark = None     # float64 luma 0-255 at current resolution
        self.master_exp = None      # exposure units the master was built at
        self.defects = None         # np.ndarray [[y, x], ...]
        self.master_flat = None     # normalised gain map (median 1) or None
        self.flat_exp = None        # exposure units the flat was built at
        self.panel_win = None       # software flat panel window, if open

        # stack state (written by the light worker; UI reads snapshots
        # passed through the queue)
        self.stack_sum = None
        self.stack_n = 0
        self.stars = []

        self._photo_live = None     # keep references or Tk drops the images
        self._photo_stack = None

        self.solving = False
        self.last_solve_t = 0.0
        self.status_base = "starting…"
        self.last_evt_t = time.monotonic()
        # live display target; grows when the user enlarges the window
        self.canvas_w, self.canvas_h = CANVAS_W, CANVAS_H

        # visualisation state
        self.solve_history = []   # [{t, ra, dec, sep_arcmin}, ...]
        self.pole_xy = None       # pole pixel from last solve's WCS
        self.noise_hist = []      # [(n, stack_sigma, single_sigma), ...]
        self._last_hist_series = []   # cached so a popout can redraw on resize
        self._popouts = {}        # kind -> Canvas in an expanded popout window
        self._popout_wins = {}    # kind -> Toplevel
        self._img_popouts = {}    # "live"/"stack" -> Canvas (image windows)
        self._img_popout_wins = {}
        self._main_cw, self._main_ch = CANVAS_W, CANVAS_H

        # "known-good" solve region: when dark/defect correction can't clean
        # the whole frame, the user drags a rectangle on the LIGHT stack view
        # and only that crop is handed to the solver. Full-res stack pixels.
        self.roi = None           # (x0, y0, x1, y1) or None = whole frame
        self._roi_arm = False     # next drag on the stack canvas defines it
        self._roi_drag = None     # (x0, y0, x1, y1) live drag, canvas px
        self._xform_stack = None  # (ox, oy, s, stride) of last stack draw
        self._solve_roi = None    # crop origin snapshot for the in-flight solve

        self._build_ui()
        self._set_mode_banner()
        self._draw_bullseye()
        self._draw_noise()
        self._draw_hist([])
        root.after(100, self._poll)
        # probe the default device automatically -- a blank, disabled UI with
        # no explanation was the worst possible first impression
        root.after(300, self.on_probe)

    # ---------------- UI construction ----------------

    def _build_ui(self):
        self.banner = tk.Label(self.root, font=("TkDefaultFont", 12, "bold"),
                               pady=6)
        self.banner.pack(fill="x")

        # activity line: always says what the tool is doing right now
        self.lbl_status = tk.Label(self.root, text="starting…", anchor="w",
                                   font=("TkDefaultFont", 10, "bold"),
                                   bg="#202020", fg="#80c0ff", padx=8, pady=3)
        self.lbl_status.pack(fill="x")

        # canvases grow with the window: drag the window larger and both
        # panels expand; the rational scaler refits the image each update
        disp = tk.Frame(self.root)
        disp.pack(fill="both", expand=True, padx=4, pady=4)
        self.lf_left = tk.LabelFrame(disp,
                                     text="Raw stream (click to pop out)")
        self.lf_left.pack(side="left", padx=2, fill="both", expand=True)
        self.canvas_live = tk.Canvas(self.lf_left, width=CANVAS_W,
                                     height=CANVAS_H, bg="#141414",
                                     highlightthickness=0)
        self.canvas_live.pack(fill="both", expand=True)
        self.lf_right = tk.LabelFrame(disp, text="Dark-corrected residual")
        self.lf_right.pack(side="left", padx=2, fill="both", expand=True)
        self.canvas_stack = tk.Canvas(self.lf_right, width=CANVAS_W,
                                      height=CANVAS_H, bg="#141414",
                                      highlightthickness=0)
        self.canvas_stack.pack(fill="both", expand=True)
        self.canvas_live.bind("<Configure>", self._on_canvas_resize)
        # dedicated live image windows: single click on the raw view,
        # double-click on the stack view (single click there is ROI drag)
        self.canvas_live.bind("<Button-1>",
                              lambda e: self._popout_image("live"))
        self.canvas_stack.bind("<Double-Button-1>",
                               lambda e: self._popout_image("stack"))
        # drag a known-good solve region on the stack view
        self.canvas_stack.bind("<ButtonPress-1>", self._on_roi_press)
        self.canvas_stack.bind("<B1-Motion>", self._on_roi_drag)
        self.canvas_stack.bind("<ButtonRelease-1>", self._on_roi_release)

        # aux visualisation column: pole bullseye, noise convergence,
        # histogram -- fixed size, main canvases take the slack
        aux = tk.Frame(disp)
        aux.pack(side="left", fill="y", padx=2)
        lfp = tk.LabelFrame(aux, text="Pole offset (θ = RA)")
        lfp.pack(fill="x")
        self.canvas_pole = tk.Canvas(lfp, width=AUX_W, height=AUX_W,
                                     bg="#101018", highlightthickness=0)
        self.canvas_pole.pack()
        lfn = tk.LabelFrame(aux, text="Stack noise vs √N")
        lfn.pack(fill="x", pady=2)
        self.canvas_noise = tk.Canvas(lfn, width=AUX_W, height=110,
                                      bg="#101018", highlightthickness=0)
        self.canvas_noise.pack()
        lfh = tk.LabelFrame(aux, text="Histogram (log y)")
        lfh.pack(fill="x")
        self.canvas_hist = tk.Canvas(lfh, width=AUX_W, height=110,
                                     bg="#101018", highlightthickness=0)
        self.canvas_hist.pack()

        # click any aux graph to pop it out larger in its own resizable window
        for c, kind in ((self.canvas_pole, "pole"),
                        (self.canvas_noise, "noise"),
                        (self.canvas_hist, "hist")):
            c.configure(cursor="hand2")
            c.bind("<Button-1>",
                   lambda e, k=kind: self._popout_graph(k))

        self.root.minsize(2 * CANVAS_W + AUX_W + 60, CANVAS_H + 430)

        ctl = tk.Frame(self.root)
        ctl.pack(fill="x", padx=4)

        # capture bar: the two decisions that matter most -- which MODE the
        # session is in, and running-or-not -- get the biggest, most
        # explicit controls, on their own row above everything else
        bar = tk.Frame(ctl)
        bar.pack(fill="x", pady=(4, 2))
        tk.Label(bar, text="Mode:",
                 font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.var_mode = tk.StringVar(value="dark")
        for text, val, sel in (
                ("DARK — lens capped", "dark", "#c9c9c9"),
                ("LIGHT — capturing photons", "light", "#aecdf0")):
            tk.Radiobutton(bar, text=text, value=val,
                           variable=self.var_mode,
                           command=self.on_mode_change, indicatoron=0,
                           font=("TkDefaultFont", 10, "bold"),
                           padx=10, pady=5, selectcolor=sel
                           ).pack(side="left", padx=3)
        self.btn_start = tk.Button(bar, text="▶  Start dark preview",
                                   command=self.on_start, state="disabled",
                                   font=("TkDefaultFont", 11, "bold"),
                                   padx=16, pady=4, width=20,
                                   disabledforeground="#909090")
        self.btn_start.pack(side="left", padx=(24, 4))
        self.btn_stop = tk.Button(bar, text="■  Stop", command=self.on_stop,
                                  state="disabled",
                                  font=("TkDefaultFont", 11, "bold"),
                                  padx=16, pady=4,
                                  disabledforeground="#909090")
        self.btn_stop.pack(side="left", padx=4)
        # "that looks wrong" button: one press writes a self-describing
        # per-frame diagnostic dump of the next burst (or, idle, of the
        # session state) -- the artefact to hand over when asking why
        self.btn_deepdive = tk.Button(bar, text="Deep-dive dump",
                                      command=self.on_deepdive)
        self.btn_deepdive.pack(side="right", padx=4)

        # camera: everything about the device itself
        lf_cam = tk.LabelFrame(ctl, text="Camera")
        lf_cam.pack(fill="x", pady=2)
        r0 = tk.Frame(lf_cam); r0.pack(fill="x", pady=2)
        tk.Label(r0, text="Device:").pack(side="left")
        devs = sorted(glob.glob("/dev/video*")) or [self.args.device]
        self.var_dev = tk.StringVar(value=self.args.device if
                                    self.args.device in devs else devs[0])
        self.cmb_dev = ttk.Combobox(r0, textvariable=self.var_dev,
                                    values=devs, width=14)
        self.cmb_dev.pack(side="left", padx=2)
        self.cmb_dev.bind("<<ComboboxSelected>>", lambda e: self.on_probe())
        self.btn_probe = tk.Button(r0, text="Probe", command=self.on_probe)
        self.btn_probe.pack(side="left", padx=4)
        tk.Label(r0, text="Resolution:").pack(side="left")
        self.var_res = tk.StringVar()
        self.cmb_res = ttk.Combobox(r0, textvariable=self.var_res, width=16,
                                    state="readonly")
        self.cmb_res.pack(side="left", padx=2)
        self.cmb_res.bind("<<ComboboxSelected>>", lambda e: self.on_res_change())
        self.lbl_native = tk.Label(r0, text="")
        self.lbl_native.pack(side="left", padx=4)
        tk.Label(r0, text="Exposure (100µs units):").pack(side="left")
        self.var_exp = tk.IntVar(value=self.exp_value)
        self.var_exp.trace_add("write", self._on_exp_change)
        self.spn_exp = tk.Spinbox(r0, from_=1, to=65535, width=7,
                                  textvariable=self.var_exp)
        self.spn_exp.pack(side="left", padx=2)
        self.lbl_exp_rng = tk.Label(r0, text="")
        self.lbl_exp_rng.pack(side="left")

        # UVC processing controls, populated after Probe with the
        # device's REAL controls and ranges
        self.frm_uvc = tk.Frame(lf_cam)
        self.frm_uvc.pack(fill="x", pady=2)
        tk.Label(self.frm_uvc, text="UVC controls: (probe a device)"
                 ).pack(side="left")

        # calibration masters: dark and flat rows share one frame, and each
        # row label says which mode its acquisition needs -- that is also
        # why the wrong-mode acquire button is greyed out
        lf_cal = tk.LabelFrame(ctl, text="Calibration masters "
                                         "(per device + resolution)")
        lf_cal.pack(fill="x", pady=2)
        r1 = tk.Frame(lf_cal); r1.pack(fill="x", pady=1)
        tk.Label(r1, text="Dark (DARK mode):", width=17,
                 anchor="w").pack(side="left")
        tk.Label(r1, text="frames:").pack(side="left")
        self.var_dark_n = tk.IntVar(value=64)
        tk.Spinbox(r1, from_=8, to=1024, width=5,
                   textvariable=self.var_dark_n).pack(side="left")
        self.btn_dark = tk.Button(r1, text="Acquire master dark",
                                  command=self.on_acquire_dark,
                                  state="disabled")
        self.btn_dark.pack(side="left", padx=4)
        self.prg = ttk.Progressbar(r1, length=120, mode="determinate")
        self.prg.pack(side="left", padx=4)
        self.lbl_dark = tk.Label(r1, text="no master dark")
        self.lbl_dark.pack(side="left")
        self.btn_viewdark = tk.Button(r1, text="View dark/defects",
                                      command=self.on_view_dark,
                                      state="disabled")
        self.btn_viewdark.pack(side="left", padx=4)

        r1b = tk.Frame(lf_cal); r1b.pack(fill="x", pady=1)
        tk.Label(r1b, text="Flat (LIGHT mode):", width=17,
                 anchor="w").pack(side="left")
        tk.Label(r1b, text="frames:").pack(side="left")
        self.var_flat_n = tk.IntVar(value=64)
        tk.Spinbox(r1b, from_=8, to=1024, width=5,
                   textvariable=self.var_flat_n).pack(side="left")
        self.btn_flat = tk.Button(r1b, text="Acquire master flat",
                                  command=self.on_acquire_flat,
                                  state="disabled")
        self.btn_flat.pack(side="left", padx=4)
        # opt-in: search the exposure for mid-scale signal. Off by default
        # because the flat SHOULD be shot at the observing exposure (the
        # ISP's amplification may be nonlinear with exposure) -- the right
        # knob is normally the illumination, which pre-flight guides
        self.var_flat_autoexp = tk.BooleanVar(value=False)
        tk.Checkbutton(r1b, text="auto exposure",
                       variable=self.var_flat_autoexp).pack(side="left")
        self.lbl_flat = tk.Label(r1b, text="no master flat")
        self.lbl_flat.pack(side="left", padx=4)
        self.btn_viewflat = tk.Button(r1b, text="View flat",
                                      command=self.on_view_flat,
                                      state="disabled")
        self.btn_viewflat.pack(side="left", padx=4)
        # software flat panel: a uniform grey window on any attached screen
        # is a light source whose level the TOOL can turn -- no camera
        # setting has to change to hit the target signal band
        self.btn_panel = tk.Button(r1b, text="Flat panel",
                                   command=self.on_flat_panel)
        self.btn_panel.pack(side="left", padx=(12, 2))
        tk.Label(r1b, text="level:").pack(side="left")
        self.var_panel_level = tk.IntVar(value=180)
        self.var_panel_level.trace_add("write", self._apply_panel_level)
        tk.Spinbox(r1b, from_=0, to=255, width=4,
                   textvariable=self.var_panel_level).pack(side="left")
        self.var_panel_auto = tk.BooleanVar(value=True)
        tk.Checkbutton(r1b, text="auto level",
                       variable=self.var_panel_auto).pack(side="left")

        # integration: what the LIGHT worker does with each frame
        lf_int = tk.LabelFrame(ctl, text="Integration (LIGHT mode)")
        lf_int.pack(fill="x", pady=2)
        r2 = tk.Frame(lf_int); r2.pack(fill="x", pady=1)
        tk.Label(r2, text="Rolling window:").pack(side="left")
        self.var_stack_max = tk.IntVar(value=self.stack_max_value)
        self.var_stack_max.trace_add("write", self._on_stack_max_change)
        tk.Spinbox(r2, from_=2, to=1024, width=5,
                   textvariable=self.var_stack_max).pack(side="left")
        tk.Label(r2, text="frames").pack(side="left")
        self.var_sub = tk.BooleanVar(value=True)
        self.var_fix = tk.BooleanVar(value=True)
        self.var_align = tk.BooleanVar(value=True)
        tk.Checkbutton(r2, text="subtract dark",
                       variable=self.var_sub).pack(side="left", padx=2)
        # flat toggle + how the flat is applied (divide = true gain
        # correction on a linear path; subtract = background-scaled
        # structure removal, safe on nonlinear paths)
        self.var_flat = tk.BooleanVar(value=True)
        tk.Checkbutton(r2, text="flat correct",
                       variable=self.var_flat).pack(side="left")
        self.var_flat_mode = tk.StringVar(value="divide")
        ttk.Combobox(r2, textvariable=self.var_flat_mode, width=8,
                     state="readonly",
                     values=("divide", "subtract")).pack(side="left")
        tk.Checkbutton(r2, text="repair defects",
                       variable=self.var_fix).pack(side="left")
        tk.Checkbutton(r2, text="align (phase corr.)",
                       variable=self.var_align).pack(side="left")
        self.btn_pause = tk.Button(r2, text="Pause integration",
                                   command=self.on_pause, state="disabled")
        self.btn_pause.pack(side="left", padx=8)
        self.var_reset_resume = tk.BooleanVar(value=True)
        self.var_reset_resume.trace_add(
            "write", lambda *a: setattr(self, "reset_on_resume",
                                        bool(self.var_reset_resume.get())))
        tk.Checkbutton(r2, text="reset stack on resume",
                       variable=self.var_reset_resume).pack(side="left")
        self.btn_reset = tk.Button(r2, text="Reset stack",
                                   command=self.on_reset_stack)
        self.btn_reset.pack(side="left", padx=8)

        # integration status
        r2b = tk.Frame(lf_int); r2b.pack(fill="x", pady=1)
        # roughly how many stars the user judges visible: detection adapts
        # its threshold toward this count, with the track filter keeping
        # the lowered threshold honest
        tk.Label(r2b, text="Visible stars ≈").pack(side="left", padx=(4, 0))
        self.var_star_target = tk.IntVar(value=self.star_target_value)
        self.var_star_target.trace_add("write", self._on_star_target_change)
        tk.Spinbox(r2b, from_=1, to=300, width=4,
                   textvariable=self.var_star_target).pack(side="left",
                                                           padx=(0, 8))
        self.lbl_stack = tk.Label(r2b, text="integrating: 0 frames")
        self.lbl_stack.pack(side="left", padx=4)
        self.lbl_stars = tk.Label(r2b, text="stars: -")
        self.lbl_stars.pack(side="left", padx=8)

        # plate solving
        lf_solve = tk.LabelFrame(ctl, text="Plate solving")
        lf_solve.pack(fill="x", pady=2)
        r3 = tk.Frame(lf_solve); r3.pack(fill="x", pady=1)
        tk.Label(r3, text="solve-field:").pack(side="left")
        self.var_solver = tk.StringVar(value=self.args.solver)
        tk.Entry(r3, textvariable=self.var_solver, width=16).pack(side="left")
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

        # known-good solve region (crop fed to the solver when the
        # dark/defect correction can't clean the whole frame)
        r3b = tk.Frame(lf_solve); r3b.pack(fill="x", pady=1)
        tk.Label(r3b, text="Solve region:").pack(side="left")
        self.btn_roi = tk.Button(r3b, text="Select on stack",
                                 command=self.on_roi_select)
        self.btn_roi.pack(side="left", padx=4)
        self.btn_roi_coords = tk.Button(r3b, text="Enter coords…",
                                        command=self.on_roi_coords)
        self.btn_roi_coords.pack(side="left", padx=4)
        self.btn_roi_clear = tk.Button(r3b, text="Clear (whole frame)",
                                       command=self.on_roi_clear,
                                       state="disabled")
        self.btn_roi_clear.pack(side="left")
        self.lbl_roi = tk.Label(r3b, text="whole frame")
        self.lbl_roi.pack(side="left", padx=8)

        # sky position / polar alignment readout
        r4 = tk.Frame(lf_solve); r4.pack(fill="x", pady=2)
        self.lbl_solve = tk.Label(r4, text="sky position: (not solved)",
                                  font=("TkDefaultFont", 10, "bold"),
                                  anchor="w")
        self.lbl_solve.pack(side="left", fill="x")

        # log
        self.txt = tk.Text(self.root, height=7, state="disabled",
                           font=("TkFixedFont", 9))
        self.txt.pack(fill="x", padx=4, pady=4)

    # ---------------- small helpers ----------------

    def log(self, msg):
        self.txt.configure(state="normal")
        self.txt.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def set_status(self, text, fg="#80c0ff"):
        self.status_base = text
        self.last_evt_t = time.monotonic()
        self.lbl_status.configure(text=text, fg=fg)

    def _on_canvas_resize(self, event):
        # workers downsample to this target; _show always refits the actual
        # canvas, so a mid-capture resize just upscales until the next burst
        if event.width > 50 and event.height > 50:
            self._main_cw, self._main_ch = event.width, event.height
            self._update_display_target()

    def _update_display_target(self):
        """Workers downsample to the LARGEST open view, so a big dedicated
        image window actually receives the extra resolution."""
        cw, ch = self._main_cw, self._main_ch
        for cv in self._img_popouts.values():
            if cv.winfo_exists():
                cw = max(cw, cv.winfo_width())
                ch = max(ch, cv.winfo_height())
        self.canvas_w = max(cw, CANVAS_W)
        self.canvas_h = max(ch, CANVAS_H)

    def _on_star_target_change(self, *a):
        try:
            self.star_target_value = max(1, int(self.var_star_target.get()))
        except (tk.TclError, ValueError):
            pass

    def _on_exp_change(self, *a):
        try:
            self.exp_value = int(self.var_exp.get())
        except (tk.TclError, ValueError):
            pass

    def _on_stack_max_change(self, *a):
        try:
            self.stack_max_value = max(2, int(self.var_stack_max.get()))
        except (tk.TclError, ValueError):
            pass

    def _set_mode_banner(self):
        dark = self.var_mode.get() == "dark"
        text, bg, fg = DARK_BANNER if dark else LIGHT_BANNER
        self.banner.configure(text=text, bg=bg, fg=fg)
        self.lf_right.configure(
            text=("Dark-corrected residual" if dark
                  else "Integrated stack") + " (double-click to pop out)")
        # the Start button says what Start will actually do in this mode
        self.btn_start.configure(text="▶  Start dark preview" if dark
                                 else "▶  Start integrating")

    def _update_run_ui(self, running):
        """Start/Stop prominence tracks the capture state: exactly one of
        them is ever the coloured, obvious next click."""
        if running:
            self.btn_start.configure(state="disabled", bg="#d9d9d9",
                                     fg="black",
                                     activebackground="#d9d9d9")
            self.btn_stop.configure(state="normal", bg="#c62828",
                                    fg="white",
                                    activebackground="#e53935",
                                    activeforeground="white")
        else:
            ready = self.fourcc is not None
            self.btn_start.configure(
                state="normal" if ready else "disabled",
                bg="#2e7d32" if ready else "#d9d9d9",
                fg="white" if ready else "black",
                activebackground="#388e3c" if ready else "#d9d9d9",
                activeforeground="white")
            self.btn_stop.configure(state="disabled", bg="#d9d9d9",
                                    fg="black",
                                    activebackground="#d9d9d9")

    def _update_mode_ui(self, running=None):
        """The acquire buttons follow the mode: a master dark needs the
        lens capped (DARK), a master flat needs light (LIGHT). Greying the
        wrong one out is clearer than a log message after the click."""
        if running is None:
            running = self.busy()
        dark = self.var_mode.get() == "dark"
        ready = self.fourcc is not None and not running
        self.btn_dark.configure(
            state="normal" if (ready and dark) else "disabled")
        self.btn_flat.configure(
            state="normal" if (ready and not dark) else "disabled")

    def current_size(self):
        m = re.match(r"(\d+)x(\d+)", self.var_res.get() or "")
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    def data_dir(self):
        d = self.args.data_dir
        return os.path.join(d, self.dev_tag) if self.dev_tag else d

    def _prefs_path(self):
        # lives in the per-camera data dir, so prefs are naturally per-camera
        return os.path.join(self.data_dir(), "ui_prefs.json")

    def _load_prefs(self):
        try:
            with open(self._prefs_path()) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save_prefs(self, **kv):
        prefs = self._load_prefs()
        prefs.update(kv)
        try:
            os.makedirs(self.data_dir(), exist_ok=True)
            with open(self._prefs_path(), "w") as fh:
                json.dump(prefs, fh)
        except OSError as e:
            self.dbg.log("error", where="save_prefs", err=str(e))

    def make_capture(self, dev, w, h):
        """dev/w/h are passed in (snapshotted on the UI thread) rather than
        read from Tk variables here: workers call this, and Tk variables must
        never be touched from a worker thread -- a marshalled Tcl call can
        stall against a blocked main loop."""
        cap = cc.Capture(dev, w, h, self.fourcc)
        cap.exposure = self.exp_value
        return cap

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    @staticmethod
    def _fit_scale(w, h, cw, ch):
        """Best integer zoom/subsample pair (z, q) so the image FILLS as much
        of the canvas as possible without cropping -- e.g. a 320x240 frame on
        a 480x360 canvas gets 3/2. tk.PhotoImage only scales by integer
        ratios, hence the rational search instead of a single factor. The
        zoom range adapts to the canvas so enlarged windows keep filling."""
        # z can exceed the integer-fit factor because q divides it back down
        # (e.g. 320px into 1440px wants z/q = 9/2 = 4.5)
        zmax = max(4, min(16, (cw // max(w, 1)) * 2 + 2))
        best, best_s = (1, 1), 0.0
        for z in range(1, zmax + 1):
            for q in range(1, 5):
                s = z / q
                if w * s <= cw + 0.5 and h * s <= ch + 0.5 and s > best_s:
                    best_s, best = s, (z, q)
        return best

    def _draw_bullseye(self):
        self._render_pole(self.canvas_pole)
        if "pole" in self._popouts:
            self._render_pole(self._popouts["pole"])

    def _draw_noise(self):
        self._render_noise(self.canvas_noise)
        if "noise" in self._popouts:
            self._render_noise(self._popouts["noise"])

    def _draw_hist(self, series=None):
        if series is not None:
            self._last_hist_series = series
        self._render_hist(self.canvas_hist)
        if "hist" in self._popouts:
            self._render_hist(self._popouts["hist"])

    def _popout_graph(self, kind):
        """Open (or raise) a larger, resizable window of one aux graph. It
        re-renders on resize and tracks live data through the _draw_* fan-out
        (the same renderer draws the embedded panel and the popout)."""
        self.log(f"opening {kind} graph popout")
        win = self._popout_wins.get(kind)
        if win is not None and win.winfo_exists():
            self._raise_popout(win)
            return
        titles = {"pole": "Pole offset (θ = RA)",
                  "noise": "Stack noise vs √N", "hist": "Histogram (log y)"}
        init = {"pole": (560, 560), "noise": (720, 420), "hist": (720, 420)}
        renderers = {"pole": self._render_pole, "noise": self._render_noise,
                     "hist": self._render_hist}
        desc = {
            "pole":
                "Polar-alignment scope. Each plate solve plots a point at "
                "angle = field RA and radius = the field centre's angular "
                "distance from the celestial pole (rings: 1', 5', 10', 30', "
                "1°; √ radial scale). Newest solve is bright, the trail "
                "shows drift — adjust the mount to walk the point into the "
                "centre.",
            "noise":
                "Noise of the integrated stack (robust MAD σ) vs frame "
                "count N. Grey = the ideal 1/√N improvement from averaging; "
                "green = what's measured. Where green lifts off grey is your "
                "fixed-pattern / drift floor — past it, stacking more frames "
                "stops helping. With a solve region set, σ is measured "
                "inside it. On a near-empty frame this σ reflects residual "
                "spatial structure, not a true temporal noise floor.",
            "hist":
                "Distribution of pixel values, log count on Y. Grey = raw "
                "frame, green = integrated stack. After dark subtraction the "
                "stack should pile up near 0 ADU: a narrow spike means a "
                "clean subtraction with little signal, while a bright tail "
                "is stars (or, spread across the frame, leftover structure "
                "/ stray light).",
        }
        w0, h0 = init[kind]
        win = tk.Toplevel(self.root)
        # explicit offset so it can't open exactly behind a maximised parent
        win.geometry(f"{w0}x{h0}+140+120")
        win.title(titles[kind] + " — expanded (drag edges to resize)")
        # explanation pinned at the bottom; canvas fills the rest
        cap = tk.Label(win, text=desc[kind], justify="left", anchor="w",
                       wraplength=w0 - 16, fg="#9fb0c4", bg="#181820",
                       padx=8, pady=6, font=("TkDefaultFont", 9))
        cap.pack(side="bottom", fill="x")
        cap.bind("<Configure>",
                 lambda e, lbl=cap: lbl.configure(wraplength=e.width - 16))
        cv = tk.Canvas(win, width=w0, height=h0, bg="#101018",
                       highlightthickness=0)
        cv.pack(side="top", fill="both", expand=True)
        self._popouts[kind] = cv
        self._popout_wins[kind] = win
        render = renderers[kind]
        cv.bind("<Configure>", lambda e: render(cv))  # also fires on first map

        def on_close():
            self._popouts.pop(kind, None)
            self._popout_wins.pop(kind, None)
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)
        self._raise_popout(win)
        self.dbg.log("ui", action="graph_popout", graph=kind)

    def _popout_image(self, which):
        """Dedicated live window for one of the image views ('live' raw
        stream / 'stack'), for observing at a useful size: resizable, keeps
        updating with every burst (with the star/vector/pole overlays), and
        its size feeds the worker's downsample target so a bigger window
        genuinely shows more resolution. Display-only: ROI selection stays
        on the embedded stack view."""
        win = self._img_popout_wins.get(which)
        if win is not None and win.winfo_exists():
            self._raise_popout(win)
            return
        titles = {"live": "Raw stream — live",
                  "stack": "Stack / residual — live"}
        win = tk.Toplevel(self.root)
        win.title(titles[which] + " (drag edges to resize)")
        win.geometry("900x680+120+80")
        cv = tk.Canvas(win, bg="#141414", highlightthickness=0)
        cv.pack(fill="both", expand=True)
        self._img_popouts[which] = cv
        self._img_popout_wins[which] = win
        cv.bind("<Configure>", lambda e: self._update_display_target())

        def on_close():
            self._img_popouts.pop(which, None)
            self._img_popout_wins.pop(which, None)
            self._update_display_target()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)
        self._raise_popout(win)
        self.dbg.log("ui", action="image_popout", view=which)

    @staticmethod
    def _raise_popout(win):
        """Force a popout to the front -- new Toplevels can otherwise open
        behind a maximised/fullscreen main window on some Linux WMs."""
        win.deiconify()
        win.lift()
        win.focus_force()
        win.attributes("-topmost", True)
        win.after(400, lambda: win.winfo_exists()
                  and win.attributes("-topmost", False))

    def _render_pole(self, cv):
        """Polar-scope-style chart: rings at 1'/5'/10'/30'/1° (sqrt radial
        mapping so the inner rings stay visible), each solve plotted at
        angle=RA, radius=pole separation; trail fades, newest is bright.
        Watching the dot walk toward centre IS the polar alignment.

        Hemisphere-aware: the pole (NCP/SCP) is taken from the latest solve's
        declination sign, and the angular handedness is mirrored for the SCP
        so the chart turns the same way the southern sky does (clockwise
        facing south) instead of the northern convention."""
        cv.delete("all")
        size = min(cv.winfo_width(), cv.winfo_height())
        if size < 50:                       # not yet mapped (or tiny)
            size = AUX_W
        cx = cy = size // 2
        R = cx - 14
        pts = self.solve_history[-20:]
        south = bool(pts) and pts[-1]["dec"] < 0
        ysign = 1.0 if south else -1.0  # SCP: clockwise; NCP: counter-clockwise
        for sep, lab in ((1, "1'"), (5, "5'"), (10, "10'"),
                         (30, "30'"), (60, "1°")):
            r = R * math.sqrt(sep / 60.0)
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#2c3a4c")
            cv.create_text(cx + r - 2, cy - 7, text=lab, fill="#4a5a6c",
                           font=("TkDefaultFont", 7), anchor="e")
        cv.create_line(cx - 4, cy, cx + 4, cy, fill="#80a0c0")
        cv.create_line(cx, cy - 4, cx, cy + 4, fill="#80a0c0")
        if pts:
            cv.create_text(cx + 7, cy + 8, text=("SCP" if south else "NCP"),
                           fill="#80a0c0", anchor="w",
                           font=("TkDefaultFont", 8, "bold"))
        prev = None
        for i, p in enumerate(pts):
            r = R * math.sqrt(min(p["sep_arcmin"], 60.0) / 60.0)
            a = math.radians(p["ra"])
            x = cx + r * math.cos(a)
            y = cy + ysign * r * math.sin(a)
            if prev:
                cv.create_line(prev[0], prev[1], x, y, fill="#2e5a3a")
            last = i == len(pts) - 1
            d = 4 if last else 2
            cv.create_oval(x - d, y - d, x + d, y + d,
                           fill="#00ff60" if last else "#3a7a4a", outline="")
            prev = (x, y)
        if pts:
            sep = pts[-1]["sep_arcmin"]
            txt = f"{sep:.1f}'" if sep < 100 else f"{sep / 60.0:.2f}°"
            if sep > 60:
                txt = "off-chart  " + txt
            cv.create_text(cx, size - 9,
                           text=f"{('SCP' if south else 'NCP')}  {txt}",
                           fill="#00ff60",
                           font=("TkDefaultFont", 10, "bold"))
        else:
            cv.create_text(cx, size - 9, text="no solves yet",
                           fill="#4a5a6c")

    def _render_noise(self, cv):
        """Measured stack noise vs integration depth against the 1/sqrt(N)
        ideal. Where the green curve departs from the grey one is the
        FPN/drift floor -- the actual limit of integrating deeper."""
        cv.delete("all")
        W = cv.winfo_width()
        H = cv.winfo_height()
        W = W if W > 50 else AUX_W
        H = H if H > 30 else 110
        padl, padb = 30, 14
        data = self.noise_hist
        if len(data) < 2:
            cv.create_text(W // 2, H // 2, text="integrate to populate",
                           fill="#4a5a6c")
            return
        sig1 = float(np.median([d[2] for d in data]))
        nmax = max(d[0] for d in data)
        ymax = max(max(d[1] for d in data), sig1) * 1.1 or 1.0

        def X(n):
            return padl + (W - padl - 6) * n / nmax

        def Y(s):
            return (H - padb) - (H - padb - 8) * s / ymax

        theory = [(X(n), Y(sig1 / math.sqrt(n)))
                  for n in range(1, nmax + 1)]
        for a, b in zip(theory[:-1], theory[1:]):
            cv.create_line(*a, *b, fill="#506070")
        meas = [(X(n), Y(s)) for n, s, _ in data]
        for a, b in zip(meas[:-1], meas[1:]):
            cv.create_line(*a, *b, fill="#00d050", width=2)
        cv.create_text(padl - 2, Y(ymax / 1.1), text=f"{ymax / 1.1:.2f}",
                       fill="#4a5a6c", anchor="e",
                       font=("TkDefaultFont", 7))
        cv.create_text(padl - 2, Y(0), text="0", fill="#4a5a6c", anchor="e",
                       font=("TkDefaultFont", 7))
        cv.create_text(W - 8, H - 4, text=f"N={nmax}", fill="#4a5a6c",
                       anchor="e", font=("TkDefaultFont", 7))
        cv.create_text(padl + 4, 8, anchor="w", fill="#00d050",
                       font=("TkDefaultFont", 7),
                       text=f"σ {data[-1][1]:.2f} (ideal "
                            f"{sig1 / math.sqrt(nmax):.2f})")

    def _render_hist(self, cv):
        """Log-y histogram polylines from the last series handed to
        _draw_hist; a zero-ADU tick is drawn when the range spans zero
        (residual view). series: list of (counts, color, lo, hi, label)."""
        series = self._last_hist_series
        cv.delete("all")
        W = cv.winfo_width()
        H = cv.winfo_height()
        W = W if W > 50 else AUX_W
        H = H if H > 30 else 110
        pad = 6
        if not series:
            cv.create_text(W // 2, H // 2, text="no data", fill="#4a5a6c")
            return
        for counts, color, lo, hi, label in series:
            if counts is None:
                continue
            logc = np.log10(1.0 + counts.astype(np.float64))
            top = float(logc.max()) or 1.0
            n = len(counts)
            pts = []
            for i in range(n):
                x = pad + (W - 2 * pad) * i / (n - 1)
                y = (H - 14) - (H - 24) * (logc[i] / top)
                pts.extend((x, y))
            cv.create_line(*pts, fill=color)
            if lo < 0 < hi:
                xz = pad + (W - 2 * pad) * (0 - lo) / (hi - lo)
                cv.create_line(xz, 10, xz, H - 14, fill="#506070",
                               dash=(2, 2))
        y = H - 6
        x = pad
        for counts, color, lo, hi, label in series:
            if counts is None:
                continue
            cv.create_text(x, y, text=label, fill=color, anchor="w",
                           font=("TkDefaultFont", 7))
            x += 8 + 6 * len(label)

    def _show(self, canvas, img8, which, stars=None, stride=1, vectors=None,
              cross=None):
        """Draw onto the embedded canvas, and mirror to this view's
        dedicated image window when one is open."""
        self._render_image(canvas, img8, which, stars, stride, vectors,
                           cross, main=True)
        pop = self._img_popouts.get(which)
        if pop is not None and pop.winfo_exists():
            self._render_image(pop, img8, which, stars, stride, vectors,
                               cross, main=False)

    def _render_image(self, canvas, img8, which, stars, stride, vectors,
                      cross, main):
        h, w = img8.shape
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 64 or ch < 64:  # not yet mapped by the geometry manager
            cw, ch = CANVAS_W, CANVAS_H
        z, q = self._fit_scale(w, h, cw, ch)
        photo = tk.PhotoImage(data=pgm_bytes(img8))
        if z > 1:
            photo = photo.zoom(z, z)
        if q > 1:
            photo = photo.subsample(q, q)
        ox = (cw - photo.width()) // 2
        oy = (ch - photo.height()) // 2
        canvas.delete("all")
        canvas.create_image(ox, oy, image=photo, anchor="nw")
        s = z / q  # display px per img8 px; overlays use full-res coords
        if stars:
            for st in stars[:50]:
                x = st["x"] / stride * s + ox
                y = st["y"] / stride * s + oy
                canvas.create_oval(x - 6, y - 6, x + 6, y + 6,
                                   outline="#00ff60")
        if vectors:
            for rx, ry, cx, cy in vectors[:50]:
                canvas.create_line(rx / stride * s + ox, ry / stride * s + oy,
                                   cx / stride * s + ox, cy / stride * s + oy,
                                   fill="#ffd000", width=2, arrow=tk.LAST)
        if cross:
            # celestial pole position from the last solve's WCS
            x = cross[0] / stride * s + ox
            y = cross[1] / stride * s + oy
            if -20 <= x <= cw + 20 and -20 <= y <= ch + 20:
                canvas.create_line(x - 14, y, x + 14, y, fill="#ff50ff",
                                   width=2)
                canvas.create_line(x, y - 14, x, y + 14, fill="#ff50ff",
                                   width=2)
                canvas.create_oval(x - 7, y - 7, x + 7, y + 7,
                                   outline="#ff50ff", width=2)
                canvas.create_text(x, y - 20, text="pole", fill="#ff50ff",
                                   font=("TkDefaultFont", 8, "bold"))
        # known-good solve region: record the transform so canvas clicks map
        # back to full-res pixels, and keep the box drawn across frame
        # redraws. Only the embedded canvas takes ROI drags; the popout
        # mirror still shows the committed region.
        if which == "stack" and self.var_mode.get() == "light":
            if main:
                self._xform_stack = (ox, oy, s, stride)
            if self.roi is not None:
                rx0, ry0, rx1, ry1 = self.roi
                canvas.create_rectangle(
                    rx0 / stride * s + ox, ry0 / stride * s + oy,
                    rx1 / stride * s + ox, ry1 / stride * s + oy,
                    outline="#00e5ff", width=2)
                canvas.create_text(rx0 / stride * s + ox + 2,
                                   ry0 / stride * s + oy - 2, anchor="sw",
                                   text="solve region", fill="#00e5ff",
                                   font=("TkDefaultFont", 8, "bold"))
            if main and self._roi_drag is not None:
                canvas.create_rectangle(*self._roi_drag, outline="#00e5ff",
                                        dash=(4, 3), tags="roidrag")
        canvas._photo = photo   # per-canvas reference or Tk drops the image
        if main:
            if which == "live":
                self._photo_live = photo
            else:
                self._photo_stack = photo

    # ---------------- known-good solve region (ROI) ----------------

    def on_roi_select(self):
        """Arm a one-shot rectangle drag on the stack view. The selection
        maps the displayed (downsampled, scaled) pixels back to full-res
        stack pixels, so the solver receives exactly the chosen crop."""
        if self.var_mode.get() != "light":
            self.log("switch to LIGHT mode and integrate first, then select "
                     "the solve region on the stack view")
            self.set_status("LIGHT mode needed to pick a solve region",
                            fg="#ff8060")
            return
        self._roi_arm = True
        self.btn_roi.configure(relief="sunken")
        self.set_status("drag a rectangle on the stack view to mark the "
                        "known-good solve region", fg="#ffd000")
        self.dbg.log("ui", action="roi_arm")

    def _reset_roi_state(self):
        """Drop the in-memory ROI and reset its UI -- does NOT touch the saved
        preference (used when a resolution change invalidates the pixels)."""
        self.roi = None
        self._roi_arm = False
        self._roi_drag = None
        self.canvas_stack.delete("roidrag")
        self.btn_roi.configure(relief="raised")
        self.btn_roi_clear.configure(state="disabled")
        self.lbl_roi.configure(text="whole frame")

    def on_roi_clear(self):
        self._reset_roi_state()
        self._save_roi_pref(None)  # user intent: forget it for this resolution
        self.log("solve region cleared -- solving the whole frame")
        self.dbg.log("ui", action="roi_clear")

    def _save_roi_pref(self, roi):
        """Persist (or remove) the ROI for the CURRENT resolution in the
        per-camera prefs, keyed by resolution since ROI pixels are
        resolution-specific."""
        w, h = self.current_size()
        if w is None:
            return
        rmap = self._load_prefs().get("roi", {})
        key = f"{w}x{h}"
        if roi is None:
            rmap.pop(key, None)
        else:
            rmap[key] = [int(v) for v in roi]
        self._save_prefs(roi=rmap)

    def _canvas_to_full(self, cx, cy):
        """Stack-canvas pixel -> full-res stack pixel via the transform the
        last _show captured. None if the stack has not been drawn yet."""
        xf = self._xform_stack
        if not xf or xf[2] <= 0:
            return None
        ox, oy, s, stride = xf
        return (cx - ox) / s * stride, (cy - oy) / s * stride

    def _on_roi_press(self, event):
        if self._roi_arm:
            self._roi_drag = (event.x, event.y, event.x, event.y)

    def _on_roi_drag(self, event):
        if not self._roi_arm or self._roi_drag is None:
            return
        x0, y0, _, _ = self._roi_drag
        self._roi_drag = (x0, y0, event.x, event.y)
        self.canvas_stack.delete("roidrag")
        self.canvas_stack.create_rectangle(x0, y0, event.x, event.y,
                                           outline="#00e5ff", dash=(4, 3),
                                           tags="roidrag")

    def _on_roi_release(self, event):
        if not self._roi_arm or self._roi_drag is None:
            return
        x0c, y0c, x1c, y1c = self._roi_drag
        self._roi_drag = None
        self._roi_arm = False
        self.canvas_stack.delete("roidrag")
        self.btn_roi.configure(relief="raised")
        p0 = self._canvas_to_full(min(x0c, x1c), min(y0c, y1c))
        p1 = self._canvas_to_full(max(x0c, x1c), max(y0c, y1c))
        if p0 is None or p1 is None:
            self.log("no stack image yet -- integrate in LIGHT mode, then "
                     "select the region")
            return
        self._set_roi(p0[0], p0[1], p1[0], p1[1], source="drag")

    def _set_roi(self, x0, y0, x1, y1, source="drag"):
        """Normalise, clamp to the frame, validate (>= 16px each side) and
        commit an ROI. Shared by the drag selection and the numeric dialog;
        returns True on success."""
        w, h = self.current_size()
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        x0, y0 = max(0, x0), max(0, y0)
        if w:
            x1 = min(w, x1)
        if h:
            y1 = min(h, y1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            self.log("region too small (need >= 16px each side); try again")
            self.set_status("region too small -- give it >= 16px each side",
                            fg="#ff8060")
            return False
        self.roi = (x0, y0, x1, y1)
        self.btn_roi_clear.configure(state="normal")
        frac = 100.0 * (x1 - x0) * (y1 - y0) / max(1, (w or 1) * (h or 1))
        self.lbl_roi.configure(
            text=f"x[{x0},{x1}) y[{y0},{y1})  "
                 f"({x1 - x0}x{y1 - y0}, {frac:.0f}% of frame)")
        self.log(f"solve region set ({source}): x[{x0},{x1}) y[{y0},{y1}) "
                 f"({x1 - x0}x{y1 - y0}px) -- solves use this crop")
        self.set_status("solve region set", fg="#80ff80")
        self.dbg.log("ui", action="roi_set", roi=[x0, y0, x1, y1],
                     frame=[w, h], source=source)
        if source != "restored":  # don't rewrite what we just loaded
            self._save_roi_pref(self.roi)
        return True

    def on_roi_coords(self):
        """Pop a small dialog to type the ROI in full-res pixels -- works
        without integrating (no display transform needed), unlike the drag."""
        w, h = self.current_size()
        if w is None:
            self.log("probe a device first so the frame size is known")
            self.set_status("probe a device before entering ROI coords",
                            fg="#ff8060")
            return
        win = tk.Toplevel(self.root)
        win.title("Solve region coordinates")
        win.transient(self.root)
        win.resizable(False, False)
        tk.Label(win, text=f"Frame: {w}x{h} px.  x0,y0 = top-left, "
                 "x1,y1 = bottom-right (exclusive).").grid(
                     row=0, column=0, columnspan=4, padx=8, pady=(8, 4),
                     sticky="w")
        cur = self.roi if self.roi else (0, 0, w, h)
        fields = {}
        for col, (lab, val) in enumerate(zip(("x0", "y0", "x1", "y1"), cur)):
            tk.Label(win, text=lab).grid(row=1, column=col, padx=(8, 0))
            v = tk.IntVar(value=int(val))
            tk.Spinbox(win, from_=0, to=max(w, h), width=7,
                       textvariable=v).grid(row=2, column=col, padx=4, pady=2)
            fields[lab] = v
        msg = tk.Label(win, text="", fg="#b00000")
        msg.grid(row=3, column=0, columnspan=4, padx=8, sticky="w")

        def apply_and_close():
            try:
                vals = [fields[k].get() for k in ("x0", "y0", "x1", "y1")]
            except (tk.TclError, ValueError):
                msg.configure(text="all four values must be integers")
                return
            if self._set_roi(*vals, source="manual"):
                win.destroy()
            else:
                msg.configure(text="region too small / out of bounds "
                                   "(need >= 16px each side, within frame)")

        btns = tk.Frame(win)
        btns.grid(row=4, column=0, columnspan=4, pady=8)
        tk.Button(btns, text="Apply", command=apply_and_close).pack(
            side="left", padx=4)
        tk.Button(btns, text="Cancel", command=win.destroy).pack(
            side="left", padx=4)
        win.bind("<Return>", lambda e: apply_and_close())
        win.bind("<Escape>", lambda e: win.destroy())
        win.grab_set()

    # ---------------- device probing / calibration files ----------------

    def on_probe(self):
        if self.busy():
            self.log("stop the current capture before re-probing")
            return
        dev = self.var_dev.get()
        self.cmb_dev["values"] = sorted(glob.glob("/dev/video*")) or [dev]
        self.set_status(f"probing {dev}…")
        self.root.update_idletasks()
        try:
            ranked = cc.rank_formats(cc.enum_formats(dev))
        except OSError as e:
            self.log(f"cannot open {dev}: {e}")
            self.set_status(f"cannot open {dev} — pick a device and press "
                            "Probe", fg="#ff8060")
            return
        best = cc.best_measurement_format(ranked)
        if best is None:
            self.log(f"{dev}: no uncompressed format -- unusable for "
                     "measurement (only lossy codecs offered)")
            self.set_status(f"{dev}: only lossy formats — unusable",
                            fg="#ff8060")
            return
        self.fourcc = best["fourcc"].strip()
        fmts = query_formats(dev)
        sizes = []
        for f in fmts:
            if f["fourcc"] == self.fourcc:
                sizes = [(s["w"], s["h"]) for s in f["sizes"]]
        if not sizes:
            # fourcc bookkeeping mismatch or sparse descriptors: offer every
            # discrete size any uncompressed format advertises
            for f in fmts:
                if not f.get("compressed"):
                    sizes += [(s["w"], s["h"]) for s in f["sizes"]]
            if sizes:
                self.log(f"no size list for {self.fourcc}; offering all "
                         "uncompressed sizes")
        if not sizes:
            sizes = [(640, 480)]
            self.log("device advertises no discrete sizes; assuming 640x480")
        self.sizes = sorted(set(sizes), key=lambda s: s[0] * s[1])
        self.cmb_res["values"] = [f"{w}x{h} ({aspect_str(w, h)})"
                                  for w, h in self.sizes]
        self.dev_tag = usb_device_tag(dev)  # set first: prefs key on it
        # restore the resolution last used with THIS camera, if still offered
        saved = self._load_prefs().get("resolution")
        pick = next((wh for wh in self.sizes
                     if f"{wh[0]}x{wh[1]}" == saved), None)
        sw, sh = pick if pick else self.sizes[0]
        self.var_res.set(f"{sw}x{sh} ({aspect_str(sw, sh)})")
        if pick:
            self.log(f"restored last-used resolution {sw}x{sh} "
                     "for this camera")
        # native aspect, inferred from the largest advertised frame: smaller
        # modes with a different ratio are cropped or anamorphic
        nw, nh = self.sizes[-1]
        self.lbl_native.configure(
            text=f"native {nw}x{nh} ({aspect_str(nw, nh)})")
        self.exp_rng = cc.get_ctrl_range(dev, "exposure_time_absolute")
        if self.exp_rng:
            self.lbl_exp_rng.configure(
                text=f"[{self.exp_rng.get('min', '?')}.."
                     f"{self.exp_rng.get('max', '?')}]")
            if self.exp_rng.get("max"):
                self.var_exp.set(self.exp_rng["max"])
        self._build_uvc_panel(dev)
        self.dbg.log("probe", device=dev, identity=identify(dev),
                     fourcc=self.fourcc,
                     formats_ranked=[(f["fourcc"], f["fidelity_rank"])
                                     for f in ranked],
                     resolutions=self.sizes,
                     native=list(self.sizes[-1]),
                     native_aspect=aspect_str(*self.sizes[-1]),
                     exposure_range=self.exp_rng, dev_tag=self.dev_tag,
                     data_dir=self.data_dir())
        self.dbg.log("controls", device=dev, controls=query_controls(dev))
        self.log(f"{dev}: format {self.fourcc}, "
                 f"{len(self.sizes)} resolutions, "
                 f"device tag {self.dev_tag or '(none)'}")
        self.on_res_change()
        self._update_run_ui(running=False)
        self._update_mode_ui(running=False)
        nw, nh = self.sizes[-1]
        self.set_status(
            f"ready — {dev} {self.fourcc}, {len(self.sizes)} resolutions, "
            f"native {nw}x{nh} ({aspect_str(nw, nh)}) — choose mode, "
            "then Start", fg="#80ff80")

    def _build_uvc_panel(self, dev):
        """Populate the UVC controls row with the device's REAL processing
        controls and ranges; values apply immediately via v4l2-ctl."""
        for w in self.frm_uvc.winfo_children():
            w.destroy()
        ctrls = query_controls(dev)
        present = [c for c in UVC_PANEL_CTRLS if c in ctrls]
        if not present:
            tk.Label(self.frm_uvc,
                     text="UVC controls: none exposed by device"
                     ).pack(side="left")
            return
        tk.Label(self.frm_uvc, text="UVC controls:").pack(side="left")
        for name in present:
            d = ctrls[name]
            lo = d.get("min", 0)
            hi = d.get("max", 255)
            cur = d.get("value", d.get("default", lo))
            tk.Label(self.frm_uvc, text=f"  {name}").pack(side="left")
            var = tk.IntVar(value=cur)

            def apply(n=name, v=var):
                try:
                    val = int(v.get())
                except (tk.TclError, ValueError):
                    return
                ok, err = cc.set_ctrl(dev, n, val)
                rb = cc.get_ctrl(dev, n)
                self.dbg.log("ctrl_set", name=n, value=val, ok=ok,
                             readback=rb, error=err if not ok else "")
                if not ok:
                    self.log(f"{n}={val} rejected: {err}")

            spn = tk.Spinbox(self.frm_uvc, from_=lo, to=hi, width=6,
                             textvariable=var, command=apply)
            spn.bind("<Return>", lambda e, fn=apply: fn())
            spn.pack(side="left")

    def on_res_change(self):
        self.master_dark = None
        self.master_exp = None
        self.defects = None
        self.master_flat = None
        self.flat_exp = None
        self.pole_xy = None  # pole pixel position is resolution-specific
        self._reset_roi_state()  # ROI pixels are resolution-specific too
        self.on_reset_stack()
        w, h = self.current_size()
        if w is None:
            return
        self._save_prefs(resolution=f"{w}x{h}")  # remember per camera
        # restore the ROI last used at THIS resolution on THIS camera
        saved_roi = self._load_prefs().get("roi", {}).get(f"{w}x{h}")
        if saved_roi and len(saved_roi) == 4:
            self._set_roi(*saved_roi, source="restored")
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
        fp = os.path.join(d, f"flat_{w}x{h}.npy")
        fjp = os.path.join(d, f"flat_meta_{w}x{h}.json")
        if os.path.exists(fp):
            try:
                f = np.load(fp)
                if f.shape == (h, w):
                    # defensive re-clamp: a hand-edited or stale file must
                    # not smuggle NaN/0 into the per-frame divide
                    self.master_flat = np.clip(
                        np.nan_to_num(f.astype(np.float64), nan=1.0),
                        FLAT_GAIN_MIN, FLAT_GAIN_MAX)
                    self.log(f"loaded master flat {fp}")
                else:
                    self.log(f"{fp} shape {f.shape} does not match {w}x{h}; "
                             "ignored")
            except Exception as e:
                self.log(f"could not load {fp}: {e}")
        if os.path.exists(fjp):
            try:
                self.flat_exp = json.load(open(fjp)).get("exposure_units")
            except Exception:
                pass
        self.dbg.log(
            "calibration_load", resolution=f"{w}x{h}", dir=d,
            master_found=self.master_dark is not None,
            master_stats=(frame_stats(self.master_dark)
                          if self.master_dark is not None else None),
            master_exposure_units=self.master_exp,
            defects=0 if self.defects is None else len(self.defects),
            flat_found=self.master_flat is not None,
            flat_stats=(frame_stats(self.master_flat)
                        if self.master_flat is not None else None),
            flat_exposure_units=self.flat_exp)
        self._update_dark_label()
        self._update_flat_label()
        # NB: the loaded master dark is NOT drawn into the integrated-stack
        # panel -- doing so looked like persisted stack data across restarts.
        # Use "View dark/defects" to inspect it; the panel stays blank until
        # real integration produces a stack (on_reset_stack ran above).

    def _update_dark_label(self):
        if self.master_dark is None:
            self.lbl_dark.configure(text="no master dark")
            self.btn_viewdark.configure(state="disabled")
        else:
            nd = 0 if self.defects is None else len(self.defects)
            exp = f" @ exp {self.master_exp}" if self.master_exp else ""
            self.lbl_dark.configure(
                text=f"master dark ready{exp}, {nd} defects")
            self.btn_viewdark.configure(state="normal")

    def _apply_panel_level(self, *a):
        if self.panel_win is None or not self.panel_win.winfo_exists():
            return
        try:
            v = max(0, min(255, int(self.var_panel_level.get())))
        except (tk.TclError, ValueError):
            return
        self.panel_win.configure(bg=f"#{v:02x}{v:02x}{v:02x}")

    def on_flat_panel(self):
        """Software flat panel: a uniform grey window -- put it on any
        attached screen and aim the camera at it (slightly defocused, or
        through a paper diffuser, to even out pixel structure and backlight
        non-uniformity). Its level is a knob the tool itself can turn, so
        the flat signal can be servoed into the target band without
        touching a single camera setting."""
        if self.panel_win is not None and self.panel_win.winfo_exists():
            self._raise_popout(self.panel_win)
            return
        win = tk.Toplevel(self.root)
        win.title("Flat panel — Up/Down level, F11 or double-click "
                  "fullscreen, Esc closes")
        win.geometry("640x480+180+120")
        self.panel_win = win
        self._apply_panel_level()

        def bump(d):
            try:
                v = int(self.var_panel_level.get())
            except (tk.TclError, ValueError):
                v = 180
            self.var_panel_level.set(max(0, min(255, v + d)))

        def toggle_fs(_e=None):
            win.attributes("-fullscreen",
                           not win.attributes("-fullscreen"))

        win.bind("<Up>", lambda e: bump(+5))
        win.bind("<Down>", lambda e: bump(-5))
        win.bind("<F11>", toggle_fs)
        win.bind("<Double-Button-1>", toggle_fs)
        win.bind("<Escape>", lambda e: win.destroy())
        self._raise_popout(win)
        self.dbg.log("ui", action="flat_panel_open",
                     level=self.var_panel_level.get())
        self.log("flat panel open -- aim the camera at it (slight defocus "
                 "or a paper diffuser evens out screen structure); Up/Down "
                 "adjusts level, F11/double-click toggles fullscreen")

    def _update_flat_label(self):
        if self.master_flat is None:
            self.lbl_flat.configure(text="no master flat")
            self.btn_viewflat.configure(state="disabled")
        else:
            exp = f" @ exp {self.flat_exp}" if self.flat_exp else ""
            self.lbl_flat.configure(text=f"master flat ready{exp}")
            self.btn_viewflat.configure(state="normal")

    def on_view_flat(self):
        """Inspection + analysis window: the gain map auto-stretched (the
        vignette profile and dust shadows should be obvious; pure noise
        means signal-starved), its radial gain profile, the per-pixel noise
        the divide injects, and a corner-vs-centre verdict saying whether
        the shading is normal falloff or upstream LSC over-correction."""
        if self.master_flat is None:
            return
        w, h = self.current_size()
        top = tk.Toplevel(self.root)
        top.title(f"Master flat {w}x{h}")
        vw, vh = self.canvas_w, self.canvas_h  # match current display size
        cv = tk.Canvas(top, width=vw, height=vh, bg="#141414",
                       highlightthickness=0)
        cv.pack()
        small, _ = downsample_to_fit(self.master_flat, vw, vh)
        img8 = stretch_to_u8(small)
        z, q = self._fit_scale(img8.shape[1], img8.shape[0], vw, vh)
        photo = tk.PhotoImage(data=pgm_bytes(img8))
        if z > 1:
            photo = photo.zoom(z, z)
        if q > 1:
            photo = photo.subsample(q, q)
        cv.create_image((vw - photo.width()) // 2,
                        (vh - photo.height()) // 2, image=photo, anchor="nw")
        top._photo = photo  # keep a reference or Tk drops the image

        # radial gain profile: gain vs distance from centre, 1.0 reference
        # dashed -- the shape separates optical falloff (drooping) from LSC
        # over-correction (rising toward the corners)
        PH = 150
        pc = tk.Canvas(top, width=vw, height=PH, bg="#101018",
                       highlightthickness=0)
        pc.pack(fill="x")
        prof_r, prof_g = radial_profile(small)
        gmin = min(float(prof_g.min()), 1.0)
        gmax = max(float(prof_g.max()), 1.0)
        span = (gmax - gmin) or 0.01
        gmin -= 0.08 * span
        gmax += 0.08 * span
        padl, padb, padt = 46, 16, 8

        def X(r):
            return padl + (vw - padl - 10) * r

        def Y(g):
            return (PH - padb) - (PH - padb - padt) * (g - gmin) \
                / (gmax - gmin)

        pc.create_line(X(0.0), Y(1.0), X(1.0), Y(1.0), fill="#506070",
                       dash=(3, 3))
        pc.create_text(padl - 4, Y(1.0), text="1.00", fill="#4a5a6c",
                       anchor="e", font=("TkDefaultFont", 7))
        for gv in (float(prof_g.min()), float(prof_g.max())):
            if abs(gv - 1.0) > 0.02 * span:
                pc.create_text(padl - 4, Y(gv), text=f"{gv:.2f}",
                               fill="#4a5a6c", anchor="e",
                               font=("TkDefaultFont", 7))
        pts = []
        for r, g in zip(prof_r, prof_g):
            pts.extend((X(r), Y(g)))
        pc.create_line(*pts, fill="#00d050", width=2)
        pc.create_text(X(0.0), PH - 6, text="centre", fill="#4a5a6c",
                       anchor="w", font=("TkDefaultFont", 7))
        pc.create_text(X(1.0), PH - 6, text="corner", fill="#4a5a6c",
                       anchor="e", font=("TkDefaultFont", 7))

        centre_g = float(np.mean(prof_g[:3]))
        corner_g = float(np.mean(prof_g[-3:]))
        ratio = corner_g / centre_g if centre_g else 0.0
        noise = flat_noise_sigma(self.master_flat)
        exp = f"{self.flat_exp} units" if self.flat_exp else "unknown"
        info = (f"exposure {exp}  |  "
                f"gain min {float(self.master_flat.min()):.3f} "
                f"max {float(self.master_flat.max()):.3f}  |  "
                f"divide injects ~{noise * 100:.2f}% px noise  |  "
                "display auto-stretched")
        tk.Label(top, text=info, anchor="w", padx=6, pady=2).pack(fill="x")
        dev_pct = (ratio - 1.0) * 100.0
        if ratio > 1.02:
            verdict = (f"corners {dev_pct:+.1f}% vs centre — OVER-corrected "
                       "upstream (LSC reverse vignetting); the flat darkens "
                       "them back")
            fg = "#a05000"
        elif ratio < 0.98:
            verdict = (f"corners {dev_pct:+.1f}% vs centre — normal "
                       "falloff; the flat brightens them")
            fg = "#006000"
        else:
            verdict = (f"corners {dev_pct:+.1f}% vs centre — field already "
                       "flat to ±2%")
            fg = "#006000"
        tk.Label(top, text=verdict, anchor="w", padx=6, pady=2, fg=fg,
                 font=("TkDefaultFont", 9, "bold")).pack(fill="x")

    def on_view_dark(self):
        """Inspection window: the master dark auto-stretched, with every
        defect-map pixel marked. The stretch makes the FPN structure and the
        LSC shading visible; the marks confirm what will be repaired."""
        if self.master_dark is None:
            return
        w, h = self.current_size()
        nd = 0 if self.defects is None else len(self.defects)
        top = tk.Toplevel(self.root)
        top.title(f"Master dark {w}x{h} — {nd} defects")
        vw, vh = self.canvas_w, self.canvas_h  # match current display size
        cv = tk.Canvas(top, width=vw, height=vh, bg="#141414",
                       highlightthickness=0)
        cv.pack()
        small, stride = downsample_to_fit(self.master_dark, vw, vh)
        img8 = stretch_to_u8(small)
        z, q = self._fit_scale(img8.shape[1], img8.shape[0], vw, vh)
        photo = tk.PhotoImage(data=pgm_bytes(img8))
        if z > 1:
            photo = photo.zoom(z, z)
        if q > 1:
            photo = photo.subsample(q, q)
        ox = (vw - photo.width()) // 2
        oy = (vh - photo.height()) // 2
        cv.create_image(ox, oy, image=photo, anchor="nw")
        top._photo = photo  # keep a reference or Tk drops the image
        s = z / q
        drawn = 0
        if self.defects is not None:
            for y, x in self.defects[:3000]:
                px = x / stride * s + ox
                py = y / stride * s + oy
                cv.create_oval(px - 2, py - 2, px + 2, py + 2,
                               outline="#ff4040")
                drawn += 1
        fpn = float(self.master_dark.std())
        exp = f"{self.master_exp} units" if self.master_exp else "unknown"
        info = (f"exposure {exp}  |  FPN σ {fpn:.2f} ADU  |  "
                f"{nd} hot pixels"
                + (f" (first {drawn} marked)" if nd > drawn else " marked")
                + "  |  display auto-stretched")
        tk.Label(top, text=info, anchor="w", padx=6, pady=4).pack(fill="x")

    # ---------------- mode / actions ----------------

    def on_mode_change(self):
        self.dbg.log("ui", action="mode_change", mode=self.var_mode.get())
        if self.busy():
            self.on_stop()
        self._set_mode_banner()
        self._update_mode_ui()

    def on_reset_stack(self):
        self.stack_sum = None
        self.stack_n = 0
        self.stars = []
        self.noise_hist = []
        self._draw_noise()
        # blank the integrated-stack panel so a previous run's image can't
        # masquerade as live data (the stack lives only in memory; nothing
        # is reloaded from disk)
        self.canvas_stack.delete("all")
        cw = self.canvas_stack.winfo_width()
        ch = self.canvas_stack.winfo_height()
        cw = cw if cw > 64 else CANVAS_W
        ch = ch if ch > 64 else CANVAS_H
        self.canvas_stack.create_text(cw // 2, ch // 2, fill="#404040",
                                      text="(empty — Start to integrate)")
        self._photo_stack = None
        self.lbl_stack.configure(text="integrating: 0 frames")
        self.lbl_stars.configure(text="stars: -")
        self.btn_solve.configure(state="disabled")
        self.btn_fits.configure(state="disabled")

    def on_start(self):
        if self.busy() or self.fourcc is None:
            return
        # snapshot every Tk-variable-backed setting HERE, on the UI thread;
        # the workers receive plain values and never touch Tk variables
        dev = self.var_dev.get()
        w, h = self.current_size()
        if w is None:
            self.log("no resolution selected -- probe a device first")
            return
        self.stop_evt.clear()
        self.pause_evt.clear()
        self._update_run_ui(running=True)
        self._update_mode_ui(running=True)
        if self.var_mode.get() == "dark":
            self.worker = threading.Thread(target=self._dark_preview_worker,
                                           args=(dev, w, h), daemon=True)
            self.set_status("DARK PREVIEW — starting capture…")
            self.log("dark preview: raw vs dark-corrected residual "
                     "(fixed stretch x8 -- correction should look black)")
        else:
            self.on_reset_stack()
            exp = self.exp_value
            sub = self.var_sub.get()
            fix = self.var_fix.get()
            align = self.var_align.get()
            flt = self.var_flat.get()
            fmode = self.var_flat_mode.get()
            if self.master_dark is not None and self.master_exp \
                    and exp != self.master_exp and sub:
                self.log(f"NOTE: master dark was built at exposure "
                         f"{self.master_exp}, capturing at {exp} -- "
                         "subtraction will be biased")
            if flt and self.master_flat is not None:
                if self.flat_exp and exp != self.flat_exp:
                    self.log(f"NOTE: master flat was built at exposure "
                             f"{self.flat_exp}, capturing at {exp} -- if "
                             "the ISP's amplification is nonlinear with "
                             "exposure the flat won't fully cancel the "
                             "shading; re-acquire at the observing exposure")
                if not (sub and self.master_dark is not None):
                    self.log("NOTE: flat correction without dark "
                             "subtraction -- the flat was normalised on "
                             "dark-subtracted data, so correcting raw "
                             "frames distorts it, worst in the corners "
                             "where the LSC has already amplified the "
                             "pedestal")
            self.btn_pause.configure(state="normal")
            self.set_status("INTEGRATING — starting capture…")
            self.worker = threading.Thread(target=self._light_worker,
                                           args=(dev, w, h, sub, fix, align,
                                                 flt, fmode),
                                           daemon=True)
        self.worker.start()

    def on_stop(self):
        self.dbg.log("ui", action="stop")
        self.stop_evt.set()
        self.pause_evt.clear()

    def on_pause(self):
        if not self.busy():
            return
        if self.pause_evt.is_set():
            self.pause_evt.clear()
            self.btn_pause.configure(text="Pause integration")
            self.dbg.log("pause", state="resumed",
                         reset_stack=self.reset_on_resume)
            self.log("integration resumed"
                     + (" (stack reset)" if self.reset_on_resume else ""))
        else:
            self.pause_evt.set()
            self.btn_pause.configure(text="Resume integration")
            self.dbg.log("pause", state="paused",
                         stack_frames=self.stack_n,
                         ref_stars=len(self.stars))
            self.log("integration PAUSED -- adjust the mount; star motion "
                     "arrows show where the field is going")

    def on_acquire_dark(self):
        if self.fourcc is None:
            self.log("probe a device first")
            return
        if self.var_mode.get() != "dark":
            self.log("switch to DARK mode (and cap the lens) first")
            return
        if self.busy():
            self.on_stop()
            self.worker.join(timeout=10)
            if self.worker.is_alive():
                # starting a second worker now would have two v4l2-ctl
                # processes fighting over the device
                self.log("previous capture is still shutting down -- "
                         "try again in a moment")
                self.set_status("waiting for previous capture to stop",
                                fg="#ff8060")
                return
        dev = self.var_dev.get()
        w, h = self.current_size()
        if w is None:
            self.log("no resolution selected -- probe a device first")
            return
        n = int(self.var_dark_n.get())
        self.stop_evt.clear()
        self._update_run_ui(running=True)
        self._update_mode_ui(running=True)
        self.lf_right.configure(text="Master dark (accumulating)")
        self.set_status(f"ACQUIRING MASTER DARK 0/{n} — capturing…")
        self.worker = threading.Thread(target=self._dark_worker,
                                       args=(n, dev, w, h), daemon=True)
        self.worker.start()

    def on_acquire_flat(self):
        if self.fourcc is None:
            self.log("probe a device first")
            return
        if self.var_mode.get() != "light":
            self.log("switch to LIGHT mode and point the camera at an "
                     "evenly illuminated field (twilight sky, light panel, "
                     "defocused white surface) first")
            return
        if self.busy():
            self.on_stop()
            self.worker.join(timeout=10)
            if self.worker.is_alive():
                # starting a second worker now would have two v4l2-ctl
                # processes fighting over the device
                self.log("previous capture is still shutting down -- "
                         "try again in a moment")
                self.set_status("waiting for previous capture to stop",
                                fg="#ff8060")
                return
        dev = self.var_dev.get()
        w, h = self.current_size()
        if w is None:
            self.log("no resolution selected -- probe a device first")
            return
        n = int(self.var_flat_n.get())
        autoexp = bool(self.var_flat_autoexp.get())
        panel_auto = (bool(self.var_panel_auto.get())
                      and self.panel_win is not None
                      and self.panel_win.winfo_exists())
        exp_rng = ((self.exp_rng.get("min", 1) or 1,
                    self.exp_rng.get("max", 65535) or 65535)
                   if self.exp_rng else (1, 65535))
        if panel_auto:
            self.log("flat panel auto-level: the tool will drive the panel "
                     "brightness into the target band (no camera setting "
                     "is changed)")
        if autoexp:
            self.log("NOTE: auto-exposure flat -- the chosen exposure will "
                     "generally NOT match the observing exposure; if the "
                     "ISP's amplification is nonlinear with exposure the "
                     "flat may not fully cancel the shading")
        if self.master_dark is None:
            self.log("NOTE: no master dark -- the flat will not be "
                     "dark-subtracted, so dark structure folds into the "
                     "gain map")
        elif self.master_exp and self.exp_value != self.master_exp:
            self.log(f"NOTE: master dark was built at exposure "
                     f"{self.master_exp}, flat capturing at "
                     f"{self.exp_value} -- the flat's dark subtraction "
                     "will be biased")
        self.stop_evt.clear()
        self._update_run_ui(running=True)
        self._update_mode_ui(running=True)
        self.lf_right.configure(text="Master flat (accumulating)")
        self.set_status(f"ACQUIRING MASTER FLAT 0/{n} — capturing…")
        self.worker = threading.Thread(target=self._flat_worker,
                                       args=(n, dev, w, h, autoexp,
                                             exp_rng, panel_auto),
                                       daemon=True)
        self.worker.start()

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
        self.dbg.log("ui", action="save_stack", base=base,
                     frames=self.stack_n, stack=frame_stats(stack))
        self.log(f"stack written to {base}.fits and .npy "
                 f"({self.stack_n} frames)")

    def on_solve(self):
        self._trigger_solve(auto=False)

    def _trigger_solve(self, auto):
        if self.stack_sum is None or self.stack_n == 0 or self.solving:
            return
        stack = self.stack_sum / self.stack_n
        # crop to the known-good region (if set) so only clean data reaches
        # the solver; remember the origin to put WCS pixel coords back into
        # full-frame space afterwards
        self._solve_roi = None
        if self.roi is not None:
            h, w = stack.shape
            x0 = max(0, min(self.roi[0], w))
            y0 = max(0, min(self.roi[1], h))
            x1 = max(0, min(self.roi[2], w))
            y1 = max(0, min(self.roi[3], h))
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                stack = stack[y0:y1, x0:x1]
                self._solve_roi = (x0, y0)
        if self.master_dark is None and not auto:
            self.log("solving an unsubtracted stack -- hot pixels may "
                     "masquerade as stars")
        self.solving = True
        self.last_solve_t = time.monotonic()
        self.btn_solve.configure(state="disabled")
        self.log("plate solving..." + (" (auto)" if auto else "")
                 + (f" [region {stack.shape[1]}x{stack.shape[0]}px]"
                    if self._solve_roi else ""))
        # evaluate the Tk variables NOW, on the UI thread -- the lambda runs
        # in the solve thread, where Tk variables must not be touched
        solver = self.var_solver.get()
        try:
            sc_lo = float(self.var_sc_lo.get())
            sc_hi = float(self.var_sc_hi.get())
        except (tk.TclError, ValueError):
            sc_lo = sc_hi = 0.0  # unparseable entry -> solve unhinted
        t = threading.Thread(
            target=lambda: self.q.put(
                ("solved", solve_field(stack, solver, sc_lo, sc_hi,
                                       dbg=self.dbg))),
            daemon=True)
        t.start()

    # ---------------- deep-dive diagnostics ----------------

    def _deepdive_write(self, kind, arrays, meta):
        """Write one deep-dive dump: every array as .npy (float64 stored as
        float32) plus a manifest.json that explains each file and carries
        the full processing context. Thread-safe: touches no Tk state, so
        workers call it directly."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = os.path.join(self.data_dir(), f"deepdive_{kind}_{ts}")
        os.makedirs(d, exist_ok=True)
        files = {}
        for name, arr in arrays.items():
            if arr is None:
                continue
            a = np.asarray(arr)
            np.save(os.path.join(d, name + ".npy"),
                    a.astype(np.float32) if a.dtype == np.float64 else a)
            files[name + ".npy"] = {
                "shape": list(a.shape),
                "stats": (frame_stats(a)
                          if a.ndim == 2 and a.dtype.kind == "f" else None)}
        manifest = {"tool": "cam_observe.py", "tool_version": TOOL_VERSION,
                    "kind": kind,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "readme": DEEPDIVE_README.get(kind, ""),
                    "pixel_domain": "8-bit luma float, 0-255 ADU "
                                    "(Y16 normalised /257)",
                    "files": files}
        manifest.update(meta)
        with open(os.path.join(d, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2, default=DebugLog._default)
        self.dbg.log("deepdive", kind=kind, dir=d, files=sorted(files))
        self.q.put(("log", f"deep-dive dump: {d} ({len(files)} arrays "
                    "+ manifest.json)"))
        return d

    def on_deepdive(self):
        """One press = one comprehensive diagnostic snapshot, for 'that
        looks wrong' moments: while a capture runs, the NEXT burst is
        dumped in full (every raw and processed frame, calibration data,
        parameters, per-frame statistics); idle, the session's in-memory
        state is dumped immediately."""
        self.dbg.log("ui", action="deepdive", running=self.busy())
        if self.busy():
            self.deepdive_evt.set()
            self.log("deep-dive armed -- the next capture burst will be "
                     "dumped in full")
            self.set_status("deep-dive armed — dumping the next burst",
                            fg="#ffd000")
            return
        stack = (self.stack_sum / self.stack_n
                 if self.stack_sum is not None and self.stack_n else None)
        w, h = self.current_size()
        self._deepdive_write(
            "idle",
            {"stack": stack, "master_dark": self.master_dark,
             "master_flat": self.master_flat, "defects": self.defects},
            {"device": self.var_dev.get(),
             "resolution": f"{w}x{h}" if w else None,
             "fourcc": self.fourcc,
             "exposure_units": self.exp_value,
             "master_dark_exposure_units": self.master_exp,
             "flat_exposure_units": self.flat_exp,
             "stack_frames": self.stack_n,
             "roi": self.roi,
             "stars": [dict(s) for s in self.stars[:20]]})

    # ---------------- workers (no Tk calls in here) ----------------

    def _dark_worker(self, total, dev, w, h):
        exp = self.exp_value
        self.dbg.log("worker_start", kind="dark_acquire", device=dev,
                     exposure_units=exp, frames_requested=total,
                     resolution=f"{w}x{h}", fourcc=self.fourcc)
        cap = None
        try:
            # inside the try: if construction raises, the finally still posts
            # worker_done and the UI buttons come back
            cap = self.make_capture(dev, w, h)
            cc.set_ctrl(dev, "auto_exposure", 1)
            cc.set_ctrl(dev, "exposure_time_absolute", exp)
            mean = None
            count = 0
            while count < total and not self.stop_evt.is_set():
                burst = min(16, total - count)
                dd = self.deepdive_evt.is_set()
                if dd:
                    self.deepdive_evt.clear()
                dd_raw = []
                path, _, _ = cap.capture_run(burst, discard=1,
                                             timeout=30.0, verbose=False)
                last = None
                for fr in cap.iter_luma(path, burst, discard=1):
                    count += 1
                    if mean is None:
                        mean = np.zeros_like(fr)
                    mean += (fr - mean) / count
                    if dd and len(dd_raw) < 8:  # cap the memory cost
                        dd_raw.append(fr.copy())
                    last = fr
                try:
                    os.remove(path)
                except OSError:
                    pass
                if dd and dd_raw:
                    arrays = {f"raw_{i:02d}": a
                              for i, a in enumerate(dd_raw)}
                    arrays["mean_so_far"] = mean
                    self._deepdive_write("dark_acquire", arrays, {
                        "device": dev, "resolution": f"{w}x{h}",
                        "fourcc": self.fourcc, "exposure_units": exp,
                        "frames_done": count, "frames_requested": total,
                        "fps": round(cap.last_fps, 2)})
                # downsampled COPIES: the UI must not read arrays this loop
                # keeps mutating, and full-res frames are queue bloat anyway
                msmall, _ = downsample_to_fit(mean, self.canvas_w,
                                              self.canvas_h)
                lsmall, _ = downsample_to_fit(last, self.canvas_w,
                                              self.canvas_h)
                self.dbg.log("dark_progress", frames=count, of=total,
                             capture_s=round(cap.last_capture_s, 2),
                             fps=round(cap.last_fps, 2),
                             master=frame_stats(mean))
                self.q.put(("progress", count, total,
                            msmall.copy(), lsmall.copy()))
            if mean is None:
                self.q.put(("log", "dark acquisition produced no frames"))
                return
            # clip guard (same test as cam_characterise's run_dark): a master
            # pinned at either rail is a clamped/saturated processing path,
            # not a measurement -- it must never be installed or written into
            # the PHD2 data dir for auto-load at every connect
            floor_frac = float((mean <= 0.5).mean())
            ceil_frac = float((mean >= 254.5).mean())
            clipped = floor_frac > 0.5 or ceil_frac > 0.5
            hot = cc.hot_pixel_mask(mean)
            coords = np.argwhere(hot)
            self.dbg.log("dark_done", frames=count, hot_pixels=len(coords),
                         master=frame_stats(mean), exposure_units=exp,
                         clipped=clipped,
                         clip_floor_frac=round(floor_frac, 4),
                         clip_ceil_frac=round(ceil_frac, 4))
            self.q.put(("dark_done", mean, coords, exp, count,
                        (clipped, floor_frac, ceil_frac)))
        except Exception as e:
            self.dbg.exc("error", where="dark_worker")
            self.q.put(("log", f"dark acquisition failed: {e}"))
        finally:
            if cap is not None:
                cap.close()
            self.dbg.log("worker_end", kind="dark_acquire")
            self.q.put(("worker_done",))

    def _flat_worker(self, total, dev, w, h, autoexp, exp_rng, panel_auto):
        """Master-flat acquisition. Signal-level automation in order of
        preference: the software flat panel's auto-level servo (adjusts the
        ILLUMINATION -- no camera setting changes), else the opt-in
        auto-exposure search (explicit trade-off: the flat is then not shot
        at the observing exposure), else a pre-flight loop that holds until
        the signal is in the usable band, guiding the user which way to
        adjust the light. Then a streaming per-pixel mean is normalised
        into a median-1 gain map (dark-subtracted when a master dark is
        loaded)."""
        exp = self.exp_value
        master = self.master_dark   # snapshot for the dark subtraction
        self.dbg.log("worker_start", kind="flat_acquire", device=dev,
                     exposure_units=exp, frames_requested=total,
                     resolution=f"{w}x{h}", fourcc=self.fourcc,
                     auto_exposure_search=autoexp,
                     master_dark_loaded=master is not None,
                     master_dark_exposure_units=self.master_exp)
        cap = None
        try:
            cap = self.make_capture(dev, w, h)
            cc.set_ctrl(dev, "auto_exposure", 1)
            cc.set_ctrl(dev, "exposure_time_absolute", exp)

            def measure(nburst=4):
                """One short burst -> (last_frame, median, clipped_frac)."""
                path, _, _ = cap.capture_run(nburst, discard=1,
                                             timeout=30.0, verbose=False)
                last = None
                for fr in cap.iter_luma(path, nburst, discard=1):
                    last = fr
                try:
                    os.remove(path)
                except OSError:
                    pass
                if last is None:
                    return None, 0.0, 0.0
                return (last, float(np.median(last)),
                        float((last >= 254.5).mean()))

            def small_or_none(frame):
                if frame is None:
                    return None
                s, _ = downsample_to_fit(frame, self.canvas_w,
                                         self.canvas_h)
                return s.copy()

            if panel_auto and autoexp:
                self.q.put(("log", "flat panel auto-level active -- "
                            "auto-exposure search skipped (exposure stays "
                            "at the observing value)"))
                autoexp = False

            if panel_auto:
                # illumination servo: bisect the panel grey level (display
                # response is nonlinear but monotonic) for a mid-scale
                # median. The level change is applied by the UI thread via
                # the queue; the short sleep lets the screen repaint before
                # the next measurement.
                lo_l, hi_l = 0, 255
                for _ in range(9):
                    if self.stop_evt.is_set():
                        return
                    cur_l = (lo_l + hi_l) // 2
                    self.q.put(("panel_level", cur_l))
                    time.sleep(0.35)
                    last, med, cf = measure()
                    self.dbg.log("flat_panel_level", level=cur_l,
                                 median=round(med, 2),
                                 clip_frac=round(cf, 4),
                                 window=[lo_l, hi_l])
                    self.q.put(("flat_wait", med, cf,
                                f"auto panel level: {cur_l}",
                                small_or_none(last)))
                    if last is None:
                        continue
                    if cf > FLAT_SAT_FRAC or med > FLAT_TARGET_HI:
                        hi_l = cur_l - 1
                    elif med < FLAT_TARGET_LO:
                        lo_l = cur_l + 1
                    else:
                        break
                    if lo_l > hi_l:
                        # panel range exhausted: the wait loop below says
                        # what to do (move closer / open aperture / etc.)
                        break

            if autoexp:
                # bisect the exposure range for a mid-scale median; the
                # response is monotonic in exposure and clipping counts as
                # bright. 17 iterations fully resolve 1..65535, so even a
                # hopelessly bright source converges to the range floor
                # (the pre-flight loop then says what to do about it);
                # normal cases break out of the band check much earlier.
                lo, hi = int(exp_rng[0]), int(exp_rng[1])
                cur = exp
                for _ in range(17):
                    if self.stop_evt.is_set():
                        return
                    cur = max(lo, min(hi, (lo + hi) // 2))
                    cap.exposure = cur
                    last, med, cf = measure()
                    self.dbg.log("flat_autoexp", exposure_units=cur,
                                 median=round(med, 2),
                                 clip_frac=round(cf, 4), window=[lo, hi])
                    self.q.put(("flat_wait", med, cf,
                                f"auto-exposure search: {cur} units",
                                small_or_none(last)))
                    if last is None:
                        continue
                    if cf > FLAT_SAT_FRAC or med > FLAT_TARGET_HI:
                        hi = cur - 1
                    elif med < FLAT_TARGET_LO:
                        lo = cur + 1
                    else:
                        break
                    if lo > hi:
                        break
                exp = cur
                cap.exposure = exp
                note = (f" (observing exposure is {self.exp_value} -- "
                        "mismatch will be flagged at capture start)"
                        if exp != self.exp_value else "")
                self.q.put(("log",
                            f"auto exposure for flat: {exp} units{note}"))

            # pre-flight: hold here until the signal is usable, telling the
            # user which way to adjust the light -- never accumulate a flat
            # that the normaliser would only reject afterwards
            while not self.stop_evt.is_set():
                last, med, cf = measure()
                if last is None:
                    continue
                if cf > FLAT_SAT_FRAC or med > FLAT_OK_HI:
                    guide = "too bright / clipping -- dim the illumination"
                elif med < FLAT_OK_LO:
                    guide = "too dim -- brighten the illumination"
                else:
                    break
                self.dbg.log("flat_wait", median=round(med, 2),
                             clip_frac=round(cf, 4), guide=guide)
                self.q.put(("flat_wait", med, cf, guide,
                            small_or_none(last)))
            if self.stop_evt.is_set():
                return
            mean = None
            count = 0
            while count < total and not self.stop_evt.is_set():
                burst = min(16, total - count)
                dd = self.deepdive_evt.is_set()
                if dd:
                    self.deepdive_evt.clear()
                dd_raw = []
                path, _, _ = cap.capture_run(burst, discard=1,
                                             timeout=30.0, verbose=False)
                last = None
                for fr in cap.iter_luma(path, burst, discard=1):
                    count += 1
                    if mean is None:
                        mean = np.zeros_like(fr)
                    mean += (fr - mean) / count
                    if dd and len(dd_raw) < 8:  # cap the memory cost
                        dd_raw.append(fr.copy())
                    last = fr
                try:
                    os.remove(path)
                except OSError:
                    pass
                if dd and dd_raw:
                    arrays = {f"raw_{i:02d}": a
                              for i, a in enumerate(dd_raw)}
                    arrays["mean_so_far"] = mean
                    self._deepdive_write("flat_acquire", arrays, {
                        "device": dev, "resolution": f"{w}x{h}",
                        "fourcc": self.fourcc, "exposure_units": exp,
                        "frames_done": count, "frames_requested": total,
                        "fps": round(cap.last_fps, 2)})
                # downsampled COPIES: the UI must not read arrays this loop
                # keeps mutating, and full-res frames are queue bloat anyway
                msmall, _ = downsample_to_fit(mean, self.canvas_w,
                                              self.canvas_h)
                lsmall, _ = downsample_to_fit(last, self.canvas_w,
                                              self.canvas_h)
                med = float(np.median(last))
                cf = float((last >= 254.5).mean())
                self.dbg.log("flat_progress", frames=count, of=total,
                             capture_s=round(cap.last_capture_s, 2),
                             fps=round(cap.last_fps, 2),
                             median=round(med, 2), clip_frac=round(cf, 4),
                             master=frame_stats(mean))
                self.q.put(("flat_progress", count, total,
                            msmall.copy(), lsmall.copy(), med, cf))
            if mean is None:
                self.q.put(("log", "flat acquisition produced no frames"))
                return
            gain, reject, info = build_flat(mean, master)
            self.dbg.log("flat_done", frames=count, exposure_units=exp,
                         rejected=reject,
                         gain=(frame_stats(gain) if gain is not None
                               else None), **info)
            self.q.put(("flat_done", gain, reject, info, exp, count))
        except Exception as e:
            self.dbg.exc("error", where="flat_worker")
            self.q.put(("log", f"flat acquisition failed: {e}"))
        finally:
            if cap is not None:
                cap.close()
            self.dbg.log("worker_end", kind="flat_acquire")
            self.q.put(("worker_done",))

    def _dark_preview_worker(self, dev, w, h):
        """Continuous DARK-mode preview: raw stream + dark-corrected residual
        at a fixed stretch (ideally complete blackness), with residual stats
        saying how close to ideal the correction is."""
        master = self.master_dark
        defects = self.defects
        self.dbg.log("worker_start", kind="dark_preview", device=dev,
                     exposure_units=self.exp_value,
                     resolution=f"{w}x{h}", fourcc=self.fourcc,
                     master_loaded=master is not None,
                     defects=0 if defects is None else len(defects))
        cap = None
        try:
            cap = self.make_capture(dev, w, h)
            cc.set_ctrl(dev, "auto_exposure", 1)
            cc.set_ctrl(dev, "exposure_time_absolute", self.exp_value)
            while not self.stop_evt.is_set():
                cap.exposure = self.exp_value  # live-adjustable
                dd = self.deepdive_evt.is_set()
                if dd:
                    self.deepdive_evt.clear()
                dd_raw = []
                path, _, _ = cap.capture_run(4, discard=1,
                                             timeout=60.0, verbose=False)
                last = None
                for fr in cap.iter_luma(path, 4, discard=1):
                    if dd:
                        dd_raw.append(fr.copy())
                    last = fr
                try:
                    os.remove(path)
                except OSError:
                    pass
                if last is None:
                    continue
                raw_small, _ = downsample_to_fit(last, self.canvas_w,
                                                 self.canvas_h)
                raw8 = stretch_to_u8(raw_small)
                if master is not None:
                    resid = last - master
                    if defects is not None and len(defects):
                        repair_defects(resid, defects)
                    rs, _ = downsample_to_fit(np.clip(resid, 0, None),
                                              self.canvas_w, self.canvas_h)
                    resid8 = residual_to_u8(rs)
                    stats = (float(resid.mean()),
                             float(1.4826 * np.median(np.abs(
                                 resid - np.median(resid)))),
                             float(resid.max()))
                else:
                    resid8 = None
                    stats = None
                if dd and dd_raw:
                    arrays = {f"raw_{i:02d}": a
                              for i, a in enumerate(dd_raw)}
                    arrays.update(
                        residual=(resid if master is not None else None),
                        master_dark=master, defects=defects)
                    self._deepdive_write("dark_preview", arrays, {
                        "device": dev, "resolution": f"{w}x{h}",
                        "fourcc": self.fourcc,
                        "exposure_units": cap.exposure,
                        "master_dark_exposure_units": self.master_exp,
                        "fps": round(cap.last_fps, 2),
                        "residual_stats": (frame_stats(resid)
                                           if master is not None else None)})
                self.dbg.log(
                    "burst", kind="dark_preview",
                    exposure_units=cap.exposure,
                    capture_s=round(cap.last_capture_s, 2),
                    fps=round(cap.last_fps, 2), raw=frame_stats(last),
                    residual=(frame_stats(resid) if master is not None
                              else None))
                hraw = hist_counts(last, 0.0, 256.0)
                hres = (hist_counts(resid, -10.0, 22.0)
                        if master is not None else None)
                self.q.put(("dark_preview", raw8, resid8, stats,
                            cap.last_fps, hraw, hres))
        except Exception as e:
            self.dbg.exc("error", where="dark_preview_worker")
            self.q.put(("log", f"dark preview failed: {e}"))
        finally:
            if cap is not None:
                cap.close()
            self.dbg.log("worker_end", kind="dark_preview")
            self.q.put(("worker_done",))

    def _light_worker(self, dev, w, h, sub, fix, align, flt, fmode):
        # dev/w/h and the checkbox states arrive as plain values snapshotted
        # on the UI thread (on_start) -- no Tk variable access in here
        master = self.master_dark
        flat = self.master_flat
        defects = self.defects
        sub = sub and master is not None
        flt = flt and flat is not None
        fix = fix and defects is not None and len(defects) > 0
        burst = 6
        # rolling integration: ring buffer of the last N processed frames
        # (float32 to halve memory), running float64 sum for O(1) update
        ring = deque()
        ssum = None
        align_ref = None
        pause_ref_stars = None
        was_paused = False
        warned_drift = False   # one-shot UI warning for magnitude rejections
        # static-camera star pipeline: run the detector at an adaptive
        # threshold steered toward the user's "visible stars" count, and
        # let the temporal tracker confirm which candidates persist --
        # noise doesn't survive consecutive bursts, real stars do
        tracker = StarTracker()
        nsigma_cur = 5.0
        self.dbg.log("worker_start", kind="light", device=dev,
                     exposure_units=self.exp_value,
                     resolution=f"{w}x{h}", fourcc=self.fourcc,
                     subtract_dark=sub, repair_defects=fix, align=align,
                     flat_correct=flt, flat_mode=fmode if flt else None,
                     flat_exposure_units=self.flat_exp,
                     window_max=self.stack_max_value,
                     master_exposure_units=self.master_exp,
                     defects=0 if defects is None else len(defects))
        cap = None
        try:
            cap = self.make_capture(dev, w, h)
            cc.set_ctrl(dev, "auto_exposure", 1)
            cc.set_ctrl(dev, "exposure_time_absolute", self.exp_value)
            while not self.stop_evt.is_set():
                cap.exposure = self.exp_value  # live-adjustable
                maxn = self.stack_max_value
                path, _, _ = cap.capture_run(burst, discard=1,
                                             timeout=60.0, verbose=False)
                paused = self.pause_evt.is_set()
                if paused and not was_paused:
                    pause_ref_stars = list(self.stars)  # pre-pause positions
                if was_paused and not paused:
                    pause_ref_stars = None
                    if self.reset_on_resume:
                        ring.clear()
                        ssum = None
                        align_ref = None
                was_paused = paused

                # deep-dive: when armed, this burst is kept in full and
                # dumped with the complete processing context
                dd = self.deepdive_evt.is_set()
                if dd:
                    self.deepdive_evt.clear()
                dd_raw, dd_proc, dd_stats = [], [], []

                last_raw = None
                last_proc = None
                last_shift = (0.0, 0.0)
                last_conf = None
                last_applied = False      # a non-zero shift was actually used
                last_reject = None        # why a computed shift was NOT used
                last_ref_set = align_ref is not None
                for fr in cap.iter_luma(path, burst, discard=1):
                    if self.stop_evt.is_set():
                        break
                    proc = fr
                    if sub:
                        proc = np.clip(fr - master, 0.0, None)
                    if flt:
                        proc = apply_flat(proc, flat, fmode)
                    if fix:
                        proc = repair_defects(proc.copy(), defects)
                    last_raw = fr
                    last_proc = proc
                    if dd:
                        dd_raw.append(fr.copy())
                    if paused:
                        if dd:
                            dd_proc.append(proc.copy())
                            dd_stats.append({"raw": frame_stats(fr),
                                             "proc": frame_stats(proc)})
                        continue          # display only; no integration
                    if align:
                        small, f = downsample_to_fit(proc, ALIGN_THUMB, ALIGN_THUMB)
                        if align_ref is None:
                            align_ref = small.copy()
                            last_ref_set = True
                        else:
                            dy, dx, conf = phase_shift(align_ref, small)
                            last_conf = conf      # logged even when not applied
                            # only act on a clear peak and a plausible drift
                            # (<= 1/4 frame); otherwise a featureless field
                            # yields garbage shifts that smear the stack
                            maxsh = max(small.shape) // 4
                            if (conf >= ALIGN_MIN_CONF
                                    and abs(dy) <= maxsh and abs(dx) <= maxsh):
                                last_reject = None
                                sy, sx = dy * f, dx * f
                                if sy != 0.0 or sx != 0.0:
                                    proc = shift_fill(proc, sy, sx)
                                    last_shift = (sy, sx)
                                    last_applied = True
                            elif conf >= ALIGN_MIN_CONF:
                                # confident peak but implausibly large drift:
                                # cumulative motion has outrun the reference
                                # frame -- from here alignment is inert and
                                # the stack will smear (reset it)
                                last_reject = "magnitude"
                            else:
                                last_reject = "confidence"
                    if dd:
                        # AFTER the align block: dd_proc holds the frame as
                        # integrated, including any alignment shift
                        dd_proc.append(proc.copy())
                        dd_stats.append({"raw": frame_stats(fr),
                                         "proc": frame_stats(proc)})
                    f32 = proc.astype(np.float32)
                    # non-inplace subtract: ssum was published as
                    # self.stack_sum, which the UI thread reads for solves
                    # and saves -- mutating it in place tears that snapshot
                    while len(ring) >= maxn:   # window shrink also handled
                        ssum = ssum - ring.popleft()
                    ring.append(f32)
                    ssum = f32.astype(np.float64) if ssum is None \
                        else ssum + f32
                try:
                    os.remove(path)
                except OSError:
                    pass
                if last_raw is None:
                    continue

                if dd and dd_raw:
                    arrays = {f"raw_{i:02d}": a
                              for i, a in enumerate(dd_raw)}
                    arrays.update({f"proc_{i:02d}": a
                                   for i, a in enumerate(dd_proc)})
                    arrays.update(master_dark=master, master_flat=flat,
                                  defects=defects,
                                  stack=(ssum / len(ring) if ring else None))
                    self._deepdive_write("light", arrays, {
                        "device": dev, "resolution": f"{w}x{h}",
                        "fourcc": self.fourcc,
                        "exposure_units": cap.exposure,
                        "subtract_dark": sub, "flat_correct": flt,
                        "flat_mode": fmode if flt else None,
                        "repair_defects": fix, "align": align,
                        "master_dark_exposure_units": self.master_exp,
                        "flat_exposure_units": self.flat_exp,
                        "roi": self.roi, "paused": paused,
                        "ring": len(ring), "window_max": maxn,
                        "fps": round(cap.last_fps, 2),
                        "align_burst": {"shift": [round(last_shift[0], 3),
                                                  round(last_shift[1], 3)],
                                        "conf": last_conf,
                                        "applied": last_applied,
                                        "reject": last_reject},
                        "per_frame": dd_stats,
                        "stars_prev_burst": [dict(s)
                                             for s in self.stars[:10]]})

                # a confident shift rejected for magnitude means cumulative
                # drift has outrun the alignment reference: say so once,
                # visibly, instead of silently smearing the stack
                if last_reject == "magnitude" and not warned_drift:
                    warned_drift = True
                    self.q.put(("log",
                                "alignment: drift now exceeds the plausible "
                                "limit vs the reference frame -- correction "
                                "is inert and the stack may smear; Reset "
                                "stack to re-anchor"))
                elif last_applied:
                    warned_drift = False

                raw_small, stride = downsample_to_fit(
                    last_raw, self.canvas_w, self.canvas_h)
                raw8 = stretch_to_u8(raw_small)

                if paused:
                    # star-motion feedback: match live-frame detections to
                    # the pre-pause stack positions, draw the vectors
                    cur, _, _ = extract_stars(last_proc, nsigma=4.5,
                                              max_stars=40)
                    pairs = match_stars(pause_ref_stars or [], cur)
                    summary = motion_summary(pairs)
                    self.dbg.log("pause", state="tracking",
                                 detected=len(cur), matched=len(pairs),
                                 motion=(dict(zip(
                                     ("dx", "dy", "mag", "dir_deg"),
                                     summary)) if summary else None))
                    self.q.put(("pause_update", raw8, pairs, stride,
                                summary, cap.last_fps))
                    continue

                if not ring:
                    continue
                stack = ssum / len(ring)
                # publish for solve/save (UI thread reads these between
                # queue messages; assignment is atomic under the GIL)
                self.stack_sum = ssum
                self.stack_n = len(ring)
                # detect only within the known-good ROI when one is set:
                # background/noise come from the clean region and edge junk
                # can't masquerade as stars. Coords are shifted back to
                # full-frame so overlays and the solver line up.
                target = max(1, self.star_target_value)
                cap_n = max(60, target * 3)
                roi = self.roi
                roi_used = None
                if roi is not None:
                    sh, sw = stack.shape
                    rx0 = max(0, min(roi[0], sw))
                    ry0 = max(0, min(roi[1], sh))
                    rx1 = max(0, min(roi[2], sw))
                    ry1 = max(0, min(roi[3], sh))
                    if rx1 - rx0 >= 16 and ry1 - ry0 >= 16:
                        cand, bg, sigma = extract_stars(
                            stack[ry0:ry1, rx0:rx1], nsigma=nsigma_cur,
                            max_stars=cap_n)
                        for s in cand:
                            s["x"] += rx0
                            s["y"] += ry0
                        roi_used = [rx0, ry0, rx1, ry1]
                    else:
                        cand, bg, sigma = extract_stars(
                            stack, nsigma=nsigma_cur, max_stars=cap_n)
                else:
                    cand, bg, sigma = extract_stars(
                        stack, nsigma=nsigma_cur, max_stars=cap_n)
                # steer the threshold toward the declared visible-star
                # count: dig deeper (never below 3.5σ) while short of it,
                # back off when candidates swamp it. The tracker keeps a
                # low threshold honest -- unconfirmed candidates are never
                # reported.
                n_raw = len(cand)
                if n_raw < target and nsigma_cur > 3.5:
                    nsigma_cur = max(3.5, nsigma_cur - 0.3)
                elif n_raw > 3 * target and nsigma_cur < 8.0:
                    nsigma_cur = min(8.0, nsigma_cur + 0.5)
                stars = tracker.update(cand, time.monotonic())
                self.stars = stars
                # corner-vs-centre background of the stack: pointed at an
                # evenly lit field this is the live flat verification --
                # toggle the correction and watch the ratio walk to 1.0
                f_centre, f_corner, f_ratio = field_flatness(stack)
                self.dbg.log(
                    "burst", kind="light", roi=roi_used,
                    exposure_units=cap.exposure,
                    capture_s=round(cap.last_capture_s, 2),
                    fps=round(cap.last_fps, 2),
                    ring=len(ring), window_max=maxn,
                    # align_on distinguishes "toggle off" from a real attempt;
                    # align_applied distinguishes a no-shift result (good conf,
                    # zero drift, OR rejected low conf) from one that moved the
                    # frame. With these three an inert pipeline is unambiguous
                    # in the log: align_on false = off; on+conf+!applied = no
                    # usable shift; applied = corrected.
                    align_on=align,
                    align_ref_set=last_ref_set,
                    align_applied=last_applied,
                    align_reject=last_reject,
                    align_shift=[round(last_shift[0], 2), round(last_shift[1], 2)],
                    align_conf=(round(last_conf, 2)
                                if last_conf is not None else None),
                    raw=frame_stats(last_raw), proc=frame_stats(last_proc),
                    stack=frame_stats(stack),
                    stars=len(stars), stars_raw=n_raw,
                    nsigma=round(nsigma_cur, 2),
                    tracks=len(tracker.tracks),
                    star_target=target, bg=round(bg, 3),
                    sigma=round(sigma, 4),
                    field_centre=round(f_centre, 2),
                    field_corner=round(f_corner, 2),
                    field_corner_ratio=(round(f_ratio, 3)
                                        if f_ratio is not None else None),
                    top_stars=[{k: s[k] for k in
                                ("x", "y", "flux", "fwhm")}
                               for s in stars[:3]])
                self.q.put(("light_update", raw8, stack, stars,
                            len(ring), maxn, cap.last_fps, bg, sigma,
                            hist_counts(last_raw, 0.0, 256.0),
                            hist_counts(stack, 0.0, 256.0),
                            frame_stats(last_proc)["mad_sigma"], f_ratio,
                            n_raw, nsigma_cur))
        except Exception as e:
            self.dbg.exc("error", where="light_worker")
            self.q.put(("log", f"capture failed: {e}"))
        finally:
            if cap is not None:
                cap.close()
            self.dbg.log("worker_end", kind="light")
            self.q.put(("worker_done",))

    # ---------------- UI-thread event pump ----------------

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        # capture bursts take seconds with nothing to show; say so rather
        # than sitting silent
        if self.busy():
            stale = time.monotonic() - self.last_evt_t
            if stale > 2.5:
                self.lbl_status.configure(
                    text=f"{self.status_base}  …capturing "
                         f"({stale:.0f}s since last frame)")
        self.root.after(100, self._poll)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "log":
            self.log(msg[1])
        elif kind == "progress":
            _, count, total, msmall, lsmall = msg
            self.prg["maximum"] = total
            self.prg["value"] = count
            self.set_status(f"ACQUIRING MASTER DARK {count}/{total}")
            self._show(self.canvas_live, stretch_to_u8(lsmall), "live")
            self._show(self.canvas_stack, stretch_to_u8(msmall), "stack")
        elif kind == "dark_preview":
            _, raw8, resid8, stats, fps, hraw, hres = msg
            self.set_status(f"DARK PREVIEW @ {fps:.1f} fps"
                            + ("" if resid8 is not None
                               else " — no master dark yet"))
            if hres is not None:
                self._draw_hist([(hres, "#00d050", -10.0, 22.0,
                                  "residual -10..22 ADU")])
            else:
                self._draw_hist([(hraw, "#7a8aa0", 0.0, 256.0,
                                  "raw 0..255")])
            self._show(self.canvas_live, raw8, "live")
            if resid8 is not None:
                self._show(self.canvas_stack, resid8, "stack")
                mean, sigma, mx = stats
                self.lbl_stack.configure(
                    text=f"residual: mean {mean:+.2f} σ {sigma:.2f} "
                         f"max {mx:.1f} ADU @ {fps:.1f} fps "
                         f"(ideal: all ~0)")
            else:
                self.lbl_stack.configure(
                    text=f"@ {fps:.1f} fps -- no master dark: corrected "
                         "view unavailable (acquire one)")
        elif kind == "dark_done":
            _, mean, coords, exp, count, clip = msg
            clipped, floor_frac, ceil_frac = clip
            if clipped:
                # same refusal cam_characterise applies: a rail-pinned master
                # is a clamped processing path, not a measurement -- writing
                # it would hand PHD2 junk to auto-load at every connect
                where = ("floor (0)" if floor_frac > ceil_frac
                         else "ceiling (255)")
                self.log(f"master dark REJECTED: "
                         f"{max(floor_frac, ceil_frac) * 100:.0f}% of pixels "
                         f"pinned at the {where} -- output is clamped/"
                         "saturated, nothing installed or saved. Adjust "
                         "gamma/brightness/contrast (e.g. raise gamma above "
                         "the black-level clamp) and re-acquire.")
                self.lf_right.configure(text="Dark-corrected residual")
                self.set_status("master dark rejected — output clipped, "
                                "nothing saved", fg="#ff8060")
                return
            self.master_dark = mean
            self.master_exp = exp
            self.defects = coords if len(coords) else None
            self._save_dark(mean, coords, exp, count)
            self._update_dark_label()
            self.log(f"master dark: {count} frames, "
                     f"{len(coords)} hot pixels")
            self.lf_right.configure(text="Dark-corrected residual")
            self.set_status(f"master dark complete: {count} frames, "
                            f"{len(coords)} hot pixels — saved", fg="#80ff80")
        elif kind == "panel_level":
            # the flat worker's illumination servo: setting the Tk variable
            # repaints the panel via its write trace
            self.var_panel_level.set(int(msg[1]))
        elif kind == "flat_wait":
            _, med, cf, guide, lsmall = msg
            self.set_status(f"FLAT PRE-FLIGHT — median {med:.0f} ADU, "
                            f"clip {cf * 100:.1f}%: {guide}", fg="#ffd000")
            if lsmall is not None:
                self._show(self.canvas_live, stretch_to_u8(lsmall), "live")
        elif kind == "flat_progress":
            _, count, total, msmall, lsmall, med, cf = msg
            self.prg["maximum"] = total
            self.prg["value"] = count
            # illumination drifting out of band mid-acquisition is worth
            # shouting about while there is still time to fix it
            note = ""
            if cf > FLAT_SAT_FRAC or med > FLAT_OK_HI:
                note = " — TOO BRIGHT (illumination drifted?)"
            elif med < FLAT_OK_LO:
                note = " — TOO DIM (illumination drifted?)"
            self.set_status(f"ACQUIRING MASTER FLAT {count}/{total} — "
                            f"median {med:.0f} ADU{note}",
                            fg="#ff8060" if note else "#80c0ff")
            self._show(self.canvas_live, stretch_to_u8(lsmall), "live")
            self._show(self.canvas_stack, stretch_to_u8(msmall), "stack")
        elif kind == "flat_done":
            _, gain, reject, info, exp, count = msg
            self._set_mode_banner()  # restore the right-canvas label
            if gain is None:
                # same refusal policy as the dark: a saturated or
                # signal-starved flat is not a gain measurement, and
                # installing it would corrupt every corrected frame
                self.log(f"master flat REJECTED: {reject} -- nothing "
                         "installed or saved")
                self.set_status("master flat rejected — nothing saved",
                                fg="#ff8060")
                return
            if info["median_adu"] < FLAT_LOW_MEDIAN:
                self.log(f"NOTE: flat signal is low (median "
                         f"{info['median_adu']:.0f} ADU) -- the gain map "
                         "will be noisy; brighter illumination or a longer "
                         "exposure would help")
            self.master_flat = gain
            self.flat_exp = exp
            self._save_flat(gain, exp, count, info)
            self._update_flat_label()
            self.log(f"master flat: {count} frames, median "
                     f"{info['median_adu']:.0f} ADU, gain 5-95% "
                     f"[{info['gain_p5']:.3f}..{info['gain_p95']:.3f}]")
            self.set_status(f"master flat complete: {count} frames — saved",
                            fg="#80ff80")
        elif kind == "pause_update":
            _, raw8, pairs, stride, summary, fps = msg
            self.set_status(f"PAUSED — tracking star motion @ {fps:.1f} fps",
                            fg="#ffd000")
            self._show(self.canvas_live, raw8, "live",
                       vectors=pairs, stride=stride)
            if summary:
                mdx, mdy, mag, ang = summary
                horiz = "right" if mdx > 0 else "left"
                vert = "down" if mdy > 0 else "up"
                self.lbl_stars.configure(
                    text=f"PAUSED — field moving: ({mdx:+.1f},{mdy:+.1f})px "
                         f"= {mag:.1f}px toward {ang:.0f}° "
                         f"({horiz}/{vert}), {len(pairs)} stars tracked")
            else:
                self.lbl_stars.configure(
                    text="PAUSED — no matched stars to track "
                         "(too few / moved too far)")
            self.lbl_stack.configure(
                text=f"integration paused @ {fps:.1f} fps "
                     f"(stack frozen at {self.stack_n} frames)")
        elif kind == "light_update":
            (_, raw8, stack, stars, n, maxn, fps, bg, sigma,
             hraw, hstack, sig1, f_ratio, n_raw, nsig) = msg
            self.set_status(f"INTEGRATING {n}/{maxn} @ {fps:.1f} fps — "
                            f"{len(stars)} stars")
            self._show(self.canvas_live, raw8, "live")
            ssmall, stride = downsample_to_fit(stack, self.canvas_w,
                                               self.canvas_h)
            stack8, st_lo, st_hi = stretch_to_u8(ssmall, return_bounds=True)
            self._show(self.canvas_stack, stack8, "stack",
                       stars=stars, stride=stride, cross=self.pole_xy)
            self._draw_hist([(hraw, "#7a8aa0", 0.0, 256.0, "raw"),
                             (hstack, "#00d050", 0.0, 256.0, "stack")])
            if not self.noise_hist or n > self.noise_hist[-1][0]:
                self.noise_hist.append((n, sigma, sig1))
                self._draw_noise()
            # noise-floor readout: measured single-frame->stack noise gain
            # against theoretical sqrt(N) (on the downsampled stack: plenty
            # for a gain estimate)
            b1 = float(np.median(ssmall))
            sN = 1.4826 * float(np.median(np.abs(ssmall - b1)))
            gain = (sigma / sN) if sN > 0 else 0.0
            # stretch readout: the ADU window mapped to black..white. A tiny
            # span means the auto-stretch is amplifying a near-flat frame.
            # corners% is the corner/centre background of the stack -- on an
            # evenly lit field this walking to 100% when 'flat correct' is
            # on is the closed-loop verification the flat works
            flat_note = (f"  ·  corners {f_ratio * 100:.0f}% of centre"
                         if f_ratio is not None else "")
            self.lbl_stack.configure(
                text=f"integrating: {n}/{maxn} frames @ {fps:.1f} fps  ·  "
                     f"stretch {st_lo:.2f}..{st_hi:.2f} ADU "
                     f"(span {st_hi - st_lo:.2f}){flat_note}")
            # detection is already scoped to the ROI in the worker, so this
            # star list is the in-region set -- note that in the readout
            region_note = "  (in region)" if self.roi else ""
            if stars:
                s0 = stars[0]
                self.lbl_stars.configure(
                    text=f"stars: {len(stars)} confirmed / {n_raw} cand "
                         f"@ {nsig:.1f}σ (brightest flux {s0['flux']:.0f}, "
                         f"FWHM {s0['fwhm']}px) "
                         f"bg {bg:.1f} σ {sigma:.2f}{region_note}")
            else:
                self.lbl_stars.configure(
                    text=f"stars: 0 confirmed / {n_raw} cand @ {nsig:.1f}σ"
                         f"  bg {bg:.1f} σ {sigma:.2f}{region_note}")
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
                self.solve_history.append(
                    {"t": time.time(), "ra": info["ra"], "dec": info["dec"],
                     "sep_arcmin": (90.0 - abs(info["dec"])) * 60.0})
                del self.solve_history[:-50]
                self._draw_bullseye()
                self.pole_xy = info.get("pole_xy")
                # WCS pixels are relative to the solved crop -- shift them
                # back to full-frame coords so the crosshair lands correctly
                if self.pole_xy and self._solve_roi:
                    self.pole_xy = [self.pole_xy[0] + self._solve_roi[0],
                                    self.pole_xy[1] + self._solve_roi[1]]
                if self.pole_xy:
                    w, h = self.current_size()
                    px, py = self.pole_xy
                    inside = (w and 0 <= px < w and 0 <= py < h)
                    self.log(f"pole at pixel ({px:.0f},{py:.0f})"
                             + ("" if inside else " (outside the frame; "
                                "crosshair appears when it comes into view)"))
            elif not ok:
                self.lbl_solve.configure(
                    text="sky position: " + text.splitlines()[0],
                    fg="#802000")
            self.btn_solve.configure(state="normal")
        elif kind == "worker_done":
            self._update_run_ui(running=False)
            self._update_mode_ui(running=False)
            self.btn_pause.configure(state="disabled",
                                     text="Pause integration")
            self.pause_evt.clear()
            self.prg["value"] = 0
            self._set_mode_banner()  # restore right-canvas label
            self.set_status("stopped — ready", fg="#80ff80")

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
        self.dbg.log("dark_saved", master=mp, defects=dp, meta=jp,
                     hot_pixels=int(len(coords)), exposure_units=exp)
        self.log(f"saved {mp}, {dp}, {jp}")

    def _save_flat(self, gain, exp, count, info):
        w, h = self.current_size()
        d = self.data_dir()
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, f"flat_{w}x{h}.npy")
        np.save(fp, gain.astype(np.float32))  # median-1 gain map
        jp = os.path.join(d, f"flat_meta_{w}x{h}.json")
        with open(jp, "w") as fh:
            json.dump({"exposure_units": exp, "frames": count,
                       "median_adu": info["median_adu"],
                       "dark_subtracted": info["dark_subtracted"],
                       "gain_p5": info["gain_p5"],
                       "gain_p95": info["gain_p95"],
                       "timestamp_utc":
                           datetime.now(timezone.utc).isoformat()}, fh,
                      indent=2)
        self.dbg.log("flat_saved", flat=fp, meta=jp, exposure_units=exp,
                     frames=count)
        self.log(f"saved {fp}, {jp}")


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
    ap.add_argument("--debug", action="store_true",
                    help="write JSONL diagnostics suitable for LLM "
                         "consumption (one self-describing event per line; "
                         "schema in the first line)")
    ap.add_argument("--debug-file", default=None,
                    help="debug log path (default: "
                         "{data-dir}/camobs_debug_<UTC>.jsonl)")
    args = ap.parse_args()

    dbg = start_debug(args) if (args.debug or args.debug_file) else NullDebug()
    if dbg.enabled:
        print(f"debug log: {dbg.path}")

    root = tk.Tk()
    App(root, args, dbg)
    root.mainloop()
    dbg.log("session_end")
    dbg.close()


if __name__ == "__main__":
    main()
