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

It captures via V4L2 (uncompressed YUYV/NV12 only -- compressed formats are
refused for measurement because the codec destroys exactly the noise you are
trying to quantify). Capture uses v4l2 mmap streaming through python's fcntl;
no OpenCV required.

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
import fcntl
import hashlib
import json
import mmap
import os
import re
import select
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

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

def _IOR(t, nr, structtype):  return _IOC(_IOC_READ, t, nr, ctypes.sizeof(structtype))
def _IOW(t, nr, structtype):  return _IOC(_IOC_WRITE, t, nr, ctypes.sizeof(structtype))
def _IOWR(t, nr, structtype): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, ctypes.sizeof(structtype))

# Request codes are assigned AFTER the structs are defined (see below).

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP            = 1
V4L2_FIELD_NONE             = 1

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

class v4l2_format_pix(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32), ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32), ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32), ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32), ("xfer_func", ctypes.c_uint32),
    ]

class v4l2_format(ctypes.Structure):
    # Kernel struct is 208 bytes on 32-bit: __u32 type + 4 pad + 200-byte union.
    class _u(ctypes.Union):
        _fields_ = [("pix", v4l2_format_pix), ("raw", ctypes.c_ubyte * 200)]
    _fields_ = [("type", ctypes.c_uint32), ("_pad", ctypes.c_uint32), ("fmt", _u)]
    _pack_ = 1  # avoid surprises; we add the pad explicitly

class v4l2_requestbuffers(ctypes.Structure):
    # 4.9 kernel layout: count,type,memory,reserved[2] = 20 bytes.
    # (capabilities/flags are 5.x additions; including them breaks the _IOC size.)
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 2)]

class v4l2_timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_int32), ("tv_usec", ctypes.c_int32)]

class v4l2_timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]

class v4l2_buffer(ctypes.Structure):
    class _u(ctypes.Union):
        # Fixed width: all 4 bytes on this 32-bit kernel. c_void_p/c_ulong
        # would be 8 bytes under a 64-bit Python and break the struct size.
        _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_uint32),
                    ("planes", ctypes.c_uint32), ("fd", ctypes.c_int32)]
    _fields_ = [
        ("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32), ("timestamp", v4l2_timeval),
        ("timecode", v4l2_timecode), ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32), ("m", _u),
        ("length", ctypes.c_uint32), ("reserved2", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]

class v4l2_fmtdesc(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("description", ctypes.c_char * 32),
                ("pixelformat", ctypes.c_uint32), ("mbus_code", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 3)]


# Now that every struct is defined, compute the request codes so the encoded
# size always matches sizeof(struct) on THIS platform/kernel.
VIDIOC_ENUM_FMT  = _IOWR('V', 2,  v4l2_fmtdesc)
VIDIOC_G_FMT     = _IOWR('V', 4,  v4l2_format)
VIDIOC_S_FMT     = _IOWR('V', 5,  v4l2_format)
VIDIOC_REQBUFS   = _IOWR('V', 8,  v4l2_requestbuffers)
VIDIOC_QUERYBUF  = _IOWR('V', 9,  v4l2_buffer)
VIDIOC_QBUF      = _IOWR('V', 15, v4l2_buffer)
VIDIOC_DQBUF     = _IOWR('V', 17, v4l2_buffer)
VIDIOC_STREAMON  = _IOW('V', 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW('V', 19, ctypes.c_int)


def xioctl(fd, req, arg):
    """Issue an ioctl, operating in place for ctypes structs.

    fcntl.ioctl with a ctypes Structure and the default path can pass a
    non-writable / copied buffer; QUERYBUF and DQBUF need the kernel to write
    fields (m.offset, length, bytesused...) back into OUR struct. For ctypes
    args we wrap the struct's own memory in a writable buffer and pass it with
    mutate_flag=True so the kernel's writes land in `arg` directly. For plain
    bytes (e.g. STREAMON's packed int) we pass through unchanged.
    """
    if isinstance(arg, (bytes, bytearray)):
        while True:
            try:
                return fcntl.ioctl(fd, req, arg)
            except OSError as e:
                if e.errno == 4:
                    continue
                raise
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

def best_measurement_format(ranked):
    for f in ranked:
        if f["fidelity_rank"] <= 2:  # raw or uncompressed YUV
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
            means, varis = [], []
            for k in range(pairs_per_level):
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


def hot_pixel_mask(master, nsigma=6.0):
    """Single-pixel outliers above a locally-smoothed master.

    Thresholding the RAW master at mean+6*std fails at native resolution: the
    std includes the shading/FPN structure (the LSC gradient), which inflates
    the threshold and hides genuine hot pixels -- or, with a tilted pedestal,
    flags whole corners. So: remove structure with a separable median-of-3
    smooth (kills single-pixel spikes, follows gradients), then threshold the
    residual against a robust MAD sigma (so the hot pixels themselves cannot
    inflate the noise estimate they are tested against)."""
    m = master.astype(np.float32, copy=False)
    pl = np.pad(m, ((0, 0), (1, 1)), mode="edge")
    hsm = _med3(pl[:, :-2], pl[:, 1:-1], pl[:, 2:])
    pv = np.pad(hsm, ((1, 1), (0, 0)), mode="edge")
    smooth = _med3(pv[:-2, :], pv[1:-1, :], pv[2:, :])
    resid = m - smooth
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma = 1.4826 * mad
    if sigma <= 0:  # exactly flat residual (synthetic/clamped data)
        sigma = float(resid.std()) or 1e-6
    return resid > (med + nsigma * sigma)


def build_fps_calibration(ladder_data):
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
    return {"fps_max_bandwidth": round(fps_max, 2),
            "integration_limited_below_fps": round(thresh, 2),
            "anchors_fps": [round(f, 3) for f, _ in anchors],
            "anchors_exposure": [e for _, e in anchors],
            "valid_fps_range": [round(anchors[0][0], 2), round(anchors[-1][0], 2)]}


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


def run_dark(cap, device, dark_frames, exposure_max, slope_points,
             ladder=None, discard=2, knee=2048, ladder_frames=16,
             ladder_points=7, verbose=True):
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
    knee:          exposure above which real integration begins.
    """
    print("\n=== DARK NOISE CHARACTERISATION ===")
    print("CAP THE LENS / full darkness. Keep temperature stable across the run.")
    print(f"Deep stack: {dark_frames} frames at exposure={exposure_max} (max).")
    print(f"Ladder points: {ladder_frames} frames each. "
          f"Discarding first {discard} frame(s) per exposure (settle).\n")

    if ladder is None:
        # Two low points for the bias intercept, then ladder_points evenly
        # spaced across the integrating region (knee -> exposure_max) so the
        # dark-current slope is well-determined.
        lo = [max(1, int(exposure_max * f)) for f in (0.05, 0.15)]
        start = max(knee, int(exposure_max * 0.30))
        n = max(2, ladder_points)
        hi = [int(start + (exposure_max - start) * i / (n - 1))
              for i in range(n)]
        ladder = sorted(set(lo + hi))
    else:
        ladder = sorted(set(int(x) for x in ladder))
    if exposure_max not in ladder:
        ladder.append(exposure_max)
        ladder = sorted(set(ladder))

    cap.start()
    started = datetime.now(timezone.utc).isoformat()
    slope_data = []
    master = None
    residual_std = None
    deep_drops = 0
    deep_clipped = False
    radial = None
    fpn = hot = None
    hot_coords = None

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
            nframes = dark_frames if is_deep else max(4, ladder_frames)
            path, total, full = cap.capture_run(
                nframes, discard=discard,
                timeout=max(20.0, exp * 1e-4 * 4 + 10))

            # Streaming Welford over frames: per-pixel running mean + M2.
            # Holds only mean and M2 (one frame each) -> ~15MB, not GBs.
            count = 0
            mean_px = None   # per-pixel running mean (the master dark / FPN)
            m2_px = None     # per-pixel sum of squared deviations
            for fr in cap.iter_luma(path, nframes, discard=discard):
                count += 1
                if mean_px is None:
                    mean_px = np.zeros_like(fr)
                    m2_px = np.zeros_like(fr)
                delta = fr - mean_px
                mean_px += delta / count
                m2_px += delta * (fr - mean_px)
                del fr, delta
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
                temporal = float(np.sqrt(var_px).mean())
            else:
                temporal = float("nan")

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

            slope_data.append({"exposure_units": exp, "exposure_readback": rb,
                               "mean_adu": mean_level,
                               "temporal_noise_adu": temporal,
                               "frames": count, "drops": drops,
                               "capture_s": round(cap.last_capture_s, 3),
                               "eff_fps": round(cap.last_fps, 2),
                               "clipped": clipped,
                               "clip_floor_frac": round(floor_frac, 4),
                               "clip_ceil_frac": round(ceil_frac, 4)})
            if verbose:
                print(f"   exp={exp:5d} (rb={rb})  mean={mean_level:7.3f} ADU  "
                      f"temporal_noise={temporal:6.3f} ADU  "
                      f"{cap.last_fps:5.1f}fps  "
                      f"n={count}" + (f"  drops={drops}" if drops else "")
                      + clip_note)
            if is_deep:
                deep_drops = drops
                deep_clipped = clipped
                master = mean_px                 # FIXED-PATTERN part (per-pixel)
                # irreducible temporal noise after master-dark subtraction is
                # exactly the temporal std we already computed per-pixel:
                residual_std = temporal
                fpn = float(master.std())        # spatial FPN spread
                _hot_mask = hot_pixel_mask(master)
                hot = int(_hot_mask.sum())
                _yx = np.argwhere(_hot_mask)
                hot_coords = _yx.tolist()        # [[y, x], ...] numpy row-major convention
                if clipped:
                    radial = {"shape": "N/A - output clipped", "clipped": True}
                    if verbose:
                        print("   radial: skipped (deep stack CLIPPED -- "
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
        # Verdict: evaluate ONLY the integrating region (exposures above the
        # frame-period knee). Below the knee, exposure is clamped within a frame
        # period so fps cannot drop and mean cannot rise -- including those
        # points falsely makes any real integration look "flat/synthetic".
        integ = [d for d in slope_data if d["exposure_units"] >= knee
                 and d["eff_fps"] > 0]
        if len(integ) >= 2:
            f0, f1 = integ[0]["eff_fps"], integ[-1]["eff_fps"]
            e0, e1 = integ[0]["exposure_units"], integ[-1]["exposure_units"]
            fps_drop = f0 / f1 if f1 else float("nan")
            expo_rise = e1 / e0 if e0 else float("nan")
            expfid["integrating_region"] = {
                "exposure_from": e0, "exposure_to": e1,
                "fps_drop": round(fps_drop, 2),
                "expo_rise": round(expo_rise, 2)}
            # Real integration: fps falls roughly in proportion to exposure once
            # past the knee. Synthetic: fps stays flat while exposure climbs.
            if fps_drop >= 0.6 * expo_rise:
                expfid["verdict"] = (
                    f"in integrating region fps fell {fps_drop:.1f}x as exposure "
                    f"rose {expo_rise:.1f}x -> REAL integration")
            elif fps_drop < 1.3 and expo_rise > 2:
                expfid["verdict"] = (
                    f"in integrating region fps held ~flat while exposure rose "
                    f"{expo_rise:.1f}x -> synthetic exposure (gain/sum)")
            else:
                expfid["verdict"] = (
                    f"partial: fps fell {fps_drop:.1f}x vs exposure {expo_rise:.1f}x "
                    "-> integration real but sub-proportional (clamped/quantised)")
            if verbose:
                print(f"   -> {expfid['verdict']}")
        elif verbose:
            print("   -> too few integrating-region points for a verdict; "
                  "add exposures above the knee")

    # dark current slope: fit ONLY the integrating region (exposure >= knee).
    # Below the knee the level is the clamped pedestal and carries no dark-
    # current information; including it corrupts the slope (the r2=0.82 problem).
    integ_pts = [d for d in slope_data if d["exposure_units"] >= knee]
    if len(integ_pts) >= 2:
        xs = np.array([d["exposure_units"] for d in integ_pts], float)
        ys = np.array([d["mean_adu"] for d in integ_pts], float)
        fit_region = f"integrating (>= {knee})"
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
        if not linfit["linear"]:
            print(f"   !! note: dark level vs exposure fit is weak (r2={r2:.3f}, "
                  f"{len(xs)} pts, {fit_region}). Either too few integrating-region "
                  "points to fit a slope, or thermal drift across the run. The "
                  "exposure-fidelity verdict above (from fps vs exposure) is the "
                  "reliable real-vs-synthetic indicator, not this fit.")

    result = {
        "exposure_max_units": exposure_max,
        "exposure_max_ms": round(exposure_max * 0.1, 1),
        "pixel_depth": 8,
        "deep_stack_frames": dark_frames,
        "deep_stack_drops": deep_drops,
        "deep_stack_clipped": deep_clipped,
        "master_dark_fpn_adu": fpn if master is not None else None,
        "hot_pixels_6sigma": hot if master is not None else None,
        "hot_pixel_coords": hot_coords,  # [[y, x], ...] — x y when writing PHD2 defect file
        "irreducible_temporal_dark_noise_adu": residual_std,
        "radial_profile": radial if master is not None else None,
        "dark_current_fit": linfit,
        "exposure_fidelity": expfid,
        "ladder": slope_data,
        "stack_started_utc": started,
        "stack_finished_utc": finished,
    }
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
            m = re.search(r"\[\d+\]:\s+'(\w+)'", line)
            if m:
                cur_fourcc = m.group(1); cur_w = cur_h = None; continue
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
        modes_str = '|'.join(
            f"{m['fourcc']}:{m['width']}x{m['height']}@{m['max_fps']}"
            for m in sorted(adv, key=lambda m: (m['fourcc'], m['width'], m['height']))
        )
        cap_hash = hashlib.sha1(
            f"{modes_str};{_ctrl_capabilities_str(device)}".encode()
        ).hexdigest()[:12]

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


def main():
    ap = argparse.ArgumentParser(description="UVC camera noise/codec characteriser")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--name", default=None,
                    help="human-readable camera name; saved against the device tag "
                    "and auto-loaded on future runs for the same device")
    ap.add_argument("--list", action="store_true",
                    help="enumerate + rank formats and pick measurement format")
    ap.add_argument("--ptc", action="store_true", help="run photon transfer curve")
    ap.add_argument("--dark", action="store_true", help="run dark characterisation")
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
    ap.add_argument("--knee", type=int, default=2048,
                    help="exposure (raw units) above which real integration "
                    "begins; points below only set the bias and are excluded "
                    "from the dark-current slope fit. Default 2048.")
    ap.add_argument("--save-master", default=None,
                    help="write master dark as uint16 .npy (scaled 0-65535 for PHD2)")
    ap.add_argument("--save-defects", default=None,
                    help="write hot pixel coordinates in PHD2 defect map format (x y per line)")
    ap.add_argument("--save-dark-model", default=None,
                    help="write dark current model JSON for PHD2 auto-import at camera connect")
    ap.add_argument("--report", default=None, help="write json report here")
    ap.add_argument("--report-dir", default=None,
                    help="directory to auto-write a timestamped JSON report into "
                    "(filename encodes mode/resolution/timestamp)")
    args = ap.parse_args()

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
            mode = ("auto" if args.auto else "ptc" if args.ptc
                    else "dark" if args.dark else "list")
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            tag_part = f"{_dev_tag}_" if _dev_tag else ""
            fn = f"camchar_{tag_part}{mode}_{args.width}x{args.height}_{ts}.json"
            path = os.path.join(args.report_dir, fn)
            json.dump(report, open(path, "w"), indent=2)
            print(f"\nreport written to {path}")

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

    if args.list and not (args.ptc or args.dark or args.auto):
        _write_report()
        return

    # --- auto-exposure observation mode (do NOT force manual) ---
    if args.auto:
        print("\n=== Auto-exposure mode: leaving AE ENABLED (not forcing manual) ===")
        calib = None
        if args.calib:
            try:
                calib = json.load(open(args.calib))
                print(f"   loaded fps<->exposure calibration from {args.calib} "
                      f"(valid fps {calib.get('valid_fps_range')})")
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
    report["manual_notes"] = _manual_notes

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

    # --- dark ---
    if args.dark:
        cap = Capture(args.device, args.width, args.height, meas_fmt)
        ladder = None
        if args.ladder:
            ladder = [int(x) for x in args.ladder.replace(" ", "").split(",") if x]
        try:
            dark, master = run_dark(cap, args.device,
                                    dark_frames=args.dark_frames,
                                    exposure_max=args.exposure_max,
                                    slope_points=args.slope_points,
                                    ladder=ladder,
                                    discard=args.discard,
                                    knee=args.knee,
                                    ladder_frames=args.ladder_frames,
                                    ladder_points=args.ladder_points)
        finally:
            cap.close()
        report["dark"] = dark
        # A clipped deep stack is not a measurement: the master, hot-pixel list
        # and dark-current fit are all meaningless, and writing them anyway
        # means PHD2 silently auto-loads garbage at every camera connect.
        deep_clipped = bool(dark.get("deep_stack_clipped"))
        if deep_clipped and (args.save_master or args.save_defects
                             or args.save_dark_model):
            print("   !! deep stack CLIPPED -- refusing to write master/defects/"
                  "dark-model (fix the processing path: reduce brightness/"
                  "contrast/gamma, then rerun)")
        if args.save_master and master is not None and not deep_clipped:
            # Scale float64 luma (0-255) to uint16 (0-65535) for direct PHD2 import.
            # 257 * 255 = 65535 exactly; np.round preserves sub-ADU precision.
            master_u16 = np.clip(np.round(master * 257.0), 0, 65535).astype(np.uint16)
            _ensure_dir(args.save_master)
            np.save(args.save_master, master_u16)
            print(f"   master dark (uint16) written to {args.save_master}")
        if args.save_defects and master is not None and not deep_clipped:
            coords = dark.get("hot_pixel_coords") or []
            _ensure_dir(args.save_defects)
            with open(args.save_defects, "w") as fh:
                fh.write("# PHD2 Defect Map v1\n")
                fh.write(f"# cam_characterise.py {args.device} "
                         f"{args.width}x{args.height}\n")
                fh.write(f"# Exposure: {args.exposure_max} units "
                         f"({dark['exposure_max_ms']}ms)\n")
                fh.write(f"# Defect count: {len(coords)}\n")
                for y, x in coords:  # PHD2 format is x y (screen coords)
                    fh.write(f"{x} {y}\n")
            print(f"   defect map written to {args.save_defects} "
                  f"({len(coords)} hot pixels)")
        if args.save_dark_model and master is not None and not deep_clipped:
            dcf = dark.get("dark_current_fit") or {}
            model = {
                "exposure_max_units": args.exposure_max,
                "exposure_max_ms": dark["exposure_max_ms"],
                "pixel_depth": 8,
                "dark_current_fit": dcf,
                "exposure_fidelity_verdict": (
                    dark.get("exposure_fidelity", {}).get("verdict", "")),
            }
            _ensure_dir(args.save_dark_model)
            with open(args.save_dark_model, "w") as fh:
                json.dump(model, fh, indent=2)
            print(f"   dark current model written to {args.save_dark_model}")
        if args.save_calib:
            calib = build_fps_calibration(dark.get("ladder", []))
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
        print(f"  fixed-pattern (subtractable) FPN : "
              f"{dark['master_dark_fpn_adu']:.3f} ADU")
        print(f"  hot pixels (>6sigma)             : {dark['hot_pixels_6sigma']}")
        print(f"  IRREDUCIBLE temporal dark noise  : "
              f"{dark['irreducible_temporal_dark_noise_adu']:.3f} ADU")
        dcf = dark["dark_current_fit"]
        if dcf:
            print(f"  dark current                     : "
                  f"{dcf['dark_current_adu_per_unit']:.5f} ADU/unit "
                  f"(linear: {dcf['linear']}, r2={dcf['r2']:.3f})")

        # combined floor at the operating point, if we also have gain
        if report.get("ptc", {}).get("fit", {}).get("ok"):
            g = report["ptc"]["fit"]["gain_e_per_adu"]
            rn_e = report["ptc"]["fit"]["read_noise_e"]
            dn_e = dark["irreducible_temporal_dark_noise_adu"] * g
            floor = float(np.sqrt(rn_e**2 + dn_e**2))
            report["operating_point_floor_e"] = floor
            print(f"\n  COMBINED single-frame floor @ max exposure : "
                  f"{floor:.2f} e-  (read {rn_e:.2f} + dark-shot {dn_e:.2f} in "
                  "quadrature)")

    _write_report()


if __name__ == "__main__":
    main()
