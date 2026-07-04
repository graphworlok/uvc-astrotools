#!/usr/bin/env python3
"""
cam_manager.py -- capability-aware run planner for cam_characterise.py

Queries a UVC/V4L2 camera's real formats, resolutions, frame intervals and
control ranges, then generates a tailored set of characterisation runs:
  - reads the actual exposure_time_absolute min/max (no hard-coded ceiling)
  - picks resolutions and predicts which one best exposes integration (the
    resolution whose advertised frame rate is high enough that fps will fall
    with exposure rather than being pinned by bandwidth)
  - builds exposure ladders spanning each device's real range
  - emits dark, gamma-sweep, calibration and auto (AE) runs
  - prints the commands, optionally writes them to a script, optionally runs them

This tool does NOT itself touch the camera's pixel path beyond v4l2-ctl
introspection; it plans work for cam_characterise.py, which does the captures.

Nothing here pokes extension units or any control that could wedge a fragile
bridge -- it only reads standard descriptors and standard control ranges.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone


# ----------------------------------------------------------------------------
# Introspection via v4l2-ctl (standalone; does not import the main tool)
# ----------------------------------------------------------------------------

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


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=15).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ! {' '.join(cmd)} failed: {e}", file=sys.stderr)
        return ""


def sanitize_tag_component(s, maxlen=32):
    s = re.sub(r'[^A-Za-z0-9._-]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s[:maxlen]


# --- capability hash: the three functions below MUST stay byte-identical to
# cam_characterise.py's copies (_ctrl_capabilities_str, _enum_advertised_modes,
# device_capability_hash). If they drift, the two tools build different device
# tags and their output paths (master/defects vs report filenames) won't line
# up. Kept as a duplicate rather than an import so cam_manager stays standalone
# (no numpy/fcntl dependency just to plan a run).

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
    if adv is None:
        adv = _enum_advertised_modes(device)
    modes_str = '|'.join(
        f"{m['fourcc']}:{m['width']}x{m['height']}@{m['max_fps']}"
        for m in sorted(adv, key=lambda m: (m['fourcc'], m['width'], m['height']))
    )
    return hashlib.sha1(
        f"{modes_str};{_ctrl_capabilities_str(device)}".encode()
    ).hexdigest()[:12]


def usb_device_tag(device):
    """Return a stable identifier string for a /dev/videoN device:
    vid+pid_serial_caphash (or vid+pid_NOSERIAL-<hash>_caphash when serial is
    absent). Reads from sysfs + v4l2-ctl. Returns '' on failure so callers can
    fall back to resolution-only naming. Must match cam_characterise.py's
    build_device_tag exactly."""
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

        vid, pid = _rd("idVendor"), _rd("idProduct")
        if not vid or not pid:
            return ""
        serial = _rd("serial")
        if serial:
            suffix = sanitize_tag_component(serial)
        else:
            mfr = _rd("manufacturer") or ""
            prd = _rd("product") or ""
            bcd = _rd("bcdDevice") or ""
            h = hashlib.sha1((mfr + prd + bcd).encode()).hexdigest()[:8]
            suffix = f"NOSERIAL-{h}"
        caphash = device_capability_hash(device)
        if caphash:
            suffix = f"{suffix}_{caphash}"
        return f"{vid}{pid}_{suffix}"
    except Exception:
        return ""


def query_formats(device):
    """Parse `v4l2-ctl --list-formats-ext` into a structured list:
    [{fourcc, compressed, sizes:[{w,h,fps:[...]}, ...]}, ...]."""
    txt = _run(["v4l2-ctl", "-d", device, "--list-formats-ext"])
    formats = []
    cur = None
    size = None
    for line in txt.splitlines():
        # '([^']+)' not '(\w+)': fourccs may be space-padded ('Y16 ')
        m = re.search(r"\[\d+\]:\s+'([^']+)'\s+\(([^)]*)\)", line)
        if m:
            cur = {"fourcc": m.group(1).strip(),
                   "description": m.group(2),
                   "compressed": "compressed" in m.group(2).lower(),
                   "sizes": []}
            formats.append(cur)
            continue
        m = re.search(r"Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if m and cur is not None:
            size = {"w": int(m.group(1)), "h": int(m.group(2)), "fps": []}
            cur["sizes"].append(size)
            continue
        m = re.search(r"Interval:\s+Discrete\s+[\d.]+s\s+\(([\d.]+)\s*fps\)", line)
        if m and size is not None:
            size["fps"].append(float(m.group(1)))
    return formats


def query_controls(device):
    """Parse `v4l2-ctl --list-ctrls` into {name: {min,max,step,default,value}}."""
    txt = _run(["v4l2-ctl", "-d", device, "--list-ctrls"])
    ctrls = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\w+)\s+0x[0-9a-f]+\s+\((\w+)\)\s*:\s*(.*)", line)
        if not m:
            continue
        name, ctype, rest = m.group(1), m.group(2), m.group(3)
        d = {"type": ctype}
        for k in ("min", "max", "step", "default", "value"):
            mm = re.search(rf"{k}=(-?\d+)", rest)
            if mm:
                d[k] = int(mm.group(1))
        ctrls[name] = d
    return ctrls


def identify(device):
    """Best-effort device identity from v4l2-ctl --info."""
    txt = _run(["v4l2-ctl", "-d", device, "--info"])
    info = {}
    for key, pat in (("card", r"Card type\s*:\s*(.+)"),
                     ("driver", r"Driver name\s*:\s*(.+)"),
                     ("bus", r"Bus info\s*:\s*(.+)")):
        m = re.search(pat, txt)
        if m:
            info[key] = m.group(1).strip()
    return info


# ----------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------

def best_uncompressed(formats):
    """Pick the measurement format. Preference order mirrors
    cam_characterise.py's FORMAT_FIDELITY ranking (true mono beats YUV: no
    chroma path at all), so the planner chooses the same format the engine's
    auto-pick would -- the resolutions planned here are for the format
    actually measured."""
    pref = ("GREY", "Y800", "Y8", "Y16",
            "YUYV", "YUY2", "UYVY", "NV12", "NV21", "I420")
    unc = [f for f in formats if not f["compressed"]]
    for p in pref:
        for f in unc:
            if f["fourcc"] == p:
                return f
    return unc[0] if unc else (formats[0] if formats else None)


def ladder_for(exp_min, exp_max, n=12):
    """Even-ish ladder from just above min to max, integer, unique, sorted."""
    lo = max(exp_min, 1)
    pts = {lo}
    for i in range(1, n + 1):
        pts.add(int(round(lo + (exp_max - lo) * i / n)))
    pts.add(exp_max)
    return sorted(p for p in pts if exp_min <= p <= exp_max)


def native_factor_sizes(sizes):
    """Keep only resolutions that are an exact integer-factor decimation of the
    native (largest) mode: native_w/w == native_h/h, both integer. Those are
    true NxN bins/crops of the sensor array, so their defect map and FPN reflect
    real sensor pixels. Non-integer scales (e.g. 1280x720 from 1920x1080 = 1.5x,
    or any aspect-ratio change) are bridge interpolation and produce artefact
    defect maps. The native mode itself (factor 1) is always kept."""
    if not sizes:
        return []
    native = max(sizes, key=lambda s: s["w"] * s["h"])
    nw, nh = native["w"], native["h"]
    keep = []
    for s in sizes:
        if s["w"] <= 0 or s["h"] <= 0:
            continue
        if (nw % s["w"] == 0 and nh % s["h"] == 0
                and nw // s["w"] == nh // s["h"]):
            keep.append(s)
    return keep


def pick_resolutions(fmt, native_only=True):
    """Return (sizes, smallest, largest) discrete sizes for the chosen format,
    sorted ascending by pixel count. By default restrict to integer-factor
    decimations of the native (largest) mode -- the only resolutions whose
    defect/FPN data reflects real sensor pixels rather than bridge interpolation.
    native_only=False returns every advertised discrete size."""
    if not fmt or not fmt["sizes"]:
        return [], None, None
    sizes = sorted(fmt["sizes"], key=lambda s: s["w"] * s["h"])
    if native_only:
        clean = native_factor_sizes(sizes)
        if clean:
            sizes = sorted(clean, key=lambda s: s["w"] * s["h"])
    return sizes, sizes[0], sizes[-1]


def max_advertised_fps(size):
    return max(size["fps"]) if size and size.get("fps") else None


def plan(device, formats, ctrls, args):
    """Build the run plan as a list of dicts: {label, why, argv_tail}."""
    runs = []
    fmt = best_uncompressed(formats)
    if not fmt:
        return runs, None, None
    sizes, small, large = pick_resolutions(
        fmt, native_only=not getattr(args, "all_resolutions", False))

    exp = ctrls.get("exposure_time_absolute", {})
    exp_min = exp.get("min", 1)
    exp_max = exp.get("max", 2047)
    knee = max(exp_min + 1, int(exp_max * 0.12))   # generic guess; sweep refines

    has_gamma = "gamma" in ctrls
    g = ctrls.get("gamma", {})
    g_lo, g_hi = g.get("min", 100), g.get("max", 300)
    g_mid = (g_lo + g_hi) // 2
    # Gamma for the per-resolution dark runs. Default g_lo (min == linear), the
    # right choice for a well-behaved sensor. But some bridges (Sunplus
    # 1bcf:28c4) sit in a black-level CLAMP at min gamma: the deep stack comes
    # back dead-flat (mean ~4 ADU, zero temporal noise, zero FPN) and every
    # master/defect/dark-model written is junk. On those, raise --neutral-gamma
    # above the clamp (the clamp-sweep finds where it releases).
    neutral_gamma = (g_lo if getattr(args, "neutral_gamma", None) is None
                     else max(g_lo, min(g_hi, args.neutral_gamma)))

    # Device tag: vid+pid[_serial] from sysfs, used as output subdirectory so
    # files from different cameras never collide even at the same resolution.
    dev_tag = usb_device_tag(device)
    p = (dev_tag + "/") if dev_tag else ""  # file path prefix

    def neutral_proc():
        t = []
        if has_gamma:
            t += ["--gamma", str(neutral_gamma)]
        for c in ("brightness", "contrast"):
            if c in ctrls:
                t += [f"--{c}", "0"]
        if "sharpness" in ctrls:
            t += ["--sharpness", str(ctrls["sharpness"].get("min", 1))]
        return t

    # Pin the engine to the planner's format choice: without --format the
    # engine auto-picks its own best format, and the resolutions/ladders
    # planned here may not be native modes of what it actually measures.
    fmt_args = ["--format", fmt["fourcc"]]

    common = fmt_args + ["--width", str(small["w"]), "--height",
              str(small["h"]),
              "--discard", str(args.discard), "--ladder-frames",
              str(args.ladder_frames), "--exposure-max", str(exp_max),
              "--knee", str(knee)]
    lad = ["--ladder", ",".join(str(x) for x in ladder_for(exp_min, exp_max))]

    sw, sh = small["w"], small["h"]

    # 1..N: a full dark run at EVERY discrete resolution, so PHD2 finds
    # matching master/defects/dark-model for whatever frame size the user
    # guides at. Calibration is saved everywhere too: where the bandwidth
    # ceiling pins fps it simply fails to build (engine prints a note) and
    # PHD2 falls back gracefully — zero cost, full coverage when it works.
    for size in sizes:
        w, h = size["w"], size["h"]
        com = fmt_args + ["--width", str(w), "--height", str(h),
               "--discard", str(args.discard), "--ladder-frames",
               str(args.ladder_frames), "--exposure-max", str(exp_max),
               "--knee", str(knee)]
        if size is small:
            why = (f"clean dark + fps<->exposure calibration at {w}x{h} "
                   f"(highest bandwidth ceiling, best chance fps tracks "
                   f"integration). exposure range {exp_min}..{exp_max}.")
        elif size is large:
            why = (f"native {w}x{h} dark: true hot-pixel count and shading "
                   f"map (lower-res modes bin/scale and hide both).")
        else:
            why = (f"{w}x{h} dark: master/defects/dark-model so PHD2 has "
                   f"matching artefacts if guiding at this frame size.")
        runs.append({
            "label": f"dark_{w}x{h}",
            "why": why,
            "argv_tail": ["--dark"] + com + lad + neutral_proc()
                         + ["--save-calib",      f"{p}calib_{w}x{h}.json",
                            "--save-master",     f"{p}master_{w}x{h}.npy",
                            "--save-defects",    f"{p}defects_{w}x{h}.txt",
                            "--save-dark-model", f"{p}dark_model_{w}x{h}.json"]})

    # 3. gamma sweep (LSC-vs-gamma pipeline ordering, reverse-vignette depth)
    if has_gamma:
        for gv in (g_lo, g_mid, g_hi):
            tail = ["--dark"] + common + ["--ladder", str(exp_max)]
            tail += ["--gamma", str(gv)]
            for c in ("brightness", "contrast"):
                if c in ctrls:
                    tail += [f"--{c}", "0"]
            if "sharpness" in ctrls:
                tail += ["--sharpness", str(ctrls["sharpness"].get("min", 1))]
            tail += ["--save-master", f"{p}master_g{gv}.npy"]
            runs.append({
                "label": f"gamma_{gv}",
                "why": f"gamma={gv} at max exposure, all else neutral: isolates "
                       "where gamma sits in the pipeline vs the shading map "
                       "(watch corner/centre ratio).",
                "argv_tail": tail})

    # 3b. clamp sweep (--clamp-sweep): the FULL exposure ladder at each of
    # several gamma steps. Single-exposure gamma runs above cannot tell where
    # an ISP black-level clamp releases or whether exposure control is real;
    # this maps both jointly. Per gamma watch: does ladder mean ADU rise with
    # exposure (clamp released) or stay flat (clamped), and does the
    # exposure-fidelity verdict flip from synthetic to real.
    if has_gamma and getattr(args, "clamp_sweep", False):
        try:
            gammas = [int(x) for x in args.clamp_gammas.split(",") if x.strip()]
        except ValueError:
            gammas = [g_lo, g_mid, g_hi]
        gammas = sorted(set(max(g_lo, min(g_hi, gv)) for gv in gammas))
        for gv in gammas:
            tail = ["--dark"] + common + lad + ["--gamma", str(gv)]
            for c in ("brightness", "contrast"):
                if c in ctrls:
                    tail += [f"--{c}", "0"]
            if "sharpness" in ctrls:
                tail += ["--sharpness", str(ctrls["sharpness"].get("min", 1))]
            tail += ["--save-master", f"{p}master_clamp_g{gv}.npy"]
            runs.append({
                "label": f"clamp_g{gv}",
                "why": f"gamma={gv}, FULL exposure ladder {exp_min}..{exp_max}: "
                       "maps where the black-level clamp releases (mean ADU "
                       "starts rising with exposure) and whether exposure "
                       "becomes real (fps falls with exposure) vs synthetic.",
                "argv_tail": tail})

    # 4. AE reference at small res (uses the calibration from run 1)
    runs.append({
        "label": "auto_ae_reference",
        "why": "auto-exposure in the dark: where AE parks exposure, stability, "
               "and whether exposure_time_absolute readback agrees with the "
               "framerate-implied exposure (control-honesty check).",
        "argv_tail": ["--auto"] + fmt_args
                     + ["--width", str(sw), "--height", str(sh),
                        "--auto-iters", str(args.auto_iters),
                        "--auto-frames", str(args.auto_frames),
                        "--calib", f"{p}calib_{sw}x{sh}.json"]})

    return runs, fmt, (small, large, exp_min, exp_max, dev_tag)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Query a camera and generate a full cam_characterise.py run set.")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--name", default=None,
                    help="human-readable camera name; saved against the device tag "
                    "and auto-loaded on future runs for the same device")
    ap.add_argument("--tool", default="cam_characterise.py",
                    help="path to the characterisation tool to invoke")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--report-dir", default="camchar_reports",
                    help="passed through to each run for timestamped JSON output")
    ap.add_argument("--discard", type=int, default=5)
    ap.add_argument("--ladder-frames", type=int, default=16)
    ap.add_argument("--auto-iters", type=int, default=20)
    ap.add_argument("--auto-frames", type=int, default=30)
    ap.add_argument("--all-resolutions", action="store_true",
                    help="characterise every advertised discrete resolution. "
                    "Default: only integer-factor decimations of the native (max) "
                    "mode, whose defect/FPN data reflects real sensor pixels; "
                    "non-integer scales are bridge interpolation (artefact maps).")
    ap.add_argument("--neutral-gamma", type=int, default=None,
                    help="gamma for the per-resolution dark runs (default: device "
                    "min == linear). Raise it on bridges whose min-gamma output is "
                    "black-level clamped (dead-flat masters), e.g. --neutral-gamma "
                    "200 on the Sunplus 1bcf:28c4 family. Clamped to the gamma range.")
    ap.add_argument("--clamp-sweep", action="store_true",
                    help="add full-exposure-ladder dark runs at each --clamp-gammas "
                    "step, to map where a black-level clamp releases and whether "
                    "exposure control becomes real (vs synthetic) with gamma")
    ap.add_argument("--clamp-gammas", default="100,150,200,250,300",
                    help="comma-separated gamma values for --clamp-sweep "
                    "(clamped to the device's gamma range). Default 100..300.")
    ap.add_argument("--out", default=None,
                    help="write the generated commands to this shell script")
    ap.add_argument("--run", action="store_true",
                    help="execute the runs sequentially (default: just print)")
    ap.add_argument("--json", default=None,
                    help="write the machine-readable plan (capabilities + runs) here")
    args = ap.parse_args()

    info = identify(args.device)
    formats = query_formats(args.device)
    ctrls = query_controls(args.device)

    print(f"=== {args.device} ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  formats: " + ", ".join(
        f"{f['fourcc']}{'(c)' if f['compressed'] else ''}" for f in formats))
    exp = ctrls.get("exposure_time_absolute", {})
    print(f"  exposure_time_absolute: min={exp.get('min')} max={exp.get('max')} "
          f"default={exp.get('default')}")
    present = ", ".join(
        c for c in ("gamma", "brightness", "contrast", "sharpness", "gain")
        if c in ctrls)
    print("  processing ctrls present: " + (present or "(none)"))

    runs, fmt, meta = plan(args.device, formats, ctrls, args)
    if not runs:
        print("No uncompressed format found; cannot plan measurement runs.",
              file=sys.stderr)
        sys.exit(1)

    small, large, exp_min, exp_max, dev_tag = meta

    _names = load_device_names()
    _device_name = args.name
    if _device_name and dev_tag:
        save_device_name(dev_tag, _device_name)
        print(f"  device name: '{_device_name}' (saved)")
    elif not _device_name and dev_tag and dev_tag in _names:
        _device_name = _names[dev_tag]
        print(f"  device name: '{_device_name}'")

    print(f"\n  measurement format: {fmt['fourcc']}")
    if "gamma" in ctrls:
        g = ctrls["gamma"]
        ng = (g.get("min", 100) if args.neutral_gamma is None
              else max(g.get("min", 100), min(g.get("max", 300), args.neutral_gamma)))
        note = "" if args.neutral_gamma is None else "  (overridden)"
        print(f"  dark-run gamma: {ng}{note}  [device range "
              f"{g.get('min')}..{g.get('max')}]")
        if args.neutral_gamma is None and g.get("min", 100) == 100:
            print("    NOTE: gamma=100 is black-level clamped on some bridges "
                  "(flat masters). If the deep stack reads dead-flat, re-run "
                  "with --neutral-gamma above the clamp.")
    # show which resolutions will be characterised and which were filtered out
    all_sizes = sorted(fmt["sizes"], key=lambda s: s["w"] * s["h"])
    kept, _, _ = pick_resolutions(fmt, native_only=not args.all_resolutions)
    kept_set = {(s["w"], s["h"]) for s in kept}
    nat = max(all_sizes, key=lambda s: s["w"] * s["h"])
    if args.all_resolutions:
        print(f"  resolutions: ALL {len(kept)} advertised modes (--all-resolutions)")
    else:
        dropped = [f"{s['w']}x{s['h']}" for s in all_sizes
                   if (s["w"], s["h"]) not in kept_set]
        print(f"  resolutions: {len(kept)} native-factor modes of "
              f"{nat['w']}x{nat['h']} "
              + ", ".join(f"{s['w']}x{s['h']}(1/{nat['w']//s['w']})" for s in kept))
        if dropped:
            print(f"    skipped (non-integer bridge scales): {', '.join(dropped)}")
            print("    -> pass --all-resolutions to characterise them anyway")
    print(f"  instrument resolution (small): {small['w']}x{small['h']} "
          f"(max {max_advertised_fps(small)} fps)")
    print(f"  full resolution (large): {large['w']}x{large['h']} "
          f"(max {max_advertised_fps(large)} fps)")
    if dev_tag:
        print(f"  device tag: {dev_tag}  (output dir: {dev_tag}/)")
        if args.run:  # print-only / script modes must not touch the fs
            os.makedirs(dev_tag, exist_ok=True)
    else:
        print("  device tag: (not available — USB sysfs lookup failed; "
              "files will be in the current directory)")

    base = [args.python, args.tool, "--device", args.device]
    lines = []
    for r in runs:
        tail = list(r["argv_tail"]) + ["--report-dir", args.report_dir]
        cmd = base + tail
        lines.append((r["label"], r["why"], cmd))

    print(f"\n=== PLAN: {len(lines)} runs ===")
    for label, why, cmd in lines:
        print(f"\n# [{label}] {why}")
        print("  " + " ".join(shlex.quote(c) for c in cmd))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("#!/bin/sh\n# generated by cam_manager.py "
                     f"{datetime.now(timezone.utc).isoformat()}\n")
            fh.write(f"# device {args.device}  format {fmt['fourcc']}  "
                     f"exposure {exp_min}..{exp_max}\n")
            if dev_tag:
                fh.write(f"# device tag: {dev_tag}\n")
                fh.write(f"mkdir -p {shlex.quote(dev_tag)}\n")
            fh.write("set -e\n\n")
            for label, why, cmd in lines:
                fh.write(f"# [{label}] {why}\n")
                fh.write(" ".join(shlex.quote(c) for c in cmd) + "\n\n")
        os.chmod(args.out, 0o755)
        print(f"\nscript written to {args.out}")

    if args.json:
        plan_doc = {"device": args.device, "identity": info,
                    "device_tag": dev_tag,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "formats": formats, "controls": ctrls,
                    "measurement_format": fmt["fourcc"],
                    "runs": [{"label": l, "why": w,
                              "cmd": c} for l, w, c in lines]}
        json.dump(plan_doc, open(args.json, "w"), indent=2)
        print(f"plan JSON written to {args.json}")

    if args.run:
        print(f"\n=== EXECUTING {len(lines)} runs ===")
        for i, (label, why, cmd) in enumerate(lines):
            print(f"\n--- [{i+1}/{len(lines)}] {label} ---")
            try:
                subprocess.run(cmd, check=False)
            except KeyboardInterrupt:
                print("\ninterrupted; stopping plan.")
                break


if __name__ == "__main__":
    main()
