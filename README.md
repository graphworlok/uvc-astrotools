# CamTools — UVC / V4L2 sensor characterisation

Low-level characterisation of UVC webcam-class camera modules over V4L2, aimed
at working out what a given sensor + ISP-bridge + firmware combination *actually*
does versus what its controls claim — exposure fidelity, dark noise, fixed-pattern
and shading structure, hot pixels, and auto-exposure behaviour. Built for
evaluating modules for astrophotography / autoguiding use, where the firmware's
nominal exposure range is usually not the same as its usable one.

Two tools:

- **`cam_characterise.py`** — the measurement engine. Runs dark-noise
  characterisation, an exposure ladder, a photon-transfer-curve scaffold, and an
  auto-exposure observation mode against a single camera at a single resolution.
- **`cam_manager.py`** — a capability-aware planner. Introspects a camera's real
  formats, resolutions, frame intervals and control ranges, then generates a
  tailored set of `cam_characterise.py` runs for that specific device.

## Why

UVC camera bridges routinely lie, clamp, or quietly reinterpret their controls.
Across the modules tested here the failure modes included: a non-monotonic
exposure table where commanding *more* exposure gave *less* integration; an
exposure register that resets to pedestal on a sawtooth; auto-exposure that
optimises into the firmware's worst no-integration corner; and a processing path
whose gamma/brightness controls are invisible to the auto-exposure loop. None of
that is visible from the control descriptors — it only shows up under
measurement. These tools measure it.

## Requirements

- Linux with V4L2 and `v4l2-ctl` (`v4l-utils`) on `PATH`
- Python 3 with NumPy
- A UVC camera on `/dev/videoN`
- The lens capped / full darkness for dark runs

The capture path shells out to `v4l2-ctl --stream-mmap ... --stream-to=FILE`
rather than driving ioctls directly — this proved to be the only reliable path
on the bridges tested, and it applies format + exposure + capture in a single
invocation to avoid multi-open races.

## Quick start

Plan and run a full characterisation set for whatever camera is attached:

```sh
# see the plan only
python3 cam_manager.py --device /dev/video0

# write a runnable script
python3 cam_manager.py --device /dev/video0 --out probe.sh

# introspect, plan, and execute, writing timestamped JSON reports
python3 cam_manager.py --device /dev/video0 --run
```

Or drive the engine directly. A dark run with an exposure ladder, linear gamma,
saving a master dark, an fps↔exposure calibration, and a timestamped report:

```sh
python3 cam_characterise.py --dark \
  --width 640 --height 480 \
  --discard 5 --ladder-frames 16 \
  --exposure-max 2047 --knee 256 \
  --ladder 3,256,512,768,1024,1280,1536,1792,2047 \
  --gamma 100 --brightness 0 --contrast 0 --sharpness 1 \
  --save-master master.npy \
  --save-calib calib.json \
  --report-dir reports
```

Auto-exposure observation in the dark, using a previously built calibration to
cross-check the exposure readback against the framerate-implied exposure:

```sh
python3 cam_characterise.py --auto \
  --width 640 --height 480 \
  --auto-iters 20 --auto-frames 30 \
  --calib calib.json \
  --report-dir reports
```

## What it measures

### Format ranking
Enumerates formats and ranks them for measurement fidelity: true mono
(GREY / Y800 / Y16 — no chroma path at all, the ideal on mono guide-camera
gadgets) best, then uncompressed YUYV / YUV; lossy MJPG / H264 / HEVC refused
(they destroy the noise you are trying to measure). Y16 is normalised into the
same 0–255 luma domain at sub-ADU precision so every threshold and fit works
unchanged. The tool picks the best format automatically.

### Dark characterisation (`--dark`)
Captures a deep stack at maximum exposure plus a ladder of shorter exposures,
all with the lens capped, computing per-pixel statistics with a streaming
Welford accumulator (so a 256-frame 4K stack costs ~tens of MB, not gigabytes):

- **Bias / pedestal** — the black level the ISP clamps to.
- **Dark current** — slope of mean ADU vs exposure, fit over the integration-
  limited region only (below the frame-period knee the level is just the clamped
  pedestal and carries no dark-current information).
- **Fixed-pattern noise (FPN)** — spatial spread of the master dark; subtractable.
- **Irreducible temporal dark noise** — per-pixel temporal std after master-dark
  subtraction; the single-frame floor.
- **Hot pixels** — single-pixel outliers above 6 robust sigma of the residual
  after a local median smooth. (Thresholding the raw master at mean+6σ fails at
  native resolution: the LSC shading gradient inflates the global σ and hides
  real hot pixels; a MAD-based σ on the structure-removed residual is immune to
  both the gradient and the hot pixels themselves.) Note the count scales
  strongly with resolution: downscaled modes bin/average and hide hot pixels
  that the native array exposes.
- **Radial profile** — mean ADU binned by distance from frame centre, to expose
  shading structure. With the lens capped there is no optical vignetting, so any
  radial gradient is the bridge's lens-shading-correction (LSC) gain map showing
  through on the pedestal. A corner/centre ratio > 1 is "reverse vignetting" —
  the LSC map over-correcting a flat field.

### Exposure fidelity
The framerate is used as an independent instrument. When commanded exposure
exceeds the frame period, real integration forces the framerate down; synthetic
exposure (gain-boost or frame-summing) holds the framerate flat. The tool reports
fps-drop vs exposure-rise over the integrating region and flags real vs synthetic
exposure from that — which is robust even when the exposure-vs-mean fit is not.

**Resolution matters here.** The fps-as-exposure instrument only works at a
resolution whose bandwidth ceiling is *above* the integration knee. At a large
frame over USB2 the stream is bandwidth-limited and the framerate is pinned
regardless of exposure, masking integration entirely. Use the smallest available
resolution for exposure-fidelity work; use the native resolution for true
hot-pixel and shading maps.

### fps ↔ exposure calibration (`--save-calib` / `--calib`)
A dark run can emit a monotonic fps→exposure inverse, fit over the integration-
limited region. In auto-exposure mode this converts the measured framerate into
an effective exposure, independent of the (possibly lying) `exposure_time_absolute`
readback. The calibration is resolution-specific — rebuild it per resolution.

The framerate itself is measured from the kernel's per-buffer timestamps
(v4l2-ctl `--verbose` dqbuf lines), not wall-clock over the whole capture —
process spawn / STREAMON overhead would otherwise bias short ladder captures
low and the long deep stack less, bending the calibration curve.

### Auto-exposure observation (`--auto`)
Hands control back to AE first — UVC controls are sticky across processes, so
after a manual dark run the device is still in manual mode and would otherwise
be "observed" as a perfectly stable AE. The mode then caps the lens and logs the
AE-chosen exposure and gain, the framerate, and the resulting mean/noise over
many iterations. Reports where AE parked (as a fraction of the device's real
exposure range, and in seconds — UVC exposure is in 100µs units), whether it is
stable or hunting, and whether the exposure readback agrees with the framerate-
implied exposure (a control-honesty check).

### Photon transfer curve (`--ptc`)
Scaffold for a lit PTC (variance vs mean across illumination levels) to recover
gain and read noise in electrons. Requires a controllable light source; the dark
path cannot disambiguate a genuinely low-noise sensor from a denoised one at
8-bit, but a clean PTC slope can.

### Processing-path controls
`--gamma`, `--brightness`, `--contrast`, `--sharpness`, `--saturation` set the
ISP processing controls (for clean measurement: gamma linear, sharpness min,
brightness/contrast neutral). A **clip guard** flags any capture driven to the
floor (0) or ceiling (255) by aggressive settings and marks the measurement
invalid rather than reporting clipped zeros as data. If the deep stack itself
is clipped, the master / defects / dark-model exports are **refused** — writing
them would hand PHD2 meaningless calibration to auto-load at every connect.

## Output

Per run, with `--report-dir DIR`, a timestamped JSON report
(`camchar_<mode>_<WxH>_<UTC>.json`) containing the full ladder, per-point
mean/noise/fps/clip flags, dark-current fit, radial profile, fps calibration, and
(for `--auto`) the AE series and verdict. `--save-master FILE.npy` writes the 2D
master dark for offline analysis (e.g. differencing shading maps across settings).

## Safety notes

These tools deliberately touch **only standard V4L2 controls and descriptors**.
They do not enumerate or poke vendor extension units (XUs): on one module tested,
XU GET/SET wedged the device into a U-Boot mass-storage recovery state. If you go
XU-spelunking, do it knowingly and separately.

Long unattended sweeps on warm, uncooled modules: the tools stream to a temp file
and clean up, but a multi-hour native-resolution sweep writes a lot to disk and
heats the sensor (which shifts the dark current and hot-pixel count across the
run). Interleave or randomise exposure order if you need to separate thermal drift
from exposure dependence.

## Relationship to the PHD2 V4L2 work

The practical output of characterising a module is the **usable exposure window**,
which is what a guider actually needs to know and which the firmware's nominal
range does not give you:

- A faithful, monotonic exposure region exists only between the firmware's
  clamped low end and wherever its exposure table breaks. Outside it, commanding
  more exposure can give less integration.
- The point of peak real sensitivity is not necessarily the maximum commanded
  exposure — on one module it was well below max, with everything above it
  returning progressively *less* signal.
- Auto-exposure cannot be trusted to find that window; on the modules tested it
  railed to the ceiling in the dark, which on the broken-table module was the
  worst possible choice.

For the PHD2 fork this means: pin exposure to a measured value inside the usable
window rather than using AE, and treat each module's window as a per-device
calibration constant derived from a dark run here.

### Handoff format

`cam_manager.py` writes a per-device directory named by USB identity
(`{idVendor}{idProduct}[_{serial}]`, read from sysfs). The plan includes a dark
run at **every discrete resolution** the chosen format offers — not just
smallest and native — so PHD2 finds matching artefacts for whatever frame size
it connects at. Each run produces:

| File | Built by | Consumed by PHD2 for |
|------|----------|----------------------|
| `calib_WxH.json`      | `--save-calib`      | fps↔exposure inversion; the real integration window |
| `master_WxH.npy`      | `--save-master`     | master dark, uint16 (luma ×257) ready for `usImage` |
| `defects_WxH.txt`     | `--save-defects`    | hot-pixel defect map (`x y` per line, PHD2 v1 format) |
| `dark_model_WxH.json` | `--save-dark-model` | dark-current slope + bias, exposure tag, fidelity verdict |

Drop that directory into the PHD2 data dir (e.g.
`~/.local/share/PHD2/{idVendor}{idProduct}_{serial}/`). On camera connect PHD2
matches the directory by USB ID and the file by frame size, then:

- **Scales the master dark to each guide exposure.** Because the dark-current
  fit gives the per-pixel slope, the single deep master at `exposure_max` is
  scaled down to every exposure in PHD2's list — `dark(E) = bias +
  (master − bias)·(E / max)` — so a short guide frame is matched to a short dark
  instead of having an over-long master over-subtract its hot pixels. Scaling is
  applied **only when the dark-current fit is linear** (r² > 0.97); otherwise the
  native master is kept and longer-than-max exposures always fall back to it.
- **Cross-checks exposure honesty per frame** using the fps calibration: a guide
  frame's capture time implies an effective exposure, and a disagreement with the
  commanded value is logged (the firmware is not integrating as told).
- **Surfaces the verdict, not enforces it.** The derived real-integration window,
  the REAL/SYNTHETIC/PARTIAL fidelity verdict, and the dark-fit r² are shown in
  the status bar at connect and logged — advisory only; nothing is clamped.

## Status

Working. Format ranking, streaming memory-safe dark stacks, exposure-fidelity
verdict, regime-aware dark-current fit, fps↔exposure calibration, AE observation,
radial shading profile, clip guard, per-device range-aware reporting, and the
capability-planning manager are all implemented. PTC is scaffolded (needs a lit
rig). Raw-Bayer / electron-referred read noise is out of scope for the UVC path —
it needs raw sensor access below the bridge.
