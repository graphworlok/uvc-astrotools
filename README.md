# CamTools — UVC / V4L2 sensor characterisation

Low-level characterisation of UVC webcam-class camera modules over V4L2, aimed
at working out what a given sensor + ISP-bridge + firmware combination *actually*
does versus what its controls claim — exposure fidelity, dark noise, fixed-pattern
and shading structure, hot pixels, and auto-exposure behaviour. Built for
evaluating modules for astrophotography / autoguiding use, where the firmware's
nominal exposure range is usually not the same as its usable one.

Three tools:

- **`cam_characterise.py`** — the measurement engine. Runs dark-noise
  characterisation, an exposure ladder, a photon-transfer-curve scaffold, and an
  auto-exposure observation mode against a single camera at a single resolution.
- **`cam_manager.py`** — a capability-aware planner. Introspects a camera's real
  formats, resolutions, frame intervals and control ranges, then generates a
  tailored set of `cam_characterise.py` runs for that specific device.
- **`cam_observe.py`** — a standalone GUI observing tool built on the same
  stack: generate or load a master dark + defect map (DARK mode, lens capped),
  then live-stack calibrated frames (LIGHT mode), extract stars, and
  plate-solve the stack with a local astrometry.net. See below.

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

## GUI observing tool (`cam_observe.py`)

A standalone tkinter GUI (numpy + stdlib only) around the same capture and
analysis stack. The raw stream is always shown; the second display window is
mode-aware. Two explicitly user-declared modes — the tool cannot know whether
the lens is capped, so the human says so via a banner toggle:

- **DARK mode** — the raw stream beside the **dark-corrected residual** at a
  fixed stretch (deliberately not auto-stretched: a perfect correction should
  *look* black, with residual mean/σ/max stats saying how close it is).
  Acquire a deep master dark (streaming per-pixel mean) and derive a
  hot-pixel defect map, written in exactly the per-device layout the other
  tools use (`{vid+pid[_serial]}/master_WxH.npy` ×257 uint16,
  `defects_WxH.txt`, plus a `dark_meta_WxH.json` sidecar recording the
  exposure). Existing files for the device + resolution load automatically.
- **LIGHT mode** — the raw stream beside the **integrated stack**:
  dark-subtract → flat-correct → defect-repair → optional translation
  alignment (phase correlation) → a **rolling integration window of N
  frames**, with N visible and changeable and the `k/N` count displayed live.
  Stars are extracted from the stack (robust MAD background, connected
  components → sub-pixel centroids, flux, FWHM) and overlaid on the display,
  with a noise readout — the figure of merit when mining the noise floor
  with no stars at all.

**Static-camera star confirmation** — detection assumes a fixed mount: real
stars persist burst after burst and drift slowly and predictably, noise
doesn't. Every detection feeds an α-β-filtered track with velocity
prediction, and only tracks confirmed over several consecutive bursts are
reported (to the overlay, the auto-solve gate, and the pause-mode motion
reference). That lets the detection threshold run *lower* — steered
adaptively toward a user-set **"Visible stars ≈ N"** count (down to 3.5σ,
backing off if candidates swamp the target) — finding fainter real stars
without false positives, since unconfirmed candidates are never shown. The
status line reports `confirmed / candidates @ threshold`. Both the raw and
stack views also pop out into **dedicated live windows** (click the raw
view, double-click the stack view): resizable, fully overlaid, and their
size feeds the capture downsampling target, so a large window genuinely
receives more resolution.

**Flat-field correction** — in LIGHT mode, aim the camera at an evenly
illuminated field (twilight sky, light panel, defocused white surface) and
acquire a **master flat**: a streamed per-pixel mean, dark-subtracted when a
master dark is loaded, normalised to a median-1 gain map and saved per device
(`flat_WxH.npy` + `flat_meta_WxH.json`), auto-loaded per resolution like the
dark. Saturated or signal-starved flats are rejected outright (they are not a
gain measurement), and the gain map is clamped so a dead vignette corner can
never turn the correction into a noise amplifier. A **flat correct** toggle
applies it per frame in one of two modes: **divide** (true gain correction —
photometrically right when the processing path is linear) or **subtract**
(removes the flat's structure scaled to each frame's own background — not
photometric, but it cannot over/under-correct on the gamma'd, black-clamped
processing paths these cameras usually run).

Flat acquisition **pre-flights the signal level**: it holds before
accumulating — telling you which way to adjust the illumination (median ADU
and clip fraction live in the status line) — until the signal sits in the
usable band, so a flat that the normaliser would only reject can never be
built. The exposure itself is deliberately *not* changed: the flat should be
shot at the observing exposure (a nonlinear ISP amplification won't cancel
otherwise), so the illumination is the right knob — and the tool can turn
that knob itself: a **Flat panel** button opens a uniform grey window
(fullscreen-capable, level 0–255, movable to any attached screen) to aim the
camera at, and with **auto level** enabled the acquisition servos the panel
brightness into the target band before accumulating. Point the camera at the
screen (slightly defocused, or through a paper diffuser, to even out pixel
structure and backlight non-uniformity), press Acquire, and the flat builds
itself with every camera setting untouched. The panel isn't limited to the
capture machine's own screen: **Serve to browser** starts a small HTTP
server (`--panel-port`, default 8799) whose page turns any browser — a
large desktop monitor, a tablet clamped in front of the objective — into
the same servo-driven panel (tap for fullscreen with a screen wake lock).
The endpoint is read-only by design: the network can only ask what shade to
display. Remote panels **long-poll** — the server holds the request until
the level changes, so updates land immediately rather than on a poll grid —
and **acknowledge** each painted level, so the auto-level servo waits for
confirmation (plus a render-to-photons margin) instead of sleeping blind.
Display gamma spreads the 8-bit grey level across a wide luminance range
(level 1/255 ≈ 5×10⁻⁶ of full white on a γ2.2 panel), so in most setups the
display's hardware brightness needn't be adjusted — just disable its
auto-brightness, which would otherwise drift with ambient light
mid-acquisition. (A web page cannot control backlight hardware — no such
browser API exists.) When you can't control the light
source at all (twilight sky), an opt-in **auto exposure** checkbox instead
bisects the exposure range to a mid-scale median — accepted with a logged
warning, and the flat/observing exposure mismatch NOTE fires at capture
start.

A **test image** page rides the same plumbing: basic test patterns — a line
grid with a centre crosshair (geometry/distortion/FOV), a checkerboard
(focus/contrast), a 36-spoke Siemens star (focus through progressively finer
detail toward the centre) — with **level** (0–255), **scale** (feature size
in px) and **invert** (dark-on-bright) knobs in the *Test image* row.
**Test window** opens it locally (Up/Down level, F11/double-click
fullscreen, Esc closes), and when *Serve to browser* is running the same
patterns are at **`/test`** on the panel port, so any browser-equipped
screen becomes the target. Remote pages use the same triggered-refresh
method as the flat panel, keyed on a revision counter — pattern or knob
changes repaint immediately, each painted revision is acknowledged, and the
endpoint stays read-only.

Flat **review/analysis** comes in two layers. **View flat** opens the gain
map auto-stretched with its **radial gain profile** (the shape separates
optical falloff, drooping toward the corners, from upstream LSC
over-correction, rising toward them), the per-pixel noise the divide injects,
and a corner-vs-centre verdict. During integration the stack's
**corner/centre background ratio** is computed every burst and shown live
(and logged per burst under `--debug`) — point the camera at an evenly lit
field and toggle *flat correct*: the ratio walking to 100% is the
closed-loop proof the flat is actually correcting the field.

**Pause integration** freezes stacking while capture and display continue —
for when the sensor is about to move, e.g. adjusting an equatorial mount's
altitude/azimuth bolts. While paused, stars detected in the live frames are
matched to their pre-pause stack positions and drawn as **motion arrows**,
with a median displacement/direction readout: live feedback on where the
field is going during polar alignment. Resume optionally resets the stack
(the old integration is invalid once the sensor has moved).

An auxiliary panel column carries three live visualisations:

- **Pole bullseye** — a polar-scope-style chart with rings at 1′/5′/10′/30′/1°
  (√-radial mapping so the inner rings stay legible), plotting each solve at
  angle = RA, radius = pole separation, with a fading trail. Watching the dot
  walk toward centre as you turn the bolts *is* the polar alignment. It is
  hemisphere-aware: the pole (NCP/SCP) is read from each solve's declination
  and the angular handedness is mirrored for the SCP, so the chart turns the
  way the southern sky actually does — which is where solve-based alignment
  matters most, the SCP having no bright pole star to sight on.
- **Stack noise vs √N** — measured stack noise against the 1/√N ideal as
  integration deepens. Where the measured curve departs from ideal is the
  FPN/drift floor: the actual limit of integrating deeper, and the key figure
  when mining the noise floor.
- **Histogram (log-y)** — raw vs stack in light mode, raw vs residual in dark
  mode, showing pedestal position, clipping at either rail, and headroom at a
  glance.

When a solve yields a WCS, the **celestial pole's pixel position** is computed
from the CD matrix (exact, including rotation and parity) and drawn as a
magenta crosshair on the stack view whenever it falls within the frame — so
the live image itself becomes the polar scope.

The UVC processing controls the device actually exposes (gain, gamma,
brightness, contrast, sharpness, saturation) are presented with their real
ranges after probing and applied live; exposure is adjustable mid-capture.

Every solve attempt — success or failure, manual or auto — is **archived
independently of `--debug`** as one JSON line in the per-device data dir's
`solves.jsonl`: timestamp, session context (stack depth, exposure, ROI,
confirmed star count, calibration exposures), the full command line and
solver output, duration, and the WCS solution both parsed and as the raw
base64 FITS header block (any FITS/WCS library can reconstruct it). Grep it
for `"solved": true`, or pull fields with `jq`/`python -c`. The bulky
per-star artefacts (`.axy`/`.corr`/`.rdls`) are inventoried by name and
size but not copied — re-solving a saved stack regenerates them.

Plate solving runs a local astrometry.net (`solve-field`) on the stack
(written as linear 16-bit FITS), manually or on an auto-solve timer once
enough stars are present, and reports RA/Dec field centre, rotation, and the
**angular offset from the celestial pole** — with the camera mounted in or
coaxial to an equatorial's RA axis, that offset is the live polar-alignment
number while adjusting the mount. (Fitting the mechanical rotation centre
from solves at several RA-axis rotations is the planned refinement on top.)

A second, independent use of the same solve is **pointing tracking** for a
camera aimed anywhere — not just the polar cap `make_catalog.py` covers,
e.g. a wide-angle finder riding along on the mount. One solve arms a
catalogue-free WCS anchor; every burst after that is kept current purely by
**matching this burst's stars against the previous burst's** (nearest pixel
neighbour) and folding the fitted rigid transform (rotation + translation)
back onto the anchor — the same trimmed-outlier rigid fit the catalogue path
uses, just fed frame-to-frame star motion instead of catalogue predictions,
so it needs no re-solve and no catalogue coverage to stay live. The current
frame-centre RA/Dec is drawn on the stack view (a small crosshair + text)
and, when the **Stellarium UDP** row is armed, streamed out as
`MessageCurrentPosition` packets (Stellarium Telescope Protocol v1.0 —
24-byte little-endian LENGTH/TYPE/TIME/RA/DEC/STATUS, verified against the
canonical spec) so Stellarium's own sky chart can show a live reticle
tracking wherever the camera is pointed. Stellarium's Telescope Control
plugin only speaks TCP, so a small UDP→TCP relay sits between this tool and
Stellarium — no Stellarium plugin or rebuild needed, stock Stellarium:

```sh
# Windows (ncat, from nmap)
ncat -l -p 10001 --sh-exec "ncat --udp -l -p 5005"
# Linux/macOS (socat)
socat TCP-LISTEN:10001,fork UDP-RECVFROM:5005,fork
```

then in Stellarium: *Telescope Control* → configure a new telescope →
"External software or a remote computer" → host `127.0.0.1`, port `10001`
(the port Stellarium itself connects to; `--stellarium-port`/the UDP port
row is where *this tool* sends, default 5005 — the relay bridges the two).

```sh
python3 cam_observe.py --device /dev/video0 --data-dir ~/.local/share/PHD2
```

`--debug` writes JSONL diagnostics designed for LLM consumption: one
self-describing event per line (probe results, control writes, a per-burst
heartbeat with frame statistics before/after calibration, alignment shifts,
pause-mode star tracking, solve command lines and output tails, tracebacks),
with the schema and environment documented in the first line — paste the file
into a model session and it can reconstruct what happened without the source.

For "that looks *wrong*" moments there is a **Deep-dive dump** button: one
press captures the next burst in full — every raw frame, every processed
frame (after each enabled calibration step, including the alignment shift),
the master dark/flat/defect data in use, every parameter, and per-frame
statistics — into a timestamped directory whose `manifest.json` documents
each file and the complete processing context. All four capture modes
support it (light integration, dark preview, dark/flat acquisition); pressed
while idle it dumps the session's in-memory state instead. The directory is
self-describing by design: hand it over and the question "why does the
output look like this?" is answerable offline, without reproducing the
sky, the setup, or the moment.

Point `--data-dir` at the PHD2 data dir and the darks/defect maps built here
are the same files PHD2 auto-loads at camera connect.

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

- **Converts the master into the camera's live pixel domain.** The .npy is
  stored at 16-bit full scale, but PHD2 does not normalise camera data by bit
  depth — an 8-bit capture path delivers raw 0–255 values — so the import
  rescales the master (and the model bias) by the connected camera's bit depth
  before any subtraction.
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
