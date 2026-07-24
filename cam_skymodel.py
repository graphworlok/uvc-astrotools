#!/usr/bin/env python3
"""
cam_skymodel.py -- the sky as a prior, for cam_observe.py.

The instrument points at a known sky. After one anchor plate solve the
position of every catalogue star in (and near) the field is predictable,
and -- once the camera's flux response has been fitted -- so is its
brightness. That inverts the detection problem: instead of blindly
extracting "stars" from a noisy frame, the pipeline can CONFIRM predicted
stars at a far lower threshold (a bump only counts if it lands on a
prediction with roughly the predicted flux), track mount moves as low-DOF
rigid transforms instead of re-solving, and treat everything the model
cannot explain as either noise at the measured sigma or a transient.

This module holds the non-GUI machinery for that:

  * a minimal FITS binary-table reader for solve-field's .corr output
    (the per-solve table pairing each detected field source with the
    Tycho-2 index star it matched -- positions, catalogue magnitudes,
    measured flux). Hand-rolled like the FITS writer in cam_observe.py:
    the format subset these files use is tiny and a dependency is not
    justified for it.

  * a flux--magnitude response fit built from accumulated corr tables.
    On a nonlinear UVC path (gamma, black clamp, denoise) the response is
    NOT the textbook log10(flux) = c - 0.4*mag: the faint end is
    compressed by the clamp -- measured on the HEQ5 polar-scope rig at
    roughly half the expected flux by mag ~8.5. The fit is therefore a
    robust line (Theil-Sen: immune to the odd mismatched or saturated
    star) plus an optional quadratic faint-end term kept only when it
    actually improves the fit. The result is persisted per
    device+resolution beside the master dark/flat and gives
    flux_expected(mag) with honest scatter, so a gated detector knows
    how much flux plausibility to demand.

  * TanWCS: a small gnomonic (TAN) WCS -- CRVAL/CRPIX/CD -- built from
    the cards solve-field writes (as parsed by cam_observe's
    _parse_wcs_cards). Vectorised sky_to_pix / pix_to_sky, exact about
    parity because the CD matrix carries it (no mirror-flip ambiguity --
    same property pole_pixel_from_wcs relies on). compose_rigid() applies
    a fitted image-plane rotation+translation (a mount move tracked by
    matched stars); advance_sidereal() advances the pointing of a
    ground-fixed camera at the sidereal rate, which for a TAN projection
    is exactly a shift of CRVAL1 (the pole pixel is invariant -- a
    property the tests assert, since the whole polar-alignment scheme
    steers that pixel).

Requires: numpy only. Python 3.8+.

Usage (analysis CLI):
  python3 cam_skymodel.py corr.fits [corr2.fits ...] [--save-response DIR]
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np

SKYMODEL_VERSION = "0.1"

# Sidereal rate, degrees of RA per SI second (360 deg / 86164.0905 s).
SIDEREAL_DEG_PER_S = 360.0 / 86164.0905


# ----------------------------------------------------------------------------
# Minimal FITS binary-table reader (solve-field .corr subset)
# ----------------------------------------------------------------------------

_FITS_BLOCK = 2880
_TFORM_DTYPES = {"L": "u1", "B": "u1", "I": ">i2", "J": ">i4", "K": ">i8",
                 "E": ">f4", "D": ">f8", "A": "S"}


def _read_header(buf, off):
    """Parse one FITS header (80-byte cards, 2880-byte blocks) starting at
    off. Returns (cards_dict, data_offset). Values are kept as int/float
    where they parse, else stripped strings; COMMENT/HISTORY ignored."""
    cards = {}
    while True:
        block = buf[off:off + _FITS_BLOCK]
        if len(block) < _FITS_BLOCK:
            raise ValueError("truncated FITS header")
        off += _FITS_BLOCK
        done = False
        for i in range(0, _FITS_BLOCK, 80):
            card = block[i:i + 80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                done = True
                break
            if card[8:10] != "= " or key in ("COMMENT", "HISTORY", ""):
                continue
            val = card[10:]
            # strip inline comment outside quoted strings
            if val.lstrip().startswith("'"):
                s = val.lstrip()[1:]
                end = s.find("'")
                cards[key] = s[:end].strip()
                continue
            val = val.split("/")[0].strip()
            if val in ("T", "F"):
                cards[key] = (val == "T")
                continue
            try:
                cards[key] = int(val)
            except ValueError:
                try:
                    cards[key] = float(val)
                except ValueError:
                    cards[key] = val
        if done:
            return cards, off


def _skip_data(cards, off):
    """Advance past an HDU's data area (block-padded)."""
    naxis = cards.get("NAXIS", 0)
    if naxis == 0:
        return off
    n = abs(cards.get("BITPIX", 8)) // 8
    for i in range(1, naxis + 1):
        n *= cards.get(f"NAXIS{i}", 0)
    n *= max(1, cards.get("GCOUNT", 1))
    n += cards.get("PCOUNT", 0)
    return off + ((n + _FITS_BLOCK - 1) // _FITS_BLOCK) * _FITS_BLOCK


def read_bintable(path):
    """Read the first BINTABLE extension of a FITS file as a numpy
    structured array (big-endian fields converted to native). Supports the
    scalar column types solve-field emits (D/E/J/I/K/A(n)); array repeat
    counts other than 1 are refused rather than mis-read."""
    with open(path, "rb") as fh:
        buf = fh.read()
    cards, off = _read_header(buf, 0)          # primary HDU
    off = _skip_data(cards, off)
    while off < len(buf):
        cards, off = _read_header(buf, off)
        if cards.get("XTENSION", "").strip() == "BINTABLE":
            break
        off = _skip_data(cards, off)
    else:
        raise ValueError(f"{path}: no BINTABLE extension found")

    nrow = cards["NAXIS2"]
    rowlen = cards["NAXIS1"]
    nfield = cards["TFIELDS"]
    names, formats = [], []
    for i in range(1, nfield + 1):
        name = str(cards.get(f"TTYPE{i}", f"col{i}"))
        tform = str(cards[f"TFORM{i}"]).strip()
        j = 0
        while j < len(tform) and tform[j].isdigit():
            j += 1
        repeat = int(tform[:j]) if j else 1
        code = tform[j]
        if code == "A":
            formats.append(f"S{repeat}")
        else:
            if repeat != 1:
                raise ValueError(
                    f"{path}: column {name} has repeat {repeat}{code}; "
                    "only scalar columns are supported")
            formats.append(_TFORM_DTYPES[code])
        names.append(name)
    dt = np.dtype({"names": names, "formats": formats})
    if dt.itemsize != rowlen:
        raise ValueError(f"{path}: row size mismatch "
                         f"(dtype {dt.itemsize} vs NAXIS1 {rowlen})")
    data = np.frombuffer(buf, dtype=dt, count=nrow, offset=off)
    # native byte order copy: downstream numpy ops don't want big-endian
    return data.astype(data.dtype.newbyteorder("=")), cards


def read_corr(path):
    """Read a solve-field .corr match table. Returns a structured array
    with at least field_x/field_y, index_ra/index_dec, MAG, FLUX (the
    columns astrometry.net writes for Tycho-2-based indexes)."""
    data, _ = read_bintable(path)
    for req in ("field_x", "field_y", "index_ra", "index_dec"):
        if req not in data.dtype.names:
            raise ValueError(f"{path}: missing column {req} -- "
                             "not a solve-field corr table?")
    return data


# ----------------------------------------------------------------------------
# Flux--magnitude response
# ----------------------------------------------------------------------------

def _theil_sen(x, y):
    """Robust line fit y = a + b*x: median of pairwise slopes, then the
    median intercept. O(n^2) pairs -- fine at corr-table sizes."""
    n = len(x)
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(n) for j in range(i + 1, n)
              if x[j] != x[i]]
    if not slopes:
        raise ValueError("degenerate response data (all one magnitude)")
    b = float(np.median(slopes))
    a = float(np.median(y - b * x))
    return a, b


def fit_response(mag, flux):
    """Fit log10(flux) as a function of catalogue magnitude.

    Base model: Theil-Sen line -- robust to the occasional mismatched,
    blended, or saturated star that least squares would chase. A camera
    with a linear path and no losses would give slope -0.4; a gamma'd
    8-bit UVC path gives something shallower, and that deviation is
    itself a measurement of the path.

    Faint-end term: a quadratic in (mag - brightest) is fitted by plain
    least squares ON TOP of the data and kept only if it reduces the RMS
    residual by >=10% AND stays monotone decreasing over the fitted
    range. The black clamp compresses faint fluxes below the line; when
    the data shows that, the curve follows it, and when it doesn't the
    simpler model stands. Returns a dict (JSON-ready) -- see
    flux_expected() for evaluation."""
    mag = np.asarray(mag, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    keep = np.isfinite(mag) & np.isfinite(flux) & (flux > 0)
    mag, flux = mag[keep], flux[keep]
    if len(mag) < 4:
        raise ValueError(f"only {len(mag)} usable star(s) -- need >= 4")
    logf = np.log10(flux)
    a, b = _theil_sen(mag, logf)
    lin_res = logf - (a + b * mag)
    lin_rms = float(np.sqrt(np.mean(lin_res ** 2)))

    model = {"kind": "linear", "a": a, "b": b, "rms": lin_rms}
    m0 = float(mag.min())
    X = np.column_stack([np.ones_like(mag), mag - m0, (mag - m0) ** 2])
    coef, *_ = np.linalg.lstsq(X, logf, rcond=None)
    quad_res = logf - X @ coef
    quad_rms = float(np.sqrt(np.mean(quad_res ** 2)))
    if quad_rms < 0.9 * lin_rms:
        # monotone check: d(logf)/dmag = c1 + 2*c2*(mag-m0) must stay < 0
        mm = np.linspace(0.0, float(mag.max()) - m0, 64)
        if np.all(coef[1] + 2.0 * coef[2] * mm < 0.0):
            model = {"kind": "quad", "m0": m0,
                     "c": [float(c) for c in coef], "rms": quad_rms,
                     "linear_rms": lin_rms}
    model.update(n=len(mag),
                 mag_range=[float(mag.min()), float(mag.max())],
                 slope_expected_linear=-0.4,
                 fitted_at=datetime.now(timezone.utc).isoformat(),
                 skymodel_version=SKYMODEL_VERSION)
    return model


def flux_expected(model, mag):
    """Evaluate a fit_response() model: expected flux at catalogue mag.
    Outside the fitted range the curve extrapolates its end slope --
    callers gating detections should widen tolerance out there."""
    mag = np.asarray(mag, dtype=np.float64)
    if model["kind"] == "quad":
        c = model["c"]
        x = mag - model["m0"]
        logf = c[0] + c[1] * x + c[2] * x * x
    else:
        logf = model["a"] + model["b"] * mag
    return 10.0 ** logf


def response_path(data_dir, w, h):
    return os.path.join(data_dir, f"response_{w}x{h}.json")


def save_response(model, data_dir, w, h, sources=None):
    """Persist beside the master dark/flat, same per-device convention."""
    model = dict(model, resolution=f"{w}x{h}",
                 sources=list(sources or []))
    os.makedirs(data_dir, exist_ok=True)
    p = response_path(data_dir, w, h)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    return p


def load_response(data_dir, w, h):
    try:
        with open(response_path(data_dir, w, h), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------
# TAN WCS
# ----------------------------------------------------------------------------

class TanWCS:
    """Gnomonic WCS: CRVAL (deg), CRPIX (1-based, FITS convention, kept
    as-is internally; the pixel API is 0-based to match the rest of the
    tool), CD matrix (deg/px). The CD matrix encodes rotation AND parity,
    so every derived direction -- including where the celestial pole sits
    -- is exact, with no mirror ambiguity. Full spherical gnomonic
    formulas, not small-angle: at 86 deg declination the difference
    matters."""

    def __init__(self, crval1, crval2, crpix1, crpix2, cd):
        self.ra0 = float(crval1)
        self.dec0 = float(crval2)
        self.crpix = (float(crpix1), float(crpix2))
        self.cd = np.asarray(cd, dtype=np.float64).reshape(2, 2)
        det = np.linalg.det(self.cd)
        if abs(det) < 1e-20:
            raise ValueError("singular CD matrix")
        self.cd_inv = np.linalg.inv(self.cd)

    @classmethod
    def from_cards(cls, cards):
        """Build from the dict produced by cam_observe._parse_wcs_cards
        (or any mapping with CRVAL/CRPIX/CD1_1..CD2_2)."""
        return cls(cards["CRVAL1"], cards["CRVAL2"],
                   cards["CRPIX1"], cards["CRPIX2"],
                   [[cards["CD1_1"], cards["CD1_2"]],
                    [cards["CD2_1"], cards["CD2_2"]]])

    def to_cards(self):
        return {"CRVAL1": self.ra0, "CRVAL2": self.dec0,
                "CRPIX1": self.crpix[0], "CRPIX2": self.crpix[1],
                "CD1_1": self.cd[0, 0], "CD1_2": self.cd[0, 1],
                "CD2_1": self.cd[1, 0], "CD2_2": self.cd[1, 1]}

    # -- projections ------------------------------------------------------

    def sky_to_pix(self, ra, dec):
        """(ra, dec) degrees -> 0-based (x, y) pixels. Vectorised. Points
        on the far hemisphere (behind the tangent plane) return NaN
        rather than a spurious mirrored position."""
        ra = np.radians(np.asarray(ra, dtype=np.float64))
        dec = np.radians(np.asarray(dec, dtype=np.float64))
        ra0 = math.radians(self.ra0)
        dec0 = math.radians(self.dec0)
        dra = ra - ra0
        den = (np.sin(dec) * math.sin(dec0)
               + np.cos(dec) * math.cos(dec0) * np.cos(dra))
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = np.degrees(np.cos(dec) * np.sin(dra) / den)
            eta = np.degrees((np.sin(dec) * math.cos(dec0)
                              - np.cos(dec) * math.sin(dec0)
                              * np.cos(dra)) / den)
            xi = np.where(den > 1e-9, xi, np.nan)
            eta = np.where(den > 1e-9, eta, np.nan)
        dx = self.cd_inv[0, 0] * xi + self.cd_inv[0, 1] * eta
        dy = self.cd_inv[1, 0] * xi + self.cd_inv[1, 1] * eta
        return (dx + self.crpix[0] - 1.0), (dy + self.crpix[1] - 1.0)

    def pix_to_sky(self, x, y):
        """0-based pixels -> (ra, dec) degrees. Vectorised inverse of
        sky_to_pix (exact, not iterative -- TAN inverts in closed form)."""
        x = np.asarray(x, dtype=np.float64) - (self.crpix[0] - 1.0)
        y = np.asarray(y, dtype=np.float64) - (self.crpix[1] - 1.0)
        xi = np.radians(self.cd[0, 0] * x + self.cd[0, 1] * y)
        eta = np.radians(self.cd[1, 0] * x + self.cd[1, 1] * y)
        ra0 = math.radians(self.ra0)
        dec0 = math.radians(self.dec0)
        den = math.cos(dec0) - eta * math.sin(dec0)
        ra = ra0 + np.arctan2(xi, den)
        dec = np.arctan((np.cos(ra - ra0)
                         * (eta * math.cos(dec0) + math.sin(dec0)))
                        / np.where(np.abs(den) < 1e-15, 1e-15, den))
        return np.degrees(ra) % 360.0, np.degrees(dec)

    def pole_pixel(self):
        """0-based pixel of the celestial pole of the field's hemisphere.
        Same computation pole_pixel_from_wcs performs on raw cards; kept
        here so a tracked (composed) WCS can re-derive it live."""
        if abs(self.dec0) < 1e-6:
            return None
        pole = 90.0 if self.dec0 > 0 else -90.0
        x, y = self.sky_to_pix(self.ra0, pole)
        return float(x), float(y)

    def scale_arcsec_per_px(self):
        return math.sqrt(abs(np.linalg.det(self.cd))) * 3600.0

    # -- transforms -------------------------------------------------------

    def compose_rigid(self, dtheta_deg, dx, dy, pivot=None):
        """Return the WCS after the IMAGE has undergone a rigid transform:
        p' = R(dtheta) @ (p - pivot) + pivot + (dx, dy), pivot defaulting
        to CRPIX. This is the update for a mount move fitted from matched
        star pairs: the sky has not changed, the camera has. The tangent
        point is unchanged; CRPIX moves with the transform and CD absorbs
        the rotation (CD' = CD @ R^-1, so xi = CD(p-CRPIX) is preserved
        for every sky point).

        pivot, dx, dy are in the 0-BASED pixel frame of sky_to_pix /
        fit_rigid. CRPIX is 1-based (FITS), so it is shifted into the
        0-based frame before the transform and back after -- omitting
        that leaves a (R - I)@(1,1) error, ~0.014 px at 0.8 deg, which
        the compose/fit consistency test caught on first run."""
        th = math.radians(dtheta_deg)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        p = np.asarray(self.crpix, dtype=np.float64) - 1.0   # 0-based
        pv = (p.copy() if pivot is None
              else np.asarray(pivot, dtype=np.float64))
        p2 = R @ (p - pv) + pv + np.array([dx, dy], dtype=np.float64)
        return TanWCS(self.ra0, self.dec0, p2[0] + 1.0, p2[1] + 1.0,
                      self.cd @ np.linalg.inv(R))

    def advance_sidereal(self, dt_seconds):
        """WCS of the SAME ground-fixed camera dt seconds later. The sky
        rotates westward about the polar axis, so the camera's tangent
        point advances in RA at the sidereal rate: for a TAN projection
        that is exactly CRVAL1 += rate*dt, everything else untouched
        (xi/eta depend only on ra - ra0). The pole pixel is invariant --
        the property the alignment procedure steers by."""
        return TanWCS((self.ra0 + SIDEREAL_DEG_PER_S * dt_seconds) % 360.0,
                      self.dec0, self.crpix[0], self.crpix[1], self.cd)


def fit_rigid(ref_xy, cur_xy, weights=None):
    """Least-squares rigid transform (rotation + translation, NO scale)
    taking ref points to cur points: cur ~= R @ (ref - cref) + cref + t.
    Kabsch on 2D with optional weights (use star SNR). Returns
    (dtheta_deg, dx, dy, rms) with the rotation pivot at the weighted
    ref centroid -- pass the same pivot to TanWCS.compose_rigid. Needs
    >= 2 points; 4-5 well-spread stars give a stable fit, which is the
    whole reason a 12-star field is 'generous' for tracking even though
    it is thin for blind quad solving."""
    A = np.asarray(ref_xy, dtype=np.float64)
    B = np.asarray(cur_xy, dtype=np.float64)
    if A.shape != B.shape or A.shape[0] < 2:
        raise ValueError("need >= 2 matched pairs of equal count")
    w = (np.ones(len(A)) if weights is None
         else np.asarray(weights, dtype=np.float64))
    w = w / w.sum()
    ca = (A * w[:, None]).sum(0)
    cb = (B * w[:, None]).sum(0)
    Ac = A - ca
    Bc = B - cb
    H = (Ac * w[:, None]).T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])          # forbid reflection: mounts don't mirror
    R = Vt.T @ D @ U.T
    t = cb - ca
    th = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    resid = (Ac @ R.T + cb) - B
    rms = float(np.sqrt(np.mean(np.sum(resid ** 2, axis=1))))
    return th, float(t[0]), float(t[1]), rms, tuple(ca)


# ----------------------------------------------------------------------------
# Catalogue + prediction + gating
# ----------------------------------------------------------------------------

def load_catalog(data_dir, dec0):
    """Load the embedded polar-cap catalogue for the hemisphere the WCS
    points at: catalog_scp.npy (dec0 < 0) or catalog_ncp.npy, produced by
    make_catalog.py -- float32 rows of (ra_deg, dec_deg, vmag). Lives in
    the ROOT data dir, not per-device: the sky is shared. Returns
    (ra, dec, mag) arrays or None if the file isn't there (the prior is
    an upgrade, never a requirement)."""
    name = "catalog_scp.npy" if dec0 < 0 else "catalog_ncp.npy"
    path = os.path.join(data_dir, name)
    try:
        cat = np.load(path)
    except OSError:
        return None
    if cat.ndim != 2 or cat.shape[1] != 3:
        raise ValueError(f"{path}: expected (N, 3) array, got {cat.shape}")
    return cat[:, 0].astype(np.float64), cat[:, 1].astype(np.float64), \
        cat[:, 2].astype(np.float64)


class SkyPrior:
    """An anchored sky model: WCS from one plate solve + the polar-cap
    catalogue + (optionally) the fitted flux response. Immutable by
    convention -- workers read it from another thread, so methods return
    fresh data rather than mutating state; re-arming replaces the whole
    object (assignment is atomic under the GIL, same publication idiom
    the stack uses)."""

    def __init__(self, wcs, t0, catalog, response=None):
        self.wcs0 = wcs
        self.t0 = float(t0)
        self.cat_ra, self.cat_dec, self.cat_mag = catalog
        self.response = response

    def wcs_at(self, t):
        """The anchor advanced to wall-clock t: a ground-fixed camera's
        pointing drifts in RA at the sidereal rate and nothing else
        changes (see TanWCS.advance_sidereal). Mount MOVES are not
        modelled here yet -- a fitted compose_rigid() replaces the
        anchor instead (re-arm on the next solve or fit)."""
        return self.wcs0.advance_sidereal(t - self.t0)

    def predict(self, t, shape, mag_limit=11.0, margin=0.0, max_stars=400):
        """Catalogue stars expected on the sensor at time t.
        shape = (h, w); margin > 0 also returns off-frame stars within
        that many pixels of the edge (the 'about to arrive' set a
        steering UI wants). Returns dict of arrays x, y (0-based), mag,
        flux (expected counts via the response, or None), brightest
        first."""
        h, w = shape
        keep = self.cat_mag <= mag_limit
        wcs = self.wcs_at(t)
        x, y = wcs.sky_to_pix(self.cat_ra[keep], self.cat_dec[keep])
        m = self.cat_mag[keep]
        inb = (np.isfinite(x) & np.isfinite(y)
               & (x >= -margin) & (x < w + margin)
               & (y >= -margin) & (y < h + margin))
        ra_k = self.cat_ra[keep][inb]
        dec_k = self.cat_dec[keep][inb]
        x, y, m = x[inb], y[inb], m[inb]
        order = np.argsort(m)[:max_stars]
        x, y, m = x[order], y[order], m[order]
        ra_k, dec_k = ra_k[order], dec_k[order]
        flux = (flux_expected(self.response, m)
                if self.response is not None else None)
        return {"x": x, "y": y, "mag": m, "flux": flux,
                "ra": ra_k, "dec": dec_k}


def _trimmed_rigid_fit(P, D, weights, min_pairs=4, max_rms=3.0):
    """Shared core of refit_from_matches / refit_pointing: fit_rigid with
    iterative outlier rejection. A wide-gate re-acquisition (or a greedy
    frame-to-frame nearest-neighbour match) inevitably mismatches some
    pairs, and one bad pair drags a least-squares rigid fit arbitrarily
    far. Fit, drop pairs whose residual exceeds 3x the median residual,
    refit -- up to twice. The consensus (the real move) survives; the
    mismatches don't. Returns (dtheta_deg, dx, dy, rms, pivot, keep_mask)
    or None if it never converges under max_rms (or trims below
    min_pairs)."""
    keep = np.ones(len(P), dtype=bool)
    dth = dx = dy = rms = pivot = None
    for _ in range(3):
        if keep.sum() < min_pairs:
            return None
        dth, dx, dy, rms, pivot = fit_rigid(P[keep], D[keep],
                                            weights=(weights[keep]
                                                    if weights is not None
                                                    else None))
        if rms <= max_rms:
            break
        th = math.radians(dth)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        pv = np.asarray(pivot)
        res = np.hypot(*(((P - pv) @ R.T + pv
                          + np.array([dx, dy])) - D).T)
        cut = max(3.0 * float(np.median(res[keep])), 2.0)
        new_keep = keep & (res <= cut)
        if new_keep.sum() == keep.sum():
            break                      # nothing left to trim; verdict below
        keep = new_keep
    if rms > max_rms:
        return None
    return dth, dx, dy, rms, pivot, keep


def refit_from_matches(prior, dets, pred, matched, t,
                       min_pairs=4, max_rms=3.0):
    """Phase-4 core: turn a burst's matched (prediction, detection) pairs
    into an updated prior. The residual transform predicted->observed is
    fitted as rigid (rotation + translation -- a mount move or flex; no
    scale, optics don't breathe) and composed onto the prior advanced to
    t, which becomes the new anchor. Returns (new_prior, fit_info) or
    (None, reason): the caller keeps the old prior when the fit is thin
    (< min_pairs) or inconsistent (rms > max_rms px -- mismatched stars,
    not a rigid move; trust nothing)."""
    if len(matched) < min_pairs:
        return None, f"only {len(matched)} matched pair(s)"
    P = np.array([[pred["x"][pi], pred["y"][pi]] for _, pi, _, _ in matched])
    D = np.array([[dets[di]["x"], dets[di]["y"]] for di, _, _, _ in matched])
    wts = np.sqrt(np.array([max(dets[di].get("flux", 1.0), 1e-3)
                            for di, _, _, _ in matched]))
    fit = _trimmed_rigid_fit(P, D, wts, min_pairs, max_rms)
    if fit is None:
        return None, f"fit rms exceeds {max_rms}px after trimming"
    dth, dx, dy, rms, pivot, keep = fit
    matched = [m for m, k in zip(matched, keep) if k]
    wcs_now = prior.wcs_at(t).compose_rigid(dth, dx, dy, pivot=pivot)
    new = SkyPrior(wcs_now, t, (prior.cat_ra, prior.cat_dec,
                                prior.cat_mag), response=prior.response)
    info = {"n": len(matched), "dtheta_deg": round(dth, 5),
            "dx": round(dx, 3), "dy": round(dy, 3),
            "rms": round(rms, 3)}
    return new, info


def match_stars(ref_stars, cur_stars, max_dist=80.0):
    """Greedy nearest-neighbour matching of current star detections to a
    reference list (dicts just need x/y). Returns [(rx, ry, cx, cy), ...]
    for matched pairs -- no catalogue, no WCS: pure frame-to-frame pixel
    correspondence, used both for the paused-mode drift overlay and for
    refit_pointing's catalogue-free tracking."""
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


class PointingAnchor:
    """A WCS anchor for a camera with no catalogue coverage (make_catalog.py
    only covers the polar caps) -- kept current by folding frame-to-frame
    star correspondence into a rigid refit each burst instead of catalogue
    predictions, via refit_pointing() below. Same wcs_at()/sidereal-advance
    contract as SkyPrior.wcs_at, minus the catalogue."""

    def __init__(self, wcs, t0):
        self.wcs0 = wcs
        self.t0 = float(t0)

    def wcs_at(self, t):
        return self.wcs0.advance_sidereal(t - self.t0)


def refit_pointing(anchor, ref_stars, cur_stars, t, max_dist=80.0,
                   min_pairs=4, max_rms=3.0):
    """Catalogue-free anchor refit: match this burst's confirmed stars
    against the previous burst's by nearest pixel neighbour (match_stars)
    instead of catalogue predictions, fit_rigid the pairs (with the same
    trimmed outlier rejection refit_from_matches uses), and compose the
    result onto the anchor advanced to t. Works anywhere in the sky --
    the whole point, for a camera pointed off the polar-cap catalogue.
    Returns (new_anchor, fit_info) or (None, reason), same contract as
    refit_from_matches."""
    pairs = match_stars(ref_stars, cur_stars, max_dist=max_dist)
    if len(pairs) < min_pairs:
        return None, f"only {len(pairs)} matched pair(s)"
    P = np.array([[rx, ry] for rx, ry, cx, cy in pairs])
    D = np.array([[cx, cy] for rx, ry, cx, cy in pairs])
    fit = _trimmed_rigid_fit(P, D, None, min_pairs, max_rms)
    if fit is None:
        return None, f"fit rms exceeds {max_rms}px after trimming"
    dth, dx, dy, rms, pivot, keep = fit
    wcs_now = anchor.wcs_at(t).compose_rigid(dth, dx, dy, pivot=pivot)
    new_anchor = PointingAnchor(wcs_now, t)
    info = {"n": int(keep.sum()), "dtheta_deg": round(dth, 5),
            "dx": round(dx, 3), "dy": round(dy, 3), "rms": round(rms, 3)}
    return new_anchor, info


def gate_stars(dets, pred, gate_px=8.0, max_dex=None):
    """Match detected stars against predictions: greedy, brightest
    prediction first (a faint neighbour can't steal a bright star's
    detection -- same policy as StarTracker). A detection matches only
    if it falls within gate_px of a predicted position AND, when a flux
    model is available and max_dex given, within max_dex of the
    predicted log-flux (use ~3x the response fit's rms).

    dets: list of dicts with x/y/flux (the tool's star dicts).
    pred: predict() output. Returns (matched, unexpected, missed):
    matched = [(det_idx, pred_idx, dist_px, dex_err-or-None), ...];
    unexpected = det indices matching nothing (transient candidates);
    missed = pred indices with no detection (clouds, or the model
    over-promising)."""
    n_pred = len(pred["x"])
    used_d = set()
    matched = []
    for pi in range(n_pred):            # already brightest-first
        best_j, best_d = None, gate_px
        for j, s in enumerate(dets):
            if j in used_d:
                continue
            d = math.hypot(s["x"] - pred["x"][pi], s["y"] - pred["y"][pi])
            if d < best_d:
                best_d, best_j = d, j
        if best_j is None:
            continue
        dex = None
        if pred["flux"] is not None and max_dex is not None:
            f = dets[best_j].get("flux", 0.0)
            if f <= 0:
                continue
            dex = math.log10(f / float(pred["flux"][pi]))
            if abs(dex) > max_dex:
                continue            # right place, wrong brightness: refuse
        used_d.add(best_j)
        matched.append((best_j, pi, best_d,
                        round(dex, 3) if dex is not None else None))
    unexpected = [j for j in range(len(dets)) if j not in used_d]
    missed = [pi for pi in range(n_pred)
              if pi not in {m[1] for m in matched}]
    return matched, unexpected, missed


# ----------------------------------------------------------------------------
# RA-axis calibration (Phase 5)
# ----------------------------------------------------------------------------

def pixel_transform_between(wcs_a, wcs_b, shape, n=5):
    """Rigid transform taking image-A pixels to image-B pixels of the
    SAME sky: sample a grid across the frame, map A-pixel -> sky ->
    B-pixel, fit_rigid the pairs. rms reports how rigid the mapping
    really is over the frame (TAN reprojection between nearby pointings
    is not exactly rigid; at polar-scope scales the residual is tiny and
    the rms says so rather than assuming so)."""
    h, w = shape
    xs = np.linspace(0.05 * w, 0.95 * w, n)
    ys = np.linspace(0.05 * h, 0.95 * h, n)
    gx, gy = [a.ravel() for a in np.meshgrid(xs, ys)]
    ra, dec = wcs_a.pix_to_sky(gx, gy)
    bx, by = wcs_b.sky_to_pix(ra, dec)
    keep = np.isfinite(bx) & np.isfinite(by)
    return fit_rigid(np.column_stack([gx[keep], gy[keep]]),
                     np.column_stack([bx[keep], by[keep]]))


def fixed_point(dtheta_deg, dx, dy, pivot):
    """The pixel invariant under p' = R(p - pivot) + pivot + t: solve
    (I - R) p = (I - R) pivot + t. Near-zero rotation puts the fixed
    point at infinity (a pure translation has none) -- returns None
    below ~0.5 deg rather than an enormous garbage coordinate."""
    if abs(dtheta_deg) < 0.5:
        return None
    th = math.radians(dtheta_deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th), math.cos(th)]])
    A = np.eye(2) - R
    b = A @ np.asarray(pivot, dtype=np.float64) + np.array([dx, dy])
    p = np.linalg.solve(A, b)
    return float(p[0]), float(p[1])


def estimate_axis(wcs_a, wcs_b, shape, min_rot_deg=15.0):
    """RA-axis pixel from two solves either side of an RA rotation: the
    camera rotates with the shaft, so the pixel fixed between the two
    frames IS the axis projection. This retires the polar scope's
    bore-sight assumption -- however well or badly the optics are
    centred, the fixed point is the mechanical truth. Returns
    (axis_xy, rot_deg, rms) or (None, rot_deg, reason): small rotations
    put the fixed point far away and error-amplified, so anything under
    min_rot_deg is refused (rotate ~90 deg for a good conditioning)."""
    dth, dx, dy, rms, pivot = pixel_transform_between(wcs_a, wcs_b, shape)
    if abs(dth) < min_rot_deg:
        return None, dth, f"rotation {dth:.1f} deg < {min_rot_deg} deg"
    p = fixed_point(dth, dx, dy, pivot)
    if p is None:
        return None, dth, "degenerate rotation"
    return p, dth, rms


def axis_path(data_dir, w, h):
    return os.path.join(data_dir, f"axis_{w}x{h}.json")


def save_axis(axis_xy, rot_deg, rms, data_dir, w, h):
    """Per-device+resolution, beside dark/flat/response: the axis is a
    property of how THIS camera sits in THIS mount's bore."""
    os.makedirs(data_dir, exist_ok=True)
    p = axis_path(data_dir, w, h)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"axis_xy": [float(axis_xy[0]), float(axis_xy[1])],
                   "rot_deg": float(rot_deg), "fit_rms_px": float(rms),
                   "calibrated_at":
                       datetime.now(timezone.utc).isoformat(),
                   "skymodel_version": SKYMODEL_VERSION}, fh, indent=2)
    return p


def load_axis(data_dir, w, h):
    try:
        with open(axis_path(data_dir, w, h), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Generative frame model + expected motion (Phase 7)
# ----------------------------------------------------------------------------

def synthesize_stars(pred, shape, fwhm_px=2.5, floor=None):
    """Expected star image from a predict() output: each star a circular
    Gaussian of the response-predicted total flux (integral, not peak).
    Everything else the camera adds -- dark, flat, row banding -- belongs
    to the caller's calibration layers; this is only the sky term.
    floor, when given, clips each star's rendered flux at that value
    (the measured clamp floor: the bridge will not report less)."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.float64)
    if pred["flux"] is None:
        raise ValueError("synthesize_stars needs a fitted flux response")
    sig = fwhm_px / 2.355
    r = max(3, int(math.ceil(4.0 * sig)))
    for x, y, f in zip(pred["x"], pred["y"], pred["flux"]):
        if floor is not None:
            f = max(float(f), float(floor))
        x0, x1 = int(x) - r, int(x) + r + 1
        y0, y1 = int(y) - r, int(y) + r + 1
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x1), min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        yy = np.arange(cy0, cy1)[:, None] - y
        xx = np.arange(cx0, cx1)[None, :] - x
        g = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sig * sig))
        img[cy0:cy1, cx0:cx1] += float(f) * g / (2.0 * math.pi * sig * sig)
    return img


def expected_velocity(prior, t, x, y, dt=60.0):
    """Pixel velocity (vx, vy, px/s) a STATIC sky point at (x, y) shows
    at time t -- computed by round-tripping through the prior itself
    (pix -> sky at t, sky -> pix at t+dt), so parity, orientation and
    scale all come from the WCS with no hand-derived signs to get
    backwards. This is the reference every motion classification
    compares against."""
    ra, dec = prior.wcs_at(t).pix_to_sky(x, y)
    x2, y2 = prior.wcs_at(t + dt).sky_to_pix(ra, dec)
    return (float(x2) - float(x)) / dt, (float(y2) - float(y)) / dt


def classify_track(prior, t, x, y, vx, vy):
    """Sort a confirmed track into one of three populations against the
    prior's velocity field:

      static    -- not moving: sensor artifact (hot pixel cluster,
                   stuck banding feature) masquerading as a source
      sidereal  -- moving WITH the sky: a real fixed object the gate
                   didn't identify (uncatalogued, variable, or the
                   photometry gate refused it)
      transient -- anything else: satellite, aircraft, meteor, bird...
                   or unexplained. The interesting bin.

    Thresholds are relative to the local sidereal speed where possible
    (near the pole that speed approaches zero, so an absolute floor of
    0.02 px/s keeps the static test meaningful there)."""
    sx, sy = expected_velocity(prior, t, x, y)
    v_sid = math.hypot(sx, sy)
    v_trk = math.hypot(vx, vy)
    floor = max(0.25 * v_sid, 0.02)
    if v_trk < floor:
        return "static", v_sid
    if math.hypot(vx - sx, vy - sy) <= max(0.5 * v_sid, 0.03):
        return "sidereal", v_sid
    return "transient", v_sid


# ----------------------------------------------------------------------------
# Row-structured background (Sunplus banding) with an exact star mask
# ----------------------------------------------------------------------------

def row_background(img, star_xy=None, star_rad=6.0, min_frac=0.25):
    """Per-row robust background and sigma with KNOWN star positions
    masked out. The bridge's banding is row-correlated, so a per-row
    median removes it far better than a global background -- but a blind
    row median gets biased upward wherever a star crosses the row, biting
    flux out of exactly the faint stars being dug for. With a predicted
    star list the mask is exact, including stars below single-frame
    detectability. Rows left with fewer than min_frac unmasked pixels
    fall back to the global estimate rather than trusting a handful of
    survivors. Returns (bg[h], sigma[h])."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    masked = np.ma.masked_array(img, mask=np.zeros(img.shape, dtype=bool))
    if star_xy is not None and len(star_xy):
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        for sx, sy in star_xy:
            r = int(math.ceil(star_rad))
            y0, y1 = max(0, int(sy) - r), min(h, int(sy) + r + 1)
            x0, x1 = max(0, int(sx) - r), min(w, int(sx) + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            sub = ((yy[y0:y1] - sy) ** 2
                   + (xx[:, x0:x1] - sx) ** 2) <= star_rad ** 2
            masked.mask[y0:y1, x0:x1] |= sub
    bg = np.ma.median(masked, axis=1)
    mad = np.ma.median(np.ma.abs(masked - bg[:, None]), axis=1)
    sigma = 1.4826 * mad
    ok = (~masked.mask).sum(axis=1) >= max(8, int(min_frac * w))
    gmed = float(np.ma.median(masked))
    gsig = 1.4826 * float(np.ma.median(np.ma.abs(masked - gmed)))
    bg = np.where(ok, np.ma.filled(bg, gmed), gmed)
    sigma = np.where(ok & (sigma > 0), np.ma.filled(sigma, gsig), gsig)
    return bg, sigma


def subtract_row_background(img, star_xy=None, star_rad=6.0):
    """img minus its row_background(), clipped at zero -- the banding-
    aware pre-processing step for extraction and stacking."""
    bg, _ = row_background(img, star_xy, star_rad)
    return np.clip(np.asarray(img, dtype=np.float64) - bg[:, None],
                   0.0, None)


# ----------------------------------------------------------------------------
# Star names near the poles
# ----------------------------------------------------------------------------

# Built-in entries are limited to positions this project has verified
# (iota Oct straight from the corr index matches; sigma Oct is canonical).
# Everything else belongs in star_names.json in the root data dir --
# {"name": [ra_deg, dec_deg], ...} -- where the user can add the local
# asterism (BQ Oct, HD 99685, HD 98784, ...) from a source they trust
# rather than from anything hard-coded on faith.
BUILTIN_NAMES = {
    "sigma Oct": (317.195, -88.9565),
    "iota Oct": (193.74275, -85.12345),
}


def load_star_names(data_dir):
    """BUILTIN_NAMES merged under star_names.json (file wins)."""
    names = dict(BUILTIN_NAMES)
    try:
        with open(os.path.join(data_dir, "star_names.json"),
                  encoding="utf-8") as fh:
            for k, v in json.load(fh).items():
                names[str(k)] = (float(v[0]), float(v[1]))
    except (OSError, ValueError, TypeError, IndexError):
        pass
    return names


def nearest_name(names, ra, dec, tol_arcmin=3.0):
    """Name of the entry within tol of (ra, dec), else None. Angular
    separation with the cos(dec) factor -- near the pole an RA degree is
    almost nothing on the sky, which is exactly where this is used."""
    best, best_d = None, tol_arcmin / 60.0
    cd = math.cos(math.radians(dec))
    for name, (nra, ndec) in names.items():
        dra = abs((ra - nra + 180.0) % 360.0 - 180.0) * cd
        d = math.hypot(dra, dec - ndec)
        if d < best_d:
            best, best_d = name, d
    return best


# ----------------------------------------------------------------------------
# Collimation readout
# ----------------------------------------------------------------------------

def collimation(axis_xy, shape, arcsec_per_px):
    """How far the mount's RA axis pierces the sensor from its centre --
    the bore-sight error of the scope+camera assembly in the mount,
    which is what shimming the collar / adjusting the scope's seating
    changes. Because the camera rotates WITH the RA shaft, a frame
    direction is a fixed direction on the scope body, so the clock
    position is directly actionable at the workbench.

    Convention (stated because clock-face conventions are where
    collimation instructions go to die): 12 o'clock is the TOP of the
    displayed frame (-y), and the clock runs clockwise as seen on
    screen. Returns {"offset_px", "arcmin", "clock", "angle_deg"};
    angle_deg is measured clockwise from 12."""
    h, w = shape
    dx = float(axis_xy[0]) - (w - 1) / 2.0
    dy = float(axis_xy[1]) - (h - 1) / 2.0
    off = math.hypot(dx, dy)
    # screen y grows downward: up is -dy. Clockwise from 12: right (+x)
    # is 3 o'clock -> angle = atan2(dx, -dy)
    ang = math.degrees(math.atan2(dx, -dy)) % 360.0
    clock = int(round(ang / 30.0)) % 12 or 12
    return {"offset_px": round(off, 1),
            "arcmin": round(off * arcsec_per_px / 60.0, 2),
            "clock": clock,
            "angle_deg": round(ang, 1)}


# ----------------------------------------------------------------------------
# Bolt-vector auto-calibration
# ----------------------------------------------------------------------------

def cluster_move_axes(moves, min_len_px=10.0, angle_tol_deg=15.0,
                      min_samples=3, min_sep_deg=40.0):
    """The two bolt directions, learned from the moves the tracker has
    already measured: each significant refit displacement is a sample of
    whichever bolt was being turned (users turn one at a time -- the same
    discipline the guidance asks for). Fold vectors to a half-plane
    (a bolt turned either way is the same axis), cluster by angle, and
    when two well-separated clusters have enough samples each, their mean
    directions are the az/alt unit vectors -- tripod tilt and mechanical
    quirks included, no site geometry, no signs to get backwards.
    Returns (u_vec, v_vec) or None while the evidence is thin."""
    dirs = []
    for dx, dy in moves:
        n = math.hypot(dx, dy)
        if n < min_len_px:
            continue
        vx, vy = dx / n, dy / n
        if vy < 0 or (vy == 0 and vx < 0):
            vx, vy = -vx, -vy
        dirs.append((math.degrees(math.atan2(vy, vx)), vx, vy))
    if len(dirs) < 2 * min_samples:
        return None
    clusters = []
    used = [False] * len(dirs)
    for i, (a0, _, _) in enumerate(dirs):
        if used[i]:
            continue
        members = [j for j, (a, _, _) in enumerate(dirs)
                   if not used[j]
                   and abs((a - a0 + 90.0) % 180.0 - 90.0) <= angle_tol_deg]
        for j in members:
            used[j] = True
        clusters.append(members)
    clusters.sort(key=len, reverse=True)
    if len(clusters) < 2 or len(clusters[0]) < min_samples \
            or len(clusters[1]) < min_samples:
        return None

    def mean_dir(members):
        sx = sum(dirs[j][1] for j in members)
        sy = sum(dirs[j][2] for j in members)
        n = math.hypot(sx, sy) or 1.0
        return (sx / n, sy / n)

    u = mean_dir(clusters[0])
    v = mean_dir(clusters[1])
    ang = math.degrees(math.acos(max(-1.0, min(1.0,
                                               u[0] * v[0] + u[1] * v[1]))))
    ang = min(ang, 180.0 - ang)
    if ang < min_sep_deg:
        return None                 # both clusters are the same bolt
    return u, v


def bolts_path(data_dir, w, h):
    return os.path.join(data_dir, f"bolts_{w}x{h}.json")


def save_bolts(u, v, n_samples, data_dir, w, h):
    os.makedirs(data_dir, exist_ok=True)
    p = bolts_path(data_dir, w, h)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"u_vec": [float(u[0]), float(u[1])],
                   "v_vec": [float(v[0]), float(v[1])],
                   "n_samples": int(n_samples),
                   "calibrated_at":
                       datetime.now(timezone.utc).isoformat(),
                   "note": "u/v are pixel-space unit vectors of the two "
                           "mount adjustment axes, learned from tracked "
                           "moves; which is az and which is alt is for "
                           "the user to confirm once",
                   "skymodel_version": SKYMODEL_VERSION}, fh, indent=2)
    return p


def load_bolts(data_dir, w, h):
    try:
        with open(bolts_path(data_dir, w, h), encoding="utf-8") as fh:
            d = json.load(fh)
        return (d["u_vec"], d["v_vec"])
    except (OSError, ValueError, KeyError):
        return None


def bolt_correction(axis_xy, target_xy, arcsec_per_px, u_vec=None,
                    v_vec=None):
    """Decompose the axis-pixel -> target-pixel correction vector onto the
    two mount adjustment bolts (u_vec/v_vec, empirically calibrated by
    cluster_move_axes) -- the immediate version of plan_hops: no route,
    no star count, just "turn THIS bolt THIS much, then THIS one," for
    when the axis is already close enough that both bolts' fields of
    view overlap it. Falls back to the frame's own x/y axes (labelled
    accordingly) before the bolts are calibrated, same convention
    plan_hops uses for its legs.

    The +/- sign is only meaningful relative to whichever physical turn
    direction cluster_move_axes happened to fold to "positive" for that
    bolt -- pixel motion alone can't say which way is clockwise, so this
    is exactly as directionally honest as the star-hop legs already are:
    turn a little, and if the number grows, reverse.

    Returns {"az": leg, "alt": leg, "calibrated": bool}, each leg a dict
    of {"px", "arcmin", "label"}."""
    d = np.array([target_xy[0] - axis_xy[0], target_xy[1] - axis_xy[1]],
                dtype=np.float64)
    calibrated = u_vec is not None and v_vec is not None
    U = (np.asarray(u_vec, dtype=np.float64) if calibrated
        else np.array([1.0, 0.0]))
    V = (np.asarray(v_vec, dtype=np.float64) if calibrated
        else np.array([0.0, 1.0]))
    U = U / np.linalg.norm(U)
    V = V / np.linalg.norm(V)
    coeffs = np.linalg.solve(np.column_stack([U, V]), d)
    scale = arcsec_per_px / 60.0

    def leg(px, label):
        return {"px": round(float(px), 1),
                "arcmin": round(float(px) * scale, 2),
                "label": label}

    return {"az": leg(coeffs[0], "az" if calibrated else "frame-x"),
            "alt": leg(coeffs[1], "alt" if calibrated else "frame-y"),
            "calibrated": calibrated}


# ----------------------------------------------------------------------------
# Atmospheric refraction
# ----------------------------------------------------------------------------

def refraction_arcmin(true_alt_deg, pressure_hpa=1010.0, temp_c=10.0):
    """Atmospheric refraction, in arcminutes, at a given TRUE (airless)
    altitude: light bends toward the zenith as it enters the atmosphere,
    so an object appears HIGHER than it truly is by this amount --
    apparent_altitude = true_altitude + refraction_arcmin(true_altitude).
    Saemundsson's formula (1986) -- the true-altitude form, as opposed to
    Bennett's apparent-altitude form used for the reverse correction --
    accurate to about 0.1' from the horizon to the zenith for the
    standard atmosphere (1010 hPa, 10 deg C); pressure/temperature scale
    the result the usual first-order way (refraction ~ pressure,
    ~ 1/absolute-temperature). Zero at the zenith, ~29' at the horizon
    (true horizon -- this is NOT the ~34' figure quoted for the horizon
    in Bennett's apparent-altitude form; the two formulas take different
    inputs and are only approximate inverses of each other, a known
    property of both, not a bug here). Clamped to >= 0: right at the
    zenith the formula's own +10.3/(h+5.11) offset pushes its tan()
    argument a hair past 90 deg, which would otherwise flip the sign for
    a fraction of an arcsecond -- physically refraction never reverses,
    so 0 is the right floor, not a real negative reading."""
    alt = max(0.0, min(90.0, true_alt_deg))
    r = 1.02 / math.tan(math.radians(alt + 10.3 / (alt + 5.11)))
    return max(0.0, r * (pressure_hpa / 1010.0) * (283.0 / (273.0 + temp_c)))


# ----------------------------------------------------------------------------
# Star-hop route planning (Phase 6): no leaps of faith
# ----------------------------------------------------------------------------

def mag_limit_from_noise(response, sigma, nsigma=5.0, fwhm_px=2.5):
    """Tonight's limiting magnitude, from tonight's measured noise. A
    matched-filter detection of a Gaussian star of FWHM f in per-pixel
    noise sigma needs total flux >= nsigma * sigma * 2*sqrt(pi)*sig_psf
    (the effective noise footprint of the PSF). Fold that through the
    fitted response (inverted numerically -- it is monotone by
    construction) and the answer is conditions-aware: high cloud or a
    bright moon raises sigma and the planner routes more conservatively
    with no separate configuration. Clamped to the response's fitted
    magnitude range: extrapolating a clamp curve invites optimism."""
    sig_psf = fwhm_px / 2.355
    flux_min = nsigma * sigma * 2.0 * math.sqrt(math.pi) * sig_psf
    lo, hi = response["mag_range"]
    grid = np.linspace(lo, hi, 256)
    flux = flux_expected(response, grid)
    idx = np.searchsorted(-flux, -flux_min)     # flux decreasing in mag
    if idx <= 0:
        return float(lo)
    if idx >= len(grid):
        return float(hi)
    return float(grid[idx - 1])


def plan_hops(prior, t, shape, mag_limit, u_vec=None, v_vec=None,
              min_stars=5, step_frac=0.45, margin_frac=0.1,
              max_extent_frames=6.0):
    """Star-hop route from the current pointing to the pole pixel:
    a sequence of single-bolt legs, each of which keeps at least
    min_stars detectable catalogue stars in frame the whole way --
    the star-hopper's rule that you never let go of one handhold
    before the next is in reach.

    u_vec / v_vec are the measured bolt direction unit vectors in pixel
    space (azimuth, altitude), from empirical calibration; when absent
    the frame axes are used and the caller should present directions as
    frame-relative rather than bolt-relative. Legs alternate along the
    two vectors only (a diagonal is not a thing a human with two bolts
    can turn).

    The map: catalogue stars brighter than mag_limit are projected ONCE
    with the prior at t; a hypothetical frame centred at offset c
    contains the stars within half a frame (less margin_frac) of c.
    Dijkstra over a half-frame-step grid, cost = distance * (1 + crowd
    penalty), with cells under min_stars passable at a large penalty
    instead of forbidden -- when a void ring surrounds the pole the
    right answer is the SHORTEST blind crossing, clearly labelled, not
    a refusal.

    Returns {"reachable": bool, "legs": [...], "blind_px": float}:
    each leg carries the bolt vector multiple, the pixel displacement,
    the star count at the destination, blind flag, and an anchor -- the
    brightest star in the destination frame with its arrival pixel, the
    'turn until THIS star sits THERE' instruction."""
    h, w = shape
    wcs = prior.wcs_at(t)
    keep = prior.cat_mag <= mag_limit
    sx, sy = wcs.sky_to_pix(prior.cat_ra[keep], prior.cat_dec[keep])
    sm_ = prior.cat_mag[keep]
    fin = np.isfinite(sx) & np.isfinite(sy)
    sx, sy, sm_ = sx[fin], sy[fin], sm_[fin]

    pole = wcs.pole_pixel()
    if pole is None:
        return {"reachable": False, "legs": [], "blind_px": 0.0,
                "reason": "no pole in this projection"}
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0     # current frame centre
    target = np.array([pole[0] - cx0, pole[1] - cy0])  # offset to walk

    U = (np.asarray(u_vec, dtype=np.float64) if u_vec is not None
         else np.array([1.0, 0.0]))
    V = (np.asarray(v_vec, dtype=np.float64) if v_vec is not None
         else np.array([0.0, 1.0]))
    U = U / np.linalg.norm(U)
    V = V / np.linalg.norm(V)
    B = np.column_stack([U, V])
    Binv = np.linalg.inv(B)                     # pixel -> bolt coords

    hw = (0.5 - margin_frac) * w
    hh = (0.5 - margin_frac) * h

    def stars_at(off):
        """Count + brightest star of a frame centred at pixel offset."""
        inx = np.abs(sx - (cx0 + off[0])) < hw
        iny = np.abs(sy - (cy0 + off[1])) < hh
        idx = np.nonzero(inx & iny)[0]
        if not len(idx):
            return 0, None
        bi = idx[np.argmin(sm_[idx])]
        return len(idx), (float(sx[bi]), float(sy[bi]), float(sm_[bi]))

    # grid in bolt coordinates, half-frame steps: adjacent cells always
    # share most of a frame, so identity tracking never drops across a
    # step even before considering star counts
    step = step_frac * min(w, h)
    tgt_b = Binv @ target
    ext = int(math.ceil(max_extent_frames * min(w, h) / step))
    goal = (int(round(tgt_b[0] / step)), int(round(tgt_b[1] / step)))

    import heapq
    BLIND = 40.0                     # cost multiplier for starless cells

    def cell_off(ij):
        return B @ (np.array(ij, dtype=np.float64) * step)

    def cell_cost(ij):
        n, _ = stars_at(cell_off(ij))
        if n >= min_stars:
            return 1.0 + 2.0 / (1 + n - min_stars), False
        return BLIND, True

    start = (0, 0)
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    seen = set()
    while pq:
        d, ij = heapq.heappop(pq)
        if ij in seen:
            continue
        seen.add(ij)
        if ij == goal:
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nij = (ij[0] + di, ij[1] + dj)
            if not (min(0, goal[0]) - ext // 2 <= nij[0]
                    <= max(0, goal[0]) + ext // 2
                    and min(0, goal[1]) - ext // 2 <= nij[1]
                    <= max(0, goal[1]) + ext // 2):
                continue
            c, _ = cell_cost(nij)
            nd = d + step * c
            if nd < dist.get(nij, float("inf")):
                dist[nij] = nd
                prev[nij] = ij
                heapq.heappush(pq, (nd, nij))
    if goal not in prev and goal != start:
        return {"reachable": False, "legs": [], "blind_px": 0.0,
                "reason": "no route found inside the search extent"}

    # walk back; merge consecutive steps into legs only while BOTH the
    # direction and the blindness of the traversed cells agree -- a leg
    # that passes through a starless cell must say so, not hide it
    # behind a well-populated endpoint (the first version did exactly
    # that, and the walled-void test caught it)
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    cell_blind = {ij: (stars_at(cell_off(ij))[0] < min_stars)
                  for ij in path}
    legs = []
    blind_px = 0.0
    k = 1
    while k < len(path):
        d0 = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
        b0 = cell_blind[path[k]]
        k2 = k
        while (k2 + 1 < len(path)
               and (path[k2 + 1][0] - path[k2][0],
                    path[k2 + 1][1] - path[k2][1]) == d0
               and cell_blind[path[k2 + 1]] == b0):
            k2 += 1
        end = path[k2]
        off = cell_off(end)
        n, anchor = stars_at(off)
        seg_px = step * (k2 - k + 1)
        if b0:
            blind_px += seg_px
        axis = "az" if d0[0] != 0 else "alt"
        legs.append({
            "bolt": axis if (u_vec is not None) else
            ("frame-x" if d0[0] != 0 else "frame-y"),
            "direction": int(np.sign(d0[0] + d0[1])),
            "steps": k2 - k + 1,
            "move_px": [round(float((off - cell_off(path[k - 1]))[0]), 1),
                        round(float((off - cell_off(path[k - 1]))[1]), 1)],
            "stars_at_end": int(n),
            "blind": bool(b0),
            # anchor coordinates are where the star WILL SIT in the frame
            # once the leg completes ("turn until the mag-X star reaches
            # (x, y)") -- the catalogue position minus the frame's walk
            "anchor": (None if anchor is None else
                       {"x": round(anchor[0] - float(off[0]), 1),
                        "y": round(anchor[1] - float(off[1]), 1),
                        "mag": round(anchor[2], 2)}),
        })
        k = k2 + 1
    return {"reachable": True, "legs": legs,
            "blind_px": round(blind_px, 1),
            "mag_limit": round(float(mag_limit), 2),
            "grid_step_px": round(step, 1)}


# ----------------------------------------------------------------------------
# CLI: inspect corr tables, fit and optionally persist the response
# ----------------------------------------------------------------------------

def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(
        description="corr-table analysis: matched stars + flux response")
    ap.add_argument("corr", nargs="+", help="solve-field .corr file(s)")
    ap.add_argument("--save-response", metavar="DIR",
                    help="write response_WxH.json into DIR "
                         "(per-device data dir)")
    ap.add_argument("--resolution", default="3840x2160",
                    help="WxH label for the saved response "
                         "(default %(default)s)")
    a = ap.parse_args(argv)

    mags, fluxes = [], []
    for path in a.corr:
        d = read_corr(path)
        print(f"\n{path}: {len(d)} matched sources")
        has_phot = ("MAG" in d.dtype.names and "FLUX" in d.dtype.names)
        for r in np.sort(d, order="MAG") if has_phot else d:
            line = (f"  x={r['field_x']:7.1f} y={r['field_y']:7.1f}"
                    f"  RA={r['index_ra']:9.5f} Dec={r['index_dec']:9.5f}")
            if has_phot:
                line += f"  mag={r['MAG']:5.2f} flux={r['FLUX']:8.1f}"
                mags.append(float(r["MAG"]))
                fluxes.append(float(r["FLUX"]))
            print(line)

    if len(mags) >= 4:
        model = fit_response(mags, fluxes)
        print(f"\nresponse fit ({model['kind']}): n={model['n']}, "
              f"rms={model['rms']:.3f} dex, "
              f"mag range {model['mag_range'][0]:.2f}"
              f"..{model['mag_range'][1]:.2f}")
        if model["kind"] == "linear":
            print(f"  log10(flux) = {model['a']:.3f} "
                  f"{model['b']:+.3f}*mag   (linear path would be -0.400)")
        else:
            c = model["c"]
            print(f"  log10(flux) = {c[0]:.3f} {c[1]:+.3f}*x "
                  f"{c[2]:+.4f}*x^2, x = mag - {model['m0']:.2f} "
                  f"(linear rms was {model['linear_rms']:.3f})")
        for m in (6.0, 7.0, 8.0, 9.0, 10.0):
            print(f"  expected flux at mag {m:.0f}: "
                  f"{flux_expected(model, m):8.1f}")
        if a.save_response:
            try:
                w, h = a.resolution.split("x")
                p = save_response(model, a.save_response, int(w), int(h),
                                  sources=a.corr)
                print(f"saved {p}")
            except ValueError:
                print("bad --resolution, expected WxH", file=sys.stderr)
                return 2
    else:
        print("\n(too few photometric points for a response fit)")
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
