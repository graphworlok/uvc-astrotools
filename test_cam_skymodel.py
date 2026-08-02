#!/usr/bin/env python3
"""Tests for cam_skymodel.py. Synthetic-first: the FITS reader is fed a
table this file constructs byte-by-byte (so the reader is tested against
the FORMAT, not against astropy's opinion of it), the WCS against
closed-form identities, and the response fit against data with a known
clamp baked in. Run: python3 test_cam_skymodel.py"""

import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

import cam_skymodel as sm


def _card(key, val, quote=False):
    if quote:
        v = f"'{val}'".ljust(20)
    elif isinstance(val, bool):
        v = ("T" if val else "F").rjust(20)
    else:
        v = str(val).rjust(20)
    return f"{key:<8}= {v}".ljust(80).encode("ascii")


def _pad_block(b):
    return b + b" " * ((-len(b)) % 2880)


def synth_corr_bytes(rows):
    """Minimal but valid FITS: empty primary HDU + one BINTABLE with the
    solve-field corr columns this project consumes."""
    prim = _pad_block(_card("SIMPLE", True) + _card("BITPIX", 8)
                      + _card("NAXIS", 0) + b"END".ljust(80))
    names = ["field_x", "field_y", "index_ra", "index_dec", "MAG", "FLUX"]
    rowlen = 8 * len(names)
    hdr = (_card("XTENSION", "BINTABLE", quote=True)
           + _card("BITPIX", 8) + _card("NAXIS", 2)
           + _card("NAXIS1", rowlen) + _card("NAXIS2", len(rows))
           + _card("PCOUNT", 0) + _card("GCOUNT", 1)
           + _card("TFIELDS", len(names)))
    for i, n in enumerate(names, 1):
        hdr += _card(f"TTYPE{i}", n, quote=True)
        hdr += _card(f"TFORM{i}", "1D", quote=True)
    hdr = _pad_block(hdr + b"END".ljust(80))
    body = b"".join(struct.pack(">6d", *r) for r in rows)
    return prim + hdr + _pad_block(body)


class TestCorrReader(unittest.TestCase):
    def test_roundtrip(self):
        rows = [(2481.2, 1651.9, 193.74275, -85.12345, 5.63, 133.7),
                (3302.5, 968.8, 180.58607, -85.63176, 6.17, 116.1),
                (1571.8, 682.7, 187.25253, -83.80269, 6.63, 44.2)]
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            f.write(synth_corr_bytes(rows))
            path = f.name
        try:
            d = sm.read_corr(path)
            self.assertEqual(len(d), 3)
            self.assertAlmostEqual(float(d["index_ra"][0]), 193.74275)
            self.assertAlmostEqual(float(d["FLUX"][1]), 116.1)
            self.assertAlmostEqual(float(d["field_y"][2]), 682.7)
        finally:
            os.unlink(path)

    def test_rejects_non_corr(self):
        prim = _pad_block(_card("SIMPLE", True) + _card("BITPIX", 8)
                          + _card("NAXIS", 0) + b"END".ljust(80))
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            f.write(prim)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                sm.read_corr(path)
        finally:
            os.unlink(path)


def _wcs_like_polar_scope():
    """WCS matching the real Jan-22 solve: centre (190.849, -84.330),
    3.96 arcsec/px, rotation 64.3 deg E of N, 3840x2160 frame."""
    scale = 3.96 / 3600.0
    th = math.radians(64.3)
    cd = [[-scale * math.cos(th), scale * math.sin(th)],
          [scale * math.sin(th), scale * math.cos(th)]]
    return sm.TanWCS(190.849, -84.330, 1920.5, 1080.5, cd)


class TestTanWCS(unittest.TestCase):
    def test_roundtrip(self):
        w = _wcs_like_polar_scope()
        ras = np.array([193.74275, 180.58607, 187.25253, 190.0])
        decs = np.array([-85.12345, -85.63176, -83.80269, -84.0])
        x, y = w.sky_to_pix(ras, decs)
        r2, d2 = w.pix_to_sky(x, y)
        # sub-milliarcsecond round trip
        self.assertTrue(np.allclose(d2, decs, atol=1e-7))
        dra = (np.asarray(r2) - ras + 180.0) % 360.0 - 180.0
        self.assertTrue(np.all(np.abs(dra * np.cos(np.radians(decs)))
                               < 1e-7))

    def test_tangent_point_maps_to_crpix(self):
        w = _wcs_like_polar_scope()
        x, y = w.sky_to_pix(w.ra0, w.dec0)
        self.assertAlmostEqual(float(x), w.crpix[0] - 1.0, places=9)
        self.assertAlmostEqual(float(y), w.crpix[1] - 1.0, places=9)

    def test_pole_pixel_invariant_under_sidereal(self):
        w = _wcs_like_polar_scope()
        p0 = w.pole_pixel()
        p1 = w.advance_sidereal(3600.0).pole_pixel()
        self.assertAlmostEqual(p0[0], p1[0], places=6)
        self.assertAlmostEqual(p0[1], p1[1], places=6)

    def test_sidereal_moves_stars_not_pole(self):
        w = _wcs_like_polar_scope()
        x0, y0 = w.sky_to_pix(193.74275, -85.12345)   # iota Oct
        w1 = w.advance_sidereal(600.0)
        x1, y1 = w1.sky_to_pix(193.74275, -85.12345)
        moved = math.hypot(float(x1 - x0), float(y1 - y0))
        # 10 min at 2 deg-ish from the pole: tens of pixels, not zero
        self.assertGreater(moved, 5.0)
        # and the arc radius about the pole is preserved
        px, py = w.pole_pixel()
        r0 = math.hypot(float(x0) - px, float(y0) - py)
        p1x, p1y = w1.pole_pixel()
        r1 = math.hypot(float(x1) - p1x, float(y1) - p1y)
        self.assertAlmostEqual(r0, r1, delta=0.01)

    def test_scale(self):
        w = _wcs_like_polar_scope()
        self.assertAlmostEqual(w.scale_arcsec_per_px(), 3.96, places=6)

    def test_compose_rigid_matches_fit(self):
        w = _wcs_like_polar_scope()
        ras = np.array([193.74275, 180.58607, 187.25253, 189.0, 191.5])
        decs = np.array([-85.12345, -85.63176, -83.80269, -84.9, -83.9])
        x0, y0 = w.sky_to_pix(ras, decs)
        # simulate a mount nudge: rotate 0.8 deg about an arbitrary pivot,
        # translate (14, -9) px
        th = math.radians(0.8)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        pv = np.array([500.0, 300.0])
        P0 = np.column_stack([x0, y0])
        P1 = (P0 - pv) @ R.T + pv + np.array([14.0, -9.0])
        dth, dx, dy, rms, pivot = sm.fit_rigid(P0, P1)
        self.assertAlmostEqual(dth, 0.8, places=6)
        self.assertLess(rms, 1e-9)
        w2 = w.compose_rigid(dth, dx, dy, pivot=pivot)
        x2, y2 = w2.sky_to_pix(ras, decs)
        self.assertTrue(np.allclose(x2, P1[:, 0], atol=1e-6))
        self.assertTrue(np.allclose(y2, P1[:, 1], atol=1e-6))


class TestResponse(unittest.TestCase):
    def test_linear_recovery_with_outlier(self):
        rng = np.random.default_rng(1)
        mag = rng.uniform(5.5, 9.5, 40)
        logf = 4.4 - 0.32 * mag + rng.normal(0, 0.02, 40)
        flux = 10 ** logf
        flux[3] *= 25.0                      # one blended/mismatched star
        m = sm.fit_response(mag, flux)
        b = m["b"] if m["kind"] == "linear" else None
        self.assertIsNotNone(b, "outlier should not force the quad model")
        self.assertAlmostEqual(b, -0.32, delta=0.02)

    def test_clamp_compression_selects_quad(self):
        rng = np.random.default_rng(2)
        mag = np.linspace(5.5, 9.5, 60)
        logf = 4.4 - 0.32 * mag
        # black clamp: faint end progressively starved, ~0.25 dex by mag 9.5
        logf -= 0.02 * np.maximum(mag - 6.0, 0.0) ** 2
        flux = 10 ** (logf + rng.normal(0, 0.01, 60))
        m = sm.fit_response(mag, flux)
        self.assertEqual(m["kind"], "quad")
        # model tracks the compressed faint end, not the bright-end line
        f95 = float(sm.flux_expected(m, 9.5))
        truth = 10 ** (4.4 - 0.32 * 9.5 - 0.02 * 3.5 ** 2)
        self.assertAlmostEqual(math.log10(f95), math.log10(truth),
                               delta=0.05)

    def test_save_load(self):
        m = sm.fit_response([5.6, 6.2, 6.6, 7.4, 8.2, 8.6],
                            [134, 116, 44, 12.7, 5.9, 5.5])
        with tempfile.TemporaryDirectory() as d:
            sm.save_response(m, d, 3840, 2160, sources=["a.corr"])
            m2 = sm.load_response(d, 3840, 2160)
        self.assertEqual(m2["kind"], m["kind"])
        self.assertEqual(m2["resolution"], "3840x2160")
        f1 = float(sm.flux_expected(m, 7.0))
        f2 = float(sm.flux_expected(m2, 7.0))
        self.assertAlmostEqual(f1, f2, places=9)


class TestPriorAndGating(unittest.TestCase):
    def _prior(self, response=None):
        w = _wcs_like_polar_scope()
        # synthetic polar-cap catalogue: 60 stars within ~3 deg of centre
        rng = np.random.default_rng(7)
        ra = rng.uniform(160.0, 220.0, 60)
        dec = rng.uniform(-86.5, -82.5, 60)
        mag = rng.uniform(5.5, 10.5, 60)
        return sm.SkyPrior(w, 1000.0, (ra, dec, mag), response=response), w

    def test_predict_in_bounds_and_sorted(self):
        prior, w = self._prior()
        p = prior.predict(1000.0, (2160, 3840))
        self.assertGreater(len(p["x"]), 0)
        self.assertTrue(np.all((p["x"] >= 0) & (p["x"] < 3840)))
        self.assertTrue(np.all((p["y"] >= 0) & (p["y"] < 2160)))
        self.assertTrue(np.all(np.diff(p["mag"]) >= 0))  # brightest first

    def test_predict_advances_with_clock(self):
        prior, _ = self._prior()
        p0 = prior.predict(1000.0, (2160, 3840))
        p1 = prior.predict(1000.0 + 600.0, (2160, 3840))
        # same catalogue, ten sidereal minutes later: positions moved
        self.assertGreater(len(p0["x"]), 0)
        self.assertGreater(len(p1["x"]), 0)
        self.assertFalse(np.allclose(p0["x"][0], p1["x"][0], atol=1.0))

    def test_gate_matches_and_flags(self):
        model = sm.fit_response(np.linspace(5.5, 9.5, 20),
                                10 ** (4.4 - 0.32
                                       * np.linspace(5.5, 9.5, 20)))
        prior, _ = self._prior(response=model)
        p = prior.predict(1000.0, (2160, 3840))
        # detections: first three predictions nudged 1.5px with correct
        # flux, plus one impostor far from any prediction
        dets = []
        for i in range(3):
            dets.append({"x": float(p["x"][i]) + 1.5,
                         "y": float(p["y"][i]),
                         "flux": float(p["flux"][i]) * 1.2})
        dets.append({"x": 10.0, "y": 10.0, "flux": 50.0})
        matched, unexpected, missed = sm.gate_stars(
            dets, p, gate_px=8.0, max_dex=0.36)
        self.assertEqual(len(matched), 3)
        self.assertEqual(unexpected, [3])
        self.assertEqual(len(missed), len(p["x"]) - 3)

    def test_gate_refuses_wrong_flux(self):
        model = sm.fit_response(np.linspace(5.5, 9.5, 20),
                                10 ** (4.4 - 0.32
                                       * np.linspace(5.5, 9.5, 20)))
        prior, _ = self._prior(response=model)
        p = prior.predict(1000.0, (2160, 3840))
        # right place, 30x the predicted flux: a hot pixel or satellite
        # glint sitting on a predicted position must NOT inherit identity
        dets = [{"x": float(p["x"][0]), "y": float(p["y"][0]),
                 "flux": float(p["flux"][0]) * 30.0}]
        matched, unexpected, _ = sm.gate_stars(dets, p, gate_px=8.0,
                                               max_dex=0.36)
        self.assertEqual(matched, [])
        self.assertEqual(unexpected, [0])


class TestRowBackground(unittest.TestCase):
    def test_removes_banding_without_biting_stars(self):
        rng = np.random.default_rng(3)
        h, w = 120, 200
        band = rng.normal(0, 2.0, h)[:, None]        # row-correlated offset
        img = 10.0 + band + rng.normal(0, 1.0, (h, w))
        # a bright star spanning several rows
        yy, xx = np.mgrid[:h, :w]
        star = 80.0 * np.exp(-(((yy - 60) ** 2 + (xx - 100) ** 2)
                               / (2 * 2.5 ** 2)))
        img += star
        bg_blind, _ = sm.row_background(img)
        bg_masked, _ = sm.row_background(img, star_xy=[(100.0, 60.0)],
                                         star_rad=10.0)
        truth = 10.0 + band[:, 0]
        # masked estimate tracks the true banding closely on star rows
        err_masked = abs(bg_masked[60] - truth[60])
        self.assertLess(err_masked, 0.6)
        # and the corrected image preserves the star's peak flux
        corr = sm.subtract_row_background(img, star_xy=[(100.0, 60.0)],
                                          star_rad=10.0)
        self.assertGreater(corr[60, 100], 75.0)

    def test_sparse_row_falls_back_to_global(self):
        img = np.full((20, 30), 5.0)
        # mask out nearly an entire row
        bg, sigma = sm.row_background(
            img, star_xy=[(15.0, 10.0)], star_rad=14.0, min_frac=0.5)
        self.assertAlmostEqual(float(bg[10]), 5.0, places=6)
        self.assertEqual(len(bg), 20)


class TestAxisCalibration(unittest.TestCase):
    def test_recovers_rotation_centre(self):
        w = _wcs_like_polar_scope()
        axis_true = (2600.0, 900.0)          # off-centre, as reality is
        w2 = w.compose_rigid(-90.0, 0.0, 0.0, pivot=axis_true)
        axis, rot, rms = sm.estimate_axis(w, w2, (2160, 3840))
        self.assertIsNotNone(axis)
        self.assertAlmostEqual(rot, -90.0, delta=0.01)
        self.assertLess(abs(axis[0] - axis_true[0]), 0.5)
        self.assertLess(abs(axis[1] - axis_true[1]), 0.5)

    def test_refuses_small_rotation(self):
        w = _wcs_like_polar_scope()
        w2 = w.compose_rigid(3.0, 5.0, -2.0)
        axis, rot, reason = sm.estimate_axis(w, w2, (2160, 3840))
        self.assertIsNone(axis)

    def test_sidereal_compensation_prevents_false_axis(self):
        # an hour idle rotates the field ~15 deg about the POLE; without
        # compensation that would "calibrate" the axis onto the pole
        # pixel. Advancing the reference WCS removes the rotation
        # entirely -- what remains must be below the acceptance gate.
        w = _wcs_like_polar_scope()
        later = w.advance_sidereal(4000.0)     # ~16.7 deg, above the gate
        axis, rot, _ = sm.estimate_axis(w, later, (2160, 3840))
        self.assertIsNotNone(axis, "uncompensated: sidereal rotation "
                             "does look like a rotation (the trap)")
        ref = w.advance_sidereal(4000.0)
        axis2, rot2, _ = sm.estimate_axis(ref, later, (2160, 3840))
        self.assertIsNone(axis2, "compensated reference must show no "
                          "rotation to mistake for the shaft")
        self.assertLess(abs(rot2), 0.01)


class TestRefit(unittest.TestCase):
    def test_absorbs_a_move(self):
        prior_model = sm.fit_response(np.linspace(5.5, 9.5, 20),
                                      10 ** (4.4 - 0.32
                                             * np.linspace(5.5, 9.5, 20)))
        w = _wcs_like_polar_scope()
        rng = np.random.default_rng(11)
        ra = rng.uniform(170.0, 210.0, 80)
        dec = rng.uniform(-86.0, -83.0, 80)
        mag = rng.uniform(5.5, 9.0, 80)
        prior = sm.SkyPrior(w, 0.0, (ra, dec, mag), response=prior_model)
        pred = prior.predict(0.0, (2160, 3840))
        # the mount is nudged: the sky lands 60px right, 40px up, 0.1 deg
        dets = []
        th = math.radians(0.1)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        pv = np.array([1920.0, 1080.0])
        for i in range(min(12, len(pred["x"]))):
            p = R @ (np.array([pred["x"][i], pred["y"][i]]) - pv) + pv \
                + np.array([60.0, -40.0])
            dets.append({"x": float(p[0]), "y": float(p[1]),
                         "flux": float(pred["flux"][i])})
        # narrow gate loses lock; wide gate re-acquires (worker logic)
        m8, _, _ = sm.gate_stars(dets, pred, gate_px=8.0)
        self.assertEqual(len(m8), 0)
        m120, _, _ = sm.gate_stars(dets, pred, gate_px=120.0)
        self.assertGreaterEqual(len(m120), 10)
        new, info = sm.refit_from_matches(prior, dets, pred, m120, 0.0)
        self.assertIsNotNone(new, info)
        self.assertLess(info["rms"], 0.1)
        # after the refit, predictions land back on the detections
        pred2 = new.predict(0.0, (2160, 3840))
        m2, _, _ = sm.gate_stars(dets, pred2, gate_px=2.0)
        self.assertGreaterEqual(len(m2), 10)

    def test_refuses_nonrigid_garbage(self):
        w = _wcs_like_polar_scope()
        rng = np.random.default_rng(13)
        ra = rng.uniform(170.0, 210.0, 40)
        dec = rng.uniform(-86.0, -83.0, 40)
        mag = rng.uniform(5.5, 9.0, 40)
        prior = sm.SkyPrior(w, 0.0, (ra, dec, mag))
        pred = prior.predict(0.0, (2160, 3840))
        n = min(8, len(pred["x"]))
        dets = [{"x": float(pred["x"][i]) + rng.uniform(-40, 40),
                 "y": float(pred["y"][i]) + rng.uniform(-40, 40),
                 "flux": 10.0} for i in range(n)]
        matched = [(i, i, 0.0, None) for i in range(n)]
        new, reason = sm.refit_from_matches(prior, dets, pred, matched,
                                            0.0, max_rms=3.0)
        self.assertIsNone(new, "random scatter must not pass as a move")


class TestSynthesisAndMotion(unittest.TestCase):
    def _prior(self):
        model = sm.fit_response(np.linspace(5.5, 9.5, 20),
                                10 ** (4.4 - 0.32
                                       * np.linspace(5.5, 9.5, 20)))
        w = _wcs_like_polar_scope()
        rng = np.random.default_rng(17)
        ra = rng.uniform(175.0, 205.0, 50)
        dec = rng.uniform(-85.8, -83.2, 50)
        mag = rng.uniform(5.5, 8.0, 50)
        return sm.SkyPrior(w, 0.0, (ra, dec, mag), response=model)

    def test_synthesize_puts_flux_where_predicted(self):
        prior = self._prior()
        pred = prior.predict(0.0, (2160, 3840))
        img = sm.synthesize_stars(pred, (2160, 3840), fwhm_px=2.5)
        # brightest star: the rendered patch integrates to its flux
        x, y, f = pred["x"][0], pred["y"][0], float(pred["flux"][0])
        iy, ix = int(round(float(y))), int(round(float(x)))
        patch = img[max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9]
        self.assertAlmostEqual(float(patch.sum()) / f, 1.0, delta=0.02)

    def test_velocity_field_matches_sidereal_geometry(self):
        prior = self._prior()
        # a point ~2 deg from the pole moves at omega*r along its arc
        px, py = prior.wcs0.pole_pixel()
        x, y = px + 1800.0, py + 30.0     # ~2 deg out at 3.96"/px
        vx, vy = sm.expected_velocity(prior, 0.0, x, y)
        r_px = math.hypot(x - px, y - py)
        omega = 2.0 * math.pi / 86164.0905
        self.assertAlmostEqual(math.hypot(vx, vy), omega * r_px,
                               delta=0.02 * omega * r_px)

    def test_classify_three_populations(self):
        prior = self._prior()
        x, y = 2000.0, 1000.0
        sx, sy = sm.expected_velocity(prior, 0.0, x, y)
        self.assertEqual(sm.classify_track(prior, 0.0, x, y, 0.0, 0.0)[0],
                         "static")
        self.assertEqual(sm.classify_track(prior, 0.0, x, y, sx, sy)[0],
                         "sidereal")
        self.assertEqual(sm.classify_track(prior, 0.0, x, y, 3.0, -2.0)[0],
                         "transient")


def _detect(img, nsigma=5.0):
    """Tiny local-maximum detector for the end-to-end test: threshold on
    a row-subtracted frame, 3x3 peak test, flux-weighted 7x7 centroid.
    Deliberately independent of cam_observe's extractor (which needs
    tkinter at import time)."""
    # the frame arrives zero-clipped (subtract_row_background), which
    # collapses a plain median/MAD to ~0 and floods the peak test with
    # noise maxima -- estimate sigma from the nonzero (half-normal) tail
    nz = img[img > 0]
    med = float(np.median(nz)) if len(nz) else 0.0
    sig = 1.4826 * float(np.median(np.abs(nz - med))) if len(nz) else 1.0
    th = med + nsigma * max(sig, 1e-3)
    core = img[3:-3, 3:-3]
    peak = np.ones_like(core, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            peak &= core >= img[3 + dy:img.shape[0] - 3 + dy,
                                3 + dx:img.shape[1] - 3 + dx]
    ys, xs = np.nonzero(peak & (core > th))
    if len(ys) > 200:              # keep the loop bounded regardless
        order = np.argsort(core[ys, xs])[::-1][:200]
        ys, xs = ys[order], xs[order]
    dets = []
    for y, x in zip(ys + 3, xs + 3):
        p = img[y - 3:y + 4, x - 3:x + 4] - med
        p = np.clip(p, 0, None)
        f = float(p.sum())
        if f <= 0:
            continue
        yy, xx = np.mgrid[y - 3:y + 4, x - 3:x + 4]
        dets.append({"x": float((xx * p).sum() / f),
                     "y": float((yy * p).sum() / f), "flux": f})
    dets.sort(key=lambda s: -s["flux"])
    return dets


class TestEndToEnd(unittest.TestCase):
    """A cloudy-night observing run in software: synthesize what the
    camera WOULD see from a truth WCS (stars + banding + noise), run the
    skymodel pipeline against it, and hold lock through ten minutes of
    sidereal drift and a bolt move."""

    def test_full_night(self):
        rng = np.random.default_rng(23)
        # truth: binned polar-scope geometry, 7.95"/px on 1920x1080
        scale = 7.95 / 3600.0
        th0 = math.radians(64.3)
        cd = [[-scale * math.cos(th0), scale * math.sin(th0)],
              [scale * math.sin(th0), scale * math.cos(th0)]]
        truth0 = sm.TanWCS(190.849, -84.330, 960.5, 540.5, cd)
        ra = rng.uniform(170.0, 212.0, 250)
        dec = rng.uniform(-86.8, -82.0, 250)
        mag = rng.uniform(5.0, 8.5, 250)
        model = sm.fit_response(np.linspace(5.0, 9.0, 30),
                                10 ** (4.4 - 0.32
                                       * np.linspace(5.0, 9.0, 30)))
        shape = (1080, 1920)

        def frame_at(wcs_truth):
            tp = sm.SkyPrior(wcs_truth, 0.0, (ra, dec, mag),
                             response=model)
            pred = tp.predict(0.0, shape, mag_limit=8.5)
            img = sm.synthesize_stars(pred, shape, fwhm_px=2.5)
            img += 10.0 + rng.normal(0, 1.5, shape[0])[:, None]  # banding
            img += rng.normal(0, 0.8, shape)                     # noise
            return img

        # anchor solve at t=0: prior == truth
        prior = sm.SkyPrior(truth0, 0.0, (ra, dec, mag), response=model)
        matched_counts = []
        for burst, t in enumerate([120.0, 300.0, 600.0]):
            truth_t = truth0.advance_sidereal(t)
            img = sm.subtract_row_background(frame_at(truth_t))
            dets = _detect(img)[:60]
            pred = prior.predict(t, shape)
            m, unexp, miss = sm.gate_stars(dets, pred, gate_px=6.0,
                                           max_dex=0.5)
            matched_counts.append(len(m))
            new, info = sm.refit_from_matches(prior, dets, pred, m, t)
            if new is not None:
                self.assertLess(info["rms"], 1.0)
                prior = new
        self.assertTrue(all(c >= 8 for c in matched_counts),
                        f"lock lost during drift: {matched_counts}")

        # bolt move: 0.15 deg rotation + (45, -30)px shift of the CAMERA
        t = 660.0
        truth_moved = truth0.advance_sidereal(t).compose_rigid(
            0.15, 45.0, -30.0, pivot=(400.0, 900.0))
        img = sm.subtract_row_background(frame_at(truth_moved))
        dets = _detect(img)[:60]
        pred = prior.predict(t, shape)
        m_narrow, _, _ = sm.gate_stars(dets, pred, gate_px=6.0,
                                       max_dex=0.5)
        self.assertLess(len(m_narrow), 4, "a 50px move must break the "
                        "narrow gate, else the gate is meaningless")
        m_wide, _, _ = sm.gate_stars(dets, pred, gate_px=120.0)
        self.assertGreaterEqual(len(m_wide), 8)
        prior2, info = sm.refit_from_matches(prior, dets, pred, m_wide, t)
        self.assertIsNotNone(prior2, info)
        # lock restored at the narrow gate on the next burst
        t2 = 720.0
        truth_t2 = truth_moved.advance_sidereal(t2 - t)
        img2 = sm.subtract_row_background(frame_at(truth_t2))
        dets2 = _detect(img2)[:60]
        pred2 = prior2.predict(t2, shape)
        m2, _, _ = sm.gate_stars(dets2, pred2, gate_px=6.0, max_dex=0.5)
        self.assertGreaterEqual(len(m2), 8,
                                "re-acquisition did not restore lock")
        # and the pole pixel tracked the move coherently: truth vs prior
        pt = truth_t2.pole_pixel()
        pp = prior2.pole_pixel() if hasattr(prior2, "pole_pixel") else \
            prior2.wcs_at(t2).pole_pixel()
        self.assertLess(math.hypot(pt[0] - pp[0], pt[1] - pp[1]), 3.0)


class TestHopPlanner(unittest.TestCase):
    """Star-hopping to the pole: routes must detour around voids when a
    detour exists, and minimise -- not refuse -- the blind crossing when
    one doesn't. Catalogues are laid out in PIXEL space (then inverted
    to sky through the WCS) so the voids sit exactly where the test
    wants them."""

    SHAPE = (1080, 1920)

    def _make(self, star_px, pole_offset_px):
        # WCS whose pole lands at frame centre + pole_offset_px
        scale = 7.95 / 3600.0
        cd = [[scale, 0.0], [0.0, scale]]
        w0 = sm.TanWCS(0.0, -84.0, 960.5, 540.5, cd)
        cx, cy = 959.5, 539.5
        px, py = cx + pole_offset_px[0], cy + pole_offset_px[1]
        # choose CRVAL so the pole sits at the desired pixel: cheat by
        # translating the frame instead (rigid, exact)
        p0 = w0.pole_pixel()
        w1 = w0.compose_rigid(0.0, px - p0[0], py - p0[1])
        xs = np.array([s[0] for s in star_px], dtype=np.float64)
        ys = np.array([s[1] for s in star_px], dtype=np.float64)
        ra, dec = w1.pix_to_sky(xs, ys)
        mag = np.full(len(xs), 6.0)
        model = sm.fit_response(np.linspace(5.0, 9.0, 30),
                                10 ** (4.4 - 0.32
                                       * np.linspace(5.0, 9.0, 30)))
        return sm.SkyPrior(w1, 0.0, (ra, dec, mag), response=model)

    @staticmethod
    def _blanket(x0, x1, y0, y1, pitch=250.0):
        return [(x, y) for x in np.arange(x0, x1, pitch)
                for y in np.arange(y0, y1, pitch)]

    def test_direct_route_when_sky_is_rich(self):
        stars = self._blanket(-2000, 6000, -2000, 3200)
        prior = self._make(stars, (3000.0, 0.0))
        plan = sm.plan_hops(prior, 0.0, self.SHAPE, mag_limit=8.0)
        self.assertTrue(plan["reachable"])
        self.assertEqual(plan["blind_px"], 0.0)
        # rich sky, straight-line target: everything one bolt direction
        axes = {leg["bolt"] for leg in plan["legs"]}
        self.assertEqual(axes, {"frame-x"})
        total = sum(leg["move_px"][0] for leg in plan["legs"])
        self.assertAlmostEqual(total, 3000.0, delta=plan["grid_step_px"])

    def test_detours_around_a_void(self):
        # rich sky except a vertical void band across the direct path;
        # a rich corridor exists below it
        stars = [s for s in self._blanket(-2000, 6000, -2000, 3200)
                 if not (1200 < s[0] < 2600 and s[1] < 1600)]
        prior = self._make(stars, (3000.0, 0.0))
        plan = sm.plan_hops(prior, 0.0, self.SHAPE, mag_limit=8.0)
        self.assertTrue(plan["reachable"])
        self.assertEqual(plan["blind_px"], 0.0,
                         f"route went blind instead of detouring: "
                         f"{plan['legs']}")
        # the detour must actually use the second axis
        axes = {leg["bolt"] for leg in plan["legs"]}
        self.assertIn("frame-y", axes)
        # and every leg ends with handholds
        for leg in plan["legs"]:
            self.assertGreaterEqual(leg["stars_at_end"], 5)
            self.assertIsNotNone(leg["anchor"])

    def test_minimal_blind_crossing_when_walled(self):
        # a full void band with NO corridor: the planner must cross, and
        # must say so honestly
        stars = [s for s in self._blanket(-2000, 6000, -3000, 4200)
                 if not (1200 < s[0] < 2600)]
        prior = self._make(stars, (3000.0, 0.0))
        plan = sm.plan_hops(prior, 0.0, self.SHAPE, mag_limit=8.0)
        self.assertTrue(plan["reachable"])
        self.assertGreater(plan["blind_px"], 0.0)
        blind = [leg for leg in plan["legs"] if leg["blind"]]
        self.assertTrue(blind)
        # the blind stretch is about the void's width, not a scenic tour
        self.assertLess(plan["blind_px"], 2500.0)

    def test_bolt_vectors_relabel_legs(self):
        stars = self._blanket(-2000, 6000, -2000, 3200)
        prior = self._make(stars, (2000.0, 500.0))
        plan = sm.plan_hops(prior, 0.0, self.SHAPE, mag_limit=8.0,
                            u_vec=(0.96, 0.28), v_vec=(-0.28, 0.96))
        self.assertTrue(plan["reachable"])
        self.assertTrue({leg["bolt"] for leg in plan["legs"]}
                        <= {"az", "alt"})


class TestMagLimit(unittest.TestCase):
    def test_noise_raises_limit_sensibly(self):
        model = sm.fit_response(np.linspace(5.0, 9.5, 30),
                                10 ** (4.4 - 0.32
                                       * np.linspace(5.0, 9.5, 30)))
        quiet = sm.mag_limit_from_noise(model, sigma=0.5)
        noisy = sm.mag_limit_from_noise(model, sigma=5.0)
        self.assertGreater(quiet, noisy)   # quieter night digs fainter
        lo, hi = model["mag_range"]
        self.assertGreaterEqual(noisy, lo)
        self.assertLessEqual(quiet, hi)


class TestNamesAndBolts(unittest.TestCase):
    def test_nearest_name_polar_ra_compression(self):
        names = {"sigma Oct": (317.195, -88.9565)}
        # 2 arcmin away in DEC: found
        self.assertEqual(sm.nearest_name(names, 317.195, -88.99),
                         "sigma Oct")
        # a full DEGREE of RA at dec -88.96 is under 2 arcmin on the sky
        self.assertEqual(sm.nearest_name(names, 318.2, -88.9565),
                         "sigma Oct")
        # 2 degrees away in dec: not claimed
        self.assertIsNone(sm.nearest_name(names, 317.195, -86.9))

    def test_names_file_merges_over_builtin(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "star_names.json"), "w") as fh:
                fh.write('{"BQ Oct": [225.0, -89.8],'
                         ' "sigma Oct": [317.2, -88.96]}')
            names = sm.load_star_names(d)
        self.assertIn("BQ Oct", names)
        self.assertIn("iota Oct", names)          # builtin survives
        self.assertAlmostEqual(names["sigma Oct"][0], 317.2)  # file wins

    def test_bolt_clustering_two_axes(self):
        rng = np.random.default_rng(31)
        u = np.array([0.94, 0.34])
        v = np.array([-0.31, 0.95])
        moves = []
        for _ in range(6):        # az turns, both directions, noisy
            moves.append(tuple(u * rng.uniform(15, 80)
                               * rng.choice([-1, 1])
                               + rng.normal(0, 1.5, 2)))
        for _ in range(5):        # alt turns
            moves.append(tuple(v * rng.uniform(15, 60)
                               * rng.choice([-1, 1])
                               + rng.normal(0, 1.5, 2)))
        moves.append((2.0, 1.0))  # sub-threshold flex: ignored
        bv = sm.cluster_move_axes(moves)
        self.assertIsNotNone(bv)
        got_u, got_v = bv
        # recovered axes align with truth to a few degrees (sign-free)
        for got, truth in ((got_u, u), (got_v, v)):
            c = abs(got[0] * truth[0] + got[1] * truth[1])
            self.assertGreater(c, math.cos(math.radians(6.0)),
                               f"axis {got} vs {truth}")

    def test_bolt_clustering_refuses_one_axis(self):
        u = (0.94, 0.34)
        moves = [(u[0] * s, u[1] * s) for s in
                 (40, -55, 62, -30, 45, 70, -25)]
        self.assertIsNone(sm.cluster_move_axes(moves))

    def test_bolts_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            sm.save_bolts((0.9, 0.44), (-0.44, 0.9), 8, d, 3840, 2160)
            bv = sm.load_bolts(d, 3840, 2160)
        self.assertAlmostEqual(bv[0][0], 0.9)
        self.assertAlmostEqual(bv[1][1], 0.9)
        self.assertIsNone(sm.load_bolts("/nonexistent", 1, 1))

    def test_predict_carries_sky_coords(self):
        w = _wcs_like_polar_scope()
        prior = sm.SkyPrior(w, 0.0,
                            (np.array([190.849]), np.array([-84.33]),
                             np.array([6.0])))
        p = prior.predict(0.0, (2160, 3840))
        self.assertEqual(len(p["ra"]), 1)
        self.assertAlmostEqual(float(p["ra"][0]), 190.849)
        self.assertAlmostEqual(float(p["dec"][0]), -84.33)


class TestBoltCorrection(unittest.TestCase):
    def test_orthogonal_axes_uncalibrated(self):
        # no bolt vectors -> frame x/y, 1"/px -> 1'/px at 60"/px scale
        corr = sm.bolt_correction((100.0, 100.0), (150.0, 130.0), 60.0)
        self.assertFalse(corr["calibrated"])
        self.assertEqual(corr["az"]["label"], "frame-x")
        self.assertEqual(corr["alt"]["label"], "frame-y")
        self.assertAlmostEqual(corr["az"]["arcmin"], 50.0)
        self.assertAlmostEqual(corr["alt"]["arcmin"], 30.0)

    def test_calibrated_orthogonal_axes(self):
        corr = sm.bolt_correction((0.0, 0.0), (50.0, 30.0), 60.0,
                                  u_vec=(1.0, 0.0), v_vec=(0.0, 1.0))
        self.assertTrue(corr["calibrated"])
        self.assertEqual(corr["az"]["label"], "az")
        self.assertEqual(corr["alt"]["label"], "alt")
        self.assertAlmostEqual(corr["az"]["arcmin"], 50.0)
        self.assertAlmostEqual(corr["alt"]["arcmin"], 30.0)

    def test_nonorthogonal_bolts_round_trip(self):
        # bolts at 90 deg apart in name only -- a real mount's rarely
        # exactly orthogonal. The decomposition must still reconstruct
        # the original correction vector exactly (that's what a linear
        # solve over a non-orthogonal basis guarantees).
        u = (1.0, 0.0)
        v = (0.6, 0.8)          # ~53 deg from u, not 90
        axis = (200.0, 400.0)
        target = (235.0, 370.0)
        corr = sm.bolt_correction(axis, target, 3.0, u_vec=u, v_vec=v)
        rebuilt_x = (corr["az"]["px"] * u[0] + corr["alt"]["px"] * v[0])
        rebuilt_y = (corr["az"]["px"] * u[1] + corr["alt"]["px"] * v[1])
        self.assertAlmostEqual(rebuilt_x, target[0] - axis[0], places=1)
        self.assertAlmostEqual(rebuilt_y, target[1] - axis[1], places=1)

    def test_zero_offset_is_zero_correction(self):
        corr = sm.bolt_correction((10.0, 10.0), (10.0, 10.0), 5.0)
        self.assertAlmostEqual(corr["az"]["arcmin"], 0.0)
        self.assertAlmostEqual(corr["alt"]["arcmin"], 0.0)


class TestRefraction(unittest.TestCase):
    def test_zero_at_zenith(self):
        self.assertAlmostEqual(sm.refraction_arcmin(90.0), 0.0, delta=0.01)

    def test_never_negative(self):
        for alt in (90.0, 89.99, 85.0):
            self.assertGreaterEqual(sm.refraction_arcmin(alt), 0.0)

    def test_about_29_arcmin_at_horizon(self):
        self.assertAlmostEqual(sm.refraction_arcmin(0.0), 29.0, delta=1.0)

    def test_monotonically_decreasing_with_altitude(self):
        alts = (0.0, 10.0, 30.0, 45.0, 60.0, 80.0, 90.0)
        vals = [sm.refraction_arcmin(a) for a in alts]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_pressure_scales_linearly(self):
        full = sm.refraction_arcmin(30.0, pressure_hpa=1010.0)
        half = sm.refraction_arcmin(30.0, pressure_hpa=505.0)
        self.assertAlmostEqual(half, full / 2.0, delta=0.01)

    def test_clamps_out_of_range_altitude(self):
        # a caller passing a bogus true altitude (e.g. below the
        # horizon) gets the horizon value, not a domain error
        self.assertAlmostEqual(sm.refraction_arcmin(-5.0),
                               sm.refraction_arcmin(0.0))
        self.assertAlmostEqual(sm.refraction_arcmin(95.0),
                               sm.refraction_arcmin(90.0))


class TestCatalogLoader(unittest.TestCase):
    def test_load_by_hemisphere_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            cat = np.array([[190.0, -84.0, 6.0], [10.0, -85.0, 7.0]],
                           dtype=np.float32)
            np.save(os.path.join(d, "catalog_scp.npy"), cat)
            got = sm.load_catalog(d, -84.3)
            self.assertIsNotNone(got)
            self.assertEqual(len(got[0]), 2)
            # north cap not present -> None, not an exception
            self.assertIsNone(sm.load_catalog(d, +84.3))


class TestCollimation(unittest.TestCase):
    def test_clock_convention(self):
        shape = (2160, 3840)
        c = (3840 - 1) / 2.0, (2160 - 1) / 2.0
        # right of centre = 3 o'clock; up = 12; down-left = between 7-8
        r = sm.collimation((c[0] + 100, c[1]), shape, 3.96)
        self.assertEqual(r["clock"], 3)
        r = sm.collimation((c[0], c[1] - 100), shape, 3.96)
        self.assertEqual(r["clock"], 12)
        r = sm.collimation((c[0] - 100, c[1] + 100), shape, 3.96)
        self.assertIn(r["clock"], (7, 8))
        # magnitude: 100 px at 3.96"/px = 6.6 arcmin
        self.assertAlmostEqual(r["offset_px"], 141.4, delta=0.1)
        r = sm.collimation((c[0] + 100, c[1]), shape, 3.96)
        self.assertAlmostEqual(r["arcmin"], 6.6, delta=0.01)


class TestTimeAndPrecession(unittest.TestCase):
    """Anchored on values that are definitions, not opinions."""

    J2000 = 946728000.0        # 2000-01-01 12:00:00 UTC as a POSIX time

    def test_julian_date_at_j2000(self):
        self.assertAlmostEqual(sm.julian_date(self.J2000), 2451545.0, places=6)

    def test_gmst_at_j2000_is_the_defining_value(self):
        self.assertAlmostEqual(sm.gmst_deg(2451545.0), 280.46061837, places=6)

    def test_gmst_advances_a_sidereal_day(self):
        # one solar day advances GMST by ~360.9856 deg, i.e. ~0.9856 net
        g0 = sm.gmst_deg(2451545.0)
        g1 = sm.gmst_deg(2451546.0)
        self.assertAlmostEqual((g1 - g0) % 360.0, 0.98564736629, places=5)

    def test_lst_tracks_longitude_one_for_one(self):
        a = sm.lst_deg(self.J2000, 0.0)
        b = sm.lst_deg(self.J2000, 90.0)
        self.assertAlmostEqual((b - a) % 360.0, 90.0, places=9)

    def test_precession_is_identity_at_j2000(self):
        ra, dec = sm.precess_from_j2000(123.4, 56.7, self.J2000)
        self.assertAlmostEqual(ra, 123.4, places=6)
        self.assertAlmostEqual(dec, 56.7, places=6)

    def test_pole_precesses_by_the_expected_rate(self):
        # ~20.04"/yr, so the J2000 pole is ~8.9' off the pole of date in 2026
        t2026 = self.J2000 + 26.58 * 365.25 * 86400.0
        _ra, dec = sm.precess_from_j2000(0.0, 90.0, t2026)
        off_arcmin = (90.0 - dec) * 60.0
        self.assertAlmostEqual(off_arcmin, 2004.3109 * 0.2658 / 60.0, delta=0.05)
        self.assertGreater(off_arcmin, 8.0)   # big enough to matter, the point

    def test_precession_preserves_angular_separation(self):
        # a rotation cannot change the angle between two directions
        def sep(a, b):
            (r1, d1), (r2, d2) = a, b
            r1, d1, r2, d2 = map(math.radians, (r1, d1, r2, d2))
            return math.degrees(math.acos(max(-1.0, min(1.0,
                math.sin(d1) * math.sin(d2)
                + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)))))
        t = self.J2000 + 26.0 * 365.25 * 86400.0
        A, B = (10.0, 70.0), (200.0, 85.0)
        self.assertAlmostEqual(
            sep(A, B),
            sep(sm.precess_from_j2000(*A, t), sm.precess_from_j2000(*B, t)),
            places=9)


class TestAltAz(unittest.TestCase):
    def test_pole_sits_at_latitude_altitude_north(self):
        for lst in (0.0, 73.2, 189.0, 359.9):
            alt, az = sm.radec_to_altaz(0.0, 90.0, 51.5, lst)
            self.assertAlmostEqual(alt, 51.5, places=9)
            self.assertAlmostEqual(az % 360.0, 0.0, places=6)

    def test_pole_sits_due_south_in_the_south(self):
        for lst in (0.0, 120.0, 300.0):
            alt, az = sm.radec_to_altaz(0.0, -90.0, -34.9, lst)
            self.assertAlmostEqual(alt, 34.9, places=9)
            self.assertAlmostEqual(az, 180.0, places=6)

    def test_object_on_the_meridian_is_due_south_or_north(self):
        lst = 100.0
        # dec below the zenith for a northern observer -> due south, H=0
        alt, az = sm.radec_to_altaz(lst, 20.0, 51.5, lst)
        self.assertAlmostEqual(az, 180.0, places=6)
        self.assertAlmostEqual(alt, 90.0 - 51.5 + 20.0, places=6)

    def test_azimuth_east_and_west_of_the_meridian(self):
        # H = LST - RA, so H < 0 is still RISING (east of the meridian) and
        # H > 0 has already transited (west). Getting this backwards is the
        # classic hour-angle slip, so both sides are pinned here.
        lst, dec, lat = 100.0, 20.0, 51.5
        _a, az_east = sm.radec_to_altaz((lst + 15.0) % 360.0, dec, lat, lst)
        _a, az_west = sm.radec_to_altaz((lst - 15.0) % 360.0, dec, lat, lst)
        self.assertTrue(90.0 < az_east < 180.0, az_east)   # east of south
        self.assertTrue(180.0 < az_west < 270.0, az_west)  # west of south


class TestPolarAlignment(unittest.TestCase):
    """Round-trip: place the axis a known amount off the pole in horizon
    coordinates, hand the solver the equivalent J2000 direction, and require
    the original offsets back. Exercises precession, LST, the horizon
    transform and the sign conventions together."""

    T = 1785936600.0        # a fixed 2026 epoch

    @staticmethod
    def _P(t):
        T = (sm.julian_date(t) - 2451545.0) / 36525.0
        s = math.pi / (180.0 * 3600.0)
        ze = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T ** 3) * s
        z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T ** 3) * s
        th = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T ** 3) * s
        Rz = lambda a: np.array([[math.cos(a), -math.sin(a), 0.0],
                                 [math.sin(a), math.cos(a), 0.0],
                                 [0.0, 0.0, 1.0]])
        My = lambda a: np.array([[math.cos(a), 0.0, -math.sin(a)],
                                 [0.0, 1.0, 0.0],
                                 [math.sin(a), 0.0, math.cos(a)]])
        return Rz(z) @ My(th) @ Rz(ze)

    @classmethod
    def _deprecess(cls, ra_d, dec_d, t):
        r, d = math.radians(ra_d), math.radians(dec_d)
        v = np.array([math.cos(d) * math.cos(r), math.cos(d) * math.sin(r),
                      math.sin(d)])
        u = cls._P(t).T @ v
        return (math.degrees(math.atan2(u[1], u[0])) % 360.0,
                math.degrees(math.asin(max(-1.0, min(1.0, u[2])))))

    @staticmethod
    def _altaz_to_radec(alt, az, lat, lst):
        a, A, p = map(math.radians, (alt, az, lat))
        dec = math.asin(max(-1.0, min(1.0, math.sin(a) * math.sin(p)
                                      + math.cos(a) * math.cos(p) * math.cos(A))))
        H = math.atan2(-math.sin(A) * math.cos(a),
                       math.sin(a) * math.cos(p)
                       - math.cos(a) * math.sin(p) * math.cos(A))
        return (lst - math.degrees(H)) % 360.0, math.degrees(dec)

    def _roundtrip(self, lat, lon, d_alt, d_az):
        lst = sm.lst_deg(self.T, lon)
        alt_pole, az_pole = abs(lat), (180.0 if lat < 0 else 0.0)
        ra_d, dec_d = self._altaz_to_radec(alt_pole - d_alt,
                                           (az_pole - d_az) % 360.0, lat, lst)
        ra_j, dec_j = self._deprecess(ra_d, dec_d, self.T)
        return sm.polar_alignment(ra_j, dec_j, lat, lon, self.T)

    def test_recovers_known_offsets_both_hemispheres(self):
        for lat, lon, da, dz in ((-34.9, 138.6, -1.50, +2.00),
                                 (-34.9, 138.6, +0.75, -0.40),
                                 (+51.5, -0.1, -2.00, -1.25),
                                 (+51.5, -0.1, +0.10, +0.05),
                                 (+22.0, 120.0, -0.005, +0.004)):
            r = self._roundtrip(lat, lon, da, dz)
            self.assertAlmostEqual(r["d_alt_deg"], da, places=6)
            self.assertAlmostEqual(r["d_az_deg"], dz, places=6)

    def test_direction_words_match_the_signs(self):
        # _roundtrip places the axis at (alt_pole - d_alt), so a POSITIVE
        # d_alt puts the axis BELOW the pole and must read "point higher".
        r = self._roundtrip(-34.9, 138.6, +1.5, +2.0)   # axis low, and left
        self.assertTrue(r["alt_up"])
        self.assertIn("HIGHER", r["instruction_alt"])
        self.assertTrue(r["az_right"])
        self.assertIn("RIGHT", r["instruction_az"])
        r = self._roundtrip(-34.9, 138.6, -1.5, -2.0)   # axis high, and right
        self.assertFalse(r["alt_up"])
        self.assertIn("LOWER", r["instruction_alt"])
        self.assertFalse(r["az_right"])
        self.assertIn("LEFT", r["instruction_az"])

    def test_sky_error_is_independent_of_the_decomposition(self):
        # The true angular error should match the quadrature combination of
        # the two adjustments with azimuth foreshortened by cos(altitude) --
        # but that relation is FIRST ORDER, so at ~1 deg it is only good to
        # a few thousandths. The tolerance says so rather than pretending.
        lat = -34.9
        r = self._roundtrip(lat, 138.6, -0.8, +1.1)
        approx = math.hypot(r["d_alt_deg"],
                            r["d_az_deg"] * math.cos(math.radians(abs(lat))))
        self.assertAlmostEqual(r["sky_error_deg"], approx, delta=0.01)

    def test_perfect_alignment_reads_zero(self):
        # 1e-5 deg = 0.04 arcsec: the round trip goes through two precession
        # rotations, so this is float residue, not a modelling error
        r = self._roundtrip(-34.9, 138.6, 0.0, 0.0)
        self.assertAlmostEqual(r["d_alt_deg"], 0.0, places=5)
        self.assertAlmostEqual(r["d_az_deg"], 0.0, places=5)
        self.assertLess(r["sky_error_arcmin"], 0.01)

    def test_precession_is_actually_applied(self):
        # feeding the J2000 pole must NOT report zero error: it is ~9' from
        # the pole of date, and reporting 0 there is the bug this guards
        r = sm.polar_alignment(0.0, 90.0, 51.5, -0.1, self.T)
        self.assertGreater(r["sky_error_arcmin"], 8.0)
        self.assertLess(r["sky_error_arcmin"], 10.0)

    def test_result_is_json_serialisable(self):
        json.dumps(self._roundtrip(-34.9, 138.6, -1.0, 1.0))


class TestTleMatcher(unittest.TestCase):
    """Self-consistency: fabricate a transient record AT a satellite's
    own computed topocentric position, then require the matcher to find
    it -- validates TLE parsing, GMST, frame rotation and the match loop
    as one chain. Plus an independent GMST anchor value."""

    ISS = ("ISS (ZARYA)\n"
           "1 25544U 98067A   19343.69339541  .00001764  00000-0 "
           " 40967-4 0  9998\n"
           "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 "
           "15.50103472202482\n")

    def test_gmst_j2000_anchor(self):
        import match_tle as mt
        # GMST at J2000.0 epoch (2000-01-01 12:00 UT) is 280.46062 deg
        self.assertAlmostEqual(mt.gmst_deg(2451545.0), 280.46061837,
                               places=6)

    def test_roundtrip_match(self):
        try:
            import sgp4  # noqa: F401
        except ImportError:
            self.skipTest("python-sgp4 not installed")
        import match_tle as mt
        with tempfile.TemporaryDirectory() as d:
            tle_path = os.path.join(d, "test.tle")
            with open(tle_path, "w") as fh:
                fh.write(self.ISS)
            sats = mt.load_tles(tle_path)
            self.assertEqual(len(sats), 1)
            lat, lon = -34.93, 138.60          # Adelaide-ish
            obs = mt.geodetic_to_ecef(lat, lon, 0.05)
            # scan for a moment near the TLE epoch when the ISS is
            # actually above the local horizon -- the matcher rightly
            # refuses below-horizon satellites
            tr = iso = None
            for minutes in range(0, 24 * 60, 5):
                cand = (f"2019-12-09T{minutes // 60:02d}:"
                        f"{minutes % 60:02d}:00+00:00")
                jd, fr = mt.jd_from_iso(cand)
                t = mt.topo_radec(sats[0][1], jd, fr, obs)
                if t is not None and t[3] > 15.0:
                    tr, iso = t, cand
                    break
            self.assertIsNotNone(tr, "ISS never above horizon all day?")
            jd, fr = mt.jd_from_iso(iso)
            rec = {"t": iso, "class": "transient",
                   "ra": tr[0], "dec": tr[1],
                   "pa_deg": mt.motion_pa(sats[0][1], jd, fr, obs),
                   "x": 1, "y": 1}
            tpath = os.path.join(d, "transients.jsonl")
            with open(tpath, "w") as fh:
                fh.write(json.dumps(rec) + "\n")
            out = os.path.join(d, "out.jsonl")
            mt.main([tpath, "--lat", str(lat), "--lon", str(lon),
                     "--alt-m", "50", "--tle-file", tle_path,
                     "--out", out])
            got = json.loads(open(out).read())
            self.assertIsNotNone(got["tle_match"])
            self.assertEqual(got["tle_match"]["name"], "ISS (ZARYA)")
            self.assertLess(got["tle_match"]["sep_deg"], 0.01)
            self.assertFalse(got["tle_match"]["suspicious"])

    def test_far_position_unmatched(self):
        try:
            import sgp4  # noqa: F401
        except ImportError:
            self.skipTest("python-sgp4 not installed")
        import match_tle as mt
        with tempfile.TemporaryDirectory() as d:
            tle_path = os.path.join(d, "test.tle")
            with open(tle_path, "w") as fh:
                fh.write(self.ISS)
            rec = {"t": "2019-12-09T18:00:00+00:00",
                   "class": "transient", "ra": 10.0, "dec": 89.0,
                   "x": 1, "y": 1}
            tpath = os.path.join(d, "transients.jsonl")
            with open(tpath, "w") as fh:
                fh.write(json.dumps(rec) + "\n")
            out = os.path.join(d, "out.jsonl")
            mt.main([tpath, "--lat", "-34.93", "--lon", "138.60",
                     "--tle-file", tle_path, "--out", out])
            got = json.loads(open(out).read())
            self.assertIsNone(got["tle_match"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
