#!/usr/bin/env python3
"""Tests for cam_characterise.py's analysis layer. Synthetic-first: every
case here is a ladder or sweep constructed with the answer known in advance,
so the guards are tested against the PHYSICS they claim to detect rather than
against a recorded run. Run: python3 test_cam_characterise.py

The cases that matter are the ones drawn from real failures:
  * a run at mean 18 with zero temporal noise -- nowhere near a rail, and
    still meaningless (the clip guard passes it, the quantiser guard must not)
  * a ladder whose fps stays pinned through 256 and breaks at 384 (the knee
    that --knee 256 got wrong)
  * a defect count that moves 12 -> 1057 across a gamma change
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

import cam_characterise as cc


class TestQuantiserGuard(unittest.TestCase):
    """The guard that catches a collapsed output the clip guard sails past."""

    def test_zero_noise_is_quantiser_limited(self):
        # the real failure: mean 18, no rail in sight, temporal noise 0.000
        q = cc.quantiser_check(distinct_master=1, temporal=0.0,
                               code_step=1.0, distinct_stack=1)
        self.assertTrue(q["quantiser_limited"])
        self.assertTrue(any("distinct" in r for r in q["reasons"]))
        self.assertTrue(any("dither floor" in r for r in q["reasons"]))

    def test_two_codes_alone_trips_it(self):
        # noise is above the floor, but a 2-code master has no structure left
        q = cc.quantiser_check(distinct_master=2, temporal=0.9,
                               code_step=1.0, distinct_stack=2)
        self.assertTrue(q["quantiser_limited"])
        self.assertEqual(len(q["reasons"]), 1)

    def test_low_noise_alone_trips_it(self):
        q = cc.quantiser_check(distinct_master=4000, temporal=0.01,
                               code_step=1.0, distinct_stack=40)
        self.assertTrue(q["quantiser_limited"])

    def test_healthy_stack_passes(self):
        q = cc.quantiser_check(distinct_master=180000, temporal=1.7,
                               code_step=1.0, distinct_stack=64)
        self.assertFalse(q["quantiser_limited"])
        self.assertEqual(q["reasons"], [])
        self.assertNotIn("note", q)

    def test_boundary_is_not_tripped(self):
        # exactly at the floor is not below it
        q = cc.quantiser_check(distinct_master=99, temporal=0.05,
                               code_step=1.0, distinct_stack=99)
        self.assertFalse(q["quantiser_limited"])

    def test_nan_noise_does_not_trip_the_floor(self):
        # a single-frame point has nan temporal noise; nan < 0.05 is False in
        # Python but the check must not crash or silently flag it
        q = cc.quantiser_check(distinct_master=5000, temporal=float("nan"),
                               code_step=1.0, distinct_stack=50)
        self.assertFalse(q["quantiser_limited"])

    def test_code_step_tracks_bit_depth(self):
        self.assertEqual(cc.code_step_adu(8), 1.0)
        self.assertAlmostEqual(cc.code_step_adu(16), 1.0 / 257.0)
        self.assertEqual(cc.code_step_adu(None), 1.0)


class TestNoiseUpperBound(unittest.TestCase):
    def test_eight_bit_bound_is_half_a_code(self):
        nb = cc.noise_upper_bound(0.0, 1.0)
        self.assertEqual(nb["upper_bound_adu"], 0.5)
        self.assertIn("< 0.5 ADU", nb["statement"])
        self.assertIn("unmeasurable", nb["statement"])
        self.assertEqual(nb["measured_adu"], 0.0)

    def test_sixteen_bit_bound_is_finer(self):
        nb = cc.noise_upper_bound(0.0, 1.0 / 257.0)
        self.assertLess(nb["upper_bound_adu"], 0.002)

    def test_nan_measurement_survives_json(self):
        nb = cc.noise_upper_bound(float("nan"), 1.0)
        self.assertIsNone(nb["measured_adu"])
        json.dumps(nb)   # must not raise


class TestMinimumDetectableSlope(unittest.TestCase):
    def test_quantiser_limited_limit_is_one_code_over_the_span(self):
        xs = np.array([2000.0, 4000.0, 6000.0, 8000.0])
        ys = np.full(4, 18.0)          # flat to every decimal
        m = cc.minimum_detectable_slope(xs, ys, code_step=1.0,
                                        quantiser_limited=True)
        # one whole code across a 6000-unit span
        self.assertAlmostEqual(m["min_detectable_slope_adu_per_unit"],
                               1.0 / 6000.0)
        self.assertEqual(m["exposure_span_units"], 6000.0)
        self.assertIn("no temporal dither", m["basis"])
        self.assertIn("<", m["statement"])

    def test_noisy_ladder_uses_residual_scatter(self):
        xs = np.array([1000.0, 2000.0, 3000.0, 4000.0])
        ys = np.array([10.0, 10.2, 9.9, 10.1])
        m = cc.minimum_detectable_slope(xs, ys, code_step=1.0,
                                        quantiser_limited=False,
                                        resid_std=0.12)
        self.assertAlmostEqual(m["min_detectable_level_change_adu"], 0.36)
        self.assertAlmostEqual(m["min_detectable_slope_adu_per_unit"],
                               0.36 / 3000.0)
        self.assertIn("residual scatter", m["basis"])

    def test_per_second_conversion(self):
        xs = np.array([0.0, 10000.0])
        ys = np.array([1.0, 1.0])
        m = cc.minimum_detectable_slope(xs, ys, 1.0, True)
        # exposure units are 100us, so ADU/unit -> ADU/s is x10000
        self.assertAlmostEqual(m["min_detectable_slope_adu_per_second"],
                               m["min_detectable_slope_adu_per_unit"] * 10000.0)

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(cc.minimum_detectable_slope(
            np.array([5.0]), np.array([1.0]), 1.0, True))
        self.assertIsNone(cc.minimum_detectable_slope(
            np.array([7.0, 7.0]), np.array([1.0, 1.0]), 1.0, True))


def _ladder(pairs):
    """[(exposure, fps)] -> ladder records shaped like run_dark's."""
    return [{"exposure_units": e, "eff_fps": f, "mean_adu": 18.0,
             "temporal_noise_adu": 0.0} for e, f in pairs]


class TestDeriveKnee(unittest.TestCase):
    def test_finds_the_first_real_departure(self):
        # the 9411-range case: fps pinned through 256, breaks at 384
        d = cc.derive_knee(_ladder([(64, 30.0), (128, 30.0), (256, 29.9),
                                    (384, 24.0), (512, 18.0), (1024, 9.0)]))
        self.assertEqual(d["knee_units"], 384)
        self.assertEqual(d["last_pinned_exposure"], 256)
        self.assertEqual(d["fps_pinned_max"], 30.0)

    def test_single_dip_is_not_the_knee(self):
        # one hiccup at 128 that recovers must not be mistaken for the knee
        d = cc.derive_knee(_ladder([(64, 30.0), (128, 21.0), (256, 30.0),
                                    (384, 24.0), (512, 12.0)]))
        self.assertEqual(d["knee_units"], 384)

    def test_never_departs_returns_none(self):
        # wholly bandwidth-limited: there is no integrating region to find
        self.assertIsNone(cc.derive_knee(
            _ladder([(64, 15.0), (256, 15.0), (1024, 14.9), (4096, 15.0)])))

    def test_too_few_points_returns_none(self):
        self.assertIsNone(cc.derive_knee(_ladder([(64, 30.0), (512, 10.0)])))

    def test_ignores_points_with_no_fps(self):
        rows = _ladder([(64, 30.0), (128, 30.0), (256, 12.0), (512, 6.0)])
        rows.append({"exposure_units": 32, "eff_fps": 0.0, "mean_adu": 18.0})
        d = cc.derive_knee(rows)
        self.assertEqual(d["knee_units"], 256)

    def test_unsorted_input_is_handled(self):
        d = cc.derive_knee(_ladder([(512, 6.0), (64, 30.0), (256, 12.0),
                                    (128, 30.0)]))
        self.assertEqual(d["knee_units"], 256)


class TestTimingProbeSelection(unittest.TestCase):
    """MJPEG is used as a clock, never as a measurement -- so the only thing
    that matters in picking it is bandwidth, and the only thing that matters
    about failing to pick it is saying so."""

    def test_prefers_mjpeg(self):
        ranked = [{"fourcc": "GREY"}, {"fourcc": "YUYV"}, {"fourcc": "MJPG"}]
        adv = [{"fourcc": "MJPG", "width": 1920, "height": 1080},
               {"fourcc": "YUYV", "width": 1920, "height": 1080}]
        got = cc.pick_timing_format(ranked, adv, 1920, 1080)
        self.assertEqual(got["fourcc"], "MJPG")
        self.assertTrue(got["advertised_at_size"])

    def test_falls_through_to_h264(self):
        ranked = [{"fourcc": "YUYV"}, {"fourcc": "H264"}]
        adv = [{"fourcc": "H264", "width": 1280, "height": 720}]
        self.assertEqual(
            cc.pick_timing_format(ranked, adv, 1280, 720)["fourcc"], "H264")

    def test_no_compressed_format_returns_none(self):
        ranked = [{"fourcc": "GREY"}, {"fourcc": "YUYV"}]
        self.assertIsNone(cc.pick_timing_format(ranked, [], 1280, 720))

    def test_skips_format_not_advertised_at_this_size(self):
        ranked = [{"fourcc": "MJPG"}]
        adv = [{"fourcc": "MJPG", "width": 640, "height": 480}]
        self.assertIsNone(cc.pick_timing_format(ranked, adv, 1920, 1080))

    def test_unknown_mode_list_still_allows_the_attempt(self):
        # no advertised list at all is "unknown", not "unavailable"
        got = cc.pick_timing_format([{"fourcc": "MJPG"}], [], 1280, 720)
        self.assertEqual(got["fourcc"], "MJPG")
        self.assertIsNone(got["advertised_at_size"])

    def test_probe_class_has_no_decode_path(self):
        # the guarantee that MJPEG can never contaminate a pixel measurement is
        # structural: there is nothing on FpsProbe that returns image data
        self.assertFalse(hasattr(cc.FpsProbe, "_luma_from_frame"))
        self.assertFalse(hasattr(cc.FpsProbe, "iter_luma"))
        self.assertFalse(hasattr(cc.FpsProbe, "grab_n"))


class TestTimingLadder(unittest.TestCase):
    def test_geometric_spacing_resolves_a_low_knee(self):
        # the 384-unit knee in a 9411 range must not be stepped over
        lad = cc.build_timing_ladder(1, 9411)
        self.assertLessEqual(min(lad), 1)
        self.assertEqual(max(lad), 9411)
        below = [e for e in lad if e < 384]
        above = [e for e in lad if e > 384]
        self.assertGreaterEqual(len(below), 6)   # dense where the knee lives
        self.assertTrue(above)

    def test_is_sorted_and_unique(self):
        lad = cc.build_timing_ladder(1, 2047)
        self.assertEqual(lad, sorted(set(lad)))

    def test_degenerate_range(self):
        self.assertEqual(cc.build_timing_ladder(500, 500), [500])
        self.assertEqual(cc.build_timing_ladder(900, 100), [100])


class TestDarkLadderBuilder(unittest.TestCase):
    def test_always_includes_the_max(self):
        self.assertIn(9411, cc.build_exposure_ladder(9411))

    def test_explicit_ladder_is_honoured_and_sorted(self):
        got = cc.build_exposure_ladder(8192, ladder=[4096, 1024, 4096, 8192])
        self.assertEqual(got, [1024, 4096, 8192])

    def test_knee_override_raises_the_start(self):
        no_knee = cc.build_exposure_ladder(10000, knee=None, ladder_points=5)
        high = cc.build_exposure_ladder(10000, knee=8000, ladder_points=5)
        self.assertGreater(max(x for x in high if x < 10000),
                           min(x for x in no_knee if x > 3000))
        self.assertTrue(all(e >= 8000 for e in high if e > 2000))


class TestKneeSourcePreference(unittest.TestCase):
    """The timing probe's knee and the uncompressed ladder's knee are
    different quantities: one is the integration knee, the other the bandwidth
    ceiling. Confusing them over-excludes good points from the fit."""

    def test_probe_knee_is_lower_than_bandwidth_pinned_knee(self):
        # uncompressed: fps pinned at 5 throughout -> no departure at all
        uncompressed = _ladder([(128, 5.0), (384, 5.0), (1024, 5.0),
                                (4096, 5.0)])
        self.assertIsNone(cc.derive_knee(uncompressed))
        # same camera in MJPEG: ceiling is 30fps and the knee is visible
        mjpeg = _ladder([(128, 30.0), (279, 30.0), (384, 22.0),
                         (1024, 9.0), (4096, 2.4)])
        d = cc.derive_knee(mjpeg)
        self.assertEqual(d["knee_units"], 384)


class TestTransferCurveFit(unittest.TestCase):
    def _sweep(self, fn, levels=(0, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224)):
        return [{"level": L, "mean_adu": fn(L), "clipped": False}
                for L in levels]

    def test_pure_scaling_reads_as_linear(self):
        pts = self._sweep(lambda L: 6.0 + 0.55 * L)
        fit = cc.fit_transfer_curve(pts)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["pedestal_adu"], 6.0, places=6)
        self.assertGreater(fit["linear_fit"]["r2"], 0.999)
        self.assertAlmostEqual(fit["power_fit"]["exponent"], 1.0, places=3)
        self.assertIn("LINEAR", fit["verdict"])

    def test_power_law_is_detected(self):
        # a genuine tone-curve reshape: exponent 2.2 on top of a pedestal
        pts = self._sweep(lambda L: 6.0 + 0.02 * (L ** 2.2))
        fit = cc.fit_transfer_curve(pts)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["power_fit"]["exponent"], 2.2, places=2)
        self.assertIn("POWER LAW", fit["verdict"])

    def test_clipped_points_are_excluded(self):
        pts = self._sweep(lambda L: 6.0 + 0.55 * L)
        pts += [{"level": 240, "mean_adu": 255.0, "clipped": True},
                {"level": 255, "mean_adu": 255.0, "clipped": True}]
        fit = cc.fit_transfer_curve(pts)
        self.assertEqual(fit["n_points_used"], 11)
        self.assertGreater(fit["linear_fit"]["r2"], 0.999)

    def test_too_few_points_refuses(self):
        fit = cc.fit_transfer_curve(self._sweep(lambda L: L, levels=(0, 8, 16)))
        self.assertFalse(fit["ok"])
        self.assertIn("need >=4", fit["reason"])

    def test_caveat_is_always_carried(self):
        fit = cc.fit_transfer_curve(self._sweep(lambda L: 6.0 + 0.55 * L))
        self.assertIn("display transfer", fit["display_gamma_caveat"])


class TestDefectClassification(unittest.TestCase):
    """A flat count of 1057 defects is not actionable, because those pixels
    fail in three ways with three different remedies. Each case here plants a
    known defect of one kind and checks it lands in exactly one bucket."""

    H, W, N = 120, 160, 64

    def _stack(self, seed=7):
        rng = np.random.default_rng(seed)
        base = 18.0 + 0.4 * np.linspace(0, 1, self.H)[:, None] * np.ones(
            (1, self.W))                       # mild shading gradient
        frames = base[None, :, :] + rng.normal(0, 1.4,
                                               (self.N, self.H, self.W))
        return base, frames, rng

    def _classify(self, frames, bias=None, exp_max=9411, exp_bias=470):
        mean = frames.mean(0)
        std = frames.std(0, ddof=1)
        mx = frames.max(0)
        return cc.classify_defects(mean, std, max_px=mx, bias=bias,
                                   exposure_max=exp_max,
                                   exposure_bias=exp_bias)

    def test_stuck_pixel_is_stuck_not_hot(self):
        _b, fr, _ = self._stack()
        fr[:, 10, 10] = 200.0                  # never varies
        d = self._classify(fr)
        self.assertEqual(d["coords"]["stuck"], [[10, 10]])
        self.assertNotIn([10, 10], d["coords"]["hot"])
        self.assertNotIn([10, 10], d["coords"]["intermittent_hot"])

    def test_stable_hot_pixel_is_hot(self):
        _b, fr, _ = self._stack()
        fr[:, 20, 20] += 45.0
        d = self._classify(fr)
        self.assertEqual(d["coords"]["hot"], [[20, 20]])
        self.assertEqual(d["counts"]["stuck"], 0)

    def test_intermittent_outranks_hot(self):
        # hot in 60% of frames: the AVERAGE reads as hot, but subtraction
        # cannot fix it, so it must be classified intermittent
        _b, fr, rng = self._stack()
        idx = rng.random(self.N) < 0.60
        fr[idx, 30, 30] += 60.0
        d = self._classify(fr)
        self.assertIn([30, 30], d["coords"]["intermittent_hot"])
        self.assertNotIn([30, 30], d["coords"]["hot"])

    def test_rare_spiker_invisible_to_the_mean_is_still_caught(self):
        # hot in 1 frame of 64 by +30 ADU: the mean moves 0.47 ADU, under the
        # 6-sigma residual threshold, so a threshold on the master walks
        # straight past it. This is the case the split exists for.
        _b, fr, _ = self._stack()
        fr[:1, 40, 40] += 30.0
        mean = fr.mean(0)
        self.assertFalse(cc.hot_pixel_mask(mean)[40, 40])   # old test misses it
        d = self._classify(fr)
        self.assertIn([40, 40], d["coords"]["intermittent_hot"])

    def test_dark_current_needs_the_bias_lever_arm(self):
        base, fr, _ = self._stack()
        bias = base.astype(np.float32) - 2.0
        bias[50, 50] -= 25.0                   # steep only vs short exposure
        d = self._classify(fr, bias=bias)
        self.assertIn([50, 50], d["coords"]["dark_current"])
        self.assertIn("median_rate_adu_per_unit", d["dark_current_stats"])

    def test_no_bias_skips_dark_current_and_says_so(self):
        _b, fr, _ = self._stack()
        d = self._classify(fr, bias=None)
        self.assertEqual(d["counts"]["dark_current"], 0)
        self.assertIn("skipped", d["dark_current_stats"])

    def test_categories_are_mutually_exclusive(self):
        base, fr, rng = self._stack()
        fr[:, 10, 10] = 200.0
        fr[:, 20, 20] += 45.0
        fr[rng.random(self.N) < 0.25, 30, 30] += 60.0
        bias = base.astype(np.float32) - 2.0
        bias[50, 50] -= 25.0
        d = self._classify(fr, bias=bias)
        seen = []
        for k in ("stuck", "intermittent_hot", "hot", "dark_current"):
            seen.extend(tuple(p) for p in d["coords"][k])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(d["counts"]["total"], len(seen))

    def test_all_four_kinds_separate_cleanly(self):
        base, fr, rng = self._stack()
        fr[:, 10, 10] = 200.0
        fr[:, 20, 20] += 45.0
        fr[rng.random(self.N) < 0.25, 30, 30] += 60.0
        bias = base.astype(np.float32) - 2.0
        bias[50, 50] -= 25.0
        d = self._classify(fr, bias=bias)
        self.assertEqual(d["coords"]["stuck"], [[10, 10]])
        self.assertEqual(d["coords"]["hot"], [[20, 20]])
        self.assertEqual(d["coords"]["intermittent_hot"], [[30, 30]])
        self.assertEqual(d["coords"]["dark_current"], [[50, 50]])

    def test_interpolation_set_is_a_superset_of_the_old_flat_map(self):
        # the default --defect-classes must never DROP a pixel the old
        # single-threshold map would have written
        base, fr, rng = self._stack()
        fr[:, 10, 10] = 200.0
        fr[:, 20, 20] += 45.0
        fr[rng.random(self.N) < 0.25, 30, 30] += 60.0
        mean = fr.mean(0)
        old = {tuple(p) for p in np.argwhere(cc.hot_pixel_mask(mean)).tolist()}
        d = self._classify(fr, bias=base.astype(np.float32) - 2.0)
        new = set()
        for k in ("stuck", "intermittent_hot", "hot"):
            new |= {tuple(p) for p in d["coords"][k]}
        self.assertTrue(old <= new, f"dropped {old - new}")

    def test_quantiser_collapse_refuses_rather_than_reporting_millions(self):
        # every pixel frozen is a collapsed output, not a sensor covered in
        # stuck pixels
        flat = np.full((self.H, self.W), 18.0)
        d = cc.classify_defects(flat, np.zeros((self.H, self.W)))
        self.assertFalse(d["ok"])
        self.assertIn("quantiser-collapsed", d["reason"])
        self.assertNotIn("counts", d)

    def test_clean_sensor_yields_no_defects(self):
        _b, fr, _ = self._stack(seed=11)
        d = self._classify(fr)
        self.assertTrue(d["ok"])
        self.assertEqual(d["counts"]["total"], 0)

    def test_works_without_max_px_using_instability_alone(self):
        _b, fr, rng = self._stack()
        fr[rng.random(self.N) < 0.30, 30, 30] += 60.0
        mean, std = fr.mean(0), fr.std(0, ddof=1)
        d = cc.classify_defects(mean, std, max_px=None)
        self.assertFalse(d["thresholds"]["max_px_available"])
        self.assertIn([30, 30], d["coords"]["intermittent_hot"])

    def test_every_class_carries_a_remedy(self):
        _b, fr, _ = self._stack()
        d = self._classify(fr)
        for k in ("stuck", "intermittent_hot", "hot", "dark_current"):
            self.assertIn(k, d["remedy"])
            self.assertTrue(d["remedy"][k])

    def test_result_is_json_serialisable(self):
        base, fr, _ = self._stack()
        fr[:, 10, 10] = 200.0
        d = self._classify(fr, bias=base.astype(np.float32) - 2.0)
        json.dumps(d)


def _report(path, gamma, hot, ladder, noise=1.5, quant=False):
    """A minimal report with the fields --compare reads."""
    r = {
        "device": "/dev/video0", "timestamp_utc": "2026-07-31T00:00:00+00:00",
        "device_name": "test-cam", "width": 1280, "height": 720,
        "measurement_format": "YUYV", "argv": [],
        "processing_path": {"gamma": gamma, "brightness": 0, "contrast": 32},
        "dark": {
            "hot_pixels_6sigma": hot,
            "master_dark_fpn_adu": 0.8,
            "irreducible_temporal_dark_noise_adu": noise,
            "deep_stack_clipped": False,
            "deep_stack_quantiser_limited": quant,
            "quantiser_check": {"distinct_codes_master": 2 if quant else 5000},
            "knee": {"knee_units": 384},
            "dark_current_fit": {"dark_current_adu_per_unit": 0.0001,
                                 "r2": 0.99},
            "ladder": ladder,
        },
    }
    with open(path, "w") as fh:
        json.dump(r, fh)
    return r


def _lad(rows):
    return [{"exposure_units": e, "mean_adu": m, "temporal_noise_adu": n,
             "eff_fps": f, "distinct_codes_stack": c}
            for e, m, n, f, c in rows]


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="camchar_test_")
        self.a = os.path.join(self.tmp, "a.json")
        self.b = os.path.join(self.tmp, "b.json")

    def tearDown(self):
        for p in (self.a, self.b):
            try:
                os.remove(p)
            except OSError:
                pass
        os.rmdir(self.tmp)

    def test_matches_points_and_reports_deltas(self):
        _report(self.a, 100, 12,
                _lad([(256, 18.0, 0.0, 30.0, 1), (512, 18.0, 0.0, 15.0, 1)]),
                noise=0.0, quant=True)
        _report(self.b, 300, 1057,
                _lad([(256, 22.0, 1.4, 30.0, 41), (512, 24.0, 1.5, 15.0, 44)]))
        res = cc.compare_reports(self.a, self.b, verbose=False)
        self.assertEqual([p["exposure_units"] for p in res["per_point"]],
                         [256, 512])
        self.assertEqual(res["per_point"][0]["mean_adu"], [18.0, 22.0])
        self.assertEqual(res["per_point"][1]["eff_fps"], [15.0, 15.0])
        self.assertEqual(res["a"]["hot_pixels"], 12)
        self.assertEqual(res["b"]["hot_pixels"], 1057)
        self.assertEqual(res["processing_path"]["differs"]["gamma"],
                         [100, 300])
        self.assertTrue(res["a"]["quantiser_limited"])
        self.assertFalse(res["b"]["quantiser_limited"])

    def test_unmatched_exposures_are_listed_not_dropped(self):
        _report(self.a, 100, 5, _lad([(256, 18.0, 1.0, 30.0, 40),
                                      (512, 19.0, 1.0, 15.0, 40)]))
        _report(self.b, 100, 5, _lad([(512, 19.0, 1.0, 15.0, 40),
                                      (1024, 20.0, 1.0, 8.0, 40)]))
        res = cc.compare_reports(self.a, self.b, verbose=False)
        self.assertEqual([p["exposure_units"] for p in res["per_point"]], [512])
        self.assertEqual(res["exposures_only_in_a"], [256])
        self.assertEqual(res["exposures_only_in_b"], [1024])

    def test_identical_processing_path_shows_no_diff(self):
        lad = _lad([(256, 18.0, 1.0, 30.0, 40)])
        _report(self.a, 100, 5, lad)
        _report(self.b, 100, 6, lad)
        res = cc.compare_reports(self.a, self.b, verbose=False)
        self.assertEqual(res["processing_path"]["differs"], {})
        self.assertEqual(res["processing_path"]["a_source"], "readback")

    def test_falls_back_to_argv_for_older_reports(self):
        lad = _lad([(256, 18.0, 1.0, 30.0, 40)])
        _report(self.a, 100, 5, lad)
        _report(self.b, 100, 5, lad)
        for p, g in ((self.a, "100"), (self.b, "300")):
            r = json.load(open(p))
            del r["processing_path"]           # pre-stamp report
            r["argv"] = ["--dark", "--gamma", g, "--width", "1280"]
            json.dump(r, open(p, "w"))
        res = cc.compare_reports(self.a, self.b, verbose=False)
        self.assertEqual(res["processing_path"]["a_source"], "argv")
        self.assertEqual(res["processing_path"]["differs"]["gamma"],
                         ["100", "300"])

    def test_verbose_printing_does_not_raise(self):
        # the print path formats floats and Nones from both sides; exercise it
        _report(self.a, 100, 12,
                _lad([(256, 18.0, 0.0, 30.0, 1)]), noise=0.0, quant=True)
        _report(self.b, 300, 1057, _lad([(256, 22.0, 1.4, 30.0, 41)]))
        buf, sys.stdout = sys.stdout, open(os.devnull, "w")
        try:
            cc.compare_reports(self.a, self.b, verbose=True)
        finally:
            sys.stdout.close()
            sys.stdout = buf


class TestProcessingPathParse(unittest.TestCase):
    def test_reads_values_from_list_ctrls_output(self):
        sample = (
            "                     brightness 0x00980900 (int)    : "
            "min=-64 max=64 step=1 default=0 value=3\n"
            "                          gamma 0x00980910 (int)    : "
            "min=72 max=500 step=1 default=100 value=300\n"
            "                    unrelated_x 0x0098091b (int)    : "
            "min=0 max=1 step=1 default=0 value=1\n")
        orig = cc.list_controls
        cc.list_controls = lambda dev: sample
        try:
            got = cc.read_processing_path("/dev/video0")
        finally:
            cc.list_controls = orig
        self.assertEqual(got, {"brightness": 3, "gamma": 300})

    def test_missing_device_yields_empty(self):
        orig = cc.list_controls

        def _boom(dev):
            raise OSError("no such device")
        cc.list_controls = _boom
        try:
            self.assertEqual(cc.read_processing_path("/dev/video9"), {})
        finally:
            cc.list_controls = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
