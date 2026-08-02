#!/usr/bin/env python3
"""Tests for cam_indi.py. Synthetic-first, like test_cam_skymodel: the
stream parser is fed hand-built INDI XML, the FITS decoder hand-built
FITS bytes, and the client driven through an in-memory fake socket so the
mount/camera logic is exercised with no network and no indiserver. Run:
python3 test_cam_indi.py"""

import base64
import struct
import threading
import time
import unittest
import zlib

import numpy as np

import cam_indi as indi


# ----------------------------------------------------------------------------
# Helpers: build a minimal image FITS the way a CCD driver would
# ----------------------------------------------------------------------------

def build_fits(arr, bitpix=16, bzero=32768.0, bscale=1.0):
    """arr is a 2-D array of PHYSICAL values; encode as FITS with the given
    BITPIX and BZERO/BSCALE (default unsigned-16). Returns bytes."""
    h, w = arr.shape
    cards = [
        f"SIMPLE  = {'T':>20}",
        f"BITPIX  = {bitpix:>20}",
        f"NAXIS   = {2:>20}",
        f"NAXIS1  = {w:>20}",
        f"NAXIS2  = {h:>20}",
    ]
    if bzero:
        cards.append(f"BZERO   = {bzero:>20}")
    if bscale != 1.0:
        cards.append(f"BSCALE  = {bscale:>20}")
    cards.append("END")
    hdr = "".join(c.ljust(80) for c in cards)
    hdr = hdr.ljust(((len(hdr) + 2879) // 2880) * 2880).encode("ascii")
    stored = (arr - bzero) / bscale
    dt = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4"}[bitpix]
    body = stored.astype(dt).tobytes()
    body += b"\x00" * ((-len(body)) % 2880)
    return hdr + body


class TestFitsDecode(unittest.TestCase):
    def test_unsigned16_roundtrip(self):
        arr = np.array([[0, 1000, 65535], [32768, 200, 5]], dtype=np.float64)
        out, hdr = indi.read_fits_image(build_fits(arr, bitpix=16))
        self.assertEqual(out.shape, (2, 3))
        np.testing.assert_allclose(out, arr)
        self.assertEqual(int(hdr["BITPIX"]), 16)

    def test_axis_order_is_h_w(self):
        # a 2x5 image (h=2, w=5) must come back shaped (2, 5), i.e. NAXIS1
        # (=5) is the fast/column axis
        arr = np.arange(10, dtype=np.float64).reshape(2, 5) + 1000.0
        out, _ = indi.read_fits_image(build_fits(arr))
        self.assertEqual(out.shape, (2, 5))
        np.testing.assert_allclose(out, arr)

    def test_bscale_applied(self):
        arr = np.array([[100.0, 300.0]], dtype=np.float64)
        out, _ = indi.read_fits_image(
            build_fits(arr, bitpix=16, bzero=0.0, bscale=2.0))
        np.testing.assert_allclose(out, arr)

    def test_float32(self):
        arr = np.array([[1.5, -2.25, 3.0]], dtype=np.float64)
        out, _ = indi.read_fits_image(
            build_fits(arr, bitpix=-32, bzero=0.0))
        np.testing.assert_allclose(out, arr, rtol=1e-6)

    def test_rejects_non_fits(self):
        with self.assertRaises(ValueError):
            indi.read_fits_image(b"not a fits file" * 200)

    def test_rejects_truncated_data(self):
        good = build_fits(np.zeros((4, 4)) + 1000.0)
        with self.assertRaises(ValueError):
            indi.read_fits_image(good[:2880 + 8])   # header ok, data cut

    def test_luma_scales_16bit_to_255_domain(self):
        arr = np.array([[0.0, 65535.0]], dtype=np.float64)
        out, hdr = indi.read_fits_image(build_fits(arr))
        luma = indi.fits_to_luma(out, hdr)
        np.testing.assert_allclose(luma, [[0.0, 255.0]], atol=1e-6)

    def test_luma_averages_color_cube(self):
        # a (2,1,2) cube -> mean over the leading plane axis -> (1,2)
        cube = np.array([[[10.0, 20.0]], [[30.0, 40.0]]])
        luma = indi.fits_to_luma(cube, {"BITPIX": "8"})
        self.assertEqual(luma.shape, (1, 2))
        np.testing.assert_allclose(luma, [[20.0, 30.0]])


class TestNumberParsing(unittest.TestCase):
    def test_decimal(self):
        self.assertAlmostEqual(indi.parse_indi_number(" 123.5 "), 123.5)
        self.assertAlmostEqual(indi.parse_indi_number("-0.25"), -0.25)
        self.assertAlmostEqual(indi.parse_indi_number("1e3"), 1000.0)

    def test_sexagesimal_positive(self):
        self.assertAlmostEqual(indi.parse_indi_number("12:30:00"), 12.5)

    def test_sexagesimal_negative_leading_zero(self):
        # the sign lives on a "-00" field; a per-field sign would lose it
        self.assertAlmostEqual(indi.parse_indi_number("-00:30:00"), -0.5)

    def test_sexagesimal_negative(self):
        self.assertAlmostEqual(indi.parse_indi_number("-05:12:30"),
                               -(5 + 12 / 60 + 30 / 3600))

    def test_unparseable(self):
        self.assertIsNone(indi.parse_indi_number(""))
        self.assertIsNone(indi.parse_indi_number("abc"))


STREAM = (b'<defNumberVector device="M" name="EQUATORIAL_EOD_COORD">'
          b'<defNumber name="RA">12.5</defNumber>'
          b'<defNumber name="DEC">-45.0</defNumber>'
          b'</defNumberVector>'
          b'<message device="M" message="hi"/>')


class TestStreamParser(unittest.TestCase):
    def test_harvests_consecutive_top_level_elements(self):
        # a single TreeBuilder can only ever build ONE element; the parser
        # must start a fresh one per element or this raises "multiple
        # elements on top level" on the second
        p = indi._StreamParser()
        got = p.feed(STREAM)
        self.assertEqual([e.tag for e in got], ["defNumberVector", "message"])
        self.assertEqual(got[0].find("./defNumber[@name='RA']").text, "12.5")
        self.assertEqual(got[1].get("message"), "hi")

    def test_reassembles_across_every_two_chunk_split(self):
        """TCP can split a message at any byte, so every split point must
        still yield both elements with their content intact. This is the
        realistic boundary test; feeding one byte at a time instead tests
        expat's internal event timing (which holds a completed element back
        until more bytes arrive, regardless of buffer_text) rather than
        this parser's reassembly."""
        for cut in range(1, len(STREAM)):
            p = indi._StreamParser()
            got = list(p.feed(STREAM[:cut])) + list(p.feed(STREAM[cut:]))
            with self.subTest(cut=cut):
                self.assertEqual([e.tag for e in got],
                                 ["defNumberVector", "message"])
                self.assertEqual(
                    got[0].find("./defNumber[@name='DEC']").text, "-45.0")
                self.assertEqual(got[1].get("message"), "hi")

    def test_held_element_is_delayed_not_lost(self):
        """Under pathologically small chunks expat may hold the trailing
        element back. Guarantee the weaker-but-sufficient property: it is
        released intact once the next element arrives, so a stalled frame
        is never a corrupt or dropped frame."""
        p = indi._StreamParser()
        got = []
        for i in range(len(STREAM)):
            got.extend(p.feed(STREAM[i:i + 1]))
        got.extend(p.feed(b'<message device="M" message="next"/>'))
        self.assertEqual([e.tag for e in got],
                         ["defNumberVector", "message", "message"])
        self.assertEqual(got[0].find("./defNumber[@name='RA']").text, "12.5")


# ----------------------------------------------------------------------------
# Fake socket: lets IndiClient run its real reader thread with no network
# ----------------------------------------------------------------------------

class FakeSocket:
    """Feeds queued server bytes to recv(); records what the client sends."""

    def __init__(self):
        self._inbuf = bytearray()
        self._cv = threading.Condition()
        self.sent = bytearray()
        self._closed = False

    # -- server side (test drives these) --
    def push(self, data):
        with self._cv:
            self._inbuf.extend(data)
            self._cv.notify_all()

    def sent_text(self):
        return bytes(self.sent).decode("utf-8")

    # -- client side (IndiClient uses these) --
    def recv(self, n):
        with self._cv:
            while not self._inbuf and not self._closed:
                self._cv.wait(0.5)
            if self._closed and not self._inbuf:
                return b""
            chunk = bytes(self._inbuf[:n])
            del self._inbuf[:n]
            return chunk

    def sendall(self, data):
        self.sent.extend(data)

    def settimeout(self, _):
        pass

    def shutdown(self, _):
        pass

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()


def make_client(sock):
    c = indi.IndiClient()
    c._sock = sock
    c._running = True
    c._reader = threading.Thread(target=c._read_loop, daemon=True)
    c._reader.start()
    return c


def wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


class TestClientStore(unittest.TestCase):
    def test_number_and_switch_ingest(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(b'<defNumberVector device="Mnt" '
                   b'name="EQUATORIAL_EOD_COORD" state="Ok">'
                   b'<defNumber name="RA">6.0</defNumber>'
                   b'<defNumber name="DEC">30.0</defNumber>'
                   b'</defNumberVector>'
                   b'<defSwitchVector device="Mnt" name="CONNECTION" '
                   b'state="Ok"><defSwitch name="CONNECT">On</defSwitch>'
                   b'<defSwitch name="DISCONNECT">Off</defSwitch>'
                   b'</defSwitchVector>')
            self.assertTrue(wait_until(
                lambda: c.get_element("Mnt", "EQUATORIAL_EOD_COORD", "RA")
                == 6.0))
            self.assertEqual(
                c.get_element("Mnt", "EQUATORIAL_EOD_COORD", "DEC"), 30.0)
            self.assertIs(c.get_element("Mnt", "CONNECTION", "CONNECT"), True)
            self.assertIs(
                c.get_element("Mnt", "CONNECTION", "DISCONNECT"), False)
            self.assertEqual(c.get_state("Mnt", "CONNECTION"), "Ok")
        finally:
            c.close()

    def test_set_updates_existing_element(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(b'<defNumberVector device="Mnt" name="EQ" state="Ok">'
                   b'<defNumber name="RA">6.0</defNumber></defNumberVector>')
            self.assertTrue(wait_until(
                lambda: c.get_element("Mnt", "EQ", "RA") == 6.0))
            s.push(b'<setNumberVector device="Mnt" name="EQ" state="Busy">'
                   b'<oneNumber name="RA">7.25</oneNumber>'
                   b'</setNumberVector>')
            self.assertTrue(wait_until(
                lambda: c.get_element("Mnt", "EQ", "RA") == 7.25))
            self.assertEqual(c.get_state("Mnt", "EQ"), "Busy")
        finally:
            c.close()

    def test_delproperty_removes(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(b'<defTextVector device="D" name="P" state="Ok">'
                   b'<defText name="T">hello</defText></defTextVector>')
            self.assertTrue(wait_until(
                lambda: c.get_element("D", "P", "T") == "hello"))
            s.push(b'<delProperty device="D" name="P"/>')
            self.assertTrue(wait_until(
                lambda: c.get_vector("D", "P") is None))
        finally:
            c.close()

    def test_wait_property_blocks_then_returns(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            def push_later():
                time.sleep(0.05)
                s.push(b'<defNumberVector device="X" name="Y" state="Ok">'
                       b'<defNumber name="Z">1.0</defNumber>'
                       b'</defNumberVector>')
            threading.Thread(target=push_later, daemon=True).start()
            v = c.wait_property("X", "Y", timeout=2.0)
            self.assertIsNotNone(v)
            self.assertEqual(v["elements"]["Z"], 1.0)
        finally:
            c.close()


class TestClientCommands(unittest.TestCase):
    def test_set_switch_serialization(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            c.set_switch("Mnt", "ON_COORD_SET",
                         {"SLEW": False, "TRACK": True, "SYNC": False})
            self.assertTrue(wait_until(lambda: "ON_COORD_SET" in s.sent_text()))
            txt = s.sent_text()
            self.assertIn('newSwitchVector device="Mnt" name="ON_COORD_SET"',
                          txt)
            self.assertIn('<oneSwitch name="TRACK">On</oneSwitch>', txt)
            self.assertIn('<oneSwitch name="SLEW">Off</oneSwitch>', txt)
        finally:
            c.close()

    def test_enable_blob_serialization(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            c.enable_blob("Cam", "Also")
            self.assertTrue(wait_until(lambda: "enableBLOB" in s.sent_text()))
            self.assertIn('<enableBLOB device="Cam">Also</enableBLOB>',
                          s.sent_text())
        finally:
            c.close()


class TestMount(unittest.TestCase):
    def test_radec_hours_to_degrees(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(b'<defNumberVector device="Mnt" '
                   b'name="EQUATORIAL_EOD_COORD" state="Ok">'
                   b'<defNumber name="RA">6.0</defNumber>'
                   b'<defNumber name="DEC">-20.0</defNumber>'
                   b'</defNumberVector>')
            m = indi.IndiMount(c, "Mnt")
            self.assertTrue(wait_until(lambda: m.radec_deg() is not None))
            ra, dec = m.radec_deg()
            self.assertAlmostEqual(ra, 90.0)      # 6h * 15 = 90 deg
            self.assertAlmostEqual(dec, -20.0)
        finally:
            c.close()

    def test_sync_sends_degrees_as_hours(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            m = indi.IndiMount(c, "Mnt")
            m.sync(90.0, -20.0)                    # 90 deg -> 6h on the wire
            self.assertTrue(wait_until(
                lambda: "EQUATORIAL_EOD_COORD" in s.sent_text()))
            txt = s.sent_text()
            self.assertIn('<oneSwitch name="SYNC">On</oneSwitch>', txt)
            self.assertIn('<oneSwitch name="SLEW">Off</oneSwitch>', txt)
            # RA element carries 6.0 hours, DEC carries -20.0
            self.assertRegex(txt, r'name="RA">6\.0')
            self.assertRegex(txt, r'name="DEC">-20\.0')
        finally:
            c.close()

    def test_is_slewing_tracks_state(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(b'<defNumberVector device="Mnt" '
                   b'name="EQUATORIAL_EOD_COORD" state="Busy">'
                   b'<defNumber name="RA">6.0</defNumber>'
                   b'<defNumber name="DEC">0.0</defNumber>'
                   b'</defNumberVector>')
            m = indi.IndiMount(c, "Mnt")
            self.assertTrue(wait_until(lambda: m.is_slewing()))
            s.push(b'<setNumberVector device="Mnt" '
                   b'name="EQUATORIAL_EOD_COORD" state="Ok">'
                   b'</setNumberVector>')
            self.assertTrue(wait_until(lambda: not m.is_slewing()))
        finally:
            c.close()


class TestCameraSource(unittest.TestCase):
    def _blob_stream(self, device, prop, fits_bytes, compressed=False):
        payload = zlib.compress(fits_bytes) if compressed else fits_bytes
        fmt = ".fits.z" if compressed else ".fits"
        b64 = base64.b64encode(payload).decode("ascii")
        return (f'<setBLOBVector device="{device}" name="{prop}" '
                f'state="Ok"><oneBLOB name="{prop}" size="{len(fits_bytes)}" '
                f'format="{fmt}">{b64}</oneBLOB></setBLOBVector>'
                ).encode("ascii")

    def test_exposure_returns_luma_frame(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            arr = np.array([[0.0, 65535.0], [32768.0, 100.0]])
            fits = build_fits(arr)
            src = indi.IndiCameraSource(c, "Cam", exposure_scale=1e-3)
            src.exposure = 1000               # 1000 * 1e-3 = 1.0 s

            def serve():
                # wait for the client's CCD_EXPOSURE command, then deliver
                wait_until(lambda: "CCD_EXPOSURE" in s.sent_text(),
                           timeout=2.0)
                s.push(self._blob_stream("Cam", "CCD1", fits))
            threading.Thread(target=serve, daemon=True).start()

            path, total, full = src.capture_run(1, timeout=3.0, verbose=False)
            self.assertEqual((total, full), (1, 1))
            frames = list(src.iter_luma(path, 1))
            self.assertEqual(len(frames), 1)
            np.testing.assert_allclose(frames[0],
                                       [[0.0, 255.0],
                                        [32768.0 / 257.0, 100.0 / 257.0]],
                                       atol=1e-6)
            self.assertIn('name="CCD_EXPOSURE_VALUE"', s.sent_text())
            self.assertRegex(s.sent_text(), r'CCD_EXPOSURE_VALUE">1\.0')
        finally:
            c.close()

    def test_compressed_blob(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            arr = np.array([[1000.0, 2000.0]])
            fits = build_fits(arr)
            src = indi.IndiCameraSource(c, "Cam")
            src.exposure = 500

            def serve():
                wait_until(lambda: "CCD_EXPOSURE" in s.sent_text(),
                           timeout=2.0)
                s.push(self._blob_stream("Cam", "CCD1", fits, compressed=True))
            threading.Thread(target=serve, daemon=True).start()

            frame = src.expose(0.5, timeout=3.0)
            np.testing.assert_allclose(frame,
                                       [[1000.0 / 257.0, 2000.0 / 257.0]],
                                       atol=1e-6)
        finally:
            c.close()

    def test_exposure_timeout(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            src = indi.IndiCameraSource(c, "Cam")
            with self.assertRaises(TimeoutError):
                src.expose(0.1, timeout=0.2)      # no BLOB ever arrives
        finally:
            c.close()

    def test_dead_reader_is_named_in_the_error(self):
        """The reader thread cannot raise into the caller, so a broken
        stream must not surface as a bare 'no BLOB' timeout -- that sends
        you hunting for a camera fault when the connection is the problem."""
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(b'<defNumberVector device="Cam" name="P"></oops>')
            self.assertTrue(wait_until(lambda: c.last_error() is not None))
            src = indi.IndiCameraSource(c, "Cam")
            with self.assertRaises(RuntimeError) as cm:
                src.expose(0.1, timeout=1.0)
            self.assertIn("reader stopped", str(cm.exception))
            self.assertFalse(c.is_alive())
        finally:
            c.close()

    def test_server_hangup_is_named_in_the_error(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.close()                             # server drops the socket
            self.assertTrue(wait_until(lambda: not c.is_alive()))
            src = indi.IndiCameraSource(c, "Cam")
            with self.assertRaises(RuntimeError) as cm:
                src.expose(0.1, timeout=1.0)
            self.assertIn("connection closed by server", str(cm.exception))
        finally:
            c.close()


CONNECTED = (b'<defSwitchVector device="Cam" name="CONNECTION" state="Ok">'
             b'<defSwitch name="CONNECT">On</defSwitch>'
             b'<defSwitch name="DISCONNECT">Off</defSwitch>'
             b'</defSwitchVector>')
CCD_EXPOSURE = (b'<defNumberVector device="Cam" name="CCD_EXPOSURE" '
                b'state="Idle"><defNumber name="CCD_EXPOSURE_VALUE">0'
                b'</defNumber></defNumberVector>')


class TestConnectCamera(unittest.TestCase):
    def test_waits_for_ccd_exposure_and_enables_blobs(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(CONNECTED + CCD_EXPOSURE
                   + b'<defNumberVector device="Cam" name="CCD_INFO" '
                     b'state="Ok">'
                     b'<defNumber name="CCD_MAX_X">640</defNumber>'
                     b'<defNumber name="CCD_MAX_Y">480</defNumber>'
                     b'</defNumberVector>')
            src = indi.connect_camera(c, "Cam", timeout=2.0)
            self.assertEqual(src.frame_shape(), (480, 640))
            self.assertIn('<enableBLOB device="Cam">Also</enableBLOB>',
                          s.sent_text())
        finally:
            c.close()

    def test_missing_ccd_exposure_fails_fast_and_says_why(self):
        """A driver that connects but never defines CCD_EXPOSURE must fail
        here, not silently hand back a source whose first exposure hangs
        for its whole timeout."""
        s = FakeSocket()
        c = make_client(s)
        try:
            s.push(CONNECTED)                     # no CCD_EXPOSURE ever
            with self.assertRaises(RuntimeError) as cm:
                indi.connect_camera(c, "Cam", timeout=0.3)
            self.assertIn("CCD_EXPOSURE", str(cm.exception))
        finally:
            c.close()

    def test_unset_exposure_is_a_caller_error(self):
        s = FakeSocket()
        c = make_client(s)
        try:
            src = indi.IndiCameraSource(c, "Cam")
            with self.assertRaises(ValueError) as cm:
                src.capture_run(1, verbose=False)
            self.assertIn("exposure", str(cm.exception))
            self.assertNotIn("CCD_EXPOSURE_VALUE", s.sent_text())
        finally:
            c.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
