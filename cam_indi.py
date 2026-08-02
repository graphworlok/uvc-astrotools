#!/usr/bin/env python3
"""
cam_indi.py -- minimal INDI client for cam_observe.py.

Speaks the INDI XML/BLOB protocol directly over a TCP socket (indiserver
listens on :7624 by default) -- no pyindi-client / SWIG / libindi build,
the same hand-rolled-subset approach cam_skymodel takes for FITS and
cam_observe takes for the Stellarium protocol. INDI is native on the same
Linux host this tool already needs for v4l2, so an INDI source is a peer
of the v4l2 path, not a remote workaround; the client also works against
a remote indiserver over TCP unchanged.

Three consumers:

  * read_fits_image      -- decode a primary-HDU image FITS (the form a
                            CCD driver hands back as a BLOB) to a float64
                            array. Handles BITPIX 8/16/32/-32/-64 and
                            BZERO/BSCALE, the unsigned-16 (BZERO=32768)
                            case included.
  * IndiCameraSource     -- a CCD exposure source matching cam_characterise
                            .Capture's duck type (.exposure / capture_run /
                            iter_luma / close), so cam_observe's stacking
                            pipeline ingests INDI frames with no change.
  * IndiMount            -- read the mount's RA/Dec (and pier side / alt-az
                            when the driver exposes them) and command
                            SLEW / SYNC / ABORT, so the polar-alignment and
                            pointing features can close the loop against a
                            real mount instead of inferring from star motion.

INDI conventions worth stating because they are exactly where bugs hide:

  * EQUATORIAL_EOD_COORD carries RA in HOURS (0-24) and DEC in degrees.
    The rest of this project is degrees throughout, so IndiMount converts
    at the boundary (RA_hours*15 <-> RA_deg) and never lets hours leak in.
  * A client receives NO BLOB data until it sends <enableBLOB>. A camera
    that never delivers a frame is almost always a missing enableBLOB.
  * Setting EQUATORIAL_EOD_COORD SLEWS, SYNCS or TRACKS depending on the
    ON_COORD_SET switch; IndiMount sets that switch explicitly before each
    coordinate command rather than trusting the driver's current mode.
  * Number values may arrive in sexagesimal text ("12:30:00"), not only
    decimal; parse_indi_number handles both.
  * The wire is a rootless CONCATENATION of elements (<defX>/<setX>/...),
    not one document; _StreamParser primes a synthetic never-closed root
    and harvests each top-level element as its end tag arrives.
"""

import base64
import re
import socket
import threading
import time
import zlib
import xml.parsers.expat as expat
from xml.etree.ElementTree import TreeBuilder
from xml.sax.saxutils import quoteattr

import numpy as np

INDI_DEFAULT_PORT = 7624
INDI_PROTOCOL_VERSION = "1.7"


# ----------------------------------------------------------------------------
# Value parsing
# ----------------------------------------------------------------------------

def parse_indi_number(text):
    """INDI number text -> float, or None if unparseable. Accepts plain
    decimal ("123.4", "-0.5", "1e3") and the sexagesimal forms drivers
    use for angles ("12:30:00", "-05:12:30" -- ':' or ';' separated,
    each field 1/60 of the one before). The sign is taken from the whole
    string, so "-00:30:00" is correctly -0.5, not +0.5 (a naive per-field
    sign loses it on a "-00" leading field)."""
    s = text.strip()
    if not s:
        return None
    if ":" in s or ";" in s:
        neg = s.lstrip().startswith("-")
        parts = re.split(r"[:;]", s)
        acc = 0.0
        try:
            for i, p in enumerate(parts):
                acc += abs(float(p)) / (60.0 ** i)
        except ValueError:
            return None
        return -acc if neg else acc
    try:
        return float(s)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# FITS image decode (primary HDU)
# ----------------------------------------------------------------------------

_FITS_DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8",
               -32: ">f4", -64: ">f8"}


def read_fits_image(data):
    """Decode a primary-HDU image FITS byte string to (array, header).

    array is float64 with BSCALE/BZERO applied and shaped (..., NAXIS2,
    NAXIS1) -- FITS stores NAXIS1 (columns/x) fastest, so the numpy shape
    is the reversed axis list, giving the natural (h, w) for a 2-D image.
    header is the parsed card dict (string values, comments stripped).

    Only what a CCD driver emits is supported: a single image HDU, BITPIX
    in {8,16,32,64,-32,-64}, optional BZERO/BSCALE (the unsigned-16
    BZERO=32768 convention included). Raises ValueError on anything that
    is not a well-formed FITS image, so a truncated or non-FITS BLOB is a
    clean error, not a silent misparse."""
    if len(data) < 2880 or data[:6] != b"SIMPLE":
        raise ValueError("not a FITS stream (no SIMPLE card)")
    header = {}
    pos = 0
    saw_end = False
    while not saw_end:
        block = data[pos:pos + 2880]
        if len(block) < 2880:
            raise ValueError("truncated FITS header")
        for i in range(0, 2880, 80):
            card = block[i:i + 80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                saw_end = True
                break
            # value-indicator is '= ' in columns 9-10 (0-based 8-9)
            if card[8:10] == "= ":
                val = card[10:]
                # a quoted string value can contain '/', so only split the
                # comment off an unquoted (numeric/logical) value
                v = val.strip()
                if v.startswith("'"):
                    end = val.find("'", val.find("'") + 1)
                    header[key] = val[val.find("'") + 1:end].strip()
                else:
                    header[key] = v.split("/")[0].strip()
        pos += 2880
    try:
        bitpix = int(header["BITPIX"])
        naxis = int(header["NAXIS"])
    except (KeyError, ValueError):
        raise ValueError("FITS header missing BITPIX/NAXIS")
    if bitpix not in _FITS_DTYPE:
        raise ValueError(f"unsupported BITPIX {bitpix}")
    if naxis < 2:
        raise ValueError(f"not an image (NAXIS={naxis})")
    dims = []
    for k in range(1, naxis + 1):
        dims.append(int(header[f"NAXIS{k}"]))
    if any(d <= 0 for d in dims):        # a 0-length axis => no data HDU
        raise ValueError("FITS image has a zero-length axis")
    bzero = float(header.get("BZERO", 0.0))
    bscale = float(header.get("BSCALE", 1.0))
    count = 1
    for d in dims:
        count *= d
    nbytes = count * abs(bitpix) // 8
    body = data[pos:pos + nbytes]
    if len(body) < nbytes:
        raise ValueError("truncated FITS data segment")
    arr = np.frombuffer(body, dtype=_FITS_DTYPE[bitpix]).astype(np.float64)
    if bscale != 1.0 or bzero != 0.0:
        arr = arr * bscale + bzero
    return arr.reshape(list(reversed(dims))), header


def fits_to_luma(arr, header):
    """Reduce a decoded FITS image to a 2-D luma frame in the 0-255 float
    domain the rest of the pipeline works in (matching cam_characterise's
    Y16 path: 65535/257 = 255 exactly). Colour/planar cubes (NAXIS3) are
    averaged to luma; an already-2-D mono frame passes straight through.
    Integer frames are scaled by their type's full range so an 8-bit and
    a 16-bit camera land on the same scale; float frames (BITPIX<0, in
    physical/normalised units the driver chose) are passed through and
    only clipped negative."""
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1]).mean(axis=0)
    try:
        bitpix = int(header.get("BITPIX", "16"))
    except ValueError:
        bitpix = 16
    if bitpix > 0:
        typemax = float((1 << bitpix) - 1)
        if typemax > 255.0:
            arr = arr * (255.0 / typemax)
    else:
        arr = np.clip(arr, 0.0, None)
    return arr.astype(np.float64)


# ----------------------------------------------------------------------------
# Wire protocol: rootless element stream
# ----------------------------------------------------------------------------

class _StreamParser:
    """Feed INDI's rootless element stream; return complete top-level
    elements. The server sends a concatenation of <defX>/<setX>/<message>/
    <delProperty> with no enclosing document, so a synthetic never-closed
    <indi> root is primed and each depth-1 child assembled and handed back
    as its end tag arrives -- nothing accumulates, because each element
    gets its OWN TreeBuilder and is released as soon as it completes,
    rather than being retained under a growing root.

    A TreeBuilder builds exactly one element: once its top-level element
    has ended it treats the document as closed and raises "multiple
    elements on top level" on the next start tag. Since INDI's stream is
    precisely a run of sibling top-level elements, a fresh builder is
    created for each one.

    On expat's event timing, measured rather than assumed: expat can hold
    a completed element's end event back until more bytes arrive, and
    buffer_text makes NO difference to this (both settings behave
    identically, and neither "emits each end tag the instant it is
    parsed"). Nor can a nudge force a flush -- feeding ignorable trailing
    whitespace does not release a held element.

    The saving grace is that the hold only appears under sustained tiny
    chunks (feeding a message 1-8 bytes at a time); it is not reachable
    from a recv(65536) on a real socket. Every one of the 196 possible
    two-chunk splits of a def+BLOB stream yields both elements, and the
    element is only ever DELAYED, never lost or corrupted -- the held
    element (payload intact) is released as soon as the next element
    arrives. So a trailing BLOB does not stall in practice, and if it
    ever did the frame would still be correct when it landed."""

    def __init__(self):
        self._tb = None                    # one per top-level element
        self._depth = 0
        self._ready = []
        self._p = expat.ParserCreate()
        self._p.buffer_text = False
        self._p.StartElementHandler = self._on_start
        self._p.EndElementHandler = self._on_end
        self._p.CharacterDataHandler = self._on_data
        self._p.Parse(b"<indi>", 0)        # prime the synthetic root

    def _on_start(self, name, attrs):
        self._depth += 1
        if self._depth == 2:               # a new top-level element begins
            self._tb = TreeBuilder()
        if self._depth > 1:                # skip the synthetic <indi> root
            self._tb.start(name, attrs)

    def _on_end(self, name):
        self._depth -= 1
        if self._depth >= 1:
            elem = self._tb.end(name)
            if self._depth == 1:           # a completed top-level element
                self._ready.append(elem)

    def _on_data(self, data):
        if self._depth > 1:                # ignore inter-element whitespace
            self._tb.data(data)

    def feed(self, data):
        """Parse a chunk; return the list of top-level elements it
        completed (possibly empty). Raises expat.ExpatError on malformed
        XML so the reader can surface it and reconnect."""
        self._p.Parse(data, 0)
        out = self._ready
        self._ready = []
        return out


def _new_vector(kind, device, name, one_tag, elements, fmt=str):
    parts = [f"<new{kind}Vector device={quoteattr(device)} "
             f"name={quoteattr(name)}>"]
    for k, v in elements.items():
        parts.append(f"<one{one_tag} name={quoteattr(k)}>"
                     f"{fmt(v)}</one{one_tag}>")
    parts.append(f"</new{kind}Vector>")
    return "".join(parts).encode("utf-8")


# ----------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------

class IndiClient:
    """Threaded INDI client: one reader thread parses the server stream
    into a property store; senders and waiters run on the caller's threads.
    All published state (the store, the BLOB latches) is guarded by one
    Condition, and every waiter re-checks its predicate on wake, so a
    capture that blocks for a BLOB and a UI thread reading the mount's
    RA/Dec never trip over each other."""

    def __init__(self, host="127.0.0.1", port=INDI_DEFAULT_PORT):
        self.host = host
        self.port = int(port)
        self._sock = None
        self._reader = None
        self._cond = threading.Condition()
        self._store = {}          # device -> prop -> {type,state,elements}
        self._blobs = {}          # (device, prop) -> {format,data,element}
        self._blob_seq = {}       # (device, prop) -> monotonically++ on BLOB
        self._messages = []       # (device, text) log, most recent last
        self._running = False
        self._error = None        # last fatal reader error, if any

    # -- lifecycle --------------------------------------------------------

    def connect(self, timeout=5.0):
        """Open the socket, start the reader, and ask the server to define
        all properties. Returns self. Raises OSError if the socket cannot
        be opened."""
        self._sock = socket.create_connection((self.host, self.port),
                                               timeout=timeout)
        self._sock.settimeout(None)
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.send(b'<getProperties version="%s"/>'
                  % INDI_PROTOCOL_VERSION.encode())
        return self

    def close(self):
        self._running = False
        try:
            if self._sock is not None:
                self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass

    def send(self, payload):
        if self._sock is None:
            raise RuntimeError("INDI client not connected")
        self._sock.sendall(payload)

    def _read_loop(self):
        parser = _StreamParser()
        try:
            while self._running:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                for elem in parser.feed(chunk):
                    self._handle(elem)
        except (OSError, expat.ExpatError) as e:
            self._error = str(e)
        finally:
            with self._cond:
                self._running = False
                self._cond.notify_all()

    # -- inbound handling -------------------------------------------------

    _VEC_RE = re.compile(r"(def|set|new)(Number|Switch|Text|Light|BLOB)"
                         r"Vector$")

    def _handle(self, elem):
        tag = elem.tag
        if tag == "message":
            msg = elem.get("message")
            if msg:
                with self._cond:
                    self._messages.append((elem.get("device"), msg))
                    del self._messages[:-200]
            return
        if tag == "delProperty":
            dev, name = elem.get("device"), elem.get("name")
            with self._cond:
                if dev in self._store:
                    if name:
                        self._store[dev].pop(name, None)
                    else:
                        self._store.pop(dev, None)
                self._cond.notify_all()
            return
        m = self._VEC_RE.match(tag)
        if not m:
            return
        vtype = m.group(2).lower()
        device, name = elem.get("device"), elem.get("name")
        if not device or not name:
            return
        state = elem.get("state")
        with self._cond:
            vec = self._store.setdefault(device, {}).setdefault(name, {})
            vec["type"] = vtype
            if state:
                vec["state"] = state
            els = vec.setdefault("elements", {})
            for child in elem:
                cname = child.get("name")
                if not cname:
                    continue
                text = child.text or ""
                if vtype == "number":
                    v = parse_indi_number(text)
                    if v is not None:
                        els[cname] = v
                elif vtype == "switch":
                    els[cname] = (text.strip().lower() == "on")
                elif vtype == "blob":
                    self._store_blob(device, name, cname, child)
                else:                       # text / light
                    els[cname] = text.strip()
            self._cond.notify_all()

    def _store_blob(self, device, prop, elem_name, child):
        b64 = (child.text or "").strip()
        if not b64:
            return
        fmt = (child.get("format") or "").strip()
        raw = base64.b64decode(b64)
        if fmt.endswith(".z"):              # zlib-compressed payload
            raw = zlib.decompress(raw)
        key = (device, prop)
        self._blobs[key] = {"format": fmt, "data": raw, "element": elem_name}
        self._blob_seq[key] = self._blob_seq.get(key, 0) + 1

    # -- store access (thread-safe reads) ---------------------------------

    def get_vector(self, device, prop):
        with self._cond:
            v = self._store.get(device, {}).get(prop)
            return dict(v) if v else None

    def get_element(self, device, prop, elem):
        with self._cond:
            return self._store.get(device, {}).get(prop, {}) \
                .get("elements", {}).get(elem)

    def get_state(self, device, prop):
        with self._cond:
            return self._store.get(device, {}).get(prop, {}).get("state")

    def devices(self):
        with self._cond:
            return sorted(self._store)

    def is_alive(self):
        return self._running

    def last_error(self):
        """The reader thread's fatal error, or None. The reader dies on a
        socket or XML error and can only set this flag -- it cannot raise
        into the caller's thread -- so every timeout path reports it.
        Without that, a dead connection is indistinguishable from a camera
        that is merely slow."""
        return self._error

    def _why_dead(self):
        """Suffix naming the cause when the reader has stopped, else ''."""
        if self._error:
            return f"; reader stopped: {self._error}"
        if not self._running:
            return "; connection closed by server"
        return ""

    # -- waiting ----------------------------------------------------------

    def wait_property(self, device, prop, timeout=5.0):
        """Block until (device, prop) has been defined by the server, or
        the timeout elapses. Returns the vector dict or None."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while device not in self._store \
                    or prop not in self._store.get(device, {}):
                if not self._running:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return dict(self._store[device][prop])

    def wait_state(self, device, prop, states, timeout=30.0):
        """Block until (device, prop)'s state is one of `states`
        (e.g. {"Ok","Alert"} to wait out a Busy slew). Returns the final
        state, or None on timeout/disconnect."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                st = self._store.get(device, {}).get(prop, {}).get("state")
                if st in states:
                    return st
                if not self._running:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    # -- outbound commands ------------------------------------------------

    def set_number(self, device, prop, elements):
        self.send(_new_vector("Number", device, prop, "Number", elements,
                              fmt=lambda v: repr(float(v))))

    def set_switch(self, device, prop, elements):
        self.send(_new_vector("Switch", device, prop, "Switch", elements,
                              fmt=lambda v: "On" if v else "Off"))

    def set_text(self, device, prop, elements):
        self.send(_new_vector("Text", device, prop, "Text", elements))

    def enable_blob(self, device, mode="Also"):
        """Ask the server to deliver this device's BLOBs on THIS connection
        ('Also' = alongside other updates, 'Only' = BLOBs only, 'Never' =
        off). Without it a camera's frames never arrive."""
        self.send(f"<enableBLOB device={quoteattr(device)}>{mode}"
                  "</enableBLOB>".encode("utf-8"))

    def connect_device(self, device, timeout=10.0):
        """Drive CONNECTION.CONNECT and wait for the driver to come up
        (state Ok). Returns True on success."""
        if self.wait_property(device, "CONNECTION", timeout=timeout) is None:
            return False
        if self.get_element(device, "CONNECTION", "CONNECT") is True:
            return True
        self.set_switch(device, "CONNECTION",
                        {"CONNECT": True, "DISCONNECT": False})
        st = self.wait_state(device, "CONNECTION", {"Ok", "Alert"},
                             timeout=timeout)
        return st == "Ok" or \
            self.get_element(device, "CONNECTION", "CONNECT") is True


# ----------------------------------------------------------------------------
# Mount
# ----------------------------------------------------------------------------

class IndiMount:
    """Read and command an INDI telescope. RA is degrees at this boundary
    in BOTH directions -- the hours<->degrees conversion against
    EQUATORIAL_EOD_COORD lives here and nowhere else, so callers stay in
    the project's degrees-everywhere convention."""

    EQ = "EQUATORIAL_EOD_COORD"
    ON_SET = "ON_COORD_SET"

    def __init__(self, client, device):
        self.client = client
        self.device = device

    def connect(self, timeout=10.0):
        return self.client.connect_device(self.device, timeout=timeout)

    def radec_deg(self):
        """(ra_deg, dec_deg) of the current pointing, or None if the mount
        has not reported EQUATORIAL_EOD_COORD yet."""
        ra_h = self.client.get_element(self.device, self.EQ, "RA")
        dec = self.client.get_element(self.device, self.EQ, "DEC")
        if ra_h is None or dec is None:
            return None
        return (ra_h * 15.0) % 360.0, float(dec)

    def altaz_deg(self):
        """(alt_deg, az_deg) if the driver publishes HORIZONTAL_COORD, else
        None (many mounts only expose equatorial)."""
        alt = self.client.get_element(self.device, "HORIZONTAL_COORD", "ALT")
        az = self.client.get_element(self.device, "HORIZONTAL_COORD", "AZ")
        if alt is None or az is None:
            return None
        return float(alt), float(az)

    def pier_side(self):
        e = self.client.get_element(self.device, "TELESCOPE_PIER_SIDE",
                                    "PIER_EAST")
        if e is None:
            return None
        return "east" if e else "west"

    def is_slewing(self):
        return self.client.get_state(self.device, self.EQ) == "Busy"

    def _go(self, mode, ra_deg, dec_deg):
        self.client.set_switch(self.device, self.ON_SET,
                               {"SLEW": mode == "SLEW",
                                "TRACK": mode == "TRACK",
                                "SYNC": mode == "SYNC"})
        self.client.set_number(self.device, self.EQ,
                               {"RA": (ra_deg % 360.0) / 15.0,
                                "DEC": dec_deg})

    def slew(self, ra_deg, dec_deg):
        """Slew to (ra_deg, dec_deg) and stop tracking there."""
        self._go("SLEW", ra_deg, dec_deg)

    def track(self, ra_deg, dec_deg):
        """Slew to (ra_deg, dec_deg) and track sidereally from there."""
        self._go("TRACK", ra_deg, dec_deg)

    def sync(self, ra_deg, dec_deg):
        """Tell the mount its current pointing IS (ra_deg, dec_deg) --
        no motion, just re-anchors its model. This is the one to feed a
        fresh plate solve: it makes the mount's RA/Dec agree with the
        sky truth without moving anything."""
        self._go("SYNC", ra_deg, dec_deg)

    def abort(self):
        self.client.set_switch(self.device, "TELESCOPE_ABORT_MOTION",
                               {"ABORT": True})


# ----------------------------------------------------------------------------
# Camera source (cam_characterise.Capture duck type)
# ----------------------------------------------------------------------------

class IndiCameraSource:
    """A CCD exposure source shaped like cam_characterise.Capture, so
    cam_observe's workers drive it unchanged: set .exposure, capture_run()
    a burst, iter_luma() the frames.

    .exposure is the same integer field the UVC path uses, so one exposure
    entry serves both sources; exposure_scale is the seconds-per-unit that
    makes that literally true. It defaults to 1e-4 to match v4l2's
    exposure_time_absolute, which is DEFINED in 100us units -- pick 1e-3
    here and the same UI number silently means a 10x longer exposure on
    the INDI path than on the UVC one. Unlike the UVC path there is no warm-up
    frame to throw away -- a CCD's first frame is real and every exposure
    costs real seconds -- so `discard` is accepted for interface
    compatibility and ignored: capture_run(n) takes exactly n exposures."""

    def __init__(self, client, device, blob_prop="CCD1", exposure_scale=1e-4,
                 frame_shape=None):
        self.client = client
        self.device = device
        self.blob_prop = blob_prop
        self.exposure_scale = float(exposure_scale)
        self.exposure = None            # set by caller (same field as UVC)
        self._frames = []
        self._shape = frame_shape       # (h, w) once a frame has arrived
        self.last_fps = 0.0
        self.last_whole_frames = 0

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def exposure_seconds(self):
        return (self.exposure or 0) * self.exposure_scale

    def expose(self, seconds, timeout):
        """Trigger one exposure and return its luma frame (2-D float64,
        0-255 domain). Blocks up to `timeout` seconds for the BLOB."""
        key = (self.device, self.blob_prop)
        with self.client._cond:
            start_seq = self.client._blob_seq.get(key, 0)
        self.client.set_number(self.device, "CCD_EXPOSURE",
                               {"CCD_EXPOSURE_VALUE": seconds})
        deadline = time.monotonic() + timeout
        with self.client._cond:
            while self.client._blob_seq.get(key, 0) == start_seq:
                if not self.client._running:
                    raise RuntimeError("INDI connection lost during exposure"
                                       + self.client._why_dead())
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"no BLOB from {self.device}/{self.blob_prop} "
                        f"within {timeout:.0f}s (enableBLOB sent? camera "
                        "connected and exposing?)"
                        + self.client._why_dead())
                self.client._cond.wait(remaining)
            blob = dict(self.client._blobs[key])
        arr, header = read_fits_image(blob["data"])
        luma = fits_to_luma(arr, header)
        self._shape = luma.shape
        return luma

    def capture_run(self, n, discard=1, timeout=20.0, verbose=True):
        """Take n exposures, buffering the luma frames for iter_luma.
        Returns (token, total, full) to match the Capture contract; the
        token is a sentinel (there is no on-disk file -- frames live in
        memory, one burst at a time)."""
        # an unset .exposure would otherwise request a 0-second exposure --
        # which a driver either rejects or answers with a useless frame,
        # both of which read as a camera fault rather than a caller bug
        if not self.exposure_seconds > 0:
            raise ValueError(
                f"{self.device}: .exposure is {self.exposure!r}; set it "
                f"(units of {self.exposure_scale}s) before capturing")
        per = max(timeout, self.exposure_seconds * 2.0 + 15.0)
        t0 = time.monotonic()
        self._frames = []
        for _ in range(n):
            self._frames.append(self.expose(self.exposure_seconds, per))
        dt = time.monotonic() - t0
        self.last_fps = (n / dt) if dt > 0 else 0.0
        self.last_whole_frames = len(self._frames)
        if verbose:
            print(f"      [INDI {n} exposure(s) in {dt:.2f}s]")
        return "<indi>", n, len(self._frames)

    def iter_luma(self, path, n, discard=1):
        """Yield the frames capture_run buffered (discard ignored -- see
        the class docstring; capture_run already took exactly the frames
        that matter)."""
        for fr in self._frames[:n]:
            yield fr

    def frame_shape(self):
        return self._shape

    def close(self):
        self._frames = []


def connect_camera(client, device, blob_prop="CCD1", exposure_scale=1e-4,
                   timeout=10.0):
    """Connect a CCD device, enable its BLOBs, and return a ready
    IndiCameraSource. Raises RuntimeError if the device will not connect."""
    if not client.connect_device(device, timeout=timeout):
        raise RuntimeError(f"INDI camera '{device}' did not connect"
                           + client._why_dead())
    client.enable_blob(device, "Also")
    # a newly-connected driver defines CCD_EXPOSURE asynchronously; a
    # newNumberVector that arrives before the definition is dropped by the
    # server, so the first exposure would hang for its whole timeout with
    # nothing to show for it. Wait for the property, not just the CONNECT.
    if client.wait_property(device, "CCD_EXPOSURE", timeout=timeout) is None:
        raise RuntimeError(
            f"INDI camera '{device}' connected but never defined "
            f"CCD_EXPOSURE within {timeout:.0f}s (is it really a CCD "
            f"driver?)" + client._why_dead())
    src = IndiCameraSource(client, device, blob_prop=blob_prop,
                           exposure_scale=exposure_scale)
    # CCD_INFO, when present, gives the native frame size before the first
    # exposure -- lets the caller lay out its resolution UI immediately
    w = client.get_element(device, "CCD_INFO", "CCD_MAX_X")
    h = client.get_element(device, "CCD_INFO", "CCD_MAX_Y")
    if w and h:
        src._shape = (int(h), int(w))
    return src
