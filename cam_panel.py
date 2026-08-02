#!/usr/bin/env python3
"""
cam_panel.py  --  the software flat panel, shared by cam_observe and
cam_characterise.

A uniform grey field is the cheapest controllable light source there is, and
both tools want it for different reasons: cam_observe servos it to a target
signal level to acquire master flats, cam_characterise steps it through a
ladder to read the ISP transfer function straight off the sensor. Keeping one
implementation means the stimulus is identical in both, so a transfer curve
measured by one is directly comparable to a flat taken by the other.

Contents: the panel page, the test-image page, and PanelServer, which serves
both over HTTP so a browser on ANY device -- a desktop monitor, a tablet
clamped in front of the objective -- becomes the source.

The server is READ-ONLY by design: the network can ask what to display and
acknowledge what it painted, and nothing more. The level is owned by the
process that constructed the server (via the get_level callback), so a sweep
driver sets it in-process and the browser follows.

cam_observe imports cam_characterise, so this cannot live in either of them
without a circular import -- hence a module of its own.
"""

import http.server
import json
import socket
import threading
import time
import urllib.parse

# Self-contained page. Triggered refresh: long-polls /level (the server
# holds the request until the level changes, so updates land immediately,
# not on a poll grid) and ACKS each applied level via /shown so the
# auto-level servo can wait for confirmation instead of sleeping blind.
# Levels are plain 8-bit greys; display gamma already spreads them across
# a wide luminance range. Tap for fullscreen + screen wake lock; cursor
# hidden; disable the device's auto-brightness.
PANEL_HTML = """<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cam_observe flat panel</title>
<style>html,body{margin:0;height:100%;background:#000;cursor:none}
#p{position:fixed;inset:0;background:#000}
#hint{position:fixed;bottom:10px;width:100%;text-align:center;color:#444;
font:14px sans-serif}</style></head>
<body><div id="p"></div>
<div id="hint">cam_observe flat panel &mdash; tap for fullscreen; disable
this device's auto-brightness</div>
<script>
let lvl=null;
function render(L){
  const h=L.toString(16).padStart(2,'0');
  document.getElementById('p').style.background='#'+h+h+h;
  fetch('/shown?level='+L,{cache:'no-store'}).catch(()=>{});
}
async function loop(){
  try{
    const q=(lvl===null)?'':'&last='+lvl;
    const r=await fetch('/level?wait=25'+q,{cache:'no-store'});
    const j=await r.json();
    if(j.level!==lvl){lvl=j.level;render(lvl);}
  }catch(e){await new Promise(t=>setTimeout(t,1000));}
  loop();
}
loop();
let wl=null;
async function wake(){
  try{
    if(navigator.wakeLock&&!wl){
      wl=await navigator.wakeLock.request('screen');
      wl.addEventListener('release',()=>{wl=null;});
    }
  }catch(e){}
}
document.addEventListener('visibilitychange',
  ()=>{if(!document.hidden)wake();});
document.body.addEventListener('click',()=>{
  wake();
  const e=document.documentElement;
  if(e.requestFullscreen)e.requestFullscreen().catch(()=>{});
  document.getElementById('hint').style.display='none';
});
</script></body></html>"""

# Test image page, served at /test on the same server: basic test patterns
# (line grid, checkerboard, Siemens star) drawn on a full-window canvas.
# Same triggered-refresh method as the flat panel, keyed on a revision
# counter instead of a grey level: long-poll /pattern (held until the rev
# moves), ACK each painted rev via /pattern_shown. Read-only, fullscreen +
# wake lock on tap.
TEST_HTML = """<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cam_observe test image</title>
<style>html,body{margin:0;height:100%;background:#000;cursor:none;
overflow:hidden}
canvas{position:fixed;inset:0}
#hint{position:fixed;bottom:10px;width:100%;text-align:center;color:#444;
font:14px sans-serif}</style></head>
<body><canvas id="c"></canvas>
<div id="hint">cam_observe test image &mdash; tap for fullscreen; disable
this device's auto-brightness</div>
<script>
let st=null;
function draw(p){
  const c=document.getElementById('c');
  c.width=innerWidth;c.height=innerHeight;
  const w=c.width,h=c.height,x=c.getContext('2d');
  const lv=p.level,bg=p.invert?lv:0,fg=p.invert?0:lv;
  x.fillStyle='rgb('+bg+','+bg+','+bg+')';x.fillRect(0,0,w,h);
  const f='rgb('+fg+','+fg+','+fg+')';
  x.fillStyle=f;x.strokeStyle=f;
  const s=p.scale,cx=w/2,cy=h/2;
  if(p.kind==='grid'){
    x.lineWidth=1;x.beginPath();
    for(let gx=cx%s;gx<=w;gx+=s){x.moveTo(gx+.5,0);x.lineTo(gx+.5,h);}
    for(let gy=cy%s;gy<=h;gy+=s){x.moveTo(0,gy+.5);x.lineTo(w,gy+.5);}
    x.stroke();
    x.lineWidth=3;x.beginPath();
    x.moveTo(cx-s,cy);x.lineTo(cx+s,cy);
    x.moveTo(cx,cy-s);x.lineTo(cx,cy+s);x.stroke();
  }else if(p.kind==='checker'){
    for(let gy=0,ry=0;gy<h;gy+=s,ry++)
      for(let gx=(ry%2)*s;gx<w;gx+=2*s)
        x.fillRect(gx,gy,s,s);
  }else if(p.kind==='siemens'){
    const n=36,r=Math.min(w,h)/2-8;
    for(let i=0;i<n;i++){
      const a0=i*2*Math.PI/n;
      x.beginPath();x.moveTo(cx,cy);
      x.arc(cx,cy,r,a0,a0+Math.PI/n);x.closePath();x.fill();
    }
  }
  fetch('/pattern_shown?rev='+p.rev,{cache:'no-store'}).catch(()=>{});
}
async function loop(){
  try{
    const q=(st===null)?'':'&last='+st.rev;
    const r=await fetch('/pattern?wait=25'+q,{cache:'no-store'});
    const j=await r.json();
    if(st===null||j.rev!==st.rev){st=j;draw(st);}
  }catch(e){await new Promise(t=>setTimeout(t,1000));}
  loop();
}
loop();
window.addEventListener('resize',()=>{if(st)draw(st);});
let wl=null;
async function wake(){
  try{
    if(navigator.wakeLock&&!wl){
      wl=await navigator.wakeLock.request('screen');
      wl.addEventListener('release',()=>{wl=null;});
    }
  }catch(e){}
}
document.addEventListener('visibilitychange',
  ()=>{if(!document.hidden)wake();});
document.body.addEventListener('click',()=>{
  wake();
  const e=document.documentElement;
  if(e.requestFullscreen)e.requestFullscreen().catch(()=>{});
  document.getElementById('hint').style.display='none';
});
</script></body></html>"""


def _grey(v):
    """8-bit grey level -> Tk colour string."""
    v = int(v)
    return f"#{v:02x}{v:02x}{v:02x}"


def local_ips():
    """Best-effort list of this machine's LAN IPv4 addresses, for showing
    usable panel URLs."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))     # no packet sent; just picks a route
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ip = info[4][0]
            if "." in ip and not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def _level_eq(a, b):
    """NaN-safe equality for the ACKed-level comparison. last_shown starts as
    NaN and NaN != NaN, so a plain == would silently never match."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def panel_urls(port):
    """Panel URLs to hand the user, one per reachable LAN address."""
    return [f"http://{ip}:{port}/" for ip in local_ips()] \
        or [f"http://<this-host>:{port}/"]


class PanelServer:
    """Serves the software flat panel over HTTP so a browser on ANY device
    -- a big desktop monitor, a tablet clamped in front of the objective --
    can be the illumination source. GET / is the panel page, GET /level the
    current grey level as JSON. GET /test is the test image page, driven by
    /pattern (current pattern params + revision as JSON, long-pollable) and
    acknowledged via /pattern_shown. Read-only by design: nothing on the
    network can change tool state, it can only ask what to display. The
    last-poll timestamp tells the auto-level servo a remote display is
    following, so it allows extra settle time per step.

    get_pattern is optional: a caller that only needs the grey field (the
    transfer-curve sweep) omits it and the test-image endpoints 404.
    """

    def __init__(self, get_level, get_pattern=None, port=8088):
        self.get_level = get_level
        self.get_pattern = get_pattern   # -> dict incl. a "rev" counter
        self.port = port
        self.last_poll = 0.0
        self.last_shown = float("nan")   # last level a browser ACKed
        self.last_pattern_shown = -1     # last pattern rev a browser ACKed
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(u.query)
                if u.path == "/level":
                    outer.last_poll = time.monotonic()
                    # triggered refresh: with wait+last, hold the request
                    # until the level moves (or timeout) -- the browser
                    # sees changes immediately, not on a poll grid
                    try:
                        wait = min(float(qs.get("wait", ["0"])[0]), 30.0)
                        last = float(qs.get("last", ["nan"])[0])
                    except ValueError:
                        wait, last = 0.0, float("nan")
                    if wait > 0 and last == last:      # last is not NaN
                        deadline = time.monotonic() + wait
                        while (float(outer.get_level()) == last
                               and time.monotonic() < deadline):
                            # a held request IS an active client: keep the
                            # timestamp fresh so active_client() stays true
                            # while the browser quietly holds the poll
                            outer.last_poll = time.monotonic()
                            time.sleep(0.05)
                        outer.last_poll = time.monotonic()
                    body = json.dumps(
                        {"level": int(round(float(
                            outer.get_level())))}).encode()
                    ctype = "application/json"
                elif u.path == "/shown":
                    # the browser confirms a level is actually painted;
                    # the auto-level servo waits on this instead of a
                    # blind sleep
                    try:
                        outer.last_shown = float(
                            qs.get("level", ["nan"])[0])
                    except ValueError:
                        pass
                    outer.last_poll = time.monotonic()
                    body = b'{"ok":1}'
                    ctype = "application/json"
                elif u.path == "/pattern":
                    if outer.get_pattern is None:
                        self.send_error(404)
                        return
                    outer.last_poll = time.monotonic()
                    # same triggered refresh as /level, keyed on the
                    # pattern's revision counter
                    try:
                        wait = min(float(qs.get("wait", ["0"])[0]), 30.0)
                        last = int(qs.get("last", ["-1"])[0])
                    except ValueError:
                        wait, last = 0.0, -1
                    if wait > 0 and last >= 0:
                        deadline = time.monotonic() + wait
                        while (int(outer.get_pattern()["rev"]) == last
                               and time.monotonic() < deadline):
                            outer.last_poll = time.monotonic()
                            time.sleep(0.05)
                        outer.last_poll = time.monotonic()
                    body = json.dumps(outer.get_pattern()).encode()
                    ctype = "application/json"
                elif u.path == "/pattern_shown":
                    # the browser confirms a pattern revision is painted
                    try:
                        outer.last_pattern_shown = int(
                            qs.get("rev", ["-1"])[0])
                    except ValueError:
                        pass
                    outer.last_poll = time.monotonic()
                    body = b'{"ok":1}'
                    ctype = "application/json"
                elif u.path == "/test":
                    if outer.get_pattern is None:
                        self.send_error(404)
                        return
                    body = TEST_HTML.encode("utf-8")
                    ctype = "text/html; charset=utf-8"
                elif u.path == "/" or u.path.startswith("/index"):
                    body = PANEL_HTML.encode("utf-8")
                    ctype = "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):   # keep stdout quiet
                pass

        self.httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port),
                                                     Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()

    def active_client(self, within=5.0):
        """True when some browser polled the level recently."""
        return time.monotonic() - self.last_poll < within

    def wait_for_client(self, timeout=120.0, poll=0.25):
        """Block until a browser is following the panel. Returns True if one
        turned up inside the timeout. A sweep that starts before the display
        is live measures the previous level at every early point, so the
        driver waits here rather than trusting the operator's timing."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.active_client():
                return True
            time.sleep(poll)
        return False

    def wait_shown(self, level, timeout=10.0, poll=0.05):
        """Block until a browser ACKs that `level` is painted (or timeout).
        Returns True on a real acknowledgement. This is the difference
        between measuring the level you asked for and measuring whatever
        was still on screen."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _level_eq(self.last_shown, level):
                return True
            time.sleep(poll)
        return False

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
