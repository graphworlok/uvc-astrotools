#!/usr/bin/env python3
"""
match_tle.py -- put names to the visitors in transients.jsonl.

OFFLINE analysis tool (same contract as make_catalog.py: the observing
tool never touches the network; this runs afterwards, wherever there is
internet). Every record cam_observe.py classified as a transient carries
UTC, topocentric sky coordinates and a motion vector; this tool
propagates a satellite catalogue to each record's instant and reports
which satellite -- if any -- was at that place, moving that way. What
survives unmatched is the residue actually worth a second look:
UFO -> IFO, with named suspects and error bars.

Method: SGP4 propagation (python-sgp4 -- the one extra dependency, for
this tool only: `pip install sgp4`) gives TEME positions; the observer's
WGS84 location is rotated into TEME via GMST (IAU 1982/Meeus) and the
look vector's RA/Dec compared against the record. Coordinate honesty:
the record's WCS coordinates are J2000 while the look vector is
equator-of-date (TEME); in 2026 that difference is ~0.4 deg, well inside
the match tolerance, whose default (1.5 deg) also absorbs TLE age and
the record's own centroid/timing error. Position angles of motion are
compared as a secondary check, not a gate -- a match at 0.3 deg with the
wrong direction is reported as suspicious rather than silently kept.

Usage:
  python3 match_tle.py transients.jsonl --lat -34.93 --lon 138.60 \\
      [--alt-m 50] [--group active|visual|starlink|stations]
      [--tle-file cached.tle] [--tol-deg 1.5] [--out matched.jsonl]

TLEs are fetched from CelesTrak and cached beside the input as
tle_<group>_<date>.txt; --tle-file skips the network entirely.
"""

import argparse
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone

CELESTRAK = ("https://celestrak.org/NORAD/elements/gp.php"
             "?GROUP={group}&FORMAT=tle")

WGS84_A = 6378.137            # km
WGS84_F = 1.0 / 298.257223563


def gmst_deg(jd_ut):
    """Greenwich mean sidereal time, degrees (Meeus 12.4 / IAU 1982).
    UT1 ~= UTC is assumed: the <1 s difference is ~0.004 deg of Earth
    rotation, three orders under the match tolerance."""
    t = (jd_ut - 2451545.0) / 36525.0
    g = (280.46061837 + 360.98564736629 * (jd_ut - 2451545.0)
         + 0.000387933 * t * t - t * t * t / 38710000.0)
    return g % 360.0


def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    x = (n + alt_km) * math.cos(lat) * math.cos(lon)
    y = (n + alt_km) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - e2) + alt_km) * math.sin(lat)
    return x, y, z


def ecef_to_teme(xyz, gmst):
    """ROT3(-gmst): PEF/ECEF -> TEME (polar motion ignored: ~10 m)."""
    g = math.radians(gmst)
    x, y, z = xyz
    return (x * math.cos(g) - y * math.sin(g),
            x * math.sin(g) + y * math.cos(g), z)


def topo_radec(sat, jd, fr, obs_ecef):
    """Topocentric TEME (RA, Dec, range_km, elevation_deg) of sat at
    jd+fr, or None on propagation error (decayed / bad elements).
    Elevation is against the local geocentric zenith (spherical-Earth
    approximation, good to ~0.2 deg): a satellite below the horizon
    cannot be the explanation for an observed transient, however well
    its RA/Dec happens to line up on the far side of the planet."""
    e, r, _ = sat.sgp4(jd, fr)
    if e != 0:
        return None
    ox, oy, oz = ecef_to_teme(obs_ecef, gmst_deg(jd + fr))
    dx, dy, dz = r[0] - ox, r[1] - oy, r[2] - oz
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    om = math.sqrt(ox * ox + oy * oy + oz * oz) or 1.0
    el = math.degrees(math.asin(
        (dx * ox + dy * oy + dz * oz) / (rng * om)))
    return (math.degrees(math.atan2(dy, dx)) % 360.0,
            math.degrees(math.asin(dz / rng)), rng, el)


def ang_sep(ra1, de1, ra2, de2):
    """Angular separation, degrees (haversine -- stable at small seps)."""
    p1, p2 = math.radians(de1), math.radians(de2)
    dra = math.radians(abs((ra1 - ra2 + 180.0) % 360.0 - 180.0))
    dde = p2 - p1
    a = (math.sin(dde / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dra / 2.0) ** 2)
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(a))))


def motion_pa(sat, jd, fr, obs_ecef):
    """Position angle of the satellite's apparent motion (deg E of N),
    from a 1 s baseline -- compared against the record's pa_deg as a
    plausibility note."""
    a = topo_radec(sat, jd, fr, obs_ecef)
    b = topo_radec(sat, jd, fr + 1.0 / 86400.0, obs_ecef)
    if a is None or b is None:
        return None
    dra = ((b[0] - a[0] + 180.0) % 360.0 - 180.0) \
        * math.cos(math.radians(a[1]))
    dde = b[1] - a[1]
    if dra == 0.0 and dde == 0.0:
        return None
    return math.degrees(math.atan2(dra, dde)) % 360.0


def load_tles(path):
    sats = []
    try:
        from sgp4.api import Satrec
    except ImportError:
        sys.exit("python-sgp4 is required for this tool: pip install sgp4")
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.rstrip() for ln in fh if ln.strip()]
    i = 0
    while i + 1 < len(lines):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name = lines[i - 1].strip() if i else "?"
            try:
                sats.append((name, Satrec.twoline2rv(lines[i],
                                                     lines[i + 1])))
            except Exception:
                pass
            i += 2
        else:
            i += 1
    return sats


def fetch_tles(group, cache_dir):
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(cache_dir, f"tle_{group}_{day}.txt")
    if os.path.exists(path):
        print(f"using cached {path}")
        return path
    url = CELESTRAK.format(group=group)
    print(f"fetching {url} ...")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if len(data) < 1000:
        sys.exit("suspiciously small TLE download -- refusing to cache")
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"cached {path} ({len(data) // 1024} KB)")
    return path


def jd_from_iso(iso):
    from sgp4.api import jday
    t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    t = t.astimezone(timezone.utc)
    return jday(t.year, t.month, t.day, t.hour, t.minute,
                t.second + t.microsecond / 1e6)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="match transients.jsonl records against satellite "
                    "TLEs")
    ap.add_argument("transients", help="transients.jsonl from cam_observe")
    ap.add_argument("--lat", type=float, required=True,
                    help="observer latitude, degrees (+N)")
    ap.add_argument("--lon", type=float, required=True,
                    help="observer longitude, degrees (+E)")
    ap.add_argument("--alt-m", type=float, default=0.0)
    ap.add_argument("--group", default="active",
                    help="CelesTrak group (default %(default)s; 'visual' "
                         "is much smaller/faster)")
    ap.add_argument("--tle-file", help="use this TLE file, no network")
    ap.add_argument("--tol-deg", type=float, default=1.5)
    ap.add_argument("--all-classes", action="store_true",
                    help="also try to match 'sidereal' and 'static' "
                         "records (sanity check: they should NOT match)")
    ap.add_argument("--out", help="write annotated records here "
                    "(default <input>_matched.jsonl)")
    a = ap.parse_args(argv)

    tle_path = a.tle_file or fetch_tles(
        a.group, os.path.dirname(os.path.abspath(a.transients)) or ".")
    sats = load_tles(tle_path)
    print(f"{len(sats)} satellites loaded")
    obs_ecef = geodetic_to_ecef(a.lat, a.lon, a.alt_m / 1000.0)

    recs = []
    with open(a.transients, encoding="utf-8") as fh:
        for ln in fh:
            try:
                recs.append(json.loads(ln))
            except ValueError:
                pass
    todo = [r for r in recs if "ra" in r
            and (a.all_classes or r.get("class") == "transient")]
    print(f"{len(todo)} record(s) to match "
          f"(of {len(recs)} in file)")

    out_path = a.out or (os.path.splitext(a.transients)[0]
                         + "_matched.jsonl")
    n_matched = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in recs:
            if rec in todo:
                jd, fr = jd_from_iso(rec["t"])
                best = None            # (sep, name, range_km, pa)
                for name, sat in sats:
                    tr = topo_radec(sat, jd, fr, obs_ecef)
                    if tr is None or tr[3] < -1.0:
                        continue          # below the observer's horizon
                    sep = ang_sep(rec["ra"], rec["dec"], tr[0], tr[1])
                    if sep <= a.tol_deg and (best is None
                                             or sep < best[0]):
                        best = (sep, name, tr[2], sat)
                if best is not None:
                    sep, name, rng, sat = best
                    pa = motion_pa(sat, jd, fr, obs_ecef)
                    dpa = None
                    if pa is not None and "pa_deg" in rec:
                        dpa = abs((pa - rec["pa_deg"] + 180.0)
                                  % 360.0 - 180.0)
                    rec["tle_match"] = {
                        "name": name, "sep_deg": round(sep, 3),
                        "range_km": round(rng, 1),
                        "pa_deg": (round(pa, 1) if pa is not None
                                   else None),
                        "dpa_deg": (round(dpa, 1) if dpa is not None
                                    else None),
                        "suspicious": bool(dpa is not None
                                           and dpa > 30.0),
                        "tle_file": os.path.basename(tle_path)}
                    n_matched += 1
                    note = (f"  (MOTION MISMATCH dpa={dpa:.0f} deg)"
                            if dpa is not None and dpa > 30.0 else "")
                    print(f"{rec['t']}  {rec['class']:9s} -> {name} "
                          f"sep {sep:.2f} deg, range {rng:.0f} km{note}")
                else:
                    rec["tle_match"] = None
                    print(f"{rec['t']}  {rec['class']:9s} -> "
                          f"UNMATCHED (the interesting bin)")
            out.write(json.dumps(rec) + "\n")
    print(f"\n{n_matched}/{len(todo)} matched; annotated records "
          f"written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
