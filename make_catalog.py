#!/usr/bin/env python3
"""
make_catalog.py -- build the embedded polar-cap star catalogue.

OFFLINE, RUN-ONCE tool: downloads the Tycho-2 polar cap from VizieR and
writes catalog_scp.npy / catalog_ncp.npy -- float32 rows of
(ra_deg, dec_deg, vmag) -- plus a provenance sidecar. The observing tool
only ever reads the .npy; nothing at observation time touches the
network (a telescope in a paddock has no internet, and a sky survey is
not a runtime dependency).

Vmag is approximated from Tycho BT/VT with the standard transformation
V = VT - 0.090*(BT - VT); where BT is missing, VT is used as-is (the
error is < ~0.1 mag for the field, far inside the response fit's
scatter). Proper motion is ignored: Tycho-2 epoch positions are wrong by
at most a few arcsec today, well under one pixel at the ~4-8"/px scales
this project runs.

Dec <= -78 (or >= +78) at VT <= 11.5 is ~450 sq deg and a few tens of
thousands of stars: a few hundred KB on disk, milliseconds to project.

Usage:
  python3 make_catalog.py [--ncp] [--dec-limit 78] [--mag-limit 11.5]
                          [--out DIR]

Attribution: Tycho-2 (Hog et al. 2000, A&A 355, L27), via VizieR
catalogue I/259, CDS, Strasbourg.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np

MIRRORS = [
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "https://vizier.u-strasbg.fr/viz-bin/asu-tsv",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
]


def fetch_tycho2(dec_limit, mag_limit, south, timeout=300):
    """One VizieR TSV request for the whole cap (no paging: the cap is
    small). Tries mirrors in order; raises if all fail."""
    constraint = f"<=-{dec_limit}" if south else f">=+{dec_limit}"
    params = {
        "-source": "I/259/tyc2",
        "-out": "RAmdeg DEmdeg BTmag VTmag",
        "DEmdeg": constraint,
        "VTmag": f"<={mag_limit}",
        "-out.max": "unlimited",
    }
    q = urllib.parse.urlencode(params)
    last = None
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(f"{base}?{q}",
                                        timeout=timeout) as r:
                return r.read().decode("utf-8", "replace"), base
        except Exception as e:            # noqa: BLE001 -- try next mirror
            print(f"  {base}: {e}", file=sys.stderr)
            last = e
    raise RuntimeError(f"all VizieR mirrors failed; last error: {last}")


def parse_tsv(text):
    """VizieR TSV: '#'-prefixed metadata, a header line, a dashes line,
    then data rows. Missing values are empty fields."""
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split("\t")
        if len(parts) < 4 or not parts[0].strip()[:1].isdigit():
            continue
        try:
            ra = float(parts[0])
            dec = float(parts[1])
        except ValueError:
            continue
        try:
            bt = float(parts[2])
        except ValueError:
            bt = None
        try:
            vt = float(parts[3])
        except ValueError:
            continue                      # no VT, no magnitude: skip
        v = vt - 0.090 * (bt - vt) if bt is not None else vt
        rows.append((ra, dec, v))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ncp", action="store_true",
                    help="build the north cap instead of the south")
    ap.add_argument("--dec-limit", type=float, default=78.0)
    ap.add_argument("--mag-limit", type=float, default=11.5,
                    help="VT cut (default %(default)s)")
    ap.add_argument("--out", default=".", help="output directory")
    a = ap.parse_args(argv)

    south = not a.ncp
    name = "catalog_scp" if south else "catalog_ncp"
    print(f"fetching Tycho-2 {'south' if south else 'north'} cap "
          f"(|dec| >= {a.dec_limit}, VT <= {a.mag_limit}) ...")
    text, mirror = fetch_tycho2(a.dec_limit, a.mag_limit, south)
    rows = parse_tsv(text)
    if len(rows) < 100:
        print(f"only {len(rows)} stars parsed -- refusing to write a "
              "catalogue that thin (mirror format change?)",
              file=sys.stderr)
        return 1
    cat = np.array(rows, dtype=np.float32)
    cat = cat[np.argsort(cat[:, 2])]      # brightest first on disk
    npy = f"{a.out.rstrip('/')}/{name}.npy"
    np.save(npy, cat)
    meta = {
        "source": "Tycho-2 (Hog et al. 2000), VizieR I/259/tyc2",
        "mirror": mirror,
        "dec_limit": (-a.dec_limit if south else a.dec_limit),
        "vt_mag_limit": a.mag_limit,
        "vmag_transform": "V = VT - 0.090*(BT-VT); VT where BT missing",
        "n_stars": int(len(cat)),
        "columns": ["ra_deg", "dec_deg", "vmag"],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(f"{a.out.rstrip('/')}/{name}.json", "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {npy}: {len(cat)} stars, "
          f"mag {cat[:, 2].min():.2f}..{cat[:, 2].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
