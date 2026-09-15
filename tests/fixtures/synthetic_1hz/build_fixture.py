"""Build and pin the synthetic 1 Hz regression fixture.

The fixture is a one-day mini site carved from a real site's raw data: an
IRGASON/fine-wire table decimated from 20 Hz to 1 Hz, real samples for a
window of the day and NaN rows elsewhere (a complete day, as
``raw_processing`` writes them), gzipped; the matching 30-min T/RH table;
``siteInfo.toml`` with placeholder coordinates; ``PFinfo.json`` for the
global-planar-fit run. It is small enough to commit, so every migration
step is tested on any machine, without the 48-h inputs or a real site's
identity.

The real site, date, and file prefix used to carve it are supplied at
build time (below) and never recorded here or in the fixture itself:
timestamps and the site code are rewritten to the fixed placeholders in
:data:`OUT_DATE` and :data:`SITE` regardless of the source.

``expected/<config>.npz`` + ``.json`` pin the averaged output (every 2-D
field with its header) and a strided subset of the raw output for each
configuration in :data:`CONFIGS`. ``tests/test_pinned_fixture.py`` runs
the pipeline on the fixture and compares.

Usage (repo root, UTESpac_Plus env)::

    python tests/fixtures/synthetic_1hz/build_fixture.py build \\
        --source /path/to/a/real/site/dir --source-prefix SITECODE \\
        --source-date YYYY-MM-DD --source-hours 8 16
    python tests/fixtures/synthetic_1hz/build_fixture.py pin

Re-pinning is a deliberate act: do it only when the ledger
(tests/KNOWN_DIVERGENCES.md) has a row for the change that moved the values.
"""

import argparse
import glob
import gzip
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import date, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SITE = "TESTSITE_1Hz"             # fixture's own site/fast-table name (generic, committed)
SLOW = "TESTSITE_30min"           # fixture's own slow-table name (generic, committed)
OUT_DATE = date(2024, 6, 1)       # placeholder calendar date every fixture timestamp uses
RAW_STRIDE = 50                   # raw output rows pinned every 50th sample

# (config name, RunConfig overrides)
CONFIGS = {
    "LPF_LinDet": dict(pf={"globalCalculation": "local", "recalculateGlobalCoefficients": False},
                       flux={"detrendingFormat": "linear"}),
    "GPF_ConstDet": dict(pf={"globalCalculation": "global", "recalculateGlobalCoefficients": False},
                         flux={"detrendingFormat": "constant"}),
}

SITE_INFO_TOML = """\
# TESTSITE_1Hz -- synthetic regression fixture: one day of a real IRGASON +
# fine-wire table, decimated to 1 Hz and relabeled onto a placeholder date
# and placeholder coordinates. Built by tests/fixtures/synthetic_1hz/build_fixture.py.
tower = 180
siteElevation = 18.2
latitude = 41.15
longitude = -98.92
angle = 0.0

tableNames = ["TESTSITE_1Hz", "TESTSITE_30min"]
tableScanFrequency = [1, 0.000555555555556]
tableNumberOfColumns = [16, 6]

useTrefHMP = true
avgSlowFreq = 30
shiftzRef = false
zRefLowestSon = 10.85
ascending = false

SSITC_subAvgMin = 5
displacementHeight = 0
canopyHeight = 6.0
useCanopyITC = true

[[sonics]]
height = 10.85
orientation = 215
manufacturer = 1
hmp_height = 10.72
"""


# ── build ────────────────────────────────────────────────────────────────────

def _mmdd(d):
    return f"{d.month:02d}{d.day:02d}"


def _grid():
    """(day_offset, hm, sec) for one day: 00:00:01 (offset 0) ... 00:00:00 the
    next day (offset 1), independent of which calendar date is in play."""
    for s in range(1, 86400):
        hh, rem = divmod(s, 3600)
        mm, ss = divmod(rem, 60)
        yield (0, hh * 100 + mm, ss)
    yield (1, 0, 0)


def build(source, source_prefix, source_date, source_hours=(8, 16), out=HERE):
    """Carve the fixture from a real site's raw data.

    ``source`` is a real site directory (containing ``utespac/``),
    ``source_prefix`` its table-file prefix, ``source_date`` the real
    calendar date (``datetime.date``) to carve, ``source_hours`` the
    (start, end) hour window with real samples. None of these are
    recorded in the fixture: output timestamps use :data:`OUT_DATE` and
    the site name is :data:`SITE`/:data:`SLOW`.
    """
    src_year, src_doy = source_date.year, source_date.timetuple().tm_yday
    src_fast = glob.glob(os.path.join(source, "utespac",
                                      f"{source_prefix}_20Hz_{src_year}{_mmdd(source_date)}000000_*.txt"))
    src_slow = glob.glob(os.path.join(source, "utespac",
                                      f"{source_prefix}_30min_{src_year}{_mmdd(source_date)}000000_*.txt"))
    if not src_fast or not src_slow:
        raise FileNotFoundError(f"48-h fast/slow inputs for {source_date} not found under {source}/utespac")
    os.makedirs(os.path.join(out, "utespac"), exist_ok=True)

    # fast table: whole-second samples of the day, real inside source_hours
    rows = {}
    with open(src_fast[0], "r") as fh:
        for line in fh:
            y, d, hm, sec, rest = line.split(",", 4)
            d, hm = int(d), int(hm)
            if d not in (src_doy, src_doy + 1):
                continue
            cs = int(round(float(sec) * 100))
            if cs % 100:
                continue
            rows[(d - src_doy, hm, cs // 100)] = line.rstrip("\n")
    n_cols = 16
    blank = "," * (n_cols - 4)
    n_real = 0
    out_year, out_doy = OUT_DATE.year, OUT_DATE.timetuple().tm_yday
    next_date = OUT_DATE + timedelta(days=1)
    gz_path = os.path.join(out, "utespac", f"{SITE}_{out_year}{_mmdd(OUT_DATE)}000000_"
                           f"{next_date.year}{_mmdd(next_date)}000000.txt.gz")
    with gzip.open(gz_path, "wt", newline="\n", compresslevel=9) as gz:
        for offset, hm, ss in _grid():
            key = (offset, hm, ss)
            real = offset == 0 and source_hours[0] * 100 <= hm < source_hours[1] * 100
            doy_out = out_doy + offset
            if real and key in rows:
                rest = rows[key].split(",", 4)[4]
                gz.write(f"{out_year},{doy_out},{hm},{ss:.2f},{rest}\n")
                n_real += 1
            else:
                gz.write(f"{out_year},{doy_out},{hm},{ss:.2f}{blank}\n")
    print(f"fast: {n_real} real rows -> {gz_path} ({os.path.getsize(gz_path) / 1e6:.2f} MB)")

    # slow table: the day's 48 rows (00:30 ... next-day 00:00), relabeled onto OUT_DATE
    slow_path = os.path.join(out, "utespac", f"{SLOW}_{out_year}{_mmdd(OUT_DATE)}000000_"
                             f"{next_date.year}{_mmdd(next_date)}000000.txt")
    n_slow = 0
    with open(src_slow[0]) as fh, open(slow_path, "w", newline="\n") as oh:
        for line in fh:
            y, d, hm, sec, rest = line.split(",", 4)
            d, hm = int(d), int(hm)
            if d == src_doy or (d == src_doy + 1 and hm == 0):
                offset = 1 if (d == src_doy + 1 and hm == 0) else 0
                rest_line = rest.rstrip("\n")
                oh.write(f"{out_year},{out_doy + offset},{hm},{sec},{rest_line}\n")
                n_slow += 1
    print(f"slow: {n_slow} rows -> {slow_path}")

    # headers, site facts, planar-fit coefficients
    shutil.copy(os.path.join(source, "utespac", f"{source_prefix}_20Hz_header.dat"),
                os.path.join(out, "utespac", f"{SITE}_header.dat"))
    shutil.copy(os.path.join(source, "utespac", f"{source_prefix}_30min_header.dat"),
                os.path.join(out, "utespac", f"{SLOW}_header.dat"))
    with open(os.path.join(out, "siteInfo.toml"), "w", newline="\n") as fh:
        fh.write(SITE_INFO_TOML)
    _write_generic_pfinfo(os.path.join(source, "PFinfo.json"), os.path.join(out, "PFinfo.json"))


def _write_generic_pfinfo(src_path, dst_path):
    """Copy PFinfo.json with its site name and date range replaced: only the
    b0/b1/b2 tilt coefficients and heights are real-site-derived, and neither
    identifies the source. ``coefficients()`` is unused by the pipeline (it
    matches by height alone via ``to_legacy()``), so the placeholder range
    need not cover OUT_DATE -- it only has to be internally consistent."""
    with open(src_path) as fh:
        d = json.load(fh)
    d["site"] = SITE
    start, end = OUT_DATE - timedelta(days=2), OUT_DATE + timedelta(days=2)
    date_range = re.compile(r"\d{2}-[A-Za-z]{3}-\d{4}\s+to\s+\d{2}-[A-Za-z]{3}-\d{4}")
    placeholder_range = f"{start:%d-%b-%Y} to {end:%d-%b-%Y}"
    for r in d.get("records", []):
        r["date_start"], r["date_end"] = start.isoformat(), end.isoformat()
    for block in d.get("info_string", []):
        for i, line in enumerate(block):
            if date_range.fullmatch(line):
                block[i] = placeholder_range
    with open(dst_path, "w", newline="\n") as fh:
        json.dump(d, fh, indent=1)


# ── run ──────────────────────────────────────────────────────────────────────

def stage_site(fixture_dir, root):
    """Copy the fixture into ``<root>/<SITE>`` (gz inflated); returns the site path."""
    site = os.path.join(root, SITE)
    os.makedirs(os.path.join(site, "utespac"), exist_ok=True)
    for name in ("siteInfo.toml", "PFinfo.json"):
        shutil.copy(os.path.join(fixture_dir, name), os.path.join(site, name))
    for path in glob.glob(os.path.join(fixture_dir, "utespac", "*")):
        dst = os.path.join(site, "utespac", os.path.basename(path))
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as src, open(dst[:-3], "wb") as out:
                shutil.copyfileobj(src, out)
        else:
            shutil.copy(path, dst)
    return site


def run_fixture(config_name, root):
    """Run the pipeline on the staged site for one config; returns (avg dict, raw dict).

    ``avg`` is the run netCDF read back as the legacy dict (what the pickle
    used to hold), ``raw`` the raw products of the in-memory Run (the HF
    netCDF stores them as float32, so the pin takes them before the file)."""
    from utespac import RunConfig, run_utespac
    from utespac.run_io import read_run_legacy
    from utespac.model import raw_to_legacy
    cfg = RunConfig.from_config(rootFolder=root, saveCSV=False,
                                saveRawConditionedData=True, **CONFIGS[config_name])
    result = run_utespac(cfg, site=SITE, dates="all", keep_runs=True)
    if not result.ok:
        raise RuntimeError(f"{config_name}: {[d.error for d in result.dates if d.error]}")
    date_ = result.dates[0]
    return read_run_legacy(date_.paths["nc"]), raw_to_legacy(date_.run.raw)


# ── pin / compare ────────────────────────────────────────────────────────────

def _numeric_fields(d):
    return {k: v for k, v in d.items()
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.dtype.kind in "fb"}


def _header_of(d, field):
    for k in (f"{field}Header", f"{field}header"):
        if k in d:
            return d[k]
    return None


def snapshot(avg, raw):
    """Arrays and headers to pin: every 2-D field of the averaged output, the raw
    output strided by :data:`RAW_STRIDE` (1-D arrays whole)."""
    arrays, meta = {}, {"avg_fields": [], "raw_fields": [], "headers": {}}
    for k, v in sorted(_numeric_fields(avg).items()):
        arrays[f"avg/{k}"] = v
        meta["avg_fields"].append(k)
        hdr = _header_of(avg, k)
        if hdr is not None:
            meta["headers"][k] = hdr
    for k, v in sorted(raw.items()):
        if not isinstance(v, np.ndarray):
            continue
        arrays[f"raw/{k}"] = v[::RAW_STRIDE] if v.ndim == 2 else v[::RAW_STRIDE] if v.size > 1000 else v
        meta["raw_fields"].append(k)
    return arrays, meta


def pin(fixture_dir=HERE):
    exp = os.path.join(fixture_dir, "expected")
    os.makedirs(exp, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage_site(fixture_dir, tmp)
        for name in CONFIGS:
            avg, raw = run_fixture(name, tmp)
            arrays, meta = snapshot(avg, raw)
            np.savez_compressed(os.path.join(exp, f"{name}.npz"), **arrays)
            with open(os.path.join(exp, f"{name}.json"), "w", newline="\n") as fh:
                json.dump(meta, fh, indent=1, default=str)
            print(f"{name}: pinned {len(meta['avg_fields'])} averaged fields, "
                  f"{len(meta['raw_fields'])} raw fields")


def load_expected(config_name, fixture_dir=HERE):
    exp = os.path.join(fixture_dir, "expected")
    with np.load(os.path.join(exp, f"{config_name}.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    with open(os.path.join(exp, f"{config_name}.json")) as fh:
        meta = json.load(fh)
    return arrays, meta


def compare(avg, raw, expected, meta, rtol=1e-9, atol=1e-12):
    """Differences between a run and the pinned snapshot; empty list when none.

    The tolerance sits above floating-point summation-order noise (the
    flux stage reduces single columns, ~1e-13 absolute against the
    pre-split stage) and far below any physical change."""
    problems = []
    got, got_meta = snapshot(avg, raw)
    for kind in ("avg_fields", "raw_fields"):
        if sorted(got_meta[kind]) != sorted(meta[kind]):
            problems.append(f"{kind}: got {sorted(got_meta[kind])}, pinned {sorted(meta[kind])}")
    for field, hdr in meta["headers"].items():
        if got_meta["headers"].get(field) != hdr:
            problems.append(f"header of {field} differs")
    for key, ref in expected.items():
        if key not in got:
            continue
        val = got[key]
        if val.shape != ref.shape:
            problems.append(f"{key}: shape {val.shape} vs pinned {ref.shape}")
            continue
        if val.dtype == bool or ref.dtype == bool:
            if not np.array_equal(val.astype(bool), ref.astype(bool)):
                problems.append(f"{key}: {int((val.astype(bool) != ref.astype(bool)).sum())} flag cells differ")
            continue
        v, r = val.astype(float), ref.astype(float)
        nan_mismatch = np.isnan(v) != np.isnan(r)
        if nan_mismatch.any():
            problems.append(f"{key}: {int(nan_mismatch.sum())} NaN-pattern cells differ")
            continue
        m = ~np.isnan(v)
        if not np.allclose(v[m], r[m], rtol=rtol, atol=atol):
            diff = np.abs(v[m] - r[m])
            rel = diff / np.maximum(np.abs(r[m]), atol)
            problems.append(f"{key}: max abs {diff.max():.3g}, max rel {rel.max():.3g}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["build", "pin", "check"])
    ap.add_argument("--source", help="real site directory to carve from (build only)")
    ap.add_argument("--source-prefix", help="real site's table-file prefix, e.g. SITECODE (build only)")
    ap.add_argument("--source-date", help="real calendar date to carve, YYYY-MM-DD (build only)")
    ap.add_argument("--source-hours", nargs=2, type=int, default=[8, 16], metavar=("START", "END"),
                    help="hour window with real samples in the source day (build only, default 8 16)")
    args = ap.parse_args()
    if args.action == "build":
        if not (args.source and args.source_prefix and args.source_date):
            ap.error("build requires --source, --source-prefix and --source-date")
        y, m, d = (int(p) for p in args.source_date.split("-"))
        build(args.source, args.source_prefix, date(y, m, d), tuple(args.source_hours))
    elif args.action == "pin":
        pin()
    else:
        with tempfile.TemporaryDirectory() as tmp:
            stage_site(HERE, tmp)
            bad = 0
            for name in CONFIGS:
                avg, raw = run_fixture(name, tmp)
                probs = compare(avg, raw, *load_expected(name))
                print(f"{name}: {'OK' if not probs else chr(10).join(probs)}")
                bad += bool(probs)
            raise SystemExit(bad)
