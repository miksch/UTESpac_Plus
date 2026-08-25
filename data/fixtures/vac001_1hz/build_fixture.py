"""Build and pin the VAC001_1Hz regression fixture.

The fixture is a one-day mini site carved from ``data/VAC001`` (2023-07-08):
the IRGASON/fine-wire table decimated from 20 Hz to 1 Hz, real samples from
08:00 to 16:00 and NaN rows elsewhere (a complete day, as
``raw_processing`` writes them), gzipped; the 30-min T/RH table for the
same day; ``siteInfo.toml``; ``PFinfo.json`` for the global-planar-fit run.
It is small enough to commit, so every migration step is tested on any
machine, without the 48-h inputs or MATLAB references.

``expected/<config>.npz`` + ``.json`` pin the averaged output (every 2-D
field with its header) and a strided subset of the raw output for each
configuration in :data:`CONFIGS`. ``tests/test_pinned_fixture.py`` runs
the pipeline on the fixture and compares.

Usage (repo root, UTESpac_Plus env)::

    python data/fixtures/vac001_1hz/build_fixture.py build   # needs data/VAC001 (this machine)
    python data/fixtures/vac001_1hz/build_fixture.py pin     # re-pin after an accepted change

Re-pinning is a deliberate act: do it only when the ledger
(tests/KNOWN_DIVERGENCES.md) has a row for the change that moved the values.
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SITE = "VAC001_1Hz"
FAST = f"{SITE}"                 # fast table name = <prefix>_<table> with prefix VAC001, table 1Hz
SLOW = "VAC001_30min"
DAY = (2023, 189)                # 2023-07-08
HOURS = (8, 16)                  # real data [08:00, 16:00), NaN rows elsewhere
RAW_STRIDE = 50                  # raw output rows pinned every 50th sample

# (config name, RunConfig overrides)
CONFIGS = {
    "LPF_LinDet": dict(pf={"globalCalculation": "local", "recalculateGlobalCoefficients": False},
                       flux={"detrendingFormat": "linear"}),
    "GPF_ConstDet": dict(pf={"globalCalculation": "global", "recalculateGlobalCoefficients": False},
                         flux={"detrendingFormat": "constant"}),
}

SITE_INFO_TOML = """\
# VAC001_1Hz -- regression fixture carved from data/VAC001 (2023-07-08):
# the 20 Hz IRGASON + fine-wire table decimated to 1 Hz, real samples
# 08:00-16:00, NaN rows for the rest of the day. Site facts as in
# data/VAC001/siteInfo.toml. Built by data/fixtures/vac001_1hz/build_fixture.py.
tower = 180
siteElevation = 18.2
latitude = 38.300056
longitude = -121.9105
angle = 0.0

tableNames = ["VAC001_1Hz", "VAC001_30min"]
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

def _grid(day):
    """(doy, HM, sec) of the 1 Hz day grid: 00:00:01 ... next-day 00:00:00."""
    yr, doy = day
    for s in range(1, 86400):
        hh, rem = divmod(s, 3600)
        mm, ss = divmod(rem, 60)
        yield (doy, hh * 100 + mm, ss)
    yield (doy + 1, 0, 0)


def build(source_site=os.path.join(ROOT, "data", "VAC001"), out=HERE):
    yr, doy = DAY
    src_fast = glob.glob(os.path.join(source_site, "utespac",
                                      f"VAC001_20Hz_{yr}{_mmdd(yr, doy)}000000_*.txt"))
    src_slow = glob.glob(os.path.join(source_site, "utespac",
                                      f"{SLOW}_{yr}{_mmdd(yr, doy)}000000_*.txt"))
    if not src_fast or not src_slow:
        raise FileNotFoundError("48-h VAC001 inputs for 2023-07-08 not found under data/VAC001/utespac")
    os.makedirs(os.path.join(out, "utespac"), exist_ok=True)

    # fast table: whole-second samples of the day, real inside HOURS
    rows = {}
    with open(src_fast[0], "r") as fh:
        for line in fh:
            y, d, hm, sec, rest = line.split(",", 4)
            d, hm = int(d), int(hm)
            if d not in (doy, doy + 1):
                continue
            cs = int(round(float(sec) * 100))
            if cs % 100:
                continue
            rows[(d, hm, cs // 100)] = line.rstrip("\n")
    n_cols = 16
    blank = "," * (n_cols - 4)
    n_real = 0
    gz_path = os.path.join(out, "utespac", f"{FAST}_{yr}{_mmdd(yr, doy)}000000_"
                           f"{yr}{_mmdd(yr, doy + 1)}000000.txt.gz")
    with gzip.open(gz_path, "wt", newline="\n", compresslevel=9) as gz:
        for d, hm, ss in _grid(DAY):
            key = (d, hm, ss)
            real = d == doy and HOURS[0] * 100 <= hm < HOURS[1] * 100
            if real and key in rows:
                gz.write(rows[key] + "\n")
                n_real += 1
            else:
                gz.write(f"{yr},{d},{hm},{ss:.2f}{blank}\n")
    print(f"fast: {n_real} real rows -> {gz_path} ({os.path.getsize(gz_path) / 1e6:.2f} MB)")

    # slow table: the day's 48 rows (00:30 ... next-day 00:00)
    slow_path = os.path.join(out, "utespac", f"{SLOW}_{yr}{_mmdd(yr, doy)}000000_"
                             f"{yr}{_mmdd(yr, doy + 1)}000000.txt")
    n_slow = 0
    with open(src_slow[0]) as fh, open(slow_path, "w", newline="\n") as oh:
        for line in fh:
            y, d, hm, sec, rest = line.split(",", 4)
            d, hm = int(d), int(hm)
            if d == doy or (d == doy + 1 and hm == 0):
                oh.write(line if line.endswith("\n") else line + "\n")
                n_slow += 1
    print(f"slow: {n_slow} rows -> {slow_path}")

    # headers, site facts, planar-fit coefficients
    shutil.copy(os.path.join(source_site, "utespac", "VAC001_20Hz_header.dat"),
                os.path.join(out, "utespac", f"{FAST}_header.dat"))
    shutil.copy(os.path.join(source_site, "utespac", f"{SLOW}_header.dat"),
                os.path.join(out, "utespac", f"{SLOW}_header.dat"))
    with open(os.path.join(out, "siteInfo.toml"), "w", newline="\n") as fh:
        fh.write(SITE_INFO_TOML)
    shutil.copy(os.path.join(source_site, "PFinfo.json"), os.path.join(out, "PFinfo.json"))


def _mmdd(yr, doy):
    from datetime import date, timedelta
    d = date(yr, 1, 1) + timedelta(days=doy - 1)
    return f"{d.month:02d}{d.day:02d}"


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
    date = result.dates[0]
    return read_run_legacy(date.paths["nc"]), raw_to_legacy(date.run.raw)


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
    args = ap.parse_args()
    if args.action == "build":
        build()
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
