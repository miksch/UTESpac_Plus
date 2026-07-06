"""compare_outputs.py — Comprehensive statistical comparison of UTESpac Python vs MATLAB outputs.

After running the MATLAB pipeline (produces .mat files) and the Python pipeline (produces .pkl
files) for any site under UTESpac_MATLAB/, run this script to get an expansive field-by-field,
column-by-column statistical report.

The comparison engine lives in utespac.testkit (shared with the pytest suite);
this script is the report-generating CLI.

Usage
-----
    python3 compare_outputs.py                         # all sites, GPF only (default)
    python3 compare_outputs.py --site siteFire1        # one site, GPF only
    python3 compare_outputs.py --site siteFire1 --lpf  # LPF only
    python3 compare_outputs.py --avg                   # 30-min averaged only, GPF
    python3 compare_outputs.py --raw                   # 20 Hz raw only
    python3 compare_outputs.py --csv results.csv       # save stats table to CSV
    python3 compare_outputs.py --tol-rel 0.05          # change PASS threshold (default 0.01)

Output
------
For each matched (mat, pkl) pair the script prints a table with one row per numeric column:

  Status  Field            Column                  N_valid  Bias        RMSE        max_abs
          p99_abs     max_rel     mean_rel    R2      NaN_mismatch

Status codes:
  OK   — max_rel < tol_rel  (default 1 %)
  WARN — 1 % ≤ max_rel < 10 %
  BIG  — max_rel ≥ 10 %      (or absolute error > tol_abs when ref ≈ 0)

Exit code is 1 when any column is BIG (or when no pairs are found), so the
script can gate CI.
"""

import os
import sys
import argparse
import warnings
import csv as _csv
from datetime import datetime
import numpy as np

from utespac.testkit import (
    discover_pairs, load_pkl, load_mat_avg, load_mat_raw,
    compare_avg, compare_raw,
)

# ── paths ────────────────────────────────────────────────────────────────────

ROOT_PY    = os.path.dirname(os.path.abspath(__file__))
MATLAB_DIR = os.path.join(ROOT_PY, "UTESpac_MATLAB")


# ── printing ──────────────────────────────────────────────────────────────────

STAT_KEYS  = ["bias", "RMSE", "max_abs", "p99_abs", "max_rel", "mean_rel", "R2"]
STAT_WIDTHS = [12, 12, 12, 12, 10, 10, 8]


def _fmt(val, width):
    if not isinstance(val, float) or not np.isfinite(val):
        return f"{'nan':>{width}}"
    if abs(val) == 0:
        return f"{'0':>{width}}"
    if abs(val) >= 1000 or (abs(val) < 0.001 and val != 0):
        return f"{val:>{width}.3e}"
    return f"{val:>{width}.6f}"


def print_table(rows, site, stem, tol_rel):
    col_w = 30
    field_w = 22
    n_w = 8
    nm_w = 7

    sep = "-" * (6 + field_w + col_w + n_w + nm_w + sum(STAT_WIDTHS) + len(STAT_WIDTHS) + 4)
    header_line = (
        f"  {'':6s}{'Field':<{field_w}}{'Column':<{col_w}}"
        f"{'N_valid':>{n_w}} {'NaN_mm':>{nm_w}}"
    )
    for key, w in zip(STAT_KEYS, STAT_WIDTHS):
        header_line += f"  {key:>{w}}"

    print(f"\n{'='*len(sep)}")
    print(f"  Site: {site}   File: {stem}   tol_rel={tol_rel:.1%}")
    print(f"{'='*len(sep)}")
    print(header_line)
    print(sep)

    for row in rows:
        status  = row.get("status", "")
        field   = str(row.get("field",  ""))[:field_w]
        column  = str(row.get("column", ""))[:col_w]
        n_valid = row.get("N_valid", float("nan"))
        nan_mm  = row.get("N_nan_mismatch", float("nan"))

        n_str  = f"{n_valid:>{n_w}d}" if isinstance(n_valid, int) else f"{'nan':>{n_w}}"
        nm_str = f"{nan_mm:>{nm_w}d}" if isinstance(nan_mm, int) else f"{'nan':>{nm_w}}"

        line = f"  {status:<6s}{field:<{field_w}}{column:<{col_w}}{n_str} {nm_str}"
        for key, w in zip(STAT_KEYS, STAT_WIDTHS):
            line += f"  {_fmt(row.get(key, float('nan')), w)}"
        print(line)

    print(sep)
    ok   = sum(1 for r in rows if r.get("status","").strip() == "OK")
    warn = sum(1 for r in rows if r.get("status","").strip() == "WARN")
    big  = sum(1 for r in rows if r.get("status","").strip() == "BIG")
    skip = sum(1 for r in rows if r.get("status","").strip() in ("SKIP","MISS_PY","MISS_REF"))
    print(f"  Summary: {ok} OK  {warn} WARN  {big} BIG  {skip} SKIP/MISSING\n")


# ── CSV export ────────────────────────────────────────────────────────────────

ALL_COLS = ["site", "pf_mode", "output_type", "stem", "field", "column", "status",
            "N_total", "N_valid", "N_nan_mismatch",
            "bias", "RMSE", "max_abs", "mean_abs", "p95_abs", "p99_abs",
            "max_rel", "mean_rel", "R2"]


def save_csv(all_rows, path):
    with open(path, "w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nStats saved to: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(ROOT_PY, "comparison_logs")


class _Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, fh):
        self._fh = fh
        self._stdout = sys.stdout
    def write(self, s):
        self._stdout.write(s)
        self._fh.write(s)
    def flush(self):
        self._stdout.flush()
        self._fh.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive statistical comparison of UTESpac Python vs MATLAB outputs.")
    parser.add_argument("--site",     default=None,  help="Limit to one site folder name")
    parser.add_argument("--lpf",      action="store_true", help="LPF files only")
    parser.add_argument("--gpf",      action="store_true", help="GPF files only")
    parser.add_argument("--avg",      action="store_true", help="30-min averaged files only")
    parser.add_argument("--raw",      action="store_true", help="20 Hz raw files only")
    parser.add_argument("--csv",      default=None,  metavar="FILE", help="Save stats to CSV")
    parser.add_argument("--tol-rel",  default=0.01,  type=float,
                        help="Relative error threshold for OK/WARN boundary (default 0.01)")
    parser.add_argument("--tol-abs",  default=1e-9,  type=float,
                        help="Absolute floor used when ref ≈ 0 (default 1e-9)")
    args = parser.parse_args()

    # Auto-generate log file name: comparison_logs/<site>_<YYYYMMDD_HHMMSS>.txt
    os.makedirs(LOG_DIR, exist_ok=True)
    site_tag  = args.site if args.site else "all"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"{site_tag}_{timestamp}.txt")
    csv_path  = args.csv or os.path.join(LOG_DIR, f"{site_tag}_{timestamp}.csv")

    with open(log_path, "w") as log_fh:
        sys.stdout = _Tee(log_fh)

        # Default to GPF when neither --lpf nor --gpf is given (GPF is the mode
        # used for analyses and AmeriFlux submission).
        gpf_only = args.gpf or (not args.lpf and not args.gpf)
        pairs = discover_pairs(
            MATLAB_DIR, ROOT_PY,
            site_filter=args.site,
            lpf_only=args.lpf,
            gpf_only=gpf_only,
            avg_only=args.avg,
            raw_only=args.raw,
        )

        if not pairs:
            print(f"No matched (pkl, mat) pairs found under {MATLAB_DIR} or {ROOT_PY}. "
                  "Check that both pipelines have been run.")
            sys.stdout = sys.stdout._stdout
            sys.exit(1)

        print(f"Found {len(pairs)} matched file pair(s).")
        all_csv_rows = []

        for pair in pairs:
            site   = pair["site"]
            stem   = pair["stem"]
            pf     = pair["pf_mode"]
            otype  = pair["output_type"]

            print(f"\nLoading: {stem}")
            try:
                pkl = load_pkl(pair["pkl_path"])
            except Exception as exc:
                print(f"  ERROR loading pkl: {exc}")
                continue
            try:
                if otype == "raw":
                    mat_ref = load_mat_raw(pair["mat_path"])
                else:
                    mat_ref = load_mat_avg(pair["mat_path"])
            except Exception as exc:
                print(f"  ERROR loading mat: {exc}")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if otype == "raw":
                    rows = compare_raw(pkl, mat_ref, args.tol_rel, args.tol_abs)
                else:
                    rows = compare_avg(pkl, mat_ref, args.tol_rel, args.tol_abs)

            for r in rows:
                r.update({"site": site, "pf_mode": pf, "output_type": otype, "stem": stem})
            all_csv_rows.extend(rows)

            print_table(rows, site, stem, args.tol_rel)

        save_csv(all_csv_rows, csv_path)
        print(f"\nLog saved to:  {log_path}")
        print(f"CSV saved to:  {csv_path}")

        sys.stdout = sys.stdout._stdout

    # Non-zero exit when any column diverges badly, so CI can gate on this.
    n_big = sum(1 for r in all_csv_rows if r.get("status", "").strip() == "BIG")
    if n_big:
        print(f"{n_big} column(s) with BIG divergence.")
        sys.exit(1)


if __name__ == "__main__":
    main()
