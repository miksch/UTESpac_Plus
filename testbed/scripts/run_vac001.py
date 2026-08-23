"""Run the UTESpac pipeline on data/VAC001 non-interactively (local planar fit).

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/run_vac001.py            # all dates, LPF
    python testbed/scripts/run_vac001.py --dates 1 2 --detrend constant

Global PF: see run_vac001_gpf.py.
"""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utespac import RunConfig, run_utespac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, nargs="*", default=None,
                    help="1-based date rows to process (default: all)")
    ap.add_argument("--site", default="VAC001")
    ap.add_argument("--detrend", choices=["linear", "constant"], default="linear")
    ap.add_argument("--netcdf", action="store_true", help="also write .nc")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    config = RunConfig.from_config(
        rootFolder=os.path.join(ROOT, "data"),
        saveNetCDF=args.netcdf, saveCSV=True, saveRawConditionedData=True,
        pf={"globalCalculation": "local", "recalculateGlobalCoefficients": False},
        flux={"detrendingFormat": args.detrend},
    )
    result = run_utespac(config, site=args.site, dates=args.dates or "all")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
