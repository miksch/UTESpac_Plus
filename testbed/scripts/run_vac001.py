"""Run the UTESpac pipeline on data/VAC001 non-interactively (local planar fit).

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/run_vac001.py            # all dates, LPF
    python testbed/scripts/run_vac001.py --dates 1 2

Global PF needs the interactive sector/date prompts in find_global_pf and is
not driven from here; see the audit task doc for the planned non-interactive
path.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import utespac_main as um  # noqa: E402  (module-level info/template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, nargs="*", default=None,
                    help="1-based date rows to process (default: all)")
    ap.add_argument("--site", default="VAC001")
    ap.add_argument("--detrend", choices=["linear", "constant"], default="linear")
    ap.add_argument("--netcdf", action="store_true", help="also write .nc")
    args = ap.parse_args()

    info = um.info
    info["rootFolder"] = os.path.join(ROOT, "data")
    info["PF"]["globalCalculation"] = "local"
    info["PF"]["recalculateGlobalCoefficients"] = False
    info["detrendingFormat"] = args.detrend
    info["saveNetCDF"] = args.netcdf
    info["saveCSV"] = True
    info["saveRawConditionedData"] = True

    dates = args.dates if args.dates else "all"
    um.run_utespac(info, um.template, site=args.site, dates=dates)


if __name__ == "__main__":
    main()
