"""Run the UTESpac pipeline on data/VAC001 with the global planar fit,
non-interactively through a ScriptedPFSelection.

The default selection reproduces the EddyPro setup for this IOP: one
0–360 sector, whole record, no date barriers, use all data — the same
answers the old input()-driven run gave ("1", Enter, Enter, "1", "1", "1").

Requires LPF averaged output already present (run_vac001.py), which
find_global_pf reads through get_data(qualifier="LPF").

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/run_vac001_gpf.py [--detrend constant|linear]
                                             [--dates 1 2 ...] [--compat]
                                             [--reuse-pf] [--bins 90 270]
"""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utespac import RunConfig, run_utespac, ScriptedPFSelection  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, nargs="*", default=None)
    ap.add_argument("--site", default="VAC001")
    ap.add_argument("--detrend", choices=["linear", "constant"], default="constant")
    ap.add_argument("--compat", action="store_true",
                    help="matlabCompat=True: legacy coefficient indexing, no b0 removal, "
                         "0.61 sonic coefficient, buoyancy-flux WPL driver")
    ap.add_argument("--reuse-pf", action="store_true",
                    help="reuse existing PFinfo.pkl instead of recomputing")
    ap.add_argument("--bins", type=float, nargs="*", default=[],
                    help="direction-bin boundaries [deg] for every height (default: none)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    config = RunConfig.from_config(
        rootFolder=os.path.join(ROOT, "data"),
        saveNetCDF=False, saveCSV=True, saveRawConditionedData=True,
        matlabCompat=args.compat,
        pf={"globalCalculation": "global",
            "recalculateGlobalCoefficients": not args.reuse_pf},
        flux={"detrendingFormat": args.detrend},
    )
    selection = ScriptedPFSelection(default_bins=list(args.bins))
    result = run_utespac(config, site=args.site, dates=args.dates or "all",
                         prompter=selection)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
