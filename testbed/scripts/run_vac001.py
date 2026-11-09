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
    ap.add_argument("--root", default=None,
                    help="configuration root: one sub-folder per site with its "
                         "siteInfo.toml and PFinfo.json (default: <repo>/data when "
                         "it holds the site, else $UTESPAC_DATA_ROOT)")
    ap.add_argument("--data-root", default=os.environ.get("UTESPAC_DATA_ROOT"),
                    help="data root: <site>/utespac input files and <site>/output "
                         "products (default: $UTESPAC_DATA_ROOT, else --root)")
    ap.add_argument("--dates", type=int, nargs="*", default=None,
                    help="1-based date rows to process (default: all)")
    ap.add_argument("--site", default="VAC001")
    ap.add_argument("--detrend", choices=["linear", "constant"], default="linear")
    args = ap.parse_args()
    if args.root is None:
        repo_data = os.path.join(ROOT, "data")
        args.root = (repo_data if os.path.isdir(os.path.join(repo_data, args.site))
                     else os.environ.get("UTESPAC_DATA_ROOT", repo_data))

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    config = RunConfig.from_config(
        rootFolder=args.root, dataFolder=args.data_root,
        saveCSV=True, saveRawConditionedData=True,
        pf={"globalCalculation": "local", "recalculateGlobalCoefficients": False},
        flux={"detrendingFormat": args.detrend},
    )
    result = run_utespac(config, site=args.site, dates=args.dates or "all")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
