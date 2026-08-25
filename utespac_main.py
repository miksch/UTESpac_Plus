"""UTESpac – Utah Turbulence in Environmental Studies Process and Analysis Code.

Python port of UTESpac v5.0 (MATLAB original by Derek Jensen & Eric Pardyjak,
modified by Diane Wang). This file is the command-line wrapper; the pipeline
itself is :func:`utespac.pipeline.run_utespac` and the settings are
:class:`utespac.run_config.RunConfig` (packaged TOMLs in ``utespac/config/``,
overridable by a ``config/<stage>.toml`` in the working directory or by
flags here).

Usage::

    python utespac_main.py                          # prompts for site and dates, GPF
    python utespac_main.py --site MySite --dates all --pf local --detrend constant
    python utespac_main.py --site MySite --pf global --reuse-pf  # keep PFinfo.json
    python utespac_main.py --site MySite --no-prompts            # scripted PF selection

Interactive prompts (site, dates, planar-fit sectors/dates/confirmation) live
here only; ``--no-prompts`` runs the scripted single-sector selection that
scripts and tests use.
"""

import argparse
import logging
import os
import sys

from utespac.run_config import RunConfig
from utespac.pipeline import run_utespac
from utespac.prompts import ConsolePFPrompter, ScriptedPFSelection
from utespac.site_config import list_sites

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _parse_dates(text):
    """'0' → all; '1 3 4:7' → [1, 3, 4, 5, 6, 7]."""
    text = text.strip()
    if text in ("", "0", "all"):
        return "all"
    rows = []
    for part in text.replace(",", " ").split():
        if ":" in part:
            a, b = part.split(":", 1)
            rows.extend(range(int(a), int(b) + 1))
        else:
            rows.append(int(part))
    return rows


def _choose_site(root):
    sites = list_sites(root)
    if not sites:
        raise SystemExit(f"No site folders under {root}")
    for i, s in enumerate(sites):
        print(f"  {i + 1}. {s}")
    choice = int(input("Please indicate site number of interest: ")) - 1
    return sites[choice]


def _choose_dates(info, site):
    from utespac.find_files import find_files
    _, data_files, _, _ = find_files(dict(info), site=site, dates="all")
    print("\nAvailable dates:")
    for i, row in enumerate(data_files):
        print(f"  {i + 1}. {[os.path.basename(f) if f else None for f in row]}")
    return _parse_dates(input("Input dates of interest (e.g. '1 3 4:7') or '0' for all: "))


def _summary(config, site, dates):
    mode = config.pf.globalCalculation
    print("\n" + "=" * 60)
    print(f"  UTESpac — {'Global' if mode == 'global' else 'Local'} Planar Fit "
          f"({'GPF' if mode == 'global' else 'LPF'}) Mode")
    print("=" * 60)
    print(f"  Site              : {site}")
    print(f"  Dates             : {dates}")
    print(f"  Averaging period  : {config.avgPer} min")
    print(f"  Detrending        : {config.flux.detrendingFormat}")
    if mode == "global":
        print(f"  Recalculate PF    : {config.pf.recalculateGlobalCoefficients}")
        print(f"  PF wind range     : {config.pf.globalCalcMinWind}–{config.pf.globalCalcMaxWind} m/s")
    print("=" * 60)


def main(argv=None):
    ap = argparse.ArgumentParser(description="UTESpac processing pipeline")
    ap.add_argument("--root", default=os.path.join(REPO_ROOT, "data"),
                    help="folder with one sub-folder per site (default: <repo>/data)")
    ap.add_argument("--site", help="site folder name; prompted when omitted")
    ap.add_argument("--dates", help="'all', or rows like '1 3 4:7'; prompted when omitted")
    ap.add_argument("--pf", choices=["global", "local"], help="planar-fit mode (default: pf.toml)")
    ap.add_argument("--reuse-pf", action="store_true", help="reuse <site>/PFinfo.json (GPF)")
    ap.add_argument("--detrend", choices=["linear", "constant"], help="default: flux.toml")
    ap.add_argument("--avg-per", type=int, help="[min] averaging period (default: run.toml)")
    ap.add_argument("--run-config", help="path to a run.toml overriding the packaged one")
    ap.add_argument("--no-prompts", action="store_true",
                    help="scripted planar-fit selection (single sector, all dates), no confirmations")
    ap.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(message)s", stream=sys.stdout)

    overrides = {"rootFolder": args.root}
    if args.avg_per is not None:
        overrides["avgPer"] = args.avg_per
    pf_over = {}
    if args.pf:
        pf_over["globalCalculation"] = args.pf
    if args.reuse_pf:
        pf_over["recalculateGlobalCoefficients"] = False
    flux_over = {"detrendingFormat": args.detrend} if args.detrend else {}
    config = RunConfig.from_config(args.run_config, pf=pf_over or None,
                                   flux=flux_over or None, **overrides)

    site = args.site or _choose_site(config.rootFolder)
    dates = _parse_dates(args.dates) if args.dates else _choose_dates(config.to_info(), site)

    if not args.no_prompts:
        _summary(config, site, dates)
        if config.pf.globalCalculation == "local":
            ans = input("\nOkay to begin analysis? (yes/no): ").strip().lower()
            if ans not in ("yes", "y"):
                print("Analysis cancelled.")
                return 0
    prompter = ScriptedPFSelection() if args.no_prompts else ConsolePFPrompter()
    result = run_utespac(config, site=site, dates=dates, prompter=prompter)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
