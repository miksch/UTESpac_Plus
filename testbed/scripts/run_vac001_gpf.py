"""Run the UTESpac pipeline on data/VAC001 with the global planar fit,
driving find_global_pf's prompts from a script instead of stdin.

Answers fed to find_global_pf (single sector, whole record, no date
barriers) reproduce the EddyPro setup for this IOP (one 0-360 sector):

    Skip 10.85 m?                 -> 1 (process)
    Bin boundary ...              -> <Enter>  (no bins => single sector)
    Date barrier ...              -> <Enter>
    Use all data in range (1)?    -> 1
    Is this correct? / Okay?      -> 1, 1

Requires LPF averaged output already present (run_vac001.py), which
find_global_pf reads through get_data(qualifier="LPF").

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/run_vac001_gpf.py [--detrend constant|linear]
                                             [--dates 1 2 ...] [--compat]
"""

import argparse
import builtins
import os
import sys
from collections import deque

import matplotlib
matplotlib.use("Agg")
matplotlib.use = lambda *a, **k: None   # find_global_pf calls use("TkAgg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import utespac_main as um  # noqa: E402


class ScriptedInput:
    """Replace input() with a queue of answers; fall back to '' (Enter)."""

    def __init__(self, answers):
        self.q = deque(answers)
        self.log = []

    def __call__(self, prompt=""):
        ans = self.q.popleft() if self.q else ""
        self.log.append((prompt.strip(), ans))
        print(f"{prompt}{ans}  [scripted]")
        return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, nargs="*", default=None)
    ap.add_argument("--site", default="VAC001")
    ap.add_argument("--detrend", choices=["linear", "constant"], default="constant")
    ap.add_argument("--compat", action="store_true",
                    help="matlabCompat=True: legacy coefficient indexing, no b0 removal")
    ap.add_argument("--reuse-pf", action="store_true",
                    help="reuse existing PFinfo.pkl instead of recomputing")
    args = ap.parse_args()

    info = um.info
    info["rootFolder"] = os.path.join(ROOT, "data")
    info["PF"]["globalCalculation"] = "global"
    info["PF"]["recalculateGlobalCoefficients"] = not args.reuse_pf
    info["detrendingFormat"] = args.detrend
    info["saveNetCDF"] = False
    info["saveCSV"] = True
    info["saveRawConditionedData"] = True
    info["matlabCompat"] = args.compat

    # skip prompt -> "1"; bins -> Enter; barriers -> Enter; use-all -> "1";
    # confirm -> "1"; begin -> "1". Reuse path only asks the two confirms.
    answers = ["1", "1"] if args.reuse_pf else ["1", "", "", "1", "1", "1"]
    scripted = ScriptedInput(answers)
    builtins.input = scripted

    dates = args.dates if args.dates else "all"
    um.run_utespac(info, um.template, site=args.site, dates=dates)


if __name__ == "__main__":
    main()
