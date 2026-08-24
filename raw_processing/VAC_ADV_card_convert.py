"""VAC_ADV_card_convert.py
Convert VAC_ADV 2023 binary card files (TOB3) to TOA5 for the IOP.

Thin config over card_convert — see that module's docstring. Source is
the K: archive (READ-ONLY); converted TOA5 files land locally under
data/VAC_ADV/raw/card_convert/ (gitignored). IOP window matches
VAC_ADV_fast_process, selecting the 20230720 and 20230728 card pulls.
"""

import os
from datetime import datetime
from pathlib import Path

try:
    from card_convert import csi_card_convert, select_pull_dirs
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from card_convert import csi_card_convert, select_pull_dirs

ROOT_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Plus/

prefix  = "25982"
bin_dir = Path("K:/data/TREX/VAC/MCM/VAC_ADV/2023/binary")
out_dir = Path(ROOT_PY, "data", "VAC_ADV", "raw", "card_convert")

# 2023 IOP window (same bounds as VAC_ADV_fast_process)
start_date = datetime(2023, 7, 10)
end_date   = datetime(2023, 7, 22)

file_globs = [
    f"{prefix}_Time_Series_*.dat",
    f"{prefix}_Flux_CSFormat_*.dat",
    # f"{prefix}_Flux_AmeriFluxFormat_*.dat",
    # f"{prefix}_Flux_Notes_*.dat",
]

if __name__ == "__main__":
    out_dir.mkdir(parents=True, exist_ok=True)
    pull_dirs = select_pull_dirs(bin_dir, start_date, end_date)
    print("Pull folders:", ", ".join(d.name for d in pull_dirs))

    for g in file_globs:
        file_list = [f for d in pull_dirs for f in sorted(d.glob(g))]
        csi_card_convert(file_list, out_dir)
