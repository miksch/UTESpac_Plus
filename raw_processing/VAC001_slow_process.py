"""VAC001_slow_process.py
Process VAC001 slow (1-min TOA5) data into 48-h UTESpac 1-min files.

Thin config over toa5_tower.process_table — see that module's docstring
for the pipeline. Edit the TODO lines for each site/run; column source
names are on line 2 of any raw TOA5 file.

IMPORTANT: raw_pattern points at READ-ONLY raw data. Output goes to
           UTESpac_Plus/siteVAC001<start>_<end>/ only.
"""

import os
from datetime import datetime

try:
    from common import get_box_path
    from toa5_tower import process_table, level_columns
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import get_box_path
    from toa5_tower import process_table, level_columns

ROOT_PY  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Plus/
box_path = get_box_path()

# TODO: date range to process (advances in 48-h windows; keep identical
# to VAC001_fast_process.py so both tables land in the same site folder)
start_date = datetime(2026, 1, 1)
end_date   = datetime(2026, 1, 31)

site_folder = (f"siteVAC001{start_date.strftime('%Y%m%d')}"
               f"_{end_date.strftime('%Y%m%d')}")

# TODO: UTESpac base name → TOA5 base name for one T/RH level
HMP_MAP = {"Temp": "AirTC_Avg", "RH": "RH_Avg"}

process_table({
    # TODO: READ-ONLY raw TOA5 slow files (ascii from PC400/CardConvert)
    "raw_pattern": os.path.join(box_path, "Lab Library", "TODO",
                                "TOA5_*slow*.dat"),
    "out_dir":     os.path.join(ROOT_PY, site_folder),
    "prefix":      "VAC001",
    "table":       "1min",
    "hz":          1 / 60,
    "interpolate_limit": 2,   # bridge ≤2 missing minutes (timestamp jitter)
    # TODO: one level_columns() entry per T/RH level; add radiation or
    # other slow channels as plain "name_height": "source" entries.
    "columns": {
        **level_columns(HMP_MAP, 32.18, "_2"),
        **level_columns(HMP_MAP, 13.94, "_3"),
        **level_columns(HMP_MAP,  3,    "_1"),
    },
    "start_date": start_date,
    "end_date":   end_date,
})
