"""VAC001_fast_process.py
Process VAC001 fast (sonic/IRGA TOA5) data into 48-h UTESpac fast files.

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
    from toa5_tower import (process_table, level_columns,
                            FAST_SONIC_MAP, FAST_IRGA_MAP)
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import get_box_path
    from toa5_tower import (process_table, level_columns,
                            FAST_SONIC_MAP, FAST_IRGA_MAP)

ROOT_PY  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Plus/
box_path = get_box_path()

# TODO: date range to process (advances in 48-h windows)
start_date = datetime(2026, 1, 1)
end_date   = datetime(2026, 1, 31)

site_folder = (f"siteVAC001{start_date.strftime('%Y%m%d')}"
               f"_{end_date.strftime('%Y%m%d')}")

SONIC_GAS = {**FAST_SONIC_MAP, **FAST_IRGA_MAP}

process_table({
    # TODO: READ-ONLY raw TOA5 fast files (ascii from PC400/CardConvert)
    "raw_pattern": os.path.join(box_path, "Lab Library", "TODO",
                                "TOA5_*fast*.dat"),
    "out_dir":     os.path.join(ROOT_PY, site_folder),
    "prefix":      "VAC001",
    "table":       "20Hz",   # used in filenames, header name, and siteInfo tableNames
    "hz":          20,       # TODO: logger scan rate
    # TODO: one level_columns() entry per sonic: height [m] ← logger
    # instrument suffix. Use FAST_SONIC_MAP alone for a sonic without a
    # gas analyzer; override map entries for non-EasyFlux column names.
    "columns": {
        **level_columns(SONIC_GAS,      32.18, "_2"),
        **level_columns(SONIC_GAS,      13.94, "_3"),
        **level_columns(FAST_SONIC_MAP,  3,    "_1"),
    },
    "start_date": start_date,
    "end_date":   end_date,
})
