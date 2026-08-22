"""VAC001_slow_process.py
Process the VAC001 30-min logger statistics CSV into 48-h UTESpac slow
files (T/RH for the reference-humidity path in fluxes).

Thin config over toa5_tower.process_table. The source is not TOA5: it is
a logger export with a names row and a units row, so a small loader is
supplied. Of the three exports in data/VAC001/raw/slow/, the one with no
suffix (``VAC_001_slow_logger_vars_2023.csv``) is the least modified and is
the one used here.

Keep start/end identical to VAC001_fast_process.py so both tables land in
the same 48-h windows.
"""

import os
from datetime import datetime

import pandas as pd

try:
    from toa5_tower import process_table, level_columns
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from toa5_tower import process_table, level_columns

ROOT_PY  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Plus/
SITE_DIR = os.path.join(ROOT_PY, "data", "VAC001")

start_date = datetime(2023, 7, 6)
end_date   = datetime(2023, 7, 22)

# T/RH columns are labelled with the paired sonic height (10.85) so
# find_instruments pairs them with the sonic; siteInfo.toml's hmp_height
# (10.72) carries the true probe height for the altitude correction.
SONIC_HEIGHT = 10.85
HMP_MAP = {"Temp": "TA_1_1_1", "RH": "RH_1_1_1"}


def load_logger_csv(paths):
    """Load logger-stats CSVs (names row + units row) indexed by TIMESTAMP."""
    dfs = []
    for p in paths:
        df = pd.read_csv(p, header=0, skiprows=[1], index_col=0,
                         na_values=["NaN", "NAN"], low_memory=False)
        df.index = pd.to_datetime(df.index, format="mixed")
        dfs.append(df)
    df = pd.concat(dfs, axis=0).sort_index()
    return df[~df.index.duplicated(keep="first")]


process_table({
    "raw_pattern": os.path.join(SITE_DIR, "raw", "slow",
                                "VAC_001_slow_logger_vars_2023.csv"),
    "out_dir":     os.path.join(SITE_DIR, "utespac"),
    "prefix":      "VAC001",
    "table":       "30min",
    "hz":          1 / 1800,
    "loader":      load_logger_csv,
    "columns":     level_columns(HMP_MAP, SONIC_HEIGHT),
    "start_date":  start_date,
    "end_date":    end_date,
})
