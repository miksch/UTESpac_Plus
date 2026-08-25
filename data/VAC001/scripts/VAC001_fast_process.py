"""VAC001_fast_process.py
Process VAC001 fast (IRGASON + fine-wire TOA5 ``Time_Series``) data into
48-h UTESpac fast files.

Thin config over toa5_tower.process_table — see that module's docstring
for the pipeline. Raw files are read from data/VAC001/raw/fast/ (READ-ONLY)
and outputs go to data/VAC001/utespac/.

Logger: CR6, program 2023_VAC_EZ_v1.CR6, 20 Hz. Source column names (line 2
of any raw file): TIMESTAMP, RECORD, Ux, Uy, Uz, T_SONIC, diag_sonic,
CO2_density, CO2_density_fast_tmpr, H2O_density, diag_irga, T_SONIC_corr,
TA_1_1_1, PA, CO2_sig_strgth, H2O_sig_strgth, FW.
"""

import os
from datetime import datetime

SITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data/VAC001
ROOT_PY  = os.path.dirname(os.path.dirname(SITE_DIR))                   # UTESpac_Plus/

try:
    from toa5_tower import process_table, level_columns
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.join(ROOT_PY, "raw_processing"))
    from toa5_tower import process_table, level_columns

# 2023 IOP: raw fast data span 2023-07-06 00:00 to 2023-07-21 00:00. The
# end date is the exclusive bound of the last 48-h window.
start_date = datetime(2023, 7, 6)
end_date   = datetime(2023, 7, 22)

SONIC_HEIGHT = 10.85   # [m] surveyed; EddyPro metadata rounds to 11.00

# UTESpac base name ← this logger's TOA5 name (EasyFlux-CR6 naming differs
# from the defaults in toa5_tower.FAST_*_MAP).
VAC_FAST_MAP = {
    "Ux":         "Ux",
    "Uy":         "Uy",
    "Uz":         "Uz",
    "T_Sonic":    "T_SONIC",
    "diagnostic": "diag_sonic",
    "CO2":        "CO2_density",       # mg m-3
    "H2O":        "H2O_density",       # g m-3
    "gas_diag":   "diag_irga",
    "CO2sig":     "CO2_sig_strgth",
    "H2Osig":     "H2O_sig_strgth",
    "Pressure":   "PA",                # kPa
    "FW":         "FW",                # fine-wire thermocouple, deg C
}

process_table({
    "raw_pattern": os.path.join(SITE_DIR, "raw", "fast", "VAC__Time_Series_*.dat"),
    "out_dir":     os.path.join(SITE_DIR, "utespac"),
    "prefix":      "VAC001",
    "table":       "20Hz",   # used in filenames, header name, and siteInfo tableNames
    "hz":          20,
    "columns":     level_columns(VAC_FAST_MAP, SONIC_HEIGHT),
    "start_date":  start_date,
    "end_date":    end_date,
})
