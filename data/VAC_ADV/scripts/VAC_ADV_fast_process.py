"""VAC_ADV_fast_process.py
Process VAC_ADV fast (two-level CSAT3 tower, 2023 IOP) data into 48-h
UTESpac fast files.

Thin config over toa5_tower.process_table — see that module's docstring
for the pipeline. Raw files are read from data/VAC_ADV/raw/fast/cc_30min/
(READ-ONLY) and outputs go to data/VAC_ADV/utespac/. Unlike the VAC001
TOA5 files, these are 30-min CSV exports with a single names row, hence
``read_kwargs``.

Logger: 10 Hz, 30-min files VAC_ADV_yyyy_mm_dd_HHMMSS.dat. Two levels
(heights from eddypro/metadata/vac_adv_2023_iop.metadata; the metadata
lists both sonics as CSAT3 but the upper level is an IRGASON — user
ruling 2026-08-24):
  7.44 m — IRGASON (CO2 mg/m³, H2O g/m³, signal strengths, gas
           diagnostic, PA barometer)
  3.07 m — CSAT3 + LI-7500 (CO2 and H2O in mmol/m³ → LiCO2/LiH2O
           templates; diag_irga_2 is the LI-7500 diagnostic word, healthy
           ≈ 249). diag_sonic_2 is a free-running 0-63 sample counter,
           not a diagnostic — not mapped.
Fine-wire thermocouples FWT_6m/7m/8m ride at their own heights (no sonic
there, so no fw fluxes — profile only). Unmapped source columns:
CO2_density_fast_tmpr, T_SONIC_corr, TA_1_1_1 (slow value held at 10 Hz).
"""

import os
from datetime import datetime

SITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data/VAC_ADV
ROOT_PY  = os.path.dirname(os.path.dirname(SITE_DIR))                   # UTESpac_Plus/

try:
    from toa5_tower import process_table, level_columns
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.join(ROOT_PY, "raw_processing"))
    from toa5_tower import process_table, level_columns

# 2023 IOP: raw fast data span 2023-07-10 09:00 to 2023-07-20 23:59 with
# no missing 30-min files. The end date is the exclusive bound of the
# last 48-h window.
start_date = datetime(2023, 7, 10)
end_date   = datetime(2023, 7, 22)

UPPER_HEIGHT = 7.44   # [m] IRGASON, from the EddyPro metadata
LOWER_HEIGHT = 3.07   # [m] CSAT3 + LI-7500

UPPER_MAP = {
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
}

LOWER_MAP = {
    "Ux":          "Ux_2",
    "Uy":          "Uy_2",
    "Uz":          "Uz_2",
    "T_Sonic":     "Ts_2",
    "LiCO2":       "CO2_density_2",    # mmol m-3
    "LiH2O":       "H2O_density_2",    # mmol m-3
    "Li_gas_diag": "diag_irga_2",
    "Pressure":    "press_amb_2",      # kPa
}

FW_MAP = {
    "FW_6": "FWT_6m",
    "FW_7": "FWT_7m",
    "FW_8": "FWT_8m",
}

process_table({
    "raw_pattern": os.path.join(SITE_DIR, "raw", "fast", "cc_30min",
                                "VAC_ADV_*.dat"),
    "out_dir":     os.path.join(SITE_DIR, "utespac"),
    "prefix":      "VAC_ADV",
    "table":       "10Hz",   # used in filenames, header name, and siteInfo tableNames
    "hz":          10,
    "read_kwargs": {"skiprows": []},   # single names row, not TOA5
    "columns":     {**level_columns(UPPER_MAP, UPPER_HEIGHT),
                    **level_columns(LOWER_MAP, LOWER_HEIGHT),
                    **FW_MAP},
    "start_date":  start_date,
    "end_date":    end_date,
})
