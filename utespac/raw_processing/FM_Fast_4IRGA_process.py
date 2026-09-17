"""FM_Fast_4IRGA_process.py
Process CR1000X IRGASON fast data into 48-h UTESpac-ready files
for siteIRGA (4 IRGASON heights: 32.18, 13.94, 6.35, 4.42 m).

Steps:
  1. Load pairs of CR1000X fast files (each file ≈ 1 day)
  2. Reindex to a uniform 48-h grid at the target sample rate
  3. Write formatted fast-channel files to the siteIRGA output folder

IMPORTANT: Lab Library paths are READ-ONLY. Output is written to
           UTESpac_Python/siteIRGA<start>_<end>/ only.

CR1000X file-index reference:
  20250718_tall  HZ=10  startid=600  endid=627
  20250828_tall  HZ=20  startid=637  endid=674
"""

import os
import pandas as pd
from datetime import datetime, timedelta

try:
    from common import (get_box_path, read_toa5, validate_fast,
                        build_48h_index, timestamp_columns)
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import (get_box_path, read_toa5, validate_fast,
                        build_48h_index, timestamp_columns)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Python/

box_path = get_box_path()

# READ-ONLY: CR1000X ascii data (converted from binary with PC400)
FM_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "20250718_tall", "ascii")

# ── date range / file indices ─────────────────────────────────────────────────
# Modify these for each processing run.
startid        = 604   # CR1000X file index for start date
endid          = 607   # CR1000X file index for end date (exclusive)
HZ             = 10
decimal_places = 1     # 10 Hz → 0.1 s resolution

start_date = datetime(2025, 7, 23)   # must match the date of file at startid
end_date   = datetime(2025, 7, 25)

# Output site folder (created automatically)
site_folder_name = f"siteIRGA{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
FM_processed_dir = os.path.join(ROOT_PY, site_folder_name)
os.makedirs(FM_processed_dir, exist_ok=True)
print(f"Output → {FM_processed_dir}")

# ── main processing loop ──────────────────────────────────────────────────────

curr_date = start_date

for fi in range(startid, endid, 2):
    cr1000x_files = [
        os.path.join(FM_dir, f"TOA5_6653fast_20hz_data{fi}.dat"),
        os.path.join(FM_dir, f"TOA5_6653fast_20hz_data{fi + 1}.dat"),
    ]
    cr1000x_df = pd.concat(
        [read_toa5(f) for f in cr1000x_files],
        axis=0)
    cr1000x_df  = cr1000x_df[~cr1000x_df.index.duplicated(keep="first")]
    timestrings = pd.to_datetime(cr1000x_df.index, format="mixed")
    cr1000x_df.index = pd.to_datetime(timestrings)

    if not validate_fast(cr1000x_df, curr_date, HZ, f"fi={fi}"):
        curr_date += timedelta(days=2)
        continue

    print(f"Processing fi={fi}  ({curr_date.date()}) …")

    # Build 48-h uniform time index
    day_start       = timestrings[0].floor("D")
    full_time_index = build_48h_index(day_start, hz=HZ)
    cr1000x_df_full = cr1000x_df.reindex(full_time_index)

    # Assemble output table
    ts = cr1000x_df_full.index
    df = timestamp_columns(ts, decimals=decimal_places)

    # IRGASON at 32.18 m
    df["Ux_32.18"]         = cr1000x_df_full["Ux_2"].values
    df["Uy_32.18"]         = cr1000x_df_full["Uy_2"].values
    df["Uz_32.18"]         = cr1000x_df_full["Uz_2"].values
    df["T_Sonic_32.18"]    = cr1000x_df_full["Ts_2"].values
    df["diagnostic_32.18"] = cr1000x_df_full["diag_sonic_2"].values
    df["Pressure_32.18"]   = cr1000x_df_full["cell_press_2"].values
    df["H2O_32.18"]        = cr1000x_df_full["H2O_2"].values
    df["H2Osig_32.18"]     = cr1000x_df_full["H2O_sig_strgth_2"].values
    df["CO2_32.18"]        = cr1000x_df_full["CO2_2"].values
    df["CO2sig_32.18"]     = cr1000x_df_full["CO2_sig_strgth_2"].values
    df["gas_diag_32.18"]   = cr1000x_df_full["diag_irga_2"].values

    # IRGASON at 13.94 m
    df["Ux_13.94"]         = cr1000x_df_full["Ux_3"].values
    df["Uy_13.94"]         = cr1000x_df_full["Uy_3"].values
    df["Uz_13.94"]         = cr1000x_df_full["Uz_3"].values
    df["T_Sonic_13.94"]    = cr1000x_df_full["Ts_3"].values
    df["diagnostic_13.94"] = cr1000x_df_full["diag_sonic_3"].values
    df["Pressure_13.94"]   = cr1000x_df_full["cell_press_3"].values
    df["H2O_13.94"]        = cr1000x_df_full["H2O_3"].values
    df["H2Osig_13.94"]     = cr1000x_df_full["H2O_sig_strgth_3"].values
    df["CO2_13.94"]        = cr1000x_df_full["CO2_3"].values
    df["CO2sig_13.94"]     = cr1000x_df_full["CO2_sig_strgth_3"].values
    df["gas_diag_13.94"]   = cr1000x_df_full["diag_irga_3"].values

    # IRGASON at 6.35 m
    df["Ux_6.35"]          = cr1000x_df_full["Ux_1"].values
    df["Uy_6.35"]          = cr1000x_df_full["Uy_1"].values
    df["Uz_6.35"]          = cr1000x_df_full["Uz_1"].values
    df["T_Sonic_6.35"]     = cr1000x_df_full["Ts_1"].values
    df["diagnostic_6.35"]  = cr1000x_df_full["diag_sonic_1"].values
    df["Pressure_6.35"]    = cr1000x_df_full["cell_press_1"].values
    df["H2O_6.35"]         = cr1000x_df_full["H2O_1"].values
    df["H2Osig_6.35"]      = cr1000x_df_full["H2O_sig_strgth_1"].values
    df["CO2_6.35"]         = cr1000x_df_full["CO2_1"].values
    df["CO2sig_6.35"]      = cr1000x_df_full["CO2_sig_strgth_1"].values
    df["gas_diag_6.35"]    = cr1000x_df_full["diag_irga_1"].values

    # IRGASON at 4.42 m
    df["Ux_4.42"]          = cr1000x_df_full["Ux_4"].values
    df["Uy_4.42"]          = cr1000x_df_full["Uy_4"].values
    df["Uz_4.42"]          = cr1000x_df_full["Uz_4"].values
    df["T_Sonic_4.42"]     = cr1000x_df_full["Ts_4"].values
    df["diagnostic_4.42"]  = cr1000x_df_full["diag_sonic_4"].values
    df["Pressure_4.42"]    = cr1000x_df_full["cell_press_4"].values
    df["H2O_4.42"]         = cr1000x_df_full["H2O_4"].values
    df["H2Osig_4.42"]      = cr1000x_df_full["H2O_sig_strgth_4"].values
    df["CO2_4.42"]         = cr1000x_df_full["CO2_4"].values
    df["CO2sig_4.42"]      = cr1000x_df_full["CO2_sig_strgth_4"].values
    df["gas_diag_4.42"]    = cr1000x_df_full["diag_irga_4"].values

    datestr1 = f"{ts[0].year}{ts[0].month:02d}{ts[0].day:02d}"
    endtime  = ts[0] + pd.Timedelta(hours=48)
    datestr2 = f"{endtime.year}{endtime.month:02d}{endtime.day:02d}"
    out_path = os.path.join(FM_processed_dir,
                            f"FMDOL_{HZ}Hz_{datestr1}000000_{datestr2}000000.txt")
    df.to_csv(out_path, header=False, index=False, sep=",")
    print(f"  → {os.path.basename(out_path)}")

    curr_date += timedelta(days=2)
    del cr1000x_df, df

print("\nDone.")
