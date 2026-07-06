"""FM_1min_data_process.py
Process LICOR daqm (1-min stats) and CR1000X slow (1-min) data into
daily FM_DOL_1min/*.txt files consumed by FM_FastNSlow_*_process.py.

Steps:
  1. Optionally unzip LICOR daqm archives (see commented block below)
  2. Load LICOR daqm 1-min stats files (T/RH at 15/30/51.5 m + radiation at 51.5 m)
  3. Load CR1000X 1-min slow files (T/RH at 6.35 m, radiation at 4.22 m)
  4. Merge and write one daily combined file per day

IMPORTANT: Lab Library paths are READ-ONLY (raw data).
           Output is written to Lab Library/.../processed/FM_DOL_1min/ —
           the intermediate product consumed by FM_FastNSlow_*_process.py.
"""

import os
import glob
import platform
import pandas as pd
from zipfile import ZipFile
from datetime import datetime, timedelta

# ── paths ─────────────────────────────────────────────────────────────────────

if platform.system() == "Darwin":
    box_path = os.path.expanduser("~/Library/CloudStorage/Box-Box")
else:  # Windows
    box_path = os.path.expanduser("~/Box")

# READ-ONLY: CR1000X ascii data (converted from binary with PC400)
cr1000xFM_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "20251110_tall", "ascii")

# Processed output (consumed by FM_FastNSlow_*_process.py)
FM_processed_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "processed", "FM_DOL_1min")

# LICOR daqm zip archives and unzipped folder
licor_zipfile_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "20250828_DOL50m_Data", "2025-11-10", "Daqm")
unzipfile_dir = os.path.join(
    box_path, "Diane", "French Meadows", "data", "DOL_unziped",
    "20250901-20251110")

# ── CR1000X file-index reference ──────────────────────────────────────────────
# Update startid for each processing run.
# startid=516  → 2025-04-01  (20250527_tall ascii)
# startid=638  → 2025-07-24  (20250718_tall ascii)
# startid=681  → 2025-09-01  (20251110_tall ascii)
startid = 681

# ── date range (used for progress reporting only) ─────────────────────────────
start_date = datetime(2025, 9, 1)
end_date   = datetime(2025, 11, 9)

# ── optional: unzip daqm archives ─────────────────────────────────────────────
# Uncomment and adjust month/daylength to extract daqm files for the target range.
# moni = [11]
# daylength = [9]
# for mi in range(len(moni)):
#     for di in range(daylength[mi]):
#         zipfile = os.path.join(licor_zipfile_dir,
#                                f"2025-{moni[mi]:02}-{1+di:02}-000000-daqm.zip")
#         with ZipFile(zipfile) as z:
#             z.extractall(unzipfile_dir)

# ── main processing loop ──────────────────────────────────────────────────────

log_file_path = sorted(glob.glob(os.path.join(unzipfile_dir, "2025-*-000000-daqm.log")))
filelength    = len(log_file_path)

curr_date = start_date

for fi in range(filelength - 1):
    print(f"Processing {curr_date.date()} …")

    # LICOR daqm: current day + first row of next day (to align with CR1000X start)
    licor_df = pd.read_csv(log_file_path[fi], sep=r"\s+", engine="python",
                           header=0, skiprows=[1, 2])
    licor_df_next = pd.read_csv(log_file_path[fi + 1], sep=r"\s+", engine="python",
                                header=0, skiprows=[1], na_values=["-9999"])
    licor_df = pd.concat([licor_df, licor_df_next.iloc[[0]]], ignore_index=True)
    licor_df["TIMESTAMP"] = pd.to_datetime(licor_df["DATE"] + " " + licor_df["TIME"])
    licor_df.set_index("TIMESTAMP", inplace=True)
    licor_df = licor_df[~licor_df.index.duplicated(keep="first")]

    # CR1000X slow: file index has a gap at 718→720 (one file was skipped in the field)
    file_idx     = startid + fi
    actual_idx   = file_idx if file_idx <= 718 else file_idx + 1
    cr1000x_path = os.path.join(cr1000xFM_dir, f"TOA5_6653slow_avg_data{actual_idx}.dat")
    cr1000x_df   = pd.read_csv(cr1000x_path, skiprows=[0, 2, 3], index_col=[0],
                               na_values=["NaN", "NAN"], parse_dates=True)
    cr1000x_df   = cr1000x_df[~cr1000x_df.index.duplicated(keep="first")]
    cr1000x_df   = cr1000x_df.reindex(licor_df.index)

    # Assemble output table
    ts = licor_df.index
    df = pd.DataFrame()
    df["year"]      = [int(t.year)        for t in ts]
    df["day"]       = [int(t.day_of_year) for t in ts]
    df["HM"]        = [int(f"{t.hour:02d}{t.minute:02d}") for t in ts]
    df["second"]    = 0

    df["Temp_6.35"]   = cr1000x_df["AirTC_ee181_1_Avg"].values
    df["RH_6.35"]     = cr1000x_df["RH_ee181_1_Avg"].values
    df["Temp_15"]     = licor_df["TA_1_1_1"].values
    df["RH_15"]       = licor_df["RH_1_1_1"].values
    df["Temp_30"]     = licor_df["TA_1_2_1"].values
    df["RH_30"]       = licor_df["RH_1_2_1"].values
    df["Temp_51.5"]   = licor_df["TA_1_3_1"].values
    df["RH_51.5"]     = licor_df["RH_1_3_1"].values
    df["SWIN_4.22"]   = cr1000x_df["SWTop_Avg"].values
    df["SWOUT_4.22"]  = cr1000x_df["SWBottom_Avg"].values
    df["LWIN_4.22"]   = cr1000x_df["LWTopC_Avg"].values
    df["LWOUT_4.22"]  = cr1000x_df["LWBottomC_Avg"].values
    df["Rn_4.22"]     = cr1000x_df["Rn_Avg"].values
    df["ALBEDO_4.22"] = cr1000x_df["albedo_Avg"].values
    df["SWIN_51.5"]   = licor_df["SWIN_1_1_1"].values
    df["SWOUT_51.5"]  = licor_df["SWOUT_1_1_1"].values
    df["LWIN_51.5"]   = licor_df["LWIN_1_1_1"].values
    df["LWOUT_51.5"]  = licor_df["LWOUT_1_1_1"].values
    df["Rn_51.5"]     = licor_df["RN_1_1_1"].values
    df["ALBEDO_51.5"] = licor_df["ALB_1_1_1"].values

    datestr  = cr1000x_df.index[0].strftime("%Y%m%d")
    out_path = os.path.join(FM_processed_dir, f"FM_DOL_1min_{datestr}000000.txt")
    df.to_csv(out_path, header=True, index=False, sep=",")
    print(f"  → FM_DOL_1min_{datestr}000000.txt")

    curr_date += timedelta(days=1)
    del licor_df, cr1000x_df, df

print("\nDone.")
