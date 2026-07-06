"""FM_DSfast_data_process.py
Process CR3000 DS tower 10-Hz fast data into 48-h UTESpac-ready files
(2 sonic heights: 6.83 m, 4.13 m).

Steps:
  1. Load 48 CR3000 hourly files per 48-h period (one file per hour)
  2. Reindex to a uniform 48-h grid at 10 Hz
  3. Write formatted fast-channel files to the output folder

IMPORTANT: Lab Library paths are READ-ONLY. Output is written to
           UTESpac_Python/siteFMDS<start>_<end>/ only.
"""

import os
import platform
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Python/

if platform.system() == "Darwin":
    box_path = os.path.expanduser("~/Library/CloudStorage/Box-Box")
else:  # Windows
    box_path = os.path.expanduser("~/Box")

# READ-ONLY: CR3000 ascii data
FM_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "20250527_ds", "cr3000", "ascii")

# ── date range / file indices ─────────────────────────────────────────────────
# Modify these for each processing run. start_date must match the date of the
# file at startid (48 hourly files = 1 day, so step=48 per 48-h period).
startid        = 1380  # CR3000 file index for start date
endid          = 4787  # CR3000 file index for end date (exclusive)
HZ             = [10]
decimal_places = 1     # 10 Hz → 0.1 s resolution

start_date = datetime(2025, 1, 1)   # update to match the date of file at startid
end_date   = datetime(2025, 5, 27)  # update to match endid

# Output site folder (created automatically)
site_folder_name = f"siteFMDS{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
FM_processed_dir = os.path.join(ROOT_PY, site_folder_name)
os.makedirs(FM_processed_dir, exist_ok=True)
print(f"Output → {FM_processed_dir}")

# ── timestamp validator ───────────────────────────────────────────────────────

def _validate_fast(df, expected_date, expected_hz, label):
    """Check date alignment, sampling frequency, and coverage of fast data."""
    if df is None or len(df) < 2:
        print(f"  SKIP {label}: fast data is empty.")
        return False

    first_date = df.index[0].date()
    if first_date != expected_date.date():
        print(f"  SKIP {label}: first timestamp {first_date} ≠ expected {expected_date.date()}.")
        return False

    # Cast to ns before asi8 — pandas 3.x stores ms-resolution index as µs in asi8
    idx_ns      = df.index[:min(200, len(df))].astype('datetime64[ns]')
    intervals_s = np.diff(idx_ns.view('int64')) / 1e9
    dt_s        = float(np.median(intervals_s))
    if dt_s <= 0:
        print(f"  SKIP {label}: cannot determine sampling frequency (dt={dt_s:.4f} s).")
        return False
    hz_actual = round(1.0 / dt_s)
    if hz_actual != expected_hz:
        print(f"  SKIP {label}: measured {hz_actual} Hz ≠ expected {expected_hz} Hz.")
        return False

    expected_rows = expected_hz * 48 * 3600
    coverage      = len(df) / expected_rows
    if coverage < 0.10:
        print(f"  SKIP {label}: only {coverage:.0%} coverage "
              f"({len(df):,} / {expected_rows:,} expected rows).")
        return False

    print(f"  Timestamps OK: starts {df.index[0]}, {hz_actual} Hz, "
          f"{len(df):,} rows ({coverage:.0%} of 48 h)")
    return True


# ── main processing loop ──────────────────────────────────────────────────────

last_row_previous = None
curr_date = start_date

for fi in range(startid, endid, 48):
    cr3000_files   = [os.path.join(FM_dir, f"TOA5_5500.FastResponse_{fi + ii}.dat")
                      for ii in range(48)]
    existing_files = [f for f in cr3000_files if os.path.exists(f)]

    if not existing_files:
        print(f"Skipping fi={fi} ({curr_date.date()}): no files found.")
        curr_date += timedelta(days=2)
        continue

    try:
        cr3000_df = pd.concat(
            [pd.read_csv(f, skiprows=[0, 2, 3], index_col=[0],
                         na_values=["NaN", "NAN"], parse_dates=True)
             for f in existing_files],
            axis=0)
    except Exception as e:
        print(f"Failed to load fi={fi} ({curr_date.date()}): {e}")
        curr_date += timedelta(days=2)
        continue

    cr3000_df   = cr3000_df[~cr3000_df.index.duplicated(keep="first")]
    timestrings = pd.to_datetime(cr3000_df.index, format="mixed")

    if not _validate_fast(cr3000_df, curr_date, HZ[0], f"fi={fi}"):
        curr_date += timedelta(days=2)
        continue

    print(f"Processing fi={fi}  ({curr_date.date()}) …")

    # Build 48-h uniform time index
    start_time      = pd.to_datetime(timestrings[0].floor("D"))
    full_time_index = pd.date_range(start=start_time,
                                    end=start_time + pd.Timedelta(hours=48),
                                    freq="100ms", inclusive="left")
    cr3000_df_full  = cr3000_df.reindex(full_time_index)

    # Fill first row from previous period if missing
    if last_row_previous is not None and cr3000_df_full.iloc[0].isna().all():
        cr3000_df_full.iloc[0] = last_row_previous.values

    timestrings = cr3000_df_full.index
    print(f"  {len(full_time_index):,} expected rows, {len(cr3000_df):,} before reindex, "
          f"{len(cr3000_df_full):,} after reindex")

    # Assemble output table
    ts = timestrings
    df = pd.DataFrame()
    df["year"]   = [int(t.year)        for t in ts]
    df["day"]    = [int(t.day_of_year) for t in ts]
    df["HM"]     = [int(f"{t.hour:02d}{t.minute:02d}") for t in ts]
    df["second"] = [f"{t.second + t.microsecond / 1e6:.{decimal_places}f}" for t in ts]

    # CSAT3 at 6.83 m
    df["Ux_6.83"]         = cr3000_df_full["Ux_2_0"].values
    df["Uy_6.83"]         = cr3000_df_full["Uy_2_0"].values
    df["Uz_6.83"]         = cr3000_df_full["Uz_2_0"].values
    df["T_Sonic_6.83"]    = cr3000_df_full["Ts_2_0"].values
    df["diagnostic_6.83"] = cr3000_df_full["diag_sonic_2_0"].values

    # CSAT3 at 4.13 m
    df["Ux_4.13"]         = cr3000_df_full["Ux_1_0"].values
    df["Uy_4.13"]         = cr3000_df_full["Uy_1_0"].values
    df["Uz_4.13"]         = cr3000_df_full["Uz_1_0"].values
    df["T_Sonic_4.13"]    = cr3000_df_full["Ts_1_0"].values
    df["diagnostic_4.13"] = cr3000_df_full["diag_sonic_1_0"].values

    datestr1 = f"{ts[0].year}{ts[0].month:02d}{ts[0].day:02d}"
    endtime  = ts[0] + pd.Timedelta(hours=48)
    datestr2 = f"{endtime.year}{endtime.month:02d}{endtime.day:02d}"
    out_path = os.path.join(FM_processed_dir,
                            f"FMDS_{HZ[0]}Hz_{datestr1}000000_{datestr2}000000.txt")
    df.to_csv(out_path, header=False, index=False, sep=",")
    print(f"  → {os.path.basename(out_path)}")

    last_row_previous = cr3000_df.iloc[-1]
    curr_date += timedelta(days=2)
    del cr3000_df, df

print("\nDone.")
