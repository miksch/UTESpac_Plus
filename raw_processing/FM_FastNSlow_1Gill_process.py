"""FM_FastNSlow_1Gill_process.py
Process LICOR/Gill 50 m data (10-Hz fast + 1-min slow) into 48-h UTESpac-ready files.

Steps:
  1. Optionally unzip LICOR daqm archives (see commented block below)
  2. Load 1-min daqm stats files (LICOR slow: T/RH at 15/30/51.5 m)
  3. Load 10-Hz raw Gill/LI-7500 files
  4. Load FM_DOL_1min processed files to obtain T/RH at 6.35 m (if available)
  5. Write two output files per 48-h period:
       FMDOL_10Hz_*.txt  — 10-Hz fast channels only (13 cols)
       FMDOL_1min_*.txt  — native 1-min T/RH at 51.5/30/15/6.35 m (12 cols)

IMPORTANT: Lab Library paths are READ-ONLY. Output is written to
           UTESpac_Python/siteGill<start>_<end>/ only.
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta

try:
    from common import (get_box_path, validate_fast, build_48h_index,
                        timestamp_columns, load_daqm_files)
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import (get_box_path, validate_fast, build_48h_index,
                        timestamp_columns, load_daqm_files)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Python/

box_path = get_box_path()

# READ-ONLY: smart3-00536 GHG archives (.ghg = ZIP format, half-hourly raw 10-Hz data)
# Oct-Nov 2025 archives are in Lab Library under the 20250828_DOL50m_Data download folder.
licor_zipfile_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "20250828_DOL50m_Data", "2025-11-10", "50m_raw")

# Folder to receive unzipped smart3 files (and already contains daqm .log files)
unzipfile_dir = os.path.join(
    box_path, "Diane", "French Meadows", "data", "DOL_unziped",
    "20250901-20251110")

# READ-ONLY: processed 1-min combined data (FM_1min_data_process.py output)
# Contains Temp_6.35 / RH_6.35 from the Campbell CR1000X system.
FM_1min_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "processed", "FM_DOL_1min")

# ── date range to process ─────────────────────────────────────────────────────
# Modify these two lines for each processing run.
start_date = datetime(2025, 10, 7)
end_date   = datetime(2025, 11, 10)

# Output site folder (created automatically)
site_folder_name = f"siteGill{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
FM_processed_dir = os.path.join(ROOT_PY, site_folder_name)
os.makedirs(FM_processed_dir, exist_ok=True)
print(f"Output → {FM_processed_dir}")

# ── unzip smart3-00536 GHG archives (10-Hz raw data) ─────────────────────────
# Required for Oct-Nov: each half-hour is stored as a .ghg file (ZIP format).
# Extracts to unzipfile_dir as YYYY-MM-DDTHHMMSS_smart3-00536.data files.
# from zipfile import ZipFile
# moni       = [10, 11]      # October, November
# start_days = [7,   1]      # Oct data starts 7th; Nov from 1st
# end_days   = [31,  10]      # through Oct 31 and Nov 5 (data ends ~Nov 5 14:00)
# for mi in range(len(moni)):
#     for di in range(start_days[mi] - 1, end_days[mi]):
#         day = 1 + di
#         for hi in range(0, 24):
#             for mi30 in [0, 30]:
#                 timestr = f"{hi:02}{mi30:02}00"
#                 ghg_file = os.path.join(
#                     licor_zipfile_dir,
#                     f"2025-{moni[mi]:02}-{day:02}T{timestr}_smart3-00536.ghg")
#                 if os.path.exists(ghg_file):
#                     with ZipFile(ghg_file, "r") as z:
#                         z.extractall(unzipfile_dir)
#                 else:
#                     print(f"Missing: {os.path.basename(ghg_file)}")

# ── data loaders ──────────────────────────────────────────────────────────────

def load_10hz(hour_files):
    """Load 10-Hz raw Gill/LI-7500 .data files."""
    headers = [
        "DATAH", "Seconds", "Nanoseconds", "Sequence Number",
        "Diagnostic Value", "Diagnostic Value 2", "DS Diagnostic Value",
        "Date", "Time", "CO2 Absorptance", "H2O Absorptance",
        "CO2 (mmol/m^3)", "CO2 (mg/m^3)", "H2O (mmol/m^3)", "H2O (g/m^3)",
        "Temperature (C)", "Pressure (kPa)",
        "---1", "---2", "---3", "---4",
        "Cooler Voltage (V)", "Chopper Cooler Voltage (V)",
        "Vin SmartFlux (V)", "CO2 (umol/mol)", "H2O (mmol/mol)",
        "Dew Point (C)", "CO2 Signal Strength", "H2O Sample",
        "H2O Reference", "CO2 Sample", "CO2 Reference",
        "HIT Power (W)", "Vin HIT (V)", "Vin DSI (V)",
        "U (m/s)", "V (m/s)", "W (m/s)", "T (C)",
        "Anemometer Diagnostics", "CHK",
    ]
    dfs = []
    for f in hour_files:
        try:
            df = pd.read_csv(f, sep=r"\s+", engine="python", header=None,
                             names=headers, skiprows=list(range(8)),
                             encoding="utf-8", encoding_errors="replace")
        except Exception as exc:
            print(f"  Warning: could not read {os.path.basename(f)}: {exc} — skipped.")
            continue
        try:
            ts_temp = df["Date"] + " " + df["Time"]
            ts_fixed = [t.rsplit(":", 1)[0] + "." + t.rsplit(":", 1)[1]
                        for t in ts_temp]
            df["TIMESTAMP"] = pd.to_datetime(ts_fixed)
            df.set_index("TIMESTAMP", inplace=True)
        except Exception as exc:
            print(f"  Warning: timestamp parse failed for {os.path.basename(f)}: {exc} — skipped.")
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=0)


def load_fm1min_range(fm1min_dir, start_dt, end_dt):
    """Load FM_DOL_1min processed files covering start_dt..end_dt.

    Returns a DataFrame with a datetime index, or None if no files found.
    These files contain Temp_6.35 / RH_6.35 from the Campbell CR1000X.
    READ-ONLY access to Lab Library.
    """
    dfs = []
    curr = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while curr < end_dt:
        pattern = os.path.join(fm1min_dir,
                               f"FM_DOL_1min_{curr.strftime('%Y%m%d')}*.txt")
        for fp in sorted(glob.glob(pattern)):
            try:
                df = pd.read_csv(fp, header=0)
                def _make_ts(row):
                    hm  = int(row["HM"])
                    h   = hm // 100;  m = hm % 100
                    sec = int(float(row.get("second", 0)))
                    return (datetime(int(row["year"]), 1, 1)
                            + timedelta(days=int(row["day"]) - 1,
                                        hours=h, minutes=m, seconds=sec))
                df["TIMESTAMP"] = df.apply(_make_ts, axis=1)
                df.set_index("TIMESTAMP", inplace=True)
                dfs.append(df)
            except Exception as exc:
                print(f"  Warning: could not read {fp}: {exc}")
        curr += timedelta(days=1)

    if not dfs:
        return None
    combined = pd.concat(dfs, axis=0)
    return combined[~combined.index.duplicated(keep="first")]


def load_and_align(day_files, hour_files, fm1min_dir, expected_date, hz=10):
    """Load all sources and return grids for fast and slow output files.

    Returns (None, None, None) if timestamp validation fails so the caller
    can skip the period cleanly.

    Returns:
        df_10hz_full   — 10-Hz fast data on the 10-Hz 48-h grid
        df_1min_native — LICOR daqm T/RH on the native 1-min 48-h grid (or None)
        df_6m_native   — 6.35 m T/RH on the native 1-min 48-h grid (or None)
    """
    df_10hz = load_10hz(hour_files)
    df_10hz = df_10hz[~df_10hz.index.duplicated(keep="first")]

    if not validate_fast(df_10hz, expected_date, hz, expected_date.date()):
        return None, None, None

    day_start = df_10hz.index[0].floor("D")

    # 10-Hz grid for fast file
    full_index_10hz = build_48h_index(day_start, hz=hz)
    df_10hz_full    = df_10hz.reindex(full_index_10hz)

    # 1-min grid shared by both slow sources
    full_index_1min = build_48h_index(day_start, hz=1/60)

    # LICOR daqm T/RH at 15/30/51.5 m — optional
    if day_files:
        df_1min = load_daqm_files(day_files)
        df_1min = df_1min[~df_1min.index.duplicated(keep="first")]
        df_1min_native = (df_1min.reindex(full_index_1min)
                          .apply(pd.to_numeric, errors="coerce")
                          .interpolate(method="time", limit=2))
    else:
        df_1min_native = None
        print("  LICOR daqm: no files found — T/RH at 15/30/51.5 m omitted.")

    # 6.35 m T/RH from FM_DOL_1min (Campbell CR1000X) — optional
    end_dt    = day_start + timedelta(days=2)
    df_6m_raw = load_fm1min_range(fm1min_dir, day_start, end_dt)
    if df_6m_raw is not None and "Temp_6.35" in df_6m_raw.columns:
        df_6m_native = (df_6m_raw.reindex(full_index_1min)
                        .apply(pd.to_numeric, errors="coerce")
                        .interpolate(method="time", limit=2))
    else:
        df_6m_native = None
        if df_6m_raw is None:
            print("  6.35 m T/RH: no FM_DOL_1min files found — column omitted.")
        else:
            print("  6.35 m T/RH: Temp_6.35/RH_6.35 not found in FM_DOL_1min — column omitted.")

    return df_10hz_full, df_1min_native, df_6m_native


# ── main processing loop ──────────────────────────────────────────────────────

hz = 10
curr_date = start_date
while curr_date <= end_date - timedelta(days=2):

    # 1-min daqm files: day 0 and day 1 to cover the full 48-h window (day 0 → day 2).
    day_files = []
    for d in [curr_date, curr_date + timedelta(days=1)]:
        pattern = os.path.join(unzipfile_dir,
                               f"{d.strftime('%Y-%m-%d')}-000000-daqm*")
        day_files.extend(glob.glob(pattern))

    # 10-Hz files: 96 half-hour files for 48 hours.
    # Two naming conventions exist across periods:
    #   older: _smart3-00536_raw.data
    #   newer: _smart3-00536.data
    hour_files = []
    for i in range(96):
        h = curr_date + timedelta(minutes=30 * i)
        ts = h.strftime('%Y-%m-%dT%H%M%S')
        for suffix in ("_smart3-00536_raw.data", "_smart3-00536.data"):
            found = glob.glob(os.path.join(unzipfile_dir, ts + suffix))
            hour_files.extend(found)

    if not hour_files:
        print(f"Skipping {curr_date.date()}: missing 10-Hz files.")
        curr_date += timedelta(days=2)
        continue
    if not day_files:
        print(f"  Warning: no LICOR daqm files for {curr_date.date()} — T/RH at 15/30/51.5 m omitted.")

    print(f"Processing {curr_date.date()} …")
    df_10hz_full, df_1min_native, df_6m_native = load_and_align(
        day_files, hour_files, FM_1min_dir, curr_date, hz=hz)

    if df_10hz_full is None:
        curr_date += timedelta(days=2)
        continue

    # shared date strings
    ts0      = df_10hz_full.index[0]
    ts_end   = ts0 + pd.Timedelta(hours=48)
    datestr1 = f"{ts0.year}{ts0.month:02d}{ts0.day:02d}"
    datestr2 = f"{ts_end.year}{ts_end.month:02d}{ts_end.day:02d}"

    # ── assemble 10-Hz output (fast channels only) ────────────────────────────
    df = timestamp_columns(df_10hz_full.index, decimals=1)

    df["Ux_50"]          = df_10hz_full["U (m/s)"].values
    df["Uy_50"]          = df_10hz_full["V (m/s)"].values
    df["Uz_50"]          = df_10hz_full["W (m/s)"].values
    df["T_Sonic_50"]     = df_10hz_full["T (C)"].values
    df["diagnostic_50"]  = df_10hz_full["Anemometer Diagnostics"].values
    df["Pressure_50"]    = df_10hz_full["Pressure (kPa)"].values
    df["LiH2O_50"]       = df_10hz_full["H2O (mmol/m^3)"].values
    df["LiCO2_50"]       = df_10hz_full["CO2 (mmol/m^3)"].values
    df["Li_gas_diag_50"] = df_10hz_full["DS Diagnostic Value"].values

    out_10hz = os.path.join(FM_processed_dir,
                            f"FMDOL_{hz}Hz_{datestr1}000000_{datestr2}000000.txt")
    df.to_csv(out_10hz, header=False, index=False, sep=",")
    print(f"  → {os.path.basename(out_10hz)}")

    # ── assemble 1-min output (only write columns that have data) ────────────
    has_licor = df_1min_native is not None
    has_6m    = df_6m_native   is not None

    if not has_licor and not has_6m:
        print("  1-min file skipped: no T/RH data from LICOR daqm or FM_DOL_1min.")
    else:
        # Use whichever index is available for the time axis
        ref_index = df_1min_native.index if has_licor else df_6m_native.index
        df1 = timestamp_columns(ref_index, decimals=0)

        if has_licor:
            df1["Temp_51.5"] = df_1min_native["TA_1_3_1"].values
            df1["RH_51.5"]   = df_1min_native["RH_1_3_1"].values
            df1["Temp_30"]   = df_1min_native["TA_1_2_1"].values
            df1["RH_30"]     = df_1min_native["RH_1_2_1"].values
            df1["Temp_15"]   = df_1min_native["TA_1_1_1"].values
            df1["RH_15"]     = df_1min_native["RH_1_1_1"].values

        if has_6m:
            df1["Temp_6.35"] = df_6m_native["Temp_6.35"].values
            df1["RH_6.35"]   = df_6m_native["RH_6.35"].values

        out_1min = os.path.join(FM_processed_dir,
                                f"FMDOL_1min_{datestr1}000000_{datestr2}000000.txt")
        df1.to_csv(out_1min, header=False, index=False, sep=",")
        cols = (["Temp/RH_51.5+30+15"] if has_licor else []) + \
               (["Temp/RH_6.35"]       if has_6m    else [])
        print(f"  → {os.path.basename(out_1min)}  ({', '.join(cols)})")
        del df1

    del df
    curr_date += timedelta(days=2)

print("\nDone.")
