"""FM_FastNSlow_4IRGA_process.py
Process CR1000X IRGASON (20-Hz fast + 1-min slow) and LICOR daqm (1-min)
data into 48-h UTESpac-ready files for siteIRGA.

Steps:
  1. Load CR1000X 1-min slow files (T/RH at 6.35 m via EE181)
  2. Load CR1000X 20-Hz fast IRGASON files (4 sonic/IRGA systems)
  3. Load LICOR daqm 1-min stats files (T/RH at 15/30 m)
  4. Write formatted UTESpac input files to the target siteIRGA folder

IMPORTANT: Lab Library paths are READ-ONLY. Output is written to
           UTESpac_Python/siteIRGA<start>_<end>/ only.

CR1000X file-index reference:
  20250718_tall  HZ=10  startid=600  endid=627
  20250828_tall  HZ=20  startid=637  endid=674
  20251110_tall  HZ=20  startid=681  endid=717  (681 = 2025-09-01, confirmed)
  20251110_tall  HZ=20  startid=717  endid=753  (717 = 2025-10-07, confirmed)
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta

try:
    from common import (get_box_path, read_toa5, validate_fast, build_48h_index,
                        timestamp_columns, load_daqm_files, load_cr_files)
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import (get_box_path, read_toa5, validate_fast, build_48h_index,
                        timestamp_columns, load_daqm_files, load_cr_files)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT_PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # UTESpac_Python/

box_path = get_box_path()

# READ-ONLY: CR1000X ascii data (converted from binary with PC400)
cr1000xFM_dir = os.path.join(
    box_path, "Lab Library", "French Meadows", "Summer2025",
    "data", "20251110_tall", "ascii")

# READ-ONLY: LICOR daqm unzipped 1-min stats files
unzipfile_dir = os.path.join(
    box_path, "Diane", "French Meadows", "data", "DOL_unziped",
    "20250901-20251110")

# ── date range / file indices ─────────────────────────────────────────────────
# Modify these for each processing run.
startid        = 717   # CR1000X file index for start date (2025-10-07)
endid          = 753   # CR1000X file index for end date   (2025-11-10, exclusive)
HZ             = 20
decimal_places = 2     # 20 Hz → 0.05 s resolution

start_date = datetime(2025, 10, 7)
end_date   = datetime(2025, 11, 10)

# Output site folder (created automatically)
site_folder_name = f"siteIRGA{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
FM_processed_dir = os.path.join(ROOT_PY, site_folder_name)
os.makedirs(FM_processed_dir, exist_ok=True)
print(f"Output → {FM_processed_dir}")

# ── validate startid matches start_date (uses slow file — fast not loaded in 1-min mode) ──
_start_file = os.path.join(cr1000xFM_dir, f"TOA5_6653slow_avg_data{startid}.dat")
if not os.path.exists(_start_file):
    raise FileNotFoundError(f"Start slow file not found: {_start_file}")
_df_check = read_toa5(_start_file, nrows=1, parse_dates=False)
_first_ts = pd.to_datetime(_df_check.index[0], format="mixed").date()
if _first_ts != start_date.date():
    raise ValueError(
        f"startid={startid} mismatch: slow file starts on {_first_ts}, "
        f"expected {start_date.date()}. Update startid.")
print(f"startid={startid} validated: starts on {_first_ts}")

# ── main processing loop ──────────────────────────────────────────────────────

curr_date = start_date
fi        = startid

while fi < endid:

    # ── collect initial file lists for this 48-h period ─────────────────────
    fast_files = [os.path.join(cr1000xFM_dir, f"TOA5_6653fast_20hz_data{fi + k}.dat")
                  for k in range(2)]
    slow_files = [os.path.join(cr1000xFM_dir, f"TOA5_6653slow_avg_data{fi + k}.dat")
                  for k in range(2)]

    if not all(os.path.exists(f) for f in fast_files + slow_files):
        print(f"Skipping fi={fi} ({curr_date.date()}): missing CR1000X files.")
        fi += 2
        curr_date += timedelta(days=2)
        continue

    print(f"Processing fi={fi}  ({curr_date.date()}) …")

    # ── load CR1000X fast ────────────────────────────────────────────────────
    cr1000x_df_fast = load_cr_files(fast_files)

    # ── PRIMARY split-day detection: fast coverage drives file extension ──────
    # Power outages can split a 48-h period across 3+ files.  Extend greedily
    # (up to 4 extra) until fast coverage ≥ 80 % or the next file is outside
    # the 48-h window.  Matching slow files are appended in parallel.
    expected_fast_rows = HZ * 48 * 3600
    expected_slow_rows = 48 * 60        # 2 880 rows at 1/min
    day_start          = cr1000x_df_fast.index[0].floor("D")
    n_consumed         = 2

    while len(cr1000x_df_fast) < 0.80 * expected_fast_rows and n_consumed < 6:
        next_k    = fi + n_consumed
        next_fast = os.path.join(cr1000xFM_dir, f"TOA5_6653fast_20hz_data{next_k}.dat")
        next_slow = os.path.join(cr1000xFM_dir, f"TOA5_6653slow_avg_data{next_k}.dat")
        if not os.path.exists(next_fast):
            break
        try:
            peek    = read_toa5(next_fast, nrows=1, parse_dates=False)
            next_ts = pd.to_datetime(peek.index[0], format="mixed")
            if (next_ts - day_start).total_seconds() > 48 * 3600:
                break
        except Exception:
            break
        print(f"  Appending fast fi={next_k} (split-day recovery).")
        extra = read_toa5(next_fast)
        extra.index = pd.to_datetime(extra.index, format="mixed")
        cr1000x_df_fast = pd.concat([cr1000x_df_fast, extra])
        cr1000x_df_fast = cr1000x_df_fast[~cr1000x_df_fast.index.duplicated(keep="first")]
        fast_files.append(next_fast)
        slow_files.append(next_slow)   # keep slow list in sync
        n_consumed += 1

    # ── load CR1000X slow (list already extended by fast detection above) ────
    # Fast and slow files are created simultaneously by the same logger, so
    # slow_files already contains all needed files after the fast split-day loop.
    cr1000x_df_slow = load_cr_files(slow_files)

    print(f"  Fast: {len(cr1000x_df_fast):,} rows ({len(cr1000x_df_fast)/expected_fast_rows:.0%})  "
          f"Slow: {len(cr1000x_df_slow):,} rows ({len(cr1000x_df_slow)/expected_slow_rows:.0%})")

    if not validate_fast(cr1000x_df_fast, curr_date, HZ, f"fi={fi}"):
        fi += n_consumed
        curr_date += timedelta(days=2)
        continue

    # ── build 48-h master index (20-Hz grid + native 1-min grid) ─────────────
    full_time_index      = build_48h_index(day_start, hz=HZ)
    cr1000x_df_full_fast = cr1000x_df_fast.reindex(full_time_index)
    full_time_index_1min = build_48h_index(day_start, hz=1 / 60)

    # ── LICOR daqm 1-min (T/RH at 15/30 m) ──────────────────────────────────
    # Load day 0 and day 1 daqm files to cover the full 48-h window (day 0 → day 2).
    day_files = []
    for d in [curr_date, curr_date + timedelta(days=1)]:
        pattern = os.path.join(unzipfile_dir,
                               f"{d.strftime('%Y-%m-%d')}-000000-daqm*")
        day_files.extend(glob.glob(pattern))

    if day_files:
        df_1min_licor = load_1min_files(day_files)
        df_1min_licor = df_1min_licor[~df_1min_licor.index.duplicated(keep="first")]
        df_1min_licor_native = (df_1min_licor.reindex(full_time_index_1min)
                                .apply(pd.to_numeric, errors="coerce")
                                .interpolate(method="time", limit=2))
        has_licor = True
    else:
        print(f"  Warning: no LICOR daqm files found for {curr_date.date()} — T/RH at 15/30 m omitted.")
        has_licor = False

    # ── date strings (used for both output filenames) ─────────────────────────
    datestr1 = f"{day_start.year}{day_start.month:02d}{day_start.day:02d}"
    datestr2 = f"{(day_start + pd.Timedelta(hours=48)).year}" \
               f"{(day_start + pd.Timedelta(hours=48)).month:02d}" \
               f"{(day_start + pd.Timedelta(hours=48)).day:02d}"

    # ── assemble 20-Hz output table (48 columns) ─────────────────────────────
    timestrings = full_time_index
    df = pd.DataFrame()
    df["year"]   = [int(t.year)        for t in timestrings]
    df["day"]    = [int(t.day_of_year) for t in timestrings]
    df["HM"]     = [int(f"{t.hour:02d}{t.minute:02d}") for t in timestrings]
    df["second"] = [f"{t.second + t.microsecond / 1e6:.{decimal_places}f}" for t in timestrings]

    # IRGASON at 32.18 m
    df["Ux_32.18"]         = cr1000x_df_full_fast["Ux_2"].values
    df["Uy_32.18"]         = cr1000x_df_full_fast["Uy_2"].values
    df["Uz_32.18"]         = cr1000x_df_full_fast["Uz_2"].values
    df["T_Sonic_32.18"]    = cr1000x_df_full_fast["Ts_2"].values
    df["diagnostic_32.18"] = cr1000x_df_full_fast["diag_sonic_2"].values
    df["Pressure_32.18"]   = cr1000x_df_full_fast["cell_press_2"].values
    df["H2O_32.18"]        = cr1000x_df_full_fast["H2O_2"].values
    df["H2Osig_32.18"]     = cr1000x_df_full_fast["H2O_sig_strgth_2"].values
    df["CO2_32.18"]        = cr1000x_df_full_fast["CO2_2"].values
    df["CO2sig_32.18"]     = cr1000x_df_full_fast["CO2_sig_strgth_2"].values
    df["gas_diag_32.18"]   = cr1000x_df_full_fast["diag_irga_2"].values

    # IRGASON at 13.94 m
    df["Ux_13.94"]         = cr1000x_df_full_fast["Ux_3"].values
    df["Uy_13.94"]         = cr1000x_df_full_fast["Uy_3"].values
    df["Uz_13.94"]         = cr1000x_df_full_fast["Uz_3"].values
    df["T_Sonic_13.94"]    = cr1000x_df_full_fast["Ts_3"].values
    df["diagnostic_13.94"] = cr1000x_df_full_fast["diag_sonic_3"].values
    df["Pressure_13.94"]   = cr1000x_df_full_fast["cell_press_3"].values
    df["H2O_13.94"]        = cr1000x_df_full_fast["H2O_3"].values
    df["H2Osig_13.94"]     = cr1000x_df_full_fast["H2O_sig_strgth_3"].values
    df["CO2_13.94"]        = cr1000x_df_full_fast["CO2_3"].values
    df["CO2sig_13.94"]     = cr1000x_df_full_fast["CO2_sig_strgth_3"].values
    df["gas_diag_13.94"]   = cr1000x_df_full_fast["diag_irga_3"].values

    # IRGASON at 6.35 m
    df["Ux_6.35"]          = cr1000x_df_full_fast["Ux_1"].values
    df["Uy_6.35"]          = cr1000x_df_full_fast["Uy_1"].values
    df["Uz_6.35"]          = cr1000x_df_full_fast["Uz_1"].values
    df["T_Sonic_6.35"]     = cr1000x_df_full_fast["Ts_1"].values
    df["diagnostic_6.35"]  = cr1000x_df_full_fast["diag_sonic_1"].values
    df["Pressure_6.35"]    = cr1000x_df_full_fast["cell_press_1"].values
    df["H2O_6.35"]         = cr1000x_df_full_fast["H2O_1"].values
    df["H2Osig_6.35"]      = cr1000x_df_full_fast["H2O_sig_strgth_1"].values
    df["CO2_6.35"]         = cr1000x_df_full_fast["CO2_1"].values
    df["CO2sig_6.35"]      = cr1000x_df_full_fast["CO2_sig_strgth_1"].values
    df["gas_diag_6.35"]    = cr1000x_df_full_fast["diag_irga_1"].values

    # IRGASON at 4.42 m
    df["Ux_4.42"]          = cr1000x_df_full_fast["Ux_4"].values
    df["Uy_4.42"]          = cr1000x_df_full_fast["Uy_4"].values
    df["Uz_4.42"]          = cr1000x_df_full_fast["Uz_4"].values
    df["T_Sonic_4.42"]     = cr1000x_df_full_fast["Ts_4"].values
    df["diagnostic_4.42"]  = cr1000x_df_full_fast["diag_sonic_4"].values
    df["Pressure_4.42"]    = cr1000x_df_full_fast["cell_press_4"].values
    df["H2O_4.42"]         = cr1000x_df_full_fast["H2O_4"].values
    df["H2Osig_4.42"]      = cr1000x_df_full_fast["H2O_sig_strgth_4"].values
    df["CO2_4.42"]         = cr1000x_df_full_fast["CO2_4"].values
    df["CO2sig_4.42"]      = cr1000x_df_full_fast["CO2_sig_strgth_4"].values
    df["gas_diag_4.42"]    = cr1000x_df_full_fast["diag_irga_4"].values

    out_path = os.path.join(FM_processed_dir,
                            f"FMDOL_{HZ[0]}Hz_{datestr1}000000_{datestr2}000000.txt")
    df.to_csv(out_path, header=False, index=False, sep=",")
    print(f"  → {os.path.basename(out_path)}")

    # ── assemble 1-min output table ───────────────────────────────────────────
    # Reindex slow CR1000X to native 1-min grid (interpolate only to fill
    # minor timestamp jitter; data is already 1-min resolution).
    # limit=2 caps gap-filling at 2 consecutive missing minutes so that long
    # outages remain NaN rather than being bridged by a straight line.
    cr1000x_df_1min = (cr1000x_df_slow.reindex(full_time_index_1min)
                       .apply(pd.to_numeric, errors="coerce")
                       .interpolate(method="time", limit=2))

    # ── assemble 1-min output (only write columns that have data) ────────────
    has_6m = ("AirTC_ee181_1_Avg" in cr1000x_df_1min.columns and
              "RH_ee181_1_Avg"    in cr1000x_df_1min.columns)

    if not has_6m and not has_licor:
        print("  1-min file skipped: no T/RH data from EE181 or LICOR daqm.")
    else:
        ts1 = full_time_index_1min
        df1 = pd.DataFrame()
        df1["year"]   = [int(t.year)        for t in ts1]
        df1["day"]    = [int(t.day_of_year) for t in ts1]
        df1["HM"]     = [int(f"{t.hour:02d}{t.minute:02d}") for t in ts1]
        df1["second"] = [f"{t.second + t.microsecond / 1e6:.0f}" for t in ts1]

        if has_6m:
            df1["Temp_6.35"] = cr1000x_df_1min["AirTC_ee181_1_Avg"].values
            df1["RH_6.35"]   = cr1000x_df_1min["RH_ee181_1_Avg"].values

        if has_licor:
            df1["Temp_32.18"] = df_1min_licor_native["TA_1_2_1"].values
            df1["RH_32.18"]   = df_1min_licor_native["RH_1_2_1"].values
            df1["Temp_13.94"] = df_1min_licor_native["TA_1_1_1"].values
            df1["RH_13.94"]   = df_1min_licor_native["RH_1_1_1"].values

        out_path_1min = os.path.join(FM_processed_dir,
                                     f"FMDOL_1min_{datestr1}000000_{datestr2}000000.txt")
        df1.to_csv(out_path_1min, header=False, index=False, sep=",")
        cols = ([f"Temp/RH_6.35"] if has_6m else []) + \
               ([f"Temp/RH_32.18+13.94"] if has_licor else [])
        print(f"  → {os.path.basename(out_path_1min)}  ({', '.join(cols)})")
        del df1

    del cr1000x_df_full_fast, cr1000x_df_1min, df
    # Advance by at least 2 so the next period always reads a minimum of 2
    # fresh files (each raw file stores at most 1 day of data).
    fi        += max(2, n_consumed)
    curr_date += timedelta(days=2)

print("\nDone.")
