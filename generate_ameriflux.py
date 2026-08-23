"""generate_ameriflux.py – Create AmeriFlux BASE half-hourly CSV from UTESpac output
(the utespac-run-2 run files, read as the legacy dict through utespac.run_io).

Format requirements (ameriflux.lbl.gov/half-hourly-hourly-data-upload-format/):
  - ASCII CSV, comma-delimited, period as decimal separator
  - First two columns: TIMESTAMP_START, TIMESTAMP_END (YYYYMMDDHHMM, local standard time)
  - Single header row of variable names, no units row
  - Missing value: -9999
  - File name: <SITE_ID>_HH_<START>_<END>.csv

Site folder discovery:
  All site* sub-directories under ROOT_PY are found automatically.
  Folders are grouped by their alphabetic prefix (e.g. siteIRGA, siteGill), so
  multiple date-period folders of the same type are concatenated in chronological
  order.  New instrument types are picked up without any code changes.

Usage::

    python generate_ameriflux.py
"""

import os
import re
import glob
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── configuration ────────────────────────────────────────────────────────────

SITE_ID = "US-xFM"   # replace with official AmeriFlux site ID when registered

# ROOT_PY is automatically set to the directory containing this script.
ROOT_PY = os.path.dirname(os.path.abspath(__file__))

# !! CONFIGURE THIS PATH for your machine !!
# Absolute path to the folder containing FM_DOL_1min_*.txt files.
ROOT_1MIN = ""

OUT_DIR = os.path.join(ROOT_PY, "ameriflux_output")

PF_TYPE = "GPF"   # "LPF" or "GPF"
AVG_PER = 30      # minutes
MISSING = -9999.0
GAMMA   = 0.0098  # dry adiabatic lapse rate [K/m]
Z_REF   = 4.42    # reference height [m] used in UTESpac for both sites

# T/RH heights and 1-min column names (site-specific, configure as needed)
HMP_HEIGHTS = [6.35, 15.0, 30.0, 51.5]    # V=1..4, ascending
HMP_COLS    = {6.35: ("Temp_6.35", "RH_6.35"),
               15.0: ("Temp_15",   "RH_15"),
               30.0: ("Temp_30",   "RH_30"),
               51.5: ("Temp_51.5", "RH_51.5")}

# Radiation heights and 1-min column names (site-specific, configure as needed)
RAD_HEIGHTS = [4.22, 51.5]                 # V=1..2, ascending
RAD_COLS    = {4.22: ("SWIN_4.22",  "SWOUT_4.22",  "LWIN_4.22",  "LWOUT_4.22",  "Rn_4.22"),
               51.5: ("SWIN_51.5",  "SWOUT_51.5",  "LWIN_51.5",  "LWOUT_51.5",  "Rn_51.5")}

# ── helpers ───────────────────────────────────────────────────────────────────

def matlab_to_dt(serial):
    """Convert MATLAB serial date float → Python datetime, rounded to nearest minute."""
    raw = datetime.fromordinal(int(serial)) + timedelta(days=serial % 1) \
          - timedelta(days=366)
    seconds = raw.second + raw.microsecond / 1e6
    if seconds >= 30:
        raw = raw.replace(second=0, microsecond=0) + timedelta(minutes=1)
    else:
        raw = raw.replace(second=0, microsecond=0)
    return raw

def dt_to_ameriflux(dt):
    return dt.strftime("%Y%m%d%H%M")

def _hstr(h):
    return f"{h:g}"

def get_col(data_arr, header_list, pattern):
    """Return column whose header contains pattern, or NaN array."""
    for i, h in enumerate(header_list):
        if pattern in h:
            return data_arr[:, i].astype(float)
    return np.full(data_arr.shape[0], np.nan)

def site_col(site_data, arr_key, hdr_key, pattern, t_idx_series):
    """Extract a column from site data, aligned to the global t_idx."""
    arr     = site_data[arr_key]
    headers = site_data[hdr_key]
    col     = get_col(arr, headers, pattern)
    valid   = t_idx_series.dropna().astype(int)
    out     = np.full(len(t_idx_series), np.nan)
    for pos_t, pos_s in zip(valid.index, valid.values):
        loc = t_idx_series.index.get_loc(pos_t)
        out[loc] = col[pos_s] if pos_s < len(col) else np.nan
    return out

# ── auto-discover and load site groups ───────────────────────────────────────

def _site_prefix(folder_name):
    """siteIRGA20250901_20251007 → siteIRGA  |  siteGill → siteGill"""
    m = re.match(r'(site[A-Za-z]+)\d', folder_name)
    return m.group(1) if m else folder_name

def load_run_group(folder_list, pf_type):
    """Load and time-sort the 30-min run files (utespac-run-2 netCDF, read as
    the legacy output dict) of one planar-fit type from a list of site folders."""
    from utespac.run_io import read_run_legacy
    file_pairs = []
    for folder in folder_list:
        for fp in glob.glob(os.path.join(folder, "output",
                                          f"*_30minAvg_{pf_type}_LinDet_*.nc")):
            file_pairs.append(fp)
    if not file_pairs:
        return None

    # lexicographic sort = chronological for the YYYY_MM_DD suffix
    parts = [read_run_legacy(fp) for fp in sorted(file_pairs)]

    combined = {}
    for key in parts[0]:
        arrays = [p[key] for p in parts if key in p]
        if not arrays:
            continue
        if isinstance(arrays[0], np.ndarray) and arrays[0].ndim >= 1:
            try:
                combined[key] = np.concatenate(arrays, axis=0)
            except Exception:
                combined[key] = arrays[0]
        else:
            combined[key] = arrays[0]   # headers / scalars — take first
    return combined

def get_sonic_heights(site_data):
    """Return sorted list of sonic heights found in spdAndDirHeader."""
    heights = set()
    for h in site_data.get("spdAndDirHeader", []):
        m = re.match(r'([\d.]+)m direction', h)
        if m:
            heights.add(float(m.group(1)))
    return sorted(heights)

# Find all site* folders that have an output sub-directory
candidate_folders = sorted(
    d for d in os.listdir(ROOT_PY)
    if d.startswith("site")
    and os.path.isdir(os.path.join(ROOT_PY, d))
    and os.path.isdir(os.path.join(ROOT_PY, d, "output"))
)

# Group by site-type prefix
site_groups: dict = defaultdict(list)
for folder in candidate_folders:
    site_groups[_site_prefix(folder)].append(os.path.join(ROOT_PY, folder))

print(f"Discovered {len(site_groups)} site type(s): {sorted(site_groups)}")

# Load data for each group
loaded: dict = {}          # prefix → combined data dict
for prefix, folders in sorted(site_groups.items()):
    print(f"  Loading {prefix} ({len(folders)} folder(s))…")
    data = load_run_group(folders, PF_TYPE)
    if data is not None:
        loaded[prefix] = data
        n = data["H"].shape[0]
        heights = get_sonic_heights(data)
        print(f"    {n} periods, heights: {heights} m")
    else:
        print(f"    WARNING: no {PF_TYPE} run files found — skipping.")

if not loaded:
    raise RuntimeError(f"No {PF_TYPE} run files (*_30minAvg_{PF_TYPE}_LinDet_*.nc) found in any site* folder.")

# ── discover EC heights and map each to its source ───────────────────────────

height_to_src = {}   # float height → site prefix string
for prefix, data in loaded.items():
    for h in get_sonic_heights(data):
        if h not in height_to_src:
            height_to_src[h] = prefix
        # If height appears in multiple sites, the first (alphabetically sorted) wins.
        # Override manually here if needed.

EC_HEIGHTS = sorted(height_to_src)   # V=1 = lowest
print(f"\nEC heights (V=1=lowest): {EC_HEIGHTS}")
for h in EC_HEIGHTS:
    print(f"  V={EC_HEIGHTS.index(h)+1}  {h} m  →  {height_to_src[h]}")

# ── build continuous 30-min timestamp grid ────────────────────────────────────

# Collect all timestamps across all loaded sites
all_times = []
for data in loaded.values():
    all_times.extend([matlab_to_dt(t) for t in data["H"][:, 0]])

t_start = min(all_times)
t_end   = max(all_times)
freq    = f"{AVG_PER}min"
t_idx   = pd.date_range(start=t_start, end=t_end, freq=freq)

# Per-site timestamp index (position in that site's arrays → slot in t_idx)
site_dt_index = {}   # prefix → pd.Series of array positions aligned to t_idx
for prefix, data in loaded.items():
    dts = [matlab_to_dt(t) for t in data["H"][:, 0]]
    site_dt_index[prefix] = pd.Series(np.arange(len(dts)),
                                       index=pd.DatetimeIndex(dts)).reindex(t_idx)

# ── pre-extract PA (kPa) — search all sites for a Pressure column ─────────────

pa_timeseries = np.full(len(t_idx), np.nan)
for prefix, data in loaded.items():
    # Look for a table key that has "Pressure" in its corresponding header
    for key in list(data.keys()):
        if key.endswith("Header") or not isinstance(data[key], np.ndarray):
            continue
        hdr_key = key + "Header"
        hdr = data.get(hdr_key, [])
        hdr_flat = hdr[0] if (hdr and isinstance(hdr[0], list)) else hdr
        pa_col = next((i for i, h in enumerate(hdr_flat) if "Pressure" in str(h)), None)
        if pa_col is not None:
            pa_raw = data[key][:, pa_col].astype(float)
            med = np.nanmedian(pa_raw)
            if med > 50000:
                pa_raw = pa_raw / 1000.0
            elif med > 200:
                pa_raw = pa_raw / 10.0
            dts = [matlab_to_dt(t) for t in data["H"][:, 0]]
            pa_ser = pd.Series(pa_raw, index=pd.DatetimeIndex(dts))
            pa_timeseries = pa_ser.reindex(t_idx).values.astype(float)
            print(f"PA source: {prefix}/{key}  median={np.nanmedian(pa_timeseries):.3g} kPa")
            break
    if not np.all(np.isnan(pa_timeseries)):
        break

# ── build output DataFrame ────────────────────────────────────────────────────

df = pd.DataFrame(index=t_idx)
df["TIMESTAMP_END"]   = [dt_to_ameriflux(t) for t in t_idx]
df["TIMESTAMP_START"] = [dt_to_ameriflux(t - timedelta(minutes=AVG_PER)) for t in t_idx]

# ── turbulent fluxes — one loop per EC height ─────────────────────────────────

for v_idx, height in enumerate(EC_HEIGHTS, start=1):
    hn     = _hstr(height)
    prefix = height_to_src[height]
    src    = loaded[prefix]
    hi     = site_dt_index[prefix]

    def _col(arr_key, hdr_key, pattern):
        return site_col(src, arr_key, hdr_key, pattern, hi)

    # ---- wind sector flag (1 = tower-disturbed, 0 = clean) ----
    wd_flag = _col("spdAndDir", "spdAndDirHeader", f"{hn}m flag")

    # ---- density / thermodynamic terms from specificHum ----
    if "specificHum" in src:
        sh      = src["specificHum"]
        sh_hdr  = src["specificHumHeader"]
        # If this exact height has no HMP, use the nearest height that does
        avail_hum_heights = sorted(set(
            float(re.match(r'([\d.]+) m:', h).group(1))
            for h in sh_hdr if re.match(r'([\d.]+) m: rho_airmoistAvg', h)
        ))
        if avail_hum_heights:
            lookup_h = min(avail_hum_heights, key=lambda x: abs(x - height))
        else:
            lookup_h = height
        lhn     = f"{lookup_h} m: "
        rho_arr = get_col(sh, sh_hdr, f"{lhn}rho_airmoistAvg(kg/m^3)")
        r_arr   = get_col(sh, sh_hdr, f"{lhn}rAvg(g/kg)")
        vt_arr  = get_col(sh, sh_hdr, f"{lhn}virtualThetaAvg(K)")
        rho_col = np.full(len(t_idx), np.nan)
        r_col   = np.full(len(t_idx), np.nan)
        vt_col  = np.full(len(t_idx), np.nan)
        for pos_t, pos_s in zip(hi.dropna().index, hi.dropna().astype(int).values):
            loc = t_idx.get_loc(pos_t)
            if pos_s < len(rho_arr):
                rho_col[loc] = rho_arr[pos_s]
                r_col[loc]   = r_arr[pos_s]
                vt_col[loc]  = vt_arr[pos_s]
        cp_col = 1004.67 * (1.0 + 0.84 * r_col / 1000.0)
        Lv_col = (2.501 - 0.00237 * (vt_col - 273.15)) * 1e6
    else:
        h_arr   = src["H"]
        rho_raw = h_arr[:, 1]
        cp_raw  = h_arr[:, 2]
        lv_raw  = _col("LHflux", "LHfluxHeader", f"{hn}m Lv(J/g)")
        rho_col = np.full(len(t_idx), np.nan)
        cp_col  = np.full(len(t_idx), np.nan)
        Lv_col  = np.full(len(t_idx), np.nan)
        for pos_t, pos_s in zip(hi.dropna().index, hi.dropna().astype(int).values):
            loc = t_idx.get_loc(pos_t)
            if pos_s < len(rho_raw):
                rho_col[loc] = rho_raw[pos_s]
                cp_col[loc]  = cp_raw[pos_s]
                Lv_col[loc]  = lv_raw[loc] * 1000.0
        vt_col = np.full(len(t_idx), np.nan)
        r_col  = np.full(len(t_idx), np.nan)

    # ---- turbulent fluxes ----
    thv_wPF = _col("H",       "Hheader",       f"{hn}m son:Theta_v'wPF'")
    df[f"H_1_{v_idx}_1"] = rho_col * cp_col * thv_wPF

    q_wpl = _col("LHflux", "LHfluxHeader", f"{hn}m wPF'q_WPL'(m/s kg/m3)")
    df[f"LE_1_{v_idx}_1"] = Lv_col * q_wpl

    co2_wpl = _col("CO2flux", "CO2fluxHeader", f"{hn}m: wPF'CO2_WPL'(m/s kg/m^3)")
    df[f"FC_1_{v_idx}_1"] = co2_wpl / 44.01 * 1000.0 * 1e6

    tau_pf = _col("tau", "tauHeader", f"{hn}m :sqrt(uPF'wPF'^2+vPF'wPF'^2)")
    df[f"USTAR_1_{v_idx}_1"] = np.sqrt(np.abs(tau_pf)) * np.sign(tau_pf)
    df[f"TAU_1_{v_idx}_1"]   = rho_col * tau_pf

    df[f"WS_1_{v_idx}_1"] = _col("rotatedSonic", "rotatedSonicHeader", f"{hn}m:u")
    df[f"WD_1_{v_idx}_1"] = _col("spdAndDir",    "spdAndDirHeader",    f"{hn}m direction")

    L_col = _col("L", "Lheader", f"{hn}m L:")
    df[f"MO_LENGTH_1_{v_idx}_1"] = L_col

    su = _col("sigma", "sigmaHeader", f"{hn}m :sigma_uPF")
    sv = _col("sigma", "sigmaHeader", f"{hn}m :sigma_vPF")
    sw = _col("sigma", "sigmaHeader", f"{hn}m :sigma_wPF")
    df[f"TKE_1_{v_idx}_1"]     = 0.5 * (su**2 + sv**2 + sw**2)
    df[f"U_SIGMA_1_{v_idx}_1"] = su
    df[f"V_SIGMA_1_{v_idx}_1"] = sv
    df[f"W_SIGMA_1_{v_idx}_1"] = sw

    df[f"T_SONIC_SIGMA_1_{v_idx}_1"] = _col("sigma", "sigmaHeader", f"{hn}m :sigma_Tson")

    with np.errstate(divide="ignore", invalid="ignore"):
        df[f"ZL_1_{v_idx}_1"] = np.where(np.isfinite(L_col) & (L_col != 0.0),
                                          height / L_col, np.nan)

    df[f"FH2O_1_{v_idx}_1"] = q_wpl * 1e6 / 18.015
    df[f"H2O_1_{v_idx}_1"]  = r_col * 28.97 / 18.015

    # Molar density for gas sigma conversions
    PA_Pa = pa_timeseries * 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_mol = np.where(
            np.isfinite(PA_Pa) & np.isfinite(vt_col) & (vt_col > 200),
            PA_Pa / (8.314 * vt_col), np.nan)

    sig_co2 = _col("sigma", "sigmaHeader", f"{hn}m :sigma_CO2")
    sig_h2o = _col("sigma", "sigmaHeader", f"{hn}m :sigma_H2O")
    with np.errstate(divide="ignore", invalid="ignore"):
        df[f"CO2_SIGMA_1_{v_idx}_1"] = sig_co2 / (44.01 * 1000.0) / rho_mol * 1e6
        df[f"H2O_SIGMA_1_{v_idx}_1"] = sig_h2o / 18.015 / rho_mol * 1e3

    theta_v = _col("derivedT", "derivedTheader", f"{height} m: theta_v_son")
    df[f"T_SONIC_1_{v_idx}_1"] = theta_v - GAMMA * (height - Z_REF)

    df[f"CO2_1_{v_idx}_1"] = _col("CO2flux", "CO2fluxHeader", f"{hn}m: CO2 (ppm)")

    # ---- WD_FILTER: report only, no masking (same treatment as SSITC flags) ----
    df[f"WD_FILTER_1_{v_idx}_1"] = wd_flag.astype(float)

    # ---- SSITC quality flags ----
    for flag_var in ("TAU", "H", "LE", "FC"):
        df[f"{flag_var}_SSITC_TEST_1_{v_idx}_1"] = site_col(
            src, "fluxQC", "fluxQCHeader", f"{hn}m:{flag_var}_SSITC_TEST", hi)

# ── slow meteorology from 1-min data ─────────────────────────────────────────

print("\nLoading 1-min met data…")
min1_files = sorted(glob.glob(os.path.join(ROOT_1MIN, "FM_DOL_1min_*.txt"))) \
             if ROOT_1MIN else []
if min1_files:
    min1_frames = []
    for fp in min1_files:
        try:
            min1_frames.append(pd.read_csv(fp, header=0))
        except Exception:
            pass
    min1_all = pd.concat(min1_frames, ignore_index=True)

    def _1min_dt(row):
        hm = int(row["HM"]); h = hm // 100; m = hm % 100
        try:
            return (datetime(int(row["year"]), 1, 1)
                    + timedelta(days=int(row["day"]) - 1, hours=h, minutes=m,
                                seconds=int(row.get("second", 0))))
        except Exception:
            return pd.NaT

    min1_all["dt"] = min1_all.apply(_1min_dt, axis=1)
    min1_all = min1_all.dropna(subset=["dt"]).set_index("dt").sort_index()
    min1_30  = min1_all.resample(f"{AVG_PER}min", closed="right", label="right").mean()
    min1_30.index = pd.DatetimeIndex(min1_30.index).round(f"{AVG_PER}min")

    def _met(col_name):
        if col_name not in min1_30.columns:
            warnings.warn(f"Column '{col_name}' not found in 1-min data.")
            return np.full(len(t_idx), np.nan)
        return min1_30[col_name].reindex(t_idx).values.astype(float)

    for v_idx, height in enumerate(HMP_HEIGHTS, start=1):
        t_col, rh_col = HMP_COLS[height]
        ta  = _met(t_col);  rh = _met(rh_col)
        df[f"TA_1_{v_idx}_1"]  = ta
        df[f"RH_1_{v_idx}_1"]  = rh
        es  = 6.1078 * np.exp(17.27 * ta / (ta + 237.3))
        df[f"VPD_1_{v_idx}_1"] = np.where(np.isfinite(es), es * (1.0 - rh / 100.0), np.nan)

    for v_idx, height in enumerate(RAD_HEIGHTS, start=1):
        sw_in, sw_out, lw_in, lw_out, rn = RAD_COLS[height]
        df[f"SW_IN_1_{v_idx}_1"]  = _met(sw_in)
        df[f"SW_OUT_1_{v_idx}_1"] = _met(sw_out)
        df[f"LW_IN_1_{v_idx}_1"]  = _met(lw_in)
        df[f"LW_OUT_1_{v_idx}_1"] = _met(lw_out)
        df[f"NETRAD_1_{v_idx}_1"] = _met(rn)

    for v_idx, height in enumerate(RAD_HEIGHTS, start=1):
        sw_in_v  = df[f"SW_IN_1_{v_idx}_1"].values.astype(float)
        sw_out_v = df[f"SW_OUT_1_{v_idx}_1"].values.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"ALB_1_{v_idx}_1"] = np.where(
                (sw_in_v != MISSING) & (sw_out_v != MISSING) & (sw_in_v > 5.0),
                100.0 * sw_out_v / sw_in_v, np.nan)
else:
    warnings.warn("ROOT_1MIN not set or no FM_DOL_1min files found — "
                  "met/radiation columns will be absent.")

# PA always from the UTESpac output (independent of 1-min data)
df["PA_1_1_1"] = pa_timeseries

# ── replace NaN with -9999, enforce column order, write CSV ──────────────────

ts_cols   = ["TIMESTAMP_START", "TIMESTAMP_END"]
data_cols = [c for c in df.columns if c not in ts_cols]
df = df[ts_cols + data_cols].fillna(MISSING)

for col in data_cols:
    vals = df[col].values.astype(float)
    mask = vals != MISSING
    if mask.any():
        vals[mask] = np.round(vals[mask], 6)
    df[col] = vals

os.makedirs(OUT_DIR, exist_ok=True)
t_file_start = df["TIMESTAMP_START"].iloc[0]
t_file_end   = df["TIMESTAMP_END"].iloc[-1]
fname    = f"{SITE_ID}_HH_{t_file_start}_{t_file_end}.csv"
out_path = os.path.join(OUT_DIR, fname)
df.to_csv(out_path, index=False)

print(f"\nWrote {len(df)} half-hourly rows × {len(df.columns)} columns")
print(f"Output: {out_path}")

print("\nVariable coverage (non-missing rows / total):")
total = len(df)
for col in data_cols:
    n_valid = (df[col].values != MISSING).sum()
    if n_valid < total:
        pct = 100 * n_valid / total
        print(f"  {col:40s}  {n_valid:4d}/{total}  ({pct:.0f}%)")
print("  (columns with 100% coverage not shown)")
