"""generate_ameriflux_hf.py – AmeriFlux high-frequency CSV export from the UTESpac HF netCDF (*_hf_*.nc).

Format references:
  https://ameriflux.lbl.gov/data/how-to-upload-data/uploading-high-frequency-data/
  https://ameriflux.lbl.gov/half-hourly-hourly-data-upload-format/

Output structure:
  One CSV per 30-min period: <SITE_ID>_HF_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>.csv
  Flat zip archive (no subdirectories) per site type — AmeriFlux HF standard.
  Individual CSVs are removed after zipping.

Variable convention (V = 1 = lowest height, AmeriFlux positional convention):
  TIMESTAMP           – YYYYMMDDHHMMSS.cc  (local standard time, no DST; cc = centiseconds)
  U_1_{V}_1           – planar-fit u wind component          [m s⁻¹]  standard unit
  V_1_{V}_1           – planar-fit v wind component          [m s⁻¹]  (~0 after PF rotation)
  W_1_{V}_1           – planar-fit w wind component          [m s⁻¹]  (~0 after PF rotation)
  T_SONIC_1_{V}_1     – sonic temperature                    [°C]     standard unit
  H2O_IU_1_{V}_1      – water vapour density                 [g m⁻³]  instrument unit (_IU)
  CO2_IU_1_{V}_1      – CO2 density                         [mg m⁻³] instrument unit (_IU)
  PA_1_1_1            – atmospheric pressure (one barometer) [kPa]    standard unit

_IU qualifier (AmeriFlux):
  Applied to H2O and CO2 because they are submitted in density units (g m⁻³, mg m⁻³)
  rather than the AmeriFlux standard mole-fraction units (mmol mol⁻¹, µmol mol⁻¹).
  Per AmeriFlux documentation, use of _IU in HF uploads should be discussed with the
  data team (ameriflux-support@lbl.gov) before submission.

Unit notes — HF netCDF convention (utespac.export_hf):
  rhov:    g m⁻³   (vapour density)
  rhoCO2:  mg m⁻³

Missing data: -9999  (NaN and ±Inf replaced before writing)
"""

import os
import re
import glob
import sys
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utespac.campbell_date import datetime64_to_matlab_datenum   # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────

SITE_ID = "US-xFM"    # replace with official AmeriFlux site ID when registered

ROOT_PY = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_PY, "ameriflux_hf_output")

PF_TYPE = "GPF"       # "LPF" or "GPF"
MISSING = -9999.0

MATLAB_EPOCH = 719529.0 # MATLAB days to Unix epoch (1970-01-01)

# ── helpers ────────────────────────────────────────────────────────────────────

def matlab_to_unix(t_arr):
    return (np.asarray(t_arr) - MATLAB_EPOCH) * 86400.0


def make_timestamps(t_arr):
    """Vectorised MATLAB serial → 'YYYYMMDDHHMMSS.cc' pandas Series."""
    unix_s = matlab_to_unix(t_arr)
    dti    = pd.to_datetime(unix_s, unit="s")
    # compute centiseconds from float — pd.microsecond has rounding error in pandas 3.x
    cs     = pd.Index(np.round((unix_s % 1.0) * 100).astype(int)).astype(str).str.zfill(2)
    return dti.strftime("%Y%m%d%H%M%S") + "." + cs


def file_ts(t_val, offset_s=0.0):
    """Single MATLAB serial → YYYYMMDDHHMM for file naming."""
    return pd.Timestamp(matlab_to_unix(t_val) + offset_s, unit="s").strftime("%Y%m%d%H%M")


def safe(arr):
    """Replace NaN / ±Inf with MISSING (-9999)."""
    return np.where(np.isfinite(arr), arr, MISSING)


# ── main ───────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

hf_files = sorted(glob.glob(
    os.path.join(ROOT_PY, "site*", "output", f"*_hf_{PF_TYPE}_*.nc")))

if not hf_files:
    print(f"No {PF_TYPE} HF netCDF files (*_hf_{PF_TYPE}_*.nc) found under {ROOT_PY}/site*/output/")
    raise SystemExit(1)

print(f"Found {len(hf_files)} {PF_TYPE} HF netCDF file(s).")

site_csvs = defaultdict(list)

for hf_path in hf_files:
    site_dir  = os.path.basename(os.path.dirname(os.path.dirname(hf_path)))
    site_type = re.match(r"(site[A-Za-z]+)\d", site_dir)
    site_type = site_type.group(1) if site_type else site_dir

    print(f"\nProcessing {os.path.basename(hf_path)}  [{site_dir}] …")

    with xr.open_dataset(hf_path, engine="netcdf4") as ds:
        ds = ds.load()

    t     = datetime64_to_matlab_datenum(ds["time"].values)
    z     = ds["height"].values                      # ascending in the HF file
    N     = ds.sizes["time"]

    def _var(name, dim="height"):
        """(N, n_<dim>) array of *name*, NaN when the file lacks it."""
        if name not in ds:
            return np.full((N, ds.sizes.get(dim, len(z))), np.nan)
        return ds[name].values

    uPF   = _var("u")
    vPF   = _var("v")
    wPF   = _var("w")
    Ts_C  = _var("Ts")                               # °C
    P_kPa = ds["P"].values if "P" in ds else np.full(N, np.nan)   # kPa, on the HF time axis
    rhov  = _var("rhov", "height_rhov")              # g m⁻³
    rCO2  = _var("rhoCO2", "height_rhoCO2")          # mg m⁻³

    n_sonics = len(z)
    N        = len(t)

    dt_s  = float(np.nanmedian(np.diff(t[:min(2000, N)]))) * 86400.0
    hz    = int(round(1.0 / dt_s))
    block = hz * 30 * 60
    n_blocks = N // block

    print(f"  {hz} Hz  ·  {n_sonics} sonic(s) at {list(z)} m  ·  {n_blocks} × 30-min files")

    sorted_z = sorted(z)
    v_of     = {zi: sorted_z.index(zi) + 1 for zi in z}

    # ── prepare output arrays (no unit conversion needed for _IU variables) ──
    u_out   = safe(uPF)
    v_out   = safe(vPF)
    w_out   = safe(wPF)
    ts_out  = safe(Ts_C)        # °C  (standard)
    pa_out  = safe(P_kPa)       # kPa (standard)
    h2o_out = safe(rhov)        # g m⁻³   (_IU)
    co2_out = safe(rCO2)        # mg m⁻³  (_IU)

    # ── build column names ────────────────────────────────────────────────────
    col_names = ["TIMESTAMP"]
    for zi in z:
        v = v_of[zi]
        col_names += [
            f"U_1_{v}_1", f"V_1_{v}_1", f"W_1_{v}_1",
            f"T_SONIC_1_{v}_1",
            f"H2O_IU_1_{v}_1",
            f"CO2_IU_1_{v}_1",
        ]
    col_names.append("PA_1_1_1")

    # ── write one CSV per 30-min block ────────────────────────────────────────
    for bi in range(n_blocks):
        s0, s1 = bi * block, (bi + 1) * block
        t_blk  = t[s0:s1]

        t_start = file_ts(t_blk[0])
        t_end   = file_ts(t_blk[-1], offset_s=1.0 / hz)

        data = {"TIMESTAMP": make_timestamps(t_blk)}
        for si, zi in enumerate(z):
            v = v_of[zi]
            data[f"U_1_{v}_1"]         = u_out[s0:s1, si]
            data[f"V_1_{v}_1"]         = v_out[s0:s1, si]
            data[f"W_1_{v}_1"]         = w_out[s0:s1, si]
            data[f"T_SONIC_1_{v}_1"]   = ts_out[s0:s1, si]
            data[f"H2O_IU_1_{v}_1"]    = h2o_out[s0:s1, si]
            data[f"CO2_IU_1_{v}_1"]    = co2_out[s0:s1, si]
        data["PA_1_1_1"] = pa_out[s0:s1]

        df = pd.DataFrame(data, columns=col_names)

        fname    = f"{SITE_ID}_HF_{t_start}_{t_end}.csv"
        out_path = os.path.join(OUT_DIR, fname)
        df.to_csv(out_path, index=False, float_format="%.4f", lineterminator="\n")
        site_csvs[site_type].append(out_path)

    print(f"  Wrote {n_blocks} CSV files.")

# ── zip by site type (flat — no subdirectories) ───────────────────────────────
print("\nZipping …")
for site_type, csv_list in site_csvs.items():
    zip_name = f"{SITE_ID}_{site_type}_HF.zip"
    zip_path = os.path.join(OUT_DIR, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(csv_list):
            zf.write(csv_path, os.path.basename(csv_path))  # flat — no subdirs
    for csv_path in csv_list:
        os.remove(csv_path)
    print(f"  → {zip_name}  ({len(csv_list)} files)")

print("\nDone.")
