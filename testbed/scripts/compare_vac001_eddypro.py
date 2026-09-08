"""Compare UTESpac VAC001 output against the EddyPro reference run.

Reads the UTESpac 30-min run files (utespac-run-3 netCDF) in
data/VAC001/output (produced by run_vac001.py) and the EddyPro full-output CSV in data/VAC001/eddypro,
joins on period-end timestamp, and prints agreement statistics for H, LE,
Fc, u*, L, wind, sigma_w, and the tilt angles. Optionally writes a
comparison table to testbed/scratch/.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/compare_vac001_eddypro.py [--pf LPF|GPF] [--csv]

Conventions that matter for the join and units:
  * UTESpac timestamps are the last sample of each window (period end);
    EddyPro ``datetime`` is also period end. Both are rounded to the minute.
  * The UTESpac heat flux is compared kinematically: rho and cp are columns
    1-2 of sensible_heat;
    H [W/m2] = rho * cp * w'T'. Two variants are compared: w'Ts' (sonic,
    unrotated w) and w'theta_v' (planar-fit w), the latter being what
    the WPL path uses. EddyPro ``H`` is the humidity-corrected sensible
    heat flux; ``un_H`` is before spectral correction.
  * UTESpac Fc_wpl_pf is stored in kg m-2 s-1 despite the legacy header's
    "(mol/m^2s)" label; converted to umol m-2 s-1 here.
  * EddyPro q' (H2O) second moments are treated as suspect per the
    2026-08-22 ruling; LE is reported but H/u*/L anchor the judgement.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utespac.run_io import read_run_legacy, run_files   # noqa: E402
from utespac.testkit import get_header                   # noqa: E402

SITE_DIR = os.path.join(ROOT, "data", "VAC001")
EDDYPRO_GLOB = os.path.join(SITE_DIR, "eddypro", "*", "VAC_001_*_full.csv")
MATLAB_EPOCH_DAYS = 719529.0
M_CO2 = 44.01e-3   # kg/mol


def datenum_to_datetime(t):
    return pd.to_datetime((np.asarray(t, float) - MATLAB_EPOCH_DAYS) * 86400.0,
                          unit="s").round("min")


def col(out, field, label):
    """Column of out[field] whose header equals (or contains) *label*."""
    hdr = get_header(out, field)
    if hdr is None:
        raise KeyError(f"no header for {field}")
    for i, h in enumerate(hdr):
        if h == label:
            return out[field][:, i]
    for i, h in enumerate(hdr):
        if label in h:
            return out[field][:, i]
    raise KeyError(f"{label!r} not in {field} header: {hdr}")


def _frame_from_output(out):
    """Named per-period quantities from one run file (legacy dict), matched by header.

    Loading file by file (instead of utespac.get_data) matters here: when a
    sensor is dead for a whole file its all-NaN columns are trimmed, the
    matrix shape changes, and get_data's MATLAB-style shape check replaces
    the entire block for that file with NaN. Header-label lookup sidesteps
    that.
    """
    heights = sorted({float(h.rsplit("_", 1)[1]) for h in get_header(out, "sensible_heat")[3:]
                      if h and h.rsplit("_", 1)[-1][:1].isdigit()})
    z = heights[0]
    hn = f"{z:g}"
    t = datenum_to_datetime(out["sensible_heat"][:, 0])
    rho, cp = out["sensible_heat"][:, 1], out["sensible_heat"][:, 2]
    df = pd.DataFrame(index=t)

    def opt(field, label, scale=1.0):
        try:
            return col(out, field, label) * scale
        except KeyError:
            return np.full(len(t), np.nan)

    df["w_theta_v_cov_pf"]     = opt("sensible_heat", f"w_theta_v_cov_pf_{hn}")   # kinematic, K m/s
    df["H_ts_raw"]     = rho * cp * opt("sensible_heat", f"w_ts_cov_raw_{hn}")
    df["H_buoyancy_pf"]  = rho * cp * df["w_theta_v_cov_pf"]
    df["H_pf"] = rho * cp * opt("sensible_heat", f"w_t_air_cov_pf_{hn}")
    df["H_fw_pf"]   = rho * cp * opt("sensible_heat", f"w_t_fw_cov_pf_{hn}")
    df["ustar"]  = opt("scaling", f"ustar_pf_{hn}")
    if not np.isfinite(df["ustar"]).any():     # utespac-run-2 file: ustar_pf not stored yet
        df["ustar"] = np.sqrt(np.abs(opt("momentum", f"Tau_pf_{hn}")))
    df["L"]      = opt("obukhov", f"L_{hn}")
    df["zeta"]   = z / df["L"]
    df["LE_wpl_pf"] = opt("latent_heat", f"LE_wpl_pf_{hn}")
    df["Fc_wpl_pf"] = (opt("co2_flux", f"Fc_wpl_pf_{hn}") / M_CO2 * 1e6
                    if "co2_flux" in out else np.nan)
    df["sigma_w"] = opt("sigma", f"w_sigma_pf_{hn}")
    df["TKE"]     = opt("tke", f"TKE_{hn}")
    df["WS"]      = opt("spdAndDir", f"wind_speed_{hn}")
    df["WD"]      = opt("spdAndDir", f"wind_dir_{hn}")
    return df, z


def load_utespac(pf, det="LinDet"):
    files = run_files(os.path.join(ROOT, "data"), "VAC001", avg_per=30, qualifier=f"{pf}_{det}")
    if not files:
        raise FileNotFoundError(
            f"no run files *_30minAvg_{pf}_{det}_*.nc under {SITE_DIR}/output")
    frames, z, last = [], None, None
    for f in files:
        last = read_run_legacy(f)
        df, z = _frame_from_output(last)
        frames.append(df)
    ut = pd.concat(frames).sort_index()
    ut = ut[~ut.index.duplicated(keep="last")]
    return ut, z, last


def load_eddypro():
    files = sorted(glob.glob(EDDYPRO_GLOB))
    if not files:
        raise FileNotFoundError(EDDYPRO_GLOB)
    ep = pd.read_csv(files[-1], na_values=["-9999", "-9999.0"])
    ep.index = pd.DatetimeIndex(pd.to_datetime(ep["datetime"])).round("min")
    keep = {"H": "H", "un_H": "un_H", "LE": "LE", "un_LE": "un_LE", "LE_scf": "LE_scf",
            "co2_flux": "Fc", "un_co2_flux": "un_Fc", "co2_scf": "Fc_scf", "u*": "ustar",
            "L": "L", "(z-d)/L": "zeta", "wind_speed": "WS", "wind_dir": "WD", "pitch": "pitch",
            "roll": "roll", "w_var": "w_var", "ts_var": "ts_var",
            "w/ts_cov": "wts_cov", "TKE": "TKE", "qc_H": "qc_H", "qc_LE": "qc_LE",
            "qc_co2_flux": "qc_Fc", "qc_Tau": "qc_Tau", "sonic_temperature": "Ts",
            "air_temperature": "Ta"}
    out = ep[[k for k in keep if k in ep.columns]].rename(columns=keep)
    out["sigma_w"] = np.sqrt(out["w_var"])
    return out, files[-1]


def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[m], np.asarray(b)[m]
    if len(a) < 3:
        return dict(N=len(a), bias=np.nan, rmse=np.nan, slope=np.nan,
                    intercept=np.nan, r2=np.nan)
    slope, intercept = np.polyfit(b, a, 1)
    r2 = np.corrcoef(a, b)[0, 1] ** 2
    return dict(N=len(a), bias=float(np.mean(a - b)),
                rmse=float(np.sqrt(np.mean((a - b) ** 2))),
                slope=float(slope), intercept=float(intercept), r2=float(r2))


PAIRS = [
    # (utespac column, eddypro column, label)
    ("H_ts_raw",      "H",     "H: w'Ts' (unrotated)    vs EddyPro H"),
    ("H_buoyancy_pf", "H",     "H: w'theta_v' (pf)      vs EddyPro H"),
    ("H_pf",          "H",     "H: w'T_air' (pf)        vs EddyPro H"),
    ("H_fw_pf",       "H",     "H: w'T_fw' (pf)         vs EddyPro H"),
    ("H_buoyancy_pf", "un_H",  "H: w'theta_v' (pf)      vs EddyPro un_H (no spectral corr)"),
    ("w_theta_v_cov_pf", "wts_cov", "w'theta_v' kinematic    vs EddyPro w/ts_cov (rotated, raw)"),
    ("ustar",      "ustar", "u*"),
    ("L",          "L",     "L (Obukhov)"),
    ("zeta",       "zeta",  "z/L clipped to [-2,2]"),
    ("LE_wpl_pf",     "LE",    "LE (WPL, pf)            vs EddyPro LE  [q' caveat]"),
    ("LE_wpl_pf",     "un_LE", "LE (WPL, pf)            vs EddyPro un_LE (pre-spectral, pre-WPL)"),
    ("Fc_wpl_pf",     "Fc",    "Fc (WPL, pf)            vs EddyPro co2_flux"),
    ("Fc_wpl_pf",     "un_Fc", "Fc (WPL, pf)            vs EddyPro un_co2_flux (pre-spectral, pre-WPL)"),
    ("sigma_w",    "sigma_w", "sigma_w"),
    ("TKE",        "TKE",   "TKE"),
    ("WS",         "WS",    "wind speed"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf", default="LPF", choices=["LPF", "GPF"])
    ap.add_argument("--det", default="LinDet", choices=["LinDet", "ConstDet"],
                    help="UTESpac detrend variant to load")
    ap.add_argument("--csv", action="store_true",
                    help="write joined table to testbed/scratch/")
    ap.add_argument("--qc-max", type=int, default=2,
                    help="keep EddyPro periods with qc <= this (0,1,2)")
    args = ap.parse_args()

    ut, z, _ = load_utespac(args.pf, args.det)
    ep, ep_file = load_eddypro()
    j = ut.join(ep, how="inner", lsuffix="_ut", rsuffix="_ep")
    print(f"UTESpac ({args.pf}, z={z} m): {len(ut)} periods "
          f"{ut.index.min()} .. {ut.index.max()}")
    print(f"EddyPro: {len(ep)} periods {ep.index.min()} .. {ep.index.max()}  "
          f"[{os.path.relpath(ep_file, ROOT)}]")
    print(f"Joined: {len(j)} periods\n")

    print(f"{'quantity':52s} {'N':>4s} {'bias':>9s} {'rmse':>9s} "
          f"{'slope':>7s} {'icpt':>8s} {'r2':>6s}")
    for ucol, ecol, label in PAIRS:
        if ucol not in ut.columns or ecol not in ep.columns:
            continue
        a = j[ucol] if ucol in j.columns else j[f"{ucol}_ut"]
        b = j[ecol] if ecol in j.columns else j[f"{ecol}_ep"]
        qc_col = {"H": "qc_H", "un_H": "qc_H", "LE": "qc_LE", "Fc": "qc_Fc",
                  "ustar": "qc_Tau"}.get(ecol)
        if qc_col and qc_col in j.columns:
            ok = j[qc_col] <= args.qc_max
            a, b = a[ok], b[ok]
        if ucol == "zeta":
            a, b = a.clip(-2, 2), b.clip(-2, 2)
        s = stats(a.values, b.values)
        print(f"{label:52s} {s['N']:4d} {s['bias']:9.3f} {s['rmse']:9.3f} "
              f"{s['slope']:7.3f} {s['intercept']:8.3f} {s['r2']:6.3f}")

    # Sign agreement on L (stability class) is more telling than its magnitude
    if "L" in ut.columns and "L" in ep.columns:
        a, b = j["L_ut"], j["L_ep"]
        m = np.isfinite(a) & np.isfinite(b)
        agree = np.mean(np.sign(a[m]) == np.sign(b[m])) if m.any() else np.nan
        print(f"\nL sign agreement (stable/unstable class): {agree:.3f} over {int(m.sum())} periods")

    if "pitch" in ep.columns:
        print(f"EddyPro planar-fit pitch/roll (per period, deg): "
              f"pitch median {np.nanmedian(ep['pitch']):.2f}, "
              f"roll median {np.nanmedian(ep['roll']):.2f}")

    if args.csv:
        out_dir = os.path.join(ROOT, "testbed", "scratch")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"vac001_{args.pf}_{args.det}_vs_eddypro.csv")
        j.to_csv(out)
        print(f"\nJoined table written: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
