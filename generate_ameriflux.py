"""AmeriFlux BASE half-hourly CSV from the UTESpac run files.

A supported standardized output of the pipeline (gameplan
``tasks/active/2026-09-05_variable-name-refactor.md``, DECIDE 7 ruling).

Reads one site's averaged products through :mod:`utespac.run_io` --
``utespac-run-3`` files, or ``utespac-run-2`` files through the read shim
-- addressing them by group and variable name; the wind statistics, the
rotated period means and the averaged logger tables come from the same
files. Writes one half-hourly BASE CSV per site, plus the slow
meteorology and radiation of an optional 1-min met directory.

The output column names are AmeriFlux's, not UTESpac's. What each one is
built from (``group/variable``):

===============  =============================================================
AmeriFlux        Source
===============  =============================================================
H                ``sensible_heat/H_pf``, else ``rho_air_ref cp_ref
                 w_t_air_cov_pf``
LE               ``latent_heat/LE_wpl_pf``, else ``Lv w_q_wpl_cov_pf``
FC               ``co2_flux/Fc_wpl_pf`` -> umol m-2 s-1
TAU              rho x ``momentum/Tau_pf``
USTAR            ``scaling/ustar_pf``, else sqrt(``momentum/Tau_pf``)
WS, WD           ``rotation`` streamwise period mean, ``wind/direction``
MO_LENGTH, ZL    ``obukhov/L``; ZL = z / L
TKE              ``tke/TKE``
T_SONIC          ``temperature/theta_v`` - Gamma (z - z_ref)
T_SONIC_SIGMA    ``sigma/ts_sigma``
U/V/W_SIGMA      ``sigma/{u,v,w}_sigma_pf``
CO2, CO2_SIGMA   ``co2_flux/co2_mole_fraction``, ``sigma/co2_sigma``
H2O, H2O_SIGMA   ``humidity/r``, ``sigma/h2o_sigma``
FH2O             ``latent_heat/w_q_wpl_cov_pf`` -> mmol m-2 s-1
WD_FILTER        ``wind/shadow_flag`` (0 clean sector, 1 tower-disturbed)
SSITC tests      ``flux_qc/{Tau,H,LE,Fc}_ssitc``
PA               the averaged logger table's pressure column [kPa]
TA, RH, VPD,     the 1-min met directory (``--met-dir``), resampled
SW/LW/NETRAD/ALB
===============  =============================================================

Density and thermodynamic terms come from ``humidity`` (``rho_air_moist``,
``r``, ``theta_v_slow``) where that group has a height, and from
``sensible_heat/rho_air_ref``, ``cp_ref`` and ``latent_heat/Lv``
otherwise.

Format requirements (ameriflux.lbl.gov/half-hourly-hourly-data-upload-format/):
  - ASCII CSV, comma-delimited, period as decimal separator
  - First two columns: TIMESTAMP_START, TIMESTAMP_END (YYYYMMDDHHMM, local standard time)
  - Single header row of variable names, no units row
  - Missing value: -9999
  - File name: <SITE_ID>_HH_<START>_<END>.csv

Usage::

    python generate_ameriflux.py
    python generate_ameriflux.py --site VAC001 --qualifier GPF_ConstDet --site-id US-xVAC001
"""

import argparse
import glob
import os
import warnings
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from utespac.model import COMPONENT, HEIGHT, TIME
from utespac.run_io import read_run, run_files

# ── configuration ────────────────────────────────────────────────────────────

ROOT_PY = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(ROOT_PY, "data")
DEFAULT_SITE = "VAC001"
DEFAULT_QUALIFIER = "GPF_ConstDet"          # <PF>_<detrending> of the run files
OUT_DIR = os.path.join(ROOT_PY, "ameriflux_output")

AVG_PER = 30      # minutes
MISSING = -9999.0
GAMMA = 0.0098    # dry adiabatic lapse rate [K/m]

M_CO2 = 44.01     # [g/mol]
M_H2O = 18.015    # [g/mol]
M_AIR = 28.97     # [g/mol]
R_GAS = 8.314     # [J/(mol K)]

# 1-min met columns, site-specific: height -> column names in the 1-min files
HMP_COLS: Dict[float, tuple] = {}
RAD_COLS: Dict[float, tuple] = {}

# ── helpers ──────────────────────────────────────────────────────────────────


def dt_to_ameriflux(dt) -> str:
    """``datetime`` -> ``YYYYMMDDHHMM``."""
    return pd.Timestamp(dt).strftime("%Y%m%d%H%M")


def _height_index(ds: xr.Dataset, height: float, tol: float = 0.01) -> Optional[int]:
    """Position of *height* on a group's height coordinate, None when absent."""
    if ds is None or HEIGHT not in ds.coords:
        return None
    hs = np.asarray(ds[HEIGHT].values, dtype=float)
    if hs.size == 0:
        return None
    j = int(np.argmin(np.abs(hs - height)))
    return j if abs(hs[j] - height) <= tol else None


def column(ds: Optional[xr.Dataset], name: str, t_idx: pd.DatetimeIndex,
           height: Optional[float] = None) -> np.ndarray:
    """One product variable on the output time grid; NaN where absent.

    Parameters
    ----------
    ds : xarray.Dataset or None
        A product group (or ``wind``).
    name : str
        Variable name in that group.
    t_idx : pandas.DatetimeIndex
        Output half-hourly grid (period-end stamps).
    height : float, optional
        Level to select when the variable has a height dimension.
    """
    empty = np.full(len(t_idx), np.nan)
    if ds is None or name not in ds:
        return empty
    da = ds[name]
    if HEIGHT in da.dims:
        if height is None:
            return empty
        j = _height_index(ds, height)
        if j is None:
            return empty
        da = da.isel({HEIGHT: j})
    ser = pd.Series(np.asarray(da.values, dtype=float),
                    index=pd.DatetimeIndex(ds[TIME].values))
    return ser.reindex(t_idx).values.astype(float)


def _has(ds: Optional[xr.Dataset], name: str) -> bool:
    return ds is not None and name in ds


def _humidity_height(hum: Optional[xr.Dataset], height: float) -> Optional[float]:
    """Nearest height of the ``humidity`` group carrying data, None when the
    group is absent or empty (the level then falls back to the reference terms)."""
    if hum is None or HEIGHT not in hum.coords or "rho_air_moist" not in hum:
        return None
    hs = np.asarray(hum[HEIGHT].values, dtype=float)
    rho = np.asarray(hum["rho_air_moist"].values, dtype=float)
    good = [h for k, h in enumerate(hs) if np.isfinite(rho[:, k]).any()]
    if not good:
        return None
    return min(good, key=lambda h: abs(h - height))


# ── reading a site ───────────────────────────────────────────────────────────


class SiteRun:
    """The averaged part of one site's run files, concatenated along time.

    Attributes
    ----------
    products : dict of str to xarray.Dataset
        Product groups by their current names (``sensible_heat``, ...).
    wind, rotation : xarray.Dataset or None
    periods : dict of str to xarray.Dataset
        Per-period means of the logger tables (the pressure column lives here).
    facts : dict
        Run/site facts of the last file read.
    """

    def __init__(self, products, wind, rotation, periods, facts):
        self.products = products
        self.wind = wind
        self.rotation = rotation
        self.periods = periods
        self.facts = facts

    def group(self, name: str) -> Optional[xr.Dataset]:
        return self.products.get(name)

    def heights(self) -> List[float]:
        """Sonic heights, ascending (AmeriFlux V = 1 is the lowest)."""
        for name in ("momentum", "sensible_heat", "sigma"):
            ds = self.products.get(name)
            if ds is not None and HEIGHT in ds.coords:
                return sorted(float(h) for h in ds[HEIGHT].values)
        return []

    def z_ref(self) -> float:
        """Reference height of the potential temperatures (``flux/reference.py``)."""
        if self.facts.get("shiftzRef", False) and self.facts.get("zRefLowestSon") is not None:
            return float(self.facts["zRefLowestSon"])
        hs = self.heights()
        return float(min(hs)) if hs else 0.0


def _concat(parts: List[xr.Dataset]) -> xr.Dataset:
    """Concatenate along time, dropping duplicate stamps (later files win)."""
    ds = xr.concat(parts, dim=TIME, join="outer", data_vars="all", combine_attrs="override")
    ds = ds.sortby(TIME)
    _, keep = np.unique(np.asarray(ds[TIME].values), return_index=True)
    return ds.isel({TIME: np.sort(keep)})


def read_site(root, site: str, avg_per: int = AVG_PER,
              qualifier: str = DEFAULT_QUALIFIER) -> SiteRun:
    """Every run file of *site* matching *qualifier*, read as one :class:`SiteRun`."""
    files = run_files(root, site, avg_per=avg_per, qualifier=qualifier)
    if not files:
        raise FileNotFoundError(
            f"no run files for {site!r} ({qualifier}, {avg_per} min) under {root}")
    products: Dict[str, List[xr.Dataset]] = {}
    periods: Dict[str, List[xr.Dataset]] = {}
    wind: List[xr.Dataset] = []
    rotation: List[xr.Dataset] = []
    facts: Dict = {}
    for path in files:
        run = read_run(path)
        facts = run.site
        for name, ds in run.products.items():
            products.setdefault(name, []).append(ds)
        for name, ds in run.periods.items():
            periods.setdefault(name, []).append(ds)
        if run.wind is not None:
            wind.append(run.wind)
        if run.rotation is not None:
            rotation.append(run.rotation.drop_dims("time_hf", errors="ignore"))
    return SiteRun({k: _concat(v) for k, v in products.items()},
                   _concat(wind) if wind else None,
                   _concat(rotation) if rotation else None,
                   {k: _concat(v) for k, v in periods.items()}, facts)


# ── the AmeriFlux frame ──────────────────────────────────────────────────────


def pressure_kPa(site: SiteRun, t_idx: pd.DatetimeIndex) -> np.ndarray:
    """PA [kPa] from the first averaged logger column named ``Pressure*``.

    The logger columns keep the datalogger program's own names; the
    magnitude heuristic of the original exporter (Pa, hPa or kPa) is kept.
    """
    for name, ds in sorted(site.periods.items()):
        for var in ds.data_vars:
            if "Pressure" not in str(var):
                continue
            raw = np.asarray(ds[var].values, dtype=float)
            if raw.ndim != 1:
                continue
            med = np.nanmedian(raw)
            if med > 50000:
                raw = raw / 1000.0
            elif med > 200:
                raw = raw / 10.0
            ser = pd.Series(raw, index=pd.DatetimeIndex(ds[TIME].values))
            return ser.reindex(t_idx).values.astype(float)
    return np.full(len(t_idx), np.nan)


def ameriflux_frame(site: SiteRun, avg_per: int = AVG_PER, z_ref: Optional[float] = None,
                    met: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The AmeriFlux BASE table of one site.

    Parameters
    ----------
    site : SiteRun
        Products of the site's run files (:func:`read_site`).
    avg_per : int
        Averaging period [min]; sets TIMESTAMP_START and the output grid.
    z_ref : float, optional
        Reference height of the potential temperatures [m]; taken from the
        run facts when omitted.
    met : pandas.DataFrame, optional
        Slow meteorology already resampled to *avg_per* (:func:`load_met`).

    Returns
    -------
    pandas.DataFrame
        Indexed by period end, TIMESTAMP_START / TIMESTAMP_END first, NaN
        where a quantity is missing (replaced by -9999 on write).
    """
    heights = site.heights()
    if not heights:
        raise ValueError("no sonic heights in the run products")
    if z_ref is None:
        z_ref = site.z_ref()

    sh, lh, co2, mom = (site.group(g) for g in
                        ("sensible_heat", "latent_heat", "co2_flux", "momentum"))
    scal, sig, obk, tke = (site.group(g) for g in ("scaling", "sigma", "obukhov", "tke"))
    temp, hum, qc = (site.group(g) for g in ("temperature", "humidity", "flux_qc"))

    ref = sh if sh is not None else mom
    times = pd.DatetimeIndex(ref[TIME].values)
    t_idx = pd.date_range(start=times.min(), end=times.max(), freq=f"{avg_per}min")

    df = pd.DataFrame(index=t_idx)
    df["TIMESTAMP_END"] = [dt_to_ameriflux(t) for t in t_idx]
    df["TIMESTAMP_START"] = [dt_to_ameriflux(t - timedelta(minutes=avg_per)) for t in t_idx]

    pa = pressure_kPa(site, t_idx)

    for v_idx, height in enumerate(heights, start=1):
        def col(ds, name, h=height):
            return column(ds, name, t_idx, h)

        # ---- density and thermodynamic terms ----
        hum_h = _humidity_height(hum, height)
        if hum_h is not None:
            rho = column(hum, "rho_air_moist", t_idx, hum_h)
            r = column(hum, "r", t_idx, hum_h)                       # [g/kg]
            theta_v_slow = column(hum, "theta_v_slow", t_idx, hum_h)  # [K]
            cp = 1004.67 * (1.0 + 0.84 * r / 1000.0)
            Lv = (2.501 - 0.00237 * (theta_v_slow - 273.15)) * 1e6   # [J/kg]
        else:
            rho = column(sh, "rho_air_ref", t_idx)
            cp = column(sh, "cp_ref", t_idx)
            Lv = col(lh, "Lv") * 1000.0                              # J/g -> J/kg
            r = np.full(len(t_idx), np.nan)
            theta_v_slow = np.full(len(t_idx), np.nan)

        # ---- turbulent fluxes ----
        if _has(sh, "H_pf"):
            df[f"H_1_{v_idx}_1"] = col(sh, "H_pf")
        else:
            df[f"H_1_{v_idx}_1"] = rho * cp * col(sh, "w_t_air_cov_pf")

        q_wpl = col(lh, "w_q_wpl_cov_pf")                            # [kg m-2 s-1]
        if _has(lh, "LE_wpl_pf"):
            df[f"LE_1_{v_idx}_1"] = col(lh, "LE_wpl_pf")
        else:
            df[f"LE_1_{v_idx}_1"] = Lv * q_wpl

        df[f"FC_1_{v_idx}_1"] = col(co2, "Fc_wpl_pf") / M_CO2 * 1000.0 * 1e6

        tau_pf = col(mom, "Tau_pf")                                  # [m2 s-2] = u*^2
        if _has(scal, "ustar_pf"):
            df[f"USTAR_1_{v_idx}_1"] = col(scal, "ustar_pf")
        else:
            df[f"USTAR_1_{v_idx}_1"] = np.sqrt(np.abs(tau_pf)) * np.sign(tau_pf)
        df[f"TAU_1_{v_idx}_1"] = rho * tau_pf

        ws = np.full(len(t_idx), np.nan)
        if site.rotation is not None and "rotated_mean" in site.rotation:
            j = _height_index(site.rotation, height)
            if j is not None:
                rot_u = site.rotation["rotated_mean"].isel({HEIGHT: j}).sel({COMPONENT: "u"})
                ws = pd.Series(np.asarray(rot_u.values, dtype=float),
                               index=pd.DatetimeIndex(site.rotation[TIME].values)
                               ).reindex(t_idx).values.astype(float)
        df[f"WS_1_{v_idx}_1"] = ws
        df[f"WD_1_{v_idx}_1"] = col(site.wind, "direction")

        L_col = col(obk, "L")
        df[f"MO_LENGTH_1_{v_idx}_1"] = L_col

        df[f"TKE_1_{v_idx}_1"] = col(tke, "TKE")
        df[f"U_SIGMA_1_{v_idx}_1"] = col(sig, "u_sigma_pf")
        df[f"V_SIGMA_1_{v_idx}_1"] = col(sig, "v_sigma_pf")
        df[f"W_SIGMA_1_{v_idx}_1"] = col(sig, "w_sigma_pf")
        df[f"T_SONIC_SIGMA_1_{v_idx}_1"] = col(sig, "ts_sigma")

        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"ZL_1_{v_idx}_1"] = np.where(np.isfinite(L_col) & (L_col != 0.0),
                                             height / L_col, np.nan)

        df[f"FH2O_1_{v_idx}_1"] = q_wpl * 1e6 / M_H2O                # [mmol m-2 s-1]
        df[f"H2O_1_{v_idx}_1"] = r * M_AIR / M_H2O                   # [mmol mol-1]

        # molar density of moist air for the gas sigma conversions
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_mol = np.where(
                np.isfinite(pa * 1000.0) & np.isfinite(theta_v_slow) & (theta_v_slow > 200),
                pa * 1000.0 / (R_GAS * theta_v_slow), np.nan)

        sig_co2 = col(sig, "co2_sigma")                              # [mg m-3]
        sig_h2o = col(sig, "h2o_sigma")                              # [g m-3]
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"CO2_SIGMA_1_{v_idx}_1"] = sig_co2 / (M_CO2 * 1000.0) / rho_mol * 1e6
            df[f"H2O_SIGMA_1_{v_idx}_1"] = sig_h2o / M_H2O / rho_mol * 1e3

        df[f"T_SONIC_1_{v_idx}_1"] = col(temp, "theta_v") - GAMMA * (height - z_ref)
        df[f"CO2_1_{v_idx}_1"] = col(co2, "co2_mole_fraction")

        # ---- WD_FILTER: report only, no masking (same treatment as SSITC flags) ----
        df[f"WD_FILTER_1_{v_idx}_1"] = col(site.wind, "shadow_flag")

        # ---- SSITC quality flags ----
        for out_name, key in (("TAU", "Tau_ssitc"), ("H", "H_ssitc"),
                              ("LE", "LE_ssitc"), ("FC", "Fc_ssitc")):
            df[f"{out_name}_SSITC_TEST_1_{v_idx}_1"] = col(qc, key)

    # ---- slow meteorology and radiation ----
    if met is not None and not met.empty:
        add_met(df, met, t_idx)

    df["PA_1_1_1"] = pa
    return df


# ── slow meteorology from the 1-min files ────────────────────────────────────


def load_met(met_dir, pattern: str = "*1min*.txt", avg_per: int = AVG_PER
             ) -> Optional[pd.DataFrame]:
    """The 1-min met files of *met_dir*, block-averaged to *avg_per*.

    The files carry ``year``, ``day``, ``HM`` (and optionally ``second``)
    columns; the resampled frame is stamped at the period end, as the
    products are.
    """
    files = sorted(glob.glob(os.path.join(str(met_dir), pattern))) if met_dir else []
    if not files:
        return None
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path, header=0))
        except Exception:
            warnings.warn(f"unreadable 1-min met file: {path}")
    if not frames:
        return None
    allmin = pd.concat(frames, ignore_index=True)

    def _dt(row):
        hm = int(row["HM"])
        try:
            return (pd.Timestamp(int(row["year"]), 1, 1)
                    + timedelta(days=int(row["day"]) - 1, hours=hm // 100, minutes=hm % 100,
                                seconds=int(row.get("second", 0))))
        except Exception:
            return pd.NaT

    allmin["dt"] = allmin.apply(_dt, axis=1)
    allmin = allmin.dropna(subset=["dt"]).set_index("dt").sort_index()
    out = allmin.resample(f"{avg_per}min", closed="right", label="right").mean()
    out.index = pd.DatetimeIndex(out.index).round(f"{avg_per}min")
    return out


def add_met(df: pd.DataFrame, met: pd.DataFrame, t_idx: pd.DatetimeIndex) -> None:
    """TA, RH, VPD, radiation and albedo columns from the resampled met frame."""

    def _met(col_name):
        if col_name not in met.columns:
            warnings.warn(f"Column {col_name!r} not found in the 1-min met data.")
            return np.full(len(t_idx), np.nan)
        return met[col_name].reindex(t_idx).values.astype(float)

    for v_idx, height in enumerate(sorted(HMP_COLS), start=1):
        t_col, rh_col = HMP_COLS[height]
        ta, rh = _met(t_col), _met(rh_col)
        df[f"TA_1_{v_idx}_1"] = ta
        df[f"RH_1_{v_idx}_1"] = rh
        es = 6.1078 * np.exp(17.27 * ta / (ta + 237.3))
        df[f"VPD_1_{v_idx}_1"] = np.where(np.isfinite(es), es * (1.0 - rh / 100.0), np.nan)

    for v_idx, height in enumerate(sorted(RAD_COLS), start=1):
        sw_in, sw_out, lw_in, lw_out, rn = RAD_COLS[height]
        df[f"SW_IN_1_{v_idx}_1"] = _met(sw_in)
        df[f"SW_OUT_1_{v_idx}_1"] = _met(sw_out)
        df[f"LW_IN_1_{v_idx}_1"] = _met(lw_in)
        df[f"LW_OUT_1_{v_idx}_1"] = _met(lw_out)
        df[f"NETRAD_1_{v_idx}_1"] = _met(rn)
        with np.errstate(divide="ignore", invalid="ignore"):
            sw_in_v = df[f"SW_IN_1_{v_idx}_1"].values.astype(float)
            sw_out_v = df[f"SW_OUT_1_{v_idx}_1"].values.astype(float)
            df[f"ALB_1_{v_idx}_1"] = np.where(sw_in_v > 5.0, 100.0 * sw_out_v / sw_in_v, np.nan)


# ── writing ──────────────────────────────────────────────────────────────────


def write_base_csv(df: pd.DataFrame, out_dir, site_id: str) -> str:
    """Write the BASE CSV (-9999 fill, 6 decimals) and return its path."""
    ts_cols = ["TIMESTAMP_START", "TIMESTAMP_END"]
    data_cols = [c for c in df.columns if c not in ts_cols]
    out = df[ts_cols + data_cols].copy()
    for c in data_cols:
        vals = out[c].values.astype(float)
        mask = np.isfinite(vals)                 # NaN and +-inf are both missing
        vals[mask] = np.round(vals[mask], 6)
        vals[~mask] = MISSING
        out[c] = vals
    os.makedirs(str(out_dir), exist_ok=True)
    fname = f"{site_id}_HH_{out['TIMESTAMP_START'].iloc[0]}_{out['TIMESTAMP_END'].iloc[-1]}.csv"
    path = os.path.join(str(out_dir), fname)
    out.to_csv(path, index=False)
    return path


# ── command line ─────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=DEFAULT_ROOT, help="data root holding <SITE>/output")
    p.add_argument("--site", default=DEFAULT_SITE, help="site folder name")
    p.add_argument("--qualifier", default=DEFAULT_QUALIFIER,
                   help="run-file qualifier, <PF>_<detrending> (default: %(default)s)")
    p.add_argument("--avg-per", type=int, default=AVG_PER, help="averaging period [min]")
    p.add_argument("--site-id", default=None,
                   help="AmeriFlux site ID (default: US-x<SITE>, a placeholder until registered)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--met-dir", default=None, help="directory of 1-min met files (TA/RH/radiation)")
    p.add_argument("--z-ref", type=float, default=None,
                   help="reference height of the potential temperatures [m]")
    return p.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    site = read_site(args.root, args.site, args.avg_per, args.qualifier)
    heights = site.heights()
    print(f"{args.site} ({args.qualifier}): heights (V=1 lowest) {heights} m, "
          f"z_ref {args.z_ref if args.z_ref is not None else site.z_ref():g} m")

    met = load_met(args.met_dir, avg_per=args.avg_per) if args.met_dir else None
    if met is None and args.met_dir:
        warnings.warn(f"no 1-min met files under {args.met_dir} -- "
                      "TA/RH/VPD/radiation columns will be absent.")
    df = ameriflux_frame(site, avg_per=args.avg_per, z_ref=args.z_ref, met=met)

    site_id = args.site_id or f"US-x{args.site}"
    path = write_base_csv(df, args.out_dir, site_id)
    print(f"Wrote {len(df)} half-hourly rows x {len(df.columns)} columns")
    print(f"Output: {path}")

    print("\nVariable coverage (non-missing rows / total):")
    total = len(df)
    for c in df.columns:
        if c.startswith("TIMESTAMP"):
            continue
        n_valid = int(np.isfinite(df[c].values.astype(float)).sum())
        if n_valid < total:
            print(f"  {c:40s}  {n_valid:4d}/{total}  ({100 * n_valid / total:.0f}%)")
    print("  (columns with 100% coverage not shown)")
    return path


if __name__ == "__main__":
    main()
