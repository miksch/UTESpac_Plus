"""fluxes – compute all turbulent statistics (H, τ, LE, CO2, σ, L, η, ε, skew, …)."""

from typing import Dict, List, Optional
import numpy as np
from scipy.stats import skew as scipy_skew

from .nandetrend import nandetrend
from .simple_avg import simple_avg
from .rh_to_spec_hum import rh_to_spec_hum
from .get_virtual_pot_temp import get_virtual_pot_temp
from .calc_dissipation_rate import calc_dissipation_rate
from .calc_snsp_angle import calc_snsp_angle
from .sonic_temperature import SONIC_HUMIDITY_COEFF, air_temperature_perturbation
from .find_delta_flux import find_delta_flux
from .find_delta_time import find_delta_time
from .find_eta import find_eta
from .calc_ssitc_flags import calc_ssitc_flags
from .site_config import sonic_for

Rd = 287.058   # J/(kg·K)
Rv = 461.495   # J/(kg·K)
Mv = 18.0153   # g/mol  H2O
Md = 28.97     # g/mol  dry air


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _skew(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    if len(v) < 3:
        return np.nan
    return float(scipy_skew(v, bias=False))


def _get_sensor_col(sensor_info, key, height):
    """Return (table_idx, col_idx) for *key* at *height*, or None."""
    if key not in sensor_info:
        return None
    rows = sensor_info[key][:, 2] == height
    if not rows.any():
        return None
    si = int(np.where(rows)[0][0])
    return int(sensor_info[key][si, 0]), int(sensor_info[key][si, 1])


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------

def fluxes(
    data: List[Optional[np.ndarray]],
    rotated_sonic_data: np.ndarray,
    pf_sonic_data: np.ndarray,
    info: Dict,
    output: Dict,
    sensor_info: Dict,
    table_names: List[str],
) -> tuple:
    """Compute all turbulent fluxes and statistics for all sonic levels.

    Returns
    -------
    output : dict
    raw    : dict or None
    """
    print("\nComputing Fluxes")

    if "u" not in sensor_info:
        return output, None

    save_raw = bool(info.get("saveRawConditionedData", False))
    raw: Dict = {} if save_raw else None

    # -----------------------------------------------------------------------
    # Reference scalars (pressure, temperature, humidity, density)
    # -----------------------------------------------------------------------
    t = data[int(sensor_info["u"][0, 0])][:, 0]

    z_ref    = float(np.min(sensor_info["u"][:, 2]))
    altitude = float(info.get("siteElevation", 0))
    P_ref_kPa = 101.325 * (1 - 2.25577e-5 * (altitude + z_ref)) ** 5.25588

    # --- pressure ---
    P_kPa_avg: Optional[np.ndarray] = None
    P_raw_hf: Optional[np.ndarray] = None   # per-sample pressure (kPa), for ppm calculation
    P_t_hf:   Optional[np.ndarray] = None   # timestamps matching P_raw_hf
    if "P" in sensor_info:
        idx_P = int(np.argmin(np.abs(sensor_info["P"][:, 2] - z_ref)))
        P_tbl = int(sensor_info["P"][idx_P, 0])
        P_col = int(sensor_info["P"][idx_P, 1])
        P_raw = data[P_tbl][:, P_col].copy()
        P_t   = data[P_tbl][:, 0]
        if np.nanmedian(P_raw) > 200:
            P_raw /= 10.0
        P_avg_mat = simple_avg(np.column_stack([P_raw, P_t]), info["avgPer"], return_timestamps=False)
        P_kPa_avg = P_avg_mat[:, 0]
        if save_raw:
            raw["P"] = np.column_stack([P_t, P_raw])
        # Always keep raw samples for ppm conversion (matches MATLAB: Pson = data(:,PCol) always)
        P_raw_hf = P_raw
        P_t_hf   = P_t
        if np.nansum(~np.isnan(P_kPa_avg)) > 0 and \
                np.abs((np.nanmedian(P_kPa_avg) - P_ref_kPa) / P_ref_kPa) < 0.05:
            print(f"Barometer found. Median P = {np.nanmedian(P_kPa_avg):.3g} kPa")
        else:
            P_kPa_avg = None

    if P_kPa_avg is None:
        N_est = int(round((t[-1] - t[0]) / (info["avgPer"] / (24.0 * 60.0))))
        P_kPa_avg = np.full(N_est, P_ref_kPa)
        print(f"No valid barometer – using P_ref = {P_ref_kPa:.3g} kPa.")

    # --- reference temperature ---
    T_ref_K_avg: Optional[np.ndarray] = None
    if "T" in sensor_info:
        idx_T  = int(np.argmin(np.abs(sensor_info["T"][:, 2] - z_ref)))
        T_tbl  = int(sensor_info["T"][idx_T, 0])
        T_col  = int(sensor_info["T"][idx_T, 1])
        T_raw  = data[T_tbl][:, T_col]
        T_avg_mat = simple_avg(np.column_stack([T_raw, data[T_tbl][:, 0]]),
                               info["avgPer"], return_timestamps=False)
        T_ref_K_avg = T_avg_mat[:, 0]
        if np.nanmedian(T_ref_K_avg) < 200:
            T_ref_K_avg += 273.15
        print(f"Slow-response T found. Median T_ref = {np.nanmedian(T_ref_K_avg) - 273.15:.3g} °C")

    if T_ref_K_avg is None or np.nansum(~np.isnan(T_ref_K_avg)) == 0:
        if "Tson" in sensor_info:
            idx_T = int(np.argmin(np.abs(sensor_info["Tson"][:, 2] - z_ref)))
            T_tbl = int(sensor_info["Tson"][idx_T, 0])
            T_col = int(sensor_info["Tson"][idx_T, 1])
            T_raw = data[T_tbl][:, T_col]
            T_avg_mat = simple_avg(np.column_stack([T_raw, t]),
                                   info["avgPer"], return_timestamps=False)
            T_ref_K_avg = T_avg_mat[:, 0]
            if np.nanmedian(T_ref_K_avg) < 200:
                T_ref_K_avg += 273.15
            print(f"Using sonic T as Tref. Median = {np.nanmedian(T_ref_K_avg) - 273.15:.3g} °C")
        else:
            T_ref_K_avg = np.full(len(P_kPa_avg), 293.15)

    # --- reference specific humidity ---
    q_ref_avg: np.ndarray
    q_ref_fast: np.ndarray
    if "RH" in sensor_info:
        idx_RH = int(np.argmin(np.abs(sensor_info["RH"][:, 2] - z_ref)))
        RH_tbl = int(sensor_info["RH"][idx_RH, 0])
        RH_col = int(sensor_info["RH"][idx_RH, 1])
        RH_raw = data[RH_tbl][:, RH_col]
        RH_t   = data[RH_tbl][:, 0]
        RH_avg_mat = simple_avg(np.column_stack([RH_raw, RH_t]), info["avgPer"])
        RH_avg_t   = RH_avg_mat[:, -1]
        RH_avg     = RH_avg_mat[:, 0]
        q_ref_avg  = rh_to_spec_hum(RH_avg, P_kPa_avg, T_ref_K_avg)
        valid = ~np.isnan(q_ref_avg)
        if valid.any():
            x = np.concatenate([[np.floor(RH_avg_t[valid][0])], RH_avg_t[valid]])
            y = np.concatenate([[q_ref_avg[valid][0]], q_ref_avg[valid]])
            q_ref_fast = np.interp(t, x, y)
            print(f"RH found. Median q_ref = {1000 * np.nanmedian(q_ref_avg):.3g} g/kg")
        else:
            q_ref_avg  = np.full(len(T_ref_K_avg), info.get("qRef", 12) / 1000.0)
            q_ref_fast = np.full(len(t),            info.get("qRef", 12) / 1000.0)
    elif "irgaH2O" in sensor_info or "KH2O" in sensor_info:
        # No slow-response RH — derive q from IRGA/KH2O water vapour density (matches MATLAB)
        h2o_key = "irgaH2O" if "irgaH2O" in sensor_info else "KH2O"
        idx_h2o = int(np.argmin(np.abs(sensor_info[h2o_key][:, 2] - z_ref)))
        h2o_tbl = int(sensor_info[h2o_key][idx_h2o, 0])
        h2o_col = int(sensor_info[h2o_key][idx_h2o, 1])
        h2o_raw = data[h2o_tbl][:, h2o_col].copy()   # g/m³
        h2o_t   = data[h2o_tbl][:, 0]
        h2o_avg_mat = simple_avg(np.column_stack([h2o_raw, h2o_t]), info["avgPer"])
        h2o_avg_t   = h2o_avg_mat[:, -1]
        h2o_avg     = h2o_avg_mat[:, 0]              # g/m³ averaged
        rho_v_ref   = h2o_avg / 1000.0                                   # kg/m³ vapour density
        e_ref_Pa    = rho_v_ref * Rv * T_ref_K_avg                        # vapour pressure
        rho_d_ref   = (P_kPa_avg * 1000.0 - e_ref_Pa) / (Rd * T_ref_K_avg)  # dry-air density
        q_ref_avg   = rho_v_ref / (rho_d_ref + rho_v_ref)                # kg/kg specific humidity
        valid = ~np.isnan(q_ref_avg)
        if valid.any():
            x = np.concatenate([[np.floor(h2o_avg_t[valid][0])], h2o_avg_t[valid]])
            y = np.concatenate([[q_ref_avg[valid][0]], q_ref_avg[valid]])
            q_ref_fast = np.interp(t, x, y)
            print(f"No RH – qRef from {h2o_key}. Median q_ref = {1000 * np.nanmedian(q_ref_avg):.3g} g/kg")
        else:
            q_ref_avg  = np.full(len(T_ref_K_avg), info.get("qRef", 12) / 1000.0)
            q_ref_fast = np.full(len(t), info.get("qRef", 12) / 1000.0)
    else:
        q_ref_avg  = np.full(len(T_ref_K_avg), info.get("qRef", 12) / 1000.0)
        q_ref_fast = np.full(len(t),            info.get("qRef", 12) / 1000.0)

    # --- moist-air density ---
    P_v_avg          = q_ref_avg * P_kPa_avg / 0.622
    P_d_avg          = P_kPa_avg - P_v_avg
    rho_d_avg        = 1000.0 * P_d_avg / (Rd * T_ref_K_avg)
    rho_v_avg        = 1000.0 * P_v_avg / (Rv * T_ref_K_avg)
    rho_avg          = rho_d_avg + rho_v_avg
    T_virt_ref_K_avg = T_ref_K_avg * (1.0 + 0.61 * q_ref_avg)

    # Manual zRef override — use when running a single high sonic (e.g. 51.5 m)
    # but wanting virtual-theta / specificHum computed relative to the lowest
    # sonic on the full tower (info["zRefLowestSon"]).
    if info.get("shiftzRef", False):
        z_ref = float(info["zRefLowestSon"])

    print(f"ρ_moist = {np.nanmedian(rho_avg):.3g} kg/m³  "
          f"ρ_dry = {np.nanmedian(rho_d_avg):.3g} kg/m³  "
          f"T_virt_ref = {np.nanmedian(T_virt_ref_K_avg) - 273.15:.3g} °C  "
          f"zRef = {z_ref:.2f} m")

    # -----------------------------------------------------------------------
    # Allocate output matrices
    # -----------------------------------------------------------------------
    num_sonics    = sensor_info["u"].shape[0]
    N             = int(round((t[-1] - t[0]) / (info["avgPer"] / (24.0 * 60.0))))
    bp            = np.round(np.linspace(0, len(t), N + 1)).astype(int)
    has_fw        = "fw" in sensor_info
    # MATLAB fluxes.m: numSigmaVariables = 13 when fw exists, 12 otherwise.
    # Stride 13 leaves a NaN gap column between sonics; _trim_both removes it so
    # the final output always has 12 meaningful columns per sonic.
    num_sig_vars  = 13 if has_fw else 12
    num_LH_vars   = 8
    num_CO2_vars  = 5
    num_sk_vars   = 8
    num_fl_vars   = 5
    num_eta_vars  = 6
    num_df_vars   = 6
    num_dt_vars   = 6
    num_tr_vars   = 7
    num_tau_vars  = 14
    num_snsp_vars = 4

    H_mat      = np.full((N, 3 + num_sonics * 12),           np.nan)
    Hlat_mat   = np.full((N, 1 + num_sonics * 12),           np.nan)
    derivedT_cols: list = []   # list of (col_array, header_str) built per sonic
    tau_mat    = np.full((N, 1 + num_sonics * num_tau_vars),  np.nan)
    tke_mat    = np.full((N, 1 + num_sonics),                  np.nan)  # time + TKE per sonic (no rho/cp)
    L_mat      = np.full((N, 1 + num_sonics),                 np.nan)
    sigma_mat  = np.full((N, 1 + num_sonics * num_sig_vars),  np.nan)
    R_mat      = np.full((N, 1 + num_sonics * 17),            np.nan)  # 17 per sonic matches MATLAB
    eta_mat    = np.full((N, 1 + num_sonics * num_eta_vars),  np.nan)
    df_mat     = np.full((N, 1 + num_sonics * num_df_vars),   np.nan)
    dt_mat     = np.full((N, 1 + num_sonics * num_dt_vars),   np.nan)
    turbtr_mat = np.full((N, 1 + num_sonics * num_tr_vars),   np.nan)
    eps_mat    = np.full((N, 2 + num_sonics),                 np.nan)
    skew_mat   = np.full((N, 1 + num_sonics * num_sk_vars),   np.nan)
    H_snsp     = np.full((N, 1 + num_sonics * num_snsp_vars), np.nan)
    Flux_lat   = np.full((N, 1 + num_sonics * num_fl_vars),   np.nan)
    LH_mat     = np.full((N, 1 + num_sonics * num_LH_vars),   np.nan)
    CO2fx_mat  = np.full((N, 1 + num_sonics * num_CO2_vars),  np.nan)
    num_qc_vars = 8  # TAU, H, LE, FC × (SSITC + SS_ONLY) per sonic
    fluxQC_mat = np.full((N, 1 + num_sonics * num_qc_vars),   np.nan)

    # MATLAB parity switch: legacy 0.61 sonic-humidity coefficient and the
    # buoyancy flux in the WPL terms (findings 3-4 of the 2026-08-22 audit).
    matlab_compat = bool(info.get("matlabCompat", False))

    # Slope geometry (SiteInfo): slope angle, fall-line direction, and which
    # planar-fit horizontal axis lies along the fall line. Only the angle
    # matters on flat sites; a sloped site without the other two keys gets
    # the French Meadows values with a warning.
    angle = float(info.get("angle", 0))
    downslope_aspect = info.get("downslopeAspect")
    slope_axis = info.get("slopeAxis")
    if angle != 0 and (downslope_aspect is None or slope_axis is None):
        import warnings
        warnings.warn(
            f"angle = {angle} deg but downslopeAspect/slopeAxis are not set in "
            "siteInfo; assuming French Meadows geometry (downslopeAspect = 30, "
            "slopeAxis = 'v') for the slope-normal heat flux and L."
        )
    downslope_aspect = 30.0 if downslope_aspect is None else float(downslope_aspect)
    slope_axis = "v" if slope_axis is None else str(slope_axis)
    if slope_axis not in ("u", "v"):
        raise ValueError(f"info['slopeAxis'] must be 'u' or 'v', got {slope_axis!r}")

    # -----------------------------------------------------------------------
    # Per-sonic loop
    # -----------------------------------------------------------------------
    for ii in range(num_sonics):
        try:
            tbl_idx  = int(sensor_info["u"][ii, 0])
            height   = float(sensor_info["u"][ii, 2])
            manufact = int(sensor_info["u"][ii, 4]) if sensor_info["u"].shape[1] > 4 else 1

            mask_h = sensor_info["u"][:, 2] == height
            u_col  = int(sensor_info["u"][mask_h, 1][0])
            v_col  = int(sensor_info["v"][sensor_info["v"][:, 2] == height, 1][0])
            w_col  = int(sensor_info["w"][sensor_info["w"][:, 2] == height, 1][0])
            Ts_col = int(sensor_info["Tson"][sensor_info["Tson"][:, 2] == height, 1][0]) \
                     if "Tson" in sensor_info else None

            u_raw = data[tbl_idx][:, u_col]
            v_raw = data[tbl_idx][:, v_col]
            w_raw = data[tbl_idx][:, w_col]

            # ---- flags ----
            nf_name = f"{table_names[tbl_idx]}NanFlag"
            sf_name = f"{table_names[tbl_idx]}SpikeFlag"
            nf = output.get(nf_name, np.zeros((N, max(u_col, v_col, w_col) + 1), dtype=bool))
            sf = output.get(sf_name, np.zeros((N, max(u_col, v_col, w_col) + 1), dtype=bool))

            def _flag(col):
                return (nf[:, col] | sf[:, col]) if col < nf.shape[1] \
                       else np.zeros(N, dtype=bool)

            unrot_flag = (_flag(w_col)).astype(bool)
            rot_flag   = (_flag(u_col) | _flag(v_col) | _flag(w_col)).astype(bool)

            # Include averaged sonic diagnostic in flags (matches MATLAB fluxes.m lines 363-381).
            # Periods where mean(diagnostic) >= meanSonicDiagnosticLimit are flagged bad.
            if "sonDiagnostic" in sensor_info:
                diag_rows = sensor_info["sonDiagnostic"][:, 2] == height
                if diag_rows.any():
                    d_tbl = int(sensor_info["sonDiagnostic"][diag_rows, 0][0])
                    d_col = int(sensor_info["sonDiagnostic"][diag_rows, 1][0])
                    tbl_avg = output.get(table_names[d_tbl])
                    if tbl_avg is not None and d_col < tbl_avg.shape[1]:
                        diag_avg = tbl_avg[:, d_col].copy().astype(float)
                        diag_avg[np.isnan(diag_avg)] = 0.0
                        limit = info.get("diagnosticTest", {}).get(
                            "meanSonicDiagnosticLimit", 50)
                        diag_flag = diag_avg >= limit
                        unrot_flag = unrot_flag | diag_flag
                        rot_flag   = rot_flag   | diag_flag

            Ts_flag    = np.zeros(N, dtype=bool)

            T_son = np.full(len(t), np.nan)
            if Ts_col is not None:
                T_son = data[tbl_idx][:, Ts_col].copy()
                if np.nanmedian(T_son) > 250:
                    T_son -= 273.15
                Ts_flag = _flag(Ts_col)

            # ---- rotated and PF-only wind columns ----
            r_hdr  = output.get("rotatedSonicHeader", [])
            ru_col = rv_col = rw_col = None
            try:
                ru_col = r_hdr.index(f"{height}m:u")
                rv_col = r_hdr.index(f"{height}m:v")
                rw_col = r_hdr.index(f"{height}m:w")
                u_pf = rotated_sonic_data[:, ru_col]
                v_pf = rotated_sonic_data[:, rv_col]
                w_pf = rotated_sonic_data[:, rw_col]
            except (ValueError, IndexError):
                u_pf = v_pf = w_pf = np.full(len(t), np.nan)

            try:
                # PFSonic (before yaw rotation). u_tilt is the component along
                # the fall line, v_tilt across it: slopeAxis names the PF axis
                # that points downslope (French Meadows: v, hence the swap
                # MATLAB fluxes.m:736-737 hardcodes).
                along_col, across_col = (rv_col, ru_col) if slope_axis == "v" else (ru_col, rv_col)
                u_tilt = pf_sonic_data[:, along_col]  if along_col  is not None else np.full(len(t), np.nan)
                v_tilt = pf_sonic_data[:, across_col] if across_col is not None else np.full(len(t), np.nan)
                w_tilt = pf_sonic_data[:, rw_col]     if rw_col     is not None else np.full(len(t), np.nan)
            except Exception:
                u_tilt = v_tilt = w_tilt = np.full(len(t), np.nan)

            # ---- derived temperatures ----
            Gamma     = 0.0098
            theta_son = T_son + Gamma * (height - z_ref)
            # theta_son_air is computed after q_ref_fast_local is set (uses level-specific q)

            # ---- height-specific pressure (use co-located IRGASON P if available) ----
            P_kPa_avg_lev = P_kPa_avg
            P_raw_hf_lev  = P_raw_hf
            P_t_hf_lev    = P_t_hf
            if "P" in sensor_info:
                p_rows_lev = np.abs(sensor_info["P"][:, 2] - height) < 0.5
                if p_rows_lev.any():
                    p_tbl_lev = int(sensor_info["P"][p_rows_lev, 0][0])
                    p_col_lev = int(sensor_info["P"][p_rows_lev, 1][0])
                    P_raw_lev = data[p_tbl_lev][:, p_col_lev].copy()
                    P_t_lev   = data[p_tbl_lev][:, 0]
                    if np.nanmedian(P_raw_lev) > 200:
                        P_raw_lev /= 10.0
                    P_avg_mat_lev = simple_avg(
                        np.column_stack([P_raw_lev, P_t_lev]),
                        info["avgPer"], return_timestamps=False)
                    P_lev_check = P_avg_mat_lev[:, 0]
                    if (np.nansum(~np.isnan(P_lev_check)) > 0 and
                            np.abs((np.nanmedian(P_lev_check) - P_ref_kPa) / P_ref_kPa) < 0.05):
                        P_kPa_avg_lev = P_lev_check
                        P_raw_hf_lev  = P_raw_lev
                        P_t_hf_lev    = P_t_lev

            # ---- level-specific humidity (useTrefHMP path) ----
            q_ref_fast_local = q_ref_fast.copy()
            virtual_theta_avg_local = None
            if info.get("useTrefHMP") and "RH" in sensor_info and "T" in sensor_info:
                rh_rows = sensor_info["RH"][:, 2] == height
                t_rows  = sensor_info["T"][:, 2]  == height
                if rh_rows.any() and t_rows.any():
                    RH_tbl2 = int(sensor_info["RH"][rh_rows, 0][0])
                    RH_col2 = int(sensor_info["RH"][rh_rows, 1][0])
                    T_tbl2  = int(sensor_info["T"][t_rows, 0][0])
                    T_col2  = int(sensor_info["T"][t_rows, 1][0])
                    freq_slow = info.get("avgSlowFreq", 1)
                    RH_lev = data[RH_tbl2][:, RH_col2]
                    T_lev  = data[T_tbl2][:, T_col2]
                    t_lev  = data[RH_tbl2][:, 0]

                    # slow-freq averages for virtual theta computation
                    T1_mat  = simple_avg(np.column_stack([T_lev, t_lev]),
                                        freq_slow, return_timestamps=True)
                    RH1_mat = simple_avg(np.column_stack([RH_lev, t_lev]),
                                        freq_slow, return_timestamps=True)
                    ts1 = T1_mat[:, -1]
                    P_slow = np.interp(ts1,
                                       np.linspace(t.min(), t.max(), len(P_kPa_avg_lev)),
                                       P_kPa_avg_lev)
                    # If a paired physical HMP height is configured, use it for
                    # the altitude correction instead of the sonic height.
                    level_height = height
                    _level = sonic_for(info.get("sonics"), height)
                    if _level is not None:
                        if _level.hmp_height is not None:
                            level_height = _level.hmp_height
                    else:
                        # legacy parallel shiftsSonHeight/shiftsHMPHeight lists
                        for _sh, _hh in zip(info.get("shiftsSonHeight", []),
                                            info.get("shiftsHMPHeight", [])):
                            if abs(height - _sh) < 0.01:
                                level_height = _hh
                                break

                    vt_slow, r_slow, rho_moist_slow, rho_dry_slow, rho_H2O_slow = \
                        get_virtual_pot_temp(altitude, level_height - z_ref,
                                             T1_mat[:, 0], RH1_mat[:, 0],
                                             P_slow, use_p_elevation=False)

                    # average all slow-freq outputs to 30-min
                    def _avg30(arr):
                        m = simple_avg(np.column_stack([arr, ts1]),
                                       info["avgPer"], return_timestamps=False)
                        return m[:, 0] if m.shape[1] > 0 else np.full(N, np.nan)

                    vt_avg30        = _avg30(vt_slow)
                    r_avg30         = _avg30(r_slow)
                    rho_moist_avg30 = _avg30(rho_moist_slow)
                    rho_dry_avg30   = _avg30(rho_dry_slow)
                    rho_H2O_avg30   = _avg30(rho_H2O_slow)
                    virtual_theta_avg_local = vt_avg30

                    # 30-min q from HMP at this level → q_ref_fast_local (matches MATLAB)
                    T_avg_30_mat  = simple_avg(np.column_stack([T_lev, t_lev]),
                                               info["avgPer"], return_timestamps=True)
                    RH_avg_30_mat = simple_avg(np.column_stack([RH_lev, t_lev]),
                                               info["avgPer"], return_timestamps=True)
                    T_avg_30_K = T_avg_30_mat[:, 0].copy()
                    if np.nanmedian(T_avg_30_K) < 200:
                        T_avg_30_K = T_avg_30_K + 273.15
                    q_avg_30 = rh_to_spec_hum(RH_avg_30_mat[:, 0], P_kPa_avg_lev,
                                              T_avg_30_K)  # kg/kg, 30-min
                    valid_q = ~np.isnan(q_avg_30)
                    if valid_q.any():
                        t_q = RH_avg_30_mat[:, -1]
                        x_q = np.concatenate([[np.floor(t_q[valid_q][0])], t_q[valid_q]])
                        y_q = np.concatenate([[q_avg_30[valid_q][0]], q_avg_30[valid_q]])
                        q_ref_fast_local = np.interp(t, x_q, y_q)

                    # --- store in output["specificHum"] ---
                    def _pad(arr):
                        if len(arr) >= N:
                            return arr[:N]
                        return np.concatenate([arr, np.full(N - len(arr), np.nan)])

                    if "specificHum" not in output:
                        t_bp = np.array([t[min(bp[jj + 1] - 1, len(t) - 1)]
                                         for jj in range(N)])
                        output["specificHum"]       = t_bp.reshape(-1, 1)
                        output["specificHumHeader"] = ["time"]

                    q_col = _pad(q_avg_30 * 1000.0) if valid_q.any() else _pad(q_avg_30)
                    q_hdr = f"{height} m: q(g/kg)" if valid_q.any() else f"{height} m: q(g/g)"
                    nrows = output["specificHum"].shape[0]
                    for col_data, hdr in [
                        (q_col,                   q_hdr),
                        (_pad(vt_avg30),           f"{height} m: virtualThetaAvg(K)"),
                        (_pad(r_avg30),            f"{height} m: rAvg(g/kg)"),
                        (_pad(rho_moist_avg30),    f"{height} m: rho_airmoistAvg(kg/m^3)"),
                        (_pad(rho_dry_avg30),      f"{height} m: rho_airdryAvg(kg/m^3)"),
                        (_pad(rho_H2O_avg30),      f"{height} m: rho_H2OAvg(kg/m^3)"),
                    ]:
                        output["specificHum"] = np.column_stack([
                            output["specificHum"],
                            np.asarray(col_data)[:nrows].reshape(-1, 1),
                        ])
                        output["specificHumHeader"].append(hdr)

            # Use level-specific q for sonic air temperature (matches MATLAB qRefFastLocal)
            # Mean humidity rescale of the sonic temperature, T = T_s/(1 + 0.51 q)
            # (Schotanus et al. 1983 eq. 5; library/writeups/sonic_temperature_flux.md).
            # MATLAB fluxes.m:575 used the virtual-temperature 0.61 ("modified by Diane").
            sonic_coeff   = 0.61 if matlab_compat else SONIC_HUMIDITY_COEFF
            theta_son_air = (T_son + 273.15) / (1.0 + sonic_coeff * q_ref_fast_local) - 273.15

            # ---- fw sensor at this height ----
            fw_data  = None
            fw_flag  = np.zeros(N, dtype=bool)
            theta_fw = None
            Vtheta_fw = None
            if "fw" in sensor_info:
                fw_loc = _get_sensor_col(sensor_info, "fw", height)
                if fw_loc is not None:
                    fw_tbl, fw_col2 = fw_loc
                    fw_data = data[fw_tbl][:, fw_col2].copy()
                    nf_fw = output.get(f"{table_names[fw_tbl]}NanFlag",
                                       np.zeros((N, fw_col2 + 1), dtype=bool))
                    sf_fw = output.get(f"{table_names[fw_tbl]}SpikeFlag",
                                       np.zeros((N, fw_col2 + 1), dtype=bool))
                    fw_flag = (nf_fw[:, fw_col2] if fw_col2 < nf_fw.shape[1] else
                               np.zeros(N, dtype=bool)) | \
                              (sf_fw[:, fw_col2] if fw_col2 < sf_fw.shape[1] else
                               np.zeros(N, dtype=bool))
                    theta_fw  = fw_data + Gamma * (height - z_ref)
                    Vtheta_fw = theta_fw * (1.0 + 0.61 * q_ref_fast_local)   # MATLAB fluxes.m:566

            # ---- H2O sensor at this height ----
            h2o_data: Optional[np.ndarray] = None
            h2o_flag = np.zeros(N, dtype=bool)
            h2o_is_kh2o = False  # True when H2O comes from KH2O (krypton hygrometer)

            if "irgaH2O" in sensor_info:
                h2o_loc = _get_sensor_col(sensor_info, "irgaH2O", height)
                if h2o_loc is not None:
                    h2o_tbl, h2o_col = h2o_loc
                    h2o_data = data[h2o_tbl][:, h2o_col].copy()
                    nf_h = output.get(f"{table_names[h2o_tbl]}NanFlag",
                                      np.zeros((N, h2o_col + 1), dtype=bool))
                    sf_h = output.get(f"{table_names[h2o_tbl]}SpikeFlag",
                                      np.zeros((N, h2o_col + 1), dtype=bool))
                    h2o_nf = nf_h[:, h2o_col] if h2o_col < nf_h.shape[1] else np.zeros(N, bool)
                    h2o_sf = sf_h[:, h2o_col] if h2o_col < sf_h.shape[1] else np.zeros(N, bool)
                    h2o_sig = np.zeros(N, bool)
                    if "irgaH2OsigStrength" in sensor_info:
                        sig_loc = _get_sensor_col(sensor_info, "irgaH2OsigStrength", height)
                        if sig_loc is not None:
                            sig_tbl, sig_col = sig_loc
                            tn = table_names[sig_tbl]
                            if tn in output and sig_col < output[tn].shape[1]:
                                sv = output[tn][:, sig_col]
                                ok = ~np.isnan(sv)
                                h2o_sig[ok] = sv[ok] <= info["diagnosticTest"]["H2OminSignal"]
                    gd_flag = np.zeros(N, bool)
                    if "irgaGasDiag" in sensor_info:
                        gd_loc = _get_sensor_col(sensor_info, "irgaGasDiag", height)
                        if gd_loc is not None:
                            gd_tbl, gd_col = gd_loc
                            tn = table_names[gd_tbl]
                            if tn in output and gd_col < output[tn].shape[1]:
                                gv = output[tn][:, gd_col]
                                ok = ~np.isnan(gv)
                                gd_flag[ok] = gv[ok] >= info["diagnosticTest"]["meanGasDiagnosticLimit"]
                    h2o_flag = h2o_nf | h2o_sf | h2o_sig | gd_flag

            elif "LiH2O" in sensor_info:
                li_loc = _get_sensor_col(sensor_info, "LiH2O", height)
                if li_loc is not None:
                    li_tbl, li_col = li_loc
                    h2o_data = data[li_tbl][:, li_col] * 0.018  # mmol/m³ → g/m³
                    nf_l = output.get(f"{table_names[li_tbl]}NanFlag",
                                      np.zeros((N, li_col + 1), dtype=bool))
                    sf_l = output.get(f"{table_names[li_tbl]}SpikeFlag",
                                      np.zeros((N, li_col + 1), dtype=bool))
                    li_nf = nf_l[:, li_col] if li_col < nf_l.shape[1] else np.zeros(N, bool)
                    li_sf = sf_l[:, li_col] if li_col < sf_l.shape[1] else np.zeros(N, bool)
                    ld_flag = np.zeros(N, bool)
                    if "LiGasDiag" in sensor_info:
                        ld_loc = _get_sensor_col(sensor_info, "LiGasDiag", height)
                        if ld_loc is not None:
                            ld_tbl, ld_col = ld_loc
                            tn = table_names[ld_tbl]
                            if tn in output and ld_col < output[tn].shape[1]:
                                lv = output[tn][:, ld_col]
                                ok = ~np.isnan(lv)
                                ld_flag[ok] = lv[ok] > info["diagnosticTest"]["meanLiGasDiagnosticLimit"]
                    h2o_flag = li_nf | li_sf | ld_flag

            elif "KH2O" in sensor_info:
                kh_loc = _get_sensor_col(sensor_info, "KH2O", height)
                if kh_loc is not None:
                    kh_tbl, kh_col = kh_loc
                    h2o_data = data[kh_tbl][:, kh_col].copy()  # already g/m³
                    nf_k = output.get(f"{table_names[kh_tbl]}NanFlag",
                                      np.zeros((N, kh_col + 1), dtype=bool))
                    sf_k = output.get(f"{table_names[kh_tbl]}SpikeFlag",
                                      np.zeros((N, kh_col + 1), dtype=bool))
                    kh_nf = nf_k[:, kh_col] if kh_col < nf_k.shape[1] else np.zeros(N, bool)
                    kh_sf = sf_k[:, kh_col] if kh_col < sf_k.shape[1] else np.zeros(N, bool)
                    h2o_flag = kh_nf | kh_sf
                    h2o_is_kh2o = True

            # ---- CO2 sensor at this height ----
            co2_data: Optional[np.ndarray] = None
            co2_flag = np.zeros(N, bool)
            co2_gd   = np.zeros(N, bool)  # reused for CO2 gas-diag (shared with H2O for EC150)

            if "irgaCO2" in sensor_info and h2o_data is not None:
                co2_loc = _get_sensor_col(sensor_info, "irgaCO2", height)
                if co2_loc is not None:
                    co2_tbl, co2_col = co2_loc
                    co2_data = data[co2_tbl][:, co2_col].copy()  # mg/m³
                    nf_c = output.get(f"{table_names[co2_tbl]}NanFlag",
                                      np.zeros((N, co2_col + 1), dtype=bool))
                    sf_c = output.get(f"{table_names[co2_tbl]}SpikeFlag",
                                      np.zeros((N, co2_col + 1), dtype=bool))
                    co2_nf = nf_c[:, co2_col] if co2_col < nf_c.shape[1] else np.zeros(N, bool)
                    co2_sf = sf_c[:, co2_col] if co2_col < sf_c.shape[1] else np.zeros(N, bool)
                    co2_sig = np.zeros(N, bool)
                    if "irgaCO2sigStrength" in sensor_info:
                        cs_loc = _get_sensor_col(sensor_info, "irgaCO2sigStrength", height)
                        if cs_loc is not None:
                            cs_tbl, cs_col = cs_loc
                            tn = table_names[cs_tbl]
                            if tn in output and cs_col < output[tn].shape[1]:
                                cv = output[tn][:, cs_col]
                                ok = ~np.isnan(cv)
                                co2_sig[ok] = cv[ok] <= info["diagnosticTest"]["CO2minSignal"]
                    # reuse gas diag from H2O (same IRGA)
                    if "irgaGasDiag" in sensor_info:
                        gd_loc = _get_sensor_col(sensor_info, "irgaGasDiag", height)
                        if gd_loc is not None:
                            gd_tbl2, gd_col2 = gd_loc
                            tn2 = table_names[gd_tbl2]
                            if tn2 in output and gd_col2 < output[tn2].shape[1]:
                                gv2 = output[tn2][:, gd_col2]
                                ok2 = ~np.isnan(gv2)
                                co2_gd[ok2] = gv2[ok2] >= info["diagnosticTest"]["meanGasDiagnosticLimit"]
                    co2_flag = co2_nf | co2_sf | co2_sig | co2_gd

            elif "LiCO2" in sensor_info and h2o_data is not None:
                lco2_loc = _get_sensor_col(sensor_info, "LiCO2", height)
                if lco2_loc is not None:
                    lco2_tbl, lco2_col = lco2_loc
                    co2_data = data[lco2_tbl][:, lco2_col] * 44.0  # mmol/m³ → mg/m³
                    nf_lc = output.get(f"{table_names[lco2_tbl]}NanFlag",
                                       np.zeros((N, lco2_col + 1), dtype=bool))
                    sf_lc = output.get(f"{table_names[lco2_tbl]}SpikeFlag",
                                       np.zeros((N, lco2_col + 1), dtype=bool))
                    lco2_nf = nf_lc[:, lco2_col] if lco2_col < nf_lc.shape[1] else np.zeros(N, bool)
                    lco2_sf = sf_lc[:, lco2_col] if lco2_col < sf_lc.shape[1] else np.zeros(N, bool)
                    co2_flag = lco2_nf | lco2_sf

            # ---- initialise raw arrays on first sonic ----
            if save_raw and ii == 0:
                raw["uPF"]         = np.full((len(t), num_sonics), np.nan)
                raw["vPF"]         = np.full((len(t), num_sonics), np.nan)
                raw["wPF"]         = np.full((len(t), num_sonics), np.nan)
                raw["u_tilt"]      = np.full((len(t), num_sonics), np.nan)
                raw["v_tilt"]      = np.full((len(t), num_sonics), np.nan)
                raw["w_tilt"]      = np.full((len(t), num_sonics), np.nan)
                raw["WD"]          = np.full((len(t), num_sonics), np.nan)
                raw["spd"]         = np.full((len(t), num_sonics), np.nan)
                raw["sonTs"]       = np.full((len(t), num_sonics), np.nan)
                raw["Theta_v_son"] = np.full((len(t), num_sonics), np.nan)
                raw["t"]           = t
                raw["z"]           = np.full(num_sonics, np.nan)
                if "irgaH2O" in sensor_info or "LiH2O" in sensor_info:
                    nh = (sensor_info.get("irgaH2O", sensor_info.get("LiH2O"))).shape[0]
                    raw["rhov"]             = np.full((len(t), nh), np.nan)
                    raw["rhovPrime"]        = np.full((len(t), nh), np.nan)
                    raw["rhovextenalPrime"] = np.full((len(t), nh), np.nan)
                if "irgaCO2" in sensor_info or "LiCO2" in sensor_info:
                    nc = (sensor_info.get("irgaCO2", sensor_info.get("LiCO2"))).shape[0]
                    raw["rhoCO2"]             = np.full((len(t), nc), np.nan)
                    raw["rhoCO2Prime"]        = np.full((len(t), nc), np.nan)
                    raw["rhoCO2extenalPrime"] = np.full((len(t), nc), np.nan)

            if save_raw:
                raw["uPF"][:, ii]         = u_pf
                raw["vPF"][:, ii]         = v_pf
                raw["wPF"][:, ii]         = w_pf
                raw["u_tilt"][:, ii]      = pf_sonic_data[:, ru_col] if ru_col is not None else np.nan
                raw["v_tilt"][:, ii]      = pf_sonic_data[:, rv_col] if rv_col is not None else np.nan
                raw["w_tilt"][:, ii]      = w_tilt
                raw["sonTs"][:, ii]       = T_son
                raw["Theta_v_son"][:, ii] = theta_son
                raw["z"][ii]              = height
                bearing = float(sensor_info["u"][ii, 3]) if sensor_info["u"].shape[1] > 3 else 0.0
                if manufact == 0:    # RMYoung
                    u_wd, v_wd = v_raw, -u_raw
                elif manufact == 2:  # Gill
                    u_wd, v_wd = -u_raw, -v_raw
                else:                # Campbell CSAT3
                    u_wd, v_wd = u_raw, v_raw
                raw["WD"][:, ii]  = np.mod(np.degrees(np.arctan2(-v_wd, u_wd)) + bearing, 360.0)
                raw["spd"][:, ii] = np.sqrt(u_wd**2 + v_wd**2)

            # ---- derivedT: block-averaged derived temperatures ----
            avg_ts = simple_avg(np.column_stack([theta_son,     t]), info["avgPer"])[:, 0]
            avg_ta = simple_avg(np.column_stack([theta_son_air, t]), info["avgPer"])[:, 0]
            if has_fw and fw_data is not None and theta_fw is not None:
                avg_tfw  = simple_avg(np.column_stack([theta_fw,   t]), info["avgPer"])[:, 0]
                avg_vtfw = simple_avg(np.column_stack([Vtheta_fw, t]), info["avgPer"])[:, 0]
                derivedT_cols.append((avg_tfw,  f"{height} m: theta_fw"))
            derivedT_cols.append((avg_ts, f"{height} m: theta_v_son"))
            if has_fw and fw_data is not None and theta_fw is not None:
                derivedT_cols.append((avg_vtfw, f"{height} m: theta_v_fw"))
            derivedT_cols.append((avg_ta, f"{height} m: T_son_air"))

            # wind direction for H_SNSP
            dir_hdr = output.get("spdAndDirHeader", [])
            dir_col_idx = next((j for j, h in enumerate(dir_hdr)
                                if h == f"{height}m direction"), None)
            direction_avg = output["spdAndDir"][:, dir_col_idx] \
                            if dir_col_idx is not None else np.zeros(N)

            # =================================================================
            # Per-averaging-period loop
            # =================================================================
            det   = info.get("detrendingFormat", "linear")
            n_sub = info["avgPer"] // info.get("SSITC_subAvgMin", 5)
            d_ht  = float(info.get("displacementHeight", 0.0))

            for jj in range(N):
                s0, s1 = bp[jj], bp[jj + 1]
                if s0 >= s1:
                    continue

                ts_jj = t[s1 - 1]

                # ---- timestamp column (sonic 0 only) ----
                if ii == 0:
                    H_mat[jj, 0]    = tau_mat[jj, 0] = tke_mat[jj, 0]  = ts_jj
                    L_mat[jj, 0]    = sigma_mat[jj, 0] = R_mat[jj, 0]  = ts_jj
                    eta_mat[jj, 0]  = df_mat[jj, 0]   = dt_mat[jj, 0]  = ts_jj
                    turbtr_mat[jj, 0] = eps_mat[jj, 0] = skew_mat[jj, 0] = ts_jj
                    H_snsp[jj, 0]   = Flux_lat[jj, 0] = LH_mat[jj, 0]  = ts_jj
                    CO2fx_mat[jj, 0] = Hlat_mat[jj, 0] = fluxQC_mat[jj, 0] = ts_jj
                    # rho and cp in H columns 1,2 only (not in tke — matches MATLAB)
                    H_mat[jj, 1] = rho_avg[jj]        if jj < len(rho_avg)   else np.nan
                    H_mat[jj, 2] = 1004.67 * (1 + 0.84 * q_ref_avg[jj]) \
                                   if jj < len(q_ref_avg) else np.nan

                # ---- detrended perturbations ----
                uP    = nandetrend(u_raw[s0:s1],       det)
                vP    = nandetrend(v_raw[s0:s1],       det)
                wP    = nandetrend(w_raw[s0:s1],       det)
                TsP   = nandetrend(T_son[s0:s1],       det)
                ThvP  = nandetrend(theta_son[s0:s1],   det)
                TairP = nandetrend(theta_son_air[s0:s1], det)
                uPF_P = nandetrend(u_pf[s0:s1],        det)
                vPF_P = nandetrend(v_pf[s0:s1],        det)
                wPF_P = nandetrend(w_pf[s0:s1],        det)
                uTP   = nandetrend(u_tilt[s0:s1],      det)
                vTP   = nandetrend(v_tilt[s0:s1],      det)
                wTP   = nandetrend(w_tilt[s0:s1],      det)

                # ---- sigma ----
                c_sg = 1 + ii * num_sig_vars
                # Population std (ddof=0); MATLAB std uses N-1. At n ~ 36 000
                # per period the difference is < 2e-5 relative; kept as the
                # recorded divergence rather than changed (_ss_dev uses ddof=1).
                sigma_mat[jj, c_sg]     = np.nanstd(u_raw[s0:s1])
                sigma_mat[jj, c_sg + 1] = np.nanstd(v_raw[s0:s1])
                sigma_mat[jj, c_sg + 2] = np.nanstd(w_raw[s0:s1])
                sigma_mat[jj, c_sg + 3] = np.nanstd(u_pf[s0:s1])
                sigma_mat[jj, c_sg + 4] = np.nanstd(v_pf[s0:s1])
                sigma_mat[jj, c_sg + 5] = np.nanstd(w_pf[s0:s1])
                sigma_mat[jj, c_sg + 6] = np.nanstd(T_son[s0:s1])
                sigma_mat[jj, c_sg + 7] = np.nanmean(wPF_P * TsP ** 2)
                # Per-column flagging matching MATLAB fluxes.m lines 775-799:
                #   sigma_w uses unrotatedSonFlag (w only); sigma_Tson uses TsonFlag only.
                #   Applying rot_flag to all 8 columns would discard valid sigma_w when
                #   only u or v is bad, and discard valid sigma_Tson on wind spikes.
                if rot_flag[jj]:
                    sigma_mat[jj, c_sg]     = np.nan   # sigma_u
                    sigma_mat[jj, c_sg + 1] = np.nan   # sigma_v
                    sigma_mat[jj, c_sg + 3] = np.nan   # sigma_uPF
                    sigma_mat[jj, c_sg + 4] = np.nan   # sigma_vPF
                    sigma_mat[jj, c_sg + 5] = np.nan   # sigma_wPF
                    sigma_mat[jj, c_sg + 7] = np.nan   # wPFP_TsonP_TsonP
                if unrot_flag[jj]:
                    sigma_mat[jj, c_sg + 2] = np.nan   # sigma_w (w flag only)
                if Ts_flag[jj]:
                    sigma_mat[jj, c_sg + 6] = np.nan   # sigma_Tson (Tson flag only)
                    sigma_mat[jj, c_sg + 7] = np.nan   # wPFP_TsonP_TsonP

                # sigma_TFW at offset 12 (MATLAB col 14+(ii-1)*13) — only when fw present.
                # sigma_Theta_v goes at offset 8 unconditionally (written below).
                if has_fw and fw_data is not None:
                    sigma_mat[jj, c_sg + 12] = np.nanstd(fw_data[s0:s1])
                    if fw_flag[jj]:
                        sigma_mat[jj, c_sg + 12] = np.nan

                # ---- R: R(uPF'wPF', wPF'Thetav') ----
                c_R = 1 + ii * 16
                sigma_mat[jj, c_sg + 8] = np.nanstd(theta_son[s0:s1])   # sigma_Theta_v
                R_mat[jj, c_R] = _corr(uPF_P * wPF_P, ThvP * wPF_P)
                if Ts_flag[jj] or rot_flag[jj]:
                    R_mat[jj, c_R] = np.nan

                # ---- eta, delta_flux, delta_time: momentum and heat ----
                c_eta = 1 + ii * num_eta_vars
                c_df  = 1 + ii * num_df_vars
                c_dt  = 1 + ii * num_dt_vars

                eta_mat[jj, c_eta]     = find_eta(wPF_P, uPF_P)
                eta_mat[jj, c_eta + 1] = find_eta(wPF_P, ThvP)
                df_mat[jj,  c_df]      = find_delta_flux(wPF_P, uPF_P)
                df_mat[jj,  c_df + 1]  = find_delta_flux(wPF_P, ThvP)
                dt_mat[jj,  c_dt]      = find_delta_time(wPF_P, uPF_P)
                dt_mat[jj,  c_dt + 1]  = find_delta_time(wPF_P, ThvP)
                if Ts_flag[jj] or rot_flag[jj]:
                    eta_mat[jj, c_eta:c_eta + 2] = np.nan
                    df_mat[jj,  c_df:c_df + 2]   = np.nan
                    dt_mat[jj,  c_dt:c_dt + 2]   = np.nan

                # ---- tau ----
                c_tau = 1 + ii * num_tau_vars
                tau_mat[jj, c_tau]      = np.sqrt(np.nanmean(uP  * wP) ** 2 + np.nanmean(vP  * wP) ** 2)
                tau_mat[jj, c_tau + 1]  = np.sqrt(np.nanmean(uPF_P * wPF_P) ** 2 + np.nanmean(vPF_P * wPF_P) ** 2)
                tau_mat[jj, c_tau + 2]  = np.nanmean(uPF_P * uPF_P)
                tau_mat[jj, c_tau + 3]  = np.nanmean(vPF_P * vPF_P)
                tau_mat[jj, c_tau + 4]  = np.nanmean(wPF_P * wPF_P)
                tau_mat[jj, c_tau + 5]  = np.nanmean(uPF_P * vPF_P)
                tau_mat[jj, c_tau + 6]  = np.nanmean(uPF_P * wPF_P)
                tau_mat[jj, c_tau + 7]  = np.nanmean(vPF_P * wPF_P)
                tau_mat[jj, c_tau + 8]  = np.nanmean(uTP   * uTP)
                tau_mat[jj, c_tau + 9]  = np.nanmean(vTP   * vTP)
                tau_mat[jj, c_tau + 10] = np.nanmean(wTP   * wTP)
                tau_mat[jj, c_tau + 11] = np.nanmean(uTP   * vTP)
                tau_mat[jj, c_tau + 12] = np.nanmean(uTP   * wTP)
                tau_mat[jj, c_tau + 13] = np.nanmean(vTP   * wTP)
                if rot_flag[jj]:
                    tau_mat[jj, c_tau:c_tau + 14] = np.nan

                # ---- TKE ----
                tke_mat[jj, 1 + ii] = 0.5 * (np.nanmean(uP**2) + np.nanmean(vP**2) + np.nanmean(wP**2))
                if rot_flag[jj]:
                    tke_mat[jj, 1 + ii] = np.nan

                # ---- turbulent transport: w'e', w'u'w', w'Thv'w' ----
                c_tr = 1 + ii * num_tr_vars
                turbtr_mat[jj, c_tr]     = np.nanmean(wPF_P * 0.5 * (uPF_P**2 + vPF_P**2 + wPF_P**2))
                turbtr_mat[jj, c_tr + 1] = np.nanmean(wPF_P * (uPF_P * wPF_P))
                turbtr_mat[jj, c_tr + 2] = np.nanmean(wPF_P * (ThvP  * wPF_P))
                if rot_flag[jj]:
                    turbtr_mat[jj, c_tr:c_tr + 3] = np.nan

                # ---- dissipation ----
                if info.get("calcDissipation", False):
                    u_mean_jj = np.nanmean(u_pf[s0:s1])
                    freq      = info["tableScanFrequency"][tbl_idx]
                    eps_mat[jj, 1 + ii] = calc_dissipation_rate(uPF_P, u_mean_jj, 1.0 / freq)
                    if rot_flag[jj]:
                        eps_mat[jj, 1 + ii] = np.nan

                # ---- H_SNSP ----
                c_sn = 1 + ii * num_snsp_vars
                H_snsp[jj, c_sn]     = np.nanmean(uTP * ThvP)
                H_snsp[jj, c_sn + 1] = np.nanmean(vTP * ThvP)
                H_snsp[jj, c_sn + 2] = np.nanmean(wTP * ThvP)
                if jj < len(direction_avg):
                    a1, a2 = calc_snsp_angle(float(direction_avg[jj]), angle, downslope_aspect)
                    H_snsp[jj, c_sn + 3] = (
                        np.nanmean(wTP * ThvP) * np.cos(np.radians(angle))
                        - np.nanmean(uTP * ThvP) * np.sin(np.radians(a1))
                        - np.nanmean(vTP * ThvP) * np.sin(np.radians(a2))
                    )
                if rot_flag[jj] or Ts_flag[jj]:
                    H_snsp[jj, c_sn:c_sn + 4] = np.nan

                # ---- H: Ts'w', Thetav'wPF' ----
                c_H = 3 + ii * 12
                H_mat[jj, c_H]     = np.nanmean(wP    * TsP)
                H_mat[jj, c_H + 1] = np.nanmean(wPF_P * ThvP)   # buoyancy flux (MATLAB kinSenFlux)
                if unrot_flag[jj] or Ts_flag[jj]:
                    H_mat[jj, c_H] = np.nan
                if rot_flag[jj] or Ts_flag[jj]:
                    H_mat[jj, c_H + 1] = np.nan

                # ---- Obukhov length ----
                kappa = 0.4;  g = 9.81
                T0_L = (virtual_theta_avg_local[jj]
                        if virtual_theta_avg_local is not None and jj < len(virtual_theta_avg_local)
                        else T_virt_ref_K_avg[jj] if jj < len(T_virt_ref_K_avg) else np.nan)
                u_star_cubed = tau_mat[jj, c_tau + 1] ** (3.0 / 2.0) \
                               if not np.isnan(tau_mat[jj, c_tau + 1]) else np.nan
                H0_L = H_snsp[jj, c_sn + 3]
                if not (np.isnan(u_star_cubed) or np.isnan(H0_L) or H0_L == 0 or np.isnan(T0_L)):
                    L_mat[jj, 1 + ii] = -u_star_cubed / (kappa * g / T0_L * H0_L)
                if rot_flag[jj] or Ts_flag[jj]:
                    L_mat[jj, 1 + ii] = np.nan

                # ---- H: T_air'w', T_air'wPF' ----
                H_mat[jj, c_H + 2] = np.nanmean(wP    * TairP)
                H_mat[jj, c_H + 3] = np.nanmean(wPF_P * TairP)
                if unrot_flag[jj] or Ts_flag[jj]:
                    H_mat[jj, c_H + 2] = np.nan
                if rot_flag[jj] or Ts_flag[jj]:
                    H_mat[jj, c_H + 3] = np.nan

                # ---- H: Th_s'w', Th_s'wPF'  (theta_son = thetaSon) ----
                H_mat[jj, c_H + 6] = np.nanmean(wP    * ThvP)
                H_mat[jj, c_H + 7] = np.nanmean(wPF_P * ThvP)
                if unrot_flag[jj] or Ts_flag[jj]:
                    H_mat[jj, c_H + 6] = np.nan
                if rot_flag[jj] or Ts_flag[jj]:
                    H_mat[jj, c_H + 7] = np.nan

                # ---- Hlat: Ts'u', Ts'uPF', Ts'v', Ts'vPF' ----
                c_hl = 1 + ii * 12
                Hlat_mat[jj, c_hl]     = np.nanmean(uP    * TsP)
                Hlat_mat[jj, c_hl + 1] = np.nanmean(uPF_P * TsP)
                Hlat_mat[jj, c_hl + 2] = np.nanmean(vP    * TsP)
                Hlat_mat[jj, c_hl + 3] = np.nanmean(vPF_P * TsP)
                if rot_flag[jj] or Ts_flag[jj]:
                    Hlat_mat[jj, c_hl:c_hl + 4] = np.nan

                # ---- Hlat: Th_s'u', Th_s'uPF', Th_s'v', Th_s'vPF' ----
                Hlat_mat[jj, c_hl + 4] = np.nanmean(uP    * ThvP)
                Hlat_mat[jj, c_hl + 5] = np.nanmean(uPF_P * ThvP)
                Hlat_mat[jj, c_hl + 6] = np.nanmean(vP    * ThvP)
                Hlat_mat[jj, c_hl + 7] = np.nanmean(vPF_P * ThvP)
                if rot_flag[jj] or Ts_flag[jj]:
                    Hlat_mat[jj, c_hl + 4:c_hl + 8] = np.nan

                # ---- fw-based fluxes ----
                if fw_data is not None:
                    fwP    = nandetrend(fw_data[s0:s1],    det)
                    thFwP  = nandetrend(theta_fw[s0:s1],   det)
                    VthFwP = nandetrend(Vtheta_fw[s0:s1],  det)

                    H_mat[jj, c_H + 4]  = np.nanmean(wP    * fwP)
                    H_mat[jj, c_H + 5]  = np.nanmean(wPF_P * fwP)
                    H_mat[jj, c_H + 8]  = np.nanmean(wP    * thFwP)
                    H_mat[jj, c_H + 9]  = np.nanmean(wPF_P * thFwP)
                    H_mat[jj, c_H + 10] = np.nanmean(wP    * VthFwP)
                    H_mat[jj, c_H + 11] = np.nanmean(wPF_P * VthFwP)
                    if unrot_flag[jj] or fw_flag[jj]:
                        H_mat[jj, [c_H+4, c_H+8, c_H+10]] = np.nan
                    if rot_flag[jj] or fw_flag[jj]:
                        H_mat[jj, [c_H+5, c_H+9, c_H+11]] = np.nan

                    Hlat_mat[jj, c_hl + 8]  = np.nanmean(uP    * thFwP)
                    Hlat_mat[jj, c_hl + 9]  = np.nanmean(uPF_P * thFwP)
                    Hlat_mat[jj, c_hl + 10] = np.nanmean(vP    * thFwP)
                    Hlat_mat[jj, c_hl + 11] = np.nanmean(vPF_P * thFwP)
                    if rot_flag[jj] or fw_flag[jj]:
                        Hlat_mat[jj, c_hl + 8:c_hl + 12] = np.nan

                    if save_raw:
                        if "fwThPrime" not in raw:
                            raw["fwThPrime"] = np.full((len(t), num_sonics), np.nan)
                            raw["fwTh"]      = np.full((len(t), num_sonics), np.nan)
                            raw["fwT"]       = np.full((len(t), num_sonics), np.nan)
                        raw["fwThPrime"][s0:s1, ii] = thFwP
                        raw["fwTh"][s0:s1, ii]      = theta_fw[s0:s1]
                        raw["fwT"][s0:s1, ii]        = fw_data[s0:s1]

                # ---- raw sonic/temperature storage ----
                if save_raw:
                    raw["sonTs"][s0:s1, ii]       = T_son[s0:s1]
                    raw["Theta_v_son"][s0:s1, ii] = theta_son[s0:s1]

                # ====================================================================
                # H2O-based fluxes (LH, H2O sigma/eta/R, skew, Flux_lat, turbtr)
                # ====================================================================
                if h2o_data is not None:
                    H2Op = nandetrend(h2o_data[s0:s1], det)

                    rho_v_j = rho_v_avg[jj] if jj < len(rho_v_avg) else np.nan
                    rho_d_j = rho_d_avg[jj] if jj < len(rho_d_avg) else np.nan
                    T_ref_j = T_ref_K_avg[jj] if jj < len(T_ref_K_avg) else np.nan

                    # Schotanus correction with high-frequency humidity:
                    # T' = Ts' - 0.51 T q' (Schotanus 1983 eq. 6; Liu et al. 2001
                    # eq. 10), so w'T' = w'Ts' - 0.51 T w'q' (eq. 8 / eq. 12). This
                    # replaces the mean-humidity rescale for the T_air columns and
                    # is the temperature flux every WPL term below uses. Under
                    # matlabCompat the rescaled TairP and the buoyancy flux stay.
                    if not matlab_compat:
                        qP = (H2Op / 1000.0) / (rho_d_j + rho_v_j)          # kg/kg
                        T_air_mean_K = np.nanmean(theta_son_air[s0:s1]) + 273.15
                        TairP = air_temperature_perturbation(TsP, qP, T_air_mean_K)
                        H_mat[jj, c_H + 2] = np.nanmean(wP    * TairP)
                        H_mat[jj, c_H + 3] = np.nanmean(wPF_P * TairP)
                        if unrot_flag[jj] or Ts_flag[jj] or h2o_flag[jj]:
                            H_mat[jj, c_H + 2] = np.nan
                        if rot_flag[jj] or Ts_flag[jj] or h2o_flag[jj]:
                            H_mat[jj, c_H + 3] = np.nan

                    # WPL external H2O fluctuation
                    rhov_ext = (Md / Mv * (rho_v_j / rho_d_j) * (H2Op / 1000.0)
                                + rho_v_j * (1.0 + Md / Mv * rho_v_j / rho_d_j)
                                * TairP / T_ref_j)

                    # sigma H2O
                    sigma_mat[jj, c_sg + 9] = np.nanstd(h2o_data[s0:s1])
                    if Ts_flag[jj] or rot_flag[jj]:
                        sigma_mat[jj, c_sg + 9] = np.nan

                    # R H2O — order matches MATLAB: pair H2O at +1,2; individual at +5,6,7
                    R_mat[jj, c_R + 1] = _corr(uPF_P * wPF_P, H2Op * wPF_P)
                    R_mat[jj, c_R + 2] = _corr(H2Op  * wPF_P, ThvP * wPF_P)
                    R_mat[jj, c_R + 5] = _corr(wPF_P, uPF_P)
                    R_mat[jj, c_R + 6] = _corr(wPF_P, ThvP)
                    R_mat[jj, c_R + 7] = _corr(wPF_P, H2Op)
                    if Ts_flag[jj] or rot_flag[jj]:
                        R_mat[jj, c_R + 1:c_R + 3] = np.nan
                        R_mat[jj, c_R + 5:c_R + 8] = np.nan

                    # eta H2O, delta_flux H2O, delta_time H2O
                    eta_mat[jj, c_eta + 2] = find_eta(wPF_P, H2Op)
                    eta_mat[jj, c_eta + 3] = find_eta(wPF_P, H2Op + rhov_ext * 1e3)
                    df_mat[jj,  c_df + 2]  = find_delta_flux(wPF_P, H2Op)
                    df_mat[jj,  c_df + 3]  = find_delta_flux(wPF_P, H2Op + rhov_ext * 1e3)
                    dt_mat[jj,  c_dt + 2]  = find_delta_time(wPF_P, H2Op)
                    dt_mat[jj,  c_dt + 3]  = find_delta_time(wPF_P, H2Op + rhov_ext * 1e3)
                    if Ts_flag[jj] or rot_flag[jj]:
                        eta_mat[jj, c_eta + 2:c_eta + 4] = np.nan
                        df_mat[jj,  c_df  + 2:c_df  + 4] = np.nan
                        dt_mat[jj,  c_dt  + 2:c_dt  + 4] = np.nan

                    # LHflux (all-NaN periods produce nan silently)
                    with np.errstate(all="ignore"):
                        E   = np.nanmean(wP    * H2Op)
                        EPF = np.nanmean(wPF_P * H2Op)
                    Lv  = (2.501 - 0.00237 * (T_ref_j - 273.15)) * 1e3
                    # Temperature flux for the WPL terms (Webb et al. 1980 eqs. 24-25,
                    # 44): w'T', i.e. the Schotanus-corrected T_air'wPF' column.
                    # MATLAB (fluxes.m:1073) used the buoyancy flux Theta_v'wPF'.
                    kin_sen_flux = H_mat[jj, c_H + 1] if matlab_compat else H_mat[jj, c_H + 3]
                    wpl = 1.0 + Md / Mv * rho_v_j / rho_d_j

                    c_lh = 1 + ii * num_LH_vars
                    LH_mat[jj, c_lh]     = Lv
                    LH_mat[jj, c_lh + 1] = E
                    if unrot_flag[jj] or h2o_flag[jj]:
                        LH_mat[jj, c_lh + 1] = np.nan
                    LH_mat[jj, c_lh + 2] = EPF
                    if rot_flag[jj] or h2o_flag[jj]:
                        LH_mat[jj, c_lh + 2] = np.nan
                    LH_mat[jj, c_lh + 3] = wpl * (np.nanmean(wPF_P * H2Op) / 1e3
                                                    + rho_v_j / T_ref_j * np.nanmean(wPF_P * TairP))
                    if rot_flag[jj] or h2o_flag[jj]:
                        LH_mat[jj, c_lh + 3] = np.nan
                    LH_mat[jj, c_lh + 4] = (1000.0 * Lv * wpl
                                             * (E   / 1000.0 + rho_v_j / T_ref_j * kin_sen_flux))
                    if unrot_flag[jj] or h2o_flag[jj]:
                        LH_mat[jj, c_lh + 4] = np.nan
                    LH_mat[jj, c_lh + 5] = (1000.0 * Lv * wpl
                                             * (EPF / 1000.0 + rho_v_j / T_ref_j * kin_sen_flux))
                    if unrot_flag[jj] or h2o_flag[jj]:
                        LH_mat[jj, c_lh + 5] = np.nan

                    # KH2O O2 correction (Tanner et al. 1993)
                    if h2o_is_kh2o:
                        ko = -0.0045; kw = -0.153
                        CkO = 0.23 * ko / kw
                        O2_corr = CkO * rho_d_j / T_ref_j * kin_sen_flux * 1000.0
                        E   += O2_corr
                        EPF += O2_corr
                        LH_mat[jj, c_lh + 6] = Lv * E
                        LH_mat[jj, c_lh + 7] = Lv * EPF

                    # raw H2O
                    if save_raw and "rhov" in raw:
                        h2o_si = 0
                        if "irgaH2O" in sensor_info:
                            rows = sensor_info["irgaH2O"][:, 2] == height
                            if rows.any():
                                h2o_si = int(np.where(rows)[0][0])
                        raw["rhov"][s0:s1, h2o_si]             = h2o_data[s0:s1]
                        raw["rhovPrime"][s0:s1, h2o_si]        = H2Op
                        raw["rhovextenalPrime"][s0:s1, h2o_si] = rhov_ext * 1e3

                    # ================================================================
                    # CO2-based fluxes (requires H2O to exist first for evapFlux)
                    # ================================================================
                    if co2_data is not None:
                        rho_CO2_seg  = co2_data[s0:s1]        # mg/m³
                        rho_CO2p     = nandetrend(rho_CO2_seg, det) / 1e6  # kg/m³
                        rho_CO2_avg  = np.nanmean(rho_CO2_seg) / 1e6       # kg/m³
                        evap_flux    = LH_mat[jj, c_lh + 5] / Lv / 1000.0  # WPL wPF'' (kg/m²/s)

                        # WPL external CO2 fluctuation
                        rhoc_ext = (Md / Mv * (rho_CO2_avg / rho_d_j) * (H2Op / 1000.0)
                                    + rho_CO2_avg * (1.0 + Md / Mv * rho_v_j / rho_d_j)
                                    * TairP / T_ref_j)

                        # ppm CO2 (period mean) — use height-specific per-sample pressure
                        if P_raw_hf_lev is not None:
                            P_seg_kPa = np.interp(t[s0:s1], P_t_hf_lev, P_raw_hf_lev)
                        elif jj < len(P_kPa_avg_lev):
                            P_seg_kPa = np.full(s1 - s0, P_kPa_avg_lev[jj])
                        else:
                            P_seg_kPa = np.full(s1 - s0, P_ref_kPa)
                        ppm_CO2 = np.nanmean(rho_CO2_seg * 1000.0 * 8.314
                                             * (theta_son_air[s0:s1] + 273.15)
                                             / (44.01 * P_seg_kPa * 1e3))

                        # sigma CO2, CO2 WPL
                        sigma_mat[jj, c_sg + 10] = np.nanstd(rho_CO2_seg)
                        sigma_mat[jj, c_sg + 11] = np.nanstd(rho_CO2_seg / 1e6
                                                              + rhoc_ext) * 1e6
                        if Ts_flag[jj] or rot_flag[jj]:
                            sigma_mat[jj, c_sg + 10:c_sg + 12] = np.nan

                        # R CO2 — order matches MATLAB cols 5-17:
                        # c_R+3,4 = CO2_WPL pair; c_R+8..12 = individual/raw; c_R+13..15 = H2O_WPL
                        R_mat[jj, c_R + 3]  = _corr(uPF_P * wPF_P, (rho_CO2p + rhoc_ext) * wPF_P)
                        R_mat[jj, c_R + 4]  = _corr((rho_CO2p + rhoc_ext) * wPF_P, ThvP * wPF_P)
                        R_mat[jj, c_R + 8]  = _corr(wPF_P, rho_CO2p)
                        R_mat[jj, c_R + 9]  = _corr(wPF_P, rho_CO2p + rhoc_ext)
                        R_mat[jj, c_R + 10] = _corr(uPF_P * wPF_P, rho_CO2p * wPF_P)
                        R_mat[jj, c_R + 11] = _corr(rho_CO2p * wPF_P, ThvP * wPF_P)
                        R_mat[jj, c_R + 12] = _corr(wPF_P, rho_CO2p)  # duplicate (MATLAB col 14)
                        R_mat[jj, c_R + 13] = _corr(uPF_P * wPF_P,
                                                     (H2Op + rhov_ext * 1e3) * wPF_P)
                        R_mat[jj, c_R + 14] = _corr((H2Op + rhov_ext * 1e3) * wPF_P,
                                                     ThvP * wPF_P)
                        R_mat[jj, c_R + 15] = _corr(wPF_P, H2Op + rhov_ext * 1e3)
                        if Ts_flag[jj] or rot_flag[jj]:
                            R_mat[jj, c_R + 3:c_R + 5] = np.nan
                            R_mat[jj, c_R + 8:c_R + 16] = np.nan

                        # eta, delta_flux, delta_time CO2
                        eta_mat[jj, c_eta + 4] = find_eta(wPF_P, rho_CO2p + rhoc_ext)
                        eta_mat[jj, c_eta + 5] = find_eta(wPF_P, rho_CO2p)
                        df_mat[jj,  c_df  + 4] = find_delta_flux(wPF_P, rho_CO2p + rhoc_ext)
                        df_mat[jj,  c_df  + 5] = find_delta_flux(wPF_P, rho_CO2p)
                        dt_mat[jj,  c_dt  + 4] = find_delta_time(wPF_P, rho_CO2p + rhoc_ext)
                        dt_mat[jj,  c_dt  + 5] = find_delta_time(wPF_P, rho_CO2p)
                        if Ts_flag[jj] or rot_flag[jj]:
                            eta_mat[jj, c_eta + 4:c_eta + 6] = np.nan
                            df_mat[jj,  c_df  + 4:c_df  + 6] = np.nan
                            dt_mat[jj,  c_dt  + 4:c_dt  + 6] = np.nan

                        # skewness
                        c_sk = 1 + ii * num_sk_vars
                        skew_mat[jj, c_sk]     = _skew(uPF_P)
                        skew_mat[jj, c_sk + 1] = _skew(vPF_P)
                        skew_mat[jj, c_sk + 2] = _skew(wPF_P)
                        skew_mat[jj, c_sk + 3] = _skew(ThvP)
                        skew_mat[jj, c_sk + 4] = _skew(H2Op)
                        skew_mat[jj, c_sk + 5] = _skew(H2Op + rhov_ext * 1e3)
                        skew_mat[jj, c_sk + 6] = _skew(rho_CO2p)
                        skew_mat[jj, c_sk + 7] = _skew(rho_CO2p + rhoc_ext)
                        if rot_flag[jj]:
                            skew_mat[jj, c_sk:c_sk + 8] = np.nan

                        # lateral flux contributions
                        c_fl = 1 + ii * num_fl_vars
                        Flux_lat[jj, c_fl]     = np.nanmean(uPF_P * ThvP)
                        Flux_lat[jj, c_fl + 1] = np.nanmean(uPF_P * H2Op)
                        Flux_lat[jj, c_fl + 2] = np.nanmean(uPF_P * (H2Op + rhov_ext * 1e3))
                        Flux_lat[jj, c_fl + 3] = np.nanmean(uPF_P * rho_CO2p)
                        Flux_lat[jj, c_fl + 4] = np.nanmean(uPF_P * (rho_CO2p + rhoc_ext))
                        if rot_flag[jj]:
                            Flux_lat[jj, c_fl:c_fl + 5] = np.nan

                        # turbulent transport H2O and CO2
                        turbtr_mat[jj, c_tr + 3] = np.nanmean(wPF_P * (H2Op             * wPF_P))
                        turbtr_mat[jj, c_tr + 4] = np.nanmean(wPF_P * ((H2Op + rhov_ext * 1e3) * wPF_P))
                        turbtr_mat[jj, c_tr + 5] = np.nanmean(wPF_P * (rho_CO2p         * wPF_P))
                        turbtr_mat[jj, c_tr + 6] = np.nanmean(wPF_P * ((rho_CO2p + rhoc_ext) * wPF_P))
                        if rot_flag[jj]:
                            turbtr_mat[jj, c_tr + 3:c_tr + 7] = np.nan

                        # CO2 flux
                        c_co2 = 1 + ii * num_CO2_vars
                        CO2fx_mat[jj, c_co2]     = np.nanmean(wP    * rho_CO2p)
                        if unrot_flag[jj] or co2_flag[jj]:
                            CO2fx_mat[jj, c_co2] = np.nan
                        CO2fx_mat[jj, c_co2 + 1] = np.nanmean(wPF_P * rho_CO2p)
                        if rot_flag[jj] or co2_flag[jj]:
                            CO2fx_mat[jj, c_co2 + 1] = np.nan
                        CO2fx_mat[jj, c_co2 + 2] = (CO2fx_mat[jj, c_co2 + 1]
                                                     + Md / Mv * (rho_CO2_avg / rho_d_j) * evap_flux
                                                     + (1.0 + Md / Mv * rho_v_j / rho_d_j)
                                                     * (rho_CO2_avg / T_ref_j) * kin_sen_flux)
                        if rot_flag[jj] or co2_flag[jj]:
                            CO2fx_mat[jj, c_co2 + 2] = np.nan
                        CO2fx_mat[jj, c_co2 + 3] = (CO2fx_mat[jj, c_co2 + 1]
                                                     + np.nanmean(wPF_P * rhoc_ext))
                        if rot_flag[jj] or co2_flag[jj]:
                            CO2fx_mat[jj, c_co2 + 3] = np.nan
                        CO2fx_mat[jj, c_co2 + 4] = ppm_CO2

                        # raw CO2
                        # Unit convention (Python vs MATLAB difference):
                        #   rhoCO2            — mg/m³  in both Python and MATLAB
                        #   rhoCO2Prime       — mg/m³  in Python  (rho_CO2p * 1e6)
                        #                       kg/m³  in MATLAB  (fluxes.m line 1212, no reconversion)
                        #   rhoCO2extenalPrime— mg/m³  in Python  (rhoc_ext * 1e6)
                        #                       kg/m³  in MATLAB  (same pattern)
                        # The WPL computation internally uses kg/m³ (rho_CO2p = ... / 1e6).
                        # Python reconverts to mg/m³ for storage so that rhoCO2Prime is
                        # in the same units as rhoCO2 and is consistent with rhov/rhovPrime
                        # (both in g/m³). MATLAB omits the reconversion step.
                        if save_raw and "rhoCO2" in raw:
                            co2_si = 0
                            if "irgaCO2" in sensor_info:
                                rows = sensor_info["irgaCO2"][:, 2] == height
                                if rows.any():
                                    co2_si = int(np.where(rows)[0][0])
                            raw["rhoCO2"][s0:s1, co2_si]             = rho_CO2_seg
                            raw["rhoCO2Prime"][s0:s1, co2_si]        = rho_CO2p * 1e6
                            raw["rhoCO2extenalPrime"][s0:s1, co2_si] = rhoc_ext * 1e6

                # ---- SSITC + SS-only quality flags (ForestComplexTerrain) ----
                ustar_jj = (np.sqrt(tau_mat[jj, c_tau + 1])
                            if not np.isnan(tau_mat[jj, c_tau + 1])
                            and tau_mat[jj, c_tau + 1] >= 0 else np.nan)
                _H2Op = H2Op     if h2o_data is not None else None
                _rhov = rhov_ext if h2o_data is not None else None
                _co2p = rho_CO2p if (h2o_data is not None and co2_data is not None) else None
                _rhoc = rhoc_ext if (h2o_data is not None and co2_data is not None) else None
                (tau_ssitc, tau_ss, H_ssitc, H_ss,
                 LE_ssitc,  LE_ss,  FC_ssitc, FC_ss) = calc_ssitc_flags(
                    wPF_P, uPF_P, vPF_P, ThvP,
                    _H2Op, _rhov, _co2p, _rhoc,
                    ustar_jj, L_mat[jj, 1 + ii],
                    bool(rot_flag[jj]), bool(Ts_flag[jj]),
                    bool(h2o_flag[jj]), bool(co2_flag[jj]),
                    n_sub, height, d_ht,
                    canopy_height=float(info.get("canopyHeight", np.nan)),
                    use_canopy_itc=bool(info.get("useCanopyITC", True)),
                    latitude=info.get("latitude"),
                )
                c_qc = 1 + ii * num_qc_vars
                fluxQC_mat[jj, c_qc]     = tau_ssitc
                fluxQC_mat[jj, c_qc + 1] = tau_ss
                fluxQC_mat[jj, c_qc + 2] = H_ssitc
                fluxQC_mat[jj, c_qc + 3] = H_ss
                fluxQC_mat[jj, c_qc + 4] = LE_ssitc
                fluxQC_mat[jj, c_qc + 5] = LE_ss
                fluxQC_mat[jj, c_qc + 6] = FC_ssitc
                fluxQC_mat[jj, c_qc + 7] = FC_ss

        except Exception as exc:
            import warnings
            warnings.warn(f"Flux computation failed at height {height}m: {exc}")
            import traceback; traceback.print_exc()

    # -----------------------------------------------------------------------
    # Store outputs — trim all-NaN columns, generate MATLAB-matching headers.
    # -----------------------------------------------------------------------
    store_extra = bool(info.get("storeExtraStats", True))

    def _trim_both(mat, hdr):
        keep = np.any(~np.isnan(mat), axis=0)
        return mat[:, keep], [h for h, k in zip(hdr, keep) if k]

    def _h(height):
        return f"{height:g}"

    sonic_heights = [float(sensor_info["u"][ii, 2]) for ii in range(num_sonics)]

    # H  (3 fixed + 12 per sonic)
    H_hdr = ["time", "rho", "cp"]
    for h in sonic_heights:
        hn = _h(h)
        H_hdr += [
            f"{hn}m son:Ts'w'",      f"{hn}m son:Theta_v'wPF'",
            f"{hn}m son:T_air'w'",   f"{hn}m son:T_air'wPF'",
            f"{hn}m fw:T'w'",        f"{hn}m fw:T'wPF'",
            f"{hn}m son:Th_s'w'",    f"{hn}m son:Th_s'wPF'",
            f"{hn}m fw:Th'w'",       f"{hn}m fw:Th'wPF'",
            f"{hn}m fw:VTh'w'",      f"{hn}m fw:VTh'wPF'",
        ]
    # Hlat  (1 + 12 per sonic)
    Hlat_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        Hlat_hdr += [
            f"{hn}m son:Ts'u'",   f"{hn}m son:Ts'uPF'",
            f"{hn}m son:Ts'v'",   f"{hn}m son:Ts'vPF'",
            f"{hn}m son:Th_s'u'", f"{hn}m son:Th_s'uPF'",
            f"{hn}m son:Th_s'v'", f"{hn}m son:Th_s'vPF'",
            f"{hn}m fw:Th'u'",    f"{hn}m fw:Th'uPF'",
            f"{hn}m fw:Th'v'",    f"{hn}m fw:Th'vPF'",
        ]
    # tau  (1 + 14 per sonic)
    tau_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        tau_hdr += [
            f"{hn}m :sqrt(u'w'^2+v'w'^2)",
            f"{hn}m :sqrt(uPF'wPF'^2+vPF'wPF'^2)",
            f"{hn}m :uPF'uPF'",   f"{hn}m :vPF'vPF'",   f"{hn}m :wPF'wPF'",
            f"{hn}m :uPF'vPF'",   f"{hn}m :uPF'wPF'",   f"{hn}m :vPF'wPF'",
            f"{hn}m :uTilt'uTilt'", f"{hn}m :vTilt'vTilt'", f"{hn}m :wTilt'wTilt'",
            f"{hn}m :uTilt'vTilt'", f"{hn}m :uTilt'wTilt'", f"{hn}m :vTilt'wTilt'",
        ]
    # tke  (1 + 1 per sonic)
    tke_hdr = ["time"] + [f"{_h(h)}m :0.5(u'^2+v'^2+w'^2)" for h in sonic_heights]
    # sigma  (1 + 12 per sonic without fw; 1 + 13 per sonic with fw)
    sigma_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        sigma_hdr += [
            f"{hn}m :sigma_u",   f"{hn}m :sigma_v",   f"{hn}m :sigma_w",
            f"{hn}m :sigma_uPF", f"{hn}m :sigma_vPF", f"{hn}m :sigma_wPF",
            f"{hn}m :sigma_Tson",
            f"{hn}m :wPFP_TsonP_TsonP",
            f"{hn}m :sigma_Theta_v",
            f"{hn}m :sigma_H2O", f"{hn}m :sigma_CO2", f"{hn}m :sigma_CO2_WPL",
        ]
        if has_fw:
            sigma_hdr.append(f"{hn}m :sigma_TFW")
    # R  (1 + 16 per sonic)
    R_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        R_hdr += [
            f"{hn}m :R_uPFwPF_wPFThetav",
            f"{hn}m :R_uPFwPF_wPFH2O",
            f"{hn}m :R_wPFH2O_wPFThetav",
            f"{hn}m :R_uPFwPF_wPFCO2_WPL",
            f"{hn}m :R_wPFCO2_WPL_wPFThetav",
            f"{hn}m :R_wPF_uPF",
            f"{hn}m :R_wPF_Theta_v",
            f"{hn}m :R_wPF_H2O",
            f"{hn}m :R_wPF_CO2",
            f"{hn}m :R_wPF_CO2_WPL",
            f"{hn}m :R_uPFwPF_wPFCO2",
            f"{hn}m :R_wPFCO2_wPFThetav",
            f"{hn}m :R_wPF_CO2",              # duplicate (MATLAB col 14)
            f"{hn}m :R_uPFwPF_wPFH2O_WPL",
            f"{hn}m :R_wPFH2O_WPL_wPFThetav",
            f"{hn}m :R_wPF_H2O_WPL",
        ]
    # L  (1 + 1 per sonic)
    L_hdr = ["time"] + [
        f"{_h(h)}m L:sqrt(uPF'*wPF'+vPF'*wPF')^3/2*Th_v/(k*g*wThv_vert)"
        for h in sonic_heights]
    # eta  (1 + 6 per sonic)
    eta_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        eta_hdr += [
            f"{hn}m :eta_wPFuPF",   f"{hn}m :eta_wPFThetav",
            f"{hn}m :eta_wPFH2O",   f"{hn}m :eta_wPFH2O_WPL",
            f"{hn}m :eta_wPFCO2_WPL", f"{hn}m :eta_wPFCO2",
        ]
    # delta_flux_ctrb  (1 + 6 per sonic)
    df_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        df_hdr += [
            f"{hn}m :S_wPFuPF",    f"{hn}m :S_wPFThetav",
            f"{hn}m :S_wPFH2O",    f"{hn}m :S_wPFH2O_WPL",
            f"{hn}m :S_wPFCO2_WPL", f"{hn}m :S_wPFCO2",
        ]
    # delta_time_ctrb  (1 + 6 per sonic)
    dt_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        dt_hdr += [
            f"{hn}m :D_wPFuPF",    f"{hn}m :D_wPFThetav",
            f"{hn}m :D_wPFH2O",    f"{hn}m :D_wPFH2O_WPL",
            f"{hn}m :D_wPFCO2_WPL", f"{hn}m :D_wPFCO2",
        ]
    # turbtr  (1 + 7 per sonic)
    turbtr_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        turbtr_hdr += [
            f"{hn}m :mean(w'e')",
            f"{hn}m :mean(w'u'w')",
            f"{hn}m :mean(w'theta_v'w')",
            f"{hn}m :mean(w'H2O'w')",
            f"{hn}m :mean(w'H2O_WPL'w')",
            f"{hn}m :mean(w'CO2'w')",
            f"{hn}m :mean(w'CO2_WPL'w')",
        ]
    # epsilon  (1 + 1 per sonic)
    eps_hdr = ["time"] + [f"{_h(h)}m :epsilon" for h in sonic_heights]
    # skew  (1 + 8 per sonic)
    skew_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        skew_hdr += [
            f"{hn}m :skew_uPF",      f"{hn}m :skew_vPF",  f"{hn}m :skew_wPF",
            f"{hn}m :skew_Theata_v",           # MATLAB typo preserved
            f"{hn}m :skew_H2O",      f"{hn}m :skew_H2O_WPL",
            f"{hn}m :skew_CO2",      f"{hn}m :skew_CO2_WPL",
        ]
    # H_SNSP  (1 + 4 per sonic)
    snsp_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        snsp_hdr += [
            f"{hn}m :uTHv", f"{hn}m :vTHv",
            f"{hn}m :wTHv", f"{hn}m :wTHv_vert",
        ]
    # Flux_lat  (1 + 5 per sonic)
    fl_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        fl_hdr += [
            f"{hn}m :uPF'theta_v'",
            f"{hn}m :uPF'H2O'",   f"{hn}m :uPF'H2O_WPL'",
            f"{hn}m :uPF'CO2'",   f"{hn}m :uPF'CO2_WPL'",
        ]
    # LHflux  (1 + 8 per sonic)
    lh_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        lh_hdr += [
            f"{hn}m Lv(J/g)",
            f"{hn}m w':E(g/m^2s)",    f"{hn}m wPF':E(g/m^2s)",
            f"{hn}m wPF'q_WPL'(m/s kg/m3)",
            f"{hn}m WPL, w' (W/m^2)", f"{hn}m WPL, wPF' (W/m^2)",
            f"{hn}m O2 no WPL,w' (W/m^2)", f"{hn}m O2 no WPL,wPF' (W/m^2)",
        ]
    # CO2flux  (1 + 5 per sonic)
    co2_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        co2_hdr += [
            f"{hn}m w':CO2(kg/m^2s)",
            f"{hn}m wPF':CO2(kg/m^2s)",
            f"{hn}m WPL,wPF':CO2(mol/m^2s)",
            f"{hn}m: wPF'CO2_WPL'(m/s kg/m^3)",
            f"{hn}m: CO2 (ppm, moist-air molar ratio)",
        ]

    # --- store with trimmed headers ---
    output["H"],     output["Hheader"]   = _trim_both(H_mat,     H_hdr)
    output["Hlat"],  output["HlatHeader"]= _trim_both(Hlat_mat,  Hlat_hdr)
    output["tau"],   output["tauHeader"] = _trim_both(tau_mat,   tau_hdr)
    output["tke"],   output["tkeHeader"] = _trim_both(tke_mat,   tke_hdr)
    output["sigma"], output["sigmaHeader"]= _trim_both(sigma_mat, sigma_hdr)

    if store_extra:
        output["R"],    output["RHeader"]              = _trim_both(R_mat,      R_hdr)
        output["L"],    output["Lheader"]              = _trim_both(L_mat,      L_hdr)
        output["eta"],  output["etaHeader"]            = _trim_both(eta_mat,    eta_hdr)
        output["delta_flux_ctrb"],  output["delta_flux_ctrbHeader"] = _trim_both(df_mat, df_hdr)
        output["delta_time_ctrb"],  output["delta_time_ctrbHeader"] = _trim_both(dt_mat, dt_hdr)
        output["turbtr"],   output["turbtrHeader"]     = _trim_both(turbtr_mat, turbtr_hdr)
        output["epsilon"],  output["epsilonHeader"]    = _trim_both(eps_mat,    eps_hdr)
        output["skew"],     output["skewHeader"]       = _trim_both(skew_mat,   skew_hdr)
        output["H_SNSP"],   output["H_SNSPHeader"]     = _trim_both(H_snsp,     snsp_hdr)

    output["Flux_lat"],  output["Flux_latHeader"] = _trim_both(Flux_lat,  fl_hdr)
    output["LHflux"],    output["LHfluxHeader"]   = _trim_both(LH_mat,    lh_hdr)

    qc_hdr = ["time"]
    for h in sonic_heights:
        hn = _h(h)
        qc_hdr += [
            f"{hn}m:TAU_SSITC_TEST", f"{hn}m:TAU_SS_ONLY_TEST",
            f"{hn}m:H_SSITC_TEST",   f"{hn}m:H_SS_ONLY_TEST",
            f"{hn}m:LE_SSITC_TEST",  f"{hn}m:LE_SS_ONLY_TEST",
            f"{hn}m:FC_SSITC_TEST",  f"{hn}m:FC_SS_ONLY_TEST",
        ]
    output["fluxQC"],   output["fluxQCHeader"]   = _trim_both(fluxQC_mat, qc_hdr)

    if np.any(~np.isnan(CO2fx_mat[:, 1:])):
        output["CO2flux"],   output["CO2fluxHeader"]  = _trim_both(CO2fx_mat, co2_hdr)

    # derivedT: block-averaged derived temperatures (theta_v_son, T_son_air, fw temps)
    if derivedT_cols:
        timestamps = simple_avg(np.column_stack([t, t]), info["avgPer"])[:, 1]
        dt_arrays  = np.column_stack([timestamps] + [c for c, _ in derivedT_cols])
        dt_headers = ["time"] + [h for _, h in derivedT_cols]
        output["derivedT"], output["derivedTheader"] = _trim_both(dt_arrays, dt_headers)

    return output, raw
