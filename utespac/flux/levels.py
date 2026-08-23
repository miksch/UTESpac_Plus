"""What one sonic level brings to the flux computation.

:func:`build_level` gathers, for the ``ii``-th sonic, the high-frequency
series (raw, planar-fit and tilt winds, sonic and derived temperatures,
fine-wire, hygrometer, CO2), the per-period quality flags, the level's own
pressure and humidity, and the block-averaged columns that go to the
``specificHum`` and ``derivedT`` outputs, into a :class:`LevelInputs`.
The per-period engine (:mod:`.engine`) then works on slices of it.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..averaging import block_average, block_mean, period_bounds
from ..get_virtual_pot_temp import get_virtual_pot_temp
from ..rh_to_spec_hum import rh_to_spec_hum
from ..site_config import sonic_for
from ..sonic_temperature import SONIC_HUMIDITY_COEFF
from .reference import ReferenceState

log = logging.getLogger("utespac")

GAMMA_DRY = 0.0098     # [K/m] dry adiabatic lapse rate for theta


@dataclass
class LevelInputs:
    index: int
    height: float
    table_index: int
    manufacturer: int
    bearing: float
    # fast series
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    T_son: np.ndarray
    theta_son: np.ndarray
    theta_son_air: np.ndarray
    u_pf: np.ndarray            # planar fit + yaw
    v_pf: np.ndarray
    w_pf: np.ndarray
    pf_u: np.ndarray            # planar fit only, as rotated (for the raw product)
    pf_v: np.ndarray
    u_tilt: np.ndarray          # planar-fit frame, along the fall line
    v_tilt: np.ndarray          # across it
    w_tilt: np.ndarray
    # per-period flags
    unrot_flag: np.ndarray
    rot_flag: np.ndarray
    Ts_flag: np.ndarray
    fw_flag: np.ndarray
    h2o_flag: np.ndarray
    co2_flag: np.ndarray
    # optional sensors
    fw: Optional[np.ndarray] = None
    theta_fw: Optional[np.ndarray] = None
    Vtheta_fw: Optional[np.ndarray] = None
    h2o: Optional[np.ndarray] = None          # [g/m³]
    h2o_is_kh2o: bool = False
    h2o_sensor_index: int = 0                 # column in the raw rhov arrays
    co2: Optional[np.ndarray] = None          # [mg/m³]
    co2_sensor_index: int = 0
    # level pressure (per period / per sample)
    P_kPa: Optional[np.ndarray] = None
    P_raw_hf: Optional[np.ndarray] = None
    P_t_hf: Optional[np.ndarray] = None
    # level humidity and virtual temperature
    q_fast_local: Optional[np.ndarray] = None
    virtual_theta_avg: Optional[np.ndarray] = None   # per period, from the level HMP
    direction_avg: Optional[np.ndarray] = None
    # averaged output columns contributed by this level
    specific_hum_cols: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    derived_T_cols: List[Tuple[str, np.ndarray]] = field(default_factory=list)


def _sensor_col(sensor_info, key, height):
    """(table_idx, col_idx) for *key* at *height*, or None."""
    if key not in sensor_info:
        return None
    rows = sensor_info[key][:, 2] == height
    if not rows.any():
        return None
    si = int(np.where(rows)[0][0])
    return int(sensor_info[key][si, 0]), int(sensor_info[key][si, 1])


def _period_flag(output, table_name, col, N, which):
    """Per-period ``<table><which>`` flag column, zeros when absent."""
    arr = output.get(f"{table_name}{which}")
    if arr is None or col >= arr.shape[1]:
        return np.zeros(N, dtype=bool)
    return arr[:, col].astype(bool)


def _avg_table_flag(output, sensor_info, key, height, table_names, N, limit, op):
    """Flag from an averaged diagnostic/signal column: ``op(value, limit)`` where
    the averaged value is not NaN."""
    loc = _sensor_col(sensor_info, key, height)
    flag = np.zeros(N, dtype=bool)
    if loc is None:
        return flag
    tbl, col = loc
    tn = table_names[tbl]
    if tn in output and col < output[tn].shape[1]:
        v = output[tn][:, col]
        ok = ~np.isnan(v)
        flag[ok] = op(v[ok], limit)
    return flag


def build_level(ii: int, data, output: Dict, sensor_info: Dict, info: Dict,
                table_names: List[str], ref: ReferenceState,
                rotated: np.ndarray, pf_only: np.ndarray, N: int, t: np.ndarray,
                slope_axis: str, matlab_compat: bool) -> LevelInputs:
    """Assemble the ``ii``-th sonic level (see module docstring)."""
    tbl_idx = int(sensor_info["u"][ii, 0])
    height = float(sensor_info["u"][ii, 2])
    manufact = int(sensor_info["u"][ii, 4]) if sensor_info["u"].shape[1] > 4 else 1
    bearing = float(sensor_info["u"][ii, 3]) if sensor_info["u"].shape[1] > 3 else 0.0
    tname = table_names[tbl_idx]
    avg_per = info["avgPer"]
    diag_cfg = info.get("diagnosticTest", {})

    mask_h = sensor_info["u"][:, 2] == height
    u_col = int(sensor_info["u"][mask_h, 1][0])
    v_col = int(sensor_info["v"][sensor_info["v"][:, 2] == height, 1][0])
    w_col = int(sensor_info["w"][sensor_info["w"][:, 2] == height, 1][0])
    Ts_col = int(sensor_info["Tson"][sensor_info["Tson"][:, 2] == height, 1][0]) \
        if "Tson" in sensor_info else None

    u_raw = data[tbl_idx][:, u_col]
    v_raw = data[tbl_idx][:, v_col]
    w_raw = data[tbl_idx][:, w_col]

    # ---- flags ----
    def _flag(col):
        return _period_flag(output, tname, col, N, "NanFlag") | _period_flag(output, tname, col, N, "SpikeFlag")

    unrot_flag = _flag(w_col)
    rot_flag = _flag(u_col) | _flag(v_col) | _flag(w_col)

    # averaged sonic diagnostic (MATLAB fluxes.m lines 363-381): periods whose
    # mean diagnostic reaches meanSonicDiagnosticLimit are bad
    if "sonDiagnostic" in sensor_info:
        diag_rows = sensor_info["sonDiagnostic"][:, 2] == height
        if diag_rows.any():
            d_tbl = int(sensor_info["sonDiagnostic"][diag_rows, 0][0])
            d_col = int(sensor_info["sonDiagnostic"][diag_rows, 1][0])
            tbl_avg = output.get(table_names[d_tbl])
            if tbl_avg is not None and d_col < tbl_avg.shape[1]:
                diag_avg = tbl_avg[:, d_col].copy().astype(float)
                diag_avg[np.isnan(diag_avg)] = 0.0
                diag_flag = diag_avg >= diag_cfg.get("meanSonicDiagnosticLimit", 50)
                unrot_flag = unrot_flag | diag_flag
                rot_flag = rot_flag | diag_flag

    Ts_flag = np.zeros(N, dtype=bool)
    T_son = np.full(len(t), np.nan)
    if Ts_col is not None:
        T_son = data[tbl_idx][:, Ts_col].copy()
        if np.nanmedian(T_son) > 250:
            T_son -= 273.15
        Ts_flag = _flag(Ts_col)

    # ---- rotated and PF-only wind columns ----
    r_hdr = output.get("rotatedSonicHeader", [])
    nan_series = np.full(len(t), np.nan)
    try:
        ru_col = r_hdr.index(f"{height}m:u")
        rv_col = r_hdr.index(f"{height}m:v")
        rw_col = r_hdr.index(f"{height}m:w")
        u_pf, v_pf, w_pf = rotated[:, ru_col], rotated[:, rv_col], rotated[:, rw_col]
    except (ValueError, IndexError):
        ru_col = rv_col = rw_col = None
        u_pf = v_pf = w_pf = nan_series
    try:
        # u_tilt is the component along the fall line, v_tilt across it:
        # slopeAxis names the PF axis that points downslope (French Meadows: v,
        # hence the swap MATLAB fluxes.m:736-737 hardcodes).
        along_col, across_col = (rv_col, ru_col) if slope_axis == "v" else (ru_col, rv_col)
        u_tilt = pf_only[:, along_col] if along_col is not None else nan_series
        v_tilt = pf_only[:, across_col] if across_col is not None else nan_series
        w_tilt = pf_only[:, rw_col] if rw_col is not None else nan_series
    except Exception:
        u_tilt = v_tilt = w_tilt = nan_series
    pf_u = pf_only[:, ru_col] if ru_col is not None else nan_series
    pf_v = pf_only[:, rv_col] if rv_col is not None else nan_series

    # ---- potential sonic temperature ----
    theta_son = T_son + GAMMA_DRY * (height - ref.z_ref)

    # ---- height-specific pressure (co-located IRGASON barometer) ----
    P_kPa_lev, P_raw_hf_lev, P_t_hf_lev = ref.P_kPa, ref.P_raw_hf, ref.P_t_hf
    if "P" in sensor_info:
        p_rows = np.abs(sensor_info["P"][:, 2] - height) < 0.5
        if p_rows.any():
            p_tbl = int(sensor_info["P"][p_rows, 0][0])
            p_col = int(sensor_info["P"][p_rows, 1][0])
            P_raw_lev = data[p_tbl][:, p_col].copy()
            P_t_lev = data[p_tbl][:, 0]
            if np.nanmedian(P_raw_lev) > 200:
                P_raw_lev /= 10.0
            _, P_lev_check = block_average(P_t_lev, P_raw_lev, avg_per)
            P_lev_check = P_lev_check[:, 0]
            if (np.nansum(~np.isnan(P_lev_check)) > 0 and
                    np.abs((np.nanmedian(P_lev_check) - ref.P_ref_kPa) / ref.P_ref_kPa) < 0.05):
                P_kPa_lev, P_raw_hf_lev, P_t_hf_lev = P_lev_check, P_raw_lev, P_t_lev

    # ---- level-specific humidity and virtual theta (useTrefHMP) ----
    q_fast_local = ref.q_fast.copy()
    virtual_theta_avg = None
    specific_hum_cols: List[Tuple[str, np.ndarray]] = []
    if info.get("useTrefHMP") and "RH" in sensor_info and "T" in sensor_info:
        rh_rows = sensor_info["RH"][:, 2] == height
        t_rows = sensor_info["T"][:, 2] == height
        if rh_rows.any() and t_rows.any():
            RH_tbl2 = int(sensor_info["RH"][rh_rows, 0][0])
            RH_col2 = int(sensor_info["RH"][rh_rows, 1][0])
            T_tbl2 = int(sensor_info["T"][t_rows, 0][0])
            T_col2 = int(sensor_info["T"][t_rows, 1][0])
            freq_slow = info.get("avgSlowFreq", 1)
            RH_lev = data[RH_tbl2][:, RH_col2]
            T_lev = data[T_tbl2][:, T_col2]
            t_lev = data[RH_tbl2][:, 0]

            # slow-frequency averages for the virtual theta computation
            ts1, T1 = block_average(t_lev, T_lev, freq_slow)
            _, RH1 = block_average(t_lev, RH_lev, freq_slow)
            T1, RH1 = T1[:, 0], RH1[:, 0]
            P_slow = np.interp(ts1, np.linspace(t.min(), t.max(), len(P_kPa_lev)), P_kPa_lev)
            # paired physical HMP height for the altitude correction, if configured
            level_height = height
            _level = sonic_for(info.get("sonics"), height)
            if _level is not None:
                if _level.hmp_height is not None:
                    level_height = _level.hmp_height
            else:
                for _sh, _hh in zip(info.get("shiftsSonHeight", []), info.get("shiftsHMPHeight", [])):
                    if abs(height - _sh) < 0.01:
                        level_height = _hh
                        break
            vt_slow, r_slow, rho_moist_slow, rho_dry_slow, rho_H2O_slow = get_virtual_pot_temp(
                ref.altitude, level_height - ref.z_ref, T1, RH1, P_slow, use_p_elevation=False)

            def _avg30(arr):
                _, m = block_average(ts1, arr, avg_per)
                return m[:, 0] if m.shape[1] > 0 else np.full(N, np.nan)

            vt_avg30 = _avg30(vt_slow)
            virtual_theta_avg = vt_avg30

            # 30-min q from the HMP at this level -> q_fast_local (MATLAB qRefFastLocal)
            _, T30 = block_average(t_lev, T_lev, avg_per)
            t_q, RH30 = block_average(t_lev, RH_lev, avg_per)
            T30_K = T30[:, 0].copy()
            if np.nanmedian(T30_K) < 200:
                T30_K = T30_K + 273.15
            q30 = rh_to_spec_hum(RH30[:, 0], P_kPa_lev, T30_K)      # kg/kg per period
            valid_q = ~np.isnan(q30)
            if valid_q.any():
                x_q = np.concatenate([[np.floor(t_q[valid_q][0])], t_q[valid_q]])
                y_q = np.concatenate([[q30[valid_q][0]], q30[valid_q]])
                q_fast_local = np.interp(t, x_q, y_q)

            def _pad(arr):
                if len(arr) >= N:
                    return arr[:N]
                return np.concatenate([arr, np.full(N - len(arr), np.nan)])

            q_col = _pad(q30 * 1000.0) if valid_q.any() else _pad(q30)
            q_hdr = f"{height} m: q(g/kg)" if valid_q.any() else f"{height} m: q(g/g)"
            specific_hum_cols = [
                (q_hdr, q_col),
                (f"{height} m: virtualThetaAvg(K)", _pad(vt_avg30)),
                (f"{height} m: rAvg(g/kg)", _pad(_avg30(r_slow))),
                (f"{height} m: rho_airmoistAvg(kg/m^3)", _pad(_avg30(rho_moist_slow))),
                (f"{height} m: rho_airdryAvg(kg/m^3)", _pad(_avg30(rho_dry_slow))),
                (f"{height} m: rho_H2OAvg(kg/m^3)", _pad(_avg30(rho_H2O_slow))),
            ]

    # Mean-humidity rescale of the sonic temperature, T = T_s/(1 + 0.51 q)
    # (Schotanus et al. 1983 eq. 5; library/writeups/sonic_temperature_flux.md).
    # MATLAB fluxes.m:575 used the virtual-temperature 0.61 ("modified by Diane").
    sonic_coeff = 0.61 if matlab_compat else SONIC_HUMIDITY_COEFF
    theta_son_air = (T_son + 273.15) / (1.0 + sonic_coeff * q_fast_local) - 273.15

    # ---- fine-wire at this height ----
    fw = theta_fw = Vtheta_fw = None
    fw_flag = np.zeros(N, dtype=bool)
    fw_loc = _sensor_col(sensor_info, "fw", height)
    if fw_loc is not None:
        fw_tbl, fw_col = fw_loc
        fw = data[fw_tbl][:, fw_col].copy()
        fw_flag = (_period_flag(output, table_names[fw_tbl], fw_col, N, "NanFlag")
                   | _period_flag(output, table_names[fw_tbl], fw_col, N, "SpikeFlag"))
        theta_fw = fw + GAMMA_DRY * (height - ref.z_ref)
        Vtheta_fw = theta_fw * (1.0 + 0.61 * q_fast_local)      # MATLAB fluxes.m:566

    # ---- H2O at this height ----
    h2o = None
    h2o_flag = np.zeros(N, dtype=bool)
    h2o_is_kh2o = False
    h2o_si = 0
    if "irgaH2O" in sensor_info:
        loc = _sensor_col(sensor_info, "irgaH2O", height)
        if loc is not None:
            h_tbl, h_col = loc
            h2o = data[h_tbl][:, h_col].copy()
            h2o_flag = (_period_flag(output, table_names[h_tbl], h_col, N, "NanFlag")
                        | _period_flag(output, table_names[h_tbl], h_col, N, "SpikeFlag")
                        | _avg_table_flag(output, sensor_info, "irgaH2OsigStrength", height, table_names,
                                          N, diag_cfg.get("H2OminSignal"), np.less_equal)
                        | _avg_table_flag(output, sensor_info, "irgaGasDiag", height, table_names,
                                          N, diag_cfg.get("meanGasDiagnosticLimit"), np.greater_equal))
            h2o_si = int(np.where(sensor_info["irgaH2O"][:, 2] == height)[0][0])
    elif "LiH2O" in sensor_info:
        loc = _sensor_col(sensor_info, "LiH2O", height)
        if loc is not None:
            l_tbl, l_col = loc
            h2o = data[l_tbl][:, l_col] * 0.018          # mmol/m³ -> g/m³
            h2o_flag = (_period_flag(output, table_names[l_tbl], l_col, N, "NanFlag")
                        | _period_flag(output, table_names[l_tbl], l_col, N, "SpikeFlag")
                        | _avg_table_flag(output, sensor_info, "LiGasDiag", height, table_names,
                                          N, diag_cfg.get("meanLiGasDiagnosticLimit"), np.greater))
    elif "KH2O" in sensor_info:
        loc = _sensor_col(sensor_info, "KH2O", height)
        if loc is not None:
            k_tbl, k_col = loc
            h2o = data[k_tbl][:, k_col].copy()           # already g/m³
            h2o_flag = (_period_flag(output, table_names[k_tbl], k_col, N, "NanFlag")
                        | _period_flag(output, table_names[k_tbl], k_col, N, "SpikeFlag"))
            h2o_is_kh2o = True

    # ---- CO2 at this height (needs the hygrometer for the WPL terms) ----
    co2 = None
    co2_flag = np.zeros(N, dtype=bool)
    co2_si = 0
    if "irgaCO2" in sensor_info and h2o is not None:
        loc = _sensor_col(sensor_info, "irgaCO2", height)
        if loc is not None:
            c_tbl, c_col = loc
            co2 = data[c_tbl][:, c_col].copy()           # mg/m³
            co2_flag = (_period_flag(output, table_names[c_tbl], c_col, N, "NanFlag")
                        | _period_flag(output, table_names[c_tbl], c_col, N, "SpikeFlag")
                        | _avg_table_flag(output, sensor_info, "irgaCO2sigStrength", height, table_names,
                                          N, diag_cfg.get("CO2minSignal"), np.less_equal)
                        | _avg_table_flag(output, sensor_info, "irgaGasDiag", height, table_names,
                                          N, diag_cfg.get("meanGasDiagnosticLimit"), np.greater_equal))
            co2_si = int(np.where(sensor_info["irgaCO2"][:, 2] == height)[0][0])
    elif "LiCO2" in sensor_info and h2o is not None:
        loc = _sensor_col(sensor_info, "LiCO2", height)
        if loc is not None:
            c_tbl, c_col = loc
            co2 = data[c_tbl][:, c_col] * 44.0           # mmol/m³ -> mg/m³
            co2_flag = (_period_flag(output, table_names[c_tbl], c_col, N, "NanFlag")
                        | _period_flag(output, table_names[c_tbl], c_col, N, "SpikeFlag"))

    # ---- derivedT: block-averaged derived temperatures ----
    def _avg(series):
        return block_average(t, series, avg_per)[1][:, 0]
    derived: List[Tuple[str, np.ndarray]] = []
    has_fw_here = fw is not None and theta_fw is not None
    if has_fw_here:
        derived.append((f"{height} m: theta_fw", _avg(theta_fw)))
    derived.append((f"{height} m: theta_v_son", _avg(theta_son)))
    if has_fw_here:
        derived.append((f"{height} m: theta_v_fw", _avg(Vtheta_fw)))
    derived.append((f"{height} m: T_son_air", _avg(theta_son_air)))

    # ---- period-mean wind direction (H_SNSP) ----
    dir_hdr = output.get("spdAndDirHeader", [])
    dir_col = next((j for j, h in enumerate(dir_hdr) if h == f"{height}m direction"), None)
    direction_avg = output["spdAndDir"][:, dir_col] if dir_col is not None else np.zeros(N)

    return LevelInputs(
        ii, height, tbl_idx, manufact, bearing,
        u_raw, v_raw, w_raw, T_son, theta_son, theta_son_air,
        u_pf, v_pf, w_pf, pf_u, pf_v, u_tilt, v_tilt, w_tilt,
        unrot_flag, rot_flag, Ts_flag, fw_flag, h2o_flag, co2_flag,
        fw=fw, theta_fw=theta_fw, Vtheta_fw=Vtheta_fw,
        h2o=h2o, h2o_is_kh2o=h2o_is_kh2o, h2o_sensor_index=h2o_si,
        co2=co2, co2_sensor_index=co2_si,
        P_kPa=P_kPa_lev, P_raw_hf=P_raw_hf_lev, P_t_hf=P_t_hf_lev,
        q_fast_local=q_fast_local, virtual_theta_avg=virtual_theta_avg,
        direction_avg=direction_avg,
        specific_hum_cols=specific_hum_cols, derived_T_cols=derived,
    )
