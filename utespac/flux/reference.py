"""Site-reference thermodynamic state per averaging period.

The flux computation needs, for every period, a reference pressure,
temperature and specific humidity at the lowest sonic and the dry/vapour/
moist densities built from them. :func:`reference_state` assembles them
from whatever the site logs (barometer, slow T/RH probe, IRGA or KH2O
humidity, sonic temperature as the fallback) exactly as ``fluxes.m`` did.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..averaging import block_average, n_periods
from ..rh_to_spec_hum import rh_to_spec_hum

log = logging.getLogger("utespac")

Rd = 287.058   # J/(kg·K)
Rv = 461.495   # J/(kg·K)
Mv = 18.0153   # g/mol  H2O
Md = 28.97     # g/mol  dry air


@dataclass
class ReferenceState:
    """Per-period reference quantities (length ``N``) and the per-sample
    humidity interpolated onto the fast time axis."""
    z_ref: float                   # [m] lowest sonic height (or the configured zRefLowestSon)
    altitude: float                # [m] site elevation
    P_ref_kPa: float               # standard-atmosphere pressure at altitude + z_ref
    P_kPa: np.ndarray              # [kPa] per period (barometer or P_ref)
    T_K: np.ndarray                # [K] per period
    q: np.ndarray                  # [kg/kg] per period
    q_fast: np.ndarray             # [kg/kg] per sample
    rho_d: np.ndarray              # [kg/m³] dry air
    rho_v: np.ndarray              # [kg/m³] vapour
    rho: np.ndarray                # [kg/m³] moist air
    T_virt_K: np.ndarray           # [K] virtual temperature
    P_raw_hf: Optional[np.ndarray] = None   # [kPa] barometer samples, for the ppm conversion
    P_t_hf: Optional[np.ndarray] = None     # their timestamps
    raw_P: Optional[np.ndarray] = None      # (t, P) columns for the raw product

    @property
    def n(self) -> int:
        return len(self.P_kPa)


def _nearest(sensor_info: Dict, key: str, z_ref: float):
    """(table, column) of the *key* sensor nearest to z_ref."""
    idx = int(np.argmin(np.abs(sensor_info[key][:, 2] - z_ref)))
    return int(sensor_info[key][idx, 0]), int(sensor_info[key][idx, 1])


def _period_means(values, t, avg_per):
    """Block means (with period-end times) on the series' own time axis."""
    t_end, m = block_average(t, values, avg_per)
    return m[:, 0], t_end


def _interp_to_fast(q_avg, t_avg, t_fast):
    """Step/linear interpolation of a per-period series onto the fast axis,
    anchored at the midnight before the first valid period (as fluxes.m)."""
    valid = ~np.isnan(q_avg)
    x = np.concatenate([[np.floor(t_avg[valid][0])], t_avg[valid]])
    y = np.concatenate([[q_avg[valid][0]], q_avg[valid]])
    return np.interp(t_fast, x, y)


def reference_state(data: List[Optional[np.ndarray]], sensor_info: Dict, info: Dict,
                    t: np.ndarray) -> ReferenceState:
    """Build the :class:`ReferenceState` for one run (fast time axis *t*)."""
    avg_per = info["avgPer"]
    z_ref = float(np.min(sensor_info["u"][:, 2]))
    altitude = float(info.get("siteElevation", 0))
    P_ref_kPa = 101.325 * (1 - 2.25577e-5 * (altitude + z_ref)) ** 5.25588

    # --- pressure ---
    P_kPa_avg: Optional[np.ndarray] = None
    P_raw_hf = P_t_hf = raw_P = None
    if "P" in sensor_info:
        P_tbl, P_col = _nearest(sensor_info, "P", z_ref)
        P_raw = data[P_tbl][:, P_col].copy()
        P_t = data[P_tbl][:, 0]
        if np.nanmedian(P_raw) > 200:
            P_raw /= 10.0
        P_kPa_avg, _ = _period_means(P_raw, P_t, avg_per)
        raw_P = np.column_stack([P_t, P_raw])
        # Always keep raw samples for the ppm conversion (MATLAB: Pson = data(:,PCol) always)
        P_raw_hf, P_t_hf = P_raw, P_t
        if np.nansum(~np.isnan(P_kPa_avg)) > 0 and \
                np.abs((np.nanmedian(P_kPa_avg) - P_ref_kPa) / P_ref_kPa) < 0.05:
            log.info(f"Barometer found. Median P = {np.nanmedian(P_kPa_avg):.3g} kPa")
        else:
            P_kPa_avg = None

    if P_kPa_avg is None:
        N_est = n_periods(t, avg_per, whole_days=False)
        P_kPa_avg = np.full(N_est, P_ref_kPa)
        log.info(f"No valid barometer – using P_ref = {P_ref_kPa:.3g} kPa.")

    # --- reference temperature ---
    T_ref_K_avg: Optional[np.ndarray] = None
    if "T" in sensor_info:
        T_tbl, T_col = _nearest(sensor_info, "T", z_ref)
        T_ref_K_avg, _ = _period_means(data[T_tbl][:, T_col], data[T_tbl][:, 0], avg_per)
        if np.nanmedian(T_ref_K_avg) < 200:
            T_ref_K_avg += 273.15
        log.info(f"Slow-response T found. Median T_ref = {np.nanmedian(T_ref_K_avg) - 273.15:.3g} °C")

    if T_ref_K_avg is None or np.nansum(~np.isnan(T_ref_K_avg)) == 0:
        if "Tson" in sensor_info:
            T_tbl, T_col = _nearest(sensor_info, "Tson", z_ref)
            T_ref_K_avg, _ = _period_means(data[T_tbl][:, T_col], t, avg_per)
            if np.nanmedian(T_ref_K_avg) < 200:
                T_ref_K_avg += 273.15
            log.info(f"Using sonic T as Tref. Median = {np.nanmedian(T_ref_K_avg) - 273.15:.3g} °C")
        else:
            T_ref_K_avg = np.full(len(P_kPa_avg), 293.15)

    # --- reference specific humidity ---
    q_default = info.get("qRef", 12) / 1000.0
    q_ref_avg = q_ref_fast = None
    if "RH" in sensor_info:
        RH_tbl, RH_col = _nearest(sensor_info, "RH", z_ref)
        RH_avg, RH_avg_t = _period_means(data[RH_tbl][:, RH_col], data[RH_tbl][:, 0], avg_per)
        q_ref_avg = rh_to_spec_hum(RH_avg, P_kPa_avg, T_ref_K_avg)
        if (~np.isnan(q_ref_avg)).any():
            q_ref_fast = _interp_to_fast(q_ref_avg, RH_avg_t, t)
            log.info(f"RH found. Median q_ref = {1000 * np.nanmedian(q_ref_avg):.3g} g/kg")
        else:
            q_ref_avg = None
    elif "irgaH2O" in sensor_info or "KH2O" in sensor_info:
        # No slow-response RH — q from the IRGA/KH2O vapour density (matches MATLAB),
        # q = rho_v / (rho_d + rho_v) with rho_d from P - e.
        h2o_key = "irgaH2O" if "irgaH2O" in sensor_info else "KH2O"
        h2o_tbl, h2o_col = _nearest(sensor_info, h2o_key, z_ref)
        h2o_avg, h2o_avg_t = _period_means(data[h2o_tbl][:, h2o_col].copy(),
                                           data[h2o_tbl][:, 0], avg_per)        # g/m³
        rho_v_ref = h2o_avg / 1000.0                                             # kg/m³
        e_ref_Pa = rho_v_ref * Rv * T_ref_K_avg
        rho_d_ref = (P_kPa_avg * 1000.0 - e_ref_Pa) / (Rd * T_ref_K_avg)
        q_ref_avg = rho_v_ref / (rho_d_ref + rho_v_ref)
        if (~np.isnan(q_ref_avg)).any():
            q_ref_fast = _interp_to_fast(q_ref_avg, h2o_avg_t, t)
            log.info(f"No RH – qRef from {h2o_key}. Median q_ref = {1000 * np.nanmedian(q_ref_avg):.3g} g/kg")
        else:
            q_ref_avg = None
    if q_ref_avg is None:
        q_ref_avg = np.full(len(T_ref_K_avg), q_default)
        q_ref_fast = np.full(len(t), q_default)

    # --- moist-air density ---
    P_v_avg = q_ref_avg * P_kPa_avg / 0.622
    P_d_avg = P_kPa_avg - P_v_avg
    rho_d_avg = 1000.0 * P_d_avg / (Rd * T_ref_K_avg)
    rho_v_avg = 1000.0 * P_v_avg / (Rv * T_ref_K_avg)
    rho_avg = rho_d_avg + rho_v_avg
    T_virt_ref_K_avg = T_ref_K_avg * (1.0 + 0.61 * q_ref_avg)

    # Manual zRef override — a single high sonic run wanting virtual theta /
    # specificHum relative to the lowest sonic of the full tower.
    if info.get("shiftzRef", False):
        z_ref = float(info["zRefLowestSon"])

    log.info(f"ρ_moist = {np.nanmedian(rho_avg):.3g} kg/m³  "
             f"ρ_dry = {np.nanmedian(rho_d_avg):.3g} kg/m³  "
             f"T_virt_ref = {np.nanmedian(T_virt_ref_K_avg) - 273.15:.3g} °C  "
             f"zRef = {z_ref:.2f} m")

    return ReferenceState(z_ref, altitude, P_ref_kPa, P_kPa_avg, T_ref_K_avg, q_ref_avg,
                          q_ref_fast, rho_d_avg, rho_v_avg, rho_avg, T_virt_ref_K_avg,
                          P_raw_hf, P_t_hf, raw_P)
