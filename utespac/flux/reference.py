"""Site-reference thermodynamic state per averaging period.

The flux computation needs, for every period, a reference pressure,
temperature and specific humidity at the lowest sonic and the dry/vapour/
moist densities built from them. :func:`reference_state` assembles them
from whatever the site logs (barometer, slow T/RH probe, IRGA or KH2O
humidity, sonic temperature as the fallback) exactly as ``fluxes.m`` did,
reading the sensors of a :class:`~utespac.model.Run`.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..averaging import block_average, n_periods
from ..model import Run, Sensor
from ..rh_to_spec_hum import rh_to_spec_hum
from ..sonic_temperature import air_temperature_from_sonic
from .. import units

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
    def cp(self) -> np.ndarray:
        """[J kg-1 K-1] specific heat of moist air, ``1004.67 (1 + 0.84 q)``."""
        return 1004.67 * (1.0 + 0.84 * self.q)

    @property
    def n(self) -> int:
        return len(self.P_kPa)


def nearest(run: Run, field: str, z_ref: float) -> Sensor:
    """The *field* sensor nearest to z_ref."""
    sensors = run.sensors.by_field(field)
    return min(sensors, key=lambda s: abs(s.height - z_ref))


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


def reference_state(run: Run) -> ReferenceState:
    """Build the :class:`ReferenceState` for one run."""
    units.reset_warnings()   # one fallback warning per sensor per run
    info = run.site
    avg_per = info["avgPer"]
    t = run.time_hf_datenum
    z_ref = float(min(run.sonic_heights()))
    altitude = float(info.get("siteElevation", 0))
    P_ref_kPa = 101.325 * (1 - 2.25577e-5 * (altitude + z_ref)) ** 5.25588

    # --- pressure ---
    P_kPa_avg: Optional[np.ndarray] = None
    P_raw_hf = P_t_hf = raw_P = None
    if run.sensors.has("P"):
        sP = nearest(run, "P", z_ref)
        P_raw = units.convert(run.hf(sP), sP.units, "pressure", sensor=sP)
        P_t = run.hf_time(sP.table)
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
    if run.sensors.has("T"):
        sT = nearest(run, "T", z_ref)
        T_ref_K_avg, _ = _period_means(run.hf(sT), run.hf_time(sT.table), avg_per)
        T_ref_K_avg = units.convert(T_ref_K_avg, sT.units, "temperature", "K", sensor=sT)
        log.info(f"Slow-response T found. Median T_ref = {np.nanmedian(T_ref_K_avg) - 273.15:.3g} °C")

    T_ref_is_sonic = False
    if T_ref_K_avg is None or np.nansum(~np.isnan(T_ref_K_avg)) == 0:
        if run.sensors.has("Tson"):
            sTs = nearest(run, "Tson", z_ref)
            T_ref_K_avg, _ = _period_means(run.hf(sTs), t, avg_per)
            T_ref_K_avg = units.convert(T_ref_K_avg, sTs.units, "temperature", "K", sensor=sTs)
            T_ref_is_sonic = True
            log.info(f"Using sonic T as Tref. Median = {np.nanmedian(T_ref_K_avg) - 273.15:.3g} °C")
        else:
            T_ref_K_avg = np.full(len(P_kPa_avg), 293.15)

    # --- reference specific humidity ---
    q_default = info.get("qRef", 12) / 1000.0
    q_ref_avg = q_ref_fast = None
    if run.sensors.has("RH"):
        sRH = nearest(run, "RH", z_ref)
        RH_avg, RH_avg_t = _period_means(run.hf(sRH), run.hf_time(sRH.table), avg_per)
        RH_avg = units.convert(RH_avg, sRH.units, "humidity", sensor=sRH)
        q_ref_avg = rh_to_spec_hum(RH_avg, P_kPa_avg, T_ref_K_avg)
        if (~np.isnan(q_ref_avg)).any():
            q_ref_fast = _interp_to_fast(q_ref_avg, RH_avg_t, t)
            log.info(f"RH found. Median q_ref = {1000 * np.nanmedian(q_ref_avg):.3g} g/kg")
        else:
            q_ref_avg = None
    elif run.sensors.has("irgaH2O") or run.sensors.has("KH2O"):
        # No slow-response RH — q from the IRGA/KH2O vapour density (matches MATLAB),
        # q = rho_v / (rho_d + rho_v) with rho_d from P - e.
        h2o_key = "irgaH2O" if run.sensors.has("irgaH2O") else "KH2O"
        sH = nearest(run, h2o_key, z_ref)
        h2o_hf = units.convert(run.hf(sH), sH.units, "h2o_density", sensor=sH)     # g/m³
        h2o_avg, h2o_avg_t = _period_means(h2o_hf, run.hf_time(sH.table), avg_per)
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

    # Sonic-fallback Tref is a sonic (virtual-like) temperature; convert to
    # air temperature via Schotanus once q_ref is resolved, so the reference
    # densities and T_virt below start from actual T (upstream 874f54b).
    if T_ref_is_sonic:
        T_ref_K_avg = air_temperature_from_sonic(T_ref_K_avg, q_ref_avg)

    # --- moist-air density ---
    P_v_avg = q_ref_avg * P_kPa_avg / 0.622
    P_d_avg = P_kPa_avg - P_v_avg
    rho_d_avg = 1000.0 * P_d_avg / (Rd * T_ref_K_avg)
    rho_v_avg = 1000.0 * P_v_avg / (Rv * T_ref_K_avg)
    rho_avg = rho_d_avg + rho_v_avg
    T_virt_ref_K_avg = T_ref_K_avg * (1.0 + 0.61 * q_ref_avg)

    # Manual zRef override — a single high sonic run wanting virtual theta /
    # humidity products relative to the lowest sonic of the full tower.
    if info.get("shiftzRef", False):
        z_ref = float(info["zRefLowestSon"])

    log.info(f"ρ_moist = {np.nanmedian(rho_avg):.3g} kg/m³  "
             f"ρ_dry = {np.nanmedian(rho_d_avg):.3g} kg/m³  "
             f"T_virt_ref = {np.nanmedian(T_virt_ref_K_avg) - 273.15:.3g} °C  "
             f"zRef = {z_ref:.2f} m")

    return ReferenceState(z_ref, altitude, P_ref_kPa, P_kPa_avg, T_ref_K_avg, q_ref_avg,
                          q_ref_fast, rho_d_avg, rho_v_avg, rho_avg, T_virt_ref_K_avg,
                          P_raw_hf, P_t_hf, raw_P)
