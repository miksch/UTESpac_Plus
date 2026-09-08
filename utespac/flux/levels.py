"""What one sonic level brings to the flux computation.

:func:`build_level` gathers, for one sonic of a :class:`~utespac.model.Run`,
the high-frequency series (raw, planar-fit and tilt winds, sonic and
derived temperatures, fine-wire, hygrometer, CO2), the per-period quality
flags, the level's own pressure and humidity, and the block-averaged
columns that go to the ``humidity`` and ``temperature`` outputs, into a
:class:`LevelInputs`. The per-period engine (:mod:`.engine`) then works on
slices of it.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..averaging import block_average
from ..get_virtual_pot_temp import get_virtual_pot_temp
from ..model import CO2_FIELDS, COMPONENT, H2O_FIELDS, HEIGHT, Run, Sensor
from ..rh_to_spec_hum import rh_to_spec_hum
from ..site_config import sonic_for
from .. import units
from ..sonic_temperature import SONIC_HUMIDITY_COEFF
from .reference import ReferenceState
from .tables import height_label

log = logging.getLogger("utespac")

GAMMA_DRY = 0.0098     # [K/m] dry adiabatic lapse rate for theta


@dataclass
class LevelInputs:
    index: int
    height: float
    scan_hz: float
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
    h2o_sensor_index: Optional[int] = None    # column in the raw rhov arrays (None: not carried)
    co2: Optional[np.ndarray] = None          # [mg/m³]
    co2_sensor_index: Optional[int] = None
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


def _threshold_flag(run: Run, sensor: Optional[Sensor], N: int, limit, op) -> np.ndarray:
    """Flag from an averaged diagnostic/signal column: ``op(value, limit)`` where
    the period mean is not NaN."""
    flag = np.zeros(N, dtype=bool)
    if sensor is None:
        return flag
    v = run.period_mean(sensor)
    if v is None:
        return flag
    ok = ~np.isnan(v)
    flag[ok] = op(v[ok], limit)
    return flag


def _flags(run: Run, sensor: Sensor, N: int) -> np.ndarray:
    return run.flag(sensor, "nan", N) | run.flag(sensor, "spike", N)


def build_level(run: Run, ii: int, ref: ReferenceState, N: int, slope_axis: str) -> LevelInputs:
    """Assemble the ``ii``-th sonic level (see module docstring)."""
    info = run.site
    avg_per = info["avgPer"]
    diag_cfg = info.get("diagnosticTest", {})
    su = run.sensors.by_field("u")[ii]
    height = su.height
    hn = height_label(height)
    sv = run.sensors.at("v", height)
    sw = run.sensors.at("w", height)
    sTs = run.sensors.at("Tson", height)
    t = run.time_hf_datenum
    n = len(t)
    nan_series = np.full(n, np.nan)
    scan_hz = float(run.tables[su.table].attrs.get("scan_hz", 20.0))

    u_raw, v_raw, w_raw = run.hf(su), run.hf(sv), run.hf(sw)

    # ---- flags ----
    unrot_flag = _flags(run, sw, N)
    rot_flag = _flags(run, su, N) | _flags(run, sv, N) | _flags(run, sw, N)

    # averaged sonic diagnostic (MATLAB fluxes.m lines 363-381): periods whose
    # mean diagnostic reaches meanSonicDiagnosticLimit are bad
    sd = run.sensors.at("sonDiagnostic", height)
    if sd is not None:
        diag_avg = run.period_mean(sd)
        if diag_avg is not None:
            diag_avg = diag_avg.copy().astype(float)
            diag_avg[np.isnan(diag_avg)] = 0.0
            diag_flag = diag_avg >= diag_cfg.get("meanSonicDiagnosticLimit", 50)
            unrot_flag = unrot_flag | diag_flag
            rot_flag = rot_flag | diag_flag

    Ts_flag = np.zeros(N, dtype=bool)
    T_son = np.full(n, np.nan)
    if sTs is not None:
        T_son = units.convert(run.hf(sTs), sTs.units, "temperature", sensor=sTs)
        Ts_flag = _flags(run, sTs, N)

    # ---- rotated and PF-only wind columns ----
    rot = run.rotation
    if rot is not None and height in [float(h) for h in rot[HEIGHT].values]:
        hi = [float(h) for h in rot[HEIGHT].values].index(height)
        comps = list(rot[COMPONENT].values)
        rotated = rot["rotated"].values[:, hi, :]
        pf = rot["pf"].values[:, hi, :]
        u_pf, v_pf, w_pf = (rotated[:, comps.index(c)] for c in ("u", "v", "w"))
        pf_u, pf_v, pf_w = (pf[:, comps.index(c)] for c in ("u", "v", "w"))
    else:
        u_pf = v_pf = w_pf = pf_u = pf_v = pf_w = nan_series
    # u_tilt is the component along the fall line, v_tilt across it: slopeAxis
    # names the PF axis that points downslope (French Meadows: v, hence the
    # swap MATLAB fluxes.m:736-737 hardcodes).
    u_tilt, v_tilt = (pf_v, pf_u) if slope_axis == "v" else (pf_u, pf_v)
    w_tilt = pf_w

    # ---- potential sonic temperature ----
    theta_son = T_son + GAMMA_DRY * (height - ref.z_ref)

    # ---- height-specific pressure (co-located IRGASON barometer) ----
    P_kPa_lev, P_raw_hf_lev, P_t_hf_lev = ref.P_kPa, ref.P_raw_hf, ref.P_t_hf
    sP = next((s for s in run.sensors.by_field("P") if abs(s.height - height) < 0.5), None)
    if sP is not None:
        P_raw_lev = units.convert(run.hf(sP), sP.units, "pressure", sensor=sP)
        P_t_lev = run.hf_time(sP.table)
        _, P_lev_check = block_average(P_t_lev, P_raw_lev, avg_per)
        P_lev_check = P_lev_check[:, 0]
        if (np.nansum(~np.isnan(P_lev_check)) > 0 and
                np.abs((np.nanmedian(P_lev_check) - ref.P_ref_kPa) / ref.P_ref_kPa) < 0.05):
            P_kPa_lev, P_raw_hf_lev, P_t_hf_lev = P_lev_check, P_raw_lev, P_t_lev

    # ---- level-specific humidity and virtual theta (useTrefHMP) ----
    q_fast_local = ref.q_fast.copy()
    virtual_theta_avg = None
    specific_hum_cols: List[Tuple[str, np.ndarray]] = []
    sRH, sT = run.sensors.at("RH", height), run.sensors.at("T", height)
    if info.get("useTrefHMP") and sRH is not None and sT is not None:
        freq_slow = info.get("avgSlowFreq", 1)
        RH_lev, T_lev = run.hf(sRH), run.hf(sT)
        t_lev = run.hf_time(sRH.table)

        # slow-frequency averages for the virtual theta computation, in the
        # units get_virtual_pot_temp documents (T in K, RH in percent)
        ts1, T1 = block_average(t_lev, T_lev, freq_slow)
        _, RH1 = block_average(t_lev, RH_lev, freq_slow)
        T1 = units.convert(T1[:, 0], sT.units, "temperature", "K", sensor=sT)
        RH1 = units.convert(RH1[:, 0], sRH.units, "humidity", sensor=sRH)
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
        T30_K = units.convert(T30[:, 0], sT.units, "temperature", "K", sensor=sT)
        RH30_pct = units.convert(RH30[:, 0], sRH.units, "humidity", sensor=sRH)
        q30 = rh_to_spec_hum(RH30_pct, P_kPa_lev, T30_K)        # kg/kg per period
        valid_q = ~np.isnan(q30)
        if valid_q.any():
            x_q = np.concatenate([[np.floor(t_q[valid_q][0])], t_q[valid_q]])
            y_q = np.concatenate([[q30[valid_q][0]], q30[valid_q]])
            q_fast_local = np.interp(t, x_q, y_q)

        def _pad(arr):
            if len(arr) >= N:
                return arr[:N]
            return np.concatenate([arr, np.full(N - len(arr), np.nan)])

        # q is g/kg; the all-NaN fallback keeps the kg/kg array, which carries
        # no value (units come from names.py, DECIDE 6).
        q_col = _pad(q30 * 1000.0) if valid_q.any() else _pad(q30)
        specific_hum_cols = [
            (f"q_{hn}", q_col),
            (f"theta_v_slow_{hn}", _pad(vt_avg30)),
            (f"r_{hn}", _pad(_avg30(r_slow))),
            (f"rho_air_moist_{hn}", _pad(_avg30(rho_moist_slow))),
            (f"rho_air_dry_{hn}", _pad(_avg30(rho_dry_slow))),
            (f"rho_h2o_{hn}", _pad(_avg30(rho_H2O_slow))),
        ]

    # Mean-humidity rescale of the sonic temperature, T = T_s/(1 + 0.51 q)
    # (Schotanus et al. 1983 eq. 5; library/writeups/sonic_temperature_flux.md).
    # MATLAB fluxes.m:575 used the virtual-temperature 0.61 ("modified by Diane").
    theta_son_air = (T_son + 273.15) / (1.0 + SONIC_HUMIDITY_COEFF * q_fast_local) - 273.15

    # ---- fine-wire at this height ----
    fw = theta_fw = Vtheta_fw = None
    fw_flag = np.zeros(N, dtype=bool)
    sfw = run.sensors.at("fw", height)
    if sfw is not None:
        fw = run.hf(sfw).copy()
        fw_flag = _flags(run, sfw, N)
        theta_fw = fw + GAMMA_DRY * (height - ref.z_ref)
        Vtheta_fw = theta_fw * (1.0 + 0.61 * q_fast_local)      # MATLAB fluxes.m:566

    # ---- H2O at this height ----
    # Resolved per height, not per run: a tower may carry an EC150-style
    # IRGA on one level and a LI-7500 (LiH2O, mmol/m³) on another.
    # The raw rhov / rhoCO2 columns are indexed over every IRGA level of the
    # tower (Sensors.heights_of), not within one family.
    h2o = None
    h2o_flag = np.zeros(N, dtype=bool)
    h2o_is_kh2o = False
    h2o_si = None
    h2o_levels = run.sensors.heights_of(H2O_FIELDS)
    if (s := run.sensors.at("irgaH2O", height)) is not None:
        h2o = units.convert(run.hf(s), s.units, "h2o_density", sensor=s)
        h2o_flag = (_flags(run, s, N)
                    | _threshold_flag(run, run.sensors.at("irgaH2OsigStrength", height), N,
                                      diag_cfg.get("H2OminSignal"), np.less_equal)
                    | _threshold_flag(run, run.sensors.at("irgaGasDiag", height), N,
                                      diag_cfg.get("meanGasDiagnosticLimit"), np.greater_equal))
        h2o_si = h2o_levels.index(height)
    elif (s := run.sensors.at("LiH2O", height)) is not None:
        h2o = run.hf(s) * 0.018          # mmol/m³ -> g/m³
        # LI-7500 diagnostic decreases with problems: full strength is 255,
        # ≤ meanLiGasDiagnosticLimit is bad (MATLAB fluxes.m:610 zeroes the
        # flag where diag > limit).
        h2o_flag = (_flags(run, s, N)
                    | _threshold_flag(run, run.sensors.at("LiGasDiag", height), N,
                                      diag_cfg.get("meanLiGasDiagnosticLimit"), np.less_equal))
        h2o_si = h2o_levels.index(height)
    elif (s := run.sensors.at("KH2O", height)) is not None:
        h2o = units.convert(run.hf(s), s.units, "h2o_density", sensor=s)
        h2o_flag = _flags(run, s, N)
        h2o_is_kh2o = True

    # ---- CO2 at this height (needs the hygrometer for the WPL terms) ----
    co2 = None
    co2_flag = np.zeros(N, dtype=bool)
    co2_si = None
    co2_levels = run.sensors.heights_of(CO2_FIELDS)
    if h2o is not None and (s := run.sensors.at("irgaCO2", height)) is not None:
        co2 = units.convert(run.hf(s), s.units, "co2_density", sensor=s)
        co2_flag = (_flags(run, s, N)
                    | _threshold_flag(run, run.sensors.at("irgaCO2sigStrength", height), N,
                                      diag_cfg.get("CO2minSignal"), np.less_equal)
                    | _threshold_flag(run, run.sensors.at("irgaGasDiag", height), N,
                                      diag_cfg.get("meanGasDiagnosticLimit"), np.greater_equal))
        co2_si = co2_levels.index(height)
    elif h2o is not None and (s := run.sensors.at("LiCO2", height)) is not None:
        co2 = run.hf(s) * 44.0           # mmol/m³ -> mg/m³
        co2_flag = (_flags(run, s, N)
                    | _threshold_flag(run, run.sensors.at("LiGasDiag", height), N,
                                      diag_cfg.get("meanLiGasDiagnosticLimit"), np.less_equal))
        co2_si = co2_levels.index(height)

    # ---- temperature: block-averaged derived temperatures ----
    def _avg(series):
        return block_average(t, series, avg_per)[1][:, 0]
    derived: List[Tuple[str, np.ndarray]] = []
    has_fw_here = fw is not None and theta_fw is not None
    if has_fw_here:
        derived.append((f"theta_fw_{hn}", _avg(theta_fw)))
    derived.append((f"theta_v_{hn}", _avg(theta_son)))
    if has_fw_here:
        derived.append((f"theta_v_fw_{hn}", _avg(Vtheta_fw)))
    derived.append((f"t_air_{hn}", _avg(theta_son_air)))

    # ---- period-mean wind direction (slope-normal heat flux) ----
    direction_avg = np.zeros(N)
    if run.wind is not None:
        hts = [float(h) for h in run.wind[HEIGHT].values]
        if height in hts:
            direction_avg = run.wind["direction"].values[:, hts.index(height)]

    return LevelInputs(
        ii, height, scan_hz, su.manufacturer if su.manufacturer is not None else 1,
        su.orientation if su.orientation is not None else 0.0,
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
