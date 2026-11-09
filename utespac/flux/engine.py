"""Per-period flux computation for one sonic level.

:func:`compute_period` takes a :class:`~.levels.LevelInputs`, the
:class:`~.reference.ReferenceState`, the run :class:`FluxOptions` and one
period's sample range and returns a :class:`PeriodResult`: the named
values of every output group (names as in :mod:`.tables`) and the
per-sample series the raw product stores. Formulas and flag masking are
those of ``fluxes.m`` with the audit corrections (Schotanus temperature
flux, WPL on w'T', slope geometry from SiteInfo).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import skew as scipy_skew

from ..calc_dissipation_rate import calc_dissipation_rate
from ..calc_snsp_angle import calc_snsp_angle
from ..calc_ssitc_flags import calc_ssitc_flags
from ..find_delta_flux import find_delta_flux
from ..find_delta_time import find_delta_time
from ..find_eta import find_eta
from ..nandetrend import nandetrend
from ..sonic_temperature import air_temperature_perturbation
from ..stability import psi_h, psi_m
from ..units import M_CO2
from .covariance import lag_max_abs_cov
from .levels import LevelInputs
from .reference import Md, Mv, ReferenceState
from .spectral_correction import massman2000_factor

KAPPA = 0.4
G = 9.81

SPECTRAL_CORRECTIONS = ("off", "massman2000")

# Geometry options a "massman2000" run needs, with the run-config key each
# one is read from; none has a default [HOUSE].
_MASSMAN_GEOMETRY = (
    ("sonic_path_len", "flux.sonicPathLength"),
    ("irga_path_len", "flux.irgaPathLength"),
    ("irga_sep_lateral", "flux.irgaSeparationLateral"),
    ("irga_sep_longitudinal", "flux.irgaSeparationLongitudinal"),
)


@dataclass
class FluxOptions:
    detrend: str = "linear"
    n_sub: int = 6                      # SSITC sub-periods per averaging period
    displacement_height: float = 0.0
    canopy_height: float = np.nan
    use_canopy_itc: bool = True
    latitude: Optional[float] = None
    calc_dissipation: bool = False
    scan_freq: float = 20.0             # [Hz] of the sonic's table (dissipation lag)
    angle: float = 0.0                  # slope angle [deg]
    downslope_aspect: float = 30.0      # fall-line direction from north [deg]
    stability_source: str = "hoegstroem1988"    # psi_m/psi_h coefficient set
    lag_search: str = "off"             # "off", "h2o_co2" or "all" (adds fine wire)
    lag_max_s: float = 2.0              # lag search half-window [s]
    avg_period_s: float = 1800.0        # averaging period [s]
    spectral_correction: str = "off"    # "off" or "massman2000"
    sonic_path_len: Optional[float] = None          # [m] vertical path
    irga_path_len: Optional[float] = None           # [m]
    irga_sep_lateral: Optional[float] = None        # [m]
    irga_sep_longitudinal: Optional[float] = None   # [m]

    def __post_init__(self):
        if self.spectral_correction not in SPECTRAL_CORRECTIONS:
            raise ValueError(
                "spectral_correction must be one of "
                f"{SPECTRAL_CORRECTIONS}, got {self.spectral_correction!r}")
        for attr, key in _MASSMAN_GEOMETRY:
            value = getattr(self, attr)
            if value is None:
                if self.spectral_correction == "massman2000":
                    raise ValueError(
                        f"spectral_correction 'massman2000' needs {key}")
            elif value < 0:
                raise ValueError(f"{key} must not be negative, got {value!r}")


@dataclass
class PeriodResult:
    values: Dict[str, Dict[str, float]] = field(default_factory=dict)
    samples: Dict[str, np.ndarray] = field(default_factory=dict)

    def put(self, table: str, key: str, value) -> None:
        self.values.setdefault(table, {})[key] = value

    def blank(self, table: str, *keys: str) -> None:
        for k in keys:
            self.values.setdefault(table, {})[k] = np.nan

    def get(self, table: str, key: str) -> float:
        return self.values.get(table, {}).get(key, np.nan)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """*x* advanced by *k* samples, ``y[i] = x[i + k]``, NaN where undefined."""
    y = np.full(len(x), np.nan)
    if abs(k) >= len(x):
        return y
    if k >= 0:
        y[:len(x) - k] = x[k:]
    else:
        y[-k:] = x[:len(x) + k]
    return y


def _align(w: np.ndarray, c: np.ndarray, fs: float, max_lag_s: float):
    """Scalar perturbation *c* shifted onto *w* by its covariance-peak lag.

    Returns the aligned series and the lag [s] from
    :func:`~.covariance.lag_max_abs_cov`; a zero lag returns *c* unchanged.
    """
    lag = lag_max_abs_cov(w, c, fs, max_lag_s)
    k = int(round(lag * fs))
    return (c if k == 0 else _shift(c, k)), lag


def _skew(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    if len(v) < 3:
        return np.nan
    return float(scipy_skew(v, bias=False))


def compute_period(lev: LevelInputs, ref: ReferenceState, opts: FluxOptions,
                   jj: int, s0: int, s1: int, t: np.ndarray) -> PeriodResult:
    """All flux statistics of level *lev* for period *jj* (samples ``s0:s1``)."""
    r = PeriodResult()
    det = opts.detrend
    unrot = bool(lev.unrot_flag[jj])
    rot = bool(lev.rot_flag[jj])
    tsf = bool(lev.Ts_flag[jj])
    fwf = bool(lev.fw_flag[jj])
    h2of = bool(lev.h2o_flag[jj])
    co2f = bool(lev.co2_flag[jj])

    # ---- detrended perturbations ----
    uP = nandetrend(lev.u[s0:s1], det)
    vP = nandetrend(lev.v[s0:s1], det)
    wP = nandetrend(lev.w[s0:s1], det)
    TsP = nandetrend(lev.T_son[s0:s1], det)
    ThvP = nandetrend(lev.theta_son[s0:s1], det)
    TairP = nandetrend(lev.theta_son_air[s0:s1], det)
    uPF_P = nandetrend(lev.u_pf[s0:s1], det)
    vPF_P = nandetrend(lev.v_pf[s0:s1], det)
    wPF_P = nandetrend(lev.w_pf[s0:s1], det)
    uTP = nandetrend(lev.u_tilt[s0:s1], det)
    vTP = nandetrend(lev.v_tilt[s0:s1], det)
    wTP = nandetrend(lev.w_tilt[s0:s1], det)

    # ---- sigma ----
    # Population std (ddof=0); MATLAB std uses N-1 (< 2e-5 relative at
    # n ~ 36 000; recorded divergence). Per-column flagging as fluxes.m
    # 775-799: w_sigma_raw by the w flag only, ts_sigma by the Tson flag only.
    r.put("sigma", "u_sigma_raw", np.nanstd(lev.u[s0:s1]))
    r.put("sigma", "v_sigma_raw", np.nanstd(lev.v[s0:s1]))
    r.put("sigma", "w_sigma_raw", np.nanstd(lev.w[s0:s1]))
    r.put("sigma", "u_sigma_pf", np.nanstd(lev.u_pf[s0:s1]))
    r.put("sigma", "v_sigma_pf", np.nanstd(lev.v_pf[s0:s1]))
    r.put("sigma", "w_sigma_pf", np.nanstd(lev.w_pf[s0:s1]))
    r.put("sigma", "ts_sigma", np.nanstd(lev.T_son[s0:s1]))
    r.put("transport", "ts_var_transport_pf", np.nanmean(wPF_P * TsP ** 2))
    if rot:
        r.blank("sigma", "u_sigma_raw", "v_sigma_raw", "u_sigma_pf", "v_sigma_pf", "w_sigma_pf")
        r.blank("transport", "ts_var_transport_pf")
    if unrot:
        r.blank("sigma", "w_sigma_raw")
    if tsf:
        r.blank("sigma", "ts_sigma")
        r.blank("transport", "ts_var_transport_pf")
    if lev.fw is not None:
        r.put("sigma", "t_fw_sigma", np.nan if fwf else np.nanstd(lev.fw[s0:s1]))
    r.put("sigma", "theta_v_sigma", np.nanstd(lev.theta_son[s0:s1]))

    # ---- R(uPF'wPF', wPF'Thetav') ----
    r.put("correlation", "u_w__w_theta_v_corr_pf", np.nan if (tsf or rot) else _corr(uPF_P * wPF_P, ThvP * wPF_P))

    # ---- eta, delta_flux, delta_time: momentum and heat ----
    if tsf or rot:
        r.blank("eta", "u_w_eta_pf", "w_theta_v_eta_pf")
        r.blank("delta_flux", "u_w_delta_flux_pf", "w_theta_v_delta_flux_pf")
        r.blank("delta_time", "u_w_delta_time_pf", "w_theta_v_delta_time_pf")
    else:
        r.put("eta", "u_w_eta_pf", find_eta(wPF_P, uPF_P))
        r.put("eta", "w_theta_v_eta_pf", find_eta(wPF_P, ThvP))
        r.put("delta_flux", "u_w_delta_flux_pf", find_delta_flux(wPF_P, uPF_P))
        r.put("delta_flux", "w_theta_v_delta_flux_pf", find_delta_flux(wPF_P, ThvP))
        r.put("delta_time", "u_w_delta_time_pf", find_delta_time(wPF_P, uPF_P))
        r.put("delta_time", "w_theta_v_delta_time_pf", find_delta_time(wPF_P, ThvP))

    # ---- tau ----
    tau_pf = np.sqrt(np.nanmean(uPF_P * wPF_P) ** 2 + np.nanmean(vPF_P * wPF_P) ** 2)
    tau_vals = {
        "Tau_raw": np.sqrt(np.nanmean(uP * wP) ** 2 + np.nanmean(vP * wP) ** 2),
        "Tau_pf": tau_pf,
        "u_var_pf": np.nanmean(uPF_P * uPF_P), "v_var_pf": np.nanmean(vPF_P * vPF_P),
        "w_var_pf": np.nanmean(wPF_P * wPF_P), "u_v_cov_pf": np.nanmean(uPF_P * vPF_P),
        "u_w_cov_pf": np.nanmean(uPF_P * wPF_P), "v_w_cov_pf": np.nanmean(vPF_P * wPF_P),
        "u_var_tilt": np.nanmean(uTP * uTP), "v_var_tilt": np.nanmean(vTP * vTP), "w_var_tilt": np.nanmean(wTP * wTP),
        "u_v_cov_tilt": np.nanmean(uTP * vTP), "u_w_cov_tilt": np.nanmean(uTP * wTP), "v_w_cov_tilt": np.nanmean(vTP * wTP),
    }
    if rot:
        tau_vals = {k: np.nan for k in tau_vals}
        tau_pf = np.nan
    r.values["momentum"] = tau_vals

    # ---- TKE ----
    r.put("tke", "TKE", np.nan if rot else
          0.5 * (np.nanmean(uP ** 2) + np.nanmean(vP ** 2) + np.nanmean(wP ** 2)))

    # ---- turbulent transport: w'e', w'u'w', w'theta_v'w' ----
    if rot:
        r.blank("transport", "TKE_transport_pf", "u_w_transport_pf", "w_theta_v_transport_pf")
    else:
        r.put("transport", "TKE_transport_pf", np.nanmean(wPF_P * 0.5 * (uPF_P ** 2 + vPF_P ** 2 + wPF_P ** 2)))
        r.put("transport", "u_w_transport_pf", np.nanmean(wPF_P * (uPF_P * wPF_P)))
        r.put("transport", "w_theta_v_transport_pf", np.nanmean(wPF_P * (ThvP * wPF_P)))

    # ---- dissipation ----
    if opts.calc_dissipation:
        u_mean_jj = np.nanmean(lev.u_pf[s0:s1])
        eps = calc_dissipation_rate(uPF_P, u_mean_jj, 1.0 / opts.scan_freq)
        r.put("dissipation", "epsilon", np.nan if rot else eps)

    # ---- slope-normal heat flux ----
    uTHv = np.nanmean(uTP * ThvP)
    vTHv = np.nanmean(vTP * ThvP)
    wTHv = np.nanmean(wTP * ThvP)
    wTHv_vert = np.nan
    if lev.direction_avg is not None and jj < len(lev.direction_avg):
        a1, a2 = calc_snsp_angle(float(lev.direction_avg[jj]), opts.angle, opts.downslope_aspect)
        wTHv_vert = (wTHv * np.cos(np.radians(opts.angle))
                     - uTHv * np.sin(np.radians(a1))
                     - vTHv * np.sin(np.radians(a2)))
    if rot or tsf:
        uTHv = vTHv = wTHv = wTHv_vert = np.nan
    r.values["sensible_heat_snsp"] = {
        "u_theta_v_cov_snsp": uTHv, "v_theta_v_cov_snsp": vTHv,
        "w_theta_v_cov_snsp": wTHv, "w_theta_v_cov_vertical": wTHv_vert}

    # ---- kinematic heat flux: w'Ts', w'theta_v' ----
    w_ts_cov = np.nanmean(wP * TsP)
    w_theta_v_cov = np.nanmean(wPF_P * ThvP)    # buoyancy flux (MATLAB kinSenFlux)
    r.put("sensible_heat", "w_ts_cov_raw", np.nan if (unrot or tsf) else w_ts_cov)
    r.put("sensible_heat", "w_theta_v_cov_pf", np.nan if (rot or tsf) else w_theta_v_cov)

    # ---- Obukhov length ----
    T0_L = (lev.virtual_theta_avg[jj]
            if lev.virtual_theta_avg is not None and jj < len(lev.virtual_theta_avg)
            else ref.T_virt_K[jj] if jj < len(ref.T_virt_K) else np.nan)
    u_star_cubed = tau_pf ** 1.5 if not np.isnan(tau_pf) else np.nan
    L_val = np.nan
    if not (np.isnan(u_star_cubed) or np.isnan(wTHv_vert) or wTHv_vert == 0 or np.isnan(T0_L)):
        L_val = -u_star_cubed / (KAPPA * G / T0_L * wTHv_vert)
    if rot or tsf:
        L_val = np.nan
    r.put("obukhov", "L", L_val)

    # ---- surface-layer scales (Stull 1988 pp. 356-357) ----
    # theta_star = -w'theta_v'/u* [K]; q_star below, once w'h2o' exists.
    ustar = np.sqrt(tau_pf) if (not np.isnan(tau_pf) and tau_pf > 0) else np.nan
    r.put("scaling", "ustar_pf", ustar)
    r.put("scaling", "theta_star",
          np.nan if (rot or tsf or np.isnan(ustar)) else -w_theta_v_cov / ustar)

    # ---- integrated stability functions psi_m(z/L), psi_h(z/L) ----
    # (Foken 2008 eqs. 2.85-2.89); NaN wherever L is (rot/tsf masks included)
    zeta = lev.height / L_val if not np.isnan(L_val) else np.nan
    r.put("scaling", "psi_m", psi_m(zeta, opts.stability_source))
    r.put("scaling", "psi_h", psi_h(zeta, opts.stability_source))

    # ---- w'T_air' (mean-humidity rescale; replaced below
    #      by the Schotanus form where high-frequency humidity exists) ----
    r.put("sensible_heat", "w_t_air_cov_raw", np.nan if (unrot or tsf) else np.nanmean(wP * TairP))
    r.put("sensible_heat", "w_t_air_cov_pf", np.nan if (rot or tsf) else np.nanmean(wPF_P * TairP))

    # ---- w'theta_s' ----
    r.put("sensible_heat", "w_theta_s_cov_raw", np.nan if (unrot or tsf) else np.nanmean(wP * ThvP))
    r.put("sensible_heat", "w_theta_s_cov_pf", np.nan if (rot or tsf) else np.nanmean(wPF_P * ThvP))

    # ---- lateral heat flux: u'Ts', v'Ts', u'theta_s', v'theta_s' ----
    hl = {"u_ts_cov_raw": np.nanmean(uP * TsP), "u_ts_cov_pf": np.nanmean(uPF_P * TsP),
          "v_ts_cov_raw": np.nanmean(vP * TsP), "v_ts_cov_pf": np.nanmean(vPF_P * TsP),
          "u_theta_s_cov_raw": np.nanmean(uP * ThvP), "u_theta_s_cov_pf": np.nanmean(uPF_P * ThvP),
          "v_theta_s_cov_raw": np.nanmean(vP * ThvP), "v_theta_s_cov_pf": np.nanmean(vPF_P * ThvP)}
    if rot or tsf:
        hl = {k: np.nan for k in hl}
    r.values["sensible_heat_lateral"] = hl

    # ---- fine-wire fluxes ----
    if lev.fw is not None:
        fwP = nandetrend(lev.fw[s0:s1], det)
        thFwP = nandetrend(lev.theta_fw[s0:s1], det)
        VthFwP = nandetrend(lev.Vtheta_fw[s0:s1], det)
        if opts.lag_search == "all":
            # One lag for the three fine-wire series, found on theta_fw',
            # searched only in mode "all" [ASSUMED].
            thFwP, lag_fw = _align(wPF_P, thFwP, opts.scan_freq, opts.lag_max_s)
            k_fw = int(round(lag_fw * opts.scan_freq))
            if k_fw:
                fwP, VthFwP = _shift(fwP, k_fw), _shift(VthFwP, k_fw)
        r.put("sensible_heat", "w_t_fw_cov_raw", np.nanmean(wP * fwP))
        r.put("sensible_heat", "w_t_fw_cov_pf", np.nanmean(wPF_P * fwP))
        r.put("sensible_heat", "w_theta_fw_cov_raw", np.nanmean(wP * thFwP))
        r.put("sensible_heat", "w_theta_fw_cov_pf", np.nanmean(wPF_P * thFwP))
        r.put("sensible_heat", "w_theta_v_fw_cov_raw", np.nanmean(wP * VthFwP))
        r.put("sensible_heat", "w_theta_v_fw_cov_pf", np.nanmean(wPF_P * VthFwP))
        if unrot or fwf:
            r.blank("sensible_heat", "w_t_fw_cov_raw", "w_theta_fw_cov_raw", "w_theta_v_fw_cov_raw")
        if rot or fwf:
            r.blank("sensible_heat", "w_t_fw_cov_pf", "w_theta_fw_cov_pf", "w_theta_v_fw_cov_pf")
        r.put("sensible_heat_lateral", "u_theta_fw_cov_raw", np.nanmean(uP * thFwP))
        r.put("sensible_heat_lateral", "u_theta_fw_cov_pf", np.nanmean(uPF_P * thFwP))
        r.put("sensible_heat_lateral", "v_theta_fw_cov_raw", np.nanmean(vP * thFwP))
        r.put("sensible_heat_lateral", "v_theta_fw_cov_pf", np.nanmean(vPF_P * thFwP))
        if rot or fwf:
            r.blank("sensible_heat_lateral", "u_theta_fw_cov_raw", "u_theta_fw_cov_pf",
                    "v_theta_fw_cov_raw", "v_theta_fw_cov_pf")
        r.samples["fwThPrime"] = thFwP
        r.samples["fwTh"] = lev.theta_fw[s0:s1]
        r.samples["fwT"] = lev.fw[s0:s1]

    # ====================================================================
    # H2O-based fluxes (latent_heat, sigma/eta/correlation, skewness,
    # scalar_flux_lateral, transport)
    # ====================================================================
    H2Op = rhov_ext = rho_CO2p = rhoc_ext = None
    if lev.h2o is not None:
        H2Op = nandetrend(lev.h2o[s0:s1], det)
        if opts.lag_search != "off":
            # Aligned onto wPF' before every product that uses it (Schotanus
            # T', WPL terms, covariances, SSITC); the stored perturbation
            # samples are the aligned series. Sigma and means stay on the
            # unshifted record, except co2_wpl_sigma, whose rho_c is shifted
            # by the CO2 lag to pair with rhoc_ext (built from aligned H2O')
            # [ASSUMED].
            H2Op, lag_h2o = _align(wPF_P, H2Op, opts.scan_freq, opts.lag_max_s)
            r.put("latent_heat", "lag_h2o", np.nan if (rot or h2of) else lag_h2o)
        rho_v_j = ref.rho_v[jj] if jj < len(ref.rho_v) else np.nan
        rho_d_j = ref.rho_d[jj] if jj < len(ref.rho_d) else np.nan
        T_ref_j = ref.T_K[jj] if jj < len(ref.T_K) else np.nan

        # Schotanus correction with high-frequency humidity: T' = Ts' - 0.51 T q'
        # (Schotanus 1983 eq. 6; Liu et al. 2001 eq. 10), so w'T' = w'Ts' -
        # 0.51 T w'q' (eq. 8 / eq. 12). It replaces the mean-humidity rescale in
        # the T_air columns and is the temperature flux every WPL term uses.
        qP = (H2Op / 1000.0) / (rho_d_j + rho_v_j)                       # kg/kg
        T_air_mean_K = np.nanmean(lev.theta_son_air[s0:s1]) + 273.15
        TairP = air_temperature_perturbation(TsP, qP, T_air_mean_K)
        r.put("sensible_heat", "w_t_air_cov_raw", np.nan if (unrot or tsf or h2of) else np.nanmean(wP * TairP))
        r.put("sensible_heat", "w_t_air_cov_pf", np.nan if (rot or tsf or h2of) else np.nanmean(wPF_P * TairP))

        # WPL external H2O fluctuation
        rhov_ext = (Md / Mv * (rho_v_j / rho_d_j) * (H2Op / 1000.0)
                    + rho_v_j * (1.0 + Md / Mv * rho_v_j / rho_d_j) * TairP / T_ref_j)

        r.put("sigma", "h2o_sigma", np.nan if (tsf or rot) else np.nanstd(lev.h2o[s0:s1]))

        if tsf or rot:
            r.blank("correlation", "u_w__w_h2o_corr_pf", "w_h2o__w_theta_v_corr_pf",
                    "w_u_corr_pf", "w_theta_v_corr_pf", "w_h2o_corr_pf")
            r.blank("eta", "w_h2o_eta_pf", "w_h2o_wpl_eta_pf")
            r.blank("delta_flux", "w_h2o_delta_flux_pf", "w_h2o_wpl_delta_flux_pf")
            r.blank("delta_time", "w_h2o_delta_time_pf", "w_h2o_wpl_delta_time_pf")
        else:
            r.put("correlation", "u_w__w_h2o_corr_pf", _corr(uPF_P * wPF_P, H2Op * wPF_P))
            r.put("correlation", "w_h2o__w_theta_v_corr_pf", _corr(H2Op * wPF_P, ThvP * wPF_P))
            r.put("correlation", "w_u_corr_pf", _corr(wPF_P, uPF_P))
            r.put("correlation", "w_theta_v_corr_pf", _corr(wPF_P, ThvP))
            r.put("correlation", "w_h2o_corr_pf", _corr(wPF_P, H2Op))
            r.put("eta", "w_h2o_eta_pf", find_eta(wPF_P, H2Op))
            r.put("eta", "w_h2o_wpl_eta_pf", find_eta(wPF_P, H2Op + rhov_ext * 1e3))
            r.put("delta_flux", "w_h2o_delta_flux_pf", find_delta_flux(wPF_P, H2Op))
            r.put("delta_flux", "w_h2o_wpl_delta_flux_pf", find_delta_flux(wPF_P, H2Op + rhov_ext * 1e3))
            r.put("delta_time", "w_h2o_delta_time_pf", find_delta_time(wPF_P, H2Op))
            r.put("delta_time", "w_h2o_wpl_delta_time_pf", find_delta_time(wPF_P, H2Op + rhov_ext * 1e3))

        # latent heat (all-NaN periods produce nan silently)
        with np.errstate(all="ignore"):
            E = np.nanmean(wP * H2Op)
            EPF = np.nanmean(wPF_P * H2Op)
        Lv = (2.501 - 0.00237 * (T_ref_j - 273.15)) * 1e3
        # Temperature flux for the WPL terms (Webb et al. 1980 eqs. 24-25, 44):
        # w'T', i.e. the Schotanus-corrected T_air'wPF' column. MATLAB
        # (fluxes.m:1073) used the buoyancy flux Theta_v'wPF'.
        kin_sen_flux = r.get("sensible_heat", "w_t_air_cov_pf")
        wpl = 1.0 + Md / Mv * rho_v_j / rho_d_j

        r.put("latent_heat", "Lv", Lv)
        r.put("latent_heat", "w_h2o_cov_raw", np.nan if (unrot or h2of) else E)
        r.put("latent_heat", "w_h2o_cov_pf", np.nan if (rot or h2of) else EPF)
        r.put("latent_heat", "w_h2o_wpl_cov_pf", np.nan if (rot or h2of) else
              np.nanmean(wPF_P * (H2Op + rhov_ext * 1e3)))
        r.put("latent_heat", "w_q_wpl_cov_pf", np.nan if (rot or h2of) else
              wpl * (np.nanmean(wPF_P * H2Op) / 1e3 + rho_v_j / T_ref_j * np.nanmean(wPF_P * TairP)))
        LE_w = 1000.0 * Lv * wpl * (E / 1000.0 + rho_v_j / T_ref_j * kin_sen_flux)
        LE_pf = 1000.0 * Lv * wpl * (EPF / 1000.0 + rho_v_j / T_ref_j * kin_sen_flux)
        r.put("latent_heat", "LE_wpl_raw", np.nan if (unrot or h2of) else LE_w)
        r.put("latent_heat", "LE_wpl_pf", np.nan if (unrot or h2of) else LE_pf)

        # q_star = -w'q'/u* [g/kg] with w'q' = w'h2o'_pf / rho_moist (specific
        # humidity from the raw wPF'' covariance; Stull 1988 pp. 356-357)
        r.put("scaling", "q_star",
              np.nan if (rot or h2of or np.isnan(ustar)) else
              -(EPF / (rho_d_j + rho_v_j)) / ustar)

        # KH2O O2 correction (Tanner et al. 1993)
        if lev.h2o_is_kh2o:
            ko = -0.0045
            kw = -0.153
            CkO = 0.23 * ko / kw
            O2_corr = CkO * rho_d_j / T_ref_j * kin_sen_flux * 1000.0
            E += O2_corr
            EPF += O2_corr
            r.put("latent_heat", "LE_o2_raw", Lv * E)
            r.put("latent_heat", "LE_o2_pf", Lv * EPF)

        r.samples["rhov"] = lev.h2o[s0:s1]
        r.samples["rhovPrime"] = H2Op
        r.samples["rhovextenalPrime"] = rhov_ext * 1e3

        # ================================================================
        # CO2-based fluxes (requires H2O to exist first for evapFlux)
        # ================================================================
        if lev.co2 is not None:
            rho_CO2_seg = lev.co2[s0:s1]                      # mg/m³
            rho_CO2p = nandetrend(rho_CO2_seg, det) / 1e6     # kg/m³
            k_co2 = 0
            if opts.lag_search != "off":
                rho_CO2p, lag_co2 = _align(wPF_P, rho_CO2p, opts.scan_freq, opts.lag_max_s)
                k_co2 = int(round(lag_co2 * opts.scan_freq))
                r.put("co2_flux", "lag_co2", np.nan if (rot or co2f) else lag_co2)
            rho_CO2_avg = np.nanmean(rho_CO2_seg) / 1e6       # kg/m³
            evap_flux = r.get("latent_heat", "LE_wpl_pf") / Lv / 1000.0   # WPL wPF'' (kg/m²/s)

            # WPL external CO2 fluctuation
            rhoc_ext = (Md / Mv * (rho_CO2_avg / rho_d_j) * (H2Op / 1000.0)
                        + rho_CO2_avg * (1.0 + Md / Mv * rho_v_j / rho_d_j) * TairP / T_ref_j)

            # ppm CO2 (period mean) with the height-specific per-sample pressure
            if lev.P_raw_hf is not None:
                P_seg_kPa = np.interp(t[s0:s1], lev.P_t_hf, lev.P_raw_hf)
            elif lev.P_kPa is not None and jj < len(lev.P_kPa):
                P_seg_kPa = np.full(s1 - s0, lev.P_kPa[jj])
            else:
                P_seg_kPa = np.full(s1 - s0, ref.P_ref_kPa)
            ppm_CO2 = np.nanmean(rho_CO2_seg * 1000.0 * 8.314
                                 * (lev.theta_son_air[s0:s1] + 273.15)
                                 / (M_CO2 * P_seg_kPa * 1e3))

            r.put("sigma", "co2_sigma", np.nanstd(rho_CO2_seg))
            r.put("sigma", "co2_wpl_sigma",
                  np.nanstd((_shift(rho_CO2_seg, k_co2) if k_co2 else rho_CO2_seg) / 1e6 + rhoc_ext) * 1e6)
            if tsf or rot:
                r.blank("sigma", "co2_sigma", "co2_wpl_sigma")

            # R CO2 (MATLAB cols 5-17 order)
            if tsf or rot:
                r.blank("correlation", "u_w__w_co2_wpl_corr_pf", "w_co2_wpl__w_theta_v_corr_pf",
                        "w_co2_corr_pf", "w_co2_wpl_corr_pf",
                        "u_w__w_co2_corr_pf", "w_co2__w_theta_v_corr_pf",
                        "u_w__w_h2o_wpl_corr_pf", "w_h2o_wpl__w_theta_v_corr_pf", "w_h2o_wpl_corr_pf")
                r.blank("eta", "w_co2_wpl_eta_pf", "w_co2_eta_pf")
                r.blank("delta_flux", "w_co2_wpl_delta_flux_pf", "w_co2_delta_flux_pf")
                r.blank("delta_time", "w_co2_wpl_delta_time_pf", "w_co2_delta_time_pf")
            else:
                r.put("correlation", "u_w__w_co2_wpl_corr_pf", _corr(uPF_P * wPF_P, (rho_CO2p + rhoc_ext) * wPF_P))
                r.put("correlation", "w_co2_wpl__w_theta_v_corr_pf", _corr((rho_CO2p + rhoc_ext) * wPF_P, ThvP * wPF_P))
                r.put("correlation", "w_co2_corr_pf", _corr(wPF_P, rho_CO2p))
                r.put("correlation", "w_co2_wpl_corr_pf", _corr(wPF_P, rho_CO2p + rhoc_ext))
                r.put("correlation", "u_w__w_co2_corr_pf", _corr(uPF_P * wPF_P, rho_CO2p * wPF_P))
                r.put("correlation", "w_co2__w_theta_v_corr_pf", _corr(rho_CO2p * wPF_P, ThvP * wPF_P))
                r.put("correlation", "u_w__w_h2o_wpl_corr_pf", _corr(uPF_P * wPF_P, (H2Op + rhov_ext * 1e3) * wPF_P))
                r.put("correlation", "w_h2o_wpl__w_theta_v_corr_pf", _corr((H2Op + rhov_ext * 1e3) * wPF_P, ThvP * wPF_P))
                r.put("correlation", "w_h2o_wpl_corr_pf", _corr(wPF_P, H2Op + rhov_ext * 1e3))
                r.put("eta", "w_co2_wpl_eta_pf", find_eta(wPF_P, rho_CO2p + rhoc_ext))
                r.put("eta", "w_co2_eta_pf", find_eta(wPF_P, rho_CO2p))
                r.put("delta_flux", "w_co2_wpl_delta_flux_pf", find_delta_flux(wPF_P, rho_CO2p + rhoc_ext))
                r.put("delta_flux", "w_co2_delta_flux_pf", find_delta_flux(wPF_P, rho_CO2p))
                r.put("delta_time", "w_co2_wpl_delta_time_pf", find_delta_time(wPF_P, rho_CO2p + rhoc_ext))
                r.put("delta_time", "w_co2_delta_time_pf", find_delta_time(wPF_P, rho_CO2p))

            # skewness, lateral flux contributions, turbulent transport
            if rot:
                r.blank("skewness", "u_skew_pf", "v_skew_pf", "w_skew_pf", "theta_v_skew",
                        "h2o_skew", "h2o_wpl_skew", "co2_skew", "co2_wpl_skew")
                r.blank("scalar_flux_lateral", "u_theta_v_cov_pf", "v_theta_v_cov_pf",
                        "u_h2o_cov_pf", "v_h2o_cov_pf", "u_h2o_wpl_cov_pf", "v_h2o_wpl_cov_pf",
                        "u_co2_cov_pf", "v_co2_cov_pf", "u_co2_wpl_cov_pf", "v_co2_wpl_cov_pf")
                r.blank("transport", "w_h2o_transport_pf", "w_h2o_wpl_transport_pf",
                        "w_co2_transport_pf", "w_co2_wpl_transport_pf")
            else:
                r.values["skewness"] = {
                    "u_skew_pf": _skew(uPF_P), "v_skew_pf": _skew(vPF_P), "w_skew_pf": _skew(wPF_P),
                    "theta_v_skew": _skew(ThvP), "h2o_skew": _skew(H2Op),
                    "h2o_wpl_skew": _skew(H2Op + rhov_ext * 1e3),
                    "co2_skew": _skew(rho_CO2p), "co2_wpl_skew": _skew(rho_CO2p + rhoc_ext)}
                r.values["scalar_flux_lateral"] = {
                    "u_theta_v_cov_pf": np.nanmean(uPF_P * ThvP),
                    "v_theta_v_cov_pf": np.nanmean(vPF_P * ThvP),
                    "u_h2o_cov_pf": np.nanmean(uPF_P * H2Op),
                    "v_h2o_cov_pf": np.nanmean(vPF_P * H2Op),
                    "u_h2o_wpl_cov_pf": np.nanmean(uPF_P * (H2Op + rhov_ext * 1e3)),
                    "v_h2o_wpl_cov_pf": np.nanmean(vPF_P * (H2Op + rhov_ext * 1e3)),
                    "u_co2_cov_pf": np.nanmean(uPF_P * rho_CO2p),
                    "v_co2_cov_pf": np.nanmean(vPF_P * rho_CO2p),
                    "u_co2_wpl_cov_pf": np.nanmean(uPF_P * (rho_CO2p + rhoc_ext)),
                    "v_co2_wpl_cov_pf": np.nanmean(vPF_P * (rho_CO2p + rhoc_ext))}
                r.put("transport", "w_h2o_transport_pf", np.nanmean(wPF_P * (H2Op * wPF_P)))
                r.put("transport", "w_h2o_wpl_transport_pf", np.nanmean(wPF_P * ((H2Op + rhov_ext * 1e3) * wPF_P)))
                r.put("transport", "w_co2_transport_pf", np.nanmean(wPF_P * (rho_CO2p * wPF_P)))
                r.put("transport", "w_co2_wpl_transport_pf", np.nanmean(wPF_P * ((rho_CO2p + rhoc_ext) * wPF_P)))

            # CO2 flux
            w_co2_cov_raw = np.nanmean(wP * rho_CO2p)
            w_co2_cov_pf = np.nanmean(wPF_P * rho_CO2p)
            r.put("co2_flux", "w_co2_cov_raw", np.nan if (unrot or co2f) else w_co2_cov_raw)
            r.put("co2_flux", "w_co2_cov_pf", np.nan if (rot or co2f) else w_co2_cov_pf)
            w_co2_cov_pf_stored = r.get("co2_flux", "w_co2_cov_pf")
            r.put("co2_flux", "Fc_wpl_pf", np.nan if (rot or co2f) else
                  (w_co2_cov_pf_stored + Md / Mv * (rho_CO2_avg / rho_d_j) * evap_flux
                   + (1.0 + Md / Mv * rho_v_j / rho_d_j) * (rho_CO2_avg / T_ref_j) * kin_sen_flux))
            r.put("co2_flux", "w_co2_wpl_cov_pf", np.nan if (rot or co2f) else
                  w_co2_cov_pf_stored + np.nanmean(wPF_P * rhoc_ext))
            r.put("co2_flux", "co2_mole_fraction", ppm_CO2)

            # raw CO2: stored in mg/m³ (rhoCO2Prime = rho_CO2p * 1e6), unlike
            # MATLAB which left the kg/m³ of the WPL arithmetic (fluxes.m:1212).
            r.samples["rhoCO2"] = rho_CO2_seg
            r.samples["rhoCO2Prime"] = rho_CO2p * 1e6
            r.samples["rhoCO2extenalPrime"] = rhoc_ext * 1e6

    # ---- H in W m-2 (DECIDE 10): rho cp times the temperature covariance,
    #      with the reference-level rho and cp the group also stores ----
    rho_cp = ((ref.rho[jj] if jj < len(ref.rho) else np.nan)
              * (ref.cp[jj] if jj < len(ref.cp) else np.nan))
    r.put("sensible_heat", "H_raw", rho_cp * r.get("sensible_heat", "w_t_air_cov_raw"))
    r.put("sensible_heat", "H_pf", rho_cp * r.get("sensible_heat", "w_t_air_cov_pf"))
    r.put("sensible_heat", "H_buoyancy_pf", rho_cp * r.get("sensible_heat", "w_theta_v_cov_pf"))

    # ---- SSITC + SS-only quality flags (ForestComplexTerrain) ----
    has_co2 = lev.h2o is not None and lev.co2 is not None
    (tau_ssitc, tau_ss, H_ssitc, H_ss, LE_ssitc, LE_ss, FC_ssitc, FC_ss) = calc_ssitc_flags(
        wPF_P, uPF_P, vPF_P, ThvP,
        H2Op, rhov_ext,
        rho_CO2p if has_co2 else None, rhoc_ext if has_co2 else None,
        ustar, L_val,
        rot, tsf, h2of, co2f,
        opts.n_sub, lev.height, opts.displacement_height,
        canopy_height=opts.canopy_height,
        use_canopy_itc=opts.use_canopy_itc,
        latitude=opts.latitude,
    )
    r.values["flux_qc"] = {"Tau_ssitc": tau_ssitc, "Tau_ss": tau_ss, "H_ssitc": H_ssitc, "H_ss": H_ss,
                           "LE_ssitc": LE_ssitc, "LE_ss": LE_ss, "Fc_ssitc": FC_ssitc, "Fc_ss": FC_ss}

    # ---- Massman (2000) spectral correction factors: stored beside the
    #      flags, never multiplied into the stored fluxes. Their z/L is
    #      (z - d)/L, the height the factor's f_x also takes. Each factor
    #      follows the stored flux it is named for, so a flag that masks
    #      the flux masks the factor ----
    if opts.spectral_correction == "massman2000":
        r.values["flux_qc"].update(_massman_factors(
            lev, opts, s0, s1,
            zeta * (lev.height - opts.displacement_height) / lev.height,
            {"H": r.get("sensible_heat", "H_pf"),
             "LE": r.get("latent_heat", "LE_wpl_pf"),
             "Fc": r.get("co2_flux", "Fc_wpl_pf")}, has_co2))
    return r


def _massman_factors(lev, opts, s0, s1, zeta, fluxes, has_co2):
    """Massman (2000) spectral correction factors of H, LE and Fc.

    Each factor is that of the leading covariance of its flux: w'Ts' for
    ``H_pf``, w'h2o' for ``LE_wpl_pf`` and w'co2' for ``Fc_wpl_pf``.

    Parameters
    ----------
    lev : LevelInputs
        The sonic level.
    opts : FluxOptions
        Run options with ``spectral_correction == "massman2000"``.
    s0, s1 : int
        Sample range of the period.
    zeta : float
        Stability ``(z - d)/L`` of the period; NaN where ``L`` is, which
        includes every rotation- or Ts-flagged period.
    fluxes : dict of str to float
        The stored ``H_pf``, ``LE_wpl_pf`` and ``Fc_wpl_pf`` of the
        period, keyed ``"H"``, ``"LE"`` and ``"Fc"``.
    has_co2 : bool
        Whether the level carries CO2 next to H2O.

    Returns
    -------
    dict
        ``scf_H``, plus ``scf_LE`` where the level has H2O and
        ``scf_Fc`` where it has CO2; NaN where the stored flux is NaN,
        where z/L is undefined or where ``U`` is not positive.
    """
    keys = ["H"] + (["LE"] if lev.h2o is not None else []) \
        + (["Fc"] if has_co2 else [])
    out = {f"scf_{k}": np.nan for k in keys}
    # U is the rotated mean along-wind speed, as for dissipation; z_eff is
    # z - d, the height zeta also uses (Massman 2000 takes one z for both).
    # tau_sensor stays at 0 (fast open-path) and the anti-noise filter off
    # [ASSUMED].
    common = dict(U=float(np.nanmean(lev.u_pf[s0:s1])),
                  z_eff=lev.height - opts.displacement_height,
                  zeta=zeta, fs=opts.scan_freq,
                  avg_period_s=opts.avg_period_s, detrend=opts.detrend)
    # Sonic temperature is line-averaged on the sonic path, with no
    # separation from w [ASSUMED].
    geometry = {"H": dict(path_len_sonic=opts.sonic_path_len,
                          path_len_scalar=opts.sonic_path_len)}
    # The separations are taken as wind-direction independent, the same
    # for every period, and one geometry serves every sonic level
    # [ASSUMED]. The longitudinal term is kept whether or not lagSearch
    # shifted the gas series [ASSUMED].
    irga = dict(path_len_sonic=opts.sonic_path_len,
                path_len_scalar=opts.irga_path_len,
                sep_lateral=opts.irga_sep_lateral,
                sep_longitudinal=opts.irga_sep_longitudinal)
    geometry["LE"] = geometry["Fc"] = irga
    for k in keys:
        if not np.isnan(fluxes[k]):
            out[f"scf_{k}"] = float(
                massman2000_factor(**common, **geometry[k]))
    return out
