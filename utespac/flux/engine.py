"""Per-period flux computation for one sonic level.

:func:`compute_period` takes a :class:`~.levels.LevelInputs`, the
:class:`~.reference.ReferenceState`, the run :class:`FluxOptions` and one
period's sample range and returns a :class:`PeriodResult`: the named
values of every output table (keys as in :mod:`.tables`) and the
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
from .levels import LevelInputs
from .reference import Md, Mv, ReferenceState

KAPPA = 0.4
G = 9.81


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
    # 775-799: sigma_w by the w flag only, sigma_Tson by the Tson flag only.
    r.put("sigma", "sigma_u", np.nanstd(lev.u[s0:s1]))
    r.put("sigma", "sigma_v", np.nanstd(lev.v[s0:s1]))
    r.put("sigma", "sigma_w", np.nanstd(lev.w[s0:s1]))
    r.put("sigma", "sigma_uPF", np.nanstd(lev.u_pf[s0:s1]))
    r.put("sigma", "sigma_vPF", np.nanstd(lev.v_pf[s0:s1]))
    r.put("sigma", "sigma_wPF", np.nanstd(lev.w_pf[s0:s1]))
    r.put("sigma", "sigma_Tson", np.nanstd(lev.T_son[s0:s1]))
    r.put("sigma", "wPFP_TsonP_TsonP", np.nanmean(wPF_P * TsP ** 2))
    if rot:
        r.blank("sigma", "sigma_u", "sigma_v", "sigma_uPF", "sigma_vPF", "sigma_wPF", "wPFP_TsonP_TsonP")
    if unrot:
        r.blank("sigma", "sigma_w")
    if tsf:
        r.blank("sigma", "sigma_Tson", "wPFP_TsonP_TsonP")
    if lev.fw is not None:
        r.put("sigma", "sigma_TFW", np.nan if fwf else np.nanstd(lev.fw[s0:s1]))
    r.put("sigma", "sigma_Theta_v", np.nanstd(lev.theta_son[s0:s1]))

    # ---- R(uPF'wPF', wPF'Thetav') ----
    r.put("R", "R_uPFwPF_wPFThetav", np.nan if (tsf or rot) else _corr(uPF_P * wPF_P, ThvP * wPF_P))

    # ---- eta, delta_flux, delta_time: momentum and heat ----
    if tsf or rot:
        r.blank("eta", "eta_wPFuPF", "eta_wPFThetav")
        r.blank("delta_flux_ctrb", "S_wPFuPF", "S_wPFThetav")
        r.blank("delta_time_ctrb", "D_wPFuPF", "D_wPFThetav")
    else:
        r.put("eta", "eta_wPFuPF", find_eta(wPF_P, uPF_P))
        r.put("eta", "eta_wPFThetav", find_eta(wPF_P, ThvP))
        r.put("delta_flux_ctrb", "S_wPFuPF", find_delta_flux(wPF_P, uPF_P))
        r.put("delta_flux_ctrb", "S_wPFThetav", find_delta_flux(wPF_P, ThvP))
        r.put("delta_time_ctrb", "D_wPFuPF", find_delta_time(wPF_P, uPF_P))
        r.put("delta_time_ctrb", "D_wPFThetav", find_delta_time(wPF_P, ThvP))

    # ---- tau ----
    tau_pf = np.sqrt(np.nanmean(uPF_P * wPF_P) ** 2 + np.nanmean(vPF_P * wPF_P) ** 2)
    tau_vals = {
        "tau_raw": np.sqrt(np.nanmean(uP * wP) ** 2 + np.nanmean(vP * wP) ** 2),
        "tau_PF": tau_pf,
        "uPF_uPF": np.nanmean(uPF_P * uPF_P), "vPF_vPF": np.nanmean(vPF_P * vPF_P),
        "wPF_wPF": np.nanmean(wPF_P * wPF_P), "uPF_vPF": np.nanmean(uPF_P * vPF_P),
        "uPF_wPF": np.nanmean(uPF_P * wPF_P), "vPF_wPF": np.nanmean(vPF_P * wPF_P),
        "uT_uT": np.nanmean(uTP * uTP), "vT_vT": np.nanmean(vTP * vTP), "wT_wT": np.nanmean(wTP * wTP),
        "uT_vT": np.nanmean(uTP * vTP), "uT_wT": np.nanmean(uTP * wTP), "vT_wT": np.nanmean(vTP * wTP),
    }
    if rot:
        tau_vals = {k: np.nan for k in tau_vals}
        tau_pf = np.nan
    r.values["tau"] = tau_vals

    # ---- TKE ----
    r.put("tke", "tke", np.nan if rot else
          0.5 * (np.nanmean(uP ** 2) + np.nanmean(vP ** 2) + np.nanmean(wP ** 2)))

    # ---- turbulent transport: w'e', w'u'w', w'Thv'w' ----
    if rot:
        r.blank("turbtr", "w_e", "w_uw", "w_thvw")
    else:
        r.put("turbtr", "w_e", np.nanmean(wPF_P * 0.5 * (uPF_P ** 2 + vPF_P ** 2 + wPF_P ** 2)))
        r.put("turbtr", "w_uw", np.nanmean(wPF_P * (uPF_P * wPF_P)))
        r.put("turbtr", "w_thvw", np.nanmean(wPF_P * (ThvP * wPF_P)))

    # ---- dissipation ----
    if opts.calc_dissipation:
        u_mean_jj = np.nanmean(lev.u_pf[s0:s1])
        eps = calc_dissipation_rate(uPF_P, u_mean_jj, 1.0 / opts.scan_freq)
        r.put("epsilon", "epsilon", np.nan if rot else eps)

    # ---- H_SNSP (slope-normal heat flux) ----
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
    r.values["H_SNSP"] = {"uTHv": uTHv, "vTHv": vTHv, "wTHv": wTHv, "wTHv_vert": wTHv_vert}

    # ---- H: Ts'w', Thetav'wPF' ----
    Ts_w = np.nanmean(wP * TsP)
    Thv_wPF = np.nanmean(wPF_P * ThvP)          # buoyancy flux (MATLAB kinSenFlux)
    r.put("H", "Ts_w", np.nan if (unrot or tsf) else Ts_w)
    r.put("H", "Thv_wPF", np.nan if (rot or tsf) else Thv_wPF)

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
    r.put("L", "L", L_val)

    # ---- H: T_air'w', T_air'wPF' (mean-humidity rescale; replaced below
    #      by the Schotanus form where high-frequency humidity exists) ----
    r.put("H", "Tair_w", np.nan if (unrot or tsf) else np.nanmean(wP * TairP))
    r.put("H", "Tair_wPF", np.nan if (rot or tsf) else np.nanmean(wPF_P * TairP))

    # ---- H: Th_s'w', Th_s'wPF' ----
    r.put("H", "Ths_w", np.nan if (unrot or tsf) else np.nanmean(wP * ThvP))
    r.put("H", "Ths_wPF", np.nan if (rot or tsf) else np.nanmean(wPF_P * ThvP))

    # ---- Hlat: Ts'u', Ts'uPF', Ts'v', Ts'vPF'; Th_s'... ----
    hl = {"Ts_u": np.nanmean(uP * TsP), "Ts_uPF": np.nanmean(uPF_P * TsP),
          "Ts_v": np.nanmean(vP * TsP), "Ts_vPF": np.nanmean(vPF_P * TsP),
          "Ths_u": np.nanmean(uP * ThvP), "Ths_uPF": np.nanmean(uPF_P * ThvP),
          "Ths_v": np.nanmean(vP * ThvP), "Ths_vPF": np.nanmean(vPF_P * ThvP)}
    if rot or tsf:
        hl = {k: np.nan for k in hl}
    r.values["Hlat"] = hl

    # ---- fine-wire fluxes ----
    if lev.fw is not None:
        fwP = nandetrend(lev.fw[s0:s1], det)
        thFwP = nandetrend(lev.theta_fw[s0:s1], det)
        VthFwP = nandetrend(lev.Vtheta_fw[s0:s1], det)
        r.put("H", "fwT_w", np.nanmean(wP * fwP))
        r.put("H", "fwT_wPF", np.nanmean(wPF_P * fwP))
        r.put("H", "fwTh_w", np.nanmean(wP * thFwP))
        r.put("H", "fwTh_wPF", np.nanmean(wPF_P * thFwP))
        r.put("H", "fwVTh_w", np.nanmean(wP * VthFwP))
        r.put("H", "fwVTh_wPF", np.nanmean(wPF_P * VthFwP))
        if unrot or fwf:
            r.blank("H", "fwT_w", "fwTh_w", "fwVTh_w")
        if rot or fwf:
            r.blank("H", "fwT_wPF", "fwTh_wPF", "fwVTh_wPF")
        r.put("Hlat", "fwTh_u", np.nanmean(uP * thFwP))
        r.put("Hlat", "fwTh_uPF", np.nanmean(uPF_P * thFwP))
        r.put("Hlat", "fwTh_v", np.nanmean(vP * thFwP))
        r.put("Hlat", "fwTh_vPF", np.nanmean(vPF_P * thFwP))
        if rot or fwf:
            r.blank("Hlat", "fwTh_u", "fwTh_uPF", "fwTh_v", "fwTh_vPF")
        r.samples["fwThPrime"] = thFwP
        r.samples["fwTh"] = lev.theta_fw[s0:s1]
        r.samples["fwT"] = lev.fw[s0:s1]

    # ====================================================================
    # H2O-based fluxes (LH, H2O sigma/eta/R, skew, Flux_lat, turbtr)
    # ====================================================================
    H2Op = rhov_ext = rho_CO2p = rhoc_ext = None
    if lev.h2o is not None:
        H2Op = nandetrend(lev.h2o[s0:s1], det)
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
        r.put("H", "Tair_w", np.nan if (unrot or tsf or h2of) else np.nanmean(wP * TairP))
        r.put("H", "Tair_wPF", np.nan if (rot or tsf or h2of) else np.nanmean(wPF_P * TairP))

        # WPL external H2O fluctuation
        rhov_ext = (Md / Mv * (rho_v_j / rho_d_j) * (H2Op / 1000.0)
                    + rho_v_j * (1.0 + Md / Mv * rho_v_j / rho_d_j) * TairP / T_ref_j)

        r.put("sigma", "sigma_H2O", np.nan if (tsf or rot) else np.nanstd(lev.h2o[s0:s1]))

        if tsf or rot:
            r.blank("R", "R_uPFwPF_wPFH2O", "R_wPFH2O_wPFThetav", "R_wPF_uPF", "R_wPF_Theta_v", "R_wPF_H2O")
            r.blank("eta", "eta_wPFH2O", "eta_wPFH2O_WPL")
            r.blank("delta_flux_ctrb", "S_wPFH2O", "S_wPFH2O_WPL")
            r.blank("delta_time_ctrb", "D_wPFH2O", "D_wPFH2O_WPL")
        else:
            r.put("R", "R_uPFwPF_wPFH2O", _corr(uPF_P * wPF_P, H2Op * wPF_P))
            r.put("R", "R_wPFH2O_wPFThetav", _corr(H2Op * wPF_P, ThvP * wPF_P))
            r.put("R", "R_wPF_uPF", _corr(wPF_P, uPF_P))
            r.put("R", "R_wPF_Theta_v", _corr(wPF_P, ThvP))
            r.put("R", "R_wPF_H2O", _corr(wPF_P, H2Op))
            r.put("eta", "eta_wPFH2O", find_eta(wPF_P, H2Op))
            r.put("eta", "eta_wPFH2O_WPL", find_eta(wPF_P, H2Op + rhov_ext * 1e3))
            r.put("delta_flux_ctrb", "S_wPFH2O", find_delta_flux(wPF_P, H2Op))
            r.put("delta_flux_ctrb", "S_wPFH2O_WPL", find_delta_flux(wPF_P, H2Op + rhov_ext * 1e3))
            r.put("delta_time_ctrb", "D_wPFH2O", find_delta_time(wPF_P, H2Op))
            r.put("delta_time_ctrb", "D_wPFH2O_WPL", find_delta_time(wPF_P, H2Op + rhov_ext * 1e3))

        # LHflux (all-NaN periods produce nan silently)
        with np.errstate(all="ignore"):
            E = np.nanmean(wP * H2Op)
            EPF = np.nanmean(wPF_P * H2Op)
        Lv = (2.501 - 0.00237 * (T_ref_j - 273.15)) * 1e3
        # Temperature flux for the WPL terms (Webb et al. 1980 eqs. 24-25, 44):
        # w'T', i.e. the Schotanus-corrected T_air'wPF' column. MATLAB
        # (fluxes.m:1073) used the buoyancy flux Theta_v'wPF'.
        kin_sen_flux = r.get("H", "Tair_wPF")
        wpl = 1.0 + Md / Mv * rho_v_j / rho_d_j

        r.put("LHflux", "Lv", Lv)
        r.put("LHflux", "E_w", np.nan if (unrot or h2of) else E)
        r.put("LHflux", "E_wPF", np.nan if (rot or h2of) else EPF)
        r.put("LHflux", "wPF_qWPL", np.nan if (rot or h2of) else
              wpl * (np.nanmean(wPF_P * H2Op) / 1e3 + rho_v_j / T_ref_j * np.nanmean(wPF_P * TairP)))
        LE_w = 1000.0 * Lv * wpl * (E / 1000.0 + rho_v_j / T_ref_j * kin_sen_flux)
        LE_wPF = 1000.0 * Lv * wpl * (EPF / 1000.0 + rho_v_j / T_ref_j * kin_sen_flux)
        r.put("LHflux", "LE_WPL_w", np.nan if (unrot or h2of) else LE_w)
        r.put("LHflux", "LE_WPL_wPF", np.nan if (unrot or h2of) else LE_wPF)

        # KH2O O2 correction (Tanner et al. 1993)
        if lev.h2o_is_kh2o:
            ko = -0.0045
            kw = -0.153
            CkO = 0.23 * ko / kw
            O2_corr = CkO * rho_d_j / T_ref_j * kin_sen_flux * 1000.0
            E += O2_corr
            EPF += O2_corr
            r.put("LHflux", "LE_O2_w", Lv * E)
            r.put("LHflux", "LE_O2_wPF", Lv * EPF)

        r.samples["rhov"] = lev.h2o[s0:s1]
        r.samples["rhovPrime"] = H2Op
        r.samples["rhovextenalPrime"] = rhov_ext * 1e3

        # ================================================================
        # CO2-based fluxes (requires H2O to exist first for evapFlux)
        # ================================================================
        if lev.co2 is not None:
            rho_CO2_seg = lev.co2[s0:s1]                      # mg/m³
            rho_CO2p = nandetrend(rho_CO2_seg, det) / 1e6     # kg/m³
            rho_CO2_avg = np.nanmean(rho_CO2_seg) / 1e6       # kg/m³
            evap_flux = r.get("LHflux", "LE_WPL_wPF") / Lv / 1000.0   # WPL wPF'' (kg/m²/s)

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
                                 / (44.01 * P_seg_kPa * 1e3))

            r.put("sigma", "sigma_CO2", np.nanstd(rho_CO2_seg))
            r.put("sigma", "sigma_CO2_WPL", np.nanstd(rho_CO2_seg / 1e6 + rhoc_ext) * 1e6)
            if tsf or rot:
                r.blank("sigma", "sigma_CO2", "sigma_CO2_WPL")

            # R CO2 (MATLAB cols 5-17 order)
            if tsf or rot:
                r.blank("R", "R_uPFwPF_wPFCO2_WPL", "R_wPFCO2_WPL_wPFThetav", "R_wPF_CO2", "R_wPF_CO2_WPL",
                        "R_uPFwPF_wPFCO2", "R_wPFCO2_wPFThetav",
                        "R_uPFwPF_wPFH2O_WPL", "R_wPFH2O_WPL_wPFThetav", "R_wPF_H2O_WPL")
                r.blank("eta", "eta_wPFCO2_WPL", "eta_wPFCO2")
                r.blank("delta_flux_ctrb", "S_wPFCO2_WPL", "S_wPFCO2")
                r.blank("delta_time_ctrb", "D_wPFCO2_WPL", "D_wPFCO2")
            else:
                r.put("R", "R_uPFwPF_wPFCO2_WPL", _corr(uPF_P * wPF_P, (rho_CO2p + rhoc_ext) * wPF_P))
                r.put("R", "R_wPFCO2_WPL_wPFThetav", _corr((rho_CO2p + rhoc_ext) * wPF_P, ThvP * wPF_P))
                r.put("R", "R_wPF_CO2", _corr(wPF_P, rho_CO2p))
                r.put("R", "R_wPF_CO2_WPL", _corr(wPF_P, rho_CO2p + rhoc_ext))
                r.put("R", "R_uPFwPF_wPFCO2", _corr(uPF_P * wPF_P, rho_CO2p * wPF_P))
                r.put("R", "R_wPFCO2_wPFThetav", _corr(rho_CO2p * wPF_P, ThvP * wPF_P))
                r.put("R", "R_uPFwPF_wPFH2O_WPL", _corr(uPF_P * wPF_P, (H2Op + rhov_ext * 1e3) * wPF_P))
                r.put("R", "R_wPFH2O_WPL_wPFThetav", _corr((H2Op + rhov_ext * 1e3) * wPF_P, ThvP * wPF_P))
                r.put("R", "R_wPF_H2O_WPL", _corr(wPF_P, H2Op + rhov_ext * 1e3))
                r.put("eta", "eta_wPFCO2_WPL", find_eta(wPF_P, rho_CO2p + rhoc_ext))
                r.put("eta", "eta_wPFCO2", find_eta(wPF_P, rho_CO2p))
                r.put("delta_flux_ctrb", "S_wPFCO2_WPL", find_delta_flux(wPF_P, rho_CO2p + rhoc_ext))
                r.put("delta_flux_ctrb", "S_wPFCO2", find_delta_flux(wPF_P, rho_CO2p))
                r.put("delta_time_ctrb", "D_wPFCO2_WPL", find_delta_time(wPF_P, rho_CO2p + rhoc_ext))
                r.put("delta_time_ctrb", "D_wPFCO2", find_delta_time(wPF_P, rho_CO2p))

            # skewness, lateral flux contributions, turbulent transport
            if rot:
                r.blank("skew", "skew_uPF", "skew_vPF", "skew_wPF", "skew_Theta_v",
                        "skew_H2O", "skew_H2O_WPL", "skew_CO2", "skew_CO2_WPL")
                r.blank("Flux_lat", "uPF_thv", "uPF_H2O", "uPF_H2O_WPL", "uPF_CO2", "uPF_CO2_WPL")
                r.blank("turbtr", "w_H2Ow", "w_H2OWPLw", "w_CO2w", "w_CO2WPLw")
            else:
                r.values["skew"] = {
                    "skew_uPF": _skew(uPF_P), "skew_vPF": _skew(vPF_P), "skew_wPF": _skew(wPF_P),
                    "skew_Theta_v": _skew(ThvP), "skew_H2O": _skew(H2Op),
                    "skew_H2O_WPL": _skew(H2Op + rhov_ext * 1e3),
                    "skew_CO2": _skew(rho_CO2p), "skew_CO2_WPL": _skew(rho_CO2p + rhoc_ext)}
                r.values["Flux_lat"] = {
                    "uPF_thv": np.nanmean(uPF_P * ThvP), "uPF_H2O": np.nanmean(uPF_P * H2Op),
                    "uPF_H2O_WPL": np.nanmean(uPF_P * (H2Op + rhov_ext * 1e3)),
                    "uPF_CO2": np.nanmean(uPF_P * rho_CO2p),
                    "uPF_CO2_WPL": np.nanmean(uPF_P * (rho_CO2p + rhoc_ext))}
                r.put("turbtr", "w_H2Ow", np.nanmean(wPF_P * (H2Op * wPF_P)))
                r.put("turbtr", "w_H2OWPLw", np.nanmean(wPF_P * ((H2Op + rhov_ext * 1e3) * wPF_P)))
                r.put("turbtr", "w_CO2w", np.nanmean(wPF_P * (rho_CO2p * wPF_P)))
                r.put("turbtr", "w_CO2WPLw", np.nanmean(wPF_P * ((rho_CO2p + rhoc_ext) * wPF_P)))

            # CO2 flux
            Fc_w = np.nanmean(wP * rho_CO2p)
            Fc_wPF = np.nanmean(wPF_P * rho_CO2p)
            r.put("CO2flux", "Fc_w", np.nan if (unrot or co2f) else Fc_w)
            r.put("CO2flux", "Fc_wPF", np.nan if (rot or co2f) else Fc_wPF)
            Fc_wPF_stored = r.get("CO2flux", "Fc_wPF")
            r.put("CO2flux", "Fc_WPL", np.nan if (rot or co2f) else
                  (Fc_wPF_stored + Md / Mv * (rho_CO2_avg / rho_d_j) * evap_flux
                   + (1.0 + Md / Mv * rho_v_j / rho_d_j) * (rho_CO2_avg / T_ref_j) * kin_sen_flux))
            r.put("CO2flux", "wPF_CO2WPL", np.nan if (rot or co2f) else
                  Fc_wPF_stored + np.nanmean(wPF_P * rhoc_ext))
            r.put("CO2flux", "ppm", ppm_CO2)

            # raw CO2: stored in mg/m³ (rhoCO2Prime = rho_CO2p * 1e6), unlike
            # MATLAB which left the kg/m³ of the WPL arithmetic (fluxes.m:1212).
            r.samples["rhoCO2"] = rho_CO2_seg
            r.samples["rhoCO2Prime"] = rho_CO2p * 1e6
            r.samples["rhoCO2extenalPrime"] = rhoc_ext * 1e6

    # ---- SSITC + SS-only quality flags (ForestComplexTerrain) ----
    ustar_jj = np.sqrt(tau_pf) if (not np.isnan(tau_pf) and tau_pf >= 0) else np.nan
    has_co2 = lev.h2o is not None and lev.co2 is not None
    (tau_ssitc, tau_ss, H_ssitc, H_ss, LE_ssitc, LE_ss, FC_ssitc, FC_ss) = calc_ssitc_flags(
        wPF_P, uPF_P, vPF_P, ThvP,
        H2Op, rhov_ext,
        rho_CO2p if has_co2 else None, rhoc_ext if has_co2 else None,
        ustar_jj, L_val,
        rot, tsf, h2of, co2f,
        opts.n_sub, lev.height, opts.displacement_height,
        canopy_height=opts.canopy_height,
        use_canopy_itc=opts.use_canopy_itc,
        latitude=opts.latitude,
    )
    r.values["fluxQC"] = {"TAU_SSITC": tau_ssitc, "TAU_SS": tau_ss, "H_SSITC": H_ssitc, "H_SS": H_ss,
                          "LE_SSITC": LE_ssitc, "LE_SS": LE_ss, "FC_SSITC": FC_ssitc, "FC_SS": FC_ss}
    return r
