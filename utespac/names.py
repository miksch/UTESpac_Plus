"""Output names of the flux products: the single source of truth.

One convention for the run-file groups, their variables, the CSV headers
and the high-frequency netCDF (gameplan:
``tasks/active/2026-09-05_variable-name-refactor.md``). Names are
``[A-Za-z0-9_]`` only; lowercase snake_case except the field's flux and
scale symbols (:data:`SYMBOLS`), which keep their case as whole tokens.
Within a name the order is operands, statistic, qualifiers; the frame
qualifier (``_raw`` unrotated sonic axes, ``_pf`` planar fit plus yaw,
``_tilt`` planar-fit frame only) is always last.

:data:`GROUPS` maps the legacy table names to the group names,
:data:`VARIABLES` lists every variable of every group as a :class:`Var`
(legacy key, name, units, ``long_name``, legacy label template), and
:data:`HF_VARIABLES` maps the ``utespac-hf-1`` variable names to the
``utespac-hf-2`` ones. :func:`legacy_to_new` resolves a legacy
``(table, key)`` pair; :func:`csv_header` renders the CSV column of a
variable at a height.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# flux and scale symbols that keep their conventional case
SYMBOLS = frozenset({"H", "LE", "Fc", "Tau", "TKE", "L", "Lv"})

# legacy table -> group
GROUPS: Dict[str, str] = {
    "H": "sensible_heat",
    "Hlat": "sensible_heat_lateral",
    "tau": "momentum",
    "tke": "tke",
    "sigma": "sigma",
    "R": "correlation",
    "L": "obukhov",
    "scaling": "scaling",
    "eta": "eta",
    "delta_flux_ctrb": "delta_flux",
    "delta_time_ctrb": "delta_time",
    "turbtr": "transport",
    "epsilon": "dissipation",
    "skew": "skewness",
    "H_SNSP": "sensible_heat_snsp",
    "Flux_lat": "scalar_flux_lateral",
    "LHflux": "latent_heat",
    "fluxQC": "flux_qc",
    "CO2flux": "co2_flux",
    "derivedT": "temperature",
    "specificHum": "humidity",
}
GROUPS_INVERSE: Dict[str, str] = {v: k for k, v in GROUPS.items()}


@dataclass(frozen=True)
class Var:
    """One product variable.

    Attributes
    ----------
    name : str
        Variable name in the run file and the CSV header stem.
    units : str
        CF-style units string.
    long_name : str
    legacy_key : str or None
        Column key of the legacy ``TABLE_SPECS`` (``utespac-run-2``
        variable name); None for columns introduced with the rename.
    legacy_label : str or None
        Legacy CSV label template (``{hn}`` is the height label).
    legacy_table : str or None
        Legacy table the key lived in when it differs from the group's
        own legacy table (the third moment moved from ``sigma``).
    """
    name: str
    units: str
    long_name: str
    legacy_key: Optional[str] = None
    legacy_label: Optional[str] = None
    legacy_table: Optional[str] = None


def _v(name, units, long_name, legacy_key=None, legacy_label=None, legacy_table=None):
    return Var(name, units, long_name, legacy_key, legacy_label, legacy_table)


_KMS = "K m s-1"
_M2S2 = "m2 s-2"

VARIABLES: Dict[str, List[Var]] = {
    "sensible_heat": [
        _v("rho_air_ref", "kg m-3", "moist air density at the reference level", "rho", "rho"),
        _v("cp_ref", "J kg-1 K-1", "specific heat of moist air at the reference level", "cp", "cp"),
        _v("w_ts_cov_raw", _KMS, "covariance w'Ts', unrotated sonic axes", "Ts_w", "{hn}m son:Ts'w'"),
        _v("w_theta_v_cov_pf", _KMS, "covariance w'theta_v' (buoyancy flux), planar fit + yaw", "Thv_wPF", "{hn}m son:Theta_v'wPF'"),
        _v("w_t_air_cov_raw", _KMS, "covariance w'T_air' (humidity-corrected sonic temperature), unrotated sonic axes", "Tair_w", "{hn}m son:T_air'w'"),
        _v("w_t_air_cov_pf", _KMS, "covariance w'T_air' (humidity-corrected sonic temperature), planar fit + yaw", "Tair_wPF", "{hn}m son:T_air'wPF'"),
        _v("w_t_fw_cov_raw", _KMS, "covariance w'T_fw' (fine wire), unrotated sonic axes", "fwT_w", "{hn}m fw:T'w'"),
        _v("w_t_fw_cov_pf", _KMS, "covariance w'T_fw' (fine wire), planar fit + yaw", "fwT_wPF", "{hn}m fw:T'wPF'"),
        _v("w_theta_s_cov_raw", _KMS, "covariance w'theta_s' (sonic potential temperature), unrotated sonic axes", "Ths_w", "{hn}m son:Th_s'w'"),
        _v("w_theta_s_cov_pf", _KMS, "covariance w'theta_s' (sonic potential temperature), planar fit + yaw", "Ths_wPF", "{hn}m son:Th_s'wPF'"),
        _v("w_theta_fw_cov_raw", _KMS, "covariance w'theta_fw' (fine-wire potential temperature), unrotated sonic axes", "fwTh_w", "{hn}m fw:Th'w'"),
        _v("w_theta_fw_cov_pf", _KMS, "covariance w'theta_fw' (fine-wire potential temperature), planar fit + yaw", "fwTh_wPF", "{hn}m fw:Th'wPF'"),
        _v("w_theta_v_fw_cov_raw", _KMS, "covariance w'theta_v_fw' (fine-wire virtual potential temperature), unrotated sonic axes", "fwVTh_w", "{hn}m fw:VTh'w'"),
        _v("w_theta_v_fw_cov_pf", _KMS, "covariance w'theta_v_fw' (fine-wire virtual potential temperature), planar fit + yaw", "fwVTh_wPF", "{hn}m fw:VTh'wPF'"),
        _v("H_raw", "W m-2", "sensible heat flux rho_air_ref cp_ref w'T_air', unrotated sonic axes"),
        _v("H_pf", "W m-2", "sensible heat flux rho_air_ref cp_ref w'T_air', planar fit + yaw"),
        _v("H_buoyancy_pf", "W m-2", "buoyancy flux rho_air_ref cp_ref w'theta_v', planar fit + yaw"),
    ],
    "sensible_heat_lateral": [
        _v("u_ts_cov_raw", _KMS, "covariance u'Ts', unrotated sonic axes", "Ts_u", "{hn}m son:Ts'u'"),
        _v("u_ts_cov_pf", _KMS, "covariance u'Ts', planar fit + yaw", "Ts_uPF", "{hn}m son:Ts'uPF'"),
        _v("v_ts_cov_raw", _KMS, "covariance v'Ts', unrotated sonic axes", "Ts_v", "{hn}m son:Ts'v'"),
        _v("v_ts_cov_pf", _KMS, "covariance v'Ts', planar fit + yaw", "Ts_vPF", "{hn}m son:Ts'vPF'"),
        _v("u_theta_s_cov_raw", _KMS, "covariance u'theta_s', unrotated sonic axes", "Ths_u", "{hn}m son:Th_s'u'"),
        _v("u_theta_s_cov_pf", _KMS, "covariance u'theta_s', planar fit + yaw", "Ths_uPF", "{hn}m son:Th_s'uPF'"),
        _v("v_theta_s_cov_raw", _KMS, "covariance v'theta_s', unrotated sonic axes", "Ths_v", "{hn}m son:Th_s'v'"),
        _v("v_theta_s_cov_pf", _KMS, "covariance v'theta_s', planar fit + yaw", "Ths_vPF", "{hn}m son:Th_s'vPF'"),
        _v("u_theta_fw_cov_raw", _KMS, "covariance u'theta_fw', unrotated sonic axes", "fwTh_u", "{hn}m fw:Th'u'"),
        _v("u_theta_fw_cov_pf", _KMS, "covariance u'theta_fw', planar fit + yaw", "fwTh_uPF", "{hn}m fw:Th'uPF'"),
        _v("v_theta_fw_cov_raw", _KMS, "covariance v'theta_fw', unrotated sonic axes", "fwTh_v", "{hn}m fw:Th'v'"),
        _v("v_theta_fw_cov_pf", _KMS, "covariance v'theta_fw', planar fit + yaw", "fwTh_vPF", "{hn}m fw:Th'vPF'"),
    ],
    "momentum": [
        _v("Tau_raw", _M2S2, "kinematic momentum flux magnitude sqrt(u'w'^2 + v'w'^2) = u*^2, unrotated sonic axes", "tau_raw", "{hn}m :sqrt(u'w'^2+v'w'^2)"),
        _v("Tau_pf", _M2S2, "kinematic momentum flux magnitude sqrt(u'w'^2 + v'w'^2) = u*^2, planar fit + yaw", "tau_PF", "{hn}m :sqrt(uPF'wPF'^2+vPF'wPF'^2)"),
        _v("u_var_pf", _M2S2, "variance of u, planar fit + yaw", "uPF_uPF", "{hn}m :uPF'uPF'"),
        _v("v_var_pf", _M2S2, "variance of v, planar fit + yaw", "vPF_vPF", "{hn}m :vPF'vPF'"),
        _v("w_var_pf", _M2S2, "variance of w, planar fit + yaw", "wPF_wPF", "{hn}m :wPF'wPF'"),
        _v("u_v_cov_pf", _M2S2, "covariance u'v', planar fit + yaw", "uPF_vPF", "{hn}m :uPF'vPF'"),
        _v("u_w_cov_pf", _M2S2, "covariance u'w', planar fit + yaw", "uPF_wPF", "{hn}m :uPF'wPF'"),
        _v("v_w_cov_pf", _M2S2, "covariance v'w', planar fit + yaw", "vPF_wPF", "{hn}m :vPF'wPF'"),
        _v("u_var_tilt", _M2S2, "variance of u, planar-fit frame (fall line)", "uT_uT", "{hn}m :uTilt'uTilt'"),
        _v("v_var_tilt", _M2S2, "variance of v, planar-fit frame (across the fall line)", "vT_vT", "{hn}m :vTilt'vTilt'"),
        _v("w_var_tilt", _M2S2, "variance of w, planar-fit frame", "wT_wT", "{hn}m :wTilt'wTilt'"),
        _v("u_v_cov_tilt", _M2S2, "covariance u'v', planar-fit frame", "uT_vT", "{hn}m :uTilt'vTilt'"),
        _v("u_w_cov_tilt", _M2S2, "covariance u'w', planar-fit frame", "uT_wT", "{hn}m :uTilt'wTilt'"),
        _v("v_w_cov_tilt", _M2S2, "covariance v'w', planar-fit frame", "vT_wT", "{hn}m :vTilt'wTilt'"),
    ],
    "tke": [
        _v("TKE", _M2S2, "turbulent kinetic energy 0.5 (u'^2 + v'^2 + w'^2)", "tke", "{hn}m :0.5(u'^2+v'^2+w'^2)"),
    ],
    "sigma": [
        _v("u_sigma_raw", "m s-1", "standard deviation of u, unrotated sonic axes", "sigma_u", "{hn}m :sigma_u"),
        _v("v_sigma_raw", "m s-1", "standard deviation of v, unrotated sonic axes", "sigma_v", "{hn}m :sigma_v"),
        _v("w_sigma_raw", "m s-1", "standard deviation of w, unrotated sonic axes", "sigma_w", "{hn}m :sigma_w"),
        _v("u_sigma_pf", "m s-1", "standard deviation of u, planar fit + yaw", "sigma_uPF", "{hn}m :sigma_uPF"),
        _v("v_sigma_pf", "m s-1", "standard deviation of v, planar fit + yaw", "sigma_vPF", "{hn}m :sigma_vPF"),
        _v("w_sigma_pf", "m s-1", "standard deviation of w, planar fit + yaw", "sigma_wPF", "{hn}m :sigma_wPF"),
        _v("ts_sigma", "K", "standard deviation of the sonic temperature", "sigma_Tson", "{hn}m :sigma_Tson"),
        _v("theta_v_sigma", "K", "standard deviation of the sonic virtual potential temperature", "sigma_Theta_v", "{hn}m :sigma_Theta_v"),
        _v("h2o_sigma", "g m-3", "standard deviation of the water vapour density", "sigma_H2O", "{hn}m :sigma_H2O"),
        _v("co2_sigma", "mg m-3", "standard deviation of the CO2 density", "sigma_CO2", "{hn}m :sigma_CO2"),
        _v("co2_wpl_sigma", "mg m-3", "standard deviation of the WPL-corrected CO2 density", "sigma_CO2_WPL", "{hn}m :sigma_CO2_WPL"),
        _v("t_fw_sigma", "K", "standard deviation of the fine-wire temperature", "sigma_TFW", "{hn}m :sigma_TFW"),
    ],
    "correlation": [
        _v("u_w__w_theta_v_corr_pf", "1", "correlation of the u'w' and w'theta_v' series, planar fit + yaw", "R_uPFwPF_wPFThetav", "{hn}m :R_uPFwPF_wPFThetav"),
        _v("u_w__w_h2o_corr_pf", "1", "correlation of the u'w' and w'h2o' series, planar fit + yaw", "R_uPFwPF_wPFH2O", "{hn}m :R_uPFwPF_wPFH2O"),
        _v("w_h2o__w_theta_v_corr_pf", "1", "correlation of the w'h2o' and w'theta_v' series, planar fit + yaw", "R_wPFH2O_wPFThetav", "{hn}m :R_wPFH2O_wPFThetav"),
        _v("u_w__w_co2_wpl_corr_pf", "1", "correlation of the u'w' and w'co2_wpl' series, planar fit + yaw", "R_uPFwPF_wPFCO2_WPL", "{hn}m :R_uPFwPF_wPFCO2_WPL"),
        _v("w_co2_wpl__w_theta_v_corr_pf", "1", "correlation of the w'co2_wpl' and w'theta_v' series, planar fit + yaw", "R_wPFCO2_WPL_wPFThetav", "{hn}m :R_wPFCO2_WPL_wPFThetav"),
        _v("w_u_corr_pf", "1", "correlation of w and u, planar fit + yaw", "R_wPF_uPF", "{hn}m :R_wPF_uPF"),
        _v("w_theta_v_corr_pf", "1", "correlation of w and theta_v, planar fit + yaw", "R_wPF_Theta_v", "{hn}m :R_wPF_Theta_v"),
        _v("w_h2o_corr_pf", "1", "correlation of w and h2o, planar fit + yaw", "R_wPF_H2O", "{hn}m :R_wPF_H2O"),
        _v("w_co2_corr_pf", "1", "correlation of w and co2, planar fit + yaw", "R_wPF_CO2", "{hn}m :R_wPF_CO2"),
        _v("w_co2_wpl_corr_pf", "1", "correlation of w and co2_wpl, planar fit + yaw", "R_wPF_CO2_WPL", "{hn}m :R_wPF_CO2_WPL"),
        _v("u_w__w_co2_corr_pf", "1", "correlation of the u'w' and w'co2' series, planar fit + yaw", "R_uPFwPF_wPFCO2", "{hn}m :R_uPFwPF_wPFCO2"),
        _v("w_co2__w_theta_v_corr_pf", "1", "correlation of the w'co2' and w'theta_v' series, planar fit + yaw", "R_wPFCO2_wPFThetav", "{hn}m :R_wPFCO2_wPFThetav"),
        _v("u_w__w_h2o_wpl_corr_pf", "1", "correlation of the u'w' and w'h2o_wpl' series, planar fit + yaw", "R_uPFwPF_wPFH2O_WPL", "{hn}m :R_uPFwPF_wPFH2O_WPL"),
        _v("w_h2o_wpl__w_theta_v_corr_pf", "1", "correlation of the w'h2o_wpl' and w'theta_v' series, planar fit + yaw", "R_wPFH2O_WPL_wPFThetav", "{hn}m :R_wPFH2O_WPL_wPFThetav"),
        _v("w_h2o_wpl_corr_pf", "1", "correlation of w and h2o_wpl, planar fit + yaw", "R_wPF_H2O_WPL", "{hn}m :R_wPF_H2O_WPL"),
    ],
    "obukhov": [
        _v("L", "m", "Obukhov length -u*^3 theta_v / (kappa g w'theta_v'_vertical)", "L",
           "{hn}m L:sqrt(uPF'*wPF'+vPF'*wPF')^3/2*Th_v/(k*g*wThv_vert)"),
    ],
    "scaling": [
        _v("ustar_pf", "m s-1", "friction velocity sqrt(Tau_pf), planar fit + yaw"),
        _v("theta_star", "K", "surface-layer temperature scale -w'theta_v'/u*", "theta_star_SL", "{hn}m :theta_star_SL(K)"),
        _v("q_star", "g kg-1", "surface-layer humidity scale -w'q'/u*", "q_star_SL", "{hn}m :q_star_SL(g/kg)"),
        _v("psi_m", "1", "integrated stability function for momentum psi_m(z/L)", "psi_m", "{hn}m :psi_m(z/L)"),
        _v("psi_h", "1", "integrated stability function for heat psi_h(z/L)", "psi_h", "{hn}m :psi_h(z/L)"),
    ],
    "eta": [
        _v("u_w_eta_pf", "1", "total / downgradient flux ratio of u'w', planar fit + yaw", "eta_wPFuPF", "{hn}m :eta_wPFuPF"),
        _v("w_theta_v_eta_pf", "1", "total / downgradient flux ratio of w'theta_v', planar fit + yaw", "eta_wPFThetav", "{hn}m :eta_wPFThetav"),
        _v("w_h2o_eta_pf", "1", "total / downgradient flux ratio of w'h2o', planar fit + yaw", "eta_wPFH2O", "{hn}m :eta_wPFH2O"),
        _v("w_h2o_wpl_eta_pf", "1", "total / downgradient flux ratio of w'h2o_wpl', planar fit + yaw", "eta_wPFH2O_WPL", "{hn}m :eta_wPFH2O_WPL"),
        _v("w_co2_wpl_eta_pf", "1", "total / downgradient flux ratio of w'co2_wpl', planar fit + yaw", "eta_wPFCO2_WPL", "{hn}m :eta_wPFCO2_WPL"),
        _v("w_co2_eta_pf", "1", "total / downgradient flux ratio of w'co2', planar fit + yaw", "eta_wPFCO2", "{hn}m :eta_wPFCO2"),
    ],
    "delta_flux": [
        _v("u_w_delta_flux_pf", "1", "(ejection - sweep) / total flux of u'w', planar fit + yaw", "S_wPFuPF", "{hn}m :S_wPFuPF"),
        _v("w_theta_v_delta_flux_pf", "1", "(ejection - sweep) / total flux of w'theta_v', planar fit + yaw", "S_wPFThetav", "{hn}m :S_wPFThetav"),
        _v("w_h2o_delta_flux_pf", "1", "(ejection - sweep) / total flux of w'h2o', planar fit + yaw", "S_wPFH2O", "{hn}m :S_wPFH2O"),
        _v("w_h2o_wpl_delta_flux_pf", "1", "(ejection - sweep) / total flux of w'h2o_wpl', planar fit + yaw", "S_wPFH2O_WPL", "{hn}m :S_wPFH2O_WPL"),
        _v("w_co2_wpl_delta_flux_pf", "1", "(ejection - sweep) / total flux of w'co2_wpl', planar fit + yaw", "S_wPFCO2_WPL", "{hn}m :S_wPFCO2_WPL"),
        _v("w_co2_delta_flux_pf", "1", "(ejection - sweep) / total flux of w'co2', planar fit + yaw", "S_wPFCO2", "{hn}m :S_wPFCO2"),
    ],
    "delta_time": [
        _v("u_w_delta_time_pf", "1", "ejection - sweep time fraction of u'w', planar fit + yaw", "D_wPFuPF", "{hn}m :D_wPFuPF"),
        _v("w_theta_v_delta_time_pf", "1", "ejection - sweep time fraction of w'theta_v', planar fit + yaw", "D_wPFThetav", "{hn}m :D_wPFThetav"),
        _v("w_h2o_delta_time_pf", "1", "ejection - sweep time fraction of w'h2o', planar fit + yaw", "D_wPFH2O", "{hn}m :D_wPFH2O"),
        _v("w_h2o_wpl_delta_time_pf", "1", "ejection - sweep time fraction of w'h2o_wpl', planar fit + yaw", "D_wPFH2O_WPL", "{hn}m :D_wPFH2O_WPL"),
        _v("w_co2_wpl_delta_time_pf", "1", "ejection - sweep time fraction of w'co2_wpl', planar fit + yaw", "D_wPFCO2_WPL", "{hn}m :D_wPFCO2_WPL"),
        _v("w_co2_delta_time_pf", "1", "ejection - sweep time fraction of w'co2', planar fit + yaw", "D_wPFCO2", "{hn}m :D_wPFCO2"),
    ],
    "transport": [
        _v("TKE_transport_pf", "m3 s-3", "turbulent transport of TKE, mean(w' e'), planar fit + yaw", "w_e", "{hn}m :mean(w'e')"),
        _v("u_w_transport_pf", "m3 s-3", "turbulent transport of u'w', mean(w' u'w'), planar fit + yaw", "w_uw", "{hn}m :mean(w'u'w')"),
        _v("w_theta_v_transport_pf", "K m2 s-2", "turbulent transport of w'theta_v', mean(w' theta_v'w'), planar fit + yaw", "w_thvw", "{hn}m :mean(w'theta_v'w')"),
        _v("w_h2o_transport_pf", "g m-1 s-2", "turbulent transport of w'h2o', mean(w' h2o'w'), planar fit + yaw", "w_H2Ow", "{hn}m :mean(w'H2O'w')"),
        _v("w_h2o_wpl_transport_pf", "g m-1 s-2", "turbulent transport of w'h2o_wpl', planar fit + yaw", "w_H2OWPLw", "{hn}m :mean(w'H2O_WPL'w')"),
        _v("w_co2_transport_pf", "kg m-1 s-2", "turbulent transport of w'co2', mean(w' co2'w'), planar fit + yaw", "w_CO2w", "{hn}m :mean(w'CO2'w')"),
        _v("w_co2_wpl_transport_pf", "kg m-1 s-2", "turbulent transport of w'co2_wpl', planar fit + yaw", "w_CO2WPLw", "{hn}m :mean(w'CO2_WPL'w')"),
        _v("ts_var_transport_pf", "K2 m s-1", "turbulent transport of the Ts variance, mean(w' Ts'^2), planar fit + yaw",
           "wPFP_TsonP_TsonP", "{hn}m :wPFP_TsonP_TsonP", legacy_table="sigma"),
    ],
    "dissipation": [
        _v("epsilon", "m2 s-3", "TKE dissipation rate from the compensated structure function", "epsilon", "{hn}m :epsilon"),
    ],
    "skewness": [
        _v("u_skew_pf", "1", "skewness of u, planar fit + yaw", "skew_uPF", "{hn}m :skew_uPF"),
        _v("v_skew_pf", "1", "skewness of v, planar fit + yaw", "skew_vPF", "{hn}m :skew_vPF"),
        _v("w_skew_pf", "1", "skewness of w, planar fit + yaw", "skew_wPF", "{hn}m :skew_wPF"),
        _v("theta_v_skew", "1", "skewness of the sonic virtual potential temperature", "skew_Theta_v", "{hn}m :skew_Theta_v"),
        _v("h2o_skew", "1", "skewness of the water vapour density", "skew_H2O", "{hn}m :skew_H2O"),
        _v("h2o_wpl_skew", "1", "skewness of the WPL-corrected water vapour density", "skew_H2O_WPL", "{hn}m :skew_H2O_WPL"),
        _v("co2_skew", "1", "skewness of the CO2 density", "skew_CO2", "{hn}m :skew_CO2"),
        _v("co2_wpl_skew", "1", "skewness of the WPL-corrected CO2 density", "skew_CO2_WPL", "{hn}m :skew_CO2_WPL"),
    ],
    "sensible_heat_snsp": [
        _v("u_theta_v_cov_snsp", _KMS, "covariance u'theta_v' in the slope-normal / slope-parallel frame", "uTHv", "{hn}m :uTHv"),
        _v("v_theta_v_cov_snsp", _KMS, "covariance v'theta_v' in the slope-normal / slope-parallel frame", "vTHv", "{hn}m :vTHv"),
        _v("w_theta_v_cov_snsp", _KMS, "covariance w'theta_v' in the slope-normal / slope-parallel frame", "wTHv", "{hn}m :wTHv"),
        _v("w_theta_v_cov_vertical", _KMS, "vertical (gravity-aligned) component of the theta_v flux", "wTHv_vert", "{hn}m :wTHv_vert"),
    ],
    "scalar_flux_lateral": [
        _v("u_theta_v_cov_pf", _KMS, "covariance u'theta_v', planar fit + yaw", "uPF_thv", "{hn}m :uPF'theta_v'"),
        _v("u_h2o_cov_pf", "g m-2 s-1", "covariance u'h2o', planar fit + yaw", "uPF_H2O", "{hn}m :uPF'H2O'"),
        _v("u_h2o_wpl_cov_pf", "g m-2 s-1", "covariance u'h2o_wpl', planar fit + yaw", "uPF_H2O_WPL", "{hn}m :uPF'H2O_WPL'"),
        _v("u_co2_cov_pf", "kg m-2 s-1", "covariance u'co2', planar fit + yaw", "uPF_CO2", "{hn}m :uPF'CO2'"),
        _v("u_co2_wpl_cov_pf", "kg m-2 s-1", "covariance u'co2_wpl', planar fit + yaw", "uPF_CO2_WPL", "{hn}m :uPF'CO2_WPL'"),
    ],
    "latent_heat": [
        _v("Lv", "J g-1", "latent heat of vaporisation at the reference temperature", "Lv", "{hn}m Lv(J/g)"),
        _v("w_h2o_cov_raw", "g m-2 s-1", "covariance w'rho_v' (water vapour flux), unrotated sonic axes", "E_w", "{hn}m w':E(g/m^2s)"),
        _v("w_h2o_cov_pf", "g m-2 s-1", "covariance w'rho_v' (water vapour flux), planar fit + yaw", "E_wPF", "{hn}m wPF':E(g/m^2s)"),
        _v("w_h2o_wpl_cov_pf", "g m-2 s-1", "covariance w'rho_v' with the WPL density term, planar fit + yaw"),
        _v("w_q_wpl_cov_pf", "m s-1 kg m-3", "covariance w'q' with the WPL terms, planar fit + yaw", "wPF_qWPL", "{hn}m wPF'q_WPL'(m/s kg/m3)"),
        _v("LE_wpl_raw", "W m-2", "latent heat flux with the WPL correction, unrotated sonic axes", "LE_WPL_w", "{hn}m WPL, w' (W/m^2)"),
        _v("LE_wpl_pf", "W m-2", "latent heat flux with the WPL correction, planar fit + yaw", "LE_WPL_wPF", "{hn}m WPL, wPF' (W/m^2)"),
        _v("LE_o2_raw", "W m-2", "latent heat flux with the oxygen correction, no WPL, unrotated sonic axes", "LE_O2_w", "{hn}m O2 no WPL,w' (W/m^2)"),
        _v("LE_o2_pf", "W m-2", "latent heat flux with the oxygen correction, no WPL, planar fit + yaw", "LE_O2_wPF", "{hn}m O2 no WPL,wPF' (W/m^2)"),
    ],
    "flux_qc": [
        _v("Tau_ssitc", "1", "SSITC flag for Tau (Foken 0/1/2)", "TAU_SSITC", "{hn}m:TAU_SSITC_TEST"),
        _v("Tau_ss", "1", "steady-state-only flag for Tau", "TAU_SS", "{hn}m:TAU_SS_ONLY_TEST"),
        _v("H_ssitc", "1", "SSITC flag for H (Foken 0/1/2)", "H_SSITC", "{hn}m:H_SSITC_TEST"),
        _v("H_ss", "1", "steady-state-only flag for H", "H_SS", "{hn}m:H_SS_ONLY_TEST"),
        _v("LE_ssitc", "1", "SSITC flag for LE (Foken 0/1/2)", "LE_SSITC", "{hn}m:LE_SSITC_TEST"),
        _v("LE_ss", "1", "steady-state-only flag for LE", "LE_SS", "{hn}m:LE_SS_ONLY_TEST"),
        _v("Fc_ssitc", "1", "SSITC flag for Fc (Foken 0/1/2)", "FC_SSITC", "{hn}m:FC_SSITC_TEST"),
        _v("Fc_ss", "1", "steady-state-only flag for Fc", "FC_SS", "{hn}m:FC_SS_ONLY_TEST"),
    ],
    "co2_flux": [
        _v("w_co2_cov_raw", "kg m-2 s-1", "covariance w'rho_c' (CO2 mass flux), unrotated sonic axes", "Fc_w", "{hn}m w':CO2(kg/m^2s)"),
        _v("w_co2_cov_pf", "kg m-2 s-1", "covariance w'rho_c' (CO2 mass flux), planar fit + yaw", "Fc_wPF", "{hn}m wPF':CO2(kg/m^2s)"),
        _v("Fc_wpl_pf", "mol m-2 s-1", "CO2 flux with the WPL correction, planar fit + yaw", "Fc_WPL", "{hn}m WPL,wPF':CO2(mol/m^2s)"),
        _v("w_co2_wpl_cov_pf", "kg m-2 s-1", "covariance w'rho_c' with the WPL density term, planar fit + yaw", "wPF_CO2WPL", "{hn}m: wPF'CO2_WPL'(m/s kg/m^3)"),
        _v("co2_mole_fraction", "umol mol-1", "CO2 mole fraction in moist air", "ppm", "{hn}m: CO2 (ppm, moist-air molar ratio)"),
    ],
    "temperature": [
        _v("theta_fw", "degC", "fine-wire potential temperature, period mean", "theta_fw", "{hn} m: theta_fw"),
        _v("theta_v", "degC", "sonic virtual potential temperature, period mean", "theta_v_son", "{hn} m: theta_v_son"),
        _v("theta_v_fw", "degC", "fine-wire virtual potential temperature, period mean", "theta_v_fw", "{hn} m: theta_v_fw"),
        _v("t_air", "degC", "humidity-corrected sonic air temperature, period mean", "T_son_air", "{hn} m: T_son_air"),
    ],
    "humidity": [
        _v("q", "g kg-1", "specific humidity from the slow T/RH probe, period mean", "q", "{hn} m: q(g/kg)"),
        _v("theta_v_slow", "K", "virtual potential temperature from the slow T/RH probe, period mean", "virtualThetaAvg", "{hn} m: virtualThetaAvg(K)"),
        _v("r", "g kg-1", "mixing ratio, period mean", "rAvg", "{hn} m: rAvg(g/kg)"),
        _v("rho_air_moist", "kg m-3", "moist air density, period mean", "rho_airmoistAvg", "{hn} m: rho_airmoistAvg(kg/m^3)"),
        _v("rho_air_dry", "kg m-3", "dry air density, period mean", "rho_airdryAvg", "{hn} m: rho_airdryAvg(kg/m^3)"),
        _v("rho_h2o", "kg m-3", "water vapour density, period mean", "rho_H2OAvg", "{hn} m: rho_H2OAvg(kg/m^3)"),
    ],
}

# storage order of the groups in the run file and the legacy dict
GROUP_ORDER: List[str] = [GROUPS[k] for k in (
    "H", "Hlat", "tau", "tke", "sigma", "R", "L", "scaling", "eta", "delta_flux_ctrb",
    "delta_time_ctrb", "turbtr", "epsilon", "skew", "H_SNSP", "Flux_lat", "LHflux",
    "fluxQC", "CO2flux", "derivedT", "specificHum")]

# high-frequency netCDF: utespac-hf-1 name -> utespac-hf-2 name
HF_VARIABLES: Dict[str, str] = {
    "u": "u_pf", "v": "v_pf", "w": "w_pf",
    "u_tilt": "u_tilt", "v_tilt": "v_tilt", "w_tilt": "w_tilt",
    "Ts": "ts", "theta_v": "theta_v", "T_fw": "t_fw", "theta_fw": "theta_fw",
    "WD": "wind_dir", "spd": "wind_speed",
    "rhov": "rho_h2o", "rhoCO2": "rho_co2", "P": "p",
    "ustar": "ustar_pf", "L": "L",
    "ssitc_tau_ssitc_test": "Tau_ssitc", "ssitc_h_ssitc_test": "H_ssitc",
    "ssitc_le_ssitc_test": "LE_ssitc", "ssitc_fc_ssitc_test": "Fc_ssitc",
    "ssitc_tau_ss_only_test": "Tau_ss", "ssitc_h_ss_only_test": "H_ss",
    "ssitc_le_ss_only_test": "LE_ss", "ssitc_fc_ss_only_test": "Fc_ss",
}

# legacy CSV-only labels (wind statistics and rotated means) -> CSV stems
LEGACY_WIND_LABELS: Dict[str, str] = {"direction": "wind_dir", "speed": "wind_speed", "flag": "shadow_flag"}

# ── lookups ──────────────────────────────────────────────────────────────────

_LEGACY_INDEX: Dict[Tuple[str, str], Tuple[str, Var]] = {}
for _group, _vars in VARIABLES.items():
    for _var in _vars:
        if _var.legacy_key is not None:
            _table = _var.legacy_table or GROUPS_INVERSE[_group]
            _LEGACY_INDEX[(_table, _var.legacy_key)] = (_group, _var)

_NAME_INDEX: Dict[Tuple[str, str], Var] = {(g, v.name): v for g, vs in VARIABLES.items() for v in vs}


def legacy_to_new(table: str, key: str) -> Tuple[str, str]:
    """``("H", "Thv_wPF")`` -> ``("sensible_heat", "w_theta_v_cov_pf")``."""
    group, var = _LEGACY_INDEX[(table, key)]
    return group, var.name


def variable(group: str, name: str) -> Var:
    return _NAME_INDEX[(group, name)]


def group_of_legacy(table: str) -> str:
    return GROUPS[table]


def csv_header(name: str, height_label: Optional[str]) -> str:
    """``("w_ts_cov_raw", "10.85")`` -> ``"w_ts_cov_raw_10.85"``; no height -> the name."""
    return name if height_label is None else f"{name}_{height_label}"


def is_valid_name(name: str) -> bool:
    """Parser-safe and on-convention: ``[A-Za-z][A-Za-z0-9_]*``, every
    token lowercase unless it is one of :data:`SYMBOLS`."""
    import re
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
        return False
    for tok in name.split("_"):
        if tok and tok != tok.lower() and tok not in SYMBOLS:
            return False
    return True
