"""Air state and open-path density corrections.

:func:`air_state` partitions a measured pressure between dry air and
vapour with the air temperature from the sonic temperature;
:func:`wpl_ch4_flux` and :func:`wpl_vapour_flux` apply the Webb et al.
(1980) density correction to a trace-gas and a vapour covariance;
:func:`latent_heat_vaporisation` and :func:`ch4_mole_fraction_ppm`
convert between flux and concentration units.
"""

import numpy as np

from ..sonic_temperature import SONIC_HUMIDITY_COEFF

R_DRY = 287.058          # [J/(kg K)] gas constant of dry air
R_VAP = 461.495          # [J/(kg K)] gas constant of water vapour
MU_WPL = 1.6077          # m_dry / m_vapour (Webb et al. 1980)
CP_DRY = 1004.67         # [J/(kg K)] specific heat of dry air at constant pressure
R_GAS_J_MOL_K = 8.314462618     # molar gas constant [J mol-1 K-1], CODATA 2018


def air_state(ts_c, h2o_g_m3, press_kpa, n_iter=3):
    """Air temperature and the dry, vapour and specific-humidity state.

    Solves the Schotanus et al. (1983) relation ``T = T_s / (1 + 0.51 q)``
    together with the ideal-gas partition of the measured total pressure
    between dry air and vapour.

    Parameters
    ----------
    ts_c : array_like
        Sonic temperature [C].
    h2o_g_m3 : array_like
        Vapour density [g/m3].
    press_kpa : array_like
        Total pressure [kPa].
    n_iter : int, optional
        Fixed-point iterations; three is ample at ambient humidity.

    Returns
    -------
    ta_k : ndarray
        Air temperature [K].
    rho_v : ndarray
        Vapour density [kg/m3].
    rho_d : ndarray
        Dry-air density [kg/m3].
    q : ndarray
        Specific humidity [kg/kg].
    """
    ts_k = np.asarray(ts_c, dtype=float) + 273.15
    rho_v = np.asarray(h2o_g_m3, dtype=float) / 1000.0
    p = np.asarray(press_kpa, dtype=float) * 1000.0
    ta_k = ts_k.copy()
    q = np.zeros_like(ts_k)
    for _ in range(n_iter):
        rho_d = (p - rho_v * R_VAP * ta_k) / (R_DRY * ta_k)
        q = rho_v / (rho_d + rho_v)
        ta_k = ts_k / (1.0 + SONIC_HUMIDITY_COEFF * q)
    rho_d = (p - rho_v * R_VAP * ta_k) / (R_DRY * ta_k)
    q = rho_v / (rho_d + rho_v)
    return ta_k, rho_v, rho_d, q


def wpl_ch4_flux(w_rhoc, w_rhov, w_temp, rho_c, rho_v, rho_d, ta_k):
    """Open-path density-corrected methane flux (Webb et al. 1980, eq. 24).

    ``F = w'rho_c' + mu (rho_c / rho_d) w'rho_v' + (1 + mu sigma) (rho_c / T) w'T'``
    with ``mu = m_d / m_v`` and ``sigma = rho_v / rho_d``. The structure is
    that of McDermitt et al. (2011) for the LI-7700 with the spectroscopic
    multipliers set to unity, so no pressure, temperature or vapour
    broadening of the absorption line is applied.

    Parameters
    ----------
    w_rhoc : float or ndarray
        Covariance of vertical velocity with methane density
        [rho_c-units m/s].
    w_rhov : float or ndarray
        Covariance of vertical velocity with vapour density [kg m-2 s-1].
    w_temp : float or ndarray
        Kinematic sensible heat flux [K m/s], from the humidity-corrected
        sonic temperature.
    rho_c : float or ndarray
        Mean methane density, in the units the flux is wanted in.
    rho_v, rho_d : float or ndarray
        Mean vapour and dry-air density [kg/m3].
    ta_k : float or ndarray
        Mean air temperature [K].

    Returns
    -------
    float or ndarray
        Flux in ``rho_c`` units per m2 per s.
    """
    sigma = rho_v / rho_d
    return (w_rhoc
            + MU_WPL * (rho_c / rho_d) * w_rhov
            + (1.0 + MU_WPL * sigma) * (rho_c / ta_k) * w_temp)


def wpl_vapour_flux(w_rhov, w_temp, rho_v, rho_d, ta_k):
    """Density-corrected water-vapour flux [kg m-2 s-1] (Webb et al. 1980 eq. 25).

    Parameters
    ----------
    w_rhov : float or ndarray
        Covariance of vertical velocity with vapour density [kg m-2 s-1].
    w_temp : float or ndarray
        Kinematic sensible heat flux [K m/s].
    rho_v, rho_d : float or ndarray
        Mean vapour and dry-air density [kg/m3].
    ta_k : float or ndarray
        Mean air temperature [K].

    Returns
    -------
    float or ndarray
        ``(1 + mu sigma) (w'rho_v' + rho_v w'T' / T)``.
    """
    sigma = rho_v / rho_d
    return (1.0 + MU_WPL * sigma) * (w_rhov + rho_v * w_temp / ta_k)


def latent_heat_vaporisation(ta_k):
    """Latent heat of vaporisation [J/kg] at air temperature *ta_k* [K].

    ``L_v = 1000 (2500.8 - 2.36 (T - 273.15))``.
    """
    return 1.0e3 * (2500.8 - 2.36 * (np.asarray(ta_k, dtype=float) - 273.15))


def ch4_mole_fraction_ppm(ch4d_mmol_m3, press_kpa, temp_c):
    """Wet methane mole fraction from number density, cell pressure and temperature.

    The ideal-gas molar density of the sampled air is ``P / (R T)``, so the
    mole fraction is ``rho_c R T / P``. The LI-7700 is an open-path sensor and
    measures in the ambient, humid air, so the result is a WET mole fraction:
    no water-vapour dilution is removed.

    Parameters
    ----------
    ch4d_mmol_m3 : array_like
        Methane number density [mmol m-3].
    press_kpa : array_like
        Cell pressure [kPa].
    temp_c : array_like
        Cell temperature [deg C].

    Returns
    -------
    numpy.ndarray
        Wet mole fraction [umol mol-1]; NaN wherever an input is NaN.
    """
    c = np.asarray(ch4d_mmol_m3, dtype=float)
    p = np.asarray(press_kpa, dtype=float)
    t = np.asarray(temp_c, dtype=float)
    return c * 1e-3 * R_GAS_J_MOL_K * (t + 273.15) / (p * 1e3) * 1e6
