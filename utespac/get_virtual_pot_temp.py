"""get_virtulPotTemp – virtual potential temperature and air densities from HMP data."""

import numpy as np

from .rh_to_spec_hum import sat_vapor_pressure


def get_virtual_pot_temp(altitude, level, Tair, RH, P_air=None, use_p_elevation=True):
    """Compute virtual potential temperature and air densities.

    Parameters
    ----------
    altitude : float
        Site elevation above sea level [m]  (info["siteElevation"]).
    level : float
        Sensor height above the reference level [m]  (sonHeight - zRef).
    Tair : array_like
        Air temperature [°C or K – auto-detected].
    RH : array_like
        Relative humidity [%].
    P_air : array_like or None
        Measured (or reference) pressure.  Only used when
        ``use_p_elevation=False``.  Values < 1000 are assumed to be in kPa
        and are converted to Pa automatically.
    use_p_elevation : bool
        If True (default), ignore ``P_air`` and compute pressure from the
        hypsometric formula using site elevation + sensor height.
        If False, use the supplied ``P_air`` value (matches MATLAB behaviour
        when a barometer is passed to get_virtulPotTemp with usePelevation=false).

    Returns
    -------
    virtual_theta  : ndarray  Virtual potential temperature [K]
    r              : ndarray  Mixing ratio [g/kg]
    rho_airmoist   : ndarray  Density of moist air [kg/m³]
    rho_airdry     : ndarray  Density of dry air [kg/m³]
    rho_H2O        : ndarray  Density of water vapour [kg/m³]
    """
    Tair = np.asarray(Tair, dtype=float)
    RH   = np.asarray(RH,   dtype=float)

    if np.nanmedian(Tair) < 200:
        Tair = Tair + 273.15          # → K

    if use_p_elevation or P_air is None:
        # Pressure from hypsometric formula [Pa]
        P_air = 101325.0 * (1.0 - 2.25577e-5 * (altitude + level)) ** 5.25588
    else:
        P_air = np.asarray(P_air, dtype=float)
        if np.nanmedian(P_air) < 1000:   # kPa → Pa
            P_air = P_air * 1000.0

    # Saturated vapour pressure [Pa] (Stull 1988 eq. 7.5.2d, shared formula)
    e_sat = 1000.0 * sat_vapor_pressure(Tair)
    e_air = e_sat * (RH / 100.0)      # actual vapour pressure [Pa]

    r = 621.97 * e_air / (P_air - e_air)  # mixing ratio [g/kg]

    Gamma = 0.0098   # dry adiabatic lapse rate [K/m]
    theta = Tair + Gamma * level

    virtual_theta = theta * (1.0 + 0.61 * r / 1000.0)  # [K]

    Rd = 287.058   # [J/(kg·K)]
    Rv = 461.495   # [J/(kg·K)]
    rho_airdry   = (P_air - e_air) / (Rd * Tair)
    rho_H2O      = e_air / (Rv * Tair)
    rho_airmoist = rho_airdry + rho_H2O

    return virtual_theta, r, rho_airmoist, rho_airdry, rho_H2O
