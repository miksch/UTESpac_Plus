"""get_virtulPotTemp – virtual potential temperature and air densities from HMP data."""

import numpy as np

from . import units
from .rh_to_spec_hum import sat_vapor_pressure


def get_virtual_pot_temp(altitude, level, Tair, RH, P_air=None, use_p_elevation=True,
                         T_unit=None, P_unit=None):
    """Compute virtual potential temperature and air densities.

    Parameters
    ----------
    altitude : float
        Site elevation above sea level [m]  (info["siteElevation"]).
    level : float
        Sensor height above the reference level [m]  (sonHeight - zRef).
    Tair : array_like
        Air temperature, in *T_unit*.
    RH : array_like
        Relative humidity [%].
    P_air : array_like or None
        Measured (or reference) pressure, in *P_unit*.  Only used when
        ``use_p_elevation=False``.
    use_p_elevation : bool
        If True (default), ignore ``P_air`` and compute pressure from the
        hypsometric formula using site elevation + sensor height.
        If False, use the supplied ``P_air`` value (matches MATLAB behaviour
        when a barometer is passed to get_virtulPotTemp with usePelevation=false).
    T_unit, P_unit : str, optional
        Declared units of *Tair* and *P_air*. When None, the magnitude
        fallback of :func:`utespac.units.convert` applies: K, as deg C below a
        median of 200; Pa, as kPa below a median of 1000.

    Returns
    -------
    virtual_theta  : ndarray  Virtual potential temperature [K]
    r              : ndarray  Mixing ratio [g/kg]
    rho_airmoist   : ndarray  Density of moist air [kg/m³]
    rho_airdry     : ndarray  Density of dry air [kg/m³]
    rho_H2O        : ndarray  Density of water vapour [kg/m³]
    """
    Tair = units.convert(Tair, T_unit, "temperature", "K", warn=False)
    RH   = np.asarray(RH,   dtype=float)

    if use_p_elevation or P_air is None:
        # Pressure from hypsometric formula [Pa]
        P_air = 101325.0 * (1.0 - 2.25577e-5 * (altitude + level)) ** 5.25588
    else:
        P_air = units.convert(P_air, P_unit, "pressure", "Pa", warn=False)

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
