"""RHtoSpecHum – convert relative humidity to specific humidity."""

import numpy as np


def sat_vapor_pressure(T):
    """Saturation vapour pressure over liquid water [kPa] at temperature *T* [K].

    Bolton's fit as given by Stull (1988), eq. 7.5.2d, p. 276:
    e_sat = 0.6112 kPa · exp[17.67 (T − 273.16) / (T − 29.66)].
    The one formula used everywhere in utespac (``rh_to_spec_hum``,
    ``get_virtual_pot_temp``).
    """
    T = np.asarray(T, dtype=float)
    return 0.6112 * np.exp(17.67 * (T - 273.16) / (T - 29.66))


def rh_to_spec_hum(rh: np.ndarray, P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Convert relative humidity to specific humidity.

    Parameters
    ----------
    rh : array_like
        Relative humidity [%].
    P : array_like
        Pressure [kPa].
    T : array_like
        Temperature [K].

    Returns
    -------
    q : ndarray
        Specific humidity [kg/kg], q = 0.622 e / (P − 0.378 e)
        (Stull 1988, eq. 7.5.2c/13.1.4b form with the moist denominator).
    """
    rh = np.asarray(rh, dtype=float)
    P  = np.asarray(P,  dtype=float)
    T  = np.asarray(T,  dtype=float)

    e = rh / 100.0 * sat_vapor_pressure(T)      # actual vapour pressure [kPa]
    return 0.622 * e / (P - 0.378 * e)
