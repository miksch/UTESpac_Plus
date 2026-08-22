"""calc_SNSP_angle – slope-normal / slope-parallel angle decomposition."""

import numpy as np


def calc_snsp_angle(phi: float, alpha: float, downslope_aspect: float):
    """Decompose wind direction angles in the slope coordinate system.

    Parameters
    ----------
    phi : float
        Meteorological wind direction from north [degrees].
    alpha : float
        Slope angle [degrees].
    downslope_aspect : float
        Direction the slope falls toward, from north [degrees]
        (``info["downslopeAspect"]``; French Meadows: 30).

    Returns
    -------
    a1 : float  Angle between u_slope-normal and true vertical [degrees].
    a2 : float  Angle between v_slope-normal and true vertical [degrees].
    """
    kesi = phi - downslope_aspect   # wind direction relative to the fall line
    a1 = np.degrees(np.arcsin(np.cos(np.radians(kesi))      * np.sin(np.radians(alpha))))
    a2 = np.degrees(np.arcsin(np.cos(np.radians(kesi - 90)) * np.sin(np.radians(alpha))))
    return a1, a2
