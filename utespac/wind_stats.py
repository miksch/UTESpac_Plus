"""Wind direction, speed and the tower-shadow flag.

:func:`wind_direction_speed` and :func:`shadow_flag` are the primitives
``utespac.stages.wind``, ``find_global_pf`` and the flux stage share.
"""

from typing import Tuple

import numpy as np

MANUFACTURER_RMYOUNG = 0
MANUFACTURER_CAMPBELL = 1
MANUFACTURER_GILL = 2


def sonic_axes(u, v, manufacturer: int):
    """``(u, v)`` in the Campbell convention (u toward the transducer array
    reference, v to the left): RMYoung swaps and negates, Gill negates both."""
    if int(manufacturer) == MANUFACTURER_RMYOUNG:
        return v, -np.asarray(u, dtype=float)
    if int(manufacturer) == MANUFACTURER_GILL:
        return -np.asarray(u, dtype=float), -np.asarray(v, dtype=float)
    return u, v


def wind_direction_speed(u, v, bearing: float, manufacturer: int = MANUFACTURER_CAMPBELL
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Meteorological wind direction [deg, 0-360) and speed from sonic u, v.

    *bearing* is the sonic head orientation (``SonicLevel.orientation``);
    the manufacturer axis convention is applied first.
    """
    u_d, v_d = sonic_axes(u, v, manufacturer)
    direction = (np.degrees(np.arctan2(-np.asarray(v_d), np.asarray(u_d))) + bearing) % 360.0
    return direction, np.hypot(u_d, v_d)


def shadow_sector(tower_bearing: float, sonic_bearing: float, envelope: float
                  ) -> Tuple[float, float]:
    """``(min_a, max_a)`` of the tower-shadow sector centred on
    ``tower_bearing + sonic_bearing``; ``min_a > max_a`` when it wraps 360."""
    centre = (float(tower_bearing) + float(sonic_bearing)) % 360.0
    if centre + envelope > 360.0:
        return centre - envelope, (centre + envelope) - 360.0
    if centre - envelope < 0.0:
        return 360.0 + centre - envelope, centre + envelope
    return centre - envelope, centre + envelope


def shadow_flag(direction, tower_bearing: float, sonic_bearing: float, envelope: float
                ) -> Tuple[np.ndarray, float, float]:
    """Flag (1.0 inside the tower-shadow sector) and the sector bounds."""
    min_a, max_a = shadow_sector(tower_bearing, sonic_bearing, envelope)
    d = np.asarray(direction, dtype=float)
    if min_a > max_a:       # wraps through north
        inside = (d > min_a) | (d < max_a)
    else:
        inside = (d > min_a) & (d < max_a)
    return inside.astype(float), min_a, max_a

