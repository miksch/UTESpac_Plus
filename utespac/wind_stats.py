"""windStats – mean wind speed, direction, and the tower-shadow flag.

:func:`wind_direction_speed` and :func:`shadow_flag` are the shared
primitives (``find_global_pf`` and ``fluxes`` use the same ones);
:func:`wind_stats` is the stage wrapper that fills ``output['spdAndDir']``.
"""

import warnings
from typing import Dict, List, Tuple

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


def wind_stats(output: Dict, sensor_info: Dict, table_names: List[str], info: Dict) -> Dict:
    """Compute wind speed and direction for each sonic and flag bad-sector data.

    Populates ``output['spdAndDir']`` and ``output['spdAndDirHeader']``.
    Also initialises ``output['warnings']``.
    """
    output.setdefault("warnings", [])

    if "u" not in sensor_info:
        return output

    num_sonics = sensor_info["u"].shape[0]
    tower_bearing = float(info.get("tower", 0))
    wind_env = float(info["windDirectionTest"]["envelopeSize"])

    for ii in range(num_sonics):
        try:
            tbl_idx = int(sensor_info["u"][ii, 0])
            bearing = float(sensor_info["u"][ii, 3])
            height = float(sensor_info["u"][ii, 2])
            manufact = int(sensor_info["u"][ii, 4]) if sensor_info["u"].shape[1] > 4 else 1

            u_col = int(sensor_info["u"][sensor_info["u"][:, 2] == height, 1][0])
            v_col = int(sensor_info["v"][sensor_info["v"][:, 2] == height, 1][0])
            tname = table_names[tbl_idx]

            t = output[tname][:, 0]
            direction, speed = wind_direction_speed(output[tname][:, u_col], output[tname][:, v_col],
                                                    bearing, manufact)
            flag, min_a, max_a = shadow_flag(direction, tower_bearing, bearing, wind_env)

            # spdAndDir: timestamps in col 0, then [dir, spd, flag] per sonic
            n = len(t)
            if "spdAndDir" not in output:
                output["spdAndDir"] = np.full((n, 1 + num_sonics * 3), np.nan)
                output["spdAndDirHeader"] = ["timeStamp"] + [""] * (num_sonics * 3)
                output["spdAndDir"][:, 0] = t

            c0 = 1 + ii * 3
            output["spdAndDir"][:n, c0] = direction
            output["spdAndDir"][:n, c0 + 1] = speed
            output["spdAndDir"][:n, c0 + 2] = flag
            output["spdAndDirHeader"][c0] = f"{height}m direction"
            output["spdAndDirHeader"][c0 + 1] = f"{height}m speed"
            output["spdAndDirHeader"][c0 + 2] = f"{height}m flag {min_a:.3g}<dir<{max_a:.3g}"

        except Exception as exc:
            msg = f"Unable to find wind stats at {height}m: {exc}"
            warnings.warn(msg)
            output["warnings"].append(msg)

    return output
