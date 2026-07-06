"""findInstruments – map sensor templates to [table, column, height] triplets."""

from typing import Dict, List
import numpy as np
from .strfndw import strfndw
from .site_config import sonic_for


def find_instruments(
    headers: List[List],
    template: Dict[str, str],
    info: Dict,
) -> Dict[str, np.ndarray]:
    """Match sensor name templates against table headers.

    Parameters
    ----------
    headers : list of [names_row, heights_row]
        One entry per data table (output of import_header).
    template : dict
        Keys are sensor field names, values are wildcard patterns (e.g. ``'Ux_*'``).
    info : dict
        UTESpac info dict.  ``info['sonicManufact']`` and
        ``info['sonicOrientation']`` are attached to sonic entries.

    Returns
    -------
    sensor_info : dict
        Each key is a template field name; value is an ndarray with columns
        [table_index (0-based), column_index (0-based), height].
        Sonic sensors also carry columns [bearing, manufacturer_code].
    """
    sensor_info: Dict[str, np.ndarray] = {}

    for tbl_idx, header in enumerate(headers):
        names = header[0]

        for field, pattern in template.items():
            if not pattern:
                continue

            matches = strfndw(names, pattern)
            if not matches:
                continue

            for col_idx in matches:
                name = names[col_idx]
                # Extract height: last numeric run in the name
                import re
                nums = re.findall(r"[\d.]+", name)
                height = float(nums[-1]) if nums else np.nan

                row = [tbl_idx, col_idx, height]

                # Attach bearing and manufacturer code for sonic u-component
                if field == "u":
                    level = sonic_for(info.get("sonics"), height)
                    if level is not None:
                        # tower profile: look up by height (order-independent)
                        bearing, manufact = level.orientation, level.manufacturer
                    else:
                        # legacy parallel lists: indexed by discovery order
                        bearing_arr = info.get("sonicOrientation", [0])
                        manufact_arr = info.get("sonicManufact", [1])
                        sonic_count = len(sensor_info.get("u", np.empty((0, 5))))
                        bearing = bearing_arr[sonic_count] if sonic_count < len(bearing_arr) else 0
                        manufact = manufact_arr[sonic_count] if sonic_count < len(manufact_arr) else 1
                    row = [tbl_idx, col_idx, height, float(bearing), float(manufact)]

                row_arr = np.array(row, dtype=float)

                if field not in sensor_info:
                    sensor_info[field] = row_arr[np.newaxis, :]
                else:
                    sensor_info[field] = np.vstack([sensor_info[field], row_arr])

    # Print summary
    for field, arr in sensor_info.items():
        print(f"  {field}: {arr.shape[0]} sensor(s) found")

    return sensor_info
