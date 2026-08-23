"""avg – block-average each data table to the configured averaging period."""

from typing import Dict, List, Optional
import logging
import numpy as np

log = logging.getLogger("utespac")


def avg(
    data: List[Optional[np.ndarray]],
    info: Dict,
    table_names: List[str],
    output: Dict,
    headers: List[List],
    sensor_info: Dict,
) -> Dict:
    """Compute block averages for every table and store in *output*.

    Wind direction from propeller anemometers (``birdDir``) is averaged
    using unit-vector method.

    Returns
    -------
    output : dict  (updated with ``<tableName>`` and ``<tableName>Header`` keys)
    """
    for ii, tbl in enumerate(data):
        tname = table_names[ii]
        log.info(f"Averaging {tname}")
        if tbl is None or tbl.size == 0:
            continue

        t = tbl[:, 0]
        # Use time-span to determine N (matches MATLAB's simpleAvg / completeTableCreate)
        dt_days  = info["avgPer"] / (24.0 * 60.0)
        n_periods = int(round((np.ceil(t[-1]) - np.floor(t[0])) / dt_days))
        bp = np.round(np.linspace(0, tbl.shape[0], n_periods + 1)).astype(int)

        avg_table = np.full((n_periods, tbl.shape[1]), np.nan)

        for jj in range(n_periods):
            chunk = tbl[bp[jj]:bp[jj + 1], :]
            if len(chunk) == 0:
                continue
            avg_table[jj, :] = np.nanmean(chunk, axis=0)
            avg_table[jj, 0] = chunk[-1, 0]  # timestamp = last sample

            # Vector-average wind direction from prop anemometers
            if "birdDir" in sensor_info:
                for k in range(sensor_info["birdDir"].shape[0]):
                    if int(sensor_info["birdDir"][k, 0]) == ii:
                        ws_col = int(sensor_info["birdSpd"][k, 1])
                        wd_col = int(sensor_info["birdDir"][k, 1])
                        ws  = chunk[:, ws_col] if ws_col < chunk.shape[1] else np.full(len(chunk), np.nan)
                        wd  = np.deg2rad(chunk[:, wd_col] if wd_col < chunk.shape[1] else np.full(len(chunk), np.nan))
                        v_m = np.nanmean(ws * np.sin(wd))
                        u_m = np.nanmean(ws * np.cos(wd))
                        avg_table[jj, wd_col] = np.degrees(np.arctan2(v_m, u_m)) % 360.0

        output[tname]              = avg_table
        output[f"{tname}Header"]   = headers[ii]

    return output
