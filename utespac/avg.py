"""avg – block-average each data table to the configured averaging period.

Wrapper over :mod:`utespac.averaging` that fills the legacy ``output`` dict
(``<table>`` matrix with the period-end timestamp in column 0 and
``<table>Header``).
"""

from typing import Dict, List, Optional
import logging
import numpy as np

from .averaging import block_average

log = logging.getLogger("utespac")


def vector_average_columns(sensor_info: Dict, table_index: int):
    """``(direction_col, speed_col)`` pairs of the propeller anemometers in
    table *table_index* (``birdDir``/``birdSpd`` templates)."""
    pairs = []
    if "birdDir" in sensor_info and "birdSpd" in sensor_info:
        for k in range(sensor_info["birdDir"].shape[0]):
            if int(sensor_info["birdDir"][k, 0]) == table_index:
                pairs.append((int(sensor_info["birdDir"][k, 1]), int(sensor_info["birdSpd"][k, 1])))
    return pairs


def avg(
    data: List[Optional[np.ndarray]],
    info: Dict,
    table_names: List[str],
    output: Dict,
    headers: List[List],
    sensor_info: Dict,
) -> Dict:
    """Store the block average of every table in *output*.

    Wind direction from propeller anemometers (``birdDir``) is averaged as a
    speed-weighted unit vector.

    Returns
    -------
    output : dict  (updated with ``<tableName>`` and ``<tableName>Header`` keys)
    """
    for ii, tbl in enumerate(data):
        tname = table_names[ii]
        log.info(f"Averaging {tname}")
        if tbl is None or tbl.size == 0:
            continue
        pairs = [(wd, ws) for wd, ws in vector_average_columns(sensor_info, ii)
                 if wd < tbl.shape[1] and ws < tbl.shape[1]]
        t_end, means = block_average(tbl[:, 0], tbl, info["avgPer"], vector_cols=pairs)
        means[:, 0] = t_end
        output[tname] = means
        output[f"{tname}Header"] = headers[ii]
    return output
