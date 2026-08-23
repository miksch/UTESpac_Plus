"""findSerialDate – convert Campbell 4-column date vector to MATLAB serial dates."""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
from .campbell_date import campbell_date_to_serial_date, matlab_datenum_to_datetime

log = logging.getLogger("utespac")


def find_serial_date(
    data: List[Optional[np.ndarray]],
    data_info: List[List[str]],
    info: Dict,
) -> Tuple[List[Optional[np.ndarray]], List[List[str]], Dict]:
    """Replace the first 4 date columns with a single serial-date column.

    Modifies *data* in-place (replaces columns 1–4 with col 1 = serial date).

    Parameters
    ----------
    data : list of ndarray or None
    data_info : list of list of str
    info : dict

    Returns
    -------
    data, data_info, info (all updated)
    """
    log.info("Storing serial date in column 1 and deleting columns 2–4")
    for i, tbl in enumerate(data):
        if tbl is None or tbl.size == 0:
            continue
        log.info(f"  Finding serial dates for table {i + 1}")

        # Columns 0–3 are [year, doy, HHMM, seconds]
        serial = campbell_date_to_serial_date(tbl[:, :4])
        # Store serial in col 0 and delete cols 1–3
        tbl[:, 0] = serial
        data[i] = np.delete(tbl, [1, 2, 3], axis=1)

        # Store date info
        beg_dt = matlab_datenum_to_datetime(data[i][0, 0])
        end_dt = matlab_datenum_to_datetime(data[i][-1, 0])
        data_info[i] = data_info[i] + [
            f"beg date: {beg_dt.strftime('%d-%b-%Y %H:%M:%S')}",
            f"end date: {end_dt.strftime('%d-%b-%Y %H:%M:%S')}",
        ]
        info["date"] = beg_dt.strftime("%Y_%m_%d")

    return data, data_info, info
