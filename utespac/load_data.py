"""loadData – import a set of CSV data files for one date."""

import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("utespac")


def load_data(
    data_files_row: List[Optional[str]],
    current_date_num: int,
    total_dates: int,
    info: Dict,
    table_names: List[str],
) -> Tuple[List[Optional[np.ndarray]], List[List[str]]]:
    """Load one row of CSV data files into numeric arrays.

    Parameters
    ----------
    data_files_row : list of str or None
        Full paths to CSV files for each table in this date row.
    current_date_num : int
        1-based index of the current date (for progress display).
    total_dates : int
        Total number of dates being processed.
    info : dict
        UTESpac configuration dict with keys:
        ``tableNames``, ``tableScanFrequency``, ``tableNumberOfColumns``.
    table_names : list of str
        Names of each table in the current site run.

    Returns
    -------
    data : list of ndarray or None
        One array per table, columns = [col1, col2, ...] (raw CSV values).
    data_info : list of list of str
        Metadata strings per table column.
    """
    log.info(
        f"\nEvaluating date {current_date_num} of {total_dates} "
        f"in folder {info.get('siteFolder', '')}"
    )

    n_tables = len(data_files_row)
    data: List[Optional[np.ndarray]] = [None] * n_tables
    data_info: List[List[str]] = [[] for _ in range(n_tables)]

    expected_cols_map: dict = {}
    freq_map: dict = {}
    for i, tname in enumerate(info.get("tableNames", [])):
        expected_cols_map[tname] = info["tableNumberOfColumns"][i]
        freq_map[tname]          = info["tableScanFrequency"][i]

    for tbl_idx, fpath in enumerate(data_files_row):
        if fpath is None or not os.path.isfile(fpath):
            continue
        try:
            tname = table_names[tbl_idx]
            tname_info = tname if tname in expected_cols_map else table_names[tbl_idx]
            expected_cols = expected_cols_map.get(tname_info, None)
            scan_freq     = freq_map.get(tname_info, None)
            log.info(
                f"\n  Loading {os.path.basename(fpath)}"
                + (f"  expected cols={expected_cols}" if expected_cols else "")
            )

            temp = np.genfromtxt(fpath, delimiter=",", filling_values=np.nan)
            if temp.ndim == 1:
                temp = temp[np.newaxis, :]

            # Replace common Campbell sentinel values with NaN
            for sentinel in (-7999.0, -99999.0):
                temp[temp == sentinel] = np.nan

            # Pad or trim columns
            if expected_cols is not None:
                if temp.shape[1] < expected_cols:
                    import warnings
                    warnings.warn(
                        f"Table {tname}: found {temp.shape[1]} cols, expected {expected_cols}. "
                        "Padding with NaN."
                    )
                    pad = np.full((temp.shape[0], expected_cols - temp.shape[1]), np.nan)
                    temp = np.hstack([temp, pad])
                elif temp.shape[1] > expected_cols:
                    import warnings
                    warnings.warn(
                        f"Table {tname}: found {temp.shape[1]} cols, expected {expected_cols}. "
                        "Trimming extra columns."
                    )
                    temp = temp[:, :expected_cols]

            # Round seconds to 2 decimal places (column index 3, 0-based)
            if temp.shape[1] >= 4:
                temp[:, 3] = np.round(temp[:, 3] * 100) / 100.0

            # Drop the midnight row (HHMM=0, secs=0.00) at the start of the file.
            # MATLAB's completeTableCreate starts its template at beginSerialDay + dt
            # (i.e. secs=0.05 at 20 Hz), so the midnight sample is never merged in.
            if temp.shape[1] >= 4 and len(temp) > 0 and temp[0, 2] == 0.0 and temp[0, 3] == 0.0:
                temp = temp[1:]

            # Remove duplicates and sort by encoded date key
            if temp.shape[1] >= 4:
                date_key = (
                    np.round(temp[:, 1] * 1e8)
                    + np.round(temp[:, 2] * 10000)
                    + np.round(temp[:, 3] * 100)
                )
                _, ia = np.unique(date_key, return_index=True)
                temp = temp[ia, :]

                # Trim to 2-day window from first row
                cut_off = np.round((date_key[0] + 2e8) / 100) * 100
                temp = temp[date_key[ia] <= cut_off, :]

            # Replace -7999 sentinel with NaN
            temp[temp == -7999] = np.nan

            # Append the final midnight row if missing.
            # MATLAB's completeTableCreate ends at endSerialDay+1 (midnight of the day
            # after the last data day). The raw file typically stops at 23:59:59.95
            # (one sample before that midnight). Appending a NaN midnight row gives
            # exactly scan_freq*86400*n_days samples — needed so dir_hf in GPF
            # rotation aligns with averaging-period boundaries.
            if temp.shape[1] >= 4 and len(temp) > 2:
                from datetime import datetime, timedelta
                year_end = int(temp[-2, 0])
                doy_end  = int(temp[-2, 1])
                next_day = datetime(year_end, 1, 1) + timedelta(days=int(doy_end))
                year_next = next_day.year
                doy_next  = (next_day - datetime(next_day.year, 1, 1)).days + 1
                last_is_midnight = (
                    temp[-1, 0] == year_next and temp[-1, 1] == doy_next
                    and temp[-1, 2] == 0.0 and temp[-1, 3] == 0.0
                )
                if not last_is_midnight:
                    midnight_row = np.full((1, temp.shape[1]), np.nan)
                    midnight_row[0, 0] = float(year_next)
                    midnight_row[0, 1] = float(doy_next)
                    midnight_row[0, 2] = 0.0
                    midnight_row[0, 3] = 0.0
                    temp = np.vstack([temp, midnight_row])

            data[tbl_idx]      = temp
            data_info[tbl_idx] = [f"file: {os.path.basename(fpath)}"]

        except Exception as exc:
            import warnings
            warnings.warn(f"Failed to load {fpath}: {exc}")

    return data, data_info
