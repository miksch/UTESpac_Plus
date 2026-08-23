"""simpleAvg – block-average a matrix based on a serial-date timestamp column.

Legacy entry point: the timestamp column is located by value (a datenum
between 2000 and 2030), the spacing is checked, and the reduction is
:func:`utespac.averaging.block_mean`. New code passes the time axis
explicitly to :func:`utespac.averaging.block_average` instead.
"""

import warnings
from datetime import datetime

import numpy as np

from .averaging import block_last, block_mean, n_periods, period_bounds
from .campbell_date import datetime_to_matlab_datenum


# MATLAB datenum bounds for identifying a timestamp column
_DATENUM_2000 = datetime_to_matlab_datenum(datetime(2000, 1, 1))
_DATENUM_2030 = datetime_to_matlab_datenum(datetime(2030, 1, 1))


def simple_avg(
    input_mat: np.ndarray,
    avg_per: float,
    return_timestamps: bool = True,
    wd_col: int = None,
    ws_col: int = None,
) -> np.ndarray:
    """Block-average *input_mat* to *avg_per*-minute intervals.

    Parameters
    ----------
    input_mat : ndarray, shape (N, M)
        Last column must be the MATLAB serial timestamp (located automatically).
    avg_per : float
        Averaging period in minutes.
    return_timestamps : bool
        If False, the timestamp column is dropped from the output.
    wd_col, ws_col : int or None
        0-based wind-direction and wind-speed columns for vector averaging.

    Returns
    -------
    avg_mat : ndarray
    """
    mat = np.asarray(input_mat, dtype=float)
    n_rows, n_cols = mat.shape

    # --- locate timestamp column (last column whose first row looks like a date)
    t_col = None
    for c in range(n_cols - 1, -1, -1):
        if _DATENUM_2000 < mat[0, c] < _DATENUM_2030:
            t_col = c
            break
    if t_col is None:
        warnings.warn("simple_avg: could not identify timestamp column; returning input unchanged.")
        return mat

    t = mat[:, t_col]
    N = n_periods(t, avg_per)
    if N == 0 or N > n_rows:
        warnings.warn("simple_avg: N out of range; returning input unchanged.")
        return mat

    # Check timestamp spacing consistency (within 0.5 s)
    half_second = 0.5 / 86400.0
    if (np.nanmax(np.diff(t)) - np.nanmin(np.diff(t))) > half_second:
        warnings.warn("simple_avg: timestamp spacing inconsistent; returning input unchanged.")
        return mat

    bounds = period_bounds(n_rows, N)
    pairs = [(wd_col, ws_col)] if wd_col is not None and ws_col is not None else None
    avg_mat = block_mean(mat, bounds, vector_cols=pairs)
    avg_mat[:, t_col] = block_last(t, bounds)

    if not return_timestamps:
        avg_mat = np.delete(avg_mat, t_col, axis=1)
    return avg_mat
