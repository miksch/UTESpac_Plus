"""simpleAvg – block-average a matrix based on a serial-date timestamp column."""

import warnings

import numpy as np
from .campbell_date import datetime_to_matlab_datenum, MATLAB_EPOCH
from datetime import datetime


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
    wd_col : int or None
        0-based column index of wind direction for vector averaging.
    ws_col : int or None
        0-based column index of wind speed for vector averaging.

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
        # Cannot find a valid timestamp column; return as-is
        warnings.warn("simple_avg: could not identify timestamp column; returning input unchanged.")
        return mat

    t = mat[:, t_col]

    # Number of averaging bins
    dt_days = avg_per / (24.0 * 60.0)
    N = round(np.ceil(t[-1]) - np.floor(t[0])) / dt_days
    N = int(round(N))

    if N == 0 or N > n_rows:
        warnings.warn("simple_avg: N out of range; returning input unchanged.")
        return mat

    # Check timestamp spacing consistency (within 0.5 s)
    half_second = 0.5 / 86400.0
    if (np.nanmax(np.diff(t)) - np.nanmin(np.diff(t))) > half_second:
        warnings.warn("simple_avg: timestamp spacing inconsistent; returning input unchanged.")
        return mat

    # Breakpoints
    bp = np.round(np.linspace(0, n_rows, N + 1)).astype(int)

    avg_mat = np.full((N, n_cols), np.nan)

    for i in range(N):
        chunk = mat[bp[i]:bp[i + 1], :]
        if len(chunk) == 0:
            continue
        avg_mat[i, :] = np.nanmean(chunk, axis=0)
        avg_mat[i, t_col] = t[bp[i + 1] - 1]  # timestamp = last sample in bin

        # Vector-average wind direction
        if wd_col is not None and ws_col is not None:
            ws = chunk[:, ws_col]
            wd_rad = np.deg2rad(chunk[:, wd_col])
            v_mean = np.nanmean(ws * np.sin(wd_rad))
            u_mean = np.nanmean(ws * np.cos(wd_rad))
            avg_mat[i, wd_col] = np.degrees(np.arctan2(v_mean, u_mean)) % 360.0

    if not return_timestamps:
        avg_mat = np.delete(avg_mat, t_col, axis=1)

    return avg_mat
