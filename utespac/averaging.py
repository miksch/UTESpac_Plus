"""Block averaging over the processing period (migration step 4, first stage).

Every stage splits a table into ``N`` equal row blocks, one per averaging
period; the files are complete days (``raw_processing`` writes the full
grid, NaN where nothing was logged), so the row split and the time split
coincide. This module holds that arithmetic once: :func:`n_periods` from
the day span, :func:`period_bounds` for the block edges, :func:`block_mean`
/ :func:`block_last` for the reductions, :func:`block_average` as the one
call the stages make. ``avg``, ``simple_avg`` and ``stp_dn`` are thin
wrappers over it.

Time is still the MATLAB serial datenum the stages carry; the day-span
rule in :func:`n_periods` is the one place that knows it.
"""

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

MINUTES_PER_DAY = 24.0 * 60.0


def period_length_days(avg_per_min: float) -> float:
    return float(avg_per_min) / MINUTES_PER_DAY


def n_periods(t: np.ndarray, avg_per_min: float, whole_days: bool = True) -> int:
    """Number of averaging periods spanned by timestamps *t* (datenum).

    ``whole_days`` (the MATLAB ``completeTableCreate`` convention used by
    ``avg``/``simple_avg``) counts from the midnight before the first
    sample to the midnight after the last; otherwise the actual span
    ``t[-1] - t[0]`` is used (``fluxes``). Equal for complete day files.
    """
    t = np.asarray(t, dtype=float)
    dt = period_length_days(avg_per_min)
    if whole_days:
        span = np.ceil(t[-1]) - np.floor(t[0])
    else:
        span = t[-1] - t[0]
    return int(round(span / dt))


def period_bounds(n_rows: int, n_per: int) -> np.ndarray:
    """Block edges ``bp`` (length ``n_per + 1``): period ``j`` is rows
    ``bp[j]:bp[j+1]``; ``round(linspace(0, n_rows, n_per + 1))``."""
    return np.round(np.linspace(0, n_rows, n_per + 1)).astype(int)


def period_slices(n_rows: int, n_per: int) -> List[slice]:
    bp = period_bounds(n_rows, n_per)
    return [slice(int(bp[j]), int(bp[j + 1])) for j in range(n_per)]


def block_mean(values: np.ndarray, bounds: np.ndarray,
               vector_cols: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """``nanmean`` of *values* over each block of *bounds*.

    *vector_cols* lists ``(direction_col, speed_col)`` pairs whose direction
    column is averaged as a unit-vector mean (wind direction in degrees);
    speed-weighted, as MATLAB ``avg`` did for propeller anemometers. Empty
    blocks stay NaN.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    n_per = len(bounds) - 1
    out = np.full((n_per, v.shape[1]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN blocks
        for j in range(n_per):
            chunk = v[bounds[j]:bounds[j + 1]]
            if len(chunk) == 0:
                continue
            out[j] = np.nanmean(chunk, axis=0)
            for wd_col, ws_col in vector_cols or ():
                out[j, wd_col] = vector_mean_direction(chunk[:, wd_col], chunk[:, ws_col])
    return out


def vector_mean_direction(direction_deg: np.ndarray, speed: np.ndarray) -> float:
    """Speed-weighted unit-vector mean of a direction series, degrees in [0, 360)."""
    wd = np.deg2rad(direction_deg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v_m = np.nanmean(speed * np.sin(wd))
        u_m = np.nanmean(speed * np.cos(wd))
    return float(np.degrees(np.arctan2(v_m, u_m)) % 360.0)


def block_last(t: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Last timestamp of each block (the period's end stamp); NaN for empty blocks."""
    t = np.asarray(t, dtype=float)
    n_per = len(bounds) - 1
    out = np.full(n_per, np.nan)
    for j in range(n_per):
        if bounds[j + 1] > bounds[j]:
            out[j] = t[bounds[j + 1] - 1]
    return out


def block_average(t: np.ndarray, values: np.ndarray, avg_per_min: float,
                  vector_cols: Optional[Sequence[Tuple[int, int]]] = None,
                  whole_days: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Period-end timestamps and block means of *values* at *avg_per_min*."""
    n_per = n_periods(t, avg_per_min, whole_days)
    bounds = period_bounds(len(t), n_per)
    return block_last(t, bounds), block_mean(values, bounds, vector_cols)


def block_mean_rows(data: np.ndarray, rows: int) -> np.ndarray:
    """Mean over consecutive groups of *rows* rows (``stp_dn``); the remainder
    is dropped."""
    d = np.asarray(data, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n_out = d.shape[0] // rows
    bounds = np.arange(0, (n_out + 1) * rows, rows)
    return block_mean(d[: n_out * rows], bounds)
