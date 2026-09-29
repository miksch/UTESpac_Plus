"""Block averaging over the processing period (migration step 4, first stage).

Every stage splits a table into ``N`` equal row blocks, one per averaging
period; the files are complete days (``raw_processing`` writes the full
grid, NaN where nothing was logged), so the row split and the time split
coincide. This module holds that arithmetic once: :func:`n_periods` from
the day span, :func:`period_bounds` for the block edges, :func:`block_mean`
/ :func:`block_last` for the reductions, :func:`block_average` as the one
call the stages make. ``simple_avg`` and ``stp_dn`` are thin legacy
wrappers over it.

The kernels take the MATLAB serial datenum the run model renders for them
(``Run.hf_time``); the day-span rule lives in :func:`n_periods` and its
``datetime64`` twin :func:`n_periods_dt64`. :func:`validate_avg_min` and
:func:`period_bounds_uniform` guard equal-length periods;
:func:`period_blocks` and :func:`sub_period_stats` cut periods on the clock
grid of a ``datetime64`` series instead of by row count, and import pandas
when called.
"""

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

MINUTES_PER_DAY = 24.0 * 60.0


def epoch_s(times) -> np.ndarray:
    """Seconds since 1970-01-01 from a ``datetime64`` array, column or index.

    Parameters
    ----------
    times : array_like of datetime64
        Timestamps at any ``datetime64`` resolution.

    Returns
    -------
    numpy.ndarray
        Float seconds, at nanosecond resolution.
    """
    return np.asarray(times).astype("datetime64[ns]").astype("int64") / 1e9


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


def n_periods_dt64(t, avg_per_min: float, whole_days: bool = True) -> int:
    """:func:`n_periods` for a ``datetime64`` axis: the day span from the
    midnight before the first stamp to the midnight at or after the last."""
    t = np.asarray(t).astype("datetime64[ns]")
    if whole_days:
        d0 = t[0].astype("datetime64[D]").astype("datetime64[ns]")
        d1 = t[-1].astype("datetime64[D]").astype("datetime64[ns]")
        if d1 != t[-1]:
            d1 = d1 + np.timedelta64(1, "D")
        span_min = (d1 - d0) / np.timedelta64(1, "m")
    else:
        span_min = (t[-1] - t[0]) / np.timedelta64(1, "m")
    return int(round(span_min / float(avg_per_min)))


def period_bounds(n_rows: int, n_per: int) -> np.ndarray:
    """Block edges ``bp`` (length ``n_per + 1``): period ``j`` is rows
    ``bp[j]:bp[j+1]``; ``round(linspace(0, n_rows, n_per + 1))``."""
    return np.round(np.linspace(0, n_rows, n_per + 1)).astype(int)


def period_slices(n_rows: int, n_per: int) -> List[slice]:
    bp = period_bounds(n_rows, n_per)
    return [slice(int(bp[j]), int(bp[j + 1])) for j in range(n_per)]


def validate_avg_min(values: Sequence[float], span_min: float) -> List[float]:
    """Averaging periods [min] that tile a span of *span_min* minutes exactly.

    Each value must be a positive whole number of minutes and must divide
    the span evenly, so every period holds the same number of samples and
    shorter series aggregate onto longer ones without a remainder block.

    Parameters
    ----------
    values : sequence of float
        Requested averaging periods [min].
    span_min : float
        Length of the span the periods tile [min].

    Returns
    -------
    list of float
        The accepted periods, duplicates removed, in the order given.

    Raises
    ------
    ValueError
        If a value is not a positive whole number of minutes, or does not
        divide the span evenly.
    """
    out = []
    for value in values:
        v = float(value)
        if v <= 0 or v != round(v):
            raise ValueError(f"avg_min {v:g}: averaging period must be a "
                             f"positive whole number of minutes")
        n = span_min / v
        if n != round(n):
            raise ValueError(f"avg_min {v:g}: does not divide the "
                             f"{span_min:.0f}-min span evenly "
                             f"({n:.4f} periods)")
        if v not in out:
            out.append(v)
    return out


def period_bounds_uniform(t_days: np.ndarray, n_rows: int, avg_min: float
                          ) -> Tuple[np.ndarray, List[slice]]:
    """Equal-length block edges and row slices for *avg_min* over a grid.

    :func:`period_bounds` over the actual span of *t_days*
    (:func:`n_periods` with ``whole_days=False``), checked so that every
    period holds the same number of rows; a period mean is then the
    unweighted mean of its sub-period means whenever the sub-periods are
    equally covered.

    Parameters
    ----------
    t_days : numpy.ndarray
        Grid timestamps as serial days.
    n_rows : int
        Length of the grid.
    avg_min : float
        Averaging period [min].

    Returns
    -------
    bounds : numpy.ndarray of int
        Block edges, length ``n_per + 1``.
    slices : list of slice
        Row slice of each period.

    Raises
    ------
    ValueError
        If the blocks are not all the same length.
    """
    n_per = n_periods(t_days, avg_min, whole_days=False)
    bounds = period_bounds(n_rows, n_per)
    widths = np.diff(bounds)
    if widths.min() != widths.max():
        raise ValueError(f"{avg_min:g}-min periods do not tile {n_rows} rows "
                         f"evenly: block lengths {widths.min()}-{widths.max()}")
    return bounds, [slice(int(bounds[j]), int(bounds[j + 1])) for j in range(n_per)]


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


# ── clock-grid periods of a timestamp series ─────────────────────────────────

def period_blocks(timestamps, avg_per_min: float):
    """Contiguous row blocks of each averaging period on the clock grid.

    Periods are cut on a monotonised copy of the stamps (a running
    maximum), so a clock that steps backward leaves every period one
    contiguous run of rows with the record order inside it preserved.
    Periods holding no rows are absent. Requires pandas.

    Parameters
    ----------
    timestamps : array_like of datetime64
        Sample stamps.
    avg_per_min : float
        Averaging period [min]; the grid is aligned to the epoch.

    Returns
    -------
    starts : pandas.DatetimeIndex
        Start stamp of each populated period.
    bounds : numpy.ndarray
        Row index of each period start, with the row count appended.
    """
    import pandas as pd

    ns = np.asarray(timestamps).astype("datetime64[ns]").astype("int64")
    mono = np.maximum.accumulate(ns)
    step = int(avg_per_min * 60 * 1e9)
    pid = mono // step
    edges = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
    starts = pd.DatetimeIndex((pid[edges] * step).astype("datetime64[ns]"))
    return starts, np.r_[edges, len(ns)]


def sub_period_stats(ts, avg_min: float, channels, min_frac, fs: float):
    """Means, standard deviations and coverage of channels over sub-periods.

    The sub-periods are the :func:`period_blocks` of *ts* at *avg_min*, so
    every sub-period is one contiguous run of rows even where the clock
    steps backward. Requires pandas.

    Parameters
    ----------
    ts : array_like of datetime64
        Sample stamps.
    avg_min : float
        Sub-period length [min].
    channels : dict of str to numpy.ndarray
        One-dimensional arrays as long as *ts*; NaN marks a missing sample.
    min_frac : float or dict of str to float
        Coverage below which a channel's mean and standard deviation are
        blanked; a dict gives one threshold per channel name.
    fs : float
        Nominal sample rate [Hz]; coverage is the finite count over
        ``avg_min * 60 * fs``.

    Returns
    -------
    pandas.DataFrame
        One row per sub-period holding samples, with ``period_start_lst``,
        ``n_scans`` and, for each channel, ``<name>_mean``, ``<name>_sd``
        (population standard deviation, ``ddof = 0``) and ``<name>_frac``.
    """
    import pandas as pd

    ts = np.asarray(ts)
    cols = ["period_start_lst", "n_scans"]
    for name in channels:
        cols += [f"{name}_mean", f"{name}_sd", f"{name}_frac"]
    if len(ts) == 0:
        return pd.DataFrame(columns=cols)
    starts, bounds = period_blocks(ts, avg_min)
    expected = float(avg_min) * 60.0 * float(fs)
    rows = []
    for k in range(len(starts)):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        row = {"period_start_lst": starts[k], "n_scans": hi - lo}
        for name, x in channels.items():
            seg = np.asarray(x[lo:hi], dtype=float)
            ok = np.isfinite(seg)
            frac = float(ok.sum()) / expected
            thresh = min_frac[name] if isinstance(min_frac, dict) else min_frac
            if ok.any() and frac >= thresh:
                mean, sd = float(seg[ok].mean()), float(seg[ok].std())
            else:
                mean, sd = np.nan, np.nan
            row.update({f"{name}_mean": mean, f"{name}_sd": sd,
                        f"{name}_frac": frac})
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)
