"""Conversion utilities between Campbell date vectors, MATLAB serial date
numbers and ``numpy.datetime64``.

The serial-datenum functions are the legacy shim: the processing stages
still carry time as MATLAB datenums (days since year 0), and the labeled
I/O boundary (:mod:`utespac.labeled`, :mod:`utespac.export_hf`) converts
to ``datetime64`` with the vectorized pair below.
"""

import numpy as np
from datetime import datetime, timedelta

# MATLAB serial date for 1970-01-01 00:00:00 UTC (well-known constant)
MATLAB_EPOCH = 719529.0


def datetime_to_matlab_datenum(dt: datetime) -> float:
    """Convert a Python datetime to a MATLAB serial date number."""
    return (dt - datetime(1970, 1, 1)).total_seconds() / 86400.0 + MATLAB_EPOCH


def matlab_datenum_to_datetime(md: float) -> datetime:
    """Convert a MATLAB serial date number to a Python datetime."""
    return datetime(1970, 1, 1) + timedelta(seconds=(md - MATLAB_EPOCH) * 86400.0)


def matlab_datenum_to_datetime64(md) -> np.ndarray:
    """Vectorized datenum → ``datetime64[ms]``; NaN → NaT.

    Rounded to the millisecond: float64 datenums resolve ~1e-5 s in this
    era and Campbell timestamps carry at most centiseconds, so the rounding
    is lossless for logger data.
    """
    md = np.asarray(md, dtype=float)
    ms = np.round((md - MATLAB_EPOCH) * 86400.0 * 1e3)
    out = np.full(md.shape, np.datetime64("NaT", "ms"))
    ok = np.isfinite(ms)
    out[ok] = np.datetime64("1970-01-01T00:00:00", "ms") + ms[ok].astype("int64").astype("timedelta64[ms]")
    return out


def datetime64_to_matlab_datenum(dt64) -> np.ndarray:
    """Vectorized ``datetime64`` → MATLAB datenum (float days); NaT → NaN."""
    scalar = np.ndim(dt64) == 0
    dt64 = np.atleast_1d(np.asarray(dt64)).astype("datetime64[ns]")
    ns = dt64.astype("int64").astype(float)
    out = ns / 86400e9 + MATLAB_EPOCH
    out[np.isnat(dt64)] = np.nan
    return float(out[0]) if scalar else out


def campbell_date_to_serial_date(mat: np.ndarray) -> np.ndarray:
    """Convert Campbell date matrix to MATLAB serial date numbers.

    Parameters
    ----------
    mat : ndarray, shape (N, 4)
        Columns: [year, day_of_year, HHMM, seconds]
        HHMM is encoded as hour*100 + minute (e.g. 1430 = 14:30).

    Returns
    -------
    serial : ndarray, shape (N,)
        MATLAB serial date numbers (float).
    """
    mat = np.asarray(mat, dtype=float)
    years = mat[:, 0]
    doy   = mat[:, 1]
    hhmm  = mat[:, 2]
    secs  = mat[:, 3]

    hours   = np.floor(hhmm / 100.0)
    minutes = hhmm - hours * 100.0

    n = len(years)
    serial = np.empty(n)
    for i in range(n):
        dt = datetime(int(years[i]), 1, 1) + timedelta(
            days=int(doy[i]) - 1,
            hours=int(hours[i]),
            minutes=int(minutes[i]),
            seconds=round(float(secs[i]), 4),
        )
        serial[i] = datetime_to_matlab_datenum(dt)
    return serial


def serial_date_to_campbell_date(t_serial: np.ndarray) -> np.ndarray:
    """Convert MATLAB serial date array to Campbell date matrix.

    Returns ndarray with columns [year, day_of_year, HHMM, seconds].
    """
    t_serial = np.asarray(t_serial, dtype=float).ravel()
    out = np.empty((len(t_serial), 4))
    for i, md in enumerate(t_serial):
        dt = matlab_datenum_to_datetime(md)
        yy = dt.year
        doy = (dt - datetime(yy, 1, 1)).days + 1
        hhmm = dt.hour * 100 + dt.minute
        ss = dt.second + dt.microsecond / 1e6
        ss = round(ss * 100) / 100  # round to 2 decimals
        out[i] = [yy, doy, hhmm, ss]
    return out
