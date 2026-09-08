"""Shared helpers for the raw_processing FM_* scripts.

Collects the boilerplate that was duplicated across the standalone
FM_*_process.py scripts: Box path resolution, TOA5/daqm/CR file loading,
fast-data timestamp validation, the 48-h uniform grid, and the
year/day/HM/second output columns.
"""

import csv
import os
import platform

import numpy as np
import pandas as pd


def get_box_path():
    """Return the platform-specific Box sync root.

    Returns
    -------
    str
        Path to the Box-Box (macOS) or Box (Windows) sync folder.
    """
    if platform.system() == "Darwin":
        return os.path.expanduser("~/Library/CloudStorage/Box-Box")
    return os.path.expanduser("~/Box")  # Windows


def header_row(path, row):
    """Read one header line of a delimited file as a list of fields.

    Parameters
    ----------
    path : str
        Path to the file.
    row : int
        0-based line index (TOA5: 1 names, 2 units, 3 aggregation).

    Returns
    -------
    list of str
        Unquoted, stripped fields; empty list when the line is absent.
    """
    with open(path, "r", newline="") as fh:
        for i, fields in enumerate(csv.reader(fh)):
            if i == row:
                return [f.strip() for f in fields]
    return []


def read_toa5(path, with_units=False, units_row=2, **kwargs):
    """Read a Campbell TOA5 .dat file (4-line header, timestamp in col 0).

    Parameters
    ----------
    path : str
        Path to the TOA5 file.
    with_units : bool, optional
        Also return the units row, which the default ``skiprows`` drops
        (default False).
    units_row : int, optional
        0-based line index of the units row (default 2, the TOA5 slot).
    **kwargs
        Overrides for the default ``pd.read_csv`` options (e.g. ``nrows=1``
        or ``parse_dates=False`` for a header peek).

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, list of str)
        Data indexed by the TIMESTAMP column; with ``with_units`` also the
        units row, one entry per raw column including the timestamp.
    """
    opts = dict(skiprows=[0, 2, 3], index_col=[0],
                na_values=["NaN", "NAN"], parse_dates=True)
    opts.update(kwargs)
    df = pd.read_csv(path, **opts)
    if with_units:
        return df, header_row(path, units_row)
    return df


def load_daqm_files(paths):
    """Load and concatenate LICOR daqm 1-min statistics files.

    Parameters
    ----------
    paths : iterable of str
        Whitespace-delimited daqm .log files (DATE/TIME columns).

    Returns
    -------
    pandas.DataFrame
        Concatenated data indexed by TIMESTAMP.
    """
    dfs = []
    for f in paths:
        df = pd.read_csv(f, sep=r"\s+", engine="python", header=0, skiprows=[1])
        df["TIMESTAMP"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
        df.set_index("TIMESTAMP", inplace=True)
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def load_cr_files(paths, **read_kwargs):
    """Load, concatenate, and de-duplicate CR1000X TOA5 files.

    Parameters
    ----------
    paths : iterable of str
        Candidate .dat files; non-existent paths are skipped.
    **read_kwargs
        Passed to :func:`read_toa5` (e.g. ``skiprows=[]`` for logger
        exports with a single names row instead of the 4-line TOA5 header).

    Returns
    -------
    pandas.DataFrame
        De-duplicated data with a parsed DatetimeIndex.
    """
    dfs = [read_toa5(f, **read_kwargs) for f in paths if os.path.exists(f)]
    df = pd.concat(dfs, axis=0)
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index, format="mixed")
    return df


def validate_fast(df, expected_date, expected_hz, label):
    """Check date alignment, sampling rate, and coverage of fast data.

    Parameters
    ----------
    df : pandas.DataFrame
        Fast data with a DatetimeIndex.
    expected_date : datetime.datetime
        Expected date of the first row.
    expected_hz : int
        Expected sampling rate in Hz.
    label : str
        Identifier printed in SKIP messages.

    Returns
    -------
    bool
        True if all checks pass; otherwise prints a SKIP message and
        returns False so the caller can skip the period.
    """
    if df is None or len(df) < 2:
        print(f"  SKIP {label}: fast data is empty.")
        return False

    first_date = df.index[0].date()
    if first_date != expected_date.date():
        print(f"  SKIP {label}: first timestamp {first_date} ≠ expected {expected_date.date()}.")
        return False

    # Cast to ns before asi8 — pandas 3.x stores ms-resolution index as µs in asi8
    idx_ns      = df.index[:min(200, len(df))].astype('datetime64[ns]')
    intervals_s = np.diff(idx_ns.view('int64')) / 1e9
    dt_s        = float(np.median(intervals_s))
    if dt_s <= 0:
        print(f"  SKIP {label}: cannot determine sampling frequency (dt={dt_s:.4f} s).")
        return False
    hz_actual = round(1.0 / dt_s)
    if hz_actual != expected_hz:
        print(f"  SKIP {label}: measured {hz_actual} Hz ≠ expected {expected_hz} Hz "
              f"(median dt = {dt_s:.4f} s).")
        return False

    expected_rows = expected_hz * 48 * 3600
    coverage      = len(df) / expected_rows
    if coverage < 0.10:
        print(f"  SKIP {label}: only {coverage:.0%} coverage "
              f"({len(df):,} / {expected_rows:,} expected rows).")
        return False

    print(f"  Timestamps OK: starts {df.index[0]}, {hz_actual} Hz, "
          f"{len(df):,} rows ({coverage:.0%} of 48 h)")
    return True


def build_48h_index(start, hz, offset=True):
    """Build a left-closed 48-h DatetimeIndex at the given sample rate.

    Parameters
    ----------
    start : pandas.Timestamp
        Day-start (floored to midnight) of the 48-h window.
    hz : float
        Sample rate in Hz. Use ``1/60`` for a 1-min grid.
    offset : bool, optional
        If True (default) shift the first sample by ``1/hz`` s — Campbell
        loggers stamp the first record one interval after midnight. The
        CR3000 DS fast data (FM_DSfast) starts on midnight, so it uses False.

    Returns
    -------
    pandas.DatetimeIndex
        Uniform grid from ``start`` up to (excluding) ``start + 48 h``.
    """
    if offset:
        start = start + pd.Timedelta(seconds=1 / hz)
    freq = f"{1000 / hz:.0f}ms"
    return pd.date_range(start=start,
                         end=start + pd.Timedelta(hours=48),
                         freq=freq, inclusive="left")


def timestamp_columns(index, decimals=1):
    """Build the leading year/day/HM/second output columns from a time index.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps for each output row.
    decimals : int, optional
        Decimal places for the fractional-second string (default 1).

    Returns
    -------
    pandas.DataFrame
        Columns year (int), day (day-of-year int), HM (hhmm int), and
        second (fixed-decimal string), with a default RangeIndex.
    """
    ts = pd.DatetimeIndex(index)
    out = pd.DataFrame()
    out["year"]   = ts.year
    out["day"]    = ts.dayofyear
    out["HM"]     = ts.hour * 100 + ts.minute
    secs          = ts.second + ts.microsecond / 1e6
    out["second"] = [f"{s:.{decimals}f}" for s in secs]
    return out
