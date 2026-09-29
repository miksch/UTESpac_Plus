"""Parser for a WitMotion IMU logger's plain-text ``DATALOG.TXT``.

The logger (WitMotion WTGAHRS2, JY901B core) writes four lines per sample in
the order ``A`` (accelerometer, g), ``G`` (gyro, deg/s), ``E`` (Euler angles,
deg), ``D`` (date/time stamp, closing the record)::

    A: <ax> <ay> <az>
    G: <gx> <gy> <gz>
    E: <roll> <pitch> <yaw>
    D: <YYYY-M-D> T: <HH:MM:SS.ms>

The millisecond field of the stamp is an unpadded integer, so ``18:11:25.35``
is 25.035 s, not 25.350 s.

Records written before the RTC was set carry the date ``200-0-0`` and are
dropped.

This tagged, one-channel-group-per-line layout is not delimited text with a
names row, so :func:`utespac.raw_processing.imu.load_imu_files` does not read
it; :func:`parse_datalog` returns lower-case channel names (:data:`CHANNELS`)
that the screening helpers in :mod:`utespac.raw_processing.imu` expect.
"""

from __future__ import annotations

import array
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "yaw")

_EPOCH = dt.date(1970, 1, 1)


def _date_to_days(token: str, cache: dict) -> int | None:
    """Days since 1970-01-01 for a ``YYYY-M-D`` token, or None if unparseable."""
    days = cache.get(token, False)
    if days is not False:
        return days
    try:
        year, month, day = (int(p) for p in token.split("-"))
        days = (dt.date(year, month, day) - _EPOCH).days
    except (ValueError, TypeError):
        days = None
    cache[token] = days
    return days


def parse_datalog(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Parse one ``DATALOG.TXT`` into a dataframe plus parse statistics.

    Parameters
    ----------
    path : path-like
        Raw IMU log file.

    Returns
    -------
    df : pandas.DataFrame
        Columns ``time`` (datetime64[ns], raw logger clock) and `CHANNELS`,
        sorted by time.
    stats : dict
        Line and record counts, including records dropped for a pre-RTC
        (``200-0-0``) stamp, incomplete four-line blocks and unparseable
        numeric fields.
    """
    path = Path(path)
    cols = {name: array.array("d") for name in CHANNELS}
    times = array.array("d")

    stats = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": 0,
        "unknown_tag_lines": 0,
        "records_ok": 0,
        "records_pre_rtc": 0,
        "records_incomplete": 0,
        "records_bad_field": 0,
    }

    pend_a = pend_g = pend_e = None
    date_cache: dict = {}

    with open(path, "rb") as handle:
        for raw in handle:
            stats["lines"] += 1
            tag = raw[:1]
            if tag == b"D":
                if pend_a is None or pend_g is None or pend_e is None:
                    stats["records_incomplete"] += 1
                    pend_a = pend_g = pend_e = None
                    continue
                stamp = _stamp(raw, date_cache, stats)
                if stamp is not None:
                    times.append(stamp)
                    for name, value in zip(CHANNELS, pend_a + pend_g + pend_e):
                        cols[name].append(value)
                    stats["records_ok"] += 1
                pend_a = pend_g = pend_e = None
            elif tag == b"A":
                pend_a = _triple(raw)
                if pend_a is None:
                    stats["records_bad_field"] += 1
            elif tag == b"G":
                pend_g = _triple(raw)
                if pend_g is None:
                    stats["records_bad_field"] += 1
            elif tag == b"E":
                pend_e = _triple(raw)
                if pend_e is None:
                    stats["records_bad_field"] += 1
            else:
                stats["unknown_tag_lines"] += 1

    stats["bad_date_tokens"] = sorted(tok for tok, days in date_cache.items() if days is None)

    df = pd.DataFrame({name: np.frombuffer(cols[name], dtype="f8") for name in CHANNELS})
    stamp_us = np.rint(np.frombuffer(times, dtype="f8") * 1e6).astype("int64")
    df.insert(0, "time", pd.to_datetime(stamp_us, unit="us"))
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    return df, stats


def _stamp(raw: bytes, date_cache: dict, stats: dict):
    """Epoch seconds from a ``D: <date> T: <time>`` line, or None if rejected.

    Rejections are tallied in `stats` as pre-RTC or malformed records.
    """
    fields = raw.split()
    if len(fields) != 4 or fields[2] != b"T:":
        stats["records_bad_field"] += 1
        return None
    days = _date_to_days(fields[1].decode("ascii", "replace"), date_cache)
    if days is None:
        stats["records_pre_rtc"] += 1
        return None
    clock = fields[3].split(b":")
    if len(clock) != 3:
        stats["records_bad_field"] += 1
        return None
    sec_field = clock[2].split(b".")
    try:
        return (
            days * 86400.0
            + int(clock[0]) * 3600.0
            + int(clock[1]) * 60.0
            + int(sec_field[0])
            + (int(sec_field[1]) / 1000.0 if len(sec_field) > 1 else 0.0)
        )
    except ValueError:
        stats["records_bad_field"] += 1
        return None


def _triple(raw: bytes):
    """Three floats from a ``X: a b c`` line, or None if the line is malformed."""
    fields = raw.split()
    if len(fields) != 4:
        return None
    try:
        return (float(fields[1]), float(fields[2]), float(fields[3]))
    except ValueError:
        return None
