"""Generic, config-driven processing of Campbell TOA5 flux-tower data.

Generalizes the FM_*_process.py scripts for towers whose raw data are
TOA5 .dat files (fast sonic/IRGA tables and slow 1-min tables). A thin
per-site script (see VAC001_*_process.py) supplies a config dict;
``process_table`` handles the rest.

Pipeline per 48-h window:
  1. Select the TOA5 files overlapping the window (indexed by first
     record timestamp — no manual startid/endid bookkeeping)
  2. Load, concatenate, de-duplicate (common.load_cr_files)
  3. Fast tables: validate date / sample rate / coverage
  4. Reindex onto the uniform 48-h grid (common.build_48h_index)
  5. Slow tables: optionally interpolate across timestamp jitter
  6. Map raw logger columns to UTESpac names (cfg["columns"])
  7. Write <prefix>_<table>_<d1>000000_<d2>000000.txt (headerless)
     and <prefix>_<table>_header.dat into the site folder
"""

import glob
import os
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    from common import (load_cr_files, read_toa5, validate_fast,
                        build_48h_index, timestamp_columns)
except ImportError:  # allow running from repo root or elsewhere
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import (load_cr_files, read_toa5, validate_fast,
                        build_48h_index, timestamp_columns)


def index_toa5_files(pattern, read_kwargs=None):
    """Index TOA5 files by their first record timestamp.

    Parameters
    ----------
    pattern : str
        Glob pattern for candidate .dat files.
    read_kwargs : dict, optional
        Extra options for :func:`read_toa5` (header-variant files).

    Returns
    -------
    list of (pandas.Timestamp, str)
        Sorted by first timestamp; unreadable files are skipped with a
        warning.
    """
    entries = []
    for path in sorted(glob.glob(pattern)):
        try:
            peek = read_toa5(path, nrows=1, parse_dates=False,
                             **(read_kwargs or {}))
            ts = pd.to_datetime(peek.index[0], format="mixed")
        except Exception as exc:
            print(f"  Warning: cannot index {os.path.basename(path)}: {exc} — skipped.")
            continue
        entries.append((ts, path))
    entries.sort(key=lambda e: e[0])
    return entries


def files_for_window(entries, start, end):
    """Select files whose data may overlap [start, end].

    Parameters
    ----------
    entries : list of (pandas.Timestamp, str)
        Output of ``index_toa5_files``.
    start, end : pandas.Timestamp
        Window bounds.

    Returns
    -------
    list of str
        Files starting inside the window, plus the last file starting
        before it (its tail may extend into the window).
    """
    before, inside = None, []
    for ts, path in entries:
        if ts < start:
            before = path
        elif ts <= end:
            inside.append(path)
    return ([before] if before else []) + inside


# UTESpac base name → typical Campbell (EasyFlux-style) TOA5 base name.
# Used with level_columns(); override entries for loggers that name
# variables differently.
FAST_SONIC_MAP = {
    "Ux":         "Ux",
    "Uy":         "Uy",
    "Uz":         "Uz",
    "T_Sonic":    "Ts",
    "diagnostic": "diag_sonic",
}
FAST_IRGA_MAP = {
    "Pressure": "cell_press",
    "H2O":      "H2O",
    "H2Osig":   "H2O_sig_strgth",
    "CO2":      "CO2",
    "CO2sig":   "CO2_sig_strgth",
    "gas_diag": "diag_irga",
}


def level_columns(base_map, height, src_suffix=""):
    """Build the ``columns`` block for one tower level.

    Parameters
    ----------
    base_map : dict
        {UTESpac base name: TOA5 base name}, e.g. ``FAST_SONIC_MAP`` or
        ``{**FAST_SONIC_MAP, **FAST_IRGA_MAP}`` for a sonic + gas level.
    height : float
        Instrument height [m], appended to the output names.
    src_suffix : str, optional
        Suffix appended to the source names when the logger numbers
        replicated instruments (e.g. ``"_1"`` → ``Ux_1``).

    Returns
    -------
    dict
        ``{f"{base}_{height}": f"{src}{src_suffix}"}`` for each entry.
    """
    h = f"{height:g}"
    return {f"{out}_{h}": f"{src}{src_suffix}" for out, src in base_map.items()}


def write_header(out_dir, table_name, columns):
    """Write the UTESpac <table>_header.dat file for a column mapping.

    Parameters
    ----------
    out_dir : str
        Site folder.
    table_name : str
        Full table name (e.g. ``"VAC001_20Hz"``).
    columns : iterable of str
        Output column names, in file order.

    Returns
    -------
    str
        Path of the header file written.
    """
    path = os.path.join(out_dir, f"{table_name}_header.dat")
    with open(path, "w") as fh:
        fh.write(",".join(f'"{c}"' for c in ["TIMESTAMP", *columns]) + "\n")
    return path


def process_table(cfg):
    """Process one TOA5 table into 48-h UTESpac input files.

    Parameters
    ----------
    cfg : dict
        raw_pattern : str
            Glob for the table's raw TOA5 .dat files (READ-ONLY).
        out_dir : str
            Site folder receiving the outputs (created if missing).
        prefix : str
            Output filename prefix (e.g. ``"VAC001"``).
        table : str
            Table label (e.g. ``"20Hz"``, ``"1min"``).
        hz : float
            Sample rate [Hz]; use ``1/60`` for 1-min tables.
        columns : dict
            ``{output name: TOA5 source column}``; output names carry the
            instrument height suffix (e.g. ``"Ux_3"``).
        start_date, end_date : datetime.datetime
            Processing range, stepped in 48-h windows.
        decimals : int, optional
            Fractional-second digits (default from hz: 20 Hz → 2,
            fast → 1, slow → 0).
        offset : bool, optional
            First sample 1/hz after midnight (default True; Campbell
            loggers stamp records at interval end).
        interpolate_limit : int, optional
            Max consecutive samples to gap-fill on slow tables
            (default 0 = off).
        validate : bool, optional
            Run fast-data validation (default: hz >= 1).
        read_kwargs : dict, optional
            Extra ``read_toa5`` options applied when indexing and loading
            (e.g. ``{"skiprows": []}`` for files with a single names row
            instead of the 4-line TOA5 header).
        loader : callable, optional
            ``loader(paths) -> DataFrame`` with a DatetimeIndex, replacing
            the TOA5 reader for non-TOA5 sources (e.g. a logger CSV with a
            names row and a units row). When given, every matched file is
            loaded for every window (no first-timestamp indexing), so use
            it for small slow tables only.

    Returns
    -------
    list of str
        Paths of the data files written.
    """
    hz       = cfg["hz"]
    columns  = cfg["columns"]
    table    = f"{cfg['prefix']}_{cfg['table']}"
    decimals = cfg.get("decimals", 2 if hz >= 20 else 1 if hz >= 1 else 0)
    offset   = cfg.get("offset", True)
    interp   = cfg.get("interpolate_limit", 0)
    validate = cfg.get("validate", hz >= 1)
    loader   = cfg.get("loader")
    read_kw  = cfg.get("read_kwargs", {})

    os.makedirs(cfg["out_dir"], exist_ok=True)
    if loader is None:
        entries = index_toa5_files(cfg["raw_pattern"], read_kwargs=read_kw)
    else:
        entries = [(None, p) for p in sorted(glob.glob(cfg["raw_pattern"]))]
    if not entries:
        raise FileNotFoundError(f"No raw files match {cfg['raw_pattern']}")
    span = (f" ({entries[0][0]} … {entries[-1][0]})" if loader is None else "")
    print(f"{table}: {len(entries)} raw files{span}")

    hdr = write_header(cfg["out_dir"], table, columns)
    print(f"  → {os.path.basename(hdr)}")
    print(f"  siteInfo: tableNames += ['{table}'], "
          f"tableScanFrequency += [{hz:g}], "
          f"tableNumberOfColumns += [{4 + len(columns)}]")

    written, missing_warned = [], set()
    curr = cfg["start_date"]
    while curr <= cfg["end_date"] - timedelta(days=2):
        w0    = pd.Timestamp(curr)
        w1    = w0 + pd.Timedelta(hours=48)
        label = f"{table} {w0.date()}"

        if loader is None:
            paths = files_for_window(entries, w0, w1)
        else:
            paths = [p for _, p in entries]
        if not paths:
            print(f"  SKIP {label}: no raw files overlap the window.")
            curr += timedelta(days=2)
            continue

        raw = load_cr_files(paths, **read_kw) if loader is None else loader(paths)
        raw = raw[(raw.index >= w0) & (raw.index <= w1)]
        if validate:
            if not validate_fast(raw, curr, round(hz), label):
                curr += timedelta(days=2)
                continue
        elif raw.empty:
            print(f"  SKIP {label}: no records in window.")
            curr += timedelta(days=2)
            continue

        grid = build_48h_index(w0, hz=hz, offset=offset)
        full = raw.reindex(grid)
        if interp:
            full = (full.apply(pd.to_numeric, errors="coerce")
                        .interpolate(method="time", limit=interp))

        out = timestamp_columns(grid, decimals=decimals)
        for name, src in columns.items():
            if src in full.columns:
                out[name] = full[src].values
            else:
                if src not in missing_warned:
                    print(f"  Warning: source column '{src}' missing — "
                          f"'{name}' filled with NaN.")
                    missing_warned.add(src)
                out[name] = np.nan

        d1, d2   = w0.strftime("%Y%m%d"), w1.strftime("%Y%m%d")
        out_path = os.path.join(cfg["out_dir"],
                                f"{table}_{d1}000000_{d2}000000.txt")
        out.to_csv(out_path, header=False, index=False, sep=",")
        written.append(out_path)
        print(f"  → {os.path.basename(out_path)}")
        curr += timedelta(days=2)

    print(f"{table}: wrote {len(written)} file(s).")
    return written
