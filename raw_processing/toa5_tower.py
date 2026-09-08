"""Generic, config-driven processing of Campbell TOA5 flux-tower data.

Generalizes the FM_*_process.py scripts for towers whose raw data are
TOA5 .dat files (fast sonic/IRGA tables and slow 1-min tables). A thin
per-site script (under ``data/<SITE>/scripts/`` in the development tree)
supplies a config dict;
``process_table`` handles the rest.

Pipeline per 48-h window:
  1. Select the TOA5 files overlapping the window (indexed by first
     record timestamp — no manual startid/endid bookkeeping)
  2. Load, concatenate, de-duplicate (common.load_cr_files)
  3. Fast tables: validate date / sample rate / coverage
  4. Reindex onto the uniform 48-h grid (common.build_48h_index)
  5. Slow tables: optionally interpolate across timestamp jitter
  6. Map raw logger columns to UTESpac names (cfg["columns"])
  7. Write <prefix>_<table>_<d1>000000_<d2>000000.txt (headerless),
     <prefix>_<table>_header.dat and <prefix>_<table>_units.dat into the
     site folder
"""

import glob
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    from common import (header_row, load_cr_files, read_toa5, validate_fast,
                        build_48h_index, timestamp_columns)
except ImportError:  # allow running from repo root or elsewhere
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import (header_row, load_cr_files, read_toa5, validate_fast,
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
        Full table name (e.g. ``"MySite_20Hz"``).
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


def header_rows(cfg):
    """Line indices of the names and units rows of a table's raw files.

    Parameters
    ----------
    cfg : dict
        Table config; ``header_row`` and ``units_row`` override the
        defaults, ``units_row = None`` marks a source with no units row.

    Returns
    -------
    (int, int or None)
        0-based line indices. Defaults: TOA5 (1, 2); a ``loader`` source
        (a pandas export with a names row and a units row) (0, 1); any
        other ``read_kwargs`` variant (0, None).
    """
    if cfg.get("loader") is not None:
        default = (0, 1)
    elif cfg.get("read_kwargs", {}).get("skiprows", [0, 2, 3]) == [0, 2, 3]:
        default = (1, 2)
    else:
        default = (0, None)
    return (cfg.get("header_row", default[0]),
            cfg["units_row"] if "units_row" in cfg else default[1])


def clean_unit(unit):
    """Normalize one declared unit to a plain string.

    ``"Unnamed: 30_level_1"`` is what a pandas export writes for an empty
    units cell; it and a bare TOA5 blank both mean "no unit declared".
    """
    unit = "" if unit is None else str(unit).strip().strip('"').strip("'").strip()
    if unit.startswith("Unnamed:") or unit.lower() == "nan":
        return ""
    return unit


def source_units(path, header_row_i=1, units_row_i=2):
    """Read ``{source column: unit}`` from a raw file's header lines.

    Parameters
    ----------
    path : str
        Raw file.
    header_row_i : int
        0-based line index of the column-names row.
    units_row_i : int or None
        0-based line index of the units row; None when the source has none.

    Returns
    -------
    dict
        Empty when there is no units row.
    """
    if units_row_i is None:
        return {}
    names = header_row(path, header_row_i)
    units = header_row(path, units_row_i)
    return {n: clean_unit(u) for n, u in zip(names, units)}


def resolve_units(cfg, path=None):
    """Units of a table's output columns, in ``cfg["columns"]`` order.

    The raw file's units row supplies the defaults; ``cfg["raw_units"]``
    ({source column: unit}) declares them where the source carries none
    (card-converted files with a names row only) and overrides them
    otherwise.

    Returns
    -------
    dict
        ``{output name: unit}``; "" where no unit is known.
    """
    h_i, u_i = header_rows(cfg)
    raw = source_units(path, h_i, u_i) if path is not None else {}
    raw.update({k: clean_unit(v) for k, v in cfg.get("raw_units", {}).items()})
    return {out: raw.get(src, "") for out, src in cfg["columns"].items()}


def write_units(out_dir, table_name, columns, units):
    """Write the <table>_units.dat file beside the header.

    Parameters
    ----------
    out_dir : str
        Site folder.
    table_name : str
        Full table name (e.g. ``"MySite_20Hz"``).
    columns : iterable of str
        Output column names, in file order (as ``write_header``).
    units : dict
        ``{output name: unit}``; missing entries are written empty. The
        synthesized TIMESTAMP column is always empty.

    Returns
    -------
    str
        Path of the units file written.
    """
    path = os.path.join(out_dir, f"{table_name}_units.dat")
    fields = [""] + [clean_unit(units.get(c, "")) for c in columns]
    with open(path, "w") as fh:
        fh.write(",".join(f'"{u}"' for u in fields) + "\n")
    return path


def write_table_units(cfg):
    """Write a table's header and units files without reading any data rows.

    The units come from the first matching raw file's units row plus
    ``cfg["raw_units"]``; use it to (re)generate ``<table>_units.dat`` for
    tables that were processed before the units file existed.

    Returns
    -------
    str
        Path of the units file written.
    """
    table = f"{cfg['prefix']}_{cfg['table']}"
    paths = sorted(glob.glob(cfg["raw_pattern"]))
    if not paths:
        raise FileNotFoundError(f"No raw files match {cfg['raw_pattern']}")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    units = resolve_units(cfg, paths[0])
    hdr = write_header(cfg["out_dir"], table, cfg["columns"])
    path = write_units(cfg["out_dir"], table, cfg["columns"], units)
    print(f"{table}: units from {os.path.basename(paths[0])}")
    print(f"  → {os.path.basename(hdr)}")
    print(f"  → {os.path.basename(path)}")
    missing = [c for c, u in units.items() if not u]
    if missing:
        print(f"  no unit declared for: {', '.join(missing)}")
    return path


def run_table(cfg, argv=None):
    """Entry point of a per-site table script.

    ``--units-only`` on the command line writes the header and units files
    from the raw header rows alone (no data is read or rewritten);
    otherwise the table is processed in full.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--units-only" in argv:
        return write_table_units(cfg)
    return process_table(cfg)


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
            Output filename prefix (e.g. ``"MySite"``).
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
        raw_units : dict, optional
            ``{TOA5 source column: unit}`` declaring the units of a source
            with no units row (card-converted files with a names row
            only); overrides the file's own units row where both exist.
        header_row, units_row : int or None, optional
            0-based line indices of the names and units rows of the raw
            files; see :func:`header_rows` for the defaults.
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
    uni = write_units(cfg["out_dir"], table, columns, resolve_units(cfg, entries[0][1]))
    print(f"  → {os.path.basename(uni)}")
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
