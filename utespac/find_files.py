"""findFiles – locate site folders, header files, and CSV/TXT data files."""

import os
import glob
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
from .import_header import import_header
from .campbell_date import datetime_to_matlab_datenum
from .site_config import (load_site_info, list_sites, resolve_site_dir,
                          site_input_dir)
from datetime import datetime


def find_files(
    info: Dict,
    site: str = None,
    dates=None,
) -> Tuple[List, List, List[str], Dict]:
    """Locate site data files and load headers.

    Parameters
    ----------
    info : dict
        Must contain ``info['rootFolder']``.
    site : str
        Site folder name (``'VAC001'``) or bare legacy id; required.
    dates : None, int, list, or ``'all'``
        Which date rows to process (1-based indices).  ``None`` / ``'all'``
        uses all available dates.  An integer selects that single date.  A
        list selects those rows. Interactive selection lives in the CLI.

    Returns
    -------
    headers_cell : list of [names, heights]
    data_files   : list of lists, shape [n_dates × n_tables], full paths
    table_names  : list of str
    info         : dict (updated with siteFolder and date)
    """
    root = info["rootFolder"]

    # ── site selection ──────────────────────────────────────────────────────
    available = list_sites(root)
    if not available:
        raise FileNotFoundError(f"No site directories found in {root}")

    if site is None:
        raise ValueError("find_files needs a site name; the CLI (utespac_main) "
                         f"lists the choices: {', '.join(available)}")
    site_dir = resolve_site_dir(root, site)

    info["siteFolder"] = site_dir
    site_path = os.path.join(root, site_dir)

    # ── load site configuration ─────────────────────────────────────────────
    try:
        load_site_info(site_path).apply_to(info)
    except FileNotFoundError:
        warnings.warn(f"No siteInfo.toml or siteInfo.py in {site_path}; "
                      "using defaults from info dict.")

    # ── find header files ───────────────────────────────────────────────────
    input_path = str(site_input_dir(site_path))
    header_files = sorted(glob.glob(os.path.join(input_path, "*header*")))
    if not header_files:
        raise FileNotFoundError(f"No header files found in {input_path}")

    headers_cell: List = []
    table_names:  List[str] = []
    all_csv_by_table: List[List[Tuple[float, str]]] = []

    for hf in header_files:
        hdr = import_header(hf)
        headers_cell.append(hdr)

        base = os.path.basename(hf)
        tname_end = base.lower().find("_header")
        table_name = base[:tname_end] if tname_end >= 0 else base.split(".")[0]
        table_names.append(table_name)

        # CSV or TXT files for this table (exclude header files)
        csv_glob = os.path.join(input_path, f"*{table_name}*")
        csv_files_raw = [
            f for f in sorted(glob.glob(csv_glob))
            if "header" not in os.path.basename(f).lower()
            and os.path.splitext(f)[1].lower() in (".csv", ".txt", ".dat", "")
        ]

        dated: List[Tuple[float, str]] = []
        for fp in csv_files_raw:
            fname = os.path.basename(fp)
            loc   = fname.find(table_name) + len(table_name)
            try:
                yr = int(fname[loc + 1: loc + 5])
                mo = int(fname[loc + 5: loc + 7])
                dy = int(fname[loc + 7: loc + 9])
                hr = int(fname[loc + 9: loc + 11])
                mn = int(fname[loc + 11: loc + 13])
                sc = int(fname[loc + 13: loc + 15])
                dn = datetime_to_matlab_datenum(datetime(yr, mo, dy, hr, mn, sc))
            except (ValueError, IndexError):
                dn = 0.0
            dated.append((dn, fp))
        all_csv_by_table.append(dated)

    if not all_csv_by_table or not all_csv_by_table[0]:
        raise FileNotFoundError(f"No CSV/TXT data files found in {input_path}")

    # ── build date-aligned file matrix ─────────────────────────────────────
    all_dates = sorted({d for tbl in all_csv_by_table for d, _ in tbl})
    date_begin = int(np.floor(min(all_dates))) - 20
    date_end   = int(np.floor(max(all_dates))) + 20
    n_days     = date_end - date_begin
    n_tables   = len(table_names)
    data_files: List[List[Optional[str]]] = [[None] * n_tables for _ in range(n_days)]

    for t_idx, dated in enumerate(all_csv_by_table):
        for dn, fp in dated:
            row = int(np.floor(dn)) - date_begin
            if 0 <= row < n_days:
                data_files[row][t_idx] = fp

    # Remove rows where all tables are None
    data_files = [row for row in data_files if any(f is not None for f in row)]

    # ── date selection ──────────────────────────────────────────────────────
    if dates is None or dates == "all":
        # non-interactive: use all dates
        pass
    elif isinstance(dates, int):
        data_files = [data_files[dates - 1]]
    elif isinstance(dates, (list, tuple)):
        data_files = [data_files[i - 1] for i in dates]

    return headers_cell, data_files, table_names, info
