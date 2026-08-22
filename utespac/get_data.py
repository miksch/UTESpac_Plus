"""getData – load and concatenate processed output files from a site folder."""

import os
import glob
import pickle
import warnings
from typing import Dict, Optional
import numpy as np
from .site_config import list_sites, resolve_site_dir


def get_data(
    root_folder: str,
    site=None,
    avg_per: int = None,
    qualifier: str = None,
    rows=None,
) -> Dict:
    """Load and vertically concatenate processed UTESpac output files.

    Reproduces MATLAB getData.m defensive behaviours:

    * **Row-count consistency check** (MATLAB lines 112-136): if the averaged
      data tables in one file have different row counts from each other, that
      file is skipped with a warning.
    * **Column-count mismatch → NaN fill** (MATLAB lines 161-171): if a
      numeric field in a file has a different shape from what has already been
      accumulated, a NaN block of the expected size is inserted so that all
      fields remain row-aligned after concatenation.
    * **Late-initialised field warning** (MATLAB lines 147-149): if a field
      appears for the first time in a file that is not the first one loaded, a
      warning is emitted.
    * Headers, ``tableNames``, and scalar/cell metadata are never concatenated
      (MATLAB lines 150-153).

    Parameters
    ----------
    root_folder : str
        Root directory containing ``site*`` sub-directories.
    site : str or int or None
        Site name (without ``'site'`` prefix) or 1-based integer index.
        If None the user is prompted to select.
    avg_per : int or None
        Averaging period in minutes (used to filter file names).
    qualifier : str or None
        Substring that must appear in output file names (e.g. ``'LPF'``).
    rows : int, list, or 0
        0 → all rows; integer → first N rows; list → specific row indices.

    Returns
    -------
    output_struct : dict
        Concatenated structure with the same keys as individual output files.
    """
    sites = list_sites(root_folder)

    if site is None:
        for i, s in enumerate(sites):
            print(f"  {i + 1}. {s}")
        choice = int(input("Please indicate site of interest: ")) - 1
        site_dir = sites[choice]
    elif isinstance(site, int):
        site_dir = sites[site - 1]
    else:
        site_dir = resolve_site_dir(root_folder, site)

    site_path = os.path.join(root_folder, site_dir, "output")
    if not os.path.isdir(site_path):
        raise FileNotFoundError(f"Output folder not found: {site_path}")

    # Build glob pattern matching MATLAB: *_{avgPer}*{qualifier}*.mat → *.pkl
    parts = ["*"]
    if avg_per is not None:
        parts.append(f"_{avg_per}")
    parts.append("*")
    if qualifier is not None:
        parts.append(f"{qualifier}*")
    parts.append(".pkl")
    pattern = os.path.join(site_path, "".join(parts))

    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f"No output files found matching {pattern}")

    # Select files by rows argument (mirrors MATLAB rows logic)
    if rows is None or rows == 0:
        selected_files = all_files
    elif isinstance(rows, int):
        selected_files = all_files[:rows]
    else:
        selected_files = [all_files[r - 1] for r in rows if 0 < r <= len(all_files)]

    # Fields that are never vertically concatenated (MATLAB lines 150-153)
    _NO_CONCAT = {"tableNames", "z", "warnings", "dataInfo", "infoString"}

    def _is_header(key: str) -> bool:
        """True for any key that ends in 'Header' or 'header'."""
        return key.endswith(("Header", "header"))

    output_struct: Dict = {}
    files_loaded = 0

    for fpath in selected_files:
        try:
            with open(fpath, "rb") as fh:
                d = pickle.load(fh)
        except Exception as exc:
            warnings.warn(f"Problem loading {fpath!r}: {exc}")
            continue

        # ── row-count consistency check (MATLAB lines 112-136) ──────────────
        # Averaged files have 'tableNames'; raw files have 't'.
        n_expected: Optional[int] = None
        if "tableNames" in d:
            row_counts = []
            for tname in d["tableNames"]:
                arr = d.get(tname)
                if arr is not None and isinstance(arr, np.ndarray) and arr.ndim >= 1:
                    row_counts.append(arr.shape[0])
            if len(set(row_counts)) > 1:
                warnings.warn(
                    f"Row count inconsistent across tables in {fpath!r} "
                    f"({row_counts}) — skipping."
                )
                continue
            n_expected = row_counts[0] if row_counts else None
        elif "t" in d:
            t_arr = d["t"]
            n_expected = int(np.asarray(t_arr).shape[0])

        # ── field-by-field concatenation ─────────────────────────────────────
        for key, val in d.items():
            # Never concatenate headers or protected metadata keys
            if _is_header(key) or key in _NO_CONCAT:
                if key not in output_struct:
                    output_struct[key] = val   # keep first occurrence
                continue

            if key not in output_struct:
                # Late-initialised field warning (MATLAB lines 147-149)
                if files_loaded > 0:
                    warnings.warn(
                        f"Field {key!r} first seen in {fpath!r} (file #{files_loaded + 1}); "
                        "row counts may be inconsistent across fields."
                    )
                output_struct[key] = val
                continue

            # Concatenate numeric ndarray fields
            existing = output_struct[key]
            if not (isinstance(val, np.ndarray) and isinstance(existing, np.ndarray)):
                continue   # scalars / lists: keep first value, same as headers

            # ── column-count / row-count mismatch → NaN fill (MATLAB 161-171) ─
            if existing.ndim == 1 and val.ndim == 1:
                # 1-D arrays (e.g. raw 't', 'z')
                exp_rows = n_expected if n_expected is not None else val.shape[0]
                if val.shape[0] != exp_rows:
                    warnings.warn(
                        f"{fpath!r} field {key!r}: expected {exp_rows} rows, "
                        f"got {val.shape[0]} — inserting NaN row."
                    )
                    val = np.full(exp_rows, np.nan)
                output_struct[key] = np.concatenate([existing, val])

            elif existing.ndim == 2 and val.ndim == 2:
                exp_rows = n_expected if n_expected is not None else val.shape[0]
                exp_cols = existing.shape[1]
                if val.shape[0] != exp_rows or val.shape[1] != exp_cols:
                    warnings.warn(
                        f"{fpath!r} field {key!r}: expected [{exp_rows}, {exp_cols}], "
                        f"got {list(val.shape)} — inserting NaN block."
                    )
                    val = np.full((exp_rows, exp_cols), np.nan)
                output_struct[key] = np.vstack([existing, val])

            # ndim mismatches (e.g. 1-D in first file, 2-D later): skip silently

        files_loaded += 1

    return output_struct
