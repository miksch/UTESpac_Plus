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
    matlab_compat: bool = False,
) -> Dict:
    """Load and vertically concatenate processed UTESpac output files.

    Fields that carry a column header (``<field>Header``/``<field>header``)
    are concatenated by header label: the output columns are the union of
    the labels seen across files, and a file missing a label contributes NaN
    in that column only. This keeps a sensor that is dead for one file
    (``fluxes`` trims its all-NaN columns) from blanking the whole block.

    Reproduces MATLAB getData.m defensive behaviours:

    * **Row-count consistency check** (MATLAB lines 112-136): if the averaged
      data tables in one file have different row counts from each other, that
      file is skipped with a warning.
    * **Column-count mismatch → NaN fill** (MATLAB lines 161-171): if a
      numeric field in a file has a different shape from what has already been
      accumulated, a NaN block of the expected size is inserted so that all
      fields remain row-aligned after concatenation. Applied to unlabeled
      fields always, and to every field when ``matlab_compat`` is True.
    * **Late-initialised field warning** (MATLAB lines 147-149): if a field
      appears for the first time in a file that is not the first one loaded, a
      warning is emitted.
    * Headers, ``tableNames``, and scalar/cell metadata are never concatenated
      (MATLAB lines 150-153).

    Parameters
    ----------
    root_folder : str
        Root directory containing ``site*`` sub-directories.
    site : str or int
        Site folder name, bare legacy id, or 1-based index into the sorted
        site list; required.
    avg_per : int or None
        Averaging period in minutes (used to filter file names).
    qualifier : str or None
        Substring that must appear in output file names (e.g. ``'LPF'``).
    rows : int, list, or 0
        0 → all rows; integer → first N rows; list → specific row indices.
    matlab_compat : bool
        Use the MATLAB shape check for labeled fields too (whole-file NaN
        block on any column-count change) instead of aligning by label.

    Returns
    -------
    output_struct : dict
        Concatenated structure with the same keys as individual output files.
    """
    sites = list_sites(root_folder)

    if site is None:
        raise ValueError("get_data needs a site name or 1-based index; "
                         f"available: {', '.join(sites)}")
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

    def _header_key(d: Dict, key: str) -> Optional[str]:
        """Header key paired with numeric field *key* in *d*, if any."""
        for cand in (f"{key}Header", f"{key}header"):
            if cand in d:
                return cand
        return None

    def _labels(hdr) -> Optional[list]:
        """Flat list of column labels, or None if *hdr* is not one."""
        if isinstance(hdr, list) and hdr and isinstance(hdr[0], list):
            hdr = hdr[0]
        if isinstance(hdr, list) and all(isinstance(h, str) for h in hdr):
            return list(hdr)
        return None

    def _align(mat: np.ndarray, labels: list, target: list) -> np.ndarray:
        """Columns of *mat* (labeled *labels*) re-indexed to *target*; NaN where absent."""
        out = np.full((mat.shape[0], len(target)), np.nan)
        pos = {lab: i for i, lab in enumerate(labels)}
        for j, lab in enumerate(target):
            if lab in pos:
                out[:, j] = mat[:, pos[lab]]
        return out

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
                hk = None if matlab_compat else _header_key(d, key)
                acc_labels = _labels(output_struct.get(hk)) if hk else None
                new_labels = _labels(d.get(hk)) if hk else None
                labeled = (
                    acc_labels is not None and new_labels is not None
                    and len(acc_labels) == existing.shape[1]
                    and len(new_labels) == val.shape[1]
                )
                if val.shape[0] != exp_rows:
                    warnings.warn(
                        f"{fpath!r} field {key!r}: expected {exp_rows} rows, "
                        f"got {val.shape[0]} — inserting NaN block."
                    )
                    val = np.full((exp_rows, existing.shape[1]), np.nan)
                    new_labels = acc_labels
                if labeled and new_labels != acc_labels:
                    # Align by header label (union, accumulated order first).
                    union = acc_labels + [h for h in new_labels if h not in acc_labels]
                    if len(union) != existing.shape[1]:
                        existing = _align(existing, acc_labels, union)
                    val = _align(val, new_labels, union)
                    output_struct[hk] = union
                elif val.shape[1] != existing.shape[1]:
                    warnings.warn(
                        f"{fpath!r} field {key!r}: expected [{exp_rows}, {existing.shape[1]}], "
                        f"got {list(val.shape)} — inserting NaN block."
                    )
                    val = np.full((exp_rows, existing.shape[1]), np.nan)
                output_struct[key] = np.vstack([existing, val])

            # ndim mismatches (e.g. 1-D in first file, 2-D later): skip silently

        files_loaded += 1

    return output_struct
