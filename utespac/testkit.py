"""Comparison engine for UTESpac Python vs MATLAB outputs.

Importable core used by both the compare_outputs.py CLI report and the
pytest regression suite. Pure functions only — no printing, no sys.exit,
no stdout redirection.
"""

import os
import pickle

import numpy as np
import scipy.io as sio

# Fields to skip (non-numeric metadata)
SKIP_FIELDS = {"tableNames", "warnings", "dataInfo"}

# Fields whose timestamp column (col 0) should be skipped
HAS_TIMESTAMP_COL0 = {
    "spdAndDir", "rotatedSonic", "PFSonic",
    "H", "Hlat", "tau", "tke", "sigma", "R", "L",
    "eta", "delta_flux_ctrb", "delta_time_ctrb", "turbtr", "epsilon",
    "skew", "H_SNSP", "Flux_lat", "LHflux", "CO2flux", "derivedT",
}

# Raw 20 Hz pkl fields compared against same-named MATLAB rawFlux fields
RAW_FIELDS = [
    "uPF", "vPF", "wPF",
    "u_tilt", "v_tilt", "w_tilt",
    "sonTs", "Theta_v_son", "P",
    "rhov", "rhovPrime", "rhoCO2", "rhoCO2Prime",
]

# For 20 Hz raw, use every Nth sample to keep memory manageable
RAW_STRIDE = 100

STAT_KEYS_ALL = [
    "N_total", "N_valid", "N_nan_mismatch",
    "bias", "RMSE", "max_abs", "mean_abs", "p95_abs", "p99_abs",
    "max_rel", "mean_rel", "R2",
]


# ── file discovery ───────────────────────────────────────────────────────────

def discover_pairs(matlab_dir, py_root, site_filter=None, lpf_only=False,
                   gpf_only=False, avg_only=False, raw_only=False):
    """Return list of dicts describing matched (pkl, mat) file pairs.

    .mat files are looked up in  matlab_dir/site/output/.
    .pkl files are looked up in  matlab_dir/site/output/  first, then in
    py_root/site/output/ (where the Python pipeline writes when rootFolder
    points at UTESpac_Python/ rather than UTESpac_MATLAB/).

    Each dict has keys: site, pf_mode, output_type, stem, pkl_path, mat_path.
    """
    pairs = []

    # Collect candidate site names from both search roots
    candidate_sites = set()
    for search_root in (matlab_dir, py_root):
        try:
            candidate_sites.update(
                d for d in os.listdir(search_root)
                if d.startswith("site") and os.path.isdir(os.path.join(search_root, d))
            )
        except FileNotFoundError:
            pass

    for site in sorted(candidate_sites):
        if site_filter and site != site_filter:
            continue

        mat_dir = os.path.join(matlab_dir, site, "output")
        if not os.path.isdir(mat_dir):
            continue

        # .mat files always come from the MATLAB output folder
        mat_files = os.listdir(mat_dir)
        mat_stems = {os.path.splitext(f)[0]: f for f in mat_files if f.endswith(".mat")}

        # .pkl files: prefer matlab_dir copy, fall back to py_root copy
        pkl_stems: dict = {}
        for pkl_root in (matlab_dir, py_root):
            pkl_dir = os.path.join(pkl_root, site, "output")
            if not os.path.isdir(pkl_dir):
                continue
            for f in os.listdir(pkl_dir):
                if f.endswith(".pkl"):
                    stem = os.path.splitext(f)[0]
                    if stem not in pkl_stems:   # matlab_dir takes priority
                        pkl_stems[stem] = os.path.join(pkl_dir, f)

        for stem in sorted(set(pkl_stems) & set(mat_stems)):
            # Parse stem: e.g. Fire1_30minAvg_LPF_LinDet_2025_06_08
            #                or Fire1_raw_GPF_LinDet_2025_06_08
            parts = stem.split("_")
            pf_mode = next((p for p in parts if p in ("LPF", "GPF")), "unknown")
            output_type = "raw" if "raw" in parts else "avg"

            if lpf_only and pf_mode != "LPF":
                continue
            if gpf_only and pf_mode != "GPF":
                continue
            if avg_only and output_type != "avg":
                continue
            if raw_only and output_type != "raw":
                continue

            pairs.append({
                "site":        site,
                "pf_mode":     pf_mode,
                "output_type": output_type,
                "stem":        stem,
                "pkl_path":    pkl_stems[stem],
                "mat_path":    os.path.join(mat_dir, mat_stems[stem]),
            })
    return pairs


# ── loaders ──────────────────────────────────────────────────────────────────

def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_mat(path, key):
    """Load a MATLAB struct (e.g. 'output' or 'rawFlux') → {field: ndarray}."""
    mat = sio.loadmat(path, squeeze_me=True)
    ref = mat[key]
    result = {}
    for name, val in zip(ref.dtype.names, ref.item()):
        try:
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            result[name] = arr
        except (TypeError, ValueError):
            pass
    return result


def load_mat_avg(path):
    """Load averaged MATLAB output struct → {field: ndarray}."""
    return load_mat(path, "output")


def load_mat_raw(path):
    """Load raw MATLAB output struct → {field: ndarray}."""
    return load_mat(path, "rawFlux")


# ── header lookup ────────────────────────────────────────────────────────────

def get_header(pkl, field):
    """Return a flat column-label list for a field from the pkl dict.

    Handles both flat lists ['a','b',...] and nested [['a','b',...]] storage.
    """
    for suffix in (f"{field}Header", f"{field}header", f"{field}Header".lower()):
        if suffix in pkl:
            h = pkl[suffix]
            # unwrap one level of nesting when stored as [['col1', 'col2', ...]]
            if h and isinstance(h[0], list):
                h = h[0]
            return h
    return None


# ── per-column statistics ─────────────────────────────────────────────────────

def empty_stats(status, n_total=float("nan")):
    """Stat row with all-NaN metrics, for SKIP/MISS_PY/MISS_REF statuses."""
    row = {k: float("nan") for k in STAT_KEYS_ALL}
    row["N_total"] = n_total
    row["status"] = status
    return row


def column_stats(py_col, ref_col, tol_rel=0.01, tol_abs=1e-9):
    """Compute comprehensive statistics comparing two 1-D arrays.

    Returns a dict; all float values are NaN when no valid pairs exist.
    """
    py  = np.asarray(py_col,  dtype=float).ravel()
    ref = np.asarray(ref_col, dtype=float).ravel()
    n   = min(len(py), len(ref))
    py, ref = py[:n], ref[:n]

    py_nan  = np.isnan(py)
    ref_nan = np.isnan(ref)
    nan_mismatch = int(np.sum(py_nan != ref_nan))
    mask = ~(py_nan | ref_nan)
    n_valid = int(mask.sum())

    null = float("nan")
    if n_valid == 0:
        row = empty_stats("SKIP", n_total=n)
        row["N_valid"] = 0
        row["N_nan_mismatch"] = nan_mismatch
        return row

    diff  = py[mask] - ref[mask]
    adiff = np.abs(diff)
    ref_m = ref[mask]
    denom = np.abs(ref_m).copy()
    denom[denom < tol_abs] = tol_abs
    rel   = adiff / denom

    bias     = float(diff.mean())
    rmse     = float(np.sqrt((diff**2).mean()))
    max_abs  = float(adiff.max())
    mean_abs = float(adiff.mean())
    p95_abs  = float(np.percentile(adiff, 95))
    p99_abs  = float(np.percentile(adiff, 99))
    max_rel  = float(rel.max())
    mean_rel = float(rel.mean())

    # R² (correlation coefficient squared) — meaningful only when ref has variance
    ref_std = float(ref_m.std())
    if ref_std > 1e-14 and n_valid > 2:
        r = float(np.corrcoef(py[mask], ref_m)[0, 1])
        r2 = r**2 if np.isfinite(r) else null
    else:
        r2 = null

    if max_rel < tol_rel:
        status = "OK  "
    elif max_rel < 0.10:
        status = "WARN"
    else:
        status = "BIG "

    return {
        "N_total": n, "N_valid": n_valid, "N_nan_mismatch": nan_mismatch,
        "bias": bias, "RMSE": rmse,
        "max_abs": max_abs, "mean_abs": mean_abs,
        "p95_abs": p95_abs, "p99_abs": p99_abs,
        "max_rel": max_rel, "mean_rel": mean_rel,
        "R2": r2,
        "status": status,
    }


# ── field comparison ──────────────────────────────────────────────────────────

def compare_field(field, py_arr, ref_arr, headers, tol_rel, tol_abs):
    """Compare two 2-D arrays column by column. Returns list of row dicts."""
    rows = []
    py  = np.asarray(py_arr,  dtype=float)
    ref = np.asarray(ref_arr, dtype=float)
    if py.ndim == 1:
        py = py.reshape(-1, 1)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    nrow = min(py.shape[0], ref.shape[0])
    ncol = min(py.shape[1], ref.shape[1])
    py  = py[:nrow, :ncol]
    ref = ref[:nrow, :ncol]

    for c in range(ncol):
        col_label = str(headers[c]) if (headers and c < len(headers)) else f"col{c}"
        stats = column_stats(py[:, c], ref[:, c], tol_rel=tol_rel, tol_abs=tol_abs)
        rows.append({"field": field, "column": col_label, **stats})
    return rows


# ── averaged output comparison ────────────────────────────────────────────────

def compare_avg(pkl, mat_ref, tol_rel=0.01, tol_abs=1e-9):
    """Compare averaged output dict (pkl) vs MATLAB struct (mat_ref). Returns stat rows."""
    rows = []
    all_fields = set(pkl) | set(mat_ref)

    for field in sorted(all_fields):
        if "Header" in field or "header" in field:
            continue
        if field in SKIP_FIELDS:
            continue
        if field not in pkl:
            rows.append({"field": field, "column": "(all)", **empty_stats("MISS_PY")})
            continue
        if field not in mat_ref:
            rows.append({"field": field, "column": "(all)", **empty_stats("MISS_REF")})
            continue

        try:
            py_arr  = np.asarray(pkl[field],     dtype=float)
            ref_arr = np.asarray(mat_ref[field], dtype=float)
        except (TypeError, ValueError):
            continue

        if py_arr.ndim == 1:
            py_arr = py_arr.reshape(-1, 1)
        if ref_arr.ndim == 1:
            ref_arr = ref_arr.reshape(-1, 1)

        headers = get_header(pkl, field)

        # Skip timestamp column 0 for fields that carry it
        col_start = 1 if field in HAS_TIMESTAMP_COL0 else 0
        if headers and col_start > 0:
            headers = headers[col_start:]

        rows.extend(compare_field(
            field,
            py_arr[:, col_start:],
            ref_arr[:, col_start:],
            headers, tol_rel, tol_abs,
        ))
    return rows


# ── raw output comparison ─────────────────────────────────────────────────────

def compare_raw(pkl, mat_ref, tol_rel=0.01, tol_abs=1e-9):
    """Compare raw 20 Hz output (pkl) vs MATLAB rawFlux struct (mat_ref). Returns stat rows."""
    rows = []
    all_keys = set(RAW_FIELDS) | set(mat_ref)

    for key in sorted(all_keys):
        py_present  = key in pkl
        ref_present = key in mat_ref

        if not py_present and not ref_present:
            continue
        if not py_present:
            rows.append({"field": key, "column": "(all)", **empty_stats("MISS_PY")})
            continue
        if not ref_present:
            rows.append({"field": key, "column": "(all)", **empty_stats("MISS_REF")})
            continue

        try:
            py_arr  = np.asarray(pkl[key],     dtype=float)
            ref_arr = np.asarray(mat_ref[key], dtype=float)
        except (TypeError, ValueError):
            continue

        if py_arr.ndim == 0:
            py_arr = py_arr.reshape(1, 1)
        if ref_arr.ndim == 0:
            ref_arr = ref_arr.reshape(1, 1)
        if py_arr.ndim == 1:
            py_arr = py_arr.reshape(-1, 1)
        if ref_arr.ndim == 1:
            ref_arr = ref_arr.reshape(-1, 1)

        # Subsample to keep comparison fast
        py_arr  = py_arr[::RAW_STRIDE]
        ref_arr = ref_arr[::RAW_STRIDE]

        ncol = min(py_arr.shape[1], ref_arr.shape[1])
        for c in range(ncol):
            col_label = f"col{c}" if ncol > 1 else key
            stats = column_stats(py_arr[:, c], ref_arr[:, c], tol_rel=tol_rel, tol_abs=tol_abs)
            rows.append({"field": key, "column": col_label, **stats})

    return rows
