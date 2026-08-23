"""Column-wise comparison statistics for UTESpac outputs.

Pure functions — no printing, no file discovery. The MATLAB golden-file
machinery that used to live here (``discover_pairs``, ``.mat`` loaders,
``compare_avg``/``compare_raw``) was retired with MATLAB parity on
2026-08-22; what remains serves the EddyPro comparison scripts and any
pkl-vs-pkl check.
"""

import numpy as np

STAT_KEYS_ALL = [
    "N_total", "N_valid", "N_nan_mismatch",
    "bias", "RMSE", "max_abs", "mean_abs", "p95_abs", "p99_abs",
    "max_rel", "mean_rel", "R2",
]


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
    py = np.asarray(py_col, dtype=float).ravel()
    ref = np.asarray(ref_col, dtype=float).ravel()
    n = min(len(py), len(ref))
    py, ref = py[:n], ref[:n]

    py_nan = np.isnan(py)
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

    diff = py[mask] - ref[mask]
    adiff = np.abs(diff)
    ref_m = ref[mask]
    denom = np.abs(ref_m).copy()
    denom[denom < tol_abs] = tol_abs
    rel = adiff / denom

    bias = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_abs = float(adiff.max())
    mean_abs = float(adiff.mean())
    p95_abs = float(np.percentile(adiff, 95))
    p99_abs = float(np.percentile(adiff, 99))
    max_rel = float(rel.max())
    mean_rel = float(rel.mean())

    # R² (correlation coefficient squared) — meaningful only when ref has variance
    ref_std = float(ref_m.std())
    if ref_std > 1e-14 and n_valid > 2:
        r = float(np.corrcoef(py[mask], ref_m)[0, 1])
        r2 = r ** 2 if np.isfinite(r) else null
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
    py = np.asarray(py_arr, dtype=float)
    ref = np.asarray(ref_arr, dtype=float)
    if py.ndim == 1:
        py = py.reshape(-1, 1)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    nrow = min(py.shape[0], ref.shape[0])
    ncol = min(py.shape[1], ref.shape[1])
    py = py[:nrow, :ncol]
    ref = ref[:nrow, :ncol]

    for c in range(ncol):
        col_label = str(headers[c]) if (headers and c < len(headers)) else f"col{c}"
        stats = column_stats(py[:, c], ref[:, c], tol_rel=tol_rel, tol_abs=tol_abs)
        rows.append({"field": field, "column": col_label, **stats})
    return rows
