"""stpDn – step-down average (block-mean by fixed row count)."""

import numpy as np


def stp_dn(data: np.ndarray, rows: int) -> np.ndarray:
    """Block-average *data* into groups of *rows* rows.

    Parameters
    ----------
    data : ndarray, shape (N, M)
    rows : int
        Number of rows per averaging block.

    Returns
    -------
    out : ndarray, shape (N // rows, M)
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    n_out = n // rows
    data2d = data.reshape(n, -1) if data.ndim == 1 else data
    out = np.full((n_out, data2d.shape[1]), np.nan)
    for i in range(n_out):
        out[i, :] = np.nanmean(data2d[i * rows:(i + 1) * rows, :], axis=0)
    return out
