"""stpDn – step-down average (block-mean by fixed row count).

Legacy name for :func:`utespac.averaging.block_mean_rows`.
"""

import numpy as np

from .averaging import block_mean_rows


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
    return block_mean_rows(data, rows)
