"""PF_coefficients – planar-fit (b0, b1, b2) per direction bin.

Legacy keyed-dict form of :func:`utespac.rotation.fit_sectors`.
"""

from typing import Dict, List
import numpy as np

from .rotation import fit_sectors


def pf_coefficients(input_mat: np.ndarray, bins: List[float]) -> Dict[str, np.ndarray]:
    """Solve for planar-fit coefficients that force w̄ → 0.

    Parameters
    ----------
    input_mat : ndarray, shape (N, 4)
        Columns: [u, v, w, direction].
    bins : list of float
        Direction bin boundaries in degrees.  Empty list = single sector.

    Returns
    -------
    coef : dict
        Keys like ``'degrees_0_to_90'``; values are 3-element arrays [b0, b1, b2]
        (NaN when a sector has fewer than four points or a singular system).
    """
    m = np.asarray(input_mat, dtype=float)
    fits = fit_sectors(m[:, 0], m[:, 1], m[:, 2], m[:, 3], list(bins))
    out: Dict[str, np.ndarray] = {}
    for (lo, hi), fit in fits.items():
        key = f"degrees_{lo:g}_to_{hi:g}"
        out[key] = (np.array([fit.b0, fit.b1, fit.b2]) if fit is not None
                    else np.array([np.nan, np.nan, np.nan]))
    return out
