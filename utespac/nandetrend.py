"""nandetrend – detrend a 1-D signal while ignoring NaN values."""

import numpy as np


def nandetrend(x: np.ndarray, order=1) -> np.ndarray:
    """Remove a linear (or constant) trend from *x*, tolerating NaN values.

    Parameters
    ----------
    x : array_like, 1-D
    order : int or str
        0 / 'constant' – remove mean only.
        1 / 'linear'   – remove best-fit line (default).

    Returns
    -------
    y : ndarray, same shape as *x*
        Detrended signal; NaN positions are preserved.
    """
    x = np.asarray(x, dtype=float).ravel()
    nan_mask = np.isnan(x)

    # If > 90 % NaN, return all-NaN
    if nan_mask.sum() / len(x) > 0.9:
        return np.full_like(x, np.nan)

    x_clean = x[~nan_mask]
    n = len(x_clean)
    idx = np.flatnonzero(~nan_mask).astype(float)   # true sample index, gaps kept

    if order in (0, "c", "constant"):
        detrended = x_clean - np.nanmean(x_clean)
    else:  # linear
        if n < 2:
            detrended = x_clean - np.nanmean(x_clean)
        else:
            coeffs = np.polyfit(idx, x_clean, 1)
            trend = np.polyval(coeffs, idx)
            detrended = x_clean - trend

    y = np.full_like(x, np.nan)
    y[~nan_mask] = detrended
    return y
