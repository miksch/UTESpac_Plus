"""Lag-resolved covariances of one averaging block.

:func:`detrend_block` forms a block-mean anomaly with zeros in place of
missing samples, :func:`lagged_covariances` the whole-block and sub-block
covariance at every lag of a window, and :func:`peak_lag` the lag of the
largest covariance and how far it stands above the outer lags.
:func:`despike_mask`
screens amplitude spikes against a rolling median, and
:func:`noise_variance_lmk` estimates the white-noise variance of a series
from its lag-zero autocovariance (Lenschow, Mann and Kristensen 2000).
"""

import numpy as np

# Lenschow, Mann and Kristensen (2000): inside the inertial subrange the
# autocovariance falls as sigma^2 - b tau^(2/3).
LMK_EXPONENT = 2.0 / 3.0


def detrend_block(x):
    """Block-mean anomaly of *x*, plus the mean and a finite-sample mask.

    Parameters
    ----------
    x : array_like
        One block of the series.

    Returns
    -------
    anomaly : ndarray
        ``x - mean`` with NaN (and any non-finite sample) replaced by zero, so
        a covariance is the sum of valid products divided by the count of
        valid pairs.
    mean : float
        Mean of the finite samples; NaN when there are none.
    ok : ndarray of bool
        Finite-sample mask.
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    mean = float(np.mean(x[ok])) if ok.any() else np.nan
    out = np.where(ok, x - mean, 0.0)
    return out, mean, ok


def lagged_covariances(w, w_ok, c, c_ok, max_lag, bounds=None):
    """Covariance of *w* with *c* delayed by each lag in ``[-max_lag, max_lag]``.

    Parameters
    ----------
    w, c : ndarray
        Block-mean anomalies with zeros in place of missing samples.
    w_ok, c_ok : ndarray of bool
        Validity of each sample of *w* and *c*.
    max_lag : int
        Half-width of the lag window, in samples. A positive lag means *c*
        is delayed relative to *w*, the sense of a sample-line transport
        delay.
    bounds : ndarray, optional
        Sub-block edges (indices into *w*) for the Foken and Wichura (1996)
        sub-covariances. Each sub-covariance is taken about that
        sub-interval's own means, which is what makes the statistic
        informative.

    Returns
    -------
    lags : ndarray, shape (2 * max_lag + 1,)
        Lags in samples.
    cov : ndarray, shape (2 * max_lag + 1,)
        Covariance over the whole block at each lag.
    sub : ndarray, shape (n_sub, 2 * max_lag + 1) or None
        Covariance over each sub-block, when *bounds* is given.
    """
    n = len(w)
    lags = np.arange(-int(max_lag), int(max_lag) + 1)
    cov = np.full(len(lags), np.nan)
    use_sub = bounds is not None and np.all(np.diff(bounds) > 0)
    n_sub = len(bounds) - 1 if use_sub else 0
    sub = np.full((n_sub, len(lags)), np.nan) if n_sub else None
    prod = np.empty(n)
    pair = np.empty(n)
    w_sel = np.empty(n)
    c_sel = np.empty(n)
    for j, k in enumerate(lags):
        for buf in (prod, pair, w_sel, c_sel):
            buf[:] = 0.0
        if k >= 0:
            m = w_ok[:n - k] & c_ok[k:]
            prod[:n - k] = w[:n - k] * c[k:]
            pair[:n - k] = m
            w_sel[:n - k] = w[:n - k] * m
            c_sel[:n - k] = c[k:] * m
        else:
            i0 = -k
            m = w_ok[i0:] & c_ok[:n - i0]
            prod[i0:] = w[i0:] * c[:n - i0]
            pair[i0:] = m
            w_sel[i0:] = w[i0:] * m
            c_sel[i0:] = c[:n - i0] * m
        cnt = pair.sum()
        cov[j] = (prod.sum() / cnt - (w_sel.sum() / cnt) * (c_sel.sum() / cnt)
                  if cnt > 0 else np.nan)
        if n_sub:
            s_cnt = np.add.reduceat(pair, bounds[:-1])
            with np.errstate(invalid="ignore", divide="ignore"):
                s_prod = np.add.reduceat(prod, bounds[:-1]) / s_cnt
                s_w = np.add.reduceat(w_sel, bounds[:-1]) / s_cnt
                s_c = np.add.reduceat(c_sel, bounds[:-1]) / s_cnt
                sub[:, j] = np.where(s_cnt > 0, s_prod - s_w * s_c, np.nan)
    return lags, cov, sub


def peak_lag(lags, cov):
    """Lag of maximum ``|cov|`` and the peak-to-background ratio.

    The background is the rms of the covariance over the outer third of the
    lag window, so a period whose peak does not stand out is recognisable.

    Parameters
    ----------
    lags : ndarray of int
        Lags in samples, as returned by :func:`lagged_covariances`.
    cov : ndarray
        Covariance at each lag.

    Returns
    -------
    lag : int or float
        Lag of the largest ``|cov|``; NaN when no covariance is finite.
    ratio : float
        ``|cov|`` at that lag over the background rms; ``inf`` when the
        background is zero or undefined, NaN when no covariance is finite.
    """
    ok = np.isfinite(cov)
    if not ok.any():
        return np.nan, np.nan
    j = np.nanargmax(np.abs(cov))
    outer = np.abs(lags) > (2.0 / 3.0) * np.abs(lags).max()
    back = np.sqrt(np.nanmean(cov[outer & ok] ** 2)) if (outer & ok).any() else np.nan
    ratio = abs(cov[j]) / back if np.isfinite(back) and back > 0 else np.inf
    return int(lags[j]), float(ratio)


def despike_mask(x, window, n_mad):
    """Samples further than *n_mad* robust deviations from a rolling median.

    A centred rolling median and a rolling median absolute deviation, scaled
    to a Gaussian standard deviation by 1.4826. A window needs at least
    ``window // 5`` finite samples to return a median.

    Parameters
    ----------
    x : array_like
        The series.
    window : int
        Rolling-window length, in samples.
    n_mad : float
        Rejection limit, in scaled median absolute deviations.

    Returns
    -------
    ndarray of bool
        True at finite samples that exceed the limit.

    Notes
    -----
    Requires pandas, which is imported on call.
    """
    import pandas as pd

    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(window, center=True, min_periods=window // 5).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window // 5).median()
    scale = 1.4826 * mad
    lim = np.where(scale.to_numpy() > 0, n_mad * scale.to_numpy(), np.inf)
    return (np.abs(s.to_numpy() - med.to_numpy()) > lim) & np.isfinite(x)


def noise_variance_lmk(x, fs, fit_lags, exponent=LMK_EXPONENT):
    """White-noise variance of a series from its lag-zero autocovariance jump.

    Lenschow, Mann and Kristensen (2000): uncorrelated instrument noise adds
    power at lag zero only, so the measured autocovariance carries a spike
    there that the atmospheric signal does not. Inside the inertial subrange
    the autocovariance falls as ``R(tau) = sigma^2 - b tau^(2/3)``, so a
    least-squares line of ``R`` against ``tau^(2/3)`` over the first few lags,
    extrapolated back to ``tau = 0``, gives the signal variance; the
    difference from the measured lag-zero value is the noise variance.

    Parameters
    ----------
    x : ndarray
        One period of the series. NaN samples are replaced by zero in the
        anomaly, which biases the estimate low in proportion to the gap
        fraction, so callers should screen on coverage first.
    fs : float
        Scan rate [Hz].
    fit_lags : tuple of int
        First and last lag, in samples, admitted to the fit.
    exponent : float, optional
        Structure-function exponent; the default 2/3 is the inertial
        subrange.

    Returns
    -------
    noise_var : float
        Estimated white-noise variance, in the squared units of *x*. Clipped
        at zero: a negative estimate means the extrapolation sits above the
        measured lag-zero value and carries no noise information.
    total_var : float
        Measured variance (the lag-zero autocovariance).
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 1000:
        return np.nan, np.nan
    a = np.where(ok, x - x[ok].mean(), 0.0)
    n = len(a)
    lo, hi = int(fit_lags[0]), int(fit_lags[1])
    if hi <= lo or hi >= n // 2:
        return np.nan, np.nan
    size = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(a, size)
    acov = np.fft.irfft(spec * np.conj(spec), size)[:hi + 1] / ok.sum()
    total = float(acov[0])
    tau = np.arange(lo, hi + 1) / float(fs)
    design = np.column_stack([np.ones(hi - lo + 1), tau ** exponent])
    coef, *_ = np.linalg.lstsq(design, acov[lo:hi + 1], rcond=None)
    signal_var = float(coef[0])
    return max(total - signal_var, 0.0), total
