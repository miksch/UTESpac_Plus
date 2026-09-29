"""Quality statistics of a per-period covariance.

:func:`stationarity_fw96` and :func:`foken_class` give the Foken and
Wichura (1996) stationarity statistic and its three-class flag,
:func:`random_error_fs2001` the random error of a covariance after
Finkelstein and Sims (2001). :func:`sliding_rssi_snr` and
:func:`rssi_threshold_below` bin a per-period statistic by signal strength
and find the highest bin that fails a limit.
"""

import numpy as np

# Foken three-class stationarity boundaries [%] on the Foken and Wichura
# (1996) statistic: class 1 up to 30 %, class 2 up to 100 %, class 3 above.
FOKEN_CLASS_EDGES = (30.0, 100.0)


def stationarity_fw96(sub_cov, total_cov):
    """Foken and Wichura (1996) stationarity statistic [%].

    The mean of the sub-interval covariances against the whole-period
    covariance; values above 30% mark a non-stationary period.

    Parameters
    ----------
    sub_cov : array_like
        Covariance over each sub-interval, each about its own means.
    total_cov : float
        Covariance over the whole period.

    Returns
    -------
    float
        ``|(mean(sub_cov) - total_cov) / total_cov| * 100``; NaN when
        *total_cov* is zero or not finite, or no sub-covariance is finite.
    """
    if not np.isfinite(total_cov) or total_cov == 0:
        return np.nan
    m = np.nanmean(sub_cov)
    if not np.isfinite(m):
        return np.nan
    return float(abs((m - total_cov) / total_cov) * 100.0)


def foken_class(stat_pct, edges=FOKEN_CLASS_EDGES):
    """Foken three-class stationarity flag from the FW96 statistic [%].

    Class 1 is at or below the first edge (30%), class 2 between the edges
    (30-100%), class 3 above the second (over 100%). A missing statistic
    returns 0.

    Parameters
    ----------
    stat_pct : float
        Statistic from :func:`stationarity_fw96` [%].
    edges : tuple of float, optional
        Upper edges of classes 1 and 2 [%], each closed.

    Returns
    -------
    int
        0, 1, 2 or 3.
    """
    if not np.isfinite(stat_pct):
        return 0
    if stat_pct <= edges[0]:
        return 1
    return 2 if stat_pct <= edges[1] else 3


def random_error_fs2001(w, c, fs, max_lag_s):
    """Random error of a covariance after Finkelstein and Sims (2001).

    Their eqs. 8-10: the variance of the covariance estimate is the integral,
    over lags out to *max_lag_s*, of the product of the two auto-covariances
    plus the product of the two cross-covariances,

    ``var = (1/n) sum_p [ g_ww(p) g_cc(p) + g_wc(p) g_cw(p) ]``

    with biased (divide-by-n) estimators evaluated by FFT.

    Parameters
    ----------
    w, c : ndarray
        Block-mean anomalies of the two series, already aligned at the chosen
        lag and with zeros in place of missing samples, so a missing sample
        contributes nothing to any lag product.
    fs : float
        Scan rate [Hz].
    max_lag_s : float
        Integration limit [s]; the sums are truncated there rather than at a
        convergence test.

    Returns
    -------
    float
        Standard error of the covariance, in the product units of *w* and *c*.
        NaN for fewer than 1000 samples, fewer than 10 integration lags, or a
        non-positive variance.
    """
    w = np.asarray(w, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(w)
    m = int(min(round(max_lag_s * fs), n // 2 - 1))
    if n < 1000 or m < 10:
        return np.nan
    size = int(2 ** np.ceil(np.log2(2 * n)))
    W = np.fft.rfft(w, size)
    C = np.fft.rfft(c, size)
    g_ww = np.fft.irfft(W * np.conj(W), size)[:m + 1] / n
    g_cc = np.fft.irfft(C * np.conj(C), size)[:m + 1] / n
    g_wc = np.fft.irfft(np.conj(W) * C, size)[:m + 1] / n      # p >= 0
    g_cw = np.fft.irfft(W * np.conj(C), size)[:m + 1] / n      # g_wc(-p)
    auto = g_ww[0] * g_cc[0] + 2.0 * float(np.sum(g_ww[1:] * g_cc[1:]))
    cross = g_wc[0] * g_cw[0] + 2.0 * float(np.sum(g_wc[1:] * g_cw[1:]))
    var = (auto + cross) / n
    return float(np.sqrt(var)) if var > 0 else np.nan


def sliding_rssi_snr(rssi, snr, width, step, min_n):
    """Median signal-to-noise ratio in a sliding RSSI bin.

    The bin centres run from the smallest to the largest observed RSSI in
    steps of *step*, so both extremes sit inside a bin.

    Parameters
    ----------
    rssi : array_like
        Signal strength of each period.
    snr : array_like
        Statistic of each period whose median is taken per bin.
    width : float
        Full bin width, in RSSI units; each bin is closed on both edges.
    step : float
        Spacing of the bin centres, in RSSI units.
    min_n : int
        Periods a bin needs for a median; fewer leave it NaN. Fewer than
        *min_n* finite pairs in total return empty arrays.

    Returns
    -------
    centres, medians, counts : ndarray
        Bin centre, median of *snr* inside it, and the number of periods.
    """
    rssi = np.asarray(rssi, dtype=float)
    snr = np.asarray(snr, dtype=float)
    ok = np.isfinite(rssi) & np.isfinite(snr)
    if ok.sum() < min_n:
        return np.array([]), np.array([]), np.array([])
    lo, hi = np.nanmin(rssi[ok]), np.nanmax(rssi[ok])
    centres = np.arange(lo, hi + step, step)
    med = np.full(len(centres), np.nan)
    cnt = np.zeros(len(centres), dtype=int)
    for i, c in enumerate(centres):
        m = ok & (rssi >= c - width / 2) & (rssi <= c + width / 2)
        cnt[i] = int(m.sum())
        if cnt[i] >= min_n:
            med[i] = float(np.median(snr[m]))
    return centres, med, cnt


def rssi_threshold_below(centres, values, limit, min_n=None, counts=None):
    """Highest RSSI at which a binned statistic still sits below *limit*.

    Periods at or below the returned value fall in the failing tier.

    Parameters
    ----------
    centres : ndarray
        Bin centres, as from :func:`sliding_rssi_snr`.
    values : ndarray
        Binned statistic at each centre.
    limit : float
        A bin fails when its value is below this.
    min_n : int, optional
        Bins with fewer than *min_n* counts are ignored, when *counts* is
        also given.
    counts : ndarray, optional
        Periods in each bin.

    Returns
    -------
    float
        Largest failing centre; NaN when no populated bin fails.
    """
    ok = np.isfinite(values)
    if counts is not None and min_n is not None:
        ok &= counts >= min_n
    bad = ok & (values < limit)
    return float(np.max(centres[bad])) if bad.any() else np.nan
