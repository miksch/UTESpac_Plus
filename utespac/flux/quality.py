"""Quality statistics of a per-period covariance.

:func:`stationarity_fw96` and :func:`foken_class` give the Foken and
Wichura (1996) stationarity statistic and its three-class flag,
:func:`flux_intermittency` the Mahrt et al. (1998) flux intermittency
index on the same sub-interval covariances, :func:`mean_w_flag` the
Mauder et al. (2013) residual mean vertical wind flag,
:func:`wind_constancy` the ratio of vector-mean to scalar-mean wind
speed, :func:`random_error_fs2001` the random error of a covariance after
Finkelstein and Sims (2001). :func:`sliding_rssi_snr` and
:func:`rssi_threshold_below` bin a per-period statistic by signal strength
and find the highest bin that fails a limit.
"""

import numpy as np

# Foken three-class stationarity boundaries [%] on the Foken and Wichura
# (1996) statistic: class 1 up to 30 %, class 2 up to 100 %, class 3 above.
FOKEN_CLASS_EDGES = (30.0, 100.0)

# Thresholds [m s-1] on the mean vertical wind after planar fit, with the
# long-term mean (offset) removed, from Mauder et al. (2013, Sect. 3.2)
# [CITED]: above the first, flux flags rise by one; above the second, they
# are set to 2.
MEAN_W_FLAG_EDGES = (0.10, 0.15)


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


def flux_intermittency(sub_cov, total_cov):
    """Flux intermittency index of Mahrt et al. (1998).

    Mahrt et al. (1998, Boundary-Layer Meteorol. 88, Sect. 3.5, p. 267,
    eq. 9), [CITED], symbols in ASCII: "FI == sigmaF/abs[F] (9) where
    sigmaF is the standard deviation of the 5-minute averaged flux and
    abs[F] is the absolute value of the one-hour average of the flux. With
    substantial intermittency, this ratio exceeds unity." The same section
    notes: "Very small values of the averaged flux can lead to particularly
    large values of the flux intermittency (Equation (9))."

    The paper was written for nocturnal stable boundary layers at a tower
    with eddy-correlation sonics at 3 and 10 m (Sect. 2, p. 256): "This
    study analyzes data collected over simple, relatively flat, grassland in
    Kansas, USA, during March of 1995 in the Microfronts project." Its
    fluxes are computed from deviations about 5-minute means (Sect. 2,
    p. 257), "and then the flux is averaged over one-hour periods."

    Applying the ratio to the sub-blocks of a shorter period, such as the
    sub-intervals of :func:`stationarity_fw96` within 30 min, is
    [ASSUMED]. *total_cov* stands in for the one-hour average of the flux,
    so a whole-period covariance about the period means, which also holds
    the between-block term, replaces the paper's mean of the 5-minute
    fluxes [ASSUMED]. The paper does not state the normalisation of the
    standard deviation; the sample form (``ddof=1``) is used [ASSUMED].

    Parameters
    ----------
    sub_cov : array_like
        Covariance over each sub-interval, each about its own means.
    total_cov : float
        Covariance over the whole period.

    Returns
    -------
    float
        ``std(sub_cov, ddof=1) / |total_cov|`` over the finite
        sub-covariances, the non-finite ones skipped [HOUSE]; NaN when
        fewer than two are finite, or when *total_cov* is zero or not
        finite; inf when *total_cov* is too small to divide by.
    """
    if not np.isfinite(total_cov) or total_cov == 0:
        return np.nan
    sub = np.asarray(sub_cov, dtype=float).ravel()
    sub = sub[np.isfinite(sub)]
    if sub.size < 2:
        return np.nan
    with np.errstate(over="ignore"):
        return float(np.std(sub, ddof=1) / abs(total_cov))


def mean_w_flag(w_mean_pf, edges=MEAN_W_FLAG_EDGES):
    """Flag on the residual mean vertical wind after planar fit.

    Mauder et al. (2013, Agric. For. Meteorol. 169, Sect. 3.2, p. 125),
    written for long-term eddy-covariance tower sites over land, [CITED],
    symbols in ASCII: "Accordingly, w-bar > 0.10 m s-1 results in an
    increase of all flux flags by 1 (flag 0 becomes 1 and flag 1 becomes
    2) and all data with w-bar > 0.15 m s-1 are generally flagged with 2."

    The returned value encodes that rule, applied to each flux flag after
    the stationarity and integral-turbulence flags are combined by their
    maximum and before the interdependence of flags between fluxes (order
    as Fig. 1, p. 124). The combination by the maximum rests on Sect. 3.2,
    p. 125, [CITED]: "The maximum is taken if the individual tests result
    in different flags for a specific flux." The sentence closes the
    paragraph on the integral-turbulence tests of u and w; reading it as
    covering the stationarity flag as well, so that the mean-w flag acts
    on the maximum of both, is [ASSUMED]. 0 leaves the flux flags as
    they are, 1 raises each by one (capped at 2 [ASSUMED]), 2 sets each
    to 2. Both edges are open, as the strict ">" in the source. The test
    is applied to the magnitude of the mean [ASSUMED]; the source writes
    w-bar without a sign convention. A missing mean returns 2 [ASSUMED].

    Parameters
    ----------
    w_mean_pf : float
        Mean vertical wind over the period after the planar-fit
        transformation, with the long-term mean (offset) removed [m s-1].
    edges : tuple of float, optional
        Lower edges of flags 1 and 2 [m s-1], each open.

    Returns
    -------
    int
        0, 1 or 2.
    """
    if not np.isfinite(w_mean_pf):
        return 2
    w = abs(w_mean_pf)
    if w > edges[1]:
        return 2
    return 1 if w > edges[0] else 0


def wind_constancy(u_vec_mean, speed_mean):
    """Ratio of the vector-mean to the scalar-mean wind speed.

    [HOUSE]: ``u_vec_mean / speed_mean``, 1 for a wind of fixed direction
    and near 0 for a wind whose direction reverses within the period.

    Parameters
    ----------
    u_vec_mean : float
        Magnitude of the period-mean horizontal wind vector [m s-1].
    speed_mean : float
        Period mean of the horizontal wind speed [m s-1].

    Returns
    -------
    float
        The ratio; NaN when *speed_mean* is zero or either input is not
        finite; inf on overflow.
    """
    if not (np.isfinite(u_vec_mean) and np.isfinite(speed_mean)) \
            or speed_mean == 0:
        return np.nan
    with np.errstate(over="ignore"):
        return float(u_vec_mean / speed_mean)


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
