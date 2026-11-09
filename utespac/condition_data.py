"""QC of the raw high-frequency tables (Vickers & Mahrt 1997 style).

:func:`qc_table` is the per-table kernel (absolute limits, spike removal
with consecutive-flag filtering, per-period spike/NaN fractions);
:func:`utespac.stages.condition` runs it over the run model. The spike
test is either the iterative mean/standard-deviation test of UTESpac
(``spikeTest.method = "sigma"``, the default) or the median absolute
deviation (MAD) test of Mauder et al. (2013) (``"mad"``).
:func:`skew_kurt_flag` is the higher-moment test of Vickers and Mahrt
(1997), section 6e, as a per-period flag.

Both sources were written for other systems than the one at hand.
Mauder et al. (2013, Agric. For. Meteorol. 169, 122-135) built the MAD
test for the long-term tower eddy-covariance sites of the TERENO network
over land: "In TERENO Campbell CSAT3 sonic anemometers, Licor LI-7500
open-path gas analysers and LI-7000 closed-path gas analysers are
deployed" (p. 124). Vickers and Mahrt (1997, J. Atmos. Oceanic Technol.
14, 512-526) set their limits on "the RASEX tower data collected off the
coast of Denmark", "the Microfronts95 tower data collected over grass
land during the springtime in Kansas" and "the BOREAS TwinOtter aircraft
data collected over the Boreal forest region of Canada" (section 1,
p. 513).
"""

from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.interpolate import interp1d

from . import units
from .averaging import block_average
from .consec_flag_removal import consec_flag_removal
from .strfndw import strfndw

#: Quantity (:data:`utespac.units.INTERNAL`) of each sensor class whose
#: absolute limits are stated in the pipeline's internal unit; the other
#: classes are compared as logged.
LIMIT_QUANTITY = {
    "Tson": "temperature", "fw": "temperature", "T": "temperature",
    "RH": "humidity", "P": "pressure",
    "irgaH2O": "h2o_density", "KH2O": "h2o_density", "irgaCO2": "co2_density",
}

#: Values of ``spikeTest.method``.
SPIKE_METHODS = ("sigma", "mad")

#: [CITED] Divisor that scales the MAD to a standard deviation, Mauder et
#: al. (2013), Eq. (1), section 3.1, p. 124: "The factor of 0.6745 relates
#: an MAD of unity to one arithmetic standard deviation if the frequency
#: distribution is Gaussian, which in reality is not always the case."
MAD_GAUSS = 0.6745

#: [CITED] Skewness and kurtosis ranges of Vickers and Mahrt (1997),
#: section 6e, p. 521: "The record is hard flagged when the skewness is
#: outside the range (-2, 2) or the kurtosis is outside the range (1, 8).
#: [...] The record is soft flagged when the skewness is outside the range
#: (-1, 1) or the kurtosis is outside (2, 5)." The paper adds "These limits
#: are empirically based on the datasets considered here."
SKEW_SOFT = (-1.0, 1.0)
KURT_SOFT = (2.0, 5.0)
SKEW_HARD = (-2.0, 2.0)
KURT_HARD = (1.0, 8.0)


def qc_table(tbl: np.ndarray, header_names: Sequence[str], info: Dict, template: Dict[str, str],
             scan_hz: float, header_units: Optional[Sequence[str]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QC one table in place (column 0 is the datenum time axis).

    Parameters
    ----------
    tbl : ndarray
        ``[datenum, columns...]`` as logged.
    header_names : sequence of str
        Column labels, aligned with the columns of *tbl*.
    info : dict
        Site and QC settings (``spikeTest``, ``absoluteLimitsTest``,
        ``nanTest``, ``avgPer``). ``spikeTest.method`` selects the spike
        statistic (see Notes); absent, it is "sigma".
    template : dict
        Sensor class -> column-label pattern.
    scan_hz : float
        Sampling rate of the table; below 1 Hz the spike test is skipped.
    header_units : sequence of str, optional
        Declared unit per column (``Run.header_units``), "" where none.
        The absolute limits of the classes in :data:`LIMIT_QUANTITY` are
        stated in the internal unit, so those columns are compared after
        :func:`utespac.units.convert`; the stored column stays as logged.
        An undeclared unit takes the magnitude fallback without a warning;
        the flux stage logs it when it converts the same column.

    Returns
    -------
    tbl : ndarray
        The conditioned table: limits applied, spikes interpolated.
    spike_flag, nan_flag : ndarray of bool
        Per-period flags ``(n_periods, n_cols)``, True where the period's
        spike (NaN) fraction reaches ``spikeTest.maxPercent``
        (``nanTest.maxPercent``); column 0 is always False.

    Raises
    ------
    ValueError
        If ``spikeTest.method`` is not one of :data:`SPIKE_METHODS`.

    Notes
    -----
    The record is cut into windows of ``avgPer * windowSizeFraction``
    minutes, overlapping by half. Within each window a sample is a
    potential spike when

    - ``"sigma"``: ``|x - mean| >= spikeDef * std``, with the per-class
      ``spikeDef``; the pass repeats up to ``maxRuns`` times over the
      interpolated record;
    - ``"mad"``: ``|x - median| > q * MAD / 0.6745`` with
      ``MAD = median(|x - median|)`` and ``q = madThreshold`` for every
      spike-tested column, Mauder et al. (2013), section 3.1, p. 124
      [CITED]: "a data point xi is flagged as a spike if it is not in the
      range:", their Eq. (1) (transcribed)
      ``<x> - q MAD/0.6745 <= xi <= <x> + q MAD/0.6745``, "where <x>
      denotes the median of x", with Eq. (2) ``MAD = <|xi - <x>|>``.
      One pass only [CITED], as the MAD test "also avoids the high
      computational cost of the iteration" (same page). A window whose
      MAD is zero flags nothing [ASSUMED]; Eq. (1) read literally then
      flags every sample off the median. The two tests part there: a
      +20 spike on a constant series, or on one with 90% of its samples
      at the median, is kept under ``"mad"`` and removed under
      ``"sigma"``, whose standard deviation the spike itself raises
      above zero.

    Both methods then drop runs longer than ``maxConsecutiveOutliers``
    and interpolate the remaining spikes linearly. The consecutive-run
    filter and the half-overlapping windows are UTESpac's [HOUSE], not part of
    Mauder et al. (2013), who computed their results "using 30-min
    windows" (p. 125). Interpolating over rejected points is this
    package's behaviour [HOUSE]; in Mauder et al. (2013) rejected points
    "were not interpolated in the time series in any form but replaced
    by an error code (e.g. NaN)" (p. 125).
    """
    method = info["spikeTest"].get("method", "sigma")
    if method not in SPIKE_METHODS:
        raise ValueError(f"unknown spikeTest.method {method!r}; "
                         f"expected one of {SPIKE_METHODS}")
    spike_def_cfg = info["spikeTest"]["spikeDef"]
    abs_lim_cfg = info["absoluteLimitsTest"]
    n_rows, n_cols = tbl.shape
    is_slow_table = scan_hz < 1  # < 1 Hz — skip spike test, apply limits + NaN flag only

    # Build per-column spike definitions and absolute limits
    spike_def = np.full(n_cols, spike_def_cfg["otherInstrument"])
    min_max = np.full((2, n_cols), np.nan)
    quantity = [None] * n_cols
    spike_def[0] = np.nan  # timestamp column is never spiked

    for field, pattern in template.items():
        if field not in spike_def_cfg or field not in abs_lim_cfg:
            continue
        if field == "otherInstrument" or not pattern:
            continue
        matched_cols = strfndw(list(header_names), pattern)
        for c in matched_cols:
            spike_def[c] = spike_def_cfg[field]
            lims = abs_lim_cfg[field]
            min_max[0, c] = lims[0]
            min_max[1, c] = lims[1]
            quantity[c] = LIMIT_QUANTITY.get(field)

    # Also mark sonic diagnostic column as NaN for spike purposes
    if "sonDiagnostic" in template:
        for c in strfndw(list(header_names), template["sonDiagnostic"]):
            spike_def[c] = np.nan

    # IMU yaw wraps through 0/360 -- the jumps are real, not spikes (its
    # absolute limits above still apply)
    if template.get("imuYaw"):
        for c in strfndw(list(header_names), template["imuYaw"]):
            spike_def[c] = np.nan

    # 1. Absolute limits, judged in the internal unit of the column's quantity
    for c in range(n_cols):
        if not np.isnan(min_max[0, c]):
            judged = tbl[:, c]
            if quantity[c] is not None:
                unit = header_units[c] if header_units is not None else None
                judged = units.convert(judged, unit, quantity[c], warn=False)
            bad = (judged < min_max[0, c]) | (judged > min_max[1, c])
            tbl[bad, c] = np.nan

    # 2. Spike removal (skipped for slow tables — too few samples per window)
    nan_flag = np.isnan(tbl)
    spike_flag = np.zeros((n_rows, n_cols), dtype=bool)

    if not is_slow_table:
        num_days = tbl[-1, 0] - tbl[0, 0]
        frac = info["spikeTest"]["windowSizeFraction"]
        num_bins = int(round(num_days / (info["avgPer"] * frac / 1440))) * 2 + 1
        bp = np.round(np.linspace(0, n_rows, num_bins)).astype(int)

        no_spike_flag = np.zeros(len(bp) - 1, dtype=bool)
        n_runs = info["spikeTest"]["maxRuns"] if method == "sigma" else 1
        tested = ~np.isnan(spike_def)

        for _run in range(n_runs):
            if no_spike_flag[:-1].all():
                break
            for k in range(len(bp) - 2):
                if no_spike_flag[k]:
                    continue
                r0, r1 = bp[k], bp[k + 2]
                local = tbl[r0:r1, :].copy()
                local_nan = nan_flag[r0:r1, :]

                pot_spikes = np.zeros_like(local, dtype=bool)
                if method == "sigma":
                    stds = np.nanstd(local, axis=0)
                    means = np.nanmean(local, axis=0)
                    with np.errstate(invalid="ignore"):
                        normed = np.abs((local - means)
                                        / np.where(stds == 0, np.nan, stds))
                    for c in range(n_cols):
                        if not np.isnan(spike_def[c]):
                            pot_spikes[:, c] = normed[:, c] >= spike_def[c]
                else:
                    sub = local[:, tested]
                    with warnings.catch_warnings():
                        # all-NaN columns
                        warnings.simplefilter("ignore", RuntimeWarning)
                        med = np.nanmedian(sub, axis=0)
                        mad = np.nanmedian(np.abs(sub - med), axis=0)
                    lim = info["spikeTest"]["madThreshold"] * mad / MAD_GAUSS
                    with np.errstate(invalid="ignore"):
                        pot_spikes[:, tested] = ((np.abs(sub - med) > lim)
                                                 & (lim > 0))

                local_spikes = consec_flag_removal(
                    pot_spikes, info["spikeTest"]["maxConsecutiveOutliers"]
                )
                spike_flag[r0:r1, :] |= local_spikes

                if not local_spikes.any():
                    no_spike_flag[k] = True
                    continue

                # Linear interpolation over spikes
                for c in range(1, n_cols):
                    bad = local_spikes[:, c] | local_nan[:, c]
                    good = ~bad
                    if good.sum() < 2 or bad.sum() / len(bad) > 0.6:
                        local[:, c] = np.nan
                        continue
                    x_all = np.arange(r1 - r0)
                    x_good = x_all[good]
                    y_good = local[good, c]
                    x_bad = x_all[local_spikes[:, c]]
                    if len(x_bad):
                        f = interp1d(x_good, y_good, bounds_error=False, fill_value=np.nan)
                        local[local_spikes[:, c], c] = f(x_bad)
                tbl[r0:r1, :] = local

    # 3. Per-period flag fractions -> booleans
    t = tbl[:, 0]
    _, mean_spike = block_average(t, spike_flag[:, 1:].astype(float), info["avgPer"])
    mean_spike = np.column_stack([np.zeros(mean_spike.shape[0]), mean_spike])
    mean_spike[mean_spike < info["spikeTest"]["maxPercent"] / 100.0] = 0
    _, mean_nan = block_average(t, nan_flag[:, 1:].astype(float), info["avgPer"])
    mean_nan = np.column_stack([np.zeros(mean_nan.shape[0]), mean_nan])
    mean_nan[mean_nan < info["nanTest"]["maxPercent"] / 100.0] = 0
    return tbl, mean_spike.astype(bool), mean_nan.astype(bool)


def skew_kurt_flag(x, skew_lim=SKEW_SOFT, kurt_lim=KURT_SOFT):
    """Higher-moment flag of one averaging period (Vickers and Mahrt 1997).

    Parameters
    ----------
    x : array_like
        One averaging period of a series [ASSUMED] (see Notes);
        non-finite samples are dropped.
    skew_lim, kurt_lim : tuple of float
        Open ranges ``(lo, hi)`` of the skewness and the kurtosis; the
        defaults are the soft-flag ranges :data:`SKEW_SOFT` and
        :data:`KURT_SOFT`, :data:`SKEW_HARD` and :data:`KURT_HARD` the
        hard-flag ones.

    Returns
    -------
    bool
        True when the skewness or the kurtosis of the linearly detrended
        period lies outside its range. False with fewer than three finite
        samples or a residual variance not above machine epsilon times
        the mean square (std below 1.5e-8 of the rms) [ASSUMED].

    Notes
    -----
    Section 6e, pp. 520-521 [CITED]: "Any linear trend in the series is
    removed prior to computing the higher moment statistics" and "The
    skewness and kurtosis of the fields are computed for the entire
    record." The record of the paper is 1 h: "Our analysis considers
    1-h data records" (sections 2 and 3, p. 513), flags being assigned
    "to the entire data record (typically 1 h or 10 km)" (section 1,
    p. 513). Applying the test to one averaging period of this package
    instead is [ASSUMED]. The paper gives no formula. The kurtosis is
    the non-excess fourth standardized moment, 3 for a Gaussian
    [ASSUMED]: both its ranges, (1, 8) and (2, 5), bracket 3. The
    moments are the biased (1/N) estimators [ASSUMED]. A value equal to
    a bound counts as outside, reading the ranges as open intervals
    [ASSUMED].
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return False
    t = np.arange(x.size, dtype=float)[ok]
    y = x[ok]
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    var = np.mean(resid ** 2)
    # round-off of a linear series
    if var <= np.finfo(float).eps * np.mean(y ** 2):
        return False
    skew = np.mean(resid ** 3) / var ** 1.5
    kurt = np.mean(resid ** 4) / var ** 2
    in_skew = skew_lim[0] < skew < skew_lim[1]
    in_kurt = kurt_lim[0] < kurt < kurt_lim[1]
    return not (in_skew and in_kurt)

