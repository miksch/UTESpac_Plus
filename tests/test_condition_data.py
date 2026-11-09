"""Tests of :mod:`utespac.condition_data`: absolute limits and spike tests
of :func:`qc_table`, and :func:`skew_kurt_flag`."""

import numpy as np
import pytest

from utespac.condition_data import (KURT_HARD, KURT_SOFT, SKEW_HARD, SKEW_SOFT,
                                    qc_table, skew_kurt_flag)
from utespac.run_config import QCConfig, _DEFAULT_TEMPLATE

T0_DATENUM = 739000.0


def _slow_table(label, values):
    """One day at one sample a minute: below 1 Hz, so the spike test is skipped."""
    hz = 1.0 / 60.0
    t = T0_DATENUM + np.arange(values.size) / 1440.0
    return np.column_stack([t, values]), ["TIMESTAMP", label], hz


@pytest.mark.parametrize("label, unit, base, planted", [
    ("T_Sonic_3", "deg C", 20.0, 100.0),          # internal unit, compared as logged
    ("T_Sonic_3", "K", 293.15, 373.15),
    ("FW_3", "K", 293.15, 373.15),
    ("FW_3", "", 293.15, 373.15),                 # undeclared: K above a median of 250
    ("H2O_3", "mg m-3", 1.0e4, 6.0e4),
    ("H2O_3", "kg m-3", 0.01, 0.06),
    ("CO2_3", "g m-3", 0.7, 1.6),
    ("CO2_3", "ug m-3", 7.0e5, 1.6e6),
])
def test_limit_is_judged_in_the_internal_unit(label, unit, base, planted):
    """With limits Tson/fw [-20, 80] deg C, irgaH2O [0, 50] g m-3 and irgaCO2
    [0, 1500] mg m-3, only the planted sample lies outside once converted,
    and the kept samples are stored as logged."""
    values = np.full(1440, base)
    values[700] = planted
    tbl, labels, hz = _slow_table(label, values)
    info = {"avgPer": 30, **QCConfig().to_info()}
    out, _, _ = qc_table(tbl.copy(), labels, info, dict(_DEFAULT_TEMPLATE), hz,
                         header_units=["", unit])
    expected = values.copy()
    expected[700] = np.nan
    np.testing.assert_array_equal(out[:, 1], expected)


# -- spike tests --------------------------------------------------------------

SINE_PERIOD = 60          # [samples]
CLEAN_N = 7200            # two hours at 1 Hz: windows of 1800 samples


def _fast_table(values, label="Ux_3"):
    """One column at 1 Hz, ``Ux_3`` by default."""
    t = T0_DATENUM + np.arange(values.size) / 86400.0
    return np.column_stack([t, values]), ["TIMESTAMP", label], 1.0


def _clean_sine():
    """A unit sine never reaches 3.5 standard deviations."""
    return 3.0 + np.sin(2 * np.pi * np.arange(CLEAN_N) / SINE_PERIOD)


def _three_level():
    """1, -1, 0, 1, -1 repeated: median 0 and MAD 1 in every window, and
    still so with one sample replaced."""
    return np.tile([1.0, -1.0, 0.0, 1.0, -1.0], CLEAN_N // 5)


def _spike_info(**spike_test):
    info = {"avgPer": 30, **QCConfig().to_info()}
    info["spikeTest"].update(spike_test)
    return info


@pytest.mark.parametrize("drop_method", [False, True],
                         ids=["sigma", "method-absent"])
def test_default_sigma_removes_a_spike_inside_the_mad_limit(drop_method):
    """A +5 spike on a unit sine is 7 standard deviations (sigma = 0.71)
    but below q MAD / 0.6745 = 7.7 (MAD of the sampled sine = 0.74): the
    sigma test, also when ``spikeTest.method`` is absent, replaces it by
    the mean of its neighbours and leaves the sine as is everywhere
    else."""
    clean = _clean_sine()
    values = clean.copy()
    values[3601] += 5.0
    tbl, labels, hz = _fast_table(values)
    info = _spike_info(method="sigma")
    if drop_method:
        del info["spikeTest"]["method"]
    out, _, _ = qc_table(tbl.copy(), labels, info, dict(_DEFAULT_TEMPLATE), hz)
    expected = clean.copy()
    expected[3601] = 0.5 * (clean[3600] + clean[3602])
    # rtol: round-off of the linear interpolation
    np.testing.assert_allclose(out[:, 1], expected, rtol=1e-12)


@pytest.mark.parametrize("spike, removed", [(6.0, False), (9.0, True)])
def test_mad_limit_scales_the_mad_by_0_6745(spike, removed):
    """On the sampled unit sine q MAD / 0.6745 = 7 * 0.743 / 0.6745 = 7.7:
    a +6 spike (6.1 from the median) is kept, a +9 spike is replaced by
    the mean of its neighbours. Without the divisor the limit is 5.2 and
    the +6 spike is removed."""
    clean = _clean_sine()
    values = clean.copy()
    values[3601] += spike
    tbl, labels, hz = _fast_table(values)
    out, _, _ = qc_table(tbl.copy(), labels, _spike_info(method="mad"),
                         dict(_DEFAULT_TEMPLATE), hz)
    expected = values.copy()
    if removed:
        expected[3601] = 0.5 * (clean[3600] + clean[3602])
    # rtol: round-off of the linear interpolation
    np.testing.assert_allclose(out[:, 1], expected, rtol=1e-12)


@pytest.mark.parametrize("q", [7.0, 3.0], ids=["q7", "q3"])
@pytest.mark.parametrize("excess, removed", [(0.0, False), (1e-9, True)],
                         ids=["on-limit", "above-limit"])
def test_mad_limit_is_q_mad_over_0_6745(q, excess, removed):
    """With median 0 and MAD 1 the limit is q / 0.6745: a sample exactly
    on it lies inside the closed range of Eq. (1) and is kept, one
    1e-9 above it is replaced by the mean of its neighbours (-1)."""
    values = _three_level()
    values[3600] = q / 0.6745 * (1.0 + excess)
    tbl, labels, hz = _fast_table(values)
    out, _, _ = qc_table(tbl.copy(), labels,
                         _spike_info(method="mad", madThreshold=q),
                         dict(_DEFAULT_TEMPLATE), hz)
    expected = values.copy()
    if removed:
        expected[3600] = -1.0
    np.testing.assert_array_equal(out[:, 1], expected)


@pytest.mark.parametrize("base", [
    np.zeros(CLEAN_N),
    (np.arange(CLEAN_N) % 10 == 0).astype(float),
], ids=["constant", "ninety-percent-at-median"])
def test_mad_zero_window_keeps_a_spike_the_sigma_test_removes(base):
    """With 90% or more of a window at its median the MAD is 0 and the
    MAD test flags nothing, so a +20 spike stays; the sigma test, whose
    standard deviation the spike raises, replaces it by the mean of its
    neighbours. The ones of the second series lie 3.0 sigma from the
    mean, below the 3.5 of ``spikeDef.u``."""
    values = base.copy()
    values[3601] += 20.0
    tbl, labels, hz = _fast_table(values)
    template = dict(_DEFAULT_TEMPLATE)
    out_mad, _, _ = qc_table(tbl.copy(), labels, _spike_info(method="mad"),
                             template, hz)
    out_sigma, _, _ = qc_table(tbl.copy(), labels, _spike_info(), template, hz)
    np.testing.assert_array_equal(out_mad[:, 1], values)
    expected = base.copy()
    expected[3601] = 0.5 * (base[3600] + base[3602])
    np.testing.assert_array_equal(out_sigma[:, 1], expected)


def test_mad_ignores_nan_in_the_window():
    """A NaN in the window leaves the median 0 and the MAD 1: a +20 spike
    is replaced by the mean of its neighbours (0.5) and the NaN stays."""
    values = _three_level()
    values[3601] = 20.0
    values[3610] = np.nan
    tbl, labels, hz = _fast_table(values)
    out, _, _ = qc_table(tbl.copy(), labels, _spike_info(method="mad"),
                         dict(_DEFAULT_TEMPLATE), hz)
    expected = values.copy()
    expected[3601] = 0.5
    np.testing.assert_array_equal(out[:, 1], expected)


#: Twenty-sample block: two zeros, three each of +-1, four each of +-2,
#: one each of +-18 and of +-40; each +-40 sits between +1 and -1.
_TWO_PASS_BLOCK = np.array([1, 40, -1, 2, -2, 18, 0, 2, -2, 1,
                            -1, -40, 1, 2, -2, -18, 0, 2, -2, -1], dtype=float)


def test_mad_test_makes_one_pass():
    """With the +-40 samples in the window the median is 0 and the MAD 2,
    a limit of 20.8: the +-40 samples are replaced by 0, the mean of their
    neighbours, and the +-18 samples stay. Without the +-40 samples the
    MAD is 1.5, a limit of 15.6, which a second pass would apply to the
    +-18 samples."""
    values = np.tile(_TWO_PASS_BLOCK, CLEAN_N // _TWO_PASS_BLOCK.size)
    tbl, labels, hz = _fast_table(values)
    out, _, _ = qc_table(tbl.copy(), labels, _spike_info(method="mad"),
                         dict(_DEFAULT_TEMPLATE), hz)
    expected = values.copy()
    expected[np.abs(values) == 40.0] = 0.0
    # atol: round-off of the linear interpolation to 0
    np.testing.assert_allclose(out[:, 1], expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("label, tested", [
    ("Ux_3", True),
    ("diagnostic_3", False),
    ("IMU_Yaw_3", False),
])
def test_mad_tests_only_spike_tested_columns(label, tested):
    """A +20 spike is replaced by the mean of its neighbours (0.5) on a
    wind component; the sonic diagnostic and the IMU yaw carry no spike
    test and keep it."""
    values = _three_level()
    values[3601] = 20.0
    tbl, labels, hz = _fast_table(values, label)
    out, _, _ = qc_table(tbl.copy(), labels, _spike_info(method="mad"),
                         dict(_DEFAULT_TEMPLATE), hz)
    expected = values.copy()
    if tested:
        expected[3601] = 0.5
    np.testing.assert_array_equal(out[:, 1], expected)


def test_mad_flags_spikes_that_mask_the_sigma_test():
    """One sample in ten raised by 20 inflates the window standard
    deviation to about 6, so no spike reaches 3.5 sigma; the median and
    MAD barely move, so q = 7 catches every one, and interpolation over
    them restores the sine to within its curvature over two samples."""
    clean = _clean_sine()
    values = clean.copy()
    spikes = np.arange(5, CLEAN_N - 5, 10)
    values[spikes] += 20.0
    tbl, labels, hz = _fast_table(values)
    template = dict(_DEFAULT_TEMPLATE)
    out_sigma, _, _ = qc_table(tbl.copy(), labels, _spike_info(), template, hz)
    out_mad, _, _ = qc_table(tbl.copy(), labels, _spike_info(method="mad"),
                             template, hz)
    np.testing.assert_array_equal(out_sigma[spikes, 1], values[spikes])
    # bound: linear interpolation error of a unit sine over one sample each
    # side
    bound = 1.0 - np.cos(2 * np.pi / SINE_PERIOD)
    np.testing.assert_array_less(np.abs(out_mad[spikes, 1] - clean[spikes]),
                                 bound * 1.0001)
    kept = np.ones(CLEAN_N, dtype=bool)
    kept[spikes] = False
    np.testing.assert_array_equal(out_mad[kept, 1], values[kept])


def test_unknown_spike_method_raises():
    tbl, labels, hz = _fast_table(_clean_sine())
    with pytest.raises(ValueError, match="unknown spikeTest.method 'median'"):
        qc_table(tbl.copy(), labels, _spike_info(method="median"),
                 dict(_DEFAULT_TEMPLATE), hz)


# -- higher-moment flag -------------------------------------------------------

def test_vickers_mahrt_ranges():
    """Section 6e, p. 521: soft skewness (-1, 1) and kurtosis (2, 5), hard
    skewness (-2, 2) and kurtosis (1, 8)."""
    assert (SKEW_SOFT, KURT_SOFT, SKEW_HARD, KURT_HARD) == (
        (-1.0, 1.0), (2.0, 5.0), (-2.0, 2.0), (1.0, 8.0))


def _from_counts(counts, reps=10):
    """``counts[v]`` samples of each value ``v`` per block, each block
    followed by its mirror: the series has no linear trend, so its
    skewness and kurtosis are those of the counts."""
    block = np.repeat(list(counts), list(counts.values())).astype(float)
    return np.tile(np.concatenate([block, block[::-1]]), reps)


def _symmetric(n_zero):
    """25 each of -1 and +1 and *n_zero* zeros: skewness 0, kurtosis
    (n_zero + 50) / 50."""
    return _from_counts({-1: 25, 0: n_zero, 1: 25})


SOFT = (SKEW_SOFT, KURT_SOFT)
HARD = (SKEW_HARD, KURT_HARD)

#: Residuals +-1 to below half an ulp: skewness 0.0 and kurtosis 1.0 exact.
ON_BOUND = np.tile([1.0, -1.0, -1.0, 1.0], 5)


@pytest.mark.parametrize("x, limits, flagged", [
    # skew 0, kurt 3
    (np.random.default_rng(0).standard_normal(18000), SOFT, False),
    # raw kurt 1.82, 3 once detrended
    (np.random.default_rng(0).standard_normal(18000)
     + np.linspace(0, 40, 18000), SOFT, False),
    # skewness at the soft bounds; kurtosis 2.99 and 3.13
    (_from_counts({3: 1, 4: 7, 6: 2}), SOFT, True),        # skew 1.021
    (_from_counts({-1: 3, 0: 7, 2: 2}), SOFT, False),      # skew 0.986
    (_from_counts({-3: 1, -4: 7, -6: 2}), SOFT, True),     # skew -1.021
    (_from_counts({1: 3, 0: 7, -2: 2}), SOFT, False),      # skew -0.986
    # kurtosis at the soft bounds, skewness 0
    (_symmetric(49), SOFT, True),                          # kurt 1.98
    (_symmetric(51), SOFT, False),                         # kurt 2.02
    (_symmetric(199), SOFT, False),                        # kurt 4.98
    (_symmetric(201), SOFT, True),                         # kurt 5.02
    # skewness at the hard bounds; kurtosis 6.09 and 5.72
    (_from_counts({-2: 1, 0: 7, 6: 1}), HARD, False),      # skew 1.984
    (_from_counts({1: 6, 2: 2, 5: 1}), HARD, True),        # skew 2.024
    (_from_counts({2: 1, 0: 7, -6: 1}), HARD, False),      # skew -1.984
    (_from_counts({-1: 6, -2: 2, -5: 1}), HARD, True),     # skew -2.024
    # kurtosis at the hard bounds, skewness 0
    (ON_BOUND, HARD, True),                                # kurt 1.0
    (_symmetric(1), HARD, False),                          # kurt 1.02
    (_symmetric(349), HARD, False),                        # kurt 7.98
    (_symmetric(351), HARD, True),                         # kurt 8.02
    # a value equal to a bound counts as outside
    (ON_BOUND, ((-1.0, 1.0), (0.0, 2.0)), False),
    (ON_BOUND, ((0.0, 1.0), (0.0, 2.0)), True),
    (ON_BOUND, ((-1.0, 0.0), (0.0, 2.0)), True),
    (ON_BOUND, ((-1.0, 1.0), (0.0, 1.0)), True),
    (np.full(18000, 4.0), SOFT, False),            # zero variance
    # skew 1.021 on an offset 1e3: variance 0.81 against a mean square of
    # 1e6 is far above the round-off cut eps * 1e6 = 2.2e-10
    (1e3 + _from_counts({3: 1, 4: 7, 6: 2}), SOFT, True),
    (np.full(18000, np.nan), SOFT, False),         # no finite sample
], ids=["gaussian", "gaussian-trend",
        "skew-soft-hi-out", "skew-soft-hi-in", "skew-soft-lo-out",
        "skew-soft-lo-in", "kurt-soft-lo-out", "kurt-soft-lo-in",
        "kurt-soft-hi-in", "kurt-soft-hi-out",
        "skew-hard-hi-in", "skew-hard-hi-out", "skew-hard-lo-in",
        "skew-hard-lo-out", "kurt-hard-lo-on", "kurt-hard-lo-in",
        "kurt-hard-hi-in", "kurt-hard-hi-out",
        "on-bound-inside", "on-skew-lo", "on-skew-hi", "on-kurt-hi",
        "constant", "offset-skew-out", "all-nan"])
def test_skew_kurt_flag(x, limits, flagged):
    """Soft ranges skewness (-1, 1), kurtosis (2, 5) and hard ranges (-2, 2),
    (1, 8) of Vickers and Mahrt (1997), each bound bracketed by a series
    just inside and one just outside; a value equal to a bound is
    outside."""
    assert skew_kurt_flag(x, *limits) is flagged


@pytest.mark.parametrize("x, flagged", [
    (_from_counts({3: 1, 4: 7, 6: 2}), True),        # skew 1.021
    (_from_counts({-1: 3, 0: 7, 2: 2}), False),      # skew 0.986
    (_from_counts({-3: 1, -4: 7, -6: 2}), True),     # skew -1.021
    (_from_counts({1: 3, 0: 7, -2: 2}), False),      # skew -0.986
    (_symmetric(49), True),                          # kurt 1.98
    (_symmetric(51), False),                         # kurt 2.02
    (_symmetric(199), False),                        # kurt 4.98
    (_symmetric(201), True),                         # kurt 5.02
], ids=["skew-hi-out", "skew-hi-in", "skew-lo-out", "skew-lo-in",
        "kurt-lo-out", "kurt-lo-in", "kurt-hi-in", "kurt-hi-out"])
def test_skew_kurt_flag_defaults_are_the_soft_ranges(x, flagged):
    """Without limits the soft ranges apply: skewness (-1, 1), kurtosis
    (2, 5)."""
    assert skew_kurt_flag(x) is flagged


@pytest.mark.parametrize("x, flagged", [
    ([1.0, 5.0], False),
    # three samples detrend to residuals proportional to (1, -2, 1):
    # kurtosis 1.5, below the soft lower bound 2
    ([1.0, 5.0, 2.0], True),
], ids=["two", "three"])
def test_skew_kurt_flag_needs_three_finite_samples(x, flagged):
    """Fewer than three finite samples return False."""
    assert skew_kurt_flag(x) is flagged


def _gapped(base, fill, slope=1e3):
    """*base*, a palindrome, on a time axis with the non-finite samples
    ``fill`` (index to value) at mirror-image indices, plus a trend of
    *slope* per sample. The finite sample times stay symmetric, so the
    base has no linear trend against them and its detrended skewness and
    kurtosis are those of the counts."""
    n = len(base) + len(fill)
    free = np.ones(n, dtype=bool)
    free[list(fill)] = False
    x = slope * np.arange(n, dtype=float)
    x[free] += base
    x[list(fill)] = list(fill.values())
    return x


@pytest.mark.parametrize("x, flagged", [
    # skew 0.986 with a NaN at index 60 of 242 and an inf at its mirror 181
    (_gapped(_from_counts({-1: 3, 0: 7, 2: 2}), {60: np.nan, 181: np.inf}),
     False),
    # skew 1.021 with a NaN at the centre index 100 of 201
    (_gapped(_from_counts({3: 1, 4: 7, 6: 2}), {100: np.nan}), True),
], ids=["skew-in-nan-inf", "skew-out-nan"])
def test_skew_kurt_flag_detrends_on_the_finite_sample_times(x, flagged):
    """NaN and inf samples are dropped and the trend is fitted against the
    times of the remaining samples, gaps included."""
    assert skew_kurt_flag(x) is flagged
