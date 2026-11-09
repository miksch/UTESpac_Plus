"""utespac.flux.spectral_correction: the Massman (2000) closed-form
spectral correction factor.

Hand-worked cases follow Eq. (9), Eq. (10), Eq. (11) and Table 2 of
Massman (2000) with the Table 1 time constants written out in the test.
"""

import math

import numpy as np
import pytest

from utespac.flux.spectral_correction import (attenuation_stable,
                                              massman2000_factor)


def test_factor_unity_without_attenuation():
    # No low-pass process (p = 0) and T_b -> inf (a, b -> inf): Eq. (11) -> 1.
    f = massman2000_factor(U=[0.5, 3.0], z_eff=4.0, zeta=[-0.5, 1.0],
                           fs=10.0, avg_period_s=np.inf)
    np.testing.assert_allclose(f, [1.0, 1.0], rtol=0, atol=1e-12)


def test_factor_increases_with_lateral_separation():
    sep = np.array([0.0, 0.1, 0.2, 0.3])
    for zeta in (-0.5, 1.0):
        f = massman2000_factor(U=2.0, z_eff=4.0, zeta=zeta, fs=10.0,
                               avg_period_s=1800.0, sep_lateral=sep,
                               path_len_sonic=0.1)
        assert np.all(np.diff(f) > 0.0)


def test_factor_increases_as_wind_decreases():
    # Open-path case (tau_sensor = 0): p is wind-independent, a and b scale
    # with U, so the high-pass loss grows as U falls.
    U = np.array([8.0, 4.0, 2.0, 1.0, 0.5])
    for zeta in (-0.5, 1.0):
        f = massman2000_factor(U=U, z_eff=4.0, zeta=zeta, fs=10.0,
                               avg_period_s=1800.0, sep_lateral=0.2,
                               path_len_sonic=0.1)
        assert np.all(np.diff(f) > 0.0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("zeta", [
    -0.5,
    -2.0,     # stable-branch n_x denominator 1 + 0.5 z/L is zero
])
def test_unstable_hand_case(zeta):
    # U = 2 m/s, z = 4 m, z/L <= 0, T_b = 1800 s, linear detrend,
    # lateral separation 0.2 m, nothing else.
    f_x = 0.085 * 2.0 / 4.0                       # n_x u / z = 0.0425 Hz
    tau_h = 1800.0 / 5.3                          # 339.62 s
    tau_b = 1800.0 / 2.8                          # 642.86 s
    tau_e = 0.2 / (1.1 * 2.0)                     # 0.090909 s, Eq. (9)
    a = (2 * math.pi * f_x * tau_h) ** 0.925      # 90.691 ** 0.925 = 64.677
    b = (2 * math.pi * f_x * tau_b) ** 0.925      # 171.67 ** 0.925 = 116.70
    p = (2 * math.pi * f_x * tau_e) ** 0.925      # 0.024276 ** 0.925
    ratio = (a * b / ((a + 1) * (b + 1))) * (a * b / ((a + p) * (b + p))) \
        * (1 / (p + 1)) * (1 - p / ((a + 1) * (a + p)))
    expected = 1 / ratio                          # 1.0578450302
    f = massman2000_factor(U=2.0, z_eff=4.0, zeta=zeta, fs=10.0,
                           avg_period_s=1800.0, sep_lateral=0.2,
                           detrend="linear")
    # Same arithmetic in another order: only rounding differs.
    assert f == pytest.approx(expected, rel=1e-12)
    # Literal rounded to 6 decimals: half a unit in the last place.
    assert f == pytest.approx(1.057845, abs=5e-7)


def test_stable_hand_case_slow_sensor():
    # U = 2 m/s, z = 4 m, z/L = 1, T_b = 1800 s, block mean only (a -> inf),
    # first-order sensor tau = 0.1 s, collocated, so tau_e = 0.1 s.
    n_x = 2.0 - 1.915 / (1 + 0.5 * 1.0)           # 0.72333
    f_x = n_x * 2.0 / 4.0                         # 0.36167 Hz
    b = 2 * math.pi * f_x * 1800.0 / 2.8          # 1460.8
    p = 2 * math.pi * f_x * 0.1                   # 0.22724
    # a -> inf: ab/((a+1)(b+1)) -> b/(b+1), ab/((a+p)(b+p)) -> b/(b+p),
    # 1 - p/((a+1)(a+p)) -> 1; Table 2 slow-sensor term (1+0.9p)/(1+p).
    ratio = b / (b + 1) * b / (b + p) / (1 + p) * (1 + 0.9 * p) / (1 + p)
    expected = 1 / ratio                          # 1.2514453463
    f = massman2000_factor(U=2.0, z_eff=4.0, zeta=1.0, fs=10.0,
                           avg_period_s=1800.0, tau_sensor=0.1,
                           detrend="constant")
    # Same arithmetic in another order: only rounding differs.
    assert f == pytest.approx(expected, rel=1e-12)
    # Literal rounded to 6 decimals: half a unit in the last place.
    assert f == pytest.approx(1.251445, abs=5e-7)


def _table2_ratio(a, b, p, alpha=1.0, slow_sensor=False):
    """Table 2 of Massman (2000), written out independently of the module."""
    a, b, p = a ** alpha, b ** alpha, p ** alpha
    ratio = (a * b / ((a + 1) * (b + 1))) * (a * b / ((a + p) * (b + p))) \
        * (1 / (p + 1)) * (1 - p / ((a + 1) * (a + p)))
    if slow_sensor:
        ratio *= (1 + 0.9 * p) / (1 + p)
    return ratio


def _closed_path_taus(U, tau_sensor):
    # Table 1 with lateral = longitudinal = sonic path = 0.15 m, first-order
    # sensor tau_sensor: sensor, l_lat/(1.1 u),
    # (4/3)[(l_lon/u)((1/2)(l_lon/u) + tau_1)]^(1/2), l_w/(8.4 u).
    t_lon = 0.15 / U
    return [tau_sensor, 0.15 / (1.1 * U),
            (4 / 3) * math.sqrt(t_lon * (0.5 * t_lon + tau_sensor)),
            0.15 / (8.4 * U)]


@pytest.mark.parametrize("zeta, tau_sensor, literal", [
    (1.0, 0.1, 1.654181),     # stable, slow-sensor term applied
    (-0.5, 0.1, 1.097525),    # unstable, alpha = 0.925
    (1.0, 0.05, 1.374222),    # stable, tau below 0.1 s: no slow-sensor term
    (1.0, 0.099, 1.588316),   # stable, just below 0.1 s: no slow-sensor term
    (0.05, 0.1, 1.124072),    # weakly stable: stable row, slow-sensor term
])
def test_closed_path_recursive_filter_hand_case(zeta, tau_sensor, literal):
    # z = 5 m, U = 5 m/s, T_b = 1800 s, recursive high-pass tau_r = 450 s.
    U, z = 5.0, 5.0
    n_x = 0.085 if zeta <= 0 else 2.0 - 1.915 / (1 + 0.5 * zeta)
    omega_x = 2 * math.pi * n_x * U / z
    tau_e = math.sqrt(sum(t ** 2 for t in _closed_path_taus(U, tau_sensor)))
    expected = 1 / _table2_ratio(omega_x * 450.0, omega_x * 1800.0 / 2.8,
                                 omega_x * tau_e,
                                 alpha=0.925 if zeta <= 0 else 1.0,
                                 slow_sensor=zeta > 0 and tau_sensor >= 0.1)
    f = massman2000_factor(U=U, z_eff=z, zeta=zeta, fs=10.0,
                           avg_period_s=1800.0, tau_sensor=tau_sensor,
                           sep_lateral=0.15, sep_longitudinal=0.15,
                           path_len_sonic=0.15, detrend=450.0)
    # Same arithmetic in another order: only rounding differs.
    assert f == pytest.approx(expected, rel=1e-12)
    # Literal rounded to 6 decimals: half a unit in the last place.
    assert f == pytest.approx(literal, abs=5e-7)


def test_open_path_all_terms_hand_case():
    # z = 5 m, U = 2 m/s, z/L = 1.5, T_b = 1800 s, linear detrend, fs = 10 Hz,
    # open path (no sensor time constant), anti-noise filter on.
    U, z, zeta = 2.0, 5.0, 1.5
    taus = [0.30 / (1.1 * U),        # lateral separation
            0.1 / (1.05 * U),        # longitudinal separation, no sensor
            0.15 / (8.4 * U),        # sonic line averaging
            0.20 / (4.0 * U),        # scalar line averaging
            1 / (math.pi * 10.0)]    # anti-noise filter at Nyquist
    tau_e = math.sqrt(sum(t ** 2 for t in taus))  # Eq. (9)
    omega_x = 2 * math.pi * (2.0 - 1.915 / (1 + 0.5 * zeta)) * U / z
    expected = 1 / _table2_ratio(omega_x * 1800.0 / 5.3,
                                 omega_x * 1800.0 / 2.8, omega_x * tau_e)
    f = massman2000_factor(U=U, z_eff=z, zeta=zeta, fs=10.0,
                           avg_period_s=1800.0, sep_lateral=0.30,
                           sep_longitudinal=0.1, path_len_sonic=0.15,
                           path_len_scalar=0.20, detrend="linear",
                           anti_noise_filter=True)
    # Same arithmetic in another order: only rounding differs.
    assert f == pytest.approx(expected, rel=1e-12)
    # Literal rounded to 6 decimals: half a unit in the last place.
    assert f == pytest.approx(1.345621, abs=5e-7)


def test_neutral_hand_case_uses_unstable_row():
    # z/L = 0 belongs to the unstable row of Table 2 (z/L <= 0): alpha = 0.925,
    # n_x = 0.085. U = 3 m/s, z = 3 m, T_b = 900 s, block mean only (a -> inf),
    # longitudinal separation 0.2 m without a sensor, anti-noise at fs = 20 Hz.
    U, z, alpha = 3.0, 3.0, 0.925
    omega_x = 2 * math.pi * 0.085 * U / z
    tau_e = math.sqrt((0.2 / (1.05 * U)) ** 2 + (1 / (math.pi * 20.0)) ** 2)
    b = (omega_x * 900.0 / 2.8) ** alpha
    p = (omega_x * tau_e) ** alpha
    # a -> inf: Eq. (11) reduces to b/(b+1) * b/(b+p) / (1+p).
    expected = 1 / (b / (b + 1) * b / (b + p) / (1 + p))   # 1.0543156613
    f = massman2000_factor(U=U, z_eff=z, zeta=0.0, fs=20.0,
                           avg_period_s=900.0, sep_longitudinal=0.2,
                           detrend="constant", anti_noise_filter=True)
    # Same arithmetic in another order: only rounding differs.
    assert f == pytest.approx(expected, rel=1e-12)
    # Literal rounded to 6 decimals: half a unit in the last place.
    assert f == pytest.approx(1.054316, abs=5e-7)


def test_shape_preserved():
    U = np.full((3, 4), 2.0)
    zeta = np.linspace(-1.0, 1.0, 12).reshape(3, 4)
    f = massman2000_factor(U=U, z_eff=4.0, zeta=zeta, fs=10.0,
                           avg_period_s=1800.0, sep_lateral=0.2)
    assert f.shape == (3, 4)
    ref = massman2000_factor(U=2.0, z_eff=4.0, zeta=zeta[1, 2], fs=10.0,
                             avg_period_s=1800.0, sep_lateral=0.2)
    assert f[1, 2] == ref


@pytest.mark.parametrize("name", ["U", "z_eff", "avg_period_s"])
@pytest.mark.parametrize("zeta, kwargs, expected", [
    (-0.5, {"sep_lateral": 0.2}, 1.057845),
    (1.0, {"tau_sensor": 0.1, "detrend": "constant"}, 1.251445),
])
def test_nonpositive_wind_height_or_period_is_nan(name, zeta, kwargs,
                                                  expected):
    args = {"U": 2.0, "z_eff": 4.0, "avg_period_s": 1800.0}
    args[name] = [-1.0, 0.0, args[name]]
    f = massman2000_factor(zeta=zeta, fs=10.0, **args, **kwargs)
    assert np.isnan(f[0]) and np.isnan(f[1])
    # Third entry is the matching hand case above; literal rounded to 6
    # decimals, half a unit in the last place.
    assert f[2] == pytest.approx(expected, abs=5e-7)


_CASE_UNSTABLE = {"U": 2.0, "z_eff": 4.0, "zeta": -0.5, "fs": 10.0,
                  "avg_period_s": 1800.0}
_CASE_STABLE = dict(_CASE_UNSTABLE, zeta=1.0, detrend="constant")
_CASE_NEUTRAL = {"U": 3.0, "z_eff": 3.0, "zeta": 0.0, "fs": 20.0,
                 "avg_period_s": 900.0, "detrend": "constant",
                 "anti_noise_filter": True}
_CASE_OPEN = {"U": 2.0, "z_eff": 5.0, "zeta": 1.5, "fs": 10.0,
              "avg_period_s": 1800.0, "sep_lateral": 0.30,
              "sep_longitudinal": 0.1, "detrend": "linear",
              "anti_noise_filter": True}


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("case, name, valid, expected", [
    (_CASE_UNSTABLE, "sep_lateral", 0.2, 1.057845),
    (_CASE_STABLE, "tau_sensor", 0.1, 1.251445),
    (_CASE_NEUTRAL, "sep_longitudinal", 0.2, 1.054316),
    # Negative tau_sensor puts a negative radicand in the Table 1 term.
    (dict(_CASE_NEUTRAL, sep_longitudinal=0.2), "tau_sensor", 0.0,
     1.054316),
    (dict(_CASE_OPEN, path_len_scalar=0.20), "path_len_sonic", 0.15,
     1.345621),
    (dict(_CASE_OPEN, path_len_sonic=0.15), "path_len_scalar", 0.20,
     1.345621),
])
def test_negative_sensor_separation_or_path_is_nan(case, name, valid,
                                                   expected):
    """Negative time constant, separation or path length gives NaN."""
    f = massman2000_factor(**case, **{name: [-0.5, valid]})
    assert np.isnan(f[0])
    # Second entry is the matching hand case above; literal rounded to 6
    # decimals, half a unit in the last place.
    assert f[1] == pytest.approx(expected, abs=5e-7)


@pytest.mark.filterwarnings("error")
def test_nonpositive_fs_with_anti_noise_filter_is_nan():
    """Non-positive fs with the anti-noise filter gives NaN, no warning."""
    case = dict(_CASE_NEUTRAL, fs=[-20.0, 0.0, 20.0], sep_longitudinal=0.2)
    f = massman2000_factor(**case)
    assert np.isnan(f[0]) and np.isnan(f[1])
    # Third entry is the neutral hand case above; literal rounded to 6
    # decimals, half a unit in the last place.
    assert f[2] == pytest.approx(1.054316, abs=5e-7)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("a, limit", [
    (np.inf, "no_trend"),     # no trend removal
    (0.0, "zero"),            # zero high-pass time constant: no flux left
])
def test_attenuation_stable_default_hand_case(a, limit):
    """Table 2 stable row, default without the slow-sensor term."""
    # b and p of the stable hand case above.
    f_x = (2.0 - 1.915 / (1 + 0.5 * 1.0)) * 2.0 / 4.0
    b = 2 * math.pi * f_x * 1800.0 / 2.8
    p = 2 * math.pi * f_x * 0.1
    if limit == "no_trend":
        # a -> inf in Eq. (11): b/(b+1) * b/(b+p) / (1+p).
        expected = b / (b + 1) * b / (b + p) / (1 + p)
    else:
        # a = 0: ab/((a+1)(b+1)) = 0 in Eq. (11).
        expected = 0.0
    ratio = attenuation_stable(a, b, p)
    # Same arithmetic in another order: only rounding differs.
    assert ratio == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("detrend, match", [
    ("quadratic", "unknown detrend"),
    (0.0, "detrend time constant must be positive"),
    (-450.0, "detrend time constant must be positive"),
    (True, "got bool"),
    (False, "got bool"),
    ("l", "unknown detrend"),
    ("c", "unknown detrend"),
    ([450.0], "detrend time constant must be positive"),
    (None, "detrend time constant must be positive"),
])
def test_invalid_detrend_raises(detrend, match):
    with pytest.raises(ValueError, match=match):
        massman2000_factor(U=2.0, z_eff=4.0, zeta=0.0, fs=10.0,
                           avg_period_s=1800.0, detrend=detrend)
