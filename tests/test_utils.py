"""Unit tests for pure helper functions in utespac/."""

import numpy as np
import pytest

from utespac.strfndw import strfndw
from utespac.stp_dn import stp_dn
from utespac.nandetrend import nandetrend
from utespac.consec_flag_removal import consec_flag_removal
from utespac.rh_to_spec_hum import rh_to_spec_hum, sat_vapor_pressure
from utespac.get_virtual_pot_temp import get_virtual_pot_temp
from utespac.find_delta_time import find_delta_time
from utespac.campbell_date import (
    MATLAB_EPOCH,
    datetime_to_matlab_datenum,
    matlab_datenum_to_datetime,
    campbell_date_to_serial_date,
    serial_date_to_campbell_date,
)
from utespac.calc_dissipation_rate import (
    calc_structure_function,
    calc_dissipation_rate,
)
from utespac.calc_snsp_angle import calc_snsp_angle
from utespac.sonic_temperature import (
    SONIC_HUMIDITY_COEFF,
    air_temperature_from_sonic,
    air_temperature_perturbation,
)

from datetime import datetime


# ── strfndw ──────────────────────────────────────────────────────────────────

class TestStrfndw:
    def test_star_wildcard(self):
        names = ["Ux_10.5", "Uy_10.5", "Ux_4.42", "T_Sonic_10.5"]
        assert strfndw(names, "Ux*") == [0, 2]

    def test_question_mark_matches_single_char(self):
        assert strfndw(["ab", "aXb", "aXYb"], "a?b") == [1]

    def test_case_insensitive(self):
        assert strfndw(["ABC", "abc"], "abc") == [0, 1]

    def test_literal_dot_not_wildcard(self):
        assert strfndw(["a.b", "aXb"], "a.b") == [0]

    def test_no_match(self):
        assert strfndw(["foo", "bar"], "baz*") == []

    def test_full_match_required(self):
        # pattern must span the whole string, like MATLAB strfndw
        assert strfndw(["prefix_Ux", "Ux"], "Ux") == [1]


# ── stp_dn ───────────────────────────────────────────────────────────────────

class TestStpDn:
    def test_block_mean(self):
        data = np.array([[1.0], [3.0], [5.0], [7.0]])
        out = stp_dn(data, 2)
        np.testing.assert_allclose(out, [[2.0], [6.0]])

    def test_remainder_rows_dropped(self):
        data = np.arange(5, dtype=float).reshape(-1, 1)
        out = stp_dn(data, 2)
        assert out.shape == (2, 1)
        np.testing.assert_allclose(out.ravel(), [0.5, 2.5])

    def test_nan_ignored_within_block(self):
        data = np.array([[1.0], [np.nan], [4.0], [6.0]])
        out = stp_dn(data, 2)
        np.testing.assert_allclose(out.ravel(), [1.0, 5.0])

    def test_1d_input(self):
        out = stp_dn(np.array([2.0, 4.0]), 2)
        assert out.shape == (1, 1)
        assert out[0, 0] == 3.0

    def test_multicolumn(self):
        data = np.array([[1.0, 10.0], [3.0, 30.0]])
        out = stp_dn(data, 2)
        np.testing.assert_allclose(out, [[2.0, 20.0]])


# ── nandetrend ───────────────────────────────────────────────────────────────

class TestNandetrend:
    def test_linear_ramp_removed(self):
        x = 2.0 * np.arange(50) + 3.0
        np.testing.assert_allclose(nandetrend(x), 0.0, atol=1e-10)

    def test_nan_positions_preserved(self):
        x = np.arange(10, dtype=float)
        x[3] = np.nan
        y = nandetrend(x)
        assert np.isnan(y[3])
        assert np.isfinite(np.delete(y, 3)).all()

    def test_constant_mode_removes_mean(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(nandetrend(x, order=0), [-1.0, 0.0, 1.0])

    def test_mostly_nan_returns_all_nan(self):
        x = np.full(100, np.nan)
        x[:5] = 1.0
        assert np.isnan(nandetrend(x)).all()


# ── consec_flag_removal ──────────────────────────────────────────────────────

class TestConsecFlagRemoval:
    def test_short_run_kept(self):
        flag = np.array([[False], [True], [False], [False]])
        out = consec_flag_removal(flag, 1)
        np.testing.assert_array_equal(out, flag)

    def test_long_run_cleared(self):
        flag = np.array([[True], [True], [False], [True]])
        out = consec_flag_removal(flag, 1)
        np.testing.assert_array_equal(out.ravel(), [False, False, False, True])

    def test_run_equal_to_limit_kept(self):
        flag = np.array([[True], [True], [False], [False]])
        out = consec_flag_removal(flag, 2)
        np.testing.assert_array_equal(out, flag)

    def test_columns_independent(self):
        flag = np.array([[True, False],
                         [True, True],
                         [False, False]])
        out = consec_flag_removal(flag, 1)
        np.testing.assert_array_equal(out[:, 0], [False, False, False])
        np.testing.assert_array_equal(out[:, 1], [False, True, False])

    def test_consec_rows_geq_n_is_noop(self):
        flag = np.array([[True], [True]])
        out = consec_flag_removal(flag, 5)
        np.testing.assert_array_equal(out, flag)


# ── rh_to_spec_hum ───────────────────────────────────────────────────────────

class TestRhToSpecHum:
    def test_known_conditions(self):
        # 25 °C, 60 % RH, sea-level pressure → q ≈ 0.0119 kg/kg
        q = rh_to_spec_hum(60.0, 101.325, 298.15)
        assert q == pytest.approx(0.0119, rel=0.02)

    def test_nearly_linear_in_rh(self):
        # q = 0.622 e / (P - 0.378 e): linear in e up to the moist denominator
        q50 = rh_to_spec_hum(50.0, 101.325, 293.15)
        q100 = rh_to_spec_hum(100.0, 101.325, 293.15)
        assert q100 == pytest.approx(2.0 * q50, rel=0.01)
        assert q100 > 2.0 * q50

    def test_increases_with_temperature(self):
        q_cold = rh_to_spec_hum(50.0, 101.325, 283.15)
        q_warm = rh_to_spec_hum(50.0, 101.325, 303.15)
        assert q_warm > q_cold


# ── campbell_date ────────────────────────────────────────────────────────────

class TestCampbellDate:
    def test_epoch_constant(self):
        assert datetime_to_matlab_datenum(datetime(1970, 1, 1)) == MATLAB_EPOCH

    def test_datenum_roundtrip(self):
        # float-day datenums carry ~µs quantization error; require < 1 ms
        dt = datetime(2023, 7, 6, 12, 30, 15)
        back = matlab_datenum_to_datetime(datetime_to_matlab_datenum(dt))
        assert abs((back - dt).total_seconds()) < 1e-3

    def test_campbell_to_serial_known_value(self):
        # 2023-07-06 12:30:15 → DOY 187, HHMM 1230
        mat = np.array([[2023, 187, 1230, 15.0]])
        serial = campbell_date_to_serial_date(mat)
        expected = datetime_to_matlab_datenum(datetime(2023, 7, 6, 12, 30, 15))
        assert serial[0] == pytest.approx(expected, abs=1e-9)

    def test_serial_to_campbell_roundtrip(self):
        mat = np.array([
            [2023, 187, 1230, 15.0],
            [2024, 60, 0, 0.0],      # leap year: DOY 60 = Feb 29
            [2025, 1, 2359, 59.5],
        ])
        serial = campbell_date_to_serial_date(mat)
        back = serial_date_to_campbell_date(serial)
        np.testing.assert_allclose(back, mat, atol=0.01)


# ── calc_dissipation_rate ────────────────────────────────────────────────────

def _kolmogorov_series(eps, u_mean, dt, n, seed, C1=0.49, L_max=100.0):
    """Synthetic streamwise series with E11(k) = C1 eps^(2/3) k^(-5/3).

    Fourier synthesis with random phases on a spatial grid dx = u_mean*dt
    (Taylor), spectrum zeroed below 2*pi/L_max so the variance stays finite.
    Returns the series and a function D(r) giving the structure function
    implied by the discrete spectrum (D = 2 sum E dk (1 - cos kr)), which
    differs from the continuum C2 (eps r)^(2/3) by the Nyquist truncation.
    """
    rng = np.random.default_rng(seed)
    dx = u_mean * dt
    k = 2 * np.pi * np.fft.rfftfreq(n, d=dx)           # rad/m
    E = np.zeros_like(k)
    band = k >= 2 * np.pi / L_max
    E[band] = C1 * eps ** (2.0 / 3.0) * k[band] ** (-5.0 / 3.0)
    dk = k[1] - k[0]
    amp = np.sqrt(E * dk / 2.0) * n                     # one-sided E -> rfft amplitude
    phase = np.exp(2j * np.pi * rng.random(len(k)))
    u = np.fft.irfft(amp * phase, n=n)

    def D(r):
        return np.array([2.0 * np.sum(E * dk * (1.0 - np.cos(k * ri))) for ri in r])

    return u, D


class TestDissipationRate:
    def test_structure_function_known_values(self):
        # u = [0,1,2,3]: D(1) = mean(1,1,1) = 1, D(2) = mean(4,4) = 4
        sf = calc_structure_function(np.array([0.0, 1.0, 2.0, 3.0]))
        np.testing.assert_allclose(sf, [1.0, 4.0])

    def test_structure_function_max_lag_and_nan(self):
        u = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
        sf = calc_structure_function(u, max_lag=2)
        assert sf.shape == (2,)
        np.testing.assert_allclose(sf, [1.0, 4.0])   # NaN pairs skipped

    @pytest.mark.parametrize("eps", [1e-3, 1e-1])
    def test_recovers_synthetic_kolmogorov_epsilon(self, eps):
        # D_LL = 2*C1*I*(eps r)^(2/3) with I = int x^(-5/3)(1-cos x)dx = 2.01,
        # i.e. C2 = 4.02*C1 = 1.97 for C1 = 0.49 (estimator uses C2 = 2.0).
        u_mean, dt = 2.0, 0.05
        u, D = _kolmogorov_series(eps, u_mean, dt, n=2 ** 17, seed=1)
        est = calc_dissipation_rate(u, u_mean, dt)
        # exact: the eps implied by the synthetic signal's own discrete D_LL
        r = u_mean * np.arange(2, 41) * dt             # lags in the 0.1-2 s window
        eps_disc = np.mean(D(r) / (2.0 * r ** (2.0 / 3.0))) ** 1.5
        assert est == pytest.approx(eps_disc, rel=0.01)
        # nominal, loose: Nyquist truncation of the synthesis costs ~13 %
        assert est == pytest.approx(eps, rel=0.2)

    def test_window_insensitive_inside_subrange(self):
        u, _ = _kolmogorov_series(1e-2, u_mean=3.0, dt=0.05, n=2 ** 17, seed=2)
        a = calc_dissipation_rate(u, 3.0, 0.05, lag_window=(0.1, 1.0))
        b = calc_dissipation_rate(u, 3.0, 0.05, lag_window=(0.5, 3.0))
        assert a == pytest.approx(b, rel=0.2)

    def test_scaling_with_amplitude(self):
        u, _ = _kolmogorov_series(1e-2, u_mean=2.0, dt=0.05, n=2 ** 15, seed=3)
        e1 = calc_dissipation_rate(u, 2.0, 0.05)
        e2 = calc_dissipation_rate(2.0 * u, 2.0, 0.05)   # D_LL x4 -> eps x8
        assert e2 == pytest.approx(8.0 * e1, rel=1e-9)

    def test_nan_cases(self):
        u = np.random.default_rng(0).standard_normal(100)
        assert np.isnan(calc_dissipation_rate(u, u_mean=0.0, dt=0.05))
        assert np.isnan(calc_dissipation_rate(u, u_mean=2.0, dt=0.05,
                                              lag_window=(10.0, 20.0)))
        assert np.isnan(calc_dissipation_rate(np.full(100, np.nan), 2.0, 0.05))

    def test_finite_positive_for_turbulent_like_signal(self):
        rng = np.random.default_rng(42)
        u = np.cumsum(rng.standard_normal(2000)) * 0.01
        eps = calc_dissipation_rate(u, u_mean=2.0, dt=0.05)
        assert np.isfinite(eps)
        assert eps > 0


# ── calc_snsp_angle ──────────────────────────────────────────────────────────

class TestSnspAngle:
    def test_flat_site_gives_zero(self):
        a1, a2 = calc_snsp_angle(phi=123.0, alpha=0.0, downslope_aspect=30.0)
        assert a1 == 0.0 and a2 == 0.0

    def test_wind_along_fall_line(self):
        # wind from the downslope direction: full slope angle on the along
        # component, none across
        a1, a2 = calc_snsp_angle(phi=30.0, alpha=8.2, downslope_aspect=30.0)
        assert a1 == pytest.approx(8.2)
        assert a2 == pytest.approx(0.0, abs=1e-9)

    def test_aspect_shifts_reference(self):
        # the decomposition depends only on phi - downslope_aspect
        assert calc_snsp_angle(120.0, 8.2, 30.0) == pytest.approx(
            calc_snsp_angle(300.0, 8.2, 210.0))


# ── saturation vapour pressure / humidity paths ──────────────────────────────

class TestSatVaporPressure:
    def test_stull_reference_points(self):
        # Stull 1988 eq. 7.5.2d: 0.6112 kPa at 273.16 K; ~2.34 kPa at 20 C
        assert sat_vapor_pressure(273.16) == pytest.approx(0.6112)
        assert sat_vapor_pressure(293.15) == pytest.approx(2.34, rel=0.01)

    def test_rh_to_spec_hum_matches_virtual_pot_temp_path(self):
        # both humidity paths must use the same e_sat and moist denominator
        T, RH, P = 293.15, 60.0, 90.0
        q = rh_to_spec_hum(RH, P, T)
        _, r, _, _, _ = get_virtual_pot_temp(0.0, 0.0, T, RH, P_air=P,
                                             use_p_elevation=False)
        r = r / 1000.0                       # g/kg -> kg/kg mixing ratio
        assert q == pytest.approx(r / (1.0 + r), rel=1e-4)   # 621.97 vs 0.622


# ── find_delta_time ──────────────────────────────────────────────────────────

class TestFindDeltaTime:
    def test_fractions_over_valid_samples(self):
        w = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
        u = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, -1.0])   # all downgradient (w'u' < 0)
        full = find_delta_time(w, u)                        # 4 ejections, 2 sweeps of 6
        assert full == pytest.approx(4 / 6 - 2 / 6)
        w2 = np.concatenate([w, [np.nan, np.nan]])
        u2 = np.concatenate([u, [1.0, 1.0]])
        assert find_delta_time(w2, u2) == pytest.approx(full)   # NaNs do not deflate


# ── nandetrend interior gaps ─────────────────────────────────────────────────

def test_nandetrend_fits_on_true_index():
    # exact line with an interior gap: a true-index fit removes it exactly
    x = 0.5 * np.arange(20.0) + 3.0
    x[5:12] = np.nan
    y = nandetrend(x)
    np.testing.assert_allclose(y[~np.isnan(x)], 0.0, atol=1e-12)
    assert np.all(np.isnan(y[5:12]))


# ── sonic temperature (Schotanus et al. 1983) ────────────────────────────────

class TestSonicTemperature:
    def test_coefficient_is_schotanus_not_virtual(self):
        assert SONIC_HUMIDITY_COEFF == 0.51

    def test_mean_relation_inverts_eq5(self):
        T, q = 293.15, 0.012
        Ts = T * (1.0 + 0.51 * q)
        assert air_temperature_from_sonic(Ts, q) == pytest.approx(T)

    def test_flux_relation_eq8(self):
        # w'T' = w'Ts' - 0.51 T w'q' holds sample-wise, hence for covariances
        rng = np.random.default_rng(0)
        w = rng.standard_normal(5000)
        TsP = 0.3 * w + rng.standard_normal(5000) * 0.1
        qP = 2e-4 * w + rng.standard_normal(5000) * 1e-4
        T = 295.0
        TP = air_temperature_perturbation(TsP, qP, T)
        wT = np.mean(w * TP)
        assert wT == pytest.approx(np.mean(w * TsP) - 0.51 * T * np.mean(w * qP), rel=1e-12)
        assert wT < np.mean(w * TsP)      # upward moisture flux inflates w'Ts'
