"""Unit tests for pure helper functions in utespac/."""

import numpy as np
import pytest

from utespac.strfndw import strfndw
from utespac.stp_dn import stp_dn
from utespac.nandetrend import nandetrend
from utespac.consec_flag_removal import consec_flag_removal
from utespac.rh_to_spec_hum import rh_to_spec_hum
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

    def test_linear_in_rh(self):
        q50 = rh_to_spec_hum(50.0, 101.325, 293.15)
        q100 = rh_to_spec_hum(100.0, 101.325, 293.15)
        assert q100 == pytest.approx(2.0 * q50)

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

class TestDissipationRate:
    def test_structure_function_known_values(self):
        # u = [0,1,2,3]: D(1) = mean(1,1,1) = 1, D(2) = mean(4,4) = 4
        sf = calc_structure_function(np.array([0.0, 1.0, 2.0, 3.0]))
        np.testing.assert_allclose(sf, [1.0, 4.0])

    def test_finite_positive_for_turbulent_like_signal(self):
        rng = np.random.default_rng(42)
        u = np.cumsum(rng.standard_normal(2000)) * 0.01
        eps = calc_dissipation_rate(u, u_mean=2.0, dt=0.05)
        assert np.isfinite(eps)
        assert eps > 0
