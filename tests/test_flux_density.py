"""utespac.flux.density: the air state, the Webb et al. (1980) density
correction, latent heat and the methane mole fraction.

The density correction is checked against cases whose true flux is zero by
construction.
"""

import numpy as np
import pytest

from utespac.flux.density import (CP_DRY, MU_WPL, R_DRY, R_GAS_J_MOL_K, R_VAP,
                                  air_state, ch4_mole_fraction_ppm,
                                  latent_heat_vaporisation, wpl_ch4_flux,
                                  wpl_vapour_flux)
from utespac.sonic_temperature import SONIC_HUMIDITY_COEFF


def _cov(a, b):
    """Covariance of two complete series about their own means."""
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def test_constants():
    assert R_DRY == 287.058
    assert R_VAP == 461.495
    assert MU_WPL == 1.6077
    assert CP_DRY == 1004.67
    assert R_GAS_J_MOL_K == 8.314462618


# ── the density correction ───────────────────────────────────────────────────

def _temperature_only_case(vapour_mole_fraction):
    """A parcel whose only fluctuation is temperature, at constant pressure.

    Every mole fraction is held constant, so no gas is transported: total
    molar density and every species density scale as 1/T. The true methane
    flux is zero, and the Webb et al. (1980) correction must return zero
    whatever the raw density covariance is.
    """
    rng = np.random.default_rng(11)
    n = 20000
    p = 100_000.0                       # [Pa]
    t = 290.0 + 0.4 * rng.standard_normal(n)
    w = 0.3 * rng.standard_normal(n)
    w = w - w.mean()

    rho_v = p * vapour_mole_fraction / (R_VAP * t)              # [kg/m3]
    rho_d = p * (1.0 - vapour_mole_fraction) / (R_DRY * t)      # [kg/m3]
    rho_c = 2.0e-6 * p / (R_GAS_J_MOL_K * t) * 1000.0           # [mmol/m3] at 2 ppm

    return {
        "w_rhoc": _cov(w, rho_c),
        "w_rhov": _cov(w, rho_v),
        "w_temp": _cov(w, t),
        "rho_c": float(rho_c.mean()),
        "rho_v": float(rho_v.mean()),
        "rho_d": float(rho_d.mean()),
        "ta_k": float(t.mean()),
    }


@pytest.mark.parametrize("vapour_mole_fraction", [0.0, 0.012])
def test_wpl_zero_for_temperature_only_fluctuation(vapour_mole_fraction):
    case = _temperature_only_case(vapour_mole_fraction)
    flux = wpl_ch4_flux(**case)
    raw = abs(case["w_rhoc"])
    assert raw > 0, "the synthetic case must carry a non-zero raw covariance"
    assert abs(flux) < 1e-3 * raw


def test_wpl_vapour_term_enters_with_the_expected_sign_and_size():
    """The vapour term is mu (rho_c / rho_d) w'rho_v', added as written."""
    base = dict(w_rhoc=0.0, w_rhov=0.0, w_temp=0.0, rho_c=0.089,
                rho_v=6.0e-3, rho_d=1.20, ta_k=290.0)
    with_vapour = dict(base, w_rhov=1.0e-4)
    delta = wpl_ch4_flux(**with_vapour) - wpl_ch4_flux(**base)
    assert delta == pytest.approx(1.6077 * (0.089 / 1.20) * 1.0e-4, rel=1e-12)


def test_wpl_heat_term_carries_the_vapour_weighting():
    """The heat term is (1 + mu sigma)(rho_c / T) w'T'."""
    base = dict(w_rhoc=0.0, w_rhov=0.0, w_temp=0.0, rho_c=0.089,
                rho_v=6.0e-3, rho_d=1.20, ta_k=290.0)
    with_heat = dict(base, w_temp=0.02)
    delta = wpl_ch4_flux(**with_heat) - wpl_ch4_flux(**base)
    sigma = 6.0e-3 / 1.20
    assert delta == pytest.approx((1 + 1.6077 * sigma) * (0.089 / 290.0) * 0.02,
                                  rel=1e-12)


def test_wpl_vapour_flux_is_zero_for_temperature_only_fluctuation():
    """With every mole fraction constant no vapour is transported either."""
    case = _temperature_only_case(0.012)
    evap = wpl_vapour_flux(case["w_rhov"], case["w_temp"], case["rho_v"],
                           case["rho_d"], case["ta_k"])
    assert abs(case["w_rhov"]) > 0
    assert abs(evap) < 1e-3 * abs(case["w_rhov"])


def test_wpl_vapour_flux_scales_the_raw_covariance_by_one_plus_mu_sigma():
    evap = wpl_vapour_flux(1.0e-4, 0.0, 6.0e-3, 1.20, 290.0)
    assert evap == pytest.approx((1 + 1.6077 * 6.0e-3 / 1.20) * 1.0e-4, rel=1e-12)


# ── air state ────────────────────────────────────────────────────────────────

def test_air_state_partitions_the_measured_pressure():
    """Dry and vapour partial pressures add back to the measured total."""
    ta_k, rho_v, rho_d, q = air_state(np.array([18.0, 12.0]),
                                      np.array([8.0, 5.0]),
                                      np.array([100.2, 100.2]))
    total = rho_d * R_DRY * ta_k + rho_v * R_VAP * ta_k
    assert total == pytest.approx(np.array([100_200.0, 100_200.0]), rel=1e-9)
    # the sonic temperature exceeds the air temperature by 0.51 q T
    assert np.all(ta_k < np.array([18.0, 12.0]) + 273.15)
    assert np.all((q > 0) & (q < 0.05))


def test_air_state_satisfies_the_schotanus_relation():
    ts_c = np.array([25.0, 5.0])
    ta_k, rho_v, rho_d, q = air_state(ts_c, np.array([15.0, 4.0]),
                                      np.array([101.0, 99.0]), n_iter=10)
    assert ta_k * (1.0 + SONIC_HUMIDITY_COEFF * q) == pytest.approx(ts_c + 273.15,
                                                                    rel=1e-10)
    assert rho_v == pytest.approx(np.array([0.015, 0.004]))


def test_air_state_of_dry_air_leaves_the_sonic_temperature_unchanged():
    ta_k, rho_v, rho_d, q = air_state(20.0, 0.0, 100.0)
    assert ta_k == pytest.approx(293.15)
    assert q == 0.0
    assert rho_d == pytest.approx(100_000.0 / (R_DRY * 293.15))


# ── latent heat and mole fraction ────────────────────────────────────────────

def test_latent_heat_at_the_reference_points():
    lv = latent_heat_vaporisation(np.array([273.15, 293.15]))
    assert lv == pytest.approx(np.array([2.5008e6, 2.5008e6 - 20 * 2.36e3]))


def test_mole_fraction_matches_a_hand_value_and_propagates_nan():
    """0.08 mmol m-3 at 300 K and 101.325 kPa is about 1.970 umol mol-1."""
    x = ch4_mole_fraction_ppm(np.array([0.08, np.nan, 0.08]),
                              np.array([101.325, 101.325, np.nan]),
                              np.array([300.0 - 273.15] * 3))
    assert x[0] == pytest.approx(1.9695, abs=1e-3)
    assert np.isnan(x[1]) and np.isnan(x[2])
