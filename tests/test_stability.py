"""utespac.stability: psi_m/psi_h closed forms against the defining
integral (Foken 2008 eq. 2.84) and pinned values; branch and NaN behavior."""

import numpy as np
import pytest
from scipy.integrate import quad

from utespac.stability import psi_h, psi_m


def test_neutral_is_zero():
    assert psi_m(0.0) == 0.0
    assert psi_h(0.0) == 0.0


def test_stable_is_linear():
    assert psi_m(0.5) == pytest.approx(-3.0)
    assert psi_h(0.5) == pytest.approx(-3.9)
    assert psi_m(1.0) == pytest.approx(-6.0)
    assert psi_h(1.0) == pytest.approx(-7.8)


def test_unstable_pinned_values():
    # closed forms evaluated independently (scratch check 2026-08-26)
    assert psi_m(-1.0) == pytest.approx(1.213415, abs=1e-5)
    assert psi_h(-1.0) == pytest.approx(1.564222, abs=1e-5)
    assert psi_m(-0.1) == pytest.approx(0.325618, abs=1e-5)
    assert psi_h(-0.1) == pytest.approx(0.361482, abs=1e-5)


def test_per_source_pinned_values():
    assert psi_m(-1.0, "hoegstroem1988") == psi_m(-1.0)         # the default
    assert psi_m(-1.0, "businger1971") == pytest.approx(1.083720, abs=1e-5)
    assert psi_h(-1.0, "businger1971") == pytest.approx(1.025698, abs=1e-5)
    assert psi_m(0.5, "businger1971") == pytest.approx(-2.35)
    assert psi_h(0.5, "businger1971") == pytest.approx(-2.35)
    assert psi_m(-1.0, "dyer1974") == pytest.approx(1.116232, abs=1e-5)
    assert psi_h(-1.0, "dyer1974") == pytest.approx(1.881227, abs=1e-5)
    assert psi_m(0.5, "dyer1974") == pytest.approx(-2.5)
    assert psi_h(0.5, "dyer1974") == pytest.approx(-2.5)


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown stability source"):
        psi_m(-0.5, "kansas1968")
    with pytest.raises(ValueError, match="unknown stability source"):
        psi_h(-0.5, "kansas1968")


@pytest.mark.parametrize("source,gamma_m", [("hoegstroem1988", 19.3),
                                            ("businger1971", 15.0),
                                            ("dyer1974", 16.0)])
def test_psi_m_matches_the_defining_integral(source, gamma_m):
    # psi_m(zeta) = -int_0^zeta [1 - phi_m(s)]/s ds with
    # phi_m = (1 - gamma_m s)^(-1/4) (Foken 2008 eqs. 2.76, 2.84, A4)
    for zeta in (-2.0, -0.5, -0.05):
        num = -quad(lambda s: (1 - (1 - gamma_m * s) ** -0.25) / s, zeta, -1e-12)[0]
        assert psi_m(zeta, source) == pytest.approx(num, abs=1e-6)


def test_psi_m_is_continuous_at_neutral():
    assert psi_m(-1e-9) == pytest.approx(0.0, abs=1e-6)


def test_nan_and_shapes():
    assert np.isnan(psi_m(np.nan)) and np.isnan(psi_h(np.nan))
    z = np.array([-1.0, np.nan, 0.5])
    for f in (psi_m, psi_h):
        out = f(z)
        assert out.shape == z.shape
        assert np.isnan(out[1]) and not np.isnan(out[0]) and not np.isnan(out[2])
    assert isinstance(psi_m(-0.5), float)
