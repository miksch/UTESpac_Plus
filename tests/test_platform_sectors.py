"""Sector bookkeeping of a sectorwise planar fit, direction windows and u*.

Covers :mod:`utespac.rotation` (``sector_keys``, ``sector_of``,
``sector_label``, ``sector_name``, ``sector_edges_from_histogram``,
``fit_planar``, ``sector_fit``, ``bootstrap_tilt``) and :mod:`utespac.wind_stats`
(``in_direction_window``, ``direction_window_fraction``, ``compass_label``,
``ustar_from_rotated``, ``ustar_double_rotation``) on synthetic data.
"""

import numpy as np
import pytest

from utespac.rotation import (PlanarFit, bootstrap_tilt, fit_planar, fit_sectors,
                              sector_edges_from_histogram, sector_fit, sector_keys,
                              sector_label, sector_mask, sector_name, sector_of,
                              yaw_rotate)
from utespac.wind_stats import (COMPASS_16, MANUFACTURER_CAMPBELL, compass_label,
                                direction_window_fraction, in_direction_window,
                                ustar_double_rotation, ustar_from_rotated,
                                wind_direction_speed)

EDGES = (60.0, 250.0)
NAMES = {(250.0, 60.0): "A", (60.0, 250.0): "B"}


# ── sector keys, labels and names ────────────────────────────────────────────

def test_sector_keys_wrap_through_the_first_edge():
    assert sector_keys(EDGES) == [(60.0, 250.0), (250.0, 60.0)]
    assert sector_keys([]) == []
    assert sector_label((60.0, 250.0)) == "60-250"
    assert sector_label((250.0, 60.0)) == "250-60"
    assert sector_label((0.0, 0.0)) == "all"
    assert sector_label(None) == ""


def test_sector_names_follow_the_declared_mapping():
    assert sector_name((250.0, 60.0), NAMES) == "A"
    assert sector_name((60.0, 250.0), NAMES) == "B"
    assert sector_name((0.0, 0.0), NAMES) == "all"
    assert sector_name(None, NAMES) == ""
    # an undeclared key falls back to its degree range
    assert sector_name((10.0, 20.0), NAMES) == "10-20"


@pytest.mark.parametrize("direction,expected", [
    (61.0, (60.0, 250.0)),
    (170.0, (60.0, 250.0)),
    (250.0, (60.0, 250.0)),        # closed on the upper edge
    (250.5, (250.0, 60.0)),
    (340.0, (250.0, 60.0)),
    (0.0, (250.0, 60.0)),          # wraps through north
    (60.0, (250.0, 60.0)),         # closed on the upper edge of the wrapping sector
    (360.0, (250.0, 60.0)),        # wrapped before the test
    (-20.0, (250.0, 60.0)),
])
def test_sector_of_matches_the_closed_upper_edge_convention(direction, expected):
    assert sector_of([direction], EDGES)[0] == expected


def test_sector_of_returns_none_for_missing_directions():
    assert sector_of([np.nan], EDGES) == [None]


def _plane_sample(n, seed):
    rng = np.random.default_rng(seed)
    direction = rng.uniform(0.0, 360.0, n)
    speed = rng.uniform(1.0, 6.0, n)
    u = speed * np.cos(np.radians(direction))
    v = -speed * np.sin(np.radians(direction))
    w = 0.03 * u - 0.02 * v + 0.001 + 0.002 * rng.standard_normal(n)
    return u, v, w, direction


def test_sector_of_selects_the_same_periods_as_fit_sectors():
    """The assignment reproduces ``fit_sectors``'s own membership exactly."""
    u, v, w, direction = _plane_sample(200, 13)
    fits = fit_sectors(u, v, w, direction, list(EDGES))
    assigned = sector_of(direction, EDGES)
    assert set(fits) == set(sector_keys(EDGES))
    for key, fit in fits.items():
        member = np.array([s == key for s in assigned])
        assert member.sum() > 10
        direct = PlanarFit.fit(u[member], v[member], w[member])
        assert direct.n == fit.n
        for attr in ("b0", "b1", "b2"):
            assert getattr(direct, attr) == pytest.approx(getattr(fit, attr), rel=1e-12)


# ── sector edges from the direction histogram ────────────────────────────────

def test_sector_edges_land_in_the_middle_of_each_empty_run():
    rng = np.random.default_rng(7)
    d = np.r_[rng.uniform(0.0, 90.0, 120), rng.uniform(190.0, 340.0, 120)]
    admit = np.ones(len(d), dtype=bool)
    edges, hist = sector_edges_from_histogram(d, admit, bin_deg=15.0)
    assert hist.sum() == len(d)
    # gaps are 90-190 (edge 140) and 340-360 (edge 350)
    assert edges == pytest.approx([140.0, 350.0], abs=8.0)


def test_sector_edges_treat_a_gap_across_north_as_one_run():
    rng = np.random.default_rng(8)
    d = rng.uniform(60.0, 300.0, 200)
    edges, _ = sector_edges_from_histogram(d, np.ones(len(d), dtype=bool), 15.0)
    assert len(edges) == 1
    assert min(abs(edges[0] - 0.0), abs(edges[0] - 360.0)) < 10.0


def test_sector_edges_are_empty_when_every_bin_is_populated():
    rng = np.random.default_rng(9)
    d = rng.uniform(0.0, 360.0, 4000)
    edges, hist = sector_edges_from_histogram(d, np.ones(len(d), dtype=bool), 15.0)
    assert (hist > 0).all()
    assert edges == []


def test_sector_edges_honour_the_admit_mask():
    d = np.r_[np.full(50, 30.0), np.full(50, 200.0)]
    admit = np.r_[np.ones(50, dtype=bool), np.zeros(50, dtype=bool)]
    edges, hist = sector_edges_from_histogram(d, admit, 15.0)
    assert hist.sum() == 50
    assert len(edges) == 1


def test_derived_edges_partition_every_admitted_direction_once():
    rng = np.random.default_rng(10)
    d = np.r_[rng.uniform(0.0, 90.0, 120), rng.uniform(190.0, 340.0, 120)]
    edges, _ = sector_edges_from_histogram(d, np.ones(len(d), dtype=bool), 15.0)
    assert len(edges) == 2
    lo, hi = edges
    a = sector_mask(d, lo, hi)
    b = sector_mask(d, hi, lo)
    assert np.all(a | b)
    assert not np.any(a & b)


# ── sectorwise fits ──────────────────────────────────────────────────────────

def test_fit_planar_returns_the_sector_and_the_single_fits_of_the_admitted_periods():
    u, v, w, direction = _plane_sample(240, 21)
    admit = np.ones(len(u), dtype=bool)
    admit[::5] = False
    sec, single = fit_planar(u, v, w, direction, EDGES, admit)
    assert set(sec) == set(sector_keys(EDGES))
    ref = PlanarFit.fit(u[admit], v[admit], w[admit])
    assert single == ref
    assert sum(f.n for f in sec.values()) == int(admit.sum())


def test_sector_fit_falls_back_to_the_single_fit():
    a, b, single = PlanarFit(0, 0.1, 0), PlanarFit(0, 0.2, 0), PlanarFit(0, 0.3, 0)
    fits = {(250.0, 60.0): a, (60.0, 250.0): None}
    assert sector_fit(fits, single, (250.0, 60.0)) is a
    assert sector_fit(fits, single, (60.0, 250.0)) is single     # sector without a fit
    assert sector_fit(fits, single, None) is single
    assert sector_fit(fits, single, (250.0, 60.0), use_sector=False) is single
    assert sector_fit({(1.0, 2.0): b}, single, (3.0, 4.0)) is single


def test_bootstrap_tilt_is_reproducible_and_scales_with_noise():
    rng = np.random.default_rng(5)
    n = 80
    u = rng.uniform(-4.0, 4.0, n)
    v = rng.uniform(-4.0, 4.0, n)
    quiet = 0.02 * u + 0.001 * rng.standard_normal(n)
    noisy = 0.02 * u + 0.05 * rng.standard_normal(n)
    sp_q, sr_q = bootstrap_tilt(u, v, quiet, n_boot=200)
    assert (sp_q, sr_q) == bootstrap_tilt(u, v, quiet, n_boot=200)
    sp_n, sr_n = bootstrap_tilt(u, v, noisy, n_boot=200)
    assert sp_n > 5 * sp_q and sr_n > 5 * sr_q


def test_bootstrap_tilt_is_nan_for_too_few_points():
    sp, sr = bootstrap_tilt(np.ones(3), np.ones(3), np.ones(3))
    assert np.isnan(sp) and np.isnan(sr)


# ── direction windows ────────────────────────────────────────────────────────

@pytest.mark.parametrize("direction,inside", [
    (180.0, True),
    (170.0, True),      # lower edge, included
    (190.0, True),      # upper edge, included
    (169.9, False),
    (190.1, False),
    (0.0, False),
    (350.0, False),
    (np.nan, False),
])
def test_direction_window_includes_both_edges(direction, inside):
    assert bool(in_direction_window([direction], 180.0, 10.0)[0]) is inside


def test_direction_window_is_wrap_safe():
    d = np.array([0.0, 5.0, 355.0, 360.0, 720.0, -5.0, 180.0])
    got = in_direction_window(d, centre=0.0, half_width=10.0)
    assert got.tolist() == [True, True, True, True, True, True, False]


def test_direction_window_fraction_counts_the_samples_in_the_window():
    n, n_in = 1000, 137
    direction = np.full(n, 20.0)
    direction[:n_in] = 178.0
    rad = np.radians(direction)
    u, v = np.cos(rad), -np.sin(rad)
    assert direction_window_fraction(u, v, 180.0, 10.0) == pytest.approx(n_in / n)
    back, _ = wind_direction_speed(u, v, 0.0, MANUFACTURER_CAMPBELL)
    assert np.allclose(back, direction)


def test_direction_window_fraction_ignores_missing_samples():
    u = np.array([np.cos(np.radians(180.0)), np.nan, np.cos(np.radians(0.0))])
    v = np.array([-np.sin(np.radians(180.0)), 0.5, -np.sin(np.radians(0.0))])
    assert direction_window_fraction(u, v, 180.0, 10.0) == pytest.approx(0.5)
    assert np.isnan(direction_window_fraction(np.array([np.nan]), np.array([np.nan]),
                                              180.0, 10.0))


@pytest.mark.parametrize("bearing,label", [
    (0.0, "N"), (11.2, "N"), (11.3, "NNE"), (90.0, "E"), (180.0, "S"),
    (265.0, "W"), (280.0, "W"), (348.75, "N"), (359.9, "N"), (-90.0, "W"),
    (720.0 + 45.0, "NE"),
])
def test_compass_label(bearing, label):
    assert compass_label(bearing) == label


def test_compass_label_is_empty_for_a_missing_bearing():
    assert compass_label(np.nan) == ""
    assert len(COMPASS_16) == 16


# ── friction velocity ────────────────────────────────────────────────────────

def _turbulent_period(n=36000, seed=3):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    u = 3.0 + 0.6 * a
    v = 0.4 * b
    w = -0.25 * a + 0.1 * b + 0.3 * c
    return u, v, w


def test_ustar_from_rotated_matches_the_covariance_definition():
    u, v, w = _turbulent_period()
    uw = np.mean((u - u.mean()) * (w - w.mean()))
    vw = np.mean((v - v.mean()) * (w - w.mean()))
    got = ustar_from_rotated(np.column_stack([u, v, w]))
    assert got == pytest.approx((uw ** 2 + vw ** 2) ** 0.25, rel=1e-12)


def test_ustar_from_rotated_skips_incomplete_rows_and_short_periods():
    u, v, w = _turbulent_period(n=2000)
    uvw = np.column_stack([u, v, w])
    gapped = uvw.copy()
    gapped[::3, 1] = np.nan
    keep = np.isfinite(gapped).all(axis=1)
    assert ustar_from_rotated(gapped) == pytest.approx(ustar_from_rotated(uvw[keep]),
                                                       rel=1e-12)
    assert np.isnan(ustar_from_rotated(uvw[:99]))


def test_ustar_double_rotation_is_invariant_to_the_instrument_heading():
    """Double rotation removes the yaw of the instrument frame."""
    u, v, w = _turbulent_period()
    ref = ustar_double_rotation(u, v, w)
    g = np.radians(40.0)
    ur = u * np.cos(g) - v * np.sin(g)
    vr = u * np.sin(g) + v * np.cos(g)
    assert ustar_double_rotation(ur, vr, w) == pytest.approx(ref, rel=1e-9)
    assert np.isnan(ustar_double_rotation(u[:99], v[:99], w[:99]))


def test_the_two_ustar_estimates_agree_on_double_rotated_winds():
    """On winds whose mean v and w are already zero, the double rotation is
    the identity and the two estimators coincide."""
    u, v, w = _turbulent_period()
    v = v - v.mean()
    w = w - w.mean()
    assert ustar_double_rotation(u, v, w) == pytest.approx(
        ustar_from_rotated(np.column_stack([u, v, w])), rel=1e-9)


def test_ustar_from_rotated_after_yaw_matches_double_rotation_without_tilt():
    """Yaw into the mean wind leaves only the pitch step of the double rotation,
    which is the identity when mean w is zero."""
    u, v, w = _turbulent_period()
    w = w - w.mean()
    g = np.radians(-30.0)
    ur = u * np.cos(g) - v * np.sin(g)
    vr = u * np.sin(g) + v * np.cos(g)
    rotated = yaw_rotate(np.column_stack([ur, vr, w]), np.array([0, len(u)]))
    assert ustar_from_rotated(rotated) == pytest.approx(
        ustar_double_rotation(ur, vr, w), rel=1e-9)
