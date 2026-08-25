"""utespac.rotation: planar fit, sectors, global-fit application, yaw rotation."""

import numpy as np
import pytest

from utespac.averaging import period_bounds
from utespac.pf_info import PFRecord, PFTable
from utespac.rotation import (PlanarFit, SonicSeries, apply_global_fit, fit_sectors,
                              rotate_sonics, sector_mask, yaw_rotate)

DAY = 739406.0   # 2024-06-03 as a datenum midnight


def _plane(b0=0.04, b1=-0.08, b2=0.03, n=400, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(1, 8, n)
    v = rng.uniform(-3, 3, n)
    return u, v, b0 + b1 * u + b2 * v


def test_fit_recovers_plane_and_angles():
    u, v, w = _plane()
    fit = PlanarFit.fit(u, v, w)
    assert (fit.b0, fit.b1, fit.b2) == pytest.approx((0.04, -0.08, 0.03), abs=1e-9)
    assert fit.n == 400
    assert fit.pitch_deg == pytest.approx(np.degrees(np.arcsin(0.08 / np.sqrt(1 + 0.08 ** 2))))
    assert fit.roll_deg == pytest.approx(np.degrees(np.arcsin(0.03 / np.sqrt(1 + 0.03 ** 2))))


def test_fit_drops_nan_rows_and_needs_four_points():
    u, v, w = _plane(n=10)
    w = w.copy()
    w[:7] = np.nan
    assert PlanarFit.fit(u, v, w) is None          # 3 valid rows
    w[0] = 0.04 + -0.08 * u[0] + 0.03 * v[0]
    assert PlanarFit.fit(u, v, w).n == 4


def test_apply_removes_offset_and_flattens_plane():
    u, v, w = _plane()
    fit = PlanarFit.fit(u, v, w)
    out = fit.apply(u, v, w)
    assert np.allclose(out[:, 2], 0.0, atol=1e-9)
    out_legacy = fit.apply(u, v, w, remove_offset=False)
    assert abs(out_legacy[:, 2].mean() - 0.04 * fit.matrix()[2, 2]) < 1e-9


def test_sector_mask_conventions():
    d = np.array([0.0, 45.0, 180.0, 350.0])
    assert sector_mask(d, 0.0, 0.0).all()
    assert sector_mask(d, 10.0, 200.0).tolist() == [False, True, True, False]
    assert sector_mask(d, 300.0, 30.0).tolist() == [True, False, False, True]   # wraps north


def test_fit_sectors_keys_and_wrap():
    u, v, w = _plane(n=600)
    d = np.linspace(0, 359.9, 600)
    fits = fit_sectors(u, v, w, d, [0.0, 180.0])
    assert set(fits) == {(0.0, 180.0), (180.0, 0.0)}
    assert all(f is not None for f in fits.values())
    assert fit_sectors(u, v, w, d, [])[(0.0, 0.0)].n == 600


def test_apply_global_fit_by_date_and_sector():
    u, v, w = _plane(n=8)
    t = DAY + np.arange(8) / 24.0            # hourly samples on 2024-06-03
    t[-1] = DAY + 1.5                        # one sample the next day
    d = np.array([100.0] * 4 + [250.0] * 4)
    recs = [PFRecord(10.0, "2024-06-03", "2024-06-03", 0.0, 180.0, 0.04, -0.08, 0.03),
            PFRecord(10.0, "2024-06-03", "2024-06-03", 180.0, 0.0, 0.0, 0.0, 0.0)]
    out = apply_global_fit(u, v, w, t, d, recs)
    assert np.allclose(out[:4, 2], 0.0, atol=1e-9)            # first sector, on the plane
    assert np.allclose(out[4:7], np.column_stack([u, v, w])[4:7])   # identity record
    assert np.isnan(out[7]).all()                                # outside every date window


def test_yaw_rotate_zeroes_period_mean_v():
    rng = np.random.default_rng(2)
    n = 3600
    wind = np.column_stack([3 + rng.normal(size=n), 1 + rng.normal(size=n), rng.normal(size=n)])
    bounds = period_bounds(n, 2)
    out = yaw_rotate(wind, bounds)
    for j in range(2):
        seg = out[bounds[j]:bounds[j + 1]]
        assert abs(seg[:, 1].mean()) < 1e-12
        assert np.allclose(np.hypot(seg[:, 0], seg[:, 1]),
                           np.hypot(wind[bounds[j]:bounds[j + 1], 0], wind[bounds[j]:bounds[j + 1], 1]))
    assert np.array_equal(out[:, 2], wind[:, 2])


def test_rotate_sonics_local_and_global():
    n = 48 * 60                                # one day at 1/60 Hz
    t = DAY + np.arange(1, n + 1) / n
    rng = np.random.default_rng(3)
    phase = np.arange(n) / n
    u = 2 + 6 * phase + 0.3 * rng.normal(size=n)                   # mean wind varies across periods
    v = 2 * np.sin(2 * np.pi * phase) + 0.3 * rng.normal(size=n)
    w = 0.04 - 0.08 * u + 0.03 * v + 0.05 * rng.normal(size=n)
    bounds = period_bounds(n, 48)
    u_bar = np.array([u[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
    v_bar = np.array([v[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
    w_bar = np.array([w[a:b].mean() for a, b in zip(bounds[:-1], bounds[1:])])
    good = np.ones(48, dtype=bool)
    s = SonicSeries(10.0, u, v, w, np.full(48, 200.0), good, u_bar, v_bar, w_bar)
    res = rotate_sonics([s], t, 30)
    assert res.headers == ["10.0m:u", "10.0m:v", "10.0m:w"]
    assert res.fits[10.0].b1 == pytest.approx(-0.08, abs=0.01)
    assert abs(np.nanmean(res.pf_only[:, 2])) < 0.05
    assert np.allclose(res.averaged("rotated")[:, 1], 0.0, atol=1e-12)

    table = PFTable([PFRecord(10.0, "2024-06-03", "2024-06-03", 0.0, 0.0, 0.04, -0.08, 0.03)])
    res_g = rotate_sonics([s], t, 30, pf_table=table)
    assert res_g.records[10.0][0].b1 == -0.08
    assert not res_g.skipped
    missing = PFTable([PFRecord(20.0, "2024-06-03", "2024-06-03", 0.0, 0.0, 0.0, 0.0, 0.0)])
    res_m = rotate_sonics([s], t, 30, pf_table=missing)
    assert 10.0 in res_m.skipped and np.isnan(res_m.rotated).all()
