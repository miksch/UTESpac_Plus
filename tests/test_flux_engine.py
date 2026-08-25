"""utespac.flux.engine.compute_period on a synthetic level: flag masking,
Schotanus vs. legacy temperature flux, WPL driver, table keys."""

import numpy as np
import pytest

from utespac.flux.engine import FluxOptions, compute_period
from utespac.flux.levels import LevelInputs
from utespac.flux.reference import ReferenceState
from utespac.flux.tables import TABLE_SPECS

N_PER = 2
N_SAMP = 1800


def _ref():
    n = N_PER
    T = np.full(n, 295.0)
    q = np.full(n, 0.008)
    rho_d, rho_v = np.full(n, 1.18), np.full(n, 0.0095)
    return ReferenceState(10.0, 20.0, 101.0, np.full(n, 101.0), T, q, np.full(N_SAMP * n, 0.008),
                          rho_d, rho_v, rho_d + rho_v, T * (1 + 0.61 * q))


def _level(seed=0, with_h2o=True, with_co2=True, with_fw=True):
    rng = np.random.default_rng(seed)
    n = N_SAMP * N_PER
    w = rng.normal(size=n)
    u = 4 + rng.normal(size=n)
    v = rng.normal(size=n)
    Ts = 22 + 0.2 * w + 0.3 * rng.normal(size=n)          # positive heat flux
    theta = Ts + 0.0098 * 0.0
    theta_air = Ts / (1 + 0.51 * 0.008)
    h2o = 9 + 0.5 * w + 0.2 * rng.normal(size=n) if with_h2o else None
    co2 = 720 - 2 * w + rng.normal(size=n) if with_co2 else None
    fw = Ts + 0.5 if with_fw else None
    zeros = np.zeros(N_PER, dtype=bool)
    return LevelInputs(
        0, 10.0, 0, 1, 215.0, u, v, w, Ts, theta, theta_air,
        u, v, w, u, v, u, v, w,
        zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
        fw=fw, theta_fw=fw, Vtheta_fw=None if fw is None else fw * (1 + 0.61 * 0.008),
        h2o=h2o, co2=co2, P_kPa=np.full(N_PER, 101.0),
        q_fast_local=np.full(n, 0.008), direction_avg=np.full(N_PER, 200.0),
    )


def _run(lev, opts=None, jj=0):
    t = np.arange(N_SAMP * N_PER, dtype=float) / 86400.0 + 739406.0
    return compute_period(lev, _ref(), opts or FluxOptions(detrend="constant"), jj,
                          jj * N_SAMP, (jj + 1) * N_SAMP, t)


def test_keys_are_known_table_columns():
    r = _run(_level())
    for table, vals in r.values.items():
        spec_keys = {k for k, _ in TABLE_SPECS[table].columns}
        unknown = set(vals) - spec_keys
        assert not unknown, (table, unknown)
    assert {"H", "tau", "tke", "sigma", "LHflux", "CO2flux", "fluxQC", "L"} <= set(r.values)
    assert {"rhov", "rhovPrime", "rhoCO2", "fwThPrime"} <= set(r.samples)


def test_covariances_have_the_built_in_signs():
    r = _run(_level())
    assert r.values["H"]["Ts_w"] > 0
    assert r.values["LHflux"]["E_wPF"] > 0
    assert r.values["CO2flux"]["Fc_wPF"] < 0
    assert r.values["L"]["L"] < 0                     # unstable
    assert r.values["tau"]["wPF_wPF"] == pytest.approx(np.nanvar(_level().w[:N_SAMP]), rel=1e-6)


def test_schotanus_temperature_flux_is_below_the_buoyancy_flux():
    r = _run(_level())
    # w'T' = w'Ts' - 0.51 T w'q' with a positive moisture flux: smaller than w'Ts'
    assert r.values["H"]["Tair_wPF"] < r.values["H"]["Ts_w"]
    # the WPL LE is driven by that w'T', so it sits below the buoyancy-driven form
    ref_T = 295.0
    rho_v, rho_d = 0.0095, 1.18
    Lv = r.values["LHflux"]["Lv"]
    wpl = 1.0 + 28.97 / 18.0153 * rho_v / rho_d
    le_buoy = 1000.0 * Lv * wpl * (r.values["LHflux"]["E_wPF"] / 1000.0
                                   + rho_v / ref_T * r.values["H"]["Thv_wPF"])
    assert r.values["LHflux"]["LE_WPL_wPF"] < le_buoy


def test_flags_blank_the_right_columns():
    lev = _level()
    lev.rot_flag[:] = True
    r = _run(lev)
    assert np.isnan(r.values["tau"]["tau_PF"]) and np.isnan(r.values["tke"]["tke"])
    assert np.isnan(r.values["sigma"]["sigma_u"])
    assert not np.isnan(r.values["sigma"]["sigma_w"])      # w flag only
    assert not np.isnan(r.values["H"]["Ts_w"])             # unrotated flux survives a rotation flag
    assert np.isnan(r.values["H"]["Thv_wPF"])
    lev2 = _level()
    lev2.Ts_flag[:] = True
    r2 = _run(lev2)
    assert np.isnan(r2.values["sigma"]["sigma_Tson"]) and np.isnan(r2.values["H"]["Ts_w"])
    assert not np.isnan(r2.values["sigma"]["sigma_u"])
    lev3 = _level()
    lev3.h2o_flag[:] = True
    r3 = _run(lev3)
    assert np.isnan(r3.values["LHflux"]["E_wPF"]) and not np.isnan(r3.values["LHflux"]["Lv"])


def test_surface_layer_scales():
    r = _run(_level())
    ustar = np.sqrt(r.values["tau"]["tau_PF"])
    # theta*_SL = -w'theta_v'/u*, q*_SL = -(E_wPF/rho_moist)/u* (Stull 1988)
    assert r.values["scaling"]["theta_star_SL"] == pytest.approx(
        -r.values["H"]["Thv_wPF"] / ustar)
    assert r.values["scaling"]["q_star_SL"] == pytest.approx(
        -(r.values["LHflux"]["E_wPF"] / (1.18 + 0.0095)) / ustar)
    assert r.values["scaling"]["theta_star_SL"] < 0        # upward heat flux
    assert r.values["scaling"]["q_star_SL"] < 0            # upward moisture flux
    lev_rot = _level()
    lev_rot.rot_flag[:] = True
    r_rot = _run(lev_rot)
    assert np.isnan(r_rot.values["scaling"]["theta_star_SL"])
    assert np.isnan(r_rot.values["scaling"]["q_star_SL"])
    lev_h2o = _level()
    lev_h2o.h2o_flag[:] = True
    r_h2o = _run(lev_h2o)
    assert np.isnan(r_h2o.values["scaling"]["q_star_SL"])
    assert not np.isnan(r_h2o.values["scaling"]["theta_star_SL"])
    r_dry = _run(_level(with_h2o=False, with_co2=False, with_fw=False))
    assert "q_star_SL" not in r_dry.values["scaling"]


def test_without_hygrometer_no_h2o_or_co2_tables():
    r = _run(_level(with_h2o=False, with_co2=False, with_fw=False))
    assert "LHflux" not in r.values and "CO2flux" not in r.values
    assert "sigma_TFW" not in r.values["sigma"]
    assert r.values["fluxQC"]["LE_SSITC"] == 9 or np.isnan(r.values["fluxQC"]["LE_SSITC"]) \
        or r.values["fluxQC"]["LE_SSITC"] >= 0
