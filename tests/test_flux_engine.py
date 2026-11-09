"""utespac.flux.engine.compute_period on a synthetic level: flag masking,
Schotanus vs. legacy temperature flux, WPL driver, table keys."""

import re

import numpy as np
import pytest
import xarray as xr

from utespac import names, stages
from utespac.flux import engine
from utespac.flux.engine import FluxOptions, compute_period
from utespac.flux.levels import LevelInputs
from utespac.flux.reference import ReferenceState
from utespac.flux.spectral_correction import massman2000_factor
from utespac.flux.tables import TABLE_SPECS
from utespac.model import (COMPONENT, HEIGHT, TIME_HF, Run, Sensor, Sensors,
                           to_datetime64)
from utespac.run_config import FluxConfig

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
    assert {"sensible_heat", "momentum", "tke", "sigma", "latent_heat", "co2_flux",
            "flux_qc", "obukhov"} <= set(r.values)
    assert {"rhov", "rhovPrime", "rhoCO2", "fwThPrime"} <= set(r.samples)


def test_covariances_have_the_built_in_signs():
    r = _run(_level())
    assert r.values["sensible_heat"]["w_ts_cov_raw"] > 0
    assert r.values["latent_heat"]["w_h2o_cov_pf"] > 0
    assert r.values["co2_flux"]["w_co2_cov_pf"] < 0
    assert r.values["obukhov"]["L"] < 0                     # unstable
    assert r.values["momentum"]["w_var_pf"] == pytest.approx(np.nanvar(_level().w[:N_SAMP]), rel=1e-6)


def test_schotanus_temperature_flux_is_below_the_buoyancy_flux():
    r = _run(_level())
    # w'T' = w'Ts' - 0.51 T w'q' with a positive moisture flux: smaller than w'Ts'
    assert r.values["sensible_heat"]["w_t_air_cov_pf"] < r.values["sensible_heat"]["w_ts_cov_raw"]
    # the WPL LE is driven by that w'T', so it sits below the buoyancy-driven form
    ref_T = 295.0
    rho_v, rho_d = 0.0095, 1.18
    Lv = r.values["latent_heat"]["Lv"]
    wpl = 1.0 + 28.97 / 18.0153 * rho_v / rho_d
    le_buoy = 1000.0 * Lv * wpl * (r.values["latent_heat"]["w_h2o_cov_pf"] / 1000.0
                                   + rho_v / ref_T * r.values["sensible_heat"]["w_theta_v_cov_pf"])
    assert r.values["latent_heat"]["LE_wpl_pf"] < le_buoy


def test_flags_blank_the_right_columns():
    lev = _level()
    lev.rot_flag[:] = True
    r = _run(lev)
    assert np.isnan(r.values["momentum"]["Tau_pf"]) and np.isnan(r.values["tke"]["TKE"])
    assert np.isnan(r.values["sigma"]["u_sigma_raw"])
    assert not np.isnan(r.values["sigma"]["w_sigma_raw"])   # w flag only
    assert not np.isnan(r.values["sensible_heat"]["w_ts_cov_raw"])   # unrotated flux survives
    assert np.isnan(r.values["sensible_heat"]["w_theta_v_cov_pf"])
    assert np.isnan(r.values["transport"]["ts_var_transport_pf"])
    lev2 = _level()
    lev2.Ts_flag[:] = True
    r2 = _run(lev2)
    assert np.isnan(r2.values["sigma"]["ts_sigma"])
    assert np.isnan(r2.values["sensible_heat"]["w_ts_cov_raw"])
    assert not np.isnan(r2.values["sigma"]["u_sigma_raw"])
    lev3 = _level()
    lev3.h2o_flag[:] = True
    r3 = _run(lev3)
    assert np.isnan(r3.values["latent_heat"]["w_h2o_cov_pf"])
    assert not np.isnan(r3.values["latent_heat"]["Lv"])


def test_surface_layer_scales():
    r = _run(_level())
    ustar = np.sqrt(r.values["momentum"]["Tau_pf"])
    # theta* = -w'theta_v'/u*, q* = -(w'h2o'/rho_moist)/u* (Stull 1988)
    assert r.values["scaling"]["ustar_pf"] == pytest.approx(ustar)
    assert r.values["scaling"]["theta_star"] == pytest.approx(
        -r.values["sensible_heat"]["w_theta_v_cov_pf"] / ustar)
    assert r.values["scaling"]["q_star"] == pytest.approx(
        -(r.values["latent_heat"]["w_h2o_cov_pf"] / (1.18 + 0.0095)) / ustar)
    assert r.values["scaling"]["theta_star"] < 0        # upward heat flux
    assert r.values["scaling"]["q_star"] < 0            # upward moisture flux
    lev_rot = _level()
    lev_rot.rot_flag[:] = True
    r_rot = _run(lev_rot)
    assert np.isnan(r_rot.values["scaling"]["theta_star"])
    assert np.isnan(r_rot.values["scaling"]["ustar_pf"])


def test_sensible_heat_in_watts_matches_its_defining_products():
    r = _run(_level())
    rho_cp = (1.18 + 0.0095) * 1004.67 * (1 + 0.84 * 0.008)   # rho and cp(q) of _ref()
    h = r.values["sensible_heat"]
    assert h["H_raw"] == pytest.approx(rho_cp * h["w_t_air_cov_raw"])
    assert h["H_pf"] == pytest.approx(rho_cp * h["w_t_air_cov_pf"])
    assert h["H_buoyancy_pf"] == pytest.approx(rho_cp * h["w_theta_v_cov_pf"])
    assert h["H_pf"] > 0 and h["H_pf"] < h["H_buoyancy_pf"]   # Schotanus below buoyancy
    lev_rot = _level()
    lev_rot.rot_flag[:] = True
    assert np.isnan(_run(lev_rot).values["sensible_heat"]["H_pf"])


def test_wpl_water_vapour_covariance_is_stored():
    r = _run(_level())
    lh = r.values["latent_heat"]
    # the WPL density term adds to the raw covariance behind LE_wpl_pf
    assert lh["w_h2o_wpl_cov_pf"] != lh["w_h2o_cov_pf"]
    assert abs(lh["w_h2o_wpl_cov_pf"] - lh["w_h2o_cov_pf"]) < abs(lh["w_h2o_cov_pf"])
    lev_h2o = _level()
    lev_h2o.h2o_flag[:] = True
    assert np.isnan(_run(lev_h2o).values["latent_heat"]["w_h2o_wpl_cov_pf"])


def test_stability_functions_follow_L():
    r = _run(_level())
    from utespac.stability import psi_h, psi_m
    zeta = 10.0 / r.values["obukhov"]["L"]                  # z = sonic height
    assert r.values["scaling"]["psi_m"] == pytest.approx(psi_m(zeta))
    assert r.values["scaling"]["psi_h"] == pytest.approx(psi_h(zeta))
    assert zeta < 0 and r.values["scaling"]["psi_m"] > 0     # unstable period
    lev_rot = _level()
    lev_rot.rot_flag[:] = True
    r_rot = _run(lev_rot)
    assert np.isnan(r_rot.values["scaling"]["psi_m"])
    assert np.isnan(r_rot.values["scaling"]["psi_h"])
    assert np.isnan(r_rot.values["scaling"]["q_star"])
    lev_h2o = _level()
    lev_h2o.h2o_flag[:] = True
    r_h2o = _run(lev_h2o)
    assert np.isnan(r_h2o.values["scaling"]["q_star"])
    assert not np.isnan(r_h2o.values["scaling"]["theta_star"])
    r_dry = _run(_level(with_h2o=False, with_co2=False, with_fw=False))
    assert "q_star" not in r_dry.values["scaling"]


def test_without_hygrometer_no_h2o_or_co2_tables():
    r = _run(_level(with_h2o=False, with_co2=False, with_fw=False))
    assert "latent_heat" not in r.values and "co2_flux" not in r.values
    assert "t_fw_sigma" not in r.values["sigma"]
    assert r.values["flux_qc"]["LE_ssitc"] == 9 or np.isnan(r.values["flux_qc"]["LE_ssitc"]) \
        or r.values["flux_qc"]["LE_ssitc"] >= 0


def test_wpl_co2_flux_is_the_covariance_plus_the_two_wpl_terms():
    from utespac import names
    from utespac.flux.reference import Md, Mv
    lev = _level()
    r = _run(lev)
    rho_d, rho_v, T = 1.18, 0.0095, 295.0                   # _ref() at period 0
    rho_c = np.nanmean(lev.co2[:N_SAMP]) / 1e6               # [kg m-3]
    evap = r.values["latent_heat"]["LE_wpl_pf"] / r.values["latent_heat"]["Lv"] / 1000.0
    kin = r.values["sensible_heat"]["w_t_air_cov_pf"]
    expected = (r.values["co2_flux"]["w_co2_cov_pf"]
                + Md / Mv * (rho_c / rho_d) * evap
                + (1.0 + Md / Mv * rho_v / rho_d) * (rho_c / T) * kin)
    # same arithmetic in another order: float64 round-off only (values ~1e-6, so no abs floor)
    assert r.values["co2_flux"]["Fc_wpl_pf"] == pytest.approx(expected, rel=1e-12, abs=0)
    assert names.variable("co2_flux", "Fc_wpl_pf").units == "kg m-2 s-1"


def _delayed_level(lag, smooth=1):
    """A level whose gases and fine wire lag w_pf by *lag* samples (20 Hz)."""
    lev = _level()
    rng = np.random.default_rng(21)
    n = N_SAMP * N_PER
    # w_pf a *smooth*-sample running mean; the unrotated w leads it by two
    lev.w_pf = np.convolve(lev.w_pf, np.ones(smooth) / smooth, mode="same")
    lev.w = np.roll(lev.w_pf, 2)
    lev.h2o[:] = 9 + 0.5 * np.roll(lev.w_pf, lag) + 0.2 * rng.normal(size=n)
    lev.co2[:] = 720 - 2 * np.roll(lev.w_pf, lag) + rng.normal(size=n)
    lev.fw[:] = (20 + 0.3 * np.roll(lev.w_pf, lag)
                 + 0.05 * rng.normal(size=n))   # theta_fw is the same array
    lev.Vtheta_fw[:] = lev.fw * (1 + 0.61 * 0.008)
    return lev


def _lagged_cov(a, b, k):
    """Covariance of a[i] with b[i + k] about the block means."""
    a = a[:N_SAMP] - a[:N_SAMP].mean()
    b = b[:N_SAMP] - b[:N_SAMP].mean()
    if k >= 0:
        return float(np.mean(a[:N_SAMP - k] * b[k:]))
    return float(np.mean(a[-k:] * b[:N_SAMP + k]))


def test_lag_search_off_leaves_the_scalars_unshifted():
    lev = _delayed_level(3)
    r = _run(lev, FluxOptions(detrend="constant", lag_search="off"))
    assert "lag_h2o" not in r.values["latent_heat"]
    assert "lag_co2" not in r.values["co2_flux"]
    # same products as the oracle at zero lag: float64 summation-order
    # noise only
    assert r.values["latent_heat"]["w_h2o_cov_pf"] == pytest.approx(
        _lagged_cov(lev.w_pf, lev.h2o, 0), rel=1e-9, abs=0)
    assert r.values["co2_flux"]["w_co2_cov_pf"] == pytest.approx(
        _lagged_cov(lev.w_pf, lev.co2, 0) / 1e6, rel=1e-9, abs=0)


@pytest.mark.parametrize("lag", [3, -3])
@pytest.mark.parametrize("table, gas, lag_key, cov_key, scale", [
    ("latent_heat", "h2o", "lag_h2o", "w_h2o_cov_pf", 1.0),
    ("co2_flux", "co2", "lag_co2", "w_co2_cov_pf", 1e-6),
])
def test_lag_search_aligns_the_gas_onto_w(table, gas, lag_key, cov_key, scale,
                                          lag):
    from utespac import names
    lev = _delayed_level(lag)
    off = _run(lev).values[table][cov_key]
    opts = FluxOptions(detrend="constant", lag_search="h2o_co2")
    vals = _run(lev, opts).values[table]
    assert vals[lag_key] == lag / 20.0
    assert names.variable(table, lag_key).units == "s"
    # the planted pairing w[i], c[i + lag]; float64 summation-order noise only
    assert vals[cov_key] == pytest.approx(
        _lagged_cov(lev.w_pf, getattr(lev, gas), lag) * scale, rel=1e-9, abs=0)
    assert abs(vals[cov_key]) > 5.0 * abs(off)


def test_lag_search_pairs_co2_wpl_sigma_with_the_aligned_record():
    lev = _level()
    rng = np.random.default_rng(21)
    lev.co2[:] = 720 - 2 * np.roll(lev.w, -5) + rng.normal(size=N_SAMP * N_PER)
    opts = FluxOptions(detrend="constant", lag_search="h2o_co2",
                       scan_freq=10.0)
    r = _run(lev, opts).values
    assert (r["latent_heat"]["lag_h2o"], r["co2_flux"]["lag_co2"]) == (
        0.0, -5 / 10.0)
    # hand-aligned: co2[i - 5] at sample i, the first five samples NaN as the
    # shift pads them
    hand = _level()
    hand.co2[:] = np.concatenate([np.full(5, np.nan), lev.co2[:-5]])
    opts_off = FluxOptions(detrend="constant", lag_search="off",
                           scan_freq=10.0)
    off = _run(hand, opts_off).values
    # the rho_c mean in rhoc_ext spans 1800 samples against 1795 by hand:
    # 2e-7 relative
    assert r["sigma"]["co2_wpl_sigma"] == pytest.approx(
        off["sigma"]["co2_wpl_sigma"], rel=1e-5, abs=0)


@pytest.mark.parametrize("mode, k", [("h2o_co2", 0), ("all", 2)])
def test_lag_search_all_also_aligns_the_fine_wire(mode, k):
    lev = _level()
    lev.fw[:] = np.roll(lev.T_son, 2) + 0.5     # theta_fw is the same array
    lev.Vtheta_fw[:] = lev.fw * (1 + 0.61 * 0.008)
    opts = FluxOptions(detrend="constant", lag_search=mode, scan_freq=10.0)
    sh = _run(lev, opts).values["sensible_heat"]
    # float64 summation-order noise only
    assert sh["w_theta_fw_cov_pf"] == pytest.approx(
        _lagged_cov(lev.w_pf, lev.theta_fw, k), rel=1e-9, abs=0)
    assert sh["w_t_fw_cov_pf"] == pytest.approx(
        _lagged_cov(lev.w_pf, lev.fw, k), rel=1e-9, abs=0)
    assert sh["w_theta_v_fw_cov_pf"] == pytest.approx(
        _lagged_cov(lev.w_pf, lev.Vtheta_fw, k), rel=1e-9, abs=0)


def test_lag_beyond_lag_max_leaves_every_scalar_unshifted():
    """A lag of 3 samples against a 2-sample window peaks on the edge."""
    lev = _delayed_level(3, smooth=5)
    opts = FluxOptions(detrend="constant", lag_search="all", lag_max_s=0.1)
    r = _run(lev, opts).values
    assert (r["latent_heat"]["lag_h2o"], r["co2_flux"]["lag_co2"]) == (0, 0)
    # float64 summation-order noise only
    assert r["sensible_heat"]["w_theta_fw_cov_pf"] == pytest.approx(
        _lagged_cov(lev.w_pf, lev.theta_fw, 0), rel=1e-9, abs=0)


@pytest.mark.parametrize("flag, lag_h2o, lag_co2", [
    (None, 3 / 20.0, 3 / 20.0),
    ("rot_flag", np.nan, np.nan),
    ("h2o_flag", np.nan, 3 / 20.0),
    ("co2_flag", 3 / 20.0, np.nan),
])
def test_flagged_period_blanks_the_lag(flag, lag_h2o, lag_co2):
    """A rotation or gas flag blanks the stored lag of the gases it covers."""
    lev = _delayed_level(3)
    if flag is not None:
        getattr(lev, flag)[0] = True
    opts = FluxOptions(detrend="constant", lag_search="h2o_co2")
    r = _run(lev, opts).values
    np.testing.assert_equal(
        (r["latent_heat"]["lag_h2o"], r["co2_flux"]["lag_co2"]),
        (lag_h2o, lag_co2))


def test_crosswind_scalar_covariances_are_stored():
    lev = _level()
    r = _run(lev)
    lat = r.values["scalar_flux_lateral"]
    for key in ("v_theta_v_cov_pf", "v_h2o_cov_pf", "v_h2o_wpl_cov_pf",
                "v_co2_cov_pf", "v_co2_wpl_cov_pf"):
        assert np.isfinite(lat[key]), key
    v = lev.v_pf[:N_SAMP]
    thv = lev.theta_son[:N_SAMP]
    # constant detrend computed independently; float64 summation-order noise only
    assert lat["v_theta_v_cov_pf"] == pytest.approx(
        np.nanmean((v - v.mean()) * (thv - thv.mean())), rel=1e-9, abs=0)
    lev_rot = _level()
    lev_rot.rot_flag[:] = True
    lat_rot = _run(lev_rot).values["scalar_flux_lateral"]
    assert all(np.isnan(lat_rot[k]) for k in lat)


# Arbitrary round geometry [m], distinct per option so a swap shows; not
# the dimensions of any instrument.
SONIC_PATH, IRGA_PATH, SEP_LAT, SEP_LON = 0.1, 0.2, 0.3, 0.4
GEOMETRY = {"sonic_path_len": SONIC_PATH, "irga_path_len": IRGA_PATH,
            "irga_sep_lateral": SEP_LAT, "irga_sep_longitudinal": SEP_LON}
FACTOR_KEYS = ("scf_H", "scf_LE", "scf_Fc")


def _massman_opts(**kw):
    kw.setdefault("detrend", "constant")
    return FluxOptions(spectral_correction="massman2000", **GEOMETRY, **kw)


@pytest.mark.parametrize("detrend", ["constant", "linear"])
def test_massman_factor_takes_the_period_quantities(monkeypatch, detrend):
    calls = []
    monkeypatch.setattr(engine, "massman2000_factor",
                        lambda **kw: calls.append(kw) or 1.0)
    lev = _level()
    lev.u_pf = lev.u + 1.0          # rotated speed apart from the sonic axes
    opts = _massman_opts(displacement_height=2.0, scan_freq=10.0,
                         avg_period_s=900.0, detrend=detrend)
    r = _run(lev, opts, jj=1)
    common = {
        # the period-1 mean; float64 summation-order noise only
        "U": pytest.approx(np.mean(lev.u_pf[N_SAMP:]), rel=1e-12, abs=0),
        # z - d = 8 m for both n_x u / z and z/L; z/L only to rounding
        "z_eff": 8.0,
        "zeta": pytest.approx(8.0 / r.values["obukhov"]["L"], rel=1e-12,
                              abs=0),
        "fs": 10.0, "avg_period_s": 900.0, "detrend": detrend}
    irga = {"path_len_sonic": SONIC_PATH, "path_len_scalar": IRGA_PATH,
            "sep_lateral": SEP_LAT, "sep_longitudinal": SEP_LON}
    assert calls == [
        {**common, "path_len_sonic": SONIC_PATH,
         "path_len_scalar": SONIC_PATH},
        {**common, **irga},
        {**common, **irga}]


@pytest.mark.parametrize("key, scalar_path, sep_lat, sep_lon", [
    ("scf_H", SONIC_PATH, 0.0, 0.0),
    ("scf_LE", IRGA_PATH, SEP_LAT, SEP_LON),
    ("scf_Fc", IRGA_PATH, SEP_LAT, SEP_LON),
])
def test_massman_factor_is_stored_per_flux(key, scalar_path, sep_lat,
                                           sep_lon):
    lev = _level()
    r = _run(lev, _massman_opts())
    expected = massman2000_factor(
        np.mean(lev.u_pf[:N_SAMP]), 10.0, 10.0 / r.values["obukhov"]["L"],
        20.0, 1800.0, sep_lateral=sep_lat, sep_longitudinal=sep_lon,
        path_len_sonic=SONIC_PATH, path_len_scalar=scalar_path,
        detrend="constant")
    # U differs by float64 summation order only
    assert r.values["flux_qc"][key] == pytest.approx(expected, rel=1e-12,
                                                     abs=0)
    assert names.variable("flux_qc", key).units == "1"


def test_massman_factor_leaves_the_fluxes_unscaled():
    """Every stored value but the factors is the same with the mode off."""
    lev = _level()
    on = _run(lev, _massman_opts()).values
    off = _run(lev).values
    assert on["flux_qc"]["scf_H"] > 1.0     # a factor that would show
    on["flux_qc"] = {k: v for k, v in on["flux_qc"].items()
                     if k not in FACTOR_KEYS}
    assert {t: set(c) for t, c in on.items()} \
        == {t: set(c) for t, c in off.items()}
    for table, columns in off.items():
        for key, value in columns.items():
            # the same arithmetic on the same inputs: bitwise equal
            np.testing.assert_array_equal(on[table][key], value,
                                          err_msg=f"{table}.{key}")


FACTOR_FLUX = {"scf_H": ("sensible_heat", "H_pf"),
               "scf_LE": ("latent_heat", "LE_wpl_pf"),
               "scf_Fc": ("co2_flux", "Fc_wpl_pf")}


# Worked from the engine's masks: H_pf by rot, Ts or h2o; LE_wpl_pf by
# unrot or h2o and through w'T'; Fc_wpl_pf by rot or co2 and through LE
# and w'T'; a rotation or Ts flag also blanks z/L.
@pytest.mark.parametrize("flag, finite", [
    (None, set(FACTOR_KEYS)),
    ("rot_flag", set()),
    ("Ts_flag", set()),
    ("h2o_flag", set()),
    ("co2_flag", {"scf_H", "scf_LE"}),
    ("unrot_flag", {"scf_H"}),
])
def test_flagged_period_blanks_its_factor(flag, finite):
    """A factor is NaN where its stored flux is NaN or z/L is undefined."""
    lev = _level()
    if flag is not None:
        getattr(lev, flag)[0] = True
    values = _run(lev, _massman_opts()).values
    qc = values["flux_qc"]
    assert {k for k in FACTOR_KEYS if np.isfinite(qc[k])} == finite
    for k in finite:
        table, flux = FACTOR_FLUX[k]
        assert np.isfinite(values[table][flux]), k


@pytest.mark.parametrize("mode, h2o, co2, keys", [
    ("off", True, True, set()),
    ("massman2000", True, True, set(FACTOR_KEYS)),
    ("massman2000", True, False, {"scf_H", "scf_LE"}),
    ("massman2000", False, False, {"scf_H"}),
])
def test_factor_columns_follow_the_mode_and_the_sensors(mode, h2o, co2,
                                                        keys):
    opts = FluxOptions(detrend="constant", spectral_correction=mode,
                       **GEOMETRY)
    lev = _level(with_h2o=h2o, with_co2=co2, with_fw=False)
    qc = _run(lev, opts).values["flux_qc"]
    assert set(qc) & set(FACTOR_KEYS) == keys


@pytest.mark.parametrize("attr, key", [
    ("sonic_path_len", "flux.sonicPathLength"),
    ("irga_path_len", "flux.irgaPathLength"),
    ("irga_sep_lateral", "flux.irgaSeparationLateral"),
    ("irga_sep_longitudinal", "flux.irgaSeparationLongitudinal"),
])
def test_massman_mode_requires_each_geometry_option(attr, key):
    geometry = {**GEOMETRY, attr: None}
    with pytest.raises(ValueError, match=re.escape(f"needs {key}")):
        FluxOptions(spectral_correction="massman2000", **geometry)


@pytest.mark.parametrize("mode", ["off", "massman2000"])
@pytest.mark.parametrize("attr, key", [
    ("sonic_path_len", "flux.sonicPathLength"),
    ("irga_path_len", "flux.irgaPathLength"),
    ("irga_sep_lateral", "flux.irgaSeparationLateral"),
    ("irga_sep_longitudinal", "flux.irgaSeparationLongitudinal"),
])
def test_negative_geometry_option_is_refused(attr, key, mode):
    assert getattr(FluxOptions(spectral_correction=mode,
                               **{**GEOMETRY, attr: 0.0}), attr) == 0.0
    with pytest.raises(ValueError,
                       match=re.escape(f"{key} must not be negative")):
        FluxOptions(spectral_correction=mode, **{**GEOMETRY, attr: -0.1})


def test_unknown_spectral_correction_raises():
    with pytest.raises(ValueError,
                       match="spectral_correction must be one of"):
        FluxOptions(spectral_correction="massman")


STAGE_Z = 3.0              # [m]
_STAGE_FIELD = {"Ux": "u", "Uy": "v", "Uz": "w", "Ts": "Tson",
                "H2O": "irgaH2O", "CO2": "irgaCO2"}


def _stage_run(info_extra):
    """One day at 1 Hz, one sonic and IRGA, rotation the identity."""
    n = 86400
    rng = np.random.default_rng(5)
    u = 2.0 + rng.normal(0, 0.3, n)
    v = rng.normal(0, 0.3, n)
    w = rng.normal(0, 0.1, n)
    columns = {"Ux": u, "Uy": v, "Uz": w,
               "Ts": 25.0 + 0.5 * w + rng.normal(0, 0.2, n),
               "H2O": 10.0 + 0.5 * w + rng.normal(0, 0.02, n),
               "CO2": 700.0 - 2.0 * w + rng.normal(0, 0.1, n)}
    t64 = to_datetime64(738000.0 + np.arange(n) / 86400.0)
    labels = {kind: f"{kind}_{STAGE_Z:g}" for kind in columns}
    ds = xr.Dataset({labels[k]: (TIME_HF, x) for k, x in columns.items()},
                    coords={TIME_HF: t64})
    ds.attrs["scan_hz"] = 1.0
    sensors = Sensors()
    for kind, name in labels.items():
        sensors.append(Sensor(_STAGE_FIELD[kind], "X_1s", name, STAGE_Z,
                              0.0, 1, None))
    header = ["TIMESTAMP"] + list(labels.values())
    info = {"siteElevation": 0.0, "qRef": 10.0,
            "detrendingFormat": "constant", **info_extra}
    run = Run(site=info, sensors=sensors, table_names=["X_1s"],
              headers={"X_1s": (header, [None] * len(header))},
              tables={"X_1s": ds})
    uvw = np.stack([u, v, w], axis=1)[:, None, :]
    dims = (TIME_HF, HEIGHT, COMPONENT)
    run.rotation = xr.Dataset({"rotated": (dims, uvw), "pf": (dims, uvw)},
                              coords={TIME_HF: t64, HEIGHT: [STAGE_Z],
                                      COMPONENT: ["u", "v", "w"]})
    return run


@pytest.mark.parametrize("flux_config, keys", [
    (FluxConfig(), set()),
    (FluxConfig(spectralCorrection="massman2000",
                sonicPathLength=SONIC_PATH, irgaPathLength=IRGA_PATH,
                irgaSeparationLateral=SEP_LAT,
                irgaSeparationLongitudinal=SEP_LON), set(FACTOR_KEYS)),
])
def test_flux_stage_threads_the_spectral_correction_settings(
        monkeypatch, flux_config, keys):
    """avgPer and the configured geometry reach the factor; it is stored."""
    calls = []
    monkeypatch.setattr(engine, "massman2000_factor",
                        lambda **kw: calls.append(kw) or 1.25)
    info = {**flux_config.to_info(), "avgPer": 60}
    qc = stages.flux(_stage_run(info)).products["flux_qc"]
    assert set(qc.data_vars) & set(FACTOR_KEYS) == keys
    for k in keys:
        assert np.all(qc[k].values == 1.25), k
    seen = {(c["avg_period_s"], c["path_len_sonic"], c["path_len_scalar"],
             c.get("sep_lateral"), c.get("sep_longitudinal"))
            for c in calls}
    assert seen == ({(3600.0, SONIC_PATH, SONIC_PATH, None, None),
                     (3600.0, SONIC_PATH, IRGA_PATH, SEP_LAT, SEP_LON)}
                    if keys else set())
