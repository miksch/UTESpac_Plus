"""generate_ameriflux: the AmeriFlux BASE export off the run files.

A synthetic two-level site is written as a ``utespac-run-3`` file, read
back through the exporter's own loader and turned into the BASE table:
the documented columns, the timestamp form, the -9999 fill and two value
checks (H = rho cp w'T', USTAR = sqrt(Tau_pf)).
"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import generate_ameriflux as ga
from utespac import model as M
from utespac.run_io import write_run

HEIGHTS = [5.0, 10.85]
N = 4
T0 = np.datetime64("2024-06-03T00:30:00")
TIMES = (T0 + np.arange(N) * np.timedelta64(30, "m")).astype("datetime64[ns]")


def _p(values):
    """(time, height) array from a per-height list of columns."""
    return np.column_stack([np.asarray(c, dtype=float) for c in values])


def _ds(data, coords):
    return xr.Dataset({k: ((M.TIME, M.HEIGHT), v) for k, v in data.items()}, coords=coords)


def _run(with_new_columns=True, with_humidity=True):
    coords = {M.TIME: TIMES, M.HEIGHT: HEIGHTS}
    rho = [1.20, 1.21, 1.22, 1.23]
    cp = [1006.0, 1006.5, 1007.0, 1007.5]
    cov = _p([[0.05, 0.06, np.nan, 0.08], [0.04, 0.05, 0.06, 0.07]])
    tau = _p([[0.09, 0.16, 0.25, 0.36], [0.04, 0.09, 0.16, 0.25]])

    sensible = xr.Dataset({"rho_air_ref": (M.TIME, rho), "cp_ref": (M.TIME, cp),
                           "w_t_air_cov_pf": ((M.TIME, M.HEIGHT), cov)}, coords=coords)
    momentum = _ds({"Tau_pf": tau}, coords)
    scaling = xr.Dataset(coords=coords)
    if with_new_columns:
        sensible["H_pf"] = ((M.TIME, M.HEIGHT),
                            np.asarray(rho)[:, None] * np.asarray(cp)[:, None] * cov)
        scaling["ustar_pf"] = ((M.TIME, M.HEIGHT), np.sqrt(tau))

    products = {
        "sensible_heat": sensible,
        "momentum": momentum,
        "scaling": scaling,
        "latent_heat": _ds({"LE_wpl_pf": _p([[100.0, 110.0, 120.0, 130.0]] * 2),
                            "Lv": _p([[2450.0] * 4] * 2),
                            "w_q_wpl_cov_pf": _p([[4e-5, 5e-5, 6e-5, 7e-5]] * 2)}, coords),
        "co2_flux": _ds({"Fc_wpl_pf": _p([[-2e-7, -3e-7, -4e-7, -5e-7]] * 2),
                         "co2_mole_fraction": _p([[410.0, 411.0, 412.0, 413.0]] * 2)}, coords),
        "sigma": _ds({"u_sigma_pf": _p([[0.5, 0.6, 0.7, 0.8]] * 2),
                      "v_sigma_pf": _p([[0.4, 0.5, 0.6, 0.7]] * 2),
                      "w_sigma_pf": _p([[0.3, 0.4, 0.5, 0.6]] * 2),
                      "ts_sigma": _p([[0.2, 0.3, 0.4, 0.5]] * 2),
                      "co2_sigma": _p([[3.0, 3.1, 3.2, 3.3]] * 2),
                      "h2o_sigma": _p([[0.5, 0.6, 0.7, 0.8]] * 2)}, coords),
        "obukhov": _ds({"L": _p([[-50.0, -40.0, 100.0, np.inf]] * 2)}, coords),
        "tke": _ds({"TKE": _p([[0.25, 0.36, 0.49, 0.64]] * 2)}, coords),
        "temperature": _ds({"theta_v": _p([[295.0, 296.0, 297.0, 298.0]] * 2)}, coords),
        "flux_qc": _ds({"Tau_ssitc": _p([[0, 1, 2, 0]] * 2), "H_ssitc": _p([[0, 0, 1, 2]] * 2),
                        "LE_ssitc": _p([[1, 1, 0, 0]] * 2), "Fc_ssitc": _p([[2, 0, 0, 1]] * 2)},
                       coords),
    }
    if with_humidity:
        products["humidity"] = _ds({"rho_air_moist": _p([[1.19, 1.20, 1.21, 1.22]] * 2),
                                    "r": _p([[7.0, 7.1, 7.2, 7.3]] * 2),
                                    "theta_v_slow": _p([[295.5, 296.5, 297.5, 298.5]] * 2)},
                                   coords)

    sensors = M.Sensors(M.Sensor("u", "X_30min", f"Ux_{h:g}", h, 180.0, 1) for h in HEIGHTS)
    run = M.Run(site={"avgPer": 30, "siteFolder": "X"}, sensors=sensors,
                table_names=["X_30min"], headers={})
    run.products = products
    run.periods = {"X_30min": xr.Dataset({"Pressure_10.85": (M.TIME, [101.1, 101.2, 101.3, 101.4])},
                                         coords={M.TIME: TIMES})}
    run.wind = _ds({"direction": _p([[190.0, 200.0, 210.0, 220.0]] * 2),
                    "shadow_flag": _p([[0.0, 0.0, 1.0, 0.0]] * 2)}, coords)
    rot = np.zeros((N, len(HEIGHTS), 3))
    rot[:, :, 0] = _p([[2.0, 2.5, 3.0, 3.5]] * 2)
    run.rotation = xr.Dataset({"rotated_mean": ((M.TIME, M.HEIGHT, M.COMPONENT), rot)},
                              coords={M.TIME: TIMES, M.HEIGHT: HEIGHTS,
                                      M.COMPONENT: ["u", "v", "w"]})
    return run


def _site(tmp_path, **kwargs):
    """Write the synthetic run file and read it back as a SiteRun."""
    out = tmp_path / "X" / "output"
    out.mkdir(parents=True)
    (tmp_path / "X" / "siteInfo.toml").write_text("tower = 180\n")
    write_run(_run(**kwargs), out / "X_30minAvg_GPF_ConstDet_2024_06_03.nc")
    return ga.read_site(tmp_path, "X", 30, "GPF_ConstDet")


DOCUMENTED = ("H", "LE", "FC", "TAU", "USTAR", "WS", "WD", "MO_LENGTH", "ZL", "TKE",
              "T_SONIC", "T_SONIC_SIGMA", "CO2", "CO2_SIGMA", "H2O", "H2O_SIGMA", "FH2O",
              "U_SIGMA", "V_SIGMA", "W_SIGMA", "WD_FILTER",
              "TAU_SSITC_TEST", "H_SSITC_TEST", "LE_SSITC_TEST", "FC_SSITC_TEST")


def test_site_run_reads_products_by_name(tmp_path):
    site = _site(tmp_path)
    assert site.heights() == HEIGHTS and site.z_ref() == 5.0
    assert "w_t_air_cov_pf" in site.group("sensible_heat")
    assert site.wind is not None and site.rotation is not None


def test_frame_has_the_documented_columns_per_height(tmp_path):
    df = ga.ameriflux_frame(_site(tmp_path))
    for v_idx in (1, 2):
        for name in DOCUMENTED:
            assert f"{name}_1_{v_idx}_1" in df.columns, name
    assert "PA_1_1_1" in df.columns
    assert list(df.columns[:2]) == ["TIMESTAMP_END", "TIMESTAMP_START"]
    assert len(df) == N


def test_timestamps_are_ameriflux_form(tmp_path):
    df = ga.ameriflux_frame(_site(tmp_path))
    for col in ("TIMESTAMP_START", "TIMESTAMP_END"):
        assert all(re.fullmatch(r"\d{12}", s) for s in df[col])
    assert df["TIMESTAMP_END"].iloc[0] == "202406030030"
    assert df["TIMESTAMP_START"].iloc[0] == "202406030000"      # end minus the averaging period
    assert df["TIMESTAMP_END"].iloc[-1] == "202406030200"


def test_values_of_the_new_columns_are_taken_as_stored(tmp_path):
    site = _site(tmp_path)
    df = ga.ameriflux_frame(site)
    sh, scal = site.group("sensible_heat"), site.group("scaling")
    assert np.allclose(df["H_1_2_1"].values, sh["H_pf"].values[:, 1], equal_nan=True)
    assert np.allclose(df["USTAR_1_2_1"].values, scal["ustar_pf"].values[:, 1])
    assert np.allclose(df["LE_1_1_1"].values, site.group("latent_heat")["LE_wpl_pf"].values[:, 0])
    assert np.allclose(df["TKE_1_1_1"].values, site.group("tke")["TKE"].values[:, 0])
    assert np.allclose(df["WD_1_1_1"].values, site.wind["direction"].values[:, 0])
    assert np.allclose(df["WD_FILTER_1_1_1"].values, site.wind["shadow_flag"].values[:, 0])
    assert np.allclose(df["H_SSITC_TEST_1_1_1"].values, site.group("flux_qc")["H_ssitc"].values[:, 0])
    assert np.allclose(df["PA_1_1_1"].values, [101.1, 101.2, 101.3, 101.4])


def test_fallbacks_are_rho_cp_cov_and_sqrt_tau(tmp_path):
    """Without the columns added by the rename (a utespac-run-2 file read
    through the shim), H is rho cp w'T_air' and USTAR is sqrt(Tau_pf)."""
    site = _site(tmp_path, with_new_columns=False, with_humidity=False)
    df = ga.ameriflux_frame(site)
    sh, mom = site.group("sensible_heat"), site.group("momentum")
    rho, cp = sh["rho_air_ref"].values, sh["cp_ref"].values
    for v_idx, k in ((1, 0), (2, 1)):
        assert np.allclose(df[f"H_1_{v_idx}_1"].values, rho * cp * sh["w_t_air_cov_pf"].values[:, k],
                           equal_nan=True)
        assert np.allclose(df[f"USTAR_1_{v_idx}_1"].values, np.sqrt(mom["Tau_pf"].values[:, k]))
        assert np.allclose(df[f"TAU_1_{v_idx}_1"].values, rho * mom["Tau_pf"].values[:, k])


def test_unit_conversions_of_the_gas_fluxes(tmp_path):
    site = _site(tmp_path)
    df = ga.ameriflux_frame(site)
    fc = site.group("co2_flux")["Fc_wpl_pf"].values[:, 0]
    assert np.allclose(df["FC_1_1_1"].values, fc / 44.01 * 1000.0 * 1e6)     # -> umol m-2 s-1
    q = site.group("latent_heat")["w_q_wpl_cov_pf"].values[:, 0]
    assert np.allclose(df["FH2O_1_1_1"].values, q * 1e6 / 18.015)            # -> mmol m-2 s-1
    r = site.group("humidity")["r"].values[:, 0]
    assert np.allclose(df["H2O_1_1_1"].values, r * 28.97 / 18.015)
    z = HEIGHTS[1]
    theta_v = site.group("temperature")["theta_v"].values[:, 1]
    assert np.allclose(df["T_SONIC_1_2_1"].values, theta_v - ga.GAMMA * (z - site.z_ref()))
    L = site.group("obukhov")["L"].values[:, 0]
    with np.errstate(invalid="ignore"):
        expected = np.where(np.isfinite(L), HEIGHTS[0] / L, np.nan)
    assert np.allclose(df["ZL_1_1_1"].values, expected, equal_nan=True)


def test_csv_is_written_with_the_base_format(tmp_path):
    site = _site(tmp_path)
    df = ga.ameriflux_frame(site)
    path = ga.write_base_csv(df, tmp_path / "ameriflux_output", "US-xX")
    assert path.endswith("US-xX_HH_202406030000_202406030200.csv")
    back = pd.read_csv(path, dtype={"TIMESTAMP_START": str, "TIMESTAMP_END": str})
    assert list(back.columns[:2]) == ["TIMESTAMP_START", "TIMESTAMP_END"]
    assert len(back) == N and not back.isna().any().any()          # every gap is filled
    assert back["H_1_1_1"].iloc[2] == ga.MISSING                   # the NaN covariance period
    assert (back["MO_LENGTH_1_1_1"].iloc[3] == ga.MISSING)         # inf L is missing, not inf
    assert back["TIMESTAMP_START"].iloc[0] == "202406030000"


def test_missing_run_files_raise(tmp_path):
    (tmp_path / "X").mkdir()
    (tmp_path / "X" / "siteInfo.toml").write_text("tower = 180\n")
    (tmp_path / "X" / "output").mkdir()
    with pytest.raises(FileNotFoundError):
        ga.read_site(tmp_path, "X", 30, "GPF_ConstDet")
