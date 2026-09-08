"""utespac.model: sensors, the Run, and the legacy converters (both directions)."""

import numpy as np
import pytest
import xarray as xr

from utespac import model as M
from utespac.campbell_date import MATLAB_EPOCH

DAY = 739406.0
HEADERS = [[["TIMESTAMP", "Ux_10.85", "Uy_10.85", "Uz_10.85", "T_Sonic_10.85", "H2O_10.85"],
            [None, 10.85, 10.85, 10.85, 10.85, 10.85],
            ["", "m s-1", "m s-1", "m s-1", "deg C", "g m-3"]],
           [["TIMESTAMP", "Temp_10.85", "RH_10.85"], [None, 10.85, 10.85]]]
TABLES = ["X_1Hz", "X_30min"]
SENSOR_INFO = {"u": np.array([[0, 1, 10.85, 215.0, 1.0]]), "v": np.array([[0, 2, 10.85]]),
               "w": np.array([[0, 3, 10.85]]), "Tson": np.array([[0, 4, 10.85]]),
               "irgaH2O": np.array([[0, 5, 10.85]]), "T": np.array([[1, 1, 10.85]]),
               "RH": np.array([[1, 2, 10.85]])}


def _data(n=120, n_slow=2):
    rng = np.random.default_rng(0)
    t = DAY + np.arange(1, n + 1) / 86400.0
    fast = np.column_stack([t, rng.normal(size=(n, 5))])
    slow = np.column_stack([DAY + np.arange(1, n_slow + 1) * 1800 / 86400.0, rng.normal(size=(n_slow, 2))])
    return [fast, slow]


def test_sensors_from_and_to_legacy():
    s = M.Sensors.from_legacy(SENSOR_INFO, HEADERS, TABLES)
    u = s.at("u", 10.85)
    assert u.column == "Ux_10.85" and u.orientation == 215.0 and u.manufacturer == 1
    assert s.at("T", 10.85).table == "X_30min"
    assert s.heights("u") == [10.85] and s.has("irgaH2O") and not s.has("fw")
    # units come from the header's third row; the slow table declares none
    assert u.units == "m s-1" and s.at("Tson", 10.85).units == "deg C"
    assert s.at("T", 10.85).units is None
    back = s.to_legacy(HEADERS, TABLES)
    for k, v in SENSOR_INFO.items():
        assert np.array_equal(back[k], np.asarray(v, dtype=float))


def test_tables_round_trip_and_datetime64_axis():
    data = _data()
    tabs = M.tables_from_legacy(data, HEADERS, TABLES, scan_hz=[1.0, 1 / 1800])
    ds = tabs["X_1Hz"]
    assert ds[M.TIME_HF].dtype == np.dtype("datetime64[ns]")
    assert list(ds.data_vars) == HEADERS[0][0][1:]
    assert ds.attrs["scan_hz"] == 1.0
    back = M.tables_to_legacy(tabs, {n: (HEADERS[i][0], HEADERS[i][1]) for i, n in enumerate(TABLES)}, TABLES)
    for a, b in zip(data, back):
        assert np.array_equal(a[:, 1:], b[:, 1:])
        assert np.abs(a[:, 0] - b[:, 0]).max() < 1e-9        # datenum <-> datetime64[ms], sub-microsecond


def test_wind_round_trip_keeps_the_column_labels():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    out = {"spdAndDir": np.column_stack([t, [200.0, 210.0], [3.0, 4.0], [0.0, 1.0]]),
           "spdAndDirHeader": ["timeStamp", "wind_dir_10.85", "wind_speed_10.85", "shadow_flag_10.85"]}
    w = M.wind_from_legacy(out, [10.85])
    assert w["direction"].dims == (M.TIME, M.HEIGHT)
    # the sector bounds are not in the labels any more; they live in the wind group
    assert np.isnan(w["sector_min"].values).all() and np.isnan(w["sector_max"].values).all()
    back = M.wind_to_legacy(w)
    assert back["spdAndDirHeader"] == out["spdAndDirHeader"]
    assert np.array_equal(back["spdAndDir"], out["spdAndDir"])


def test_rotation_labels_name_the_frame():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    hf = DAY + np.arange(1, 5) / 86400.0
    rot = M.rotation_from_legacy(np.ones((4, 3)), np.full((4, 3), 2.0),
                                 {"rotatedSonic": np.ones((2, 3)), "PFSonic": np.full((2, 3), 2.0)},
                                 [10.85], M.to_datetime64(hf), M.to_datetime64(t))
    _, _, out = M.rotation_to_legacy(rot)
    assert out["rotatedSonicHeader"] == ["u_pf_10.85", "v_pf_10.85", "w_pf_10.85"]
    assert out["PFSonicHeader"] == ["u_tilt_10.85", "v_tilt_10.85", "w_tilt_10.85"]


def test_products_round_trip_with_fixed_columns_and_trim():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    H = np.column_stack([t, [1.2, 1.2], [1005.0, 1005.0], [0.1, 0.2], [0.05, np.nan]])
    out = {"sensible_heat": H,
           "sensible_heatHeader": ["time", "rho_air_ref", "cp_ref",
                                   "w_ts_cov_raw_10.85", "w_theta_v_cov_pf_10.85"],
           "temperature": np.column_stack([t, [20.0, 21.0]]),
           "temperatureHeader": ["time", "theta_v_10.85"],
           "humidity": np.column_stack([t, [7.5, 7.6], [295.0, 295.1]]),
           "humidityHeader": ["time", "q_10.85", "theta_v_slow_10.85"]}
    p = M.products_from_legacy(out, [10.85])
    sh = p["sensible_heat"]
    assert sh["w_ts_cov_raw"].dims == (M.TIME, M.HEIGHT) and sh["rho_air_ref"].dims == (M.TIME,)
    assert sh["w_ts_cov_raw"].attrs["legacy_label"] == "{hn}m son:Ts'w'"
    assert sh["w_ts_cov_raw"].attrs["units"] == "K m s-1"
    assert sh["rho_air_ref"].attrs["units"] == "kg m-3"
    assert p["humidity"]["q"].attrs["units"] == "g kg-1"
    assert p["temperature"]["theta_v"].attrs["long_name"].startswith("sonic virtual")
    back = M.products_to_legacy(p)
    for k in out:
        if k.endswith(("Header", "header")):
            assert back[k] == out[k], k
        else:
            assert np.array_equal(back[k], out[k], equal_nan=True), k


def test_products_drop_all_nan_columns_like_the_legacy_trim():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    out = {"tke": np.column_stack([t, [0.5, 0.6]]), "tkeHeader": ["time", "TKE_10.85"]}
    p = M.products_from_legacy(out, [10.85, 2.0])        # second sonic has no data
    back = M.products_to_legacy(p)
    assert back["tkeHeader"] == out["tkeHeader"]
    assert back["tke"].shape == (2, 2)


def test_raw_round_trip():
    t = DAY + np.arange(1, 5) / 86400.0
    raw = {"t": t, "z": np.array([10.85]), "uPF": np.arange(4.0)[:, None], "z_h2o": np.array([10.85]),
           "rhov": np.ones((4, 1)), "P": np.column_stack([t, np.full(4, 101.0)])}
    ds = M.raw_from_legacy(raw, M.to_datetime64(t))
    assert ds["uPF"].dims == (M.TIME_HF, M.HEIGHT) and ds["rhov"].dims == (M.TIME_HF, "height_h2o")
    back = M.raw_to_legacy(ds)
    for k in raw:
        assert np.array_equal(back[k], raw[k]), k


def test_run_from_legacy_and_to_legacy_output():
    data = _data()
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    out = {"X_1Hz": np.column_stack([t, np.ones((2, 5))]), "X_1HzHeader": HEADERS[0],
           "X_1HzSpikeFlag": np.zeros((2, 6), bool), "X_1HzNanFlag": np.zeros((2, 6), bool),
           "spdAndDir": np.column_stack([t, [200.0, 210.0], [3.0, 4.0], [0.0, 1.0]]),
           "spdAndDirHeader": ["timeStamp", "wind_dir_10.85", "wind_speed_10.85", "shadow_flag_10.85"],
           "warnings": ["w1"]}
    run = M.run_from_legacy({"tableScanFrequency": [1.0, 1 / 1800]}, data, HEADERS, TABLES, SENSOR_INFO,
                            out, data_info=[["file: x"]])
    assert run.sonic_heights() == [10.85]
    assert "X_1Hz" in run.periods and "X_1Hz" in run.flags and run.wind is not None
    assert len(run.time_hf) == 120
    back = M.to_legacy_output(run)
    assert set(back) >= set(out)
    # the legacy <name>Header stays (labels, heights); units travel on the Run
    assert np.array_equal(back["X_1Hz"], out["X_1Hz"]) and back["X_1HzHeader"] == HEADERS[0][:2]
    assert np.array_equal(back["X_1HzSpikeFlag"], out["X_1HzSpikeFlag"])
    assert back["warnings"] == ["w1"]
