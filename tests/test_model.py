"""utespac.model: sensors, the Run, and the legacy converters (both directions)."""

import numpy as np
import pytest
import xarray as xr

from utespac import model as M
from utespac.campbell_date import MATLAB_EPOCH

DAY = 739406.0
HEADERS = [[["TIMESTAMP", "Ux_10.85", "Uy_10.85", "Uz_10.85", "T_Sonic_10.85", "H2O_10.85"],
            [None, 10.85, 10.85, 10.85, 10.85, 10.85]],
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


def test_wind_round_trip_keeps_sector_header():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    out = {"spdAndDir": np.column_stack([t, [200.0, 210.0], [3.0, 4.0], [0.0, 1.0]]),
           "spdAndDirHeader": ["timeStamp", "10.85m direction", "10.85m speed", "10.85m flag 15<dir<55"]}
    w = M.wind_from_legacy(out, [10.85])
    assert w["direction"].dims == (M.TIME, M.HEIGHT)
    assert w["sector_min"].values.tolist() == [15.0] and w["sector_max"].values.tolist() == [55.0]
    back = M.wind_to_legacy(w)
    assert back["spdAndDirHeader"] == out["spdAndDirHeader"]
    assert np.array_equal(back["spdAndDir"], out["spdAndDir"])


def test_products_round_trip_with_fixed_columns_and_trim():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    H = np.column_stack([t, [1.2, 1.2], [1005.0, 1005.0], [0.1, 0.2], [0.05, np.nan]])
    out = {"H": H, "Hheader": ["time", "rho", "cp", "10.85m son:Ts'w'", "10.85m son:Theta_v'wPF'"],
           "derivedT": np.column_stack([t, [20.0, 21.0]]),
           "derivedTheader": ["time", "10.85 m: theta_v_son"],
           "specificHum": np.column_stack([t, [7.5, 7.6], [295.0, 295.1]]),
           "specificHumHeader": ["time", "10.85 m: q(g/kg)", "10.85 m: virtualThetaAvg(K)"]}
    p = M.products_from_legacy(out, [10.85])
    assert p["H"]["Ts_w"].dims == (M.TIME, M.HEIGHT) and p["H"]["rho"].dims == (M.TIME,)
    assert p["H"]["Ts_w"].attrs["label"] == "{hn}m son:Ts'w'"
    assert p["specificHum"]["q"].attrs["units"] == "g/kg"
    back = M.products_to_legacy(p)
    for k in ("H", "Hheader", "derivedT", "derivedTheader", "specificHum", "specificHumHeader"):
        if k.endswith(("Header", "header")):
            assert back[k] == out[k], k
        else:
            assert np.array_equal(back[k], out[k], equal_nan=True), k


def test_products_drop_all_nan_columns_like_the_legacy_trim():
    t = DAY + np.arange(1, 3) * 1800 / 86400.0
    out = {"tke": np.column_stack([t, [0.5, 0.6]]), "tkeHeader": ["time", "10.85m :0.5(u'^2+v'^2+w'^2)"]}
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
           "spdAndDirHeader": ["timeStamp", "10.85m direction", "10.85m speed", "10.85m flag 15<dir<55"],
           "warnings": ["w1"]}
    run = M.run_from_legacy({"tableScanFrequency": [1.0, 1 / 1800]}, data, HEADERS, TABLES, SENSOR_INFO,
                            out, data_info=[["file: x"]])
    assert run.sonic_heights() == [10.85]
    assert "X_1Hz" in run.periods and "X_1Hz" in run.flags and run.wind is not None
    assert len(run.time_hf) == 120
    back = M.to_legacy_output(run)
    assert set(back) >= set(out)
    assert np.array_equal(back["X_1Hz"], out["X_1Hz"]) and back["X_1HzHeader"] == HEADERS[0]
    assert np.array_equal(back["X_1HzSpikeFlag"], out["X_1HzSpikeFlag"])
    assert back["warnings"] == ["w1"]
