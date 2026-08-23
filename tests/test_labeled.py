"""Tests for utespac.labeled (labeled tables, netCDF round trip) and the
``fmt="nc"`` path of get_data on ``utespac-averaged-1`` files."""

import pickle

import numpy as np
import pytest

from utespac import labeled
from utespac.campbell_date import datetime_to_matlab_datenum
from utespac.get_data import get_data, get_frames

netCDF4 = pytest.importorskip("netCDF4")

from datetime import datetime  # noqa: E402

T0 = datetime_to_matlab_datenum(datetime(2023, 7, 6, 0, 30))
T = T0 + np.arange(4) * (30 / 1440)


def _output():
    """A small averaged-output dict with every header shape the pipeline writes."""
    tbl = np.column_stack([T, np.arange(4.0), 10 + np.arange(4.0)])
    return {
        "tableNames": ["X_20Hz"],
        "X_20Hz": tbl,
        "X_20HzHeader": [["TIMESTAMP", "Ux_10.85", "T_Sonic_10.85"], [None, 10.85, 10.85]],
        "X_20HzSpikeFlag": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool),
        "H": np.column_stack([T, np.full(4, 1.2), [np.nan, 1, 2, 3]]),
        "Hheader": ["time", "rho", "10.85m son:Ts'w'"],
        "spdAndDir": np.column_stack([T, [200.0, 210, 220, 230]]),
        "spdAndDirHeader": ["timeStamp", "10.85m direction"],
        "rotatedSonic": np.ones((4, 3)),
        "rotatedSonicHeader": ["10.85m:u", "10.85m:v", "10.85m:w"],
        "epsilon": np.column_stack([T]),
        "epsilonHeader": ["time"],
        "dataInfo": [["file: X_20Hz_20230706000000_20230708000000.txt", "beg date: 06-Jul-2023"]],
        "warnings": [],
    }


def test_parse_label():
    assert labeled.parse_label("10.85m son:Ts'w'") == (10.85, "son:Ts'w'")
    assert labeled.parse_label("10.85 m: q(g/kg)") == (10.85, "q(g/kg)")
    assert labeled.parse_label("10.85m:u") == (10.85, "u")
    assert labeled.parse_label("4.42m flag 15<dir<55") == (4.42, "flag 15<dir<55")
    assert labeled.parse_label("rho") == (None, "rho")
    assert labeled.parse_label("time") == (None, "time")


def test_tables_shapes_and_time():
    tabs = labeled.tables(_output())
    assert set(tabs) == {"X_20Hz", "X_20HzSpikeFlag", "H", "spdAndDir", "rotatedSonic", "epsilon"}
    h = tabs["H"]
    assert h.time_in_col0 and h.labels == ["rho", "10.85m son:Ts'w'"]
    assert h.heights == [None, 10.85] and h.values.shape == (4, 2)
    assert h.time[0] == np.datetime64("2023-07-06T00:30:00.000")
    assert h.time_label == "time" and h.header_key == "Hheader"
    # nested table header: heights from the header row, time column dropped
    x = tabs["X_20Hz"]
    assert x.header_nested and x.labels == ["Ux_10.85", "T_Sonic_10.85"] and x.heights == [10.85, 10.85]
    # flags share the table header (time column kept, it is a flag column too)
    f = tabs["X_20HzSpikeFlag"]
    assert f.header_of == "X_20Hz" and not f.time_in_col0 and f.labels[0] == "TIMESTAMP"
    assert f.values.dtype == bool and np.array_equal(f.time, x.time)
    # no time column of its own: the reference axis
    assert np.array_equal(tabs["rotatedSonic"].time, x.time)
    assert tabs["epsilon"].values.shape == (4, 0)


def test_to_frames():
    fr = labeled.to_frames(_output())
    assert fr["H"].index.name == "time" and list(fr["H"].columns) == ["rho", "10.85m son:Ts'w'"]
    assert fr["H"].loc["2023-07-06 01:30", "10.85m son:Ts'w'"] == 2


def _assert_same(a, b):
    for k, v in a.items():
        assert k in b, k
        if isinstance(v, np.ndarray):
            assert v.shape == b[k].shape, k
            if v.dtype == bool:
                assert np.array_equal(v, b[k]), k
            else:
                np.testing.assert_allclose(b[k], v, rtol=0, atol=1e-3 / 86400, equal_nan=True,
                                           err_msg=k)
        else:
            assert b[k] == v, k
    assert not (set(b) - set(a))


def test_netcdf_round_trip(tmp_path):
    out = _output()
    info = {"siteFolder": "X", "latitude": 38.3, "longitude": -121.9, "avgPer": 30,
            "PF": {"globalCalculation": "global"}, "detrendingFormat": "constant",
            "tableScanFrequency": [20, 1 / 1800], "UTESpacVersion": "5.0-Python"}
    path = labeled.write_netcdf(out, tmp_path / "x.nc", attrs=labeled.run_attrs(info, out))
    back = labeled.read_netcdf(path)
    _assert_same(out, back)
    with netCDF4.Dataset(path) as ds:
        assert ds.getncattr("utespac_format") == labeled.FORMAT
        assert ds.getncattr("pf_type") == "GPF" and ds.getncattr("latitude") == 38.3
        assert ds.getncattr("sampling_frequency_hz") == 20.0
        assert ds.getncattr("source_files") == "X_20Hz_20230706000000_20230708000000.txt"
        assert ds["time"].units.startswith("milliseconds since 1970-01-01")
        assert list(ds["H_column"][:]) == ["rho", "10.85m son:Ts'w'"]
        hh = np.ma.filled(ds["H_height"][:], np.nan)
        assert np.isnan(hh[0]) and hh[1] == 10.85
        assert ds["X_20HzSpikeFlag"].dtype == np.int8
    frames = labeled.read_netcdf(path, frames=True)
    assert frames["spdAndDir"].iloc[2, 0] == 220


def test_run_attrs_warns_without_lat_lon():
    with pytest.warns(UserWarning, match="latitude/longitude"):
        attrs = labeled.run_attrs({"siteFolder": "X", "avgPer": 30, "PF": {"globalCalculation": "local"}})
    assert np.isnan(attrs["latitude"]) and attrs["pf_type"] == "LPF"


def test_get_data_reads_netcdf_like_pkl(tmp_path):
    site = tmp_path / "VAC001"
    (site / "output").mkdir(parents=True)
    (site / "siteInfo.toml").write_text("tower = 1\n")
    out1 = _output()
    out2 = _output()
    for k in ("X_20Hz", "H", "spdAndDir", "epsilon"):        # next file: two hours later
        out2[k][:, 0] += 2 / 24
    for i, o in enumerate((out1, out2), 1):
        with open(site / "output" / f"X_30minAvg_LPF_ConstDet_0{i}.pkl", "wb") as fh:
            pickle.dump(o, fh)
        labeled.write_netcdf(o, site / "output" / f"X_30minAvg_LPF_ConstDet_0{i}.nc")
    from_pkl = get_data(tmp_path, site="VAC001", avg_per=30, qualifier="LPF", fmt="pkl")
    from_nc = get_data(tmp_path, site="VAC001", avg_per=30, qualifier="LPF", fmt="nc")
    _assert_same(from_pkl, from_nc)
    assert from_nc["H"].shape == (8, 3)
    fr = get_frames(tmp_path, site="VAC001", avg_per=30, qualifier="LPF", fmt="nc")
    assert fr["H"].shape == (8, 2) and fr["H"].index.name == "time"
    with pytest.raises(ValueError, match="fmt"):
        get_data(tmp_path, site="VAC001", fmt="csv")
