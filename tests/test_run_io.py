"""utespac.run_io: the utespac-run-3 file round trip, the utespac-run-2 read
shim and cross-date loading."""

import numpy as np
import pytest
import xarray as xr

from utespac import model as M
from utespac.pf_info import PFRecord, PFTable
from utespac.run_io import (FORMAT, FORMAT_V2, load_products, read_run, run_files, write_run)

DAY = 739406.0
HEADERS = [[["TIMESTAMP", "Ux_10.85", "Uy_10.85", "Uz_10.85"], [None, 10.85, 10.85, 10.85],
            ["", "m s-1", "m s-1", ""]]]
TABLES = ["X_1Hz"]
SENSOR_INFO = {"u": np.array([[0, 1, 10.85, 215.0, 1.0]]), "v": np.array([[0, 2, 10.85]]),
               "w": np.array([[0, 3, 10.85]])}


def _run(day=DAY, seed=0):
    rng = np.random.default_rng(seed)
    n = 120
    t = day + np.arange(1, n + 1) / 86400.0
    data = [np.column_stack([t, rng.normal(size=(n, 3))])]
    tp = day + np.arange(1, 3) * 1800 / 86400.0
    out = {"X_1Hz": np.column_stack([tp, np.ones((2, 3))]), "X_1HzHeader": HEADERS[0],
           "X_1HzSpikeFlag": np.zeros((2, 4), bool), "X_1HzNanFlag": np.eye(2, 4, dtype=bool),
           "spdAndDir": np.column_stack([tp, [200.0, 210.0], [3.0, 4.0], [0.0, 1.0]]),
           "spdAndDirHeader": ["timeStamp", "wind_dir_10.85", "wind_speed_10.85", "shadow_flag_10.85"],
           "sensible_heat": np.column_stack([tp, [1.2, 1.2], [1005.0, 1005.0], [0.1, 0.2]]),
           "sensible_heatHeader": ["time", "rho_air_ref", "cp_ref", "w_ts_cov_raw_10.85"],
           "rotatedSonic": np.ones((2, 3)), "PFSonic": np.full((2, 3), 2.0),
           "rotatedSonicHeader": ["u_pf_10.85", "v_pf_10.85", "w_pf_10.85"],
           "warnings": ["w1"]}
    rot = np.ones((n, 3))
    run = M.run_from_legacy({"avgPer": 30, "siteFolder": "X", "tower": 180}, data, HEADERS, TABLES,
                            SENSOR_INFO, out, rot, rot * 2, None, [["file: x"]],
                            pf_table=PFTable([PFRecord(10.85, "2024-06-03", "2024-06-03", 0, 0, 0.04, -0.08, 0.03)],
                                             site="X"))
    return run


def test_write_and_read_run_round_trip(tmp_path):
    run = _run()
    path = write_run(run, tmp_path / "X_30minAvg_LPF_LinDet_2024_06_03.nc", attrs={"site_id": "X"})
    back = read_run(path)
    assert back.attrs["utespac_format"] == FORMAT and back.attrs["site_id"] == "X"
    assert back.site == {"avgPer": 30, "siteFolder": "X", "tower": 180}
    assert back.table_names == TABLES and back.headers["X_1Hz"][0] == HEADERS[0][0]
    assert back.sensors.at("u", 10.85).orientation == 215.0 and back.sensors.at("v", 10.85).orientation is None
    assert back.sensors.at("u", 10.85).units == "m s-1" and back.sensors.at("w", 10.85).units is None
    assert back.header_units["X_1Hz"] == ["", "m s-1", "m s-1", ""]
    assert back.notes == [["file: x"]] and back.warnings == ["w1"]
    assert back.pf_table.records[0].b1 == -0.08
    xr.testing.assert_identical(back.periods["X_1Hz"].drop_attrs(), run.periods["X_1Hz"].drop_attrs())
    assert np.array_equal(back.flags["X_1Hz"]["nan"].values, run.flags["X_1Hz"]["nan"].values)
    assert back.flags["X_1Hz"]["nan"].dtype == bool
    xr.testing.assert_allclose(back.wind, run.wind)
    xr.testing.assert_allclose(back.products["sensible_heat"]["w_ts_cov_raw"],
                               run.products["sensible_heat"]["w_ts_cov_raw"])
    assert "rotated_mean" in back.rotation and "rotated" not in back.rotation   # HF winds not persisted
    assert back.tables == {}
    # the legacy view of the read file equals the legacy view of the original
    a, b = M.to_legacy_output(run), M.to_legacy_output(back)
    for k in ("X_1Hz", "spdAndDir", "sensible_heat", "rotatedSonic"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert a["sensible_heatHeader"] == b["sensible_heatHeader"]
    assert a["spdAndDirHeader"] == b["spdAndDirHeader"]


def test_reads_a_utespac_run_2_file_under_the_new_names(tmp_path):
    """A file written before the rename comes back with the current group and
    variable names, the third moment of Ts moved into ``transport``."""
    import netCDF4
    run = _run()
    t = run.products["sensible_heat"][M.TIME].values
    coords = {M.TIME: t, M.HEIGHT: [10.85]}
    run.products = {
        "H": xr.Dataset({"rho": (M.TIME, [1.2, 1.2]), "cp": (M.TIME, [1005.0, 1005.0]),
                         "Ts_w": ((M.TIME, M.HEIGHT), [[0.1], [0.2]])}, coords=coords),
        "sigma": xr.Dataset({"sigma_Tson": ((M.TIME, M.HEIGHT), [[0.3], [0.4]]),
                             "wPFP_TsonP_TsonP": ((M.TIME, M.HEIGHT), [[0.5], [0.6]])}, coords=coords),
        "specificHum": xr.Dataset({"virtualThetaAvg": ((M.TIME, M.HEIGHT), [[295.0], [295.1]])},
                                  coords=coords),
    }
    path = write_run(run, tmp_path / "X_30minAvg_LPF_LinDet_2024_06_03.nc")
    with netCDF4.Dataset(path, "a") as nc:
        nc.setncattr("utespac_format", FORMAT_V2)

    back = read_run(path)
    assert set(back.products) == {"sensible_heat", "sigma", "transport", "humidity"}
    sh = back.products["sensible_heat"]
    assert np.array_equal(sh["w_ts_cov_raw"].values, [[0.1], [0.2]])
    assert sh["rho_air_ref"].dims == (M.TIME,) and sh["cp_ref"].values[0] == 1005.0
    assert sh["w_ts_cov_raw"].attrs["units"] == "K m s-1"
    assert "ts_var_transport_pf" in back.products["transport"]     # moved out of sigma
    assert "ts_var_transport_pf" not in back.products["sigma"]
    assert "ts_sigma" in back.products["sigma"]
    assert "theta_v_slow" in back.products["humidity"]
    legacy = M.to_legacy_output(back)
    assert legacy["sensible_heatHeader"] == ["time", "rho_air_ref", "cp_ref", "w_ts_cov_raw_10.85"]
    assert legacy["humidityHeader"] == ["time", "theta_v_slow_10.85"]


def test_seconds_resolution_time_survives_the_round_trip(tmp_path):
    run = _run()
    ds = run.products["sensible_heat"]
    run.products["sensible_heat"] = ds.assign_coords(time=ds["time"].values.astype("datetime64[s]"))
    path = write_run(run, tmp_path / "X_30minAvg_LPF_LinDet_2024_06_03.nc")
    back = read_run(path)
    t = back.products["sensible_heat"]["time"].values
    assert not np.isnat(t).any()
    assert np.array_equal(t.astype("datetime64[ns]"), ds["time"].values.astype("datetime64[ns]"))


def test_load_products_concatenates_dates(tmp_path):
    out_dir = tmp_path / "X" / "output"
    out_dir.mkdir(parents=True)
    (tmp_path / "X" / "siteInfo.toml").write_text("tower = 180\n")
    for k, day in enumerate((DAY, DAY + 1)):
        write_run(_run(day, seed=k), out_dir / f"X_30minAvg_LPF_LinDet_2024_06_0{3 + k}.nc")
    (out_dir / "other.nc").write_bytes(b"not a netcdf")
    assert len(run_files(tmp_path, "X", avg_per=30, qualifier="LPF")) == 2
    prod = load_products(tmp_path, "X", avg_per=30, qualifier="LPF")
    assert prod["sensible_heat"].sizes[M.TIME] == 4
    assert np.all(np.diff(prod["sensible_heat"][M.TIME].values).astype("timedelta64[m]")
                  > np.timedelta64(0, "m"))
    sh = prod["sensible_heat"]
    assert "w_ts_cov_raw" in sh and sh["w_ts_cov_raw"].dims == (M.TIME, M.HEIGHT)
    with pytest.raises(FileNotFoundError):
        load_products(tmp_path, "X", qualifier="GPF")


def test_save_data_writes_run_file_and_csv_only(tmp_path):
    """save_data: the run netCDF always, CSV on request, no pickle; the HF file
    only when the run carries raw products."""
    import os
    from utespac.save_data import save_data
    (tmp_path / "X").mkdir()
    run = _run()
    run.site.update({"rootFolder": str(tmp_path), "date": "2024_06_03", "saveCSV": True,
                     "saveRawConditionedData": True, "PF": {"globalCalculation": "local"},
                     "detrendingFormat": "linear", "latitude": 38.3, "longitude": -121.9})
    paths = save_data(run)
    assert set(paths) == {"nc", "csv"}           # raw is None -> no HF file
    assert paths["nc"].endswith(os.path.join("X", "output", "X_30minAvg_LPF_LinDet_2024_06_03.nc"))
    assert sorted(f for f in os.listdir(tmp_path / "X" / "output") if f != "csv") == \
        ["X_30minAvg_LPF_LinDet_2024_06_03.nc"]
    assert (tmp_path / "X" / "output" / "csv"
            / "X_30minAvg_LPF_LinDet_2024_06_03_sensible_heat.csv").exists()
    back = read_run(paths["nc"])
    assert back.attrs["site_id"] == "X" and back.attrs["pf_type"] == "LPF"
    assert np.array_equal(M.to_legacy_output(back)["sensible_heat"],
                          M.to_legacy_output(run)["sensible_heat"], equal_nan=True)
