"""utespac.run_io: the utespac-run-2 file round trip and cross-date loading."""

import numpy as np
import pytest
import xarray as xr

from utespac import model as M
from utespac.pf_info import PFRecord, PFTable
from utespac.run_io import FORMAT, load_products, read_run, run_files, write_run

DAY = 739406.0
HEADERS = [[["TIMESTAMP", "Ux_10.85", "Uy_10.85", "Uz_10.85"], [None, 10.85, 10.85, 10.85]]]
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
           "spdAndDirHeader": ["timeStamp", "10.85m direction", "10.85m speed", "10.85m flag 15<dir<55"],
           "H": np.column_stack([tp, [1.2, 1.2], [1005.0, 1005.0], [0.1, 0.2]]),
           "Hheader": ["time", "rho", "cp", "10.85m son:Ts'w'"],
           "rotatedSonic": np.ones((2, 3)), "PFSonic": np.full((2, 3), 2.0),
           "rotatedSonicHeader": ["10.85m:u", "10.85m:v", "10.85m:w"],
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
    assert back.notes == [["file: x"]] and back.warnings == ["w1"]
    assert back.pf_table.records[0].b1 == -0.08
    xr.testing.assert_identical(back.periods["X_1Hz"].drop_attrs(), run.periods["X_1Hz"].drop_attrs())
    assert np.array_equal(back.flags["X_1Hz"]["nan"].values, run.flags["X_1Hz"]["nan"].values)
    assert back.flags["X_1Hz"]["nan"].dtype == bool
    xr.testing.assert_allclose(back.wind, run.wind)
    xr.testing.assert_allclose(back.products["H"]["Ts_w"], run.products["H"]["Ts_w"])
    assert "rotated_mean" in back.rotation and "rotated" not in back.rotation   # HF winds not persisted
    assert back.tables == {}
    # the legacy view of the read file equals the legacy view of the original
    a, b = M.to_legacy_output(run), M.to_legacy_output(back)
    for k in ("X_1Hz", "spdAndDir", "H", "rotatedSonic"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert a["Hheader"] == b["Hheader"] and a["spdAndDirHeader"] == b["spdAndDirHeader"]


def test_load_products_concatenates_dates(tmp_path):
    out_dir = tmp_path / "X" / "output"
    out_dir.mkdir(parents=True)
    (tmp_path / "X" / "siteInfo.toml").write_text("tower = 180\n")
    for k, day in enumerate((DAY, DAY + 1)):
        write_run(_run(day, seed=k), out_dir / f"X_30minAvg_LPF_LinDet_2024_06_0{3 + k}.nc")
    (out_dir / "other.nc").write_bytes(b"not a netcdf")
    assert len(run_files(tmp_path, "X", avg_per=30, qualifier="LPF")) == 2
    prod = load_products(tmp_path, "X", avg_per=30, qualifier="LPF")
    assert prod["H"].sizes[M.TIME] == 4
    assert np.all(np.diff(prod["H"][M.TIME].values).astype("timedelta64[m]") > np.timedelta64(0, "m"))
    assert "Ts_w" in prod["H"] and prod["H"]["Ts_w"].dims == (M.TIME, M.HEIGHT)
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
    assert (tmp_path / "X" / "output" / "csv" / "X_30minAvg_LPF_LinDet_2024_06_03_H.csv").exists()
    back = read_run(paths["nc"])
    assert back.attrs["site_id"] == "X" and back.attrs["pf_type"] == "LPF"
    assert np.array_equal(M.to_legacy_output(back)["H"], M.to_legacy_output(run)["H"], equal_nan=True)
