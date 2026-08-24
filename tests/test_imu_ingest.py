"""Tests for raw_processing/imu.py (generic IMU ingest helpers)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "raw_processing"))
from imu import IMU_MAP, align_imu, imu_columns, load_imu_files  # noqa: E402


def test_imu_columns_default_and_suffix():
    cols = imu_columns()
    assert cols["IMU_Ax"] == "Ax" and set(cols) == set(IMU_MAP)
    cols = imu_columns({"IMU_Roll": "roll_deg"}, src_suffix="_1")
    assert cols == {"IMU_Roll": "roll_deg_1"}


def _write_imu_csv(path, t0, n, hz, roll):
    t = pd.date_range(t0, periods=n, freq=pd.Timedelta(seconds=1 / hz))
    pd.DataFrame({"TS": t.strftime("%Y-%m-%d %H:%M:%S.%f"),
                  "roll_deg": roll,
                  "pitch_deg": np.zeros(n)}).to_csv(path, index=False)


def test_load_imu_files_maps_and_lags(tmp_path):
    _write_imu_csv(tmp_path / "imu_001.csv", "2026-07-01 00:00:00", 100, 10.0,
                   np.arange(100.0))
    _write_imu_csv(tmp_path / "imu_002.csv", "2026-07-01 00:00:10", 100, 10.0,
                   100 + np.arange(100.0))
    df = load_imu_files(str(tmp_path / "imu_*.csv"),
                        {"IMU_Roll": "roll_deg", "IMU_Pitch": "pitch_deg",
                         "IMU_Yaw": "yaw_deg"},        # yaw missing -> NaN
                        time_column="TS", lag_s=0.5)
    assert list(df.columns) == ["IMU_Roll", "IMU_Pitch", "IMU_Yaw"]
    assert len(df) == 200 and df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2026-07-01 00:00:00.5")     # lag applied
    assert df["IMU_Roll"].iloc[150] == 150.0                        # files concatenated
    assert df["IMU_Yaw"].isna().all()


def test_load_imu_files_missing_pattern(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_imu_files(str(tmp_path / "nope_*.csv"), {"IMU_Roll": "roll_deg"})


def test_align_imu_nearest_within_tolerance():
    t_imu = pd.date_range("2026-07-01", periods=50, freq="100ms")
    imu = pd.DataFrame({"IMU_Roll": np.arange(50.0)}, index=t_imu)
    imu = imu.drop(imu.index[20:30])                    # a 1-s gap
    grid = pd.date_range("2026-07-01 00:00:00.02", periods=48, freq="100ms")
    out = align_imu(imu, grid, max_gap_s=0.05)
    assert len(out) == 48
    assert out["IMU_Roll"].iloc[0] == 0.0               # 20 ms off -> nearest
    assert out["IMU_Roll"].iloc[35] == 35.0
    assert out["IMU_Roll"].iloc[20:30].isna().all()     # gap stays NaN
    # default tolerance = one grid step: the gap center still stays NaN
    out2 = align_imu(imu, grid)
    assert np.isnan(out2["IMU_Roll"].iloc[24])
