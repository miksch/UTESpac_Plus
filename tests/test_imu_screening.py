"""Tests for the IMU record-screening helpers in utespac/raw_processing/imu.py."""

import numpy as np
import pandas as pd
import pytest

from utespac.raw_processing.imu import (
    SPIKE_ACCEL_TOL_G, blocks, clean_imu, derive_mount_tilt, flag_spikes,
    frozen_attitude_mask, hold_exit_jumps, substitute_accel_tilt)


def _records(n, t0="2024-06-01 12:00:00", hz=10.0):
    """Quiet, level IMU records at *hz*."""
    t = pd.date_range(t0, periods=n, freq=pd.Timedelta(seconds=1 / hz))
    return pd.DataFrame({
        "time": t, "ax": 0.0, "ay": 0.0, "az": 1.0,
        "gx": 0.0, "gy": 0.0, "gz": 0.0,
        "roll": -9.86, "pitch": -1.68, "yaw": 100.0,
    })


# --------------------------------------------------------------------------
# flag_spikes
# --------------------------------------------------------------------------

def test_flag_spikes_each_limit():
    d = _records(7)
    d.loc[1, "az"] = 1.0 + SPIKE_ACCEL_TOL_G + 0.01
    d.loc[2, "gy"] = -150.0
    d.loc[3, "roll"] = 46.0
    d.loc[4, "pitch"] = -46.0
    d.loc[5, "yaw"] = 181.0
    d.loc[6, "az"] = 1.0 + SPIKE_ACCEL_TOL_G - 0.01      # inside the limit
    assert flag_spikes(d).tolist() == [False, True, True, True, True, True, False]


def test_flag_spikes_limits_are_arguments():
    d = _records(3)
    d.loc[1, "gx"] = 50.0
    d.loc[2, "az"] = 9.80665
    assert not flag_spikes(d.iloc[:2]).any()
    assert flag_spikes(d.iloc[:2], gyro_max=40.0).tolist() == [False, True]
    assert not flag_spikes(d.iloc[[0, 2]].assign(az=9.80665), gravity=9.80665,
                           accel_tol=0.3).any()


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

def test_blocks_split_on_gap():
    t = pd.to_datetime(["2024-06-01 00:00:00", "2024-06-01 00:00:01",
                        "2024-06-01 00:05:00", "2024-06-01 00:05:01",
                        "2024-06-01 00:05:02"])
    assert blocks(t, 120.0) == [(0, 1), (2, 4)]
    assert blocks(t, 600.0) == [(0, 4)]


# --------------------------------------------------------------------------
# clean_imu
# --------------------------------------------------------------------------

def test_clean_imu_inclusive_and_averaging():
    d = _records(6)
    d.loc[3, "time"] = d.loc[2, "time"]              # duplicate stamp
    d.loc[2, "ax"], d.loc[3, "ax"] = 0.02, 0.04
    d.loc[4, "gz"] = 500.0                           # spike
    start = d.loc[1, "time"]

    out = clean_imu(d, start, inclusive=True, average_duplicates=True)
    assert out["time"].tolist() == [d.loc[1, "time"], d.loc[2, "time"], d.loc[5, "time"]]
    assert out.loc[1, "ax"] == pytest.approx(0.03)
    assert isinstance(out.index, pd.RangeIndex)

    out = clean_imu(d, start, inclusive=False, average_duplicates=False)
    assert out["time"].tolist() == [d.loc[2, "time"], d.loc[2, "time"], d.loc[5, "time"]]


def test_clean_imu_concatenates_per_card_starts():
    a = _records(5, "2024-06-01 16:18:59.8")
    b = _records(5, "2024-06-14 18:24:59.8")
    out = clean_imu([b, a], ["2024-06-14 18:25", "2024-06-01 16:19"],
                    inclusive=False, average_duplicates=False)
    assert len(out) == 4
    assert out["time"].is_monotonic_increasing
    assert out["time"].iloc[0] > pd.Timestamp("2024-06-01 16:19")


# --------------------------------------------------------------------------
# hold_exit_jumps
# --------------------------------------------------------------------------

def test_hold_exit_jumps_wraps_and_measures_holds():
    d = _records(1200)
    d.loc[100:, "yaw"] = 101.0          # held 10 s, then +1
    d.loc[300:, "yaw"] = 359.0          # held 20 s
    d.loc[500:, "yaw"] = 1.0            # held 20 s, then +2 across the wrap
    ex = hold_exit_jumps(d, 120.0)
    assert ex["hold_s"].to_numpy() == pytest.approx([10.0, 20.0, 20.0])
    assert ex["djump"].to_numpy() == pytest.approx([1.0, -102.0, 2.0])


def test_hold_exit_jumps_skips_short_blocks():
    d = _records(1200)
    d.loc[100:, "yaw"] = 101.0
    d.loc[300:, "yaw"] = 102.0
    d.loc[500:, "yaw"] = 103.0
    assert len(hold_exit_jumps(d, 120.0, min_records=500)) == 3
    with pytest.raises(ValueError):
        hold_exit_jumps(d, 120.0, min_records=5000)


# --------------------------------------------------------------------------
# frozen attitude and accelerometer tilt
# --------------------------------------------------------------------------

def test_frozen_attitude_mask_threshold():
    d = _records(400)
    d["roll"] = np.arange(400) * 0.01
    d.loc[50:149, "roll"] = d.loc[50, "roll"]        # 99 intervals = 9.9 s
    d.loc[200:350, "roll"] = d.loc[200, "roll"]      # 150 intervals = 15 s
    mask, dur = frozen_attitude_mask(d, 10.0)
    assert mask[200:351].all() and not mask[50:150].any()
    assert mask.sum() == 151
    assert dur.max() == pytest.approx(15.0)
    mask, _ = frozen_attitude_mask(d, 9.8)
    assert mask[50:150].all()


def test_substitute_accel_tilt():
    d = _records(3)
    d.loc[:, ["ax", "ay", "az"]] = [[-np.sin(np.radians(5.0)), 0.0, np.cos(np.radians(5.0))]] * 3
    frozen = np.array([False, True, False])
    out = substitute_accel_tilt(d, frozen)
    assert out.loc[1, "pitch"] == pytest.approx(5.0)
    assert out.loc[1, "roll"] == pytest.approx(0.0)
    assert out.loc[0, "roll"] == -9.86 and d.loc[1, "pitch"] == -1.68
    unchanged = substitute_accel_tilt(d, np.zeros(3, bool))
    pd.testing.assert_frame_equal(unchanged, d)
    assert unchanged is not d


def test_derive_mount_tilt_signs():
    d = _records(5)
    d["roll"] = [-9.0, -10.0, -9.86, -11.0, -8.0]
    roll, pitch = derive_mount_tilt(d, [-1.0, 1.0, -1.0])
    assert roll == pytest.approx(9.86) and pitch == pytest.approx(-1.68)
    assert isinstance(roll, float)
