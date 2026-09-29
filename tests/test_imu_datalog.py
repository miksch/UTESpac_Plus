"""Tests for utespac/raw_processing/imu_datalog.py (WitMotion DATALOG.TXT parser)."""

import numpy as np
import pandas as pd

from utespac.raw_processing.imu_datalog import CHANNELS, parse_datalog


def _write(path, lines):
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    return path


def test_parse_datalog_records_and_unpadded_milliseconds(tmp_path):
    path = _write(tmp_path / "DATALOG.TXT", [
        "A: 0.01 -0.02 0.99",
        "G: 1.0 2.0 3.0",
        "E: -9.8 -1.7 101.5",
        "D: 2024-6-1 T: 18:11:25.35",
        "A: 0.02 -0.03 1.01",
        "G: 4.0 5.0 6.0",
        "E: -9.9 -1.6 101.6",
        "D: 2024-6-1 T: 18:11:25.135",
    ])
    df, stats = parse_datalog(path)
    assert list(df.columns) == ["time", *CHANNELS]
    assert stats["records_ok"] == 2 and stats["lines"] == 8
    # ".35" is 35 ms, not 350 ms
    assert df["time"].iloc[0] == pd.Timestamp("2024-06-01 18:11:25.035")
    assert df["time"].iloc[1] == pd.Timestamp("2024-06-01 18:11:25.135")
    np.testing.assert_array_equal(df.loc[0, ["ax", "ay", "az"]].to_numpy(float),
                                  [0.01, -0.02, 0.99])
    np.testing.assert_array_equal(df.loc[1, ["roll", "pitch", "yaw"]].to_numpy(float),
                                  [-9.9, -1.6, 101.6])


def test_parse_datalog_drops_and_counts(tmp_path):
    path = _write(tmp_path / "DATALOG.TXT", [
        # pre-RTC stamp
        "A: 0 0 1", "G: 0 0 0", "E: 0 0 0", "D: 200-0-0 T: 0:0:1.0",
        # incomplete block (no G line)
        "A: 0 0 1", "E: 0 0 0", "D: 2024-6-1 T: 12:00:00.0",
        # malformed numeric field, then the block is incomplete
        "A: 0 x 1", "G: 0 0 0", "E: 0 0 0", "D: 2024-6-1 T: 12:00:00.100",
        # malformed stamp
        "A: 0 0 1", "G: 0 0 0", "E: 0 0 0", "D: 2024-6-1 12:00:00",
        "noise",
        # good record, written out of time order
        "A: 0 0 1", "G: 0 0 0", "E: 1 2 3", "D: 2024-6-1 T: 12:00:01.0",
        "A: 0 0 1", "G: 0 0 0", "E: 4 5 6", "D: 2024-6-1 T: 12:00:00.500",
    ])
    df, stats = parse_datalog(path)
    assert stats["records_ok"] == 2
    assert stats["records_pre_rtc"] == 1
    assert stats["records_incomplete"] == 2
    assert stats["records_bad_field"] == 2
    assert stats["unknown_tag_lines"] == 1
    assert stats["bad_date_tokens"] == ["200-0-0"]
    assert df["time"].is_monotonic_increasing
    assert df["roll"].tolist() == [4.0, 1.0]
