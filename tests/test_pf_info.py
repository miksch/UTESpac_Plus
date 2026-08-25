"""Tests for utespac.pf_info (labeled planar-fit table) and its use by
find_global_pf (save_pf_info / load_pf_info)."""

import pickle

import numpy as np
import pytest

from utespac.find_global_pf import load_pf_info, save_pf_info
from utespac.pf_info import PFRecord, PFTable

LEGACY = {
    "cm_1085": {
        "day_739404to739419": {"degrees_0_to_0": np.array([0.0425, -0.0811, 0.0295])},
    },
    "cm_3218": {
        "day_739404to739411": {"degrees_90_to_270": np.array([0.01, 0.02, 0.03]),
                               "degrees_270_to_90": np.array([0.04, 0.05, 0.06])},
        "day_739412to739419": {"degrees_90_to_270": np.array([0.07, 0.08, 0.09]),
                               "degrees_270_to_90": np.array([0.10, 0.11, 0.12])},
    },
}


def _same_legacy(a, b):
    assert set(a) == set(b)
    for cm in a:
        if cm == "infoString":
            assert a[cm] == b[cm]
            continue
        assert list(a[cm]) == list(b[cm])
        for day in a[cm]:
            assert list(a[cm][day]) == list(b[cm][day])
            for sec in a[cm][day]:
                np.testing.assert_array_equal(a[cm][day][sec], b[cm][day][sec])


def test_from_legacy_records():
    t = PFTable.from_legacy(LEGACY, site="X")
    assert t.heights == [10.85, 32.18] and len(t.records) == 5
    r = t.records[0]
    assert r == PFRecord(10.85, "2024-06-01", "2024-06-16", 0.0, 0.0, 0.0425, -0.0811, 0.0295)
    assert t.records[1].sector_lo == 90 and t.records[1].sector_hi == 270


def test_legacy_round_trip_and_info_string():
    t = PFTable.from_legacy(LEGACY)
    back = t.to_legacy()
    assert "infoString" in back and back["infoString"][0][0] == "Global PF Info"
    _same_legacy(LEGACY, {k: v for k, v in back.items() if k != "infoString"})
    # a table loaded from a pickle carrying infoString keeps it verbatim
    leg2 = dict(LEGACY, infoString=[["Global PF Info", "custom"]])
    assert PFTable.from_legacy(leg2).to_legacy()["infoString"] == leg2["infoString"]


def test_json_round_trip(tmp_path):
    t = PFTable.from_legacy(LEGACY, site="X")
    t.save(tmp_path / "PFinfo.json")
    t2 = PFTable.load(tmp_path / "PFinfo.json")
    assert t2 == t
    (tmp_path / "bad.json").write_text('{"format": "other", "records": []}')
    with pytest.raises(ValueError, match="utespac-pfinfo-1"):
        PFTable.load(tmp_path / "bad.json")


def test_coefficients_lookup():
    t = PFTable.from_legacy(LEGACY)
    r = t.coefficients(32.18, np.datetime64("2024-06-05T12:00"), 180.0)
    assert r.b0 == 0.01                                    # first window, 90-270 sector
    r = t.coefficients(32.18, 739416.3, 10.0)              # datenum, wrap-around sector
    assert r.b0 == 0.10
    assert t.coefficients(10.85, np.datetime64("2024-06-05"), 123.0).b0 == 0.0425   # all sectors
    assert t.coefficients(10.85, np.datetime64("2024-08-01"), 0.0) is None
    assert t.coefficients(99.0, np.datetime64("2024-06-05"), 0.0) is None


def test_save_and_load_prefers_json(tmp_path):
    paths = save_pf_info(LEGACY, tmp_path, site="X")
    assert (tmp_path / "PFinfo.json").is_file()
    assert not (tmp_path / "PFinfo.pkl").exists()        # the pickle is no longer written
    loaded = load_pf_info(tmp_path)
    _same_legacy(LEGACY, {k: v for k, v in loaded.items() if k != "infoString"})
    # JSON is authoritative when both exist
    (tmp_path / "PFinfo.pkl").write_bytes(pickle.dumps({"cm_1": {"day_1to2": {"degrees_0_to_0": [1, 2, 3]}}}))
    assert set(load_pf_info(tmp_path)) == {"cm_1085", "cm_3218", "infoString"}
    # legacy pickle alone still loads, with infoString added
    (tmp_path / "PFinfo.json").unlink()
    leg = load_pf_info(tmp_path)
    assert set(leg) == {"cm_1", "infoString"}
    assert load_pf_info(tmp_path / "nowhere") is None
    assert paths["json"].endswith("PFinfo.json")
