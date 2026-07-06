"""Unit tests for the utespac.testkit comparison engine."""

import os

import numpy as np
import pytest
import scipy.io as sio

from utespac.testkit import (
    column_stats, empty_stats, compare_field, compare_avg, compare_raw,
    get_header, load_mat, discover_pairs,
)


# ── column_stats ─────────────────────────────────────────────────────────────

class TestColumnStats:
    def test_identical_arrays(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        s = column_stats(x, x)
        assert s["status"].strip() == "OK"
        assert s["bias"] == 0.0
        assert s["max_rel"] == 0.0
        assert s["R2"] == pytest.approx(1.0)

    def test_all_nan_is_skip(self):
        x = np.full(5, np.nan)
        s = column_stats(x, x)
        assert s["status"] == "SKIP"
        assert s["N_valid"] == 0

    def test_nan_mismatch_counted(self):
        py = np.array([1.0, np.nan, 3.0])
        ref = np.array([1.0, 2.0, np.nan])
        s = column_stats(py, ref)
        assert s["N_nan_mismatch"] == 2
        assert s["N_valid"] == 1

    def test_length_mismatch_truncates(self):
        s = column_stats(np.ones(10), np.ones(5))
        assert s["N_total"] == 5

    def test_status_thresholds(self):
        ref = np.array([100.0, 100.0, 100.0, 100.0])
        ok = column_stats(ref * 1.001, ref, tol_rel=0.01)
        warn = column_stats(ref * 1.05, ref, tol_rel=0.01)
        big = column_stats(ref * 1.5, ref, tol_rel=0.01)
        assert ok["status"].strip() == "OK"
        assert warn["status"].strip() == "WARN"
        assert big["status"].strip() == "BIG"

    def test_bias_sign(self):
        ref = np.array([1.0, 2.0, 3.0])
        s = column_stats(ref + 0.5, ref)
        assert s["bias"] == pytest.approx(0.5)


# ── empty_stats / compare_field ──────────────────────────────────────────────

def test_empty_stats_keys_match_column_stats():
    real = column_stats(np.ones(3), np.ones(3))
    empty = empty_stats("MISS_PY")
    assert set(empty) == set(real)


def test_compare_field_labels_columns_from_header():
    py = np.column_stack([np.ones(4), np.zeros(4)])
    rows = compare_field("f", py, py, ["alpha", "beta"], 0.01, 1e-9)
    assert [r["column"] for r in rows] == ["alpha", "beta"]


def test_compare_field_falls_back_to_col_index():
    py = np.ones((4, 2))
    rows = compare_field("f", py, py, None, 0.01, 1e-9)
    assert [r["column"] for r in rows] == ["col0", "col1"]


# ── compare_avg ──────────────────────────────────────────────────────────────

class TestCompareAvg:
    def test_matching_fields_ok(self):
        arr = np.column_stack([np.arange(4.0), np.arange(4.0) * 2])
        pkl = {"specificHum": arr, "specificHumHeader": ["t", "q"]}
        rows = compare_avg(pkl, {"specificHum": arr})
        assert all(r["status"].strip() == "OK" for r in rows)

    def test_timestamp_col0_skipped(self):
        # 'H' is in HAS_TIMESTAMP_COL0: col 0 (timestamps, deliberately
        # different) must not be compared
        t_py = np.arange(4.0) + 700000.0
        t_ref = t_py + 999.0
        vals = np.arange(4.0)
        pkl = {"H": np.column_stack([t_py, vals]), "HHeader": ["t", "H"]}
        ref = {"H": np.column_stack([t_ref, vals])}
        rows = compare_avg(pkl, ref)
        assert len(rows) == 1
        assert rows[0]["column"] == "H"
        assert rows[0]["status"].strip() == "OK"

    def test_missing_field_reported(self):
        rows = compare_avg({"only_py": np.ones(3)}, {"only_ref": np.ones(3)})
        by_field = {r["field"]: r["status"] for r in rows}
        assert by_field["only_py"] == "MISS_REF"
        assert by_field["only_ref"] == "MISS_PY"

    def test_header_fields_not_compared(self):
        pkl = {"HHeader": ["t", "H"], "H": np.ones((3, 2))}
        rows = compare_avg(pkl, {"H": np.ones((3, 2))})
        assert all(r["field"] == "H" for r in rows)


# ── compare_raw ──────────────────────────────────────────────────────────────

def test_compare_raw_matching():
    x = np.arange(300.0)
    rows = compare_raw({"uPF": x}, {"uPF": x.reshape(-1, 1)})
    matched = [r for r in rows if r["field"] == "uPF"]
    assert len(matched) == 1
    assert matched[0]["status"].strip() == "OK"


# ── get_header ───────────────────────────────────────────────────────────────

def test_get_header_unwraps_nested():
    pkl = {"HHeader": [["a", "b"]]}
    assert get_header(pkl, "H") == ["a", "b"]


def test_get_header_missing_returns_none():
    assert get_header({}, "H") is None


# ── load_mat ─────────────────────────────────────────────────────────────────

def test_load_mat_roundtrip(tmp_path):
    path = str(tmp_path / "ref.mat")
    sio.savemat(path, {"output": {"H": np.ones((3, 2)), "L": np.arange(3.0)}})
    ref = load_mat(path, "output")
    assert set(ref) == {"H", "L"}
    np.testing.assert_allclose(ref["H"], np.ones((3, 2)))
    assert ref["L"].shape == (3, 1)  # 1-D fields become column vectors


# ── discover_pairs ───────────────────────────────────────────────────────────

def test_discover_pairs(tmp_path):
    matlab_dir = tmp_path / "UTESpac_MATLAB"
    out = matlab_dir / "siteFire1" / "output"
    out.mkdir(parents=True)
    stem = "Fire1_30minAvg_GPF_LinDet_2025_06_08"
    (out / f"{stem}.mat").touch()
    (out / f"{stem}.pkl").touch()
    (out / "Fire1_raw_LPF_LinDet_2025_06_08.mat").touch()  # unpaired

    pairs = discover_pairs(str(matlab_dir), str(tmp_path))
    assert len(pairs) == 1
    p = pairs[0]
    assert p["site"] == "siteFire1"
    assert p["pf_mode"] == "GPF"
    assert p["output_type"] == "avg"
    assert p["stem"] == stem
    assert os.path.isfile(p["mat_path"])


def test_discover_pairs_filters(tmp_path):
    matlab_dir = tmp_path / "UTESpac_MATLAB"
    out = matlab_dir / "siteX" / "output"
    out.mkdir(parents=True)
    for stem in ("X_30minAvg_GPF_d", "X_30minAvg_LPF_d", "X_raw_GPF_d"):
        (out / f"{stem}.mat").touch()
        (out / f"{stem}.pkl").touch()

    gpf = discover_pairs(str(matlab_dir), str(tmp_path), gpf_only=True)
    assert {p["pf_mode"] for p in gpf} == {"GPF"}
    raw = discover_pairs(str(matlab_dir), str(tmp_path), raw_only=True)
    assert {p["output_type"] for p in raw} == {"raw"}
    site = discover_pairs(str(matlab_dir), str(tmp_path), site_filter="siteNone")
    assert site == []
