"""utespac.testkit: column statistics and field comparison."""

import numpy as np
import pytest

from utespac.testkit import column_stats, empty_stats, compare_field, get_header


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


# ── empty_stats / compare_field / get_header ─────────────────────────────────

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


def test_get_header_flat_nested_and_case():
    assert get_header({"Hheader": ["time", "a"]}, "H") == ["time", "a"]
    assert get_header({"XHeader": [["t", "b"], [None, 1.0]]}, "X") == ["t", "b"]
    assert get_header({}, "H") is None
