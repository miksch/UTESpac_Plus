"""Tests for get_data concatenation across files with differing column sets."""

import pickle
import warnings

import numpy as np
import pytest

from utespac.get_data import get_data


def _write(site_out, name, hdr, mat, extra=None):
    d = {"tableNames": ["H"], "H": mat, "Hheader": hdr}
    if extra:
        d.update(extra)
    with open(site_out / name, "wb") as fh:
        pickle.dump(d, fh)


@pytest.fixture
def site(tmp_path):
    s = tmp_path / "SiteA"
    s.mkdir()
    (s / "siteInfo.toml").write_text("tower = 1\n")
    out = s / "output"
    out.mkdir()
    return tmp_path, out


def test_missing_column_fills_only_that_column(site):
    root, out = site
    full = ["time", "a", "b"]
    _write(out, "X_30minAvg_LPF_01.pkl", full, np.array([[1., 10., 100.], [2., 20., 200.]]))
    _write(out, "X_30minAvg_LPF_02.pkl", ["time", "b"], np.array([[3., 300.], [4., 400.]]))
    _write(out, "X_30minAvg_LPF_03.pkl", full, np.array([[5., 50., 500.], [6., 60., 600.]]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = get_data(root, site="SiteA", avg_per=30, qualifier="LPF", fmt="pkl")

    assert d["Hheader"] == full
    np.testing.assert_array_equal(d["H"][:, 0], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(d["H"][:, 2], [100, 200, 300, 400, 500, 600])
    np.testing.assert_array_equal(d["H"][:, 1], [10, 20, np.nan, np.nan, 50, 60])


def test_new_column_widens_accumulated_block(site):
    root, out = site
    _write(out, "X_30minAvg_LPF_01.pkl", ["time", "a"], np.array([[1., 10.]]))
    _write(out, "X_30minAvg_LPF_02.pkl", ["time", "a", "c"], np.array([[2., 20., 7.]]))

    d = get_data(root, site="SiteA", avg_per=30, qualifier="LPF", fmt="pkl")

    assert d["Hheader"] == ["time", "a", "c"]
    np.testing.assert_array_equal(d["H"], [[1, 10, np.nan], [2, 20, 7]])


def test_unlabeled_field_keeps_shape_check(site):
    root, out = site
    hdr = ["time", "a"]
    _write(out, "X_30minAvg_LPF_01.pkl", hdr, np.array([[1., 10.]]),
           extra={"tau": np.zeros((1, 3))})
    _write(out, "X_30minAvg_LPF_02.pkl", hdr, np.array([[2., 20.]]),
           extra={"tau": np.zeros((1, 2))})

    with pytest.warns(UserWarning, match="NaN block"):
        d = get_data(root, site="SiteA", avg_per=30, qualifier="LPF", fmt="pkl")

    assert d["tau"].shape == (2, 3)
    assert np.all(np.isnan(d["tau"][1]))


def test_row_count_mismatch_still_nan_block(site):
    root, out = site
    hdr = ["time", "a"]
    _write(out, "X_30minAvg_LPF_01.pkl", hdr, np.array([[1., 10.], [2., 20.]]))
    # second file: tables agree on 2 rows but H has 1 → NaN rows inserted
    _write(out, "X_30minAvg_LPF_02.pkl", ["time", "b"], np.array([[3., 30.]]),
           extra={"tableNames": ["tau", "H"], "tau": np.zeros((2, 1))})
    # tableNames contains H and tau with different row counts → file skipped
    with pytest.warns(UserWarning, match="Row count inconsistent"):
        d = get_data(root, site="SiteA", avg_per=30, qualifier="LPF", fmt="pkl")
    assert d["H"].shape == (2, 2)
