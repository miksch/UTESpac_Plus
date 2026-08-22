"""Tests for utespac.site_config.SiteInfo and load_site_info."""

import os
import glob

import pytest

from utespac.site_config import SiteInfo, load_site_info

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_py_file(tmp_path):
    (tmp_path / "siteInfo.py").write_text(
        "tower = 210\n"
        "sonicOrientation = [36]\n"
        "tableScanFrequency = [10, 1/60]\n"
    )
    site = load_site_info(tmp_path)
    assert site.tower == 210
    assert site.sonicOrientation == [36]
    assert site.tableScanFrequency == [10, pytest.approx(1 / 60)]
    assert site.canopyHeight is None  # unset field


def test_unknown_key_warns(tmp_path):
    (tmp_path / "siteInfo.py").write_text("canpoyHeight = 19.3\n")  # typo
    with pytest.warns(UserWarning, match="canpoyHeight"):
        load_site_info(tmp_path)


def test_load_toml_file(tmp_path):
    (tmp_path / "siteInfo.toml").write_text(
        'tableNames = ["FMDOL_10Hz"]\nshiftzRef = true\n'
    )
    site = load_site_info(tmp_path)
    assert site.tableNames == ["FMDOL_10Hz"]
    assert site.shiftzRef is True


def test_toml_takes_precedence(tmp_path):
    (tmp_path / "siteInfo.py").write_text("tower = 1\n")
    (tmp_path / "siteInfo.toml").write_text("tower = 2\n")
    assert load_site_info(tmp_path).tower == 2


def test_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_site_info(tmp_path)


def test_apply_to_only_sets_present_fields():
    info = {"canopyHeight": 19.3, "tower": 210}
    SiteInfo(tower="001").apply_to(info)
    assert info["tower"] == "001"       # overridden
    assert info["canopyHeight"] == 19.3  # untouched


@pytest.mark.parametrize(
    "site_dir",
    sorted(glob.glob(os.path.join(REPO_ROOT, "site*")))
    + sorted(d for d in glob.glob(os.path.join(REPO_ROOT, "data", "*"))
             if glob.glob(os.path.join(d, "siteInfo.*"))),
    ids=os.path.basename,
)
def test_repo_site_files_load_cleanly(site_dir):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # unknown keys must not warn
        site = load_site_info(site_dir)
    assert site.tableNames, "tableNames must be set for every site"
    assert len(site.tableScanFrequency) == len(site.tableNames)

def test_slope_geometry_fields(tmp_path):
    (tmp_path / "siteInfo.toml").write_text(
        'angle = 8.2\ndownslopeAspect = 30\nslopeAxis = "v"\n'
    )
    site = load_site_info(tmp_path)
    assert site.angle == 8.2
    assert site.downslopeAspect == 30
    assert site.slopeAxis == "v"
    info = site.apply_to({})
    assert info["downslopeAspect"] == 30 and info["slopeAxis"] == "v"


def test_slope_axis_validated(tmp_path):
    (tmp_path / "siteInfo.toml").write_text('slopeAxis = "w"\n')
    with pytest.raises(ValueError, match="slopeAxis"):
        load_site_info(tmp_path)
