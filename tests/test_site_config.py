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
        'latitude = 41.15\nlongitude = -98.92\n'
    )
    site = load_site_info(tmp_path)
    assert site.tableNames == ["FMDOL_10Hz"]
    assert site.shiftzRef is True
    assert site.latitude == 41.15 and site.longitude == -98.92


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


def test_split_layout_resolves_config_and_data_roots(tmp_path, monkeypatch):
    from utespac.find_files import find_files
    from utespac.run_config import RunConfig
    from utespac.save_data import output_dir
    monkeypatch.chdir(tmp_path)                   # no cwd config/ overrides
    cfg_site = tmp_path / "config" / "SiteS"
    data_site = tmp_path / "data" / "SiteS"
    (cfg_site / "utespac").mkdir(parents=True)
    (data_site / "utespac").mkdir(parents=True)
    (cfg_site / "siteInfo.toml").write_text("tower = 7\n")
    (data_site / "siteInfo.toml").write_text("tower = 99\n")    # decoy: data root
    header = '"TIMESTAMP","Ux_10.85"\n'
    (cfg_site / "utespac" / "SiteS_1Hz_header.dat").write_text(header)   # decoy
    (cfg_site / "utespac" / "SiteS_1Hz_20240601000000.txt").write_text("x\n")
    inp = data_site / "utespac"
    (inp / "SiteS_20Hz_header.dat").write_text(header)
    (inp / "SiteS_20Hz_20240601000000.txt").write_text("2024,153,0,0.0\n")

    config = RunConfig.from_config(rootFolder=str(tmp_path / "config"),
                                   dataFolder=str(tmp_path / "data"))
    _, data_files, table_names, info = find_files(config.to_info(), site="SiteS")
    assert info["siteFolder"] == "SiteS" and info["tower"] == 7
    assert table_names == ["SiteS_20Hz"]
    assert [os.path.normpath(f) for row in data_files for f in row if f] == \
        [str(inp / "SiteS_20Hz_20240601000000.txt")]
    assert (info["rootFolder"], info["dataFolder"]) == \
        (str(tmp_path / "config"), str(tmp_path / "data"))
    assert os.path.normpath(output_dir(info)) == str(data_site / "output")
