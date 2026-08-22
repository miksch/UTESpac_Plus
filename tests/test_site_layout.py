"""Tests for site discovery and input-folder resolution (data/ layout + legacy)."""

import os

import pytest

from utespac.site_config import (list_sites, resolve_site_dir, site_input_dir,
                                 has_site_info)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mk_site(root, name, layout):
    site = root / name
    site.mkdir()
    if layout in ("toml", "both"):
        (site / "siteInfo.toml").write_text("tower = 1\n")
    if layout in ("py", "both"):
        (site / "siteInfo.py").write_text("tower = 1\n")
    return site


def test_list_sites_by_siteinfo_and_legacy_prefix(tmp_path):
    _mk_site(tmp_path, "VAC001", "toml")
    _mk_site(tmp_path, "Gill", "py")
    _mk_site(tmp_path, "siteLegacy", None)      # legacy prefix, no siteInfo
    (tmp_path / "notes").mkdir()                # neither → excluded
    (tmp_path / "README.md").write_text("x")    # file → excluded
    assert list_sites(tmp_path) == ["Gill", "VAC001", "siteLegacy"]


def test_list_sites_missing_root():
    assert list_sites("/definitely/not/here") == []


def test_resolve_site_dir_accepts_folder_name_and_bare_legacy_id(tmp_path):
    _mk_site(tmp_path, "VAC001", "toml")
    _mk_site(tmp_path, "siteGill", "py")
    assert resolve_site_dir(tmp_path, "VAC001") == "VAC001"
    assert resolve_site_dir(tmp_path, "siteGill") == "siteGill"
    assert resolve_site_dir(tmp_path, "Gill") == "siteGill"
    with pytest.raises(FileNotFoundError, match="available"):
        resolve_site_dir(tmp_path, "Nope")


def test_site_input_dir_prefers_utespac_subfolder(tmp_path):
    site = _mk_site(tmp_path, "VAC001", "toml")
    assert site_input_dir(site) == site            # legacy: files at site root
    (site / "utespac").mkdir()
    assert site_input_dir(site) == site / "utespac"


def test_has_site_info(tmp_path):
    site = _mk_site(tmp_path, "A", "toml")
    assert has_site_info(site)
    assert not has_site_info(tmp_path)


def test_repo_data_tree_discovers_vac001():
    data_root = os.path.join(REPO_ROOT, "data")
    if not os.path.isdir(os.path.join(data_root, "VAC001")):
        pytest.skip("data/VAC001 not present on this machine")
    assert "VAC001" in list_sites(data_root)
    assert site_input_dir(os.path.join(data_root, "VAC001")).name == "utespac"
