"""Tests for the tower-profile SiteInfo (sonics) and its consumers."""

import os

import pytest

from utespac.site_config import load_site_info, sonic_for, SonicLevel
from utespac.find_instruments import find_instruments

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRGA_DIR = os.path.join(REPO_ROOT, "siteIRGA20250723_20250828")


# ── sonic_for lookup ─────────────────────────────────────────────────────────

class TestSonicFor:
    LEVELS = [
        SonicLevel(height=32.18, orientation=243, manufacturer=1, hmp_height=30),
        SonicLevel(height=13.94, orientation=128, manufacturer=1, hmp_height=15),
        SonicLevel(height=6.35, orientation=134, manufacturer=1),
    ]

    def test_exact_match(self):
        assert sonic_for(self.LEVELS, 13.94).orientation == 128

    def test_within_tolerance(self):
        assert sonic_for(self.LEVELS, 13.945).hmp_height == 15

    def test_no_match(self):
        assert sonic_for(self.LEVELS, 99.0) is None

    def test_none_or_empty(self):
        assert sonic_for(None, 10.0) is None
        assert sonic_for([], 10.0) is None


# ── config parsing ───────────────────────────────────────────────────────────

def test_sonics_dicts_become_soniclevel(tmp_path):
    (tmp_path / "siteInfo.py").write_text(
        "sonics = [\n"
        "    {'height': 20.0, 'orientation': 100, 'manufacturer': 1, 'hmp_height': 22},\n"
        "    {'height': 5.0,  'orientation': 110, 'manufacturer': 2},\n"
        "]\n"
    )
    site = load_site_info(tmp_path)
    assert all(isinstance(s, SonicLevel) for s in site.sonics)
    assert site.sonics[0].hmp_height == 22
    assert site.sonics[1].hmp_height is None


def test_repo_irga_loads_as_profile():
    site = load_site_info(IRGA_DIR)
    assert site.sonics is not None and len(site.sonics) == 4
    # migrated off the legacy parallel lists
    assert site.sonicOrientation is None
    assert site.shiftsSonHeight is None
    heights = {s.height for s in site.sonics}
    assert heights == {32.18, 13.94, 6.35, 4.42}
    assert sonic_for(site.sonics, 32.18).hmp_height == 30
    assert sonic_for(site.sonics, 13.94).hmp_height == 15
    assert sonic_for(site.sonics, 6.35).hmp_height is None


# ── find_instruments consumes the profile ────────────────────────────────────

def test_find_instruments_maps_by_height_order_independent():
    """Header columns in a different order than the profile must still map
    each sonic to its own orientation/manufacturer (the old code was positional)."""
    info = {}
    load_site_info(IRGA_DIR).apply_to(info)
    names = ["Ux_4.42", "Ux_32.18", "Ux_6.35", "Ux_13.94"]  # shuffled vs profile
    si = find_instruments([[names]], {"u": "Ux*"}, info)
    by_h = {round(r[2], 2): (r[3], r[4]) for r in si["u"]}
    assert by_h[4.42] == (139.0, 1.0)
    assert by_h[32.18] == (243.0, 1.0)
    assert by_h[6.35] == (134.0, 1.0)
    assert by_h[13.94] == (128.0, 1.0)


def test_find_instruments_legacy_positional_fallback():
    """With no profile, orientation/manufacturer come from the parallel lists
    indexed by discovery order."""
    info = {"sonicOrientation": [36], "sonicManufact": [2]}
    si = find_instruments([[["Ux_51.5"]]], {"u": "Ux*"}, info)
    assert si["u"][0][3] == 36.0
    assert si["u"][0][4] == 2.0
