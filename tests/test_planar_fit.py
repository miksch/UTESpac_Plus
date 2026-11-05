"""Planar-fit mechanics: coefficient contract, offset removal, and what the
legacy MATLAB indexing did (audit findings 1-2)."""

import json
import logging

import numpy as np
import pytest

import utespac.find_global_pf as fgp
from utespac.pf_coefficients import pf_coefficients
from utespac.pf_info import PFRecord, PFTable
from utespac.prompts import ScriptedPFSelection
from utespac.rotation import apply_planar_fit as _apply_pf, pf_matrix as _build_pf_matrix
from utespac.run_config import PFConfig


def _synthetic_plane(b0=0.04, b1=-0.08, b2=0.03, n=500, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(1, 8, n)
    v = rng.uniform(-3, 3, n)
    w = b0 + b1 * u + b2 * v
    return u, v, w


def test_pf_coefficients_returns_b0_b1_b2_in_order():
    u, v, w = _synthetic_plane()
    coef = pf_coefficients(np.column_stack([u, v, w, np.full_like(u, 200.0)]), [])
    b = coef["degrees_0_to_0"]
    assert b == pytest.approx([0.04, -0.08, 0.03], abs=1e-9)


def test_apply_pf_with_correct_coefficients_zeroes_mean_w():
    u, v, w = _synthetic_plane()
    b = pf_coefficients(np.column_stack([u, v, w, np.zeros_like(u)]), [])["degrees_0_to_0"]
    P = _build_pf_matrix(b[1], b[2])
    rotated = _apply_pf(P, b[0], u, v, w)
    # every point lies on the fitted plane, so rotated w vanishes pointwise
    assert np.allclose(rotated[:, 2], 0.0, atol=1e-9)


def test_b0_not_removed_leaves_offset_in_w():
    u, v, w = _synthetic_plane(b0=0.04)
    b = pf_coefficients(np.column_stack([u, v, w, np.zeros_like(u)]), [])["degrees_0_to_0"]
    P = _build_pf_matrix(b[1], b[2])
    rotated = _apply_pf(P, 0.0, u, v, w)   # the legacy (MATLAB) path
    assert abs(rotated[:, 2].mean() - 0.04 * P[2, 2]) < 1e-9


def test_legacy_index_bug_applies_wrong_tilt():
    """MATLAB sonicRotation read [b0, b1] as (b1, b2); pitch is then wrong."""
    u, v, w = _synthetic_plane(b0=0.04, b1=-0.08, b2=0.03)
    b = pf_coefficients(np.column_stack([u, v, w, np.zeros_like(u)]), [])["degrees_0_to_0"]
    P_wrong = _build_pf_matrix(b[0], b[1])
    rotated = _apply_pf(P_wrong, 0.0, u, v, w)
    assert not np.allclose(rotated[:, 2], 0.0, atol=1e-3)


def test_plane_normal_maps_to_vertical():
    b1, b2 = -0.08, 0.03
    n = np.array([-b1, -b2, 1.0]) / np.sqrt(b1**2 + b2**2 + 1)
    P = _build_pf_matrix(b1, b2)
    assert np.allclose(P @ n, [0, 0, 1], atol=1e-12)


@pytest.mark.parametrize("directions, span", [
    ([10.0, 50.0, 130.0], 120.0),
    ([350.0, 5.0, 15.0], 25.0),            # wraps through north
    ([0.0, 90.0, 180.0, 270.0], 270.0),
    ([42.0], 0.0),
    ([np.nan, 340.0, 20.0], 40.0),         # NaN ignored, wraps
    ([370.0, 20.0], 10.0),                 # taken modulo 360
    ([-10.0, 370.0], 20.0),                # modulo 360 from both sides
    ([], np.nan),
])
def test_direction_span(directions, span):
    # sums and differences of exact degrees: round-off only
    np.testing.assert_allclose(fgp.direction_span(directions), span,
                               atol=1e-9)


# Periods at sector edges; 61 has a NaN u, 200 a NaN v, the next a NaN
# direction and the last a NaN w, so none of these counts. The counted
# directions 0, 30, 60, 359 span 61 deg (largest gap 60 -> 359).
_EDGE_DIRS = [0.0, 30.0, 60.0, 61.0, 200.0, 359.0, np.nan, 120.0]
_EDGE_U = [1.0] * 3 + [np.nan] + [1.0] * 4
_EDGE_V = [1.0] * 4 + [np.nan] + [1.0] * 3
_EDGE_W = [0.0] * 7 + [np.nan]


@pytest.mark.parametrize("bins, expected", [
    ([], {"degrees_0_to_0": (4, 61.0)}),
    # (0, 60] holds 30 and 60; (60, 0] wraps north and holds 359 and 0
    ([0.0, 60.0], {"degrees_0_to_60": (2, 30.0),
                   "degrees_60_to_0": (2, 1.0)}),
    ([90.0], {"degrees_90_to_90": (4, 61.0)}),     # one edge: all of it
])
def test_pf_sample_sectors(bins, expected):
    """Each sector counts its finite periods, closed on its upper edge."""
    got = fgp.pf_sample(_EDGE_U, _EDGE_V, _EDGE_W, _EDGE_DIRS, bins)
    assert got == expected


# Global planar fit on a synthetic single-sonic LPF record: planted plane,
# planted wind speeds and directions (Campbell axes, bearing 0, so the
# direction of (s cos d, -s sin d) is d).
_Z = 2.5
_PLANE = (0.04, -0.08, 0.03)


def _gpf_data(speed, direction):
    speed = np.asarray(speed, dtype=float)
    rad = np.radians(np.asarray(direction, dtype=float))
    u, v = speed * np.cos(rad), -speed * np.sin(rad)
    w = _PLANE[0] + _PLANE[1] * u + _PLANE[2] * v
    t = 739404.0 + np.arange(speed.size) / 48.0
    return {
        "spdAndDirHeader": ["time", f"wind_speed_{_Z:g}",
                            f"shadow_flag_{_Z:g}"],
        "spdAndDir": np.column_stack([t, speed, np.zeros_like(t)]),
        "tableNames": ["sonic"],
        "sonicHeader": ["time", f"Ux_{_Z}", f"Uy_{_Z}", f"Uz_{_Z}"],
        "sonic": np.column_stack([t, u, v, w]),
    }


def _run_gpf(tmp_path, monkeypatch, speed, direction, prompter, **pf):
    data = _gpf_data(speed, direction)
    monkeypatch.setattr(fgp, "get_data", lambda *a, **k: data)
    monkeypatch.setattr(fgp, "list_sites", lambda root: ["SITE"])
    monkeypatch.setattr(fgp, "data_root", lambda info: str(tmp_path))
    (tmp_path / "SITE").mkdir()
    info = {"rootFolder": str(tmp_path), "siteFolder": "SITE", "avgPer": 30,
            "PF": PFConfig(recalculateGlobalCoefficients=True, **pf).to_info()}
    template = {"u": "Ux_*", "v": "Uy_*", "w": "Uz_*"}
    sensor_info = {"u": np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])}
    coef = fgp.find_global_pf(info, template, sensor_info, prompter=prompter)
    recs = {f"degrees_{r.sector_lo:g}_to_{r.sector_hi:g}": r
            for r in PFTable.load(tmp_path / "SITE" / "PFinfo.json").records}
    return coef[f"cm_{round(_Z * 100)}"]["day_739404to739405"], recs


# 0..354 deg, span 354
_ALL_DIRS = np.arange(60) * 6.0
# 350..15 deg, span 25
_BAND_DIRS = (350.0 + np.arange(60) * 25.0 / 59) % 360.0
# 1..5 m/s, all pass 0.5..20
_ABOVE = 1.0 + (np.arange(60) % 5)
# 0.1..0.45 m/s, none pass 0.5
_BELOW = 0.1 + 0.35 * (np.arange(60) % 5) / 4
# dropped by the 0.5 m/s filter
_CALM = np.full(50, 0.2)


def _sector_dirs(span):
    """40 periods over *span* deg in (0, 60], 40 over 240 deg in (60, 0]."""
    return np.r_[15.0 + np.linspace(0.0, span, 40),
                 np.linspace(100.0, 340.0, 40)]


_ALL = ScriptedPFSelection()
_SECTORS = ScriptedPFSelection(default_bins=[0.0, 60.0])
# day by day, first day only: its 48 periods, 0..282 deg
_FIRST_DAY = ScriptedPFSelection(use_all=False, accept_days={_Z: [739404.0]})
_FIRST_DAY_SECTORS = ScriptedPFSelection(default_bins=[0.0, 60.0],
                                         use_all=False,
                                         accept_days={_Z: [739404.0]})


# case: speed, direction, prompter, pf, sector, n, span, needed span, flagged,
# unguarded fit recovers the plane
_GUARD_CASES = {
    # well sampled, plus 10 calm periods the wind filter drops
    "well_sampled": (np.r_[_ABOVE, _CALM[:10]],
                     np.r_[_ALL_DIRS, np.zeros(10)], _ALL, {},
                     "degrees_0_to_0", 60, 354.0, 90.0, False, True),
    "all_calm": (_BELOW, _ALL_DIRS, _ALL, {},
                 "degrees_0_to_0", 0, np.nan, 90.0, True, False),
    "narrow_band": (_ABOVE, _BAND_DIRS, _ALL, {},
                    "degrees_0_to_0", 60, 25.0, 90.0, True, True),
    # a 20 deg minimum span admits the 25 deg band
    "narrow_band_min_span_20": (_ABOVE, _BAND_DIRS, _ALL,
                                {"globalCalcMinDirSpan": 20.0},
                                "degrees_0_to_0", 60, 25.0, 20.0, False,
                                True),
    "override_rescues_calm": (_BELOW, _ALL_DIRS, _ALL,
                              {"globalCalcMinWindByHeight": {_Z: 0.05}},
                              "degrees_0_to_0", 60, 354.0, 90.0, False,
                              True),
    # 10 periods round the compass: wide enough, too few
    "few_periods": (np.r_[_ABOVE[:10], _CALM],
                    np.r_[np.arange(10) * 36.0, np.zeros(50)], _ALL, {},
                    "degrees_0_to_0", 10, 324.0, 90.0, True, True),
    # a 60 deg sector needs half its width, 30 deg
    "sector_span_29": (np.r_[_ABOVE, _ABOVE[:20]], _sector_dirs(29.0),
                       _SECTORS, {},
                       "degrees_0_to_60", 40, 29.0, 30.0, True, True),
    "sector_span_31": (np.r_[_ABOVE, _ABOVE[:20]], _sector_dirs(31.0),
                       _SECTORS, {},
                       "degrees_0_to_60", 40, 31.0, 30.0, False, True),
    # the guard sees the accepted days' 48 periods, not the window's 60
    "day_by_day": (_ABOVE, _ALL_DIRS, _FIRST_DAY,
                   {"globalCalcMinPeriods": 50},
                   "degrees_0_to_0", 48, 282.0, 90.0, True, True),
    # first day's 48 periods at 3..285 deg; (0, 60] holds 3, 9, ..., 57
    "day_by_day_sectors": (_ABOVE, _ALL_DIRS + 3.0, _FIRST_DAY_SECTORS, {},
                           "degrees_0_to_60", 10, 54.0, 30.0, True, True),
}


@pytest.mark.parametrize("action", ["warn", "nan"])
@pytest.mark.parametrize("case", list(_GUARD_CASES))
def test_global_pf_sample_guard(tmp_path, monkeypatch, caplog, case, action):
    """The fit records its sample and warns, or NaNs, a thin one."""
    (speed, direction, prompter, pf, sector, n, span, need, flagged,
     fits) = _GUARD_CASES[case]
    with caplog.at_level(logging.WARNING, logger="utespac"):
        coef, recs = _run_gpf(tmp_path, monkeypatch, speed, direction,
                              prompter, globalCalcSampleAction=action, **pf)
    coef, rec = coef[sector], recs[sector]
    msgs = [r.getMessage() for r in caplog.records
            if f"sector {sector}:" in r.getMessage()]
    # the sample is recorded under either action; planted, exact
    assert rec.n_periods == n
    np.testing.assert_allclose(rec.dir_span, span, atol=1e-9)
    np.testing.assert_allclose([rec.b0, rec.b1, rec.b2], coef, atol=0)
    if flagged and action == "nan":
        assert np.isnan(coef).all()
    elif fits:
        # noise-free planted plane: the unguarded fit recovers it to round-off
        np.testing.assert_allclose(coef, _PLANE, atol=1e-9)
    else:
        assert np.isnan(coef).all()       # nothing passes the wind filter
    assert len(msgs) == int(flagged)
    if not flagged:
        return
    msg = msgs[0]
    min_n = pf.get("globalCalcMinPeriods", 30)
    min_span = pf.get("globalCalcMinDirSpan", 90.0)
    assert f"{_Z:g} m" in msg and f"{n} period(s)" in msg
    assert f"over a {span:.1f} deg direction span" in msg
    assert (f"globalCalcMinPeriods={min_n}, "
            f"globalCalcMinDirSpan={min_span:.1f} deg, "
            f"needs {need:.1f} deg for this sector" in msg)
    taken = "coefficients set to NaN" if action == "nan" else "fit kept"
    assert f"{taken} (globalCalcSampleAction='{action}')" in msg
    # more periods help only a sample short of periods
    assert ("globalCalcMinWindByHeight" in msg) == (n < min_n)


# case: sector, (n, span), flagged; thresholds 30 periods and 90 deg
_THRESHOLD_CASES = {
    "n_at_minimum": ("degrees_0_to_0", (30, 200.0), False),
    "span_at_minimum": ("degrees_0_to_0", (100, 90.0), False),
    # a 60 deg sector needs 30 deg
    "span_at_sector_need": ("degrees_0_to_60", (100, 30.0), False),
    # (300, 20] wraps north: 80 deg wide, needs 40 deg
    "span_at_wrapped_sector_need": ("degrees_300_to_20", (100, 40.0), False),
    "n_below": ("degrees_0_to_0", (29, 200.0), True),
    "narrow_at_n_minimum": ("degrees_0_to_0", (30, 10.0), True),
}


@pytest.mark.parametrize("action", ["warn", "nan"])
@pytest.mark.parametrize("case", list(_THRESHOLD_CASES))
def test_guard_pf_sample_thresholds(caplog, case, action):
    """A sample at a threshold passes; one below it is flagged."""
    sector, (n, span), flagged = _THRESHOLD_CASES[case]
    coef = {sector: np.array(_PLANE)}
    sample = {sector: (n, span)}
    # a well-sampled second sector, never flagged
    coef["degrees_60_to_0"] = np.array(_PLANE)
    sample["degrees_60_to_0"] = (100, 240.0)
    with caplog.at_level(logging.WARNING, logger="utespac"):
        out = fgp.guard_pf_sample(coef, sample, _Z, 30, 90.0, action)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == int(flagged)
    if flagged and action == "nan":
        assert np.isnan(out[sector]).all()
    else:
        np.testing.assert_array_equal(out[sector], _PLANE)
    np.testing.assert_array_equal(out["degrees_60_to_0"], _PLANE)
    if flagged:
        # more periods help only a sample short of periods
        assert ("globalCalcMinWindByHeight" in msgs[0]) == (n < 30)


def test_guard_pf_sample_rejects_unknown_action():
    """An action other than warn or nan is refused."""
    sample = {"degrees_0_to_0": (5, 10.0)}
    with pytest.raises(ValueError, match="must be 'warn' or 'nan', "
                                         "got 'drop'"):
        fgp.guard_pf_sample({"degrees_0_to_0": np.array(_PLANE)}, sample,
                            _Z, 30, 90.0, action="drop")


def _no_bare_constant(token):
    raise ValueError(f"non-strict JSON token {token}")


def test_pfinfo_sample_round_trip_is_strict_json(tmp_path):
    """Sample fields round-trip as strict JSON, a NaN span as null."""
    records = [
        PFRecord(_Z, "2024-06-01", "2024-06-02", 0.0, 60.0, *_PLANE,
                 0, float("nan")),
        PFRecord(_Z, "2024-06-01", "2024-06-02", 60.0, 0.0, *_PLANE,
                 40, 240.0),
        PFRecord(_Z, "2024-06-03", "2024-06-04", 0.0, 0.0, *_PLANE),
    ]
    path = tmp_path / "PFinfo.json"
    PFTable(records, site="SITE").save(path)
    doc = json.loads(path.read_text(encoding="utf-8"),
                     parse_constant=_no_bare_constant)
    assert [(r["n_periods"], r["dir_span"]) for r in doc["records"]] == \
        [(0, None), (40, 240.0), (None, None)]
    back = PFTable.load(path).records
    assert back[0].n_periods == 0
    assert isinstance(back[0].dir_span, float) and np.isnan(back[0].dir_span)
    assert back[1:] == records[1:]


def test_pfinfo_without_sample_keys_loads(tmp_path):
    """A PFinfo.json written before the sample fields loads them as None."""
    record = {"height": _Z, "date_start": "2024-06-01",
              "date_end": "2024-06-02", "sector_lo": 0.0, "sector_hi": 0.0,
              "b0": _PLANE[0], "b1": _PLANE[1], "b2": _PLANE[2]}
    path = tmp_path / "PFinfo.json"
    path.write_text(json.dumps({"format": "utespac-pfinfo-1", "site": "SITE",
                                "records": [record]}), encoding="utf-8")
    (back,) = PFTable.load(path).records
    assert (back.n_periods, back.dir_span) == (None, None)
    assert (back.b0, back.b1, back.b2) == _PLANE
