"""Tests for the planar-fit prompter boundary (utespac.prompts) and the
pipeline entry-point argument handling (utespac.pipeline)."""

import numpy as np
import pytest

from utespac.prompts import ScriptedPFSelection, PFHeightContext
from utespac.pipeline import run_utespac
from utespac.run_config import RunConfig


def test_scripted_defaults_single_sector_all_dates():
    p = ScriptedPFSelection()
    t = np.arange(739000.0, 739010.0, 0.5)
    assert p.skip_height(10.85) is False
    assert p.bin_boundaries(10.85) == []
    assert p.date_barriers(10.85, t) == []
    assert p.use_all_dates(10.85, t[0], t[-1]) is True
    assert p.use_day(10.85, 739001.0) is True
    assert p.confirm({"cm_1085": {}}) is None
    # the show/begin/end hooks are no-ops
    ctx = PFHeightContext(10.85, "X", t, t, t, t, t, t)
    p.begin_height(ctx); p.end_selection(10.85)
    p.begin_day_by_day(10.85, []); p.show_day(10.85, 0, 3, {}, []); p.end_day_by_day(10.85)


def test_scripted_per_height_answers():
    p = ScriptedPFSelection(bins={10.85: [90.0, 270.0, 30.0]},
                            barriers={10.85: [739005.0, 738000.0]},   # one out of range
                            default_bins=[180.0], skip=[4.42],
                            use_all=False, accept_days={10.85: [739001.0]})
    t = np.arange(739000.0, 739010.0, 0.5)
    assert p.bin_boundaries(10.85) == [30.0, 90.0, 270.0]
    assert p.bin_boundaries(32.18) == [180.0]                 # default for unlisted heights
    assert p.date_barriers(10.85, t) == [739005.0]            # out-of-range dropped
    assert p.skip_height(4.42) is True and p.skip_height(10.85) is False
    assert p.use_all_dates(10.85, t[0], t[-1]) is False
    assert p.use_day(10.85, 739001.0) is True and p.use_day(10.85, 739002.0) is False
    assert p.use_day(32.18, 739002.0) is True                 # unlisted height: accept all


def test_run_utespac_argument_contract(tmp_path):
    with pytest.raises(ValueError, match="RunConfig or a legacy info"):
        run_utespac(site="X")
    with pytest.raises(ValueError, match="template"):
        run_utespac(info={"rootFolder": str(tmp_path)}, site="X")
    with pytest.raises(ValueError, match="either config or info"):
        run_utespac(RunConfig(rootFolder=str(tmp_path)), info={}, template={}, site="X")
    # an empty root has no sites: the first stage raises before any processing
    with pytest.raises(FileNotFoundError):
        run_utespac(RunConfig(rootFolder=str(tmp_path)), site="X")
