"""Pinned-fixture regression: the pipeline on the committed VAC001_1Hz day
reproduces the expected outputs in tests/fixtures/vac001_1hz/expected/.

This is the regression safety net that runs on any machine (MATLAB parity
was retired 2026-08-22; EddyPro is the external reference, see the audit
doc). A failure means a numeric or structural change in the pipeline:
either a defect, or a deliberate change that needs a ledger row in
tests/KNOWN_DIVERGENCES.md and a re-pin (build_fixture.py pin).
"""

import os
import sys
import warnings

import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "vac001_1hz")
sys.path.insert(0, FIXTURE_DIR)

from build_fixture import CONFIGS, compare, load_expected, run_fixture, stage_site  # noqa: E402


@pytest.fixture(scope="module")
def staged_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("vac001_1hz")
    stage_site(FIXTURE_DIR, str(root))
    return str(root)


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_pipeline_matches_pinned_output(staged_root, config_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg, raw = run_fixture(config_name, staged_root)
    expected, meta = load_expected(config_name)
    problems = compare(avg, raw, expected, meta)
    assert not problems, f"{config_name}:\n  " + "\n  ".join(problems)


def test_run_file_is_the_pickles_twin(tmp_path_factory):
    """The utespac-run-2 netCDF read back through get_data(fmt="nc") equals the pickle."""
    import numpy as np
    from utespac import RunConfig, get_data, run_utespac
    from build_fixture import SITE
    root = str(tmp_path_factory.mktemp("vac001_1hz_nc"))
    stage_site(FIXTURE_DIR, root)
    cfg = RunConfig.from_config(rootFolder=root, saveCSV=False, saveNetCDF=True,
                                saveRawConditionedData=False, **CONFIGS["LPF_LinDet"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_utespac(cfg, site=SITE, dates="all")
    assert result.ok and result.dates[0].paths["nc"].endswith(".nc")
    pkl = get_data(root, site=SITE, avg_per=30, qualifier="LPF", fmt="pkl")
    nc = get_data(root, site=SITE, avg_per=30, qualifier="LPF", fmt="nc")
    for key, val in pkl.items():
        if key in ("dataInfo",):
            assert nc[key] == val, key
        elif key.endswith(("Header", "header")):
            assert nc[key] == val, key
        elif isinstance(val, np.ndarray):
            assert key in nc, key
            assert val.shape == nc[key].shape, key
            assert np.array_equal(val.astype(float), nc[key].astype(float), equal_nan=True), key
