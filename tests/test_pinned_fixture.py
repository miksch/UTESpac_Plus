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
