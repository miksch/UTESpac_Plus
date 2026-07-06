"""Golden-file regression tests: Python .pkl outputs vs MATLAB .mat references.

Requires previously generated outputs from both pipelines under
UTESpac_MATLAB/<site>/output/ (or <repo>/<site>/output/ for .pkl files).
Set UTESPAC_ROOT to point elsewhere. Skips when no pairs are found, so the
unit-test suite stays green on machines without the reference data.
"""

import os
import warnings

import pytest

from utespac.testkit import (
    discover_pairs, load_pkl, load_mat_avg, load_mat_raw,
    compare_avg, compare_raw,
)

ROOT_PY = os.environ.get(
    "UTESPAC_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
MATLAB_DIR = os.path.join(ROOT_PY, "UTESpac_MATLAB")

# Per-(field, pf_mode) max_rel tolerances for known Python/MATLAB divergences
# (e.g. ddof conventions in the GPF spike cascade). Default applies otherwise.
DEFAULT_TOL_REL = 0.01
TOL_OVERRIDES = {
    # ("H", "GPF"): 0.05,
}

_PAIRS = discover_pairs(MATLAB_DIR, ROOT_PY, gpf_only=False)


def _tol_for(field, pf_mode):
    return TOL_OVERRIDES.get((field, pf_mode), DEFAULT_TOL_REL)


@pytest.mark.skipif(not _PAIRS, reason="no matched (pkl, mat) golden-file pairs found")
@pytest.mark.parametrize("pair", _PAIRS, ids=lambda p: p["stem"])
def test_output_matches_matlab(pair):
    pkl = load_pkl(pair["pkl_path"])
    if pair["output_type"] == "raw":
        mat_ref = load_mat_raw(pair["mat_path"])
        compare = compare_raw
    else:
        mat_ref = load_mat_avg(pair["mat_path"])
        compare = compare_avg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = compare(pkl, mat_ref)

    failures = [
        f"{r['field']}[{r['column']}] max_rel={r['max_rel']:.4f}"
        f" > {_tol_for(r['field'], pair['pf_mode'])}"
        for r in rows
        if r["status"].strip() in ("WARN", "BIG")
        and r["max_rel"] > _tol_for(r["field"], pair["pf_mode"])
    ]
    assert not failures, (
        f"{pair['stem']}: {len(failures)} column(s) diverge:\n  "
        + "\n  ".join(failures)
    )
