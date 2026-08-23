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


def test_products_on_disk_are_the_runs_view(tmp_path_factory):
    """The run netCDF through get_data equals the in-memory Run's legacy view, and
    the HF netCDF holds the raw products (float32) with the per-window ancillaries."""
    import numpy as np
    import xarray as xr
    from utespac import RunConfig, get_data, run_utespac
    from utespac.model import raw_to_legacy, to_legacy_output
    from build_fixture import SITE
    root = str(tmp_path_factory.mktemp("vac001_1hz_nc"))
    stage_site(FIXTURE_DIR, root)
    cfg = RunConfig.from_config(rootFolder=root, saveCSV=True, saveRawConditionedData=True,
                                **CONFIGS["LPF_LinDet"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = run_utespac(cfg, site=SITE, dates="all", keep_runs=True)
    assert result.ok
    date = result.dates[0]
    assert set(date.paths) == {"nc", "hf", "csv"}
    assert os.path.basename(date.paths["hf"]).startswith(f"{SITE}_hf_LPF_LinDet_")
    assert not any(f.endswith(".pkl") for f in os.listdir(os.path.dirname(date.paths["nc"])))

    mem = to_legacy_output(date.run)
    disk = get_data(root, site=SITE, avg_per=30, qualifier="LPF")
    for key, val in mem.items():
        if key.endswith(("Header", "header")):
            assert disk[key] == val, key
        elif isinstance(val, np.ndarray):
            assert val.shape == disk[key].shape, key
            assert np.array_equal(val.astype(float), disk[key].astype(float), equal_nan=True), key
    assert disk["dataInfo"] == date.run.notes

    raw = raw_to_legacy(date.run.raw)
    with xr.open_dataset(date.paths["hf"], engine="netcdf4") as hf:
        hf = hf.load()
    assert hf.attrs["utespac_format"] == "utespac-hf-1" and hf.attrs["pf_type"] == "LPF"
    assert hf["time"].values[0] == date.run.raw["time_hf"].values[0]
    for nc_name, raw_key in (("u", "uPF"), ("w", "wPF"), ("Ts", "sonTs"), ("rhov", "rhov")):
        got, exp = hf[nc_name].values, np.asarray(raw[raw_key], dtype=float)
        assert got.shape == exp.shape, nc_name
        m = ~np.isnan(exp)
        assert np.array_equal(np.isnan(got), ~m), nc_name
        assert np.allclose(got[m], exp[m], rtol=2e-7, atol=0), nc_name     # float32 storage
    assert {"ustar", "L", "wdir", "spike_flag", "nan_flag"} <= set(hf.data_vars)
    assert hf["ustar"].dims == ("record", "height") and hf.sizes["record"] == 48
