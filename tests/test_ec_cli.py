"""ec_coherent.cli: full default pipeline end-to-end, argv parsing, cross-group closure."""

import numpy as np
import pytest
import xarray as xr

from ec_coherent import cli, io as ecio
from ec_helpers import make_hf

GROUPS = ("spectra", "mrd", "quadrant", "octant", "ramps", "amplitude_mod",
          "scale_separation", "coherent_flux")


@pytest.fixture(scope="module")
def coherent(tmp_path_factory):
    """One synthetic HF file pushed through main() with the packaged defaults."""
    d = tmp_path_factory.mktemp("ec_cli")
    path, truth = make_hf(d / "SYN_hf_GPF_ConstDet_2023_07_06.nc")
    assert cli.main([path]) == 0
    return ecio.output_path(path), truth


def test_main_writes_every_default_group(coherent):
    out, _ = coherent
    root = xr.open_dataset(out)
    assert root.attrs["ec_coherent_format"] == "ec-coherent-1"
    assert root.attrs["modules"] == ",".join(
        ("spectra", "mrd", "quadrant", "octant", "ramps", "ampmod", "scales", "coherent_flux"))
    root.close()
    for g in GROUPS:
        ds = ecio.read_group(out, g)
        assert "record" in ds.dims and ds.sizes["record"] == 4, g


def test_closure_identities_hold_across_groups(coherent):
    out, _ = coherent
    sp = ecio.read_group(out, "spectra")
    band_sum = (sp["S_u"] * np.diff(sp["frequency_edges"])).sum("frequency")
    assert np.allclose(band_sum, sp["var_u"], rtol=1e-5)
    mrd = ecio.read_group(out, "mrd")
    assert np.allclose(mrd["D_uw"].sum("mr_scale"), mrd["cov_uw"], rtol=1e-4)
    q = ecio.read_group(out, "quadrant").isel(hole=0)
    assert np.allclose(q["S_frac_uw"].sum("quadrant"), 1.0, atol=1e-5)
    oc = ecio.read_group(out, "octant")
    assert np.allclose(oc["flux_frac_uwTs_uw"].sum("octant"), 1.0, atol=1e-5)
    sc = ecio.read_group(out, "scale_separation")
    assert np.allclose(sc["var_frac_u"].sum("scale_band"), 1.0, atol=1e-5)
    cf = ecio.read_group(out, "coherent_flux")
    ej, sw, cs = (cf[f"{k}_uw_u"] for k in ("F_ej", "F_sw", "F_cs"))
    m = np.isfinite(cs)
    assert m.any()
    assert np.allclose((ej + sw).values[m.values], cs.values[m.values], rtol=1e-6)


def test_records_and_module_subset_and_out(tmp_path):
    path, _ = make_hf(tmp_path / "SYN_hf_GPF_ConstDet_2023_07_08.nc")
    out = str(tmp_path / "subset.nc")
    assert cli.main([path, "--modules", "spectra", "--records", "0-1,3", "--out", out]) == 0
    ds = ecio.read_group(out, "spectra")
    # the record axis is kept full; unselected records stay NaN
    assert ds.sizes["record"] == 4
    finite = np.isfinite(ds["var_u"].isel(height=0)).values
    assert list(finite) == [True, True, False, True]
    with pytest.raises(OSError):
        ecio.read_group(out, "mrd")
    with pytest.raises(SystemExit):
        cli.main([path, path, "--out", out])
    with pytest.raises(ValueError):
        cli.run_file(path, cli.ECConfig.from_config(modules=("bogus",)))


def test_parse_records():
    assert cli._parse_records(None) is None
    assert cli._parse_records("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]
