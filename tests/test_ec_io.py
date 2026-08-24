"""ec_coherent.io: HF reader, window slicing, ancillary join, analysis writer."""

import numpy as np
import pytest
import xarray as xr

from ec_coherent import ECConfig, io as ecio
from ec_helpers import make_hf


@pytest.fixture(scope="module")
def hf_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("ec_io") / "SYN_hf_GPF_ConstDet_2023_07_06.nc"
    return make_hf(p, nan_block=(1, 10, 40))


def test_open_hf_reads_attrs_and_rejects_other_formats(hf_path, tmp_path):
    path, truth = hf_path
    with ecio.open_hf(path) as hf:
        assert hf.fs == 10.0 and hf.window_s == 300.0 and hf.n_per_window == 3000
        assert list(hf.heights) == [3.0, 10.0]
        assert len(hf.records) == 4 and hf.site_id == "SYN"
    other = tmp_path / "x.nc"
    xr.Dataset(attrs={"utespac_format": "utespac-run-2"}).to_netcdf(other)
    with pytest.raises(ValueError):
        ecio.open_hf(other)


def test_windows_cover_exact_samples_and_join_ancillaries(hf_path):
    path, truth = hf_path
    n_win = truth["n_win"]
    with ecio.open_hf(path) as hf:
        wins = list(ecio.iter_windows(hf))
        assert [w.index for w in wins] == [0, 1, 2, 3]
        for w in wins:
            assert w.n == n_win
            assert w.data["u"].shape == (n_win, 2)
            assert w.record - w.start == np.timedelta64(int((n_win - 1) * 100), "ms")
            assert set(w.ancillary) >= {"ustar", "L", "spike_flag"}
            assert w.ancillary["ustar"].shape == (2,) and w.ancillary["ustar"][0] == 0.45
        # exact sample slicing against the truth arrays (float32 storage)
        w1 = wins[1]
        assert np.allclose(w1.data["w"], truth["w"][n_win:2 * n_win], atol=1e-5)
        assert np.isnan(w1.data["u"][10:40]).all() and np.isfinite(w1.data["u"][40:]).all()
        # rhov lives on its own height dim and is mapped onto the sonic heights
        assert w1.data["rhov"].shape == (n_win, 2)
        assert np.allclose(w1.scalar_heights["rhov"], [3.0, 10.0])
        sub = list(ecio.iter_windows(hf, records=[3]))
        assert len(sub) == 1 and sub[0].index == 3


def test_init_output_and_write_group_roundtrip(hf_path, tmp_path):
    path, _ = hf_path
    cfg = ECConfig.from_config()
    out = tmp_path / "SYN_coherent.nc"
    with ecio.open_hf(path) as hf:
        ecio.init_output(str(out), hf, cfg)
        g = xr.Dataset({"x": (("record", "height"), np.ones((4, 2)))},
                       coords={"record": hf.records, "height": hf.heights})
        g.attrs["note"] = "test"
        ecio.write_group(str(out), "demo", g)
    root = xr.open_dataset(out)
    assert root.attrs["ec_coherent_format"] == "ec-coherent-1"
    assert root.attrs["source_file"].startswith("SYN_hf_")
    assert root.attrs["detrend_method"] == cfg.preprocess.detrend
    assert root.attrs["sampling_frequency_hz"] == 10.0
    root.close()
    back = ecio.read_group(str(out), "demo")
    assert back.attrs["note"] == "test" and float(back["x"].sum()) == 8.0
    assert ecio.output_path("a/b/SYN_hf_GPF_ConstDet_2023_07_06.nc").endswith(
        "SYN_coherent_GPF_ConstDet_2023_07_06.nc")


def test_config_resolution_and_unknown_keys(tmp_path):
    cfg = ECConfig.from_config()
    assert cfg.modules == ("spectra", "mrd", "quadrant", "octant", "ramps", "ampmod", "scales") \
        and cfg.spectra.taper == "boxcar"
    assert cfg.mrd.grid == "trim" and cfg.quadrant.hole_norm == "rms"
    assert cfg.ramps.wavelet == "mhat" and cfg.ramps.signals == ("Ts", "u", "e")
    assert cfg.spectra.nperseg is None
    cfg2 = ECConfig.from_config({"spectra": {"nperseg": 1024, "taper": "hann"}},
                                preprocess={"detrend": "linear"})
    assert cfg2.spectra.nperseg == 1024 and cfg2.preprocess.detrend == "linear"
    with pytest.raises(ValueError):
        ECConfig.from_config({"spectra": {"bogus": 1}})
    toml = tmp_path / "ec.toml"
    toml.write_text('[run]\nmodules = ["spectra"]\n[spectra]\nn_bins_per_decade = 5\n')
    assert ECConfig.from_config(str(toml)).spectra.n_bins_per_decade == 5
