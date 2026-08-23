"""Tests for utespac.export_hf (raw pickle -> HF netCDF)."""

import pickle
from datetime import datetime

import numpy as np
import pytest

from utespac.campbell_date import datetime_to_matlab_datenum
from utespac.export_hf import export_site, parse_raw_name, raw_to_netcdf, sibling_averaged
from utespac.pf_info import PFRecord, PFTable
from utespac.site_config import SiteInfo

netCDF4 = pytest.importorskip("netCDF4")

N, FS = 600, 20.0                       # 30 s of 20 Hz
T0 = datetime_to_matlab_datenum(datetime(2023, 7, 6, 0, 0)) + 1 / (FS * 86400)
T = T0 + np.arange(N) / (FS * 86400)
Z = np.array([32.18, 10.85])            # descending on purpose


def _raw():
    col = lambda a, b: np.column_stack([np.full(N, a), np.full(N, b)])
    return {"t": T, "z": Z, "uPF": col(2.0, 1.0), "vPF": col(0.0, 0.0), "wPF": col(0.1, 0.2),
            "u_tilt": col(2.1, 1.1), "v_tilt": col(0.5, 0.6), "w_tilt": col(0.1, 0.2),
            "WD": col(200, 210), "spd": col(2.0, 1.0), "sonTs": col(20.0, 19.0),
            "Theta_v_son": col(20.1, 19.1), "rhov": np.full((N, 1), 9.0),
            "rhoCO2": np.full((N, 1), 740.0), "rhovPrime": np.zeros((N, 1)),
            "P": np.column_stack([T, np.full(N, 100.5)])}


def _avg():
    te = np.array([datetime_to_matlab_datenum(datetime(2023, 7, 6, 0, 30))])
    return {"tableNames": ["X_20Hz"],
            "X_20Hz": np.column_stack([te, [1.0], [2.0], [3.0]]),
            "X_20HzHeader": [["TIMESTAMP", "Ux_10.85", "Ux_32.18", "T_Sonic_10.85"], [None, 10.85, 32.18, 10.85]],
            "X_20HzSpikeFlag": np.array([[False, True, False, False]]),
            "X_20HzNanFlag": np.array([[False, False, False, False]]),
            "tau": np.column_stack([te, [0.04], [0.09]]),
            "tauHeader": ["time", "10.85m :sqrt(uPF'wPF'^2+vPF'wPF'^2)", "32.18m :sqrt(uPF'wPF'^2+vPF'wPF'^2)"],
            "L": np.column_stack([te, [-50.0], [-80.0]]),
            "Lheader": ["time", "10.85m L:...", "32.18m L:..."],
            "spdAndDir": np.column_stack([te, [205.0], [1.0], [215.0], [0.0]]),
            "spdAndDirHeader": ["timeStamp", "10.85m direction", "10.85m flag 15<dir<55",
                                "32.18m direction", "32.18m flag 15<dir<55"],
            "fluxQC": np.column_stack([te, [0.0], [2.0]]),
            "fluxQCHeader": ["time", "10.85m:H_SSITC_TEST", "32.18m:H_SSITC_TEST"],
            "dataInfo": [["file: X_20Hz_20230706000000_20230708000000.txt"],
                         ["10.85m b0=0.01 b1=0.02 b2=0.03 pitch=1 roll=2 deg"]]}


def test_names():
    p = parse_raw_name("data/X/output/X_raw_GPF_ConstDet_2023_07_06.pkl")
    assert p == {"site": "X", "pf": "GPF", "det": "ConstDet", "date": "2023_07_06"}
    assert parse_raw_name("X_30minAvg_GPF_ConstDet_2023_07_06.pkl") is None


def test_raw_to_netcdf(tmp_path):
    raw_path = tmp_path / "X_raw_GPF_ConstDet_2023_07_06.pkl"
    avg_path = tmp_path / "X_30minAvg_GPF_ConstDet_2023_07_06.pkl"
    raw_path.write_bytes(pickle.dumps(_raw()))
    avg_path.write_bytes(pickle.dumps(_avg()))
    assert sibling_averaged(str(raw_path)) == str(avg_path)
    si = SiteInfo(latitude=38.3, longitude=-121.9, tableScanFrequency=[20, 1 / 1800],
                  canopyHeight=6.0, siteElevation=18.2, displacementHeight=0.0)
    pf = PFTable([PFRecord(10.85, "2023-07-06", "2023-07-21", 0.0, 0.0, 0.04, -0.08, 0.03)])
    out = raw_to_netcdf(str(raw_path), tmp_path / "x.nc", site_info=si, pf_table=pf, dtype="f8")
    with netCDF4.Dataset(out) as ds:
        assert len(ds.dimensions["time"]) == N and list(ds["height"][:]) == [10.85, 32.18]
        assert ds.getncattr("site_id") == "X" and ds.getncattr("pf_type") == "GPF"
        assert ds.getncattr("detrend_upstream") == "constant"
        assert ds.getncattr("sampling_frequency_hz") == 20.0
        assert ds.getncattr("sampling_frequency_hz_measured") == pytest.approx(20.0, abs=1e-3)
        assert ds.getncattr("flux_averaging_s") == 1800.0 and ds.getncattr("canopy_height") == 6.0
        # heights sorted ascending: column 0 is 10.85 m (u = 1.0), column 1 is 32.18 m (u = 2.0)
        assert ds["u"][0, 0] == 1.0 and ds["u"][0, 1] == 2.0
        assert ds["Ts"].units == "degC" and ds["rhov"].units == "g m-3"
        # no z_h2o and one column against two sonic heights: height unknown -> NaN
        assert np.isnan(np.ma.filled(ds["height_rhov"][:], np.nan)[0])
        assert ds["P"][0] == 100.5
        t0 = netCDF4.num2date(ds["time"][0], ds["time"].units, ds["time"].calendar)
        assert (t0.hour, t0.minute, t0.second, t0.microsecond) == (0, 0, 0, 50000)
        # per-window ancillaries on record x height
        assert ds["ustar"].shape == (1, 2)
        assert ds["ustar"][0, 0] == pytest.approx(0.2) and ds["ustar"][0, 1] == pytest.approx(0.3)
        assert ds["L"][0, 1] == -80.0 and ds["wdir"][0, 0] == 205.0 and ds["wind_flag"][0, 0] == 1.0
        assert ds["spike_flag"][0, 0] == 1.0 and ds["spike_flag"][0, 1] == 0.0
        assert ds["nan_flag"][0, 1] == 0.0
        assert ds["ssitc_h_ssitc_test"][0, 1] == 2.0
        assert "rhov_prime" not in ds.variables
        pfg = ds.groups["planar_fit"]
        assert pfg["b1"][0] == -0.08 and pfg["date_end"][0] == "2023-07-21"


def test_lpf_coefficients_from_data_info_and_export_site(tmp_path):
    site = tmp_path / "X"
    (site / "output").mkdir(parents=True)
    (site / "siteInfo.toml").write_text("latitude = 38.3\nlongitude = -121.9\n")
    raw = _raw()
    raw["z_h2o"] = np.array([10.85])
    (site / "output" / "X_raw_LPF_LinDet_2023_07_06.pkl").write_bytes(pickle.dumps(raw))
    (site / "output" / "X_30minAvg_LPF_LinDet_2023_07_06.pkl").write_bytes(pickle.dumps(_avg()))
    with pytest.warns(UserWarning, match="heights unknown"):   # rhoCO2 has no z_co2 and 1 col vs 2 heights
        written = export_site(tmp_path, "X", pf="LPF", det="LinDet", include_primes=True)
    assert written == [str(site / "output" / "X_hf_LPF_LinDet_2023_07_06.nc")]
    with netCDF4.Dataset(written[0]) as ds:
        assert ds.getncattr("pf_type") == "LPF" and ds.getncattr("detrend_upstream") == "linear"
        assert ds["height_rhov"][0] == 10.85 and "rhov_prime" in ds.variables
        pfg = ds.groups["planar_fit"]
        assert pfg["height"][0] == 10.85 and pfg["b0"][0] == 0.01 and pfg["date_start"][0] == "2023-07-06"
    with pytest.raises(FileNotFoundError):
        export_site(tmp_path, "X", pf="GPF")
