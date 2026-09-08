"""Tests for utespac.export_hf (the HF netCDF of a Run)."""

from datetime import datetime

import numpy as np
import pytest

from utespac import names
from utespac.campbell_date import datetime_to_matlab_datenum
from utespac.export_hf import write_hf
from utespac.model import Run, Sensors, raw_from_legacy, to_datetime64
from utespac.pf_info import PFRecord, PFTable

netCDF4 = pytest.importorskip("netCDF4")

N, FS = 600, 20.0                       # 30 s of 20 Hz
T0 = datetime_to_matlab_datenum(datetime(2024, 6, 1, 0, 0)) + 1 / (FS * 86400)
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
    te = np.array([datetime_to_matlab_datenum(datetime(2024, 6, 1, 0, 30))])
    return {"tableNames": ["X_20Hz"],
            "X_20Hz": np.column_stack([te, [1.0], [2.0], [3.0]]),
            "X_20HzHeader": [["TIMESTAMP", "Ux_10.85", "Ux_32.18", "T_Sonic_10.85"], [None, 10.85, 32.18, 10.85]],
            "X_20HzSpikeFlag": np.array([[False, True, False, False]]),
            "X_20HzNanFlag": np.array([[False, False, False, False]]),
            "momentum": np.column_stack([te, [0.04], [0.09]]),
            "momentumHeader": ["time", "Tau_pf_10.85", "Tau_pf_32.18"],
            "obukhov": np.column_stack([te, [-50.0], [-80.0]]),
            "obukhovHeader": ["time", "L_10.85", "L_32.18"],
            "spdAndDir": np.column_stack([te, [205.0], [1.0], [215.0], [0.0]]),
            "spdAndDirHeader": ["timeStamp", "wind_dir_10.85", "shadow_flag_10.85",
                                "wind_dir_32.18", "shadow_flag_32.18"],
            "flux_qc": np.column_stack([te, [0.0], [2.0]]),
            "flux_qcHeader": ["time", "H_ssitc_10.85", "H_ssitc_32.18"],
            "dataInfo": [["file: X_20Hz_20240601000000_20240603000000.txt"],
                         ["10.85m b0=0.01 b1=0.02 b2=0.03 pitch=1 roll=2 deg"]]}


def _info(pf="global", detrend="constant"):
    return {"siteFolder": "X", "date": "2024_06_01", "avgPer": 30, "UTESpacVersion": "5.0-Python",
            "PF": {"globalCalculation": pf}, "detrendingFormat": detrend,
            "latitude": 41.15, "longitude": -98.92, "tableScanFrequency": [20, 1 / 1800],
            "canopyHeight": 6.0, "siteElevation": 550.0, "displacementHeight": 0.0}


def _run(raw, info, pf_table=None):
    """A Run carrying only what write_hf reads: site facts, raw products, notes, planar fit."""
    return Run(site=info, sensors=Sensors(), table_names=["X_20Hz"], headers={},
               raw=raw_from_legacy(raw, to_datetime64(raw["t"])), notes=_avg()["dataInfo"],
               pf_table=pf_table)


def test_write_hf(tmp_path):
    pf = PFTable([PFRecord(10.85, "2024-06-01", "2024-06-16", 0.0, 0.0, 0.04, -0.08, 0.03)])
    out = write_hf(_run(_raw(), _info(), pf), tmp_path / "x.nc", output=_avg(), dtype="f8")
    with netCDF4.Dataset(out) as ds:
        assert len(ds.dimensions["time"]) == N and list(ds["height"][:]) == [10.85, 32.18]
        assert ds.getncattr("utespac_format") == "utespac-hf-2"
        assert ds.getncattr("site_id") == "X" and ds.getncattr("pf_type") == "GPF"
        assert ds.getncattr("detrend_upstream") == "constant"
        assert ds.getncattr("sampling_frequency_hz") == 20.0
        assert ds.getncattr("sampling_frequency_hz_measured") == pytest.approx(20.0, abs=1e-3)
        assert ds.getncattr("flux_averaging_s") == 1800.0 and ds.getncattr("canopy_height") == 6.0
        assert ds.getncattr("source_files") == "X_20Hz_20240601000000_20240603000000.txt"
        # heights sorted ascending: column 0 is 10.85 m (u = 1.0), column 1 is 32.18 m (u = 2.0)
        assert ds["u_pf"][0, 0] == 1.0 and ds["u_pf"][0, 1] == 2.0
        assert ds["ts"].units == "degC" and ds["rho_h2o"].units == "g m-3"
        # no z_h2o and one column against two sonic heights: height unknown -> NaN
        assert np.isnan(np.ma.filled(ds["height_h2o"][:], np.nan)[0])
        assert ds["p"][0] == 100.5
        t0 = netCDF4.num2date(ds["time"][0], ds["time"].units, ds["time"].calendar)
        assert (t0.hour, t0.minute, t0.second, t0.microsecond) == (0, 0, 0, 50000)
        # per-window ancillaries on record x height
        assert ds["ustar_pf"].shape == (1, 2)
        assert ds["ustar_pf"][0, 0] == pytest.approx(0.2) and ds["ustar_pf"][0, 1] == pytest.approx(0.3)
        assert ds["L"][0, 1] == -80.0 and ds["wdir"][0, 0] == 205.0 and ds["wind_flag"][0, 0] == 1.0
        assert ds["spike_flag"][0, 0] == 1.0 and ds["spike_flag"][0, 1] == 0.0
        assert ds["nan_flag"][0, 1] == 0.0
        assert ds["H_ssitc"][0, 1] == 2.0 and "Tau_ssitc" not in ds.variables
        assert "rho_h2o_prime" not in ds.variables
        pfg = ds.groups["planar_fit"]
        assert pfg["b1"][0] == -0.08 and pfg["date_end"][0] == "2024-06-16"


def test_lpf_coefficients_from_notes_and_primes(tmp_path):
    raw = _raw()
    raw["z_h2o"] = np.array([10.85])
    with pytest.warns(UserWarning, match="heights unknown"):   # rhoCO2 has no z_co2 and 1 col vs 2 heights
        out = write_hf(_run(raw, _info("local", "linear")), tmp_path / "X_hf_LPF_LinDet_2024_06_01.nc",
                       output=_avg(), include_primes=True)
    with netCDF4.Dataset(out) as ds:
        assert ds.getncattr("pf_type") == "LPF" and ds.getncattr("detrend_upstream") == "linear"
        assert ds["height_h2o"][0] == 10.85 and "rho_h2o_prime" in ds.variables
        assert ds["u_pf"].dtype == np.float32
        pfg = ds.groups["planar_fit"]
        assert pfg["height"][0] == 10.85 and pfg["b0"][0] == 0.01 and pfg["date_start"][0] == "2024-06-01"


def test_write_hf_needs_raw_products(tmp_path):
    run = Run(site=_info(), sensors=Sensors(), table_names=[], headers={})
    with pytest.raises(ValueError, match="raw products"):
        write_hf(run, tmp_path / "x.nc")


def test_hf_variables_use_the_new_names(tmp_path):
    """utespac-hf-2: every variable is a names.HF_VARIABLES target, no old name survives."""
    raw = _raw()
    raw["z_h2o"] = np.array([10.85])
    raw["z_co2"] = np.array([10.85])
    out = write_hf(_run(raw, _info()), tmp_path / "x.nc", output=_avg(), dtype="f8")
    with netCDF4.Dataset(out) as ds:
        got = set(ds.variables)
        assert ds.getncattr("utespac_format") == "utespac-hf-2"
        assert {"u_pf", "v_pf", "w_pf", "u_tilt", "v_tilt", "w_tilt", "ts", "theta_v",
                "wind_dir", "wind_speed", "rho_h2o", "rho_co2", "p",
                "height_h2o", "height_co2", "ustar_pf", "L", "H_ssitc"} <= got
        stale = {old for old, new in names.HF_VARIABLES.items() if old != new} \
            | {"height_rhov", "height_rhoCO2"}
        assert not (got & stale), sorted(got & stale)
        assert not [v for v in got if v.startswith("ssitc_")]
        for var in got:
            assert names.is_valid_name(var), var
