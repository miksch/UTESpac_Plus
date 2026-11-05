"""Tests for utespac.config (packaged TOMLs) and utespac.run_config (RunConfig)."""

import os

import pytest

from utespac import config as cfg
from utespac.run_config import RunConfig, QCConfig, PFConfig, FluxConfig


# The legacy info dict utespac_main.py carried before step 2 of the
# migration (values as committed in f020dcc), minus rootFolder and dataFolder.
LEGACY_INFO = {
    "UTESpacVersion": "5.0-Python",
    "avgPer": 30,
    "saveRawConditionedData": True,
    "saveCSV": True,
    "calcDissipation": False,
    "storeExtraStats": True,
    "detrendingFormat": "linear",
    "PF": {"globalCalculation": "global", "recalculateGlobalCoefficients": True,
           "avgPer": 30, "globalCalcMaxWind": 20, "globalCalcMinWind": 0.5,
           "globalCalcMinWindByHeight": {}, "globalCalcMinPeriods": 30,
           "globalCalcMinDirSpan": 90.0, "globalCalcSampleAction": "warn"},
    "qRef": 12,
    "stabilitySource": "hoegstroem1988",
    "lagSearch": "off",
    "lagMax": 2.0,
    "spectralCorrection": "off",
    "sonicPathLength": None,
    "irgaPathLength": None,
    "irgaSeparationLateral": None,
    "irgaSeparationLongitudinal": None,
    "spikeTest": {"method": "sigma", "madThreshold": 7.0,
                  "maxRuns": 20, "windowSizeFraction": 1,
                  "maxConsecutiveOutliers": 10,
                  "maxPercent": 2,
                  "spikeDef": {"u": 3.5, "v": 3.5, "w": 5.0, "Tson": 3.5, "fw": 3.5,
                               "irgaCO2": 3.5, "irgaH2O": 3.5, "KH2O": 3.5, "cup": 3.5,
                               "birdSpd": 3.5, "imuAx": 5.0, "imuAy": 5.0, "imuAz": 5.0,
                               "imuGx": 5.0, "imuGy": 5.0, "imuGz": 5.0, "imuRoll": 5.0,
                               "imuPitch": 5.0, "imuYaw": 5.0, "otherInstrument": 5.0}},
    "absoluteLimitsTest": {"u": [-50, 50], "v": [-50, 50], "w": [-10, 10], "Tson": [-20, 80],
                           "fw": [-20, 80], "irgaCO2": [0, 1500], "irgaH2O": [0, 50],
                           "KH2O": [0, 50], "cup": [0, 50], "birdSpd": [0, 50],
                           "imuAx": [-50, 50], "imuAy": [-50, 50], "imuAz": [-50, 50],
                           "imuGx": [-500, 500], "imuGy": [-500, 500], "imuGz": [-500, 500],
                           "imuRoll": [-180, 180], "imuPitch": [-180, 180],
                           "imuYaw": [-360, 360]},
    "windDirectionTest": {"envelopeSize": 20},
    "nanTest": {"maxPercent": 55},
    "diagnosticTest": {"H2OminSignal": 0.7, "CO2minSignal": 0.7, "meanGasDiagnosticLimit": 0.1,
                       "meanSonicDiagnosticLimit": 50, "meanLiGasDiagnosticLimit": 220},
}


def test_packaged_tomls_load():
    for name in ("run", "qc", "pf", "flux"):
        raw = cfg.load_toml(name)
        assert isinstance(raw, dict) and raw
    with pytest.raises(FileNotFoundError):
        cfg.load_toml("nope")


def test_to_info_reproduces_legacy_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)           # no cwd config/ overrides
    info = RunConfig.from_config(rootFolder="data").to_info()
    assert info.pop("rootFolder") == "data"
    assert info.pop("dataFolder") == "data"
    assert info == LEGACY_INFO


@pytest.mark.parametrize("changes, data_root", [({}, "r"), ({"dataFolder": "d"}, "d")])
def test_data_folder_defaults_to_root_folder(tmp_path, monkeypatch, changes, data_root):
    monkeypatch.chdir(tmp_path)
    c = RunConfig.from_config(rootFolder="r").with_(**changes)
    assert c.rootFolder == "r" and c.data_root == data_root
    info = c.to_info()
    assert (info["rootFolder"], info["dataFolder"]) == ("r", data_root)


def test_with_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown run config parameter.*dataRoot"):
        RunConfig(rootFolder="r").with_(dataRoot="d")


def test_dataclass_defaults_match_tomls(tmp_path, monkeypatch):
    # the TOMLs mirror the dataclass defaults: building from file == defaults
    monkeypatch.chdir(tmp_path)
    assert RunConfig.from_config(rootFolder="x").with_(rootFolder=None) == RunConfig()
    assert QCConfig.from_config() == QCConfig()
    assert PFConfig.from_config() == PFConfig()
    assert FluxConfig.from_config() == FluxConfig()


def test_overrides_and_stage_dicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = RunConfig.from_config(rootFolder="r", avgPer=10, pf={"globalCalculation": "local"},
                              flux={"detrendingFormat": "constant"})
    assert c.avgPer == 10 and c.pf.globalCalculation == "local"
    assert c.flux.detrendingFormat == "constant"
    info = c.to_info()
    assert info["PF"]["globalCalculation"] == "local" and info["detrendingFormat"] == "constant"
    c2 = c.with_pf(recalculateGlobalCoefficients=False).with_flux(calcDissipation=True)
    assert c2.pf.recalculateGlobalCoefficients is False and c2.flux.calcDissipation is True
    assert c.pf.recalculateGlobalCoefficients is True     # frozen: original untouched


def test_unknown_keys_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unknown run config"):
        RunConfig.from_config(rootFolder="r", avgPeriod=30)
    with pytest.raises(ValueError, match="unknown pf config"):
        PFConfig.from_config(maxWind=5)
    with pytest.raises(ValueError, match="globalCalculation"):
        PFConfig.from_config(globalCalculation="planar")
    assert PFConfig.from_config(
        globalCalcSampleAction="nan").globalCalcSampleAction == "nan"
    msg = "pf.globalCalcSampleAction must be 'warn' or 'nan', got 'drop'"
    with pytest.raises(ValueError, match=msg):
        PFConfig.from_config(globalCalcSampleAction="drop")
    with pytest.raises(ValueError, match="detrendingFormat"):
        FluxConfig.from_config(detrendingFormat="quadratic")
    with pytest.raises(ValueError, match="stabilitySource"):
        FluxConfig.from_config(stabilitySource="kansas1968")
    with pytest.raises(ValueError,
                       match="lagSearch must be 'off', 'h2o_co2' or 'all'"):
        FluxConfig.from_config(lagSearch="sometimes")
    assert FluxConfig.from_config(lagSearch="all").lagSearch == "all"
    assert FluxConfig.from_config(lagSearch="h2o_co2").lagSearch == "h2o_co2"
    with pytest.raises(ValueError):
        RunConfig().to_info()                  # rootFolder unset


# Arbitrary round geometry [m]; not the dimensions of any instrument.
_GEOMETRY = {"sonicPathLength": 0.1, "irgaPathLength": 0.2,
             "irgaSeparationLateral": 0.3, "irgaSeparationLongitudinal": 0.4}


@pytest.mark.parametrize("missing", sorted(_GEOMETRY))
def test_massman2000_requires_each_geometry_key(tmp_path, monkeypatch,
                                                missing):
    monkeypatch.chdir(tmp_path)
    geometry = {k: v for k, v in _GEOMETRY.items() if k != missing}
    with pytest.raises(ValueError, match=f"needs flux.{missing}$"):
        FluxConfig.from_config(spectralCorrection="massman2000", **geometry)


@pytest.mark.parametrize("mode", ["off", "massman2000"])
@pytest.mark.parametrize("key", sorted(_GEOMETRY))
def test_negative_geometry_key_is_refused(tmp_path, monkeypatch, key, mode):
    monkeypatch.chdir(tmp_path)
    zero = FluxConfig.from_config(spectralCorrection=mode,
                                  **{**_GEOMETRY, key: 0.0})
    assert getattr(zero, key) == 0.0
    with pytest.raises(ValueError, match=f"flux.{key} must not be negative"):
        FluxConfig.from_config(spectralCorrection=mode,
                               **{**_GEOMETRY, key: -0.1})


@pytest.mark.parametrize("mode", ["massman", "on"])
def test_unknown_spectral_correction_is_refused(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="spectralCorrection must be 'off' "
                                         "or 'massman2000'"):
        FluxConfig.from_config(spectralCorrection=mode)


def test_cwd_override_and_partial_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "flux.toml").write_text("[flux]\nqRef = 9\n")
    (tmp_path / "config" / "qc.toml").write_text(
        "[spikeTest]\nmaxRuns = 5\n[spikeTest.spikeDef]\nw = 4.0\n[nanTest]\nmaxPercent = 40\n")
    c = RunConfig.from_config(rootFolder="r")
    assert c.flux.qRef == 9 and c.flux.detrendingFormat == "linear"   # partial file
    assert c.qc.spikeTest.maxRuns == 5
    assert c.qc.spikeTest.spikeDef["w"] == 4.0 and c.qc.spikeTest.spikeDef["u"] == 3.5
    assert c.qc.nanMaxPercent == 40
    info = c.to_info()
    assert info["spikeTest"]["spikeDef"]["w"] == 4.0 and info["nanTest"]["maxPercent"] == 40


def test_explicit_file_and_dict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "my_pf.toml"
    f.write_text("[pf]\nglobalCalcMaxWind = 12\n")
    assert PFConfig.from_config(str(f)).globalCalcMaxWind == 12
    assert PFConfig.from_config({"globalCalcMinWind": 1.0}).globalCalcMinWind == 1.0
    with pytest.raises(TypeError):
        cfg.resolve("pf", 3)


def test_template_from_run_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "run.toml").write_text('[template]\nu = "U_*"\n')
    c = RunConfig.from_config(rootFolder="r")
    assert c.template["u"] == "U_*" and c.template["v"] == "Uy_*"
