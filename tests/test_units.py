"""utespac.units and the units file: declared input units, the magnitude
fallback, and that both reach the same reference state on the same data."""

import logging
import os

import numpy as np
import pytest
import xarray as xr

from utespac import units
from utespac.flux.reference import reference_state
from utespac.import_header import import_header, import_units, units_file
from utespac.model import TIME_HF, Run, Sensor, Sensors, to_datetime64

HEADER_LINE = '"TIMESTAMP","Ux_10.85","T_Sonic_10.85","Pressure_10.85"\n'
UNITS_LINE = '"","m s-1","deg C","kPa"\n'


@pytest.fixture(autouse=True)
def _fresh_warnings():
    units.reset_warnings()


# ── conversions ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("unit,quantity,target,expected", [
    ("deg C", "temperature", None, 20.0),
    ("degC", "temperature", None, 20.0),
    ("C", "temperature", None, 20.0),
    ("°C", "temperature", None, 20.0),
    ("K", "temperature", None, 20.0 - 273.15),
    ("deg C", "temperature", "K", 20.0 + 273.15),
    ("K", "temperature", "K", 20.0),
    ("%", "humidity", None, 20.0),
    ("percent", "humidity", None, 20.0),
    ("fraction", "humidity", None, 2000.0),
    ("1", "humidity", None, 2000.0),
    ("kPa", "pressure", None, 20.0),
    ("hPa", "pressure", None, 2.0),
    ("mbar", "pressure", None, 2.0),
    ("Pa", "pressure", None, 0.02),
    ("g m-3", "h2o_density", None, 20.0),
    ("g/m^3", "h2o_density", None, 20.0),
    ("kg m-3", "h2o_density", None, 20000.0),
    ("mg m-3", "co2_density", None, 20.0),
    ("g m-3", "co2_density", None, 20000.0),
])
def test_declared_unit_conversions(unit, quantity, target, expected):
    out = units.convert(np.full(3, 20.0), unit, quantity, target)
    assert np.allclose(out, expected)


def test_declared_conversion_leaves_the_input_untouched():
    src = np.full(3, 300.0)
    out = units.convert(src, "K", "temperature")
    assert np.allclose(src, 300.0) and out is not src


def test_unknown_unit_raises():
    with pytest.raises(ValueError, match="unknown unit"):
        units.convert(np.zeros(3), "furlongs", "temperature")


def test_unknown_quantity_and_target_raise():
    with pytest.raises(ValueError, match="unknown quantity"):
        units.convert(np.zeros(3), "K", "enthalpy")
    with pytest.raises(ValueError, match="cannot convert"):
        units.convert(np.zeros(3), "K", "temperature", "hPa")


# ── fallback ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("values,quantity,target,expected", [
    (300.0, "temperature", None, 300.0 - 273.15),      # median > 250 -> K
    (20.0, "temperature", None, 20.0),
    (20.0, "temperature", "K", 20.0 + 273.15),         # median < 200 -> deg C
    (300.0, "temperature", "K", 300.0),
    (1013.0, "pressure", None, 101.3),                 # median > 200 -> hPa
    (86.0, "pressure", None, 86.0),
    (55.0, "humidity", None, 55.0),                    # always percent
    (9.0, "h2o_density", None, 9.0),
    (700.0, "co2_density", None, 700.0),
])
def test_fallback_reproduces_the_magnitude_heuristics(values, quantity, target, expected):
    out = units.convert(np.full(4, values), "", quantity, target, warn=False)
    assert np.allclose(out, expected)


def test_fallback_warns_once_per_run_and_sensor(caplog):
    sensor = Sensor("Tson", "X_20Hz", "T_Sonic_10.85", 10.85)
    other = Sensor("T", "X_30min", "Temp_10.85", 10.85)
    with caplog.at_level(logging.WARNING, logger="utespac"):
        for _ in range(3):
            units.convert(np.full(4, 20.0), None, "temperature", sensor=sensor)
        units.convert(np.full(4, 20.0), "", "temperature", sensor=other)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert "X_20Hz:T_Sonic_10.85" in messages[0] and "deg C" in messages[0]
    assert "X_30min:Temp_10.85" in messages[1]

    caplog.clear()
    units.reset_warnings()
    with caplog.at_level(logging.WARNING, logger="utespac"):
        units.convert(np.full(4, 20.0), None, "temperature", sensor=sensor)
    assert len(caplog.records) == 1


def test_declared_unit_never_warns(caplog):
    sensor = Sensor("Tson", "X_20Hz", "T_Sonic_10.85", 10.85, units="K")
    with caplog.at_level(logging.WARNING, logger="utespac"):
        units.convert(np.full(4, 300.0), sensor.units, "temperature", sensor=sensor)
    assert caplog.records == []


# ── the units file ───────────────────────────────────────────────────────────

def _write_header(tmp_path, with_units=True):
    header = tmp_path / "X_20Hz_header.dat"
    header.write_text(HEADER_LINE)
    if with_units:
        (tmp_path / "X_20Hz_units.dat").write_text(UNITS_LINE)
    return str(header)


def test_import_units_reads_the_sibling_file(tmp_path):
    header = _write_header(tmp_path)
    assert units_file(header).endswith("X_20Hz_units.dat")
    assert import_units(header) == ["", "m s-1", "deg C", "kPa"]
    assert import_header(header)[0] == ["TIMESTAMP", "Ux_10.85", "T_Sonic_10.85",
                                        "Pressure_10.85"]


def test_import_units_without_the_file_is_all_empty(tmp_path):
    header = _write_header(tmp_path, with_units=False)
    assert import_units(header) == ["", "", "", ""]


def test_import_units_pads_a_short_units_row(tmp_path):
    header = _write_header(tmp_path)
    (tmp_path / "X_20Hz_units.dat").write_text('"","m s-1"\n')
    assert import_units(header) == ["", "m s-1", "", ""]


def test_find_files_reads_the_units_row_and_ignores_the_units_file(tmp_path):
    from utespac.find_files import find_files
    site = tmp_path / "SiteU"
    (site / "utespac").mkdir(parents=True)
    (site / "siteInfo.toml").write_text("tower = 1\n")
    inp = site / "utespac"
    (inp / "SiteU_20Hz_header.dat").write_text(HEADER_LINE)
    (inp / "SiteU_20Hz_units.dat").write_text(UNITS_LINE)
    (inp / "SiteU_20Hz_20230706000000_20230708000000.txt").write_text("2023,187,0,0.0\n")

    headers, data_files, table_names, _ = find_files({"rootFolder": str(tmp_path)},
                                                     site="SiteU")
    assert table_names == ["SiteU_20Hz"]
    assert headers[0][2] == ["", "m s-1", "deg C", "kPa"]
    files = [f for row in data_files for f in row if f]
    assert [os.path.basename(f) for f in files] == \
        ["SiteU_20Hz_20230706000000_20230708000000.txt"]


def test_sensors_from_legacy_carry_the_declared_units():
    header = [["TIMESTAMP", "Ux_10.85", "T_Sonic_10.85"], [None, 10.85, 10.85],
              ["", "m s-1", "deg C"]]
    sensors = Sensors.from_legacy({"u": np.array([[0, 1, 10.85, 215.0, 1.0]]),
                                   "Tson": np.array([[0, 2, 10.85]])},
                                  [header], ["X_20Hz"])
    assert sensors.at("u", 10.85).units == "m s-1"
    assert sensors.at("Tson", 10.85).units == "deg C"


def test_sensors_from_legacy_without_a_units_row():
    header = [["TIMESTAMP", "Ux_10.85"], [None, 10.85]]
    sensors = Sensors.from_legacy({"u": np.array([[0, 1, 10.85, 215.0, 1.0]])},
                                  [header], ["X_20Hz"])
    assert sensors.at("u", 10.85).units is None


# ── declared and fallback reach the same reference state ─────────────────────

N = 1440                  # one complete day at 1 sample/min -> 48 periods
T0_DATENUM = 738000.0
T_DEGC = 20.0 + 5.0 * np.sin(np.arange(N) / N * 2 * np.pi)
RH_PCT = 45.0 + 10.0 * np.cos(np.arange(N) / N * 2 * np.pi)
P_KPA = np.full(N, 86.4)

_FIELD = {"Ux": "u", "Ts": "Tson", "AirT": "T", "RH": "RH", "PA": "P"}


def _run(columns, unit_of=None):
    """A one-table Run at 2.5 m carrying T, RH and a barometer."""
    t64 = to_datetime64(T0_DATENUM + np.arange(N) / N)
    ds = xr.Dataset({name: (TIME_HF, series) for name, series in columns.items()},
                    coords={TIME_HF: t64})
    sensors = Sensors()
    for name in columns:
        sensors.append(Sensor(_FIELD[name], "X_20Hz", name, 2.5, 0.0, 1,
                              (unit_of or {}).get(name)))
    labels = ["TIMESTAMP"] + list(columns)
    info = {"avgPer": 30, "siteElevation": 1200.0, "qRef": 10.0}
    return Run(site=info, sensors=sensors, table_names=["X_20Hz"],
               headers={"X_20Hz": (labels, [None] * len(labels))},
               tables={"X_20Hz": ds})


def test_declared_units_match_the_fallback_on_the_same_data():
    heuristic = _run({"Ux": np.full(N, 2.0), "Ts": T_DEGC, "AirT": T_DEGC,
                      "RH": RH_PCT, "PA": P_KPA})
    declared = _run({"Ux": np.full(N, 2.0), "Ts": T_DEGC + 273.15,
                     "AirT": T_DEGC + 273.15, "RH": RH_PCT / 100.0,
                     "PA": P_KPA * 10.0},
                    unit_of={"Ux": "m s-1", "Ts": "K", "AirT": "K",
                             "RH": "fraction", "PA": "hPa"})
    a, b = reference_state(heuristic), reference_state(declared)
    assert np.allclose(a.T_K, b.T_K, rtol=1e-12)
    assert np.allclose(a.P_kPa, b.P_kPa, rtol=1e-12)
    assert np.allclose(a.q, b.q, rtol=1e-12)
    assert np.allclose(a.rho, b.rho, rtol=1e-12)
