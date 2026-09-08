"""raw_processing/toa5_tower.py: the <table>_units.dat file written beside
the header, from a TOA5 units row, a pandas export or a per-column config."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "raw_processing"))

from toa5_tower import (clean_unit, header_rows, resolve_units,  # noqa: E402
                        source_units, write_header, write_table_units, write_units)

TOA5 = (
    '"TOA5","13797","CR6","13797","CR6.Std.11.01","CPU:x.CR6","42885","Time_Series"\n'
    '"TIMESTAMP","RECORD","Ux","T_SONIC","H2O_density","PA"\n'
    '"TS","RN","m s-1","deg C","g m-3","kPa"\n'
    '"","","Smp","Smp","Smp","Smp"\n'
    '"2023-07-06 00:00:00.05",0,1.5,20.1,9.2,86.4\n'
)
EXPORT = (
    "TIMESTAMP,RECORD,TA_1_1_1,RH_1_1_1,Site\n"
    "TS,RN,deg C,%,Unnamed: 4_level_1\n"
    "2023-07-06 00:00:00,0,20.1,45.0,VAC\n"
)
NAMES_ONLY = (
    "TIMESTAMP,RECORD,Ux,H2O_density_2\n"
    "2023-07-10 09:00:00.1,9920,0.75,550.5\n"
)

COLUMNS = {"Ux_10.85": "Ux", "T_Sonic_10.85": "T_SONIC",
           "H2O_10.85": "H2O_density", "Pressure_10.85": "PA"}


def test_clean_unit_drops_blanks_and_pandas_placeholders():
    assert clean_unit('"deg C"') == "deg C"
    assert clean_unit("Unnamed: 30_level_1") == ""
    assert clean_unit(None) == "" and clean_unit("  ") == ""


def test_header_rows_defaults_per_source_kind():
    assert header_rows({}) == (1, 2)                                  # TOA5
    assert header_rows({"loader": len}) == (0, 1)                     # pandas export
    assert header_rows({"read_kwargs": {"skiprows": []}}) == (0, None)  # names row only
    assert header_rows({"units_row": None}) == (1, None)              # explicit override


def test_source_units_from_a_toa5_file(tmp_path):
    path = tmp_path / "x.dat"
    path.write_text(TOA5)
    assert source_units(str(path)) == {"TIMESTAMP": "TS", "RECORD": "RN", "Ux": "m s-1",
                                       "T_SONIC": "deg C", "H2O_density": "g m-3",
                                       "PA": "kPa"}


def test_resolve_units_from_a_toa5_file(tmp_path):
    path = tmp_path / "x.dat"
    path.write_text(TOA5)
    assert resolve_units({"columns": COLUMNS}, str(path)) == {
        "Ux_10.85": "m s-1", "T_Sonic_10.85": "deg C",
        "H2O_10.85": "g m-3", "Pressure_10.85": "kPa"}


def test_resolve_units_from_a_pandas_export(tmp_path):
    path = tmp_path / "slow.csv"
    path.write_text(EXPORT)
    cfg = {"loader": len, "columns": {"Temp_7.44": "TA_1_1_1", "RH_7.44": "RH_1_1_1",
                                      "Site": "Site"}}
    assert resolve_units(cfg, str(path)) == {"Temp_7.44": "deg C", "RH_7.44": "%",
                                             "Site": ""}


def test_declared_raw_units_supply_a_source_without_a_units_row(tmp_path):
    path = tmp_path / "cc.dat"
    path.write_text(NAMES_ONLY)
    cfg = {"read_kwargs": {"skiprows": []},
           "columns": {"Ux_7.44": "Ux", "LiH2O_3.07": "H2O_density_2"},
           "raw_units": {"Ux": "m s-1", "H2O_density_2": "mmol m-3"}}
    assert resolve_units(cfg, str(path)) == {"Ux_7.44": "m s-1",
                                             "LiH2O_3.07": "mmol m-3"}


def test_write_units_matches_the_header_column_order(tmp_path):
    write_header(str(tmp_path), "X_20Hz", COLUMNS)
    path = write_units(str(tmp_path), "X_20Hz", COLUMNS,
                       {"Ux_10.85": "m s-1", "Pressure_10.85": "kPa"})
    header = (tmp_path / "X_20Hz_header.dat").read_text().strip().split(",")
    units = open(path).read().strip().split(",")
    assert len(units) == len(header)
    assert units == ['""', '"m s-1"', '""', '""', '"kPa"']


def test_write_table_units_reads_no_data_rows(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.dat").write_text(TOA5)
    out = tmp_path / "site"
    path = write_table_units({"raw_pattern": str(raw / "*.dat"), "out_dir": str(out),
                              "prefix": "X", "table": "20Hz", "columns": COLUMNS})
    assert os.path.basename(path) == "X_20Hz_units.dat"
    assert open(path).read().strip() == '"","m s-1","deg C","g m-3","kPa"'
