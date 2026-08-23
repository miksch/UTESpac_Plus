"""utespac.flux.tables: named columns, legacy labels, trimming and storage."""

import numpy as np

from utespac.flux.tables import STORE_ORDER, TABLE_SPECS, FluxTable, FluxTables, height_label


def test_height_label_matches_legacy_format():
    assert height_label(10.85) == "10.85"
    assert height_label(2.0) == "2"


def test_table_labels_follow_legacy_header_order():
    tab = FluxTable(TABLE_SPECS["H"], 4, [10.85, 2.0])
    assert tab.labels[:3] == ["time", "rho", "cp"]
    assert tab.labels[3] == "10.85m son:Ts'w'"
    assert tab.labels[14] == "10.85m fw:VTh'wPF'"
    assert tab.labels[15] == "2m son:Ts'w'"
    assert len(tab.labels) == 3 + 2 * 12


def test_set_get_and_trim_drop_all_nan_columns():
    tab = FluxTable(TABLE_SPECS["tke"], 3, [10.0])
    tab.set(0, None, "time", 1.0)
    tab.set(1, None, "time", 2.0)
    vals, labels = tab.trimmed()
    assert labels == ["time"]                       # tke column untouched -> dropped
    tab.set(1, 10.0, "tke", 0.5)
    assert tab.get(1, 10.0, "tke") == 0.5
    vals, labels = tab.trimmed()
    assert labels == ["time", "10m :0.5(u'^2+v'^2+w'^2)"]
    assert np.isnan(vals[0, 1]) and vals[1, 1] == 0.5


def test_duplicate_R_label_has_its_own_key():
    tab = FluxTable(TABLE_SPECS["R"], 1, [10.0])
    assert tab.labels.count("10m :R_wPF_CO2") == 2
    tab.set(0, 10.0, "R_wPF_CO2", 0.1)
    tab.set(0, 10.0, "R_wPF_CO2_dup", 0.2)
    i = tab.labels.index("10m :R_wPF_CO2")
    assert tab.values[0, i] == 0.1 and tab.values[0, i + 4] == 0.2


def test_flux_tables_sigma_tfw_only_with_fine_wire_and_store_rules():
    with_fw = FluxTables(2, [10.0], has_fw=True)
    without = FluxTables(2, [10.0], has_fw=False)
    assert "10m :sigma_TFW" in with_fw["sigma"].labels
    assert "10m :sigma_TFW" not in without["sigma"].labels
    assert list(with_fw.tables) == STORE_ORDER

    with_fw.set_time(0, 7.0)
    with_fw["H"].set(0, 10.0, "Ts_w", 0.1)
    out = with_fw.store({}, store_extra=False)
    assert "H" in out and out["Hheader"] == ["time", "10m son:Ts'w'"]
    assert "R" not in out and "L" not in out          # extras skipped
    assert "CO2flux" not in out                       # no CO2 data
    with_fw["CO2flux"].set(0, 10.0, "ppm", 420.0)
    out = with_fw.store({}, store_extra=True)
    assert out["CO2fluxHeader"] == ["time", "10m: CO2 (ppm, moist-air molar ratio)"]
    assert "Lheader" in out and out["L"].shape == (2, 1)   # time column only
