"""utespac.flux.tables: named columns, CSV labels, trimming and storage."""

import numpy as np

from utespac.flux.tables import STORE_ORDER, TABLE_SPECS, FluxTable, FluxTables, height_label


def test_height_label_matches_legacy_format():
    assert height_label(10.85) == "10.85"
    assert height_label(2.0) == "2"


def test_table_labels_follow_the_names_column_order():
    tab = FluxTable(TABLE_SPECS["sensible_heat"], 4, [10.85, 2.0])
    n_cols = len(TABLE_SPECS["sensible_heat"].columns)
    assert tab.labels[:3] == ["time", "rho_air_ref", "cp_ref"]
    assert tab.labels[3] == "w_ts_cov_raw_10.85"
    assert tab.labels[2 + n_cols] == "H_buoyancy_pf_10.85"
    assert tab.labels[3 + n_cols] == "w_ts_cov_raw_2"
    assert len(tab.labels) == 3 + 2 * n_cols


def test_set_get_and_trim_drop_all_nan_columns():
    tab = FluxTable(TABLE_SPECS["tke"], 3, [10.0])
    tab.set(0, None, "time", 1.0)
    tab.set(1, None, "time", 2.0)
    vals, labels = tab.trimmed()
    assert labels == ["time"]                       # TKE column untouched -> dropped
    tab.set(1, 10.0, "TKE", 0.5)
    assert tab.get(1, 10.0, "TKE") == 0.5
    vals, labels = tab.trimmed()
    assert labels == ["time", "TKE_10"]
    assert np.isnan(vals[0, 1]) and vals[1, 1] == 0.5


def test_parity_artifacts_are_gone():
    tab = FluxTable(TABLE_SPECS["correlation"], 1, [10.0])
    assert len(tab.labels) == 1 + 15                      # the duplicate MATLAB col 14 is gone
    assert len(set(tab.labels)) == len(tab.labels)
    skew = FluxTable(TABLE_SPECS["skewness"], 1, [10.0])
    assert "theta_v_skew_10" in skew.labels and not any("Theata" in lab for lab in skew.labels)


def test_third_moment_of_ts_sits_in_transport():
    sigma = FluxTable(TABLE_SPECS["sigma"], 1, [10.0])
    transport = FluxTable(TABLE_SPECS["transport"], 1, [10.0])
    assert "ts_var_transport_pf_10" in transport.labels
    assert not any("transport" in lab for lab in sigma.labels)


def test_flux_tables_t_fw_sigma_only_with_fine_wire_and_store_rules():
    with_fw = FluxTables(2, [10.0], has_fw=True)
    without = FluxTables(2, [10.0], has_fw=False)
    assert "t_fw_sigma_10" in with_fw["sigma"].labels
    assert "t_fw_sigma_10" not in without["sigma"].labels
    assert list(with_fw.tables) == STORE_ORDER

    with_fw.set_time(0, 7.0)
    with_fw["sensible_heat"].set(0, 10.0, "w_ts_cov_raw", 0.1)
    out = with_fw.store({}, store_extra=False)
    assert "sensible_heat" in out and out["sensible_heatHeader"] == ["time", "w_ts_cov_raw_10"]
    assert "correlation" not in out and "obukhov" not in out      # extras skipped
    assert "co2_flux" not in out                                  # no CO2 data
    with_fw["co2_flux"].set(0, 10.0, "co2_mole_fraction", 420.0)
    out = with_fw.store({}, store_extra=True)
    assert out["co2_fluxHeader"] == ["time", "co2_mole_fraction_10"]
    assert "obukhovHeader" in out and out["obukhov"].shape == (2, 1)   # time column only
