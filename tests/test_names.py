"""utespac.names: the rename table is complete, unique and on-convention."""

import re

from utespac import names
from utespac.flux.tables import STORE_ORDER, TABLE_SPECS


def _all_vars():
    return [(g, v) for g, vs in names.VARIABLES.items() for v in vs]


def test_every_legacy_table_has_a_group():
    assert set(STORE_ORDER) == set(names.GROUPS.values()) - {"temperature", "humidity"}
    assert set(TABLE_SPECS) == set(STORE_ORDER)
    assert len(set(names.GROUPS.values())) == len(names.GROUPS)


def test_every_legacy_column_key_maps_exactly_once():
    seen = {}
    for (table, key), (group, var) in names._LEGACY_INDEX.items():
        assert names.legacy_to_new(table, key) == (group, var.name)
        assert (table, key) not in seen
        seen[(table, key)] = (group, var.name)
    # every legacy-keyed Var is reachable through its legacy (table, key)
    for group, var in _all_vars():
        if var.legacy_key is None:
            continue
        table = var.legacy_table or names.GROUPS_INVERSE[group]
        assert (table, var.legacy_key) in seen, (group, var.name)


def test_table_specs_carry_the_names_and_their_csv_labels():
    for group, spec in TABLE_SPECS.items():
        declared = [v.name for v in names.VARIABLES[group]]
        assert list(spec.fixed[1:]) + [k for k, _ in spec.columns] == declared, group
        for key, tmpl in spec.columns:
            assert tmpl == names.csv_header(key, "{hn}"), (group, key)


def test_names_unique_within_group_and_on_convention():
    for group, vs in names.VARIABLES.items():
        assert names.is_valid_name(group) and group == group.lower()
        seen = [v.name for v in vs]
        assert len(seen) == len(set(seen)), group
        for v in vs:
            assert names.is_valid_name(v.name), (group, v.name)
            assert v.units and v.long_name
            assert "'" not in v.name and ":" not in v.name


def test_frame_qualifier_is_last_and_single():
    frames = {"raw", "pf", "tilt", "snsp", "vertical"}
    for group, v in _all_vars():
        toks = [t for t in v.name.split("_") if t]
        inside = [t for t in toks[:-1] if t in frames]
        assert not inside, (group, v.name)


def test_hf_names_on_convention():
    for old, new in names.HF_VARIABLES.items():
        assert names.is_valid_name(new), (old, new)
    assert len(set(names.HF_VARIABLES.values())) == len(names.HF_VARIABLES)


def test_symbol_whitelist():
    assert names.is_valid_name("LE_wpl_pf") and names.is_valid_name("u_w__w_theta_v_corr_pf")
    assert not names.is_valid_name("Thv_wPF")
    assert not names.is_valid_name("10.85m son:Ts'w'")
    assert not names.is_valid_name("E_pf")


def test_new_columns_present():
    new = {v.name for g, v in _all_vars() if v.legacy_key is None}
    assert new == {"H_raw", "H_pf", "H_buoyancy_pf", "ustar_pf", "w_h2o_wpl_cov_pf"}


def test_csv_header():
    assert names.csv_header("w_ts_cov_raw", "10.85") == "w_ts_cov_raw_10.85"
    assert names.csv_header("rho_air_ref", None) == "rho_air_ref"
    assert re.match(r"^[A-Za-z][A-Za-z0-9_.]*$", names.csv_header("Tau_pf", "2"))
