"""Multi-source tables: two loggers joined onto one UTESpac fast table."""

import numpy as np
import pandas as pd
import pytest

from utespac.raw_processing.toa5_tower import process_table, source_specs

HZ = 1.0
START = pd.Timestamp("2025-08-01")


def _write_toa5(path, names, index, values):
    """A 4-line-header TOA5 file with one column per entry of *names*."""
    with open(path, "w", newline="") as fh:
        fh.write('"TOA5","test","CR1000X","1","v","prog","0","tbl"\n')
        fh.write(",".join(f'"{n}"' for n in ["TIMESTAMP"] + names) + "\n")
        fh.write(",".join(f'"{u}"' for u in ["TS"] + ["m/s"] * len(names)) + "\n")
        fh.write(",".join(f'"{p}"' for p in [""] + ["Smp"] * len(names)) + "\n")
        for t, row in zip(index, values):
            stamp = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]
            fh.write(",".join([f'"{stamp}"'] + [f"{v:.4f}" for v in row]) + "\n")


def _make_logger(tmp_path, name, col, n_hours=49, shift_s=0.0):
    """Just over 48 h of data; *shift_s* fakes a logger clock running fast."""
    n = int(n_hours * 3600 * HZ)
    idx = START + pd.to_timedelta(np.arange(1, n + 1) / HZ, unit="s")
    vals = np.arange(n, dtype=float).reshape(-1, 1)
    path = tmp_path / f"{name}.dat"
    _write_toa5(path, [col], idx + pd.Timedelta(seconds=shift_s), vals)
    return str(path)


def _cfg(tmp_path, sources):
    return dict(out_dir=str(tmp_path / "out"), prefix="TEST", table="Merged",
                hz=HZ, sources=sources,
                start_date=START.to_pydatetime(),
                end_date=(START + pd.Timedelta(days=2)).to_pydatetime())


def _read_out(paths, n_cols):
    return pd.read_csv(paths[0], header=None,
                       names=["y", "d", "hm", "s"] + [f"c{i}" for i in range(n_cols)])


# ── source_specs ────────────────────────────────────────────────────────────

def test_source_specs_defaults_to_the_single_source_form():
    cfg = {"raw_pattern": "a*.dat", "columns": {"Ux_10": "Ux"}}
    specs = source_specs(cfg)
    assert len(specs) == 1
    assert specs[0]["lag_s"] == 0.0
    assert specs[0]["columns"] == {"Ux_10": "Ux"}


def test_source_specs_inherits_table_level_keys_and_keeps_absent_ones_absent():
    cfg = {"read_kwargs": {"sep": ","}, "columns": {"a": "A"},
           "sources": [{"raw_pattern": "one*.dat"},
                       {"raw_pattern": "two*.dat", "columns": {"b": "B"},
                        "lag_s": -9.5, "name": "pro"}]}
    one, two = source_specs(cfg)
    assert one["read_kwargs"] == {"sep": ","} and one["columns"] == {"a": "A"}
    assert one["name"] == "source1" and one["lag_s"] == 0.0
    assert two["columns"] == {"b": "B"} and two["lag_s"] == -9.5
    assert two["name"] == "pro"
    # header_rows() must still see these as unset, not as None
    assert "header_row" not in one and "units_row" not in one


def test_source_specs_rejects_a_duplicated_output_column():
    cfg = {"sources": [{"raw_pattern": "a*", "columns": {"Ux_10": "Ux"}},
                       {"raw_pattern": "b*", "columns": {"Ux_10": "Ux_2"}}]}
    with pytest.raises(ValueError, match="mapped by both"):
        source_specs(cfg)


def test_source_specs_rejects_an_incomplete_source():
    with pytest.raises(ValueError, match="no raw_pattern"):
        source_specs({"sources": [{"columns": {"a": "A"}}]})
    with pytest.raises(ValueError, match="no columns"):
        source_specs({"sources": [{"raw_pattern": "a*"}]})


# ── the join ────────────────────────────────────────────────────────────────

def test_two_loggers_join_into_one_table_in_source_order(tmp_path):
    a = _make_logger(tmp_path, "loggerA", "Ux")
    b = _make_logger(tmp_path, "loggerB", "Uy")
    paths = process_table(_cfg(tmp_path, [
        {"raw_pattern": a, "columns": {"Ux_10": "Ux"}, "name": "top"},
        {"raw_pattern": b, "columns": {"Uy_6": "Uy"}, "name": "pro"},
    ]))
    assert len(paths) == 1
    header = (tmp_path / "out" / "TEST_Merged_header.dat").read_text()
    assert header.strip() == '"TIMESTAMP","Ux_10","Uy_6"'
    out = _read_out(paths, 2)
    assert len(out) == int(48 * 3600 * HZ)
    assert out["c0"].notna().all() and out["c1"].notna().all()
    assert (out["c0"] == out["c1"]).all()


def test_a_lagged_source_is_shifted_onto_the_primary_clock(tmp_path):
    """A logger 2 s fast lines up with the primary once lag_s undoes it."""
    a = _make_logger(tmp_path, "loggerA", "Ux")
    b = _make_logger(tmp_path, "loggerB", "Uy", shift_s=2.0)
    cfg = _cfg(tmp_path, [
        {"raw_pattern": a, "columns": {"Ux_10": "Ux"}, "name": "top"},
        {"raw_pattern": b, "columns": {"Uy_6": "Uy"}, "name": "pro",
         "lag_s": -2.0},
    ])
    out = _read_out(process_table(cfg), 2)
    assert out["c1"].notna().all()
    assert (out["c0"] == out["c1"]).all()        # same sample, same grid point


def test_an_unshifted_lagged_source_misaligns_by_the_offset(tmp_path):
    """Without lag_s the fast logger's rows land 2 samples late."""
    a = _make_logger(tmp_path, "loggerA", "Ux")
    b = _make_logger(tmp_path, "loggerB", "Uy", shift_s=2.0)
    out = _read_out(process_table(_cfg(tmp_path, [
        {"raw_pattern": a, "columns": {"Ux_10": "Ux"}},
        {"raw_pattern": b, "columns": {"Uy_6": "Uy"}},
    ])), 2)
    offset = (out["c1"] - out["c0"]).dropna().unique()
    assert offset.tolist() == [-2.0]


def test_a_sub_sample_offset_is_reported_not_silently_dropped(tmp_path, capsys):
    """A lag that is not a whole sample matches nothing; say so loudly."""
    a = _make_logger(tmp_path, "loggerA", "Ux")
    b = _make_logger(tmp_path, "loggerB", "Uy", shift_s=2.0)
    out = _read_out(process_table(_cfg(tmp_path, [
        {"raw_pattern": a, "columns": {"Ux_10": "Ux"}},
        {"raw_pattern": b, "columns": {"Uy_6": "Uy"}, "name": "pro",
         "lag_s": -2.5},
    ])), 2)
    assert out["c1"].isna().all()
    assert "WARNING" in capsys.readouterr().out


def test_a_secondary_absent_from_the_window_leaves_its_columns_nan(tmp_path):
    """One logger down does not cost the window the other logger's levels."""
    a = _make_logger(tmp_path, "loggerA", "Ux")
    b = _make_logger(tmp_path, "loggerB", "Uy")
    # push the secondary's data ten days past the processed window
    late = pd.read_csv(b, skiprows=[0, 2, 3], index_col=[0], parse_dates=True)
    _write_toa5(b, ["Uy"], late.index + pd.Timedelta(days=10), late.values)

    out = _read_out(process_table(_cfg(tmp_path, [
        {"raw_pattern": a, "columns": {"Ux_10": "Ux"}},
        {"raw_pattern": b, "columns": {"Uy_6": "Uy"}, "name": "pro"},
    ])), 2)
    assert out["c0"].notna().all()
    assert out["c1"].isna().all()


def test_units_are_collected_from_every_source(tmp_path):
    """Each source contributes the units of its own columns.

    They are taken verbatim: two loggers spelling the same unit
    differently ("m/s" vs "m s-1") stay as spelled, so a merged table
    declares them uniformly through each source's ``raw_units``.
    """
    a = _make_logger(tmp_path, "loggerA", "Ux")
    b = _make_logger(tmp_path, "loggerB", "Uy")
    process_table(_cfg(tmp_path, [
        {"raw_pattern": a, "columns": {"Ux_10": "Ux"}},
        {"raw_pattern": b, "columns": {"Uy_6": "Uy"},
         "raw_units": {"Uy": "m s-1"}},
    ]))
    units = (tmp_path / "out" / "TEST_Merged_units.dat").read_text()
    assert units.strip() == '"","m/s","m s-1"'
