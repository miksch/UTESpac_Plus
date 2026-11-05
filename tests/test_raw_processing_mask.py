"""Sample-level sonic fault mask of the TOA5 tower processing."""

import numpy as np
import pandas as pd

from utespac.raw_processing.toa5_tower import (FAST_SONIC_MAP, SONIC_SENTINEL,
                                               level_columns, mask_sonic_faults,
                                               process_table)

HZ = 1.0
START = pd.Timestamp("2025-08-01")

LOWER = {"Ux": "Ux_2", "Uy": "Uy_2", "Uz": "Uz_2", "T_Sonic": "Ts_2",
         "diagnostic": "diag_sonic_2"}
COLUMNS = {**level_columns(FAST_SONIC_MAP, 7.44),
           **level_columns(LOWER, 3.07),
           "CO2_7.44": "CO2"}
UPPER_WIND = ["Ux", "Uy", "Uz", "Ts"]
LOWER_WIND = ["Ux_2", "Uy_2", "Uz_2", "Ts_2"]


def _frame(n=6):
    """Two clean sonic levels plus a gas column, *n* samples."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, 9)),
                      columns=UPPER_WIND + LOWER_WIND + ["CO2"])
    df["diag_sonic"] = 0
    df["diag_sonic_2"] = np.arange(n) % 64
    return df


def test_mask_nulls_only_the_faulted_level_on_the_faulted_row():
    df = _frame()
    df.loc[1, "diag_sonic_2"] = 61503
    df.loc[3, "Ux"] = SONIC_SENTINEL
    clean = df.copy()

    out, counts = mask_sonic_faults(df, COLUMNS)

    assert counts == {"3.07": 1, "7.44": 1}
    assert out.loc[1, LOWER_WIND].isna().all()
    assert out.loc[3, UPPER_WIND].isna().all()
    np.testing.assert_array_equal(out.loc[1, UPPER_WIND], clean.loc[1, UPPER_WIND])
    np.testing.assert_array_equal(out.loc[3, LOWER_WIND], clean.loc[3, LOWER_WIND])
    rows = [0, 2, 4, 5]
    pd.testing.assert_frame_equal(out.loc[rows], clean.loc[rows], check_dtype=False)
    pd.testing.assert_frame_equal(out[["CO2", "diag_sonic", "diag_sonic_2"]],
                                  clean[["CO2", "diag_sonic", "diag_sonic_2"]])


def _write_toa5(path, names, index, values):
    with open(path, "w", newline="") as fh:
        fh.write('"TOA5","test","CR1000X","1","v","prog","0","tbl"\n')
        fh.write(",".join(f'"{n}"' for n in ["TIMESTAMP"] + names) + "\n")
        fh.write(",".join(f'"{u}"' for u in ["TS"] + [""] * len(names)) + "\n")
        fh.write(",".join(f'"{p}"' for p in [""] + ["Smp"] * len(names)) + "\n")
        for t, row in zip(index, values):
            stamp = t.strftime("%Y-%m-%d %H:%M:%S")
            fh.write(",".join([f'"{stamp}"'] + [f"{v:g}" for v in row]) + "\n")


def _run(tmp_path, **extra):
    n = int(49 * 3600 * HZ)
    idx = START + pd.to_timedelta(np.arange(1, n + 1) / HZ, unit="s")
    names = UPPER_WIND + ["diag_sonic"] + LOWER_WIND + ["diag_sonic_2"]
    vals = np.ones((n, len(names)))
    vals[:, names.index("diag_sonic")] = 0
    vals[:, names.index("diag_sonic_2")] = np.arange(n) % 64
    vals[9, names.index("diag_sonic_2")] = 61503
    vals[19, names.index("Ux")] = SONIC_SENTINEL
    _write_toa5(tmp_path / "fast.dat", names, idx, vals)
    columns = {**level_columns(FAST_SONIC_MAP, 7.44), **level_columns(LOWER, 3.07)}
    cfg = dict(raw_pattern=str(tmp_path / "fast.dat"), columns=columns,
               out_dir=str(tmp_path / "out"), prefix="TEST", table="1Hz",
               hz=HZ, start_date=START.to_pydatetime(),
               end_date=(START + pd.Timedelta(days=2)).to_pydatetime(), **extra)
    paths = process_table(cfg)
    return pd.read_csv(paths[0], header=None,
                       names=["y", "d", "hm", "s"] + list(columns))


def test_process_table_masks_by_default_and_the_cfg_disables_it(tmp_path):
    on = _run(tmp_path)
    upper = ["Ux_7.44", "Uy_7.44", "Uz_7.44", "T_Sonic_7.44"]
    lower = ["Ux_3.07", "Uy_3.07", "Uz_3.07", "T_Sonic_3.07"]
    assert on.loc[9, lower].isna().all() and on.loc[9, upper].notna().all()
    assert on.loc[19, upper].isna().all() and on.loc[19, lower].notna().all()
    assert on.loc[9, "diagnostic_3.07"] == 61503
    assert on[upper + lower].isna().sum().sum() == 8

    off = _run(tmp_path, fault_mask=False)
    assert off[upper + lower].notna().all().all()
    assert off.loc[19, "Ux_7.44"] == SONIC_SENTINEL
