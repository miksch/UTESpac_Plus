"""Tests of :mod:`utespac.stages`: declared header units per table."""

import numpy as np

from utespac.run_config import QCConfig, RunConfig
from utespac.stages import condition, load_run

T0_DATENUM = 739000.0
N_MIN = 1440               # one day at one sample a minute
HZ = 1.0 / 60.0            # below 1 Hz: the spike test is skipped
TSON = "T_Sonic_3"
TSON_K = 293.15            # 20 deg C, inside Tson [-20, 80] deg C as kelvin


def _tson_run(units_by_table):
    """Run of one ``T_Sonic_3`` table per entry, each at 293.15 as logged."""
    names = list(units_by_table)
    t = T0_DATENUM + np.arange(N_MIN) / 1440.0
    tbl = np.column_stack([t, np.full(N_MIN, TSON_K)])
    labels = ["TIMESTAMP", TSON]
    headers = [[labels, [None, 3.0], ["", units_by_table[n]]] for n in names]
    info = {"avgPer": 30, "tableScanFrequency": [HZ] * len(names),
            **QCConfig().to_info()}
    return load_run(info, [tbl.copy() for _ in names], headers, names, {})


def test_condition_hands_each_table_its_own_units():
    """293.15 is kept where declared K and removed where declared deg C."""
    run = condition(_tson_run({"slow_k": "K", "slow_c": "deg C"}),
                    RunConfig().template)
    np.testing.assert_array_equal(run.tables["slow_k"][TSON].values,
                                  np.full(N_MIN, TSON_K))
    np.testing.assert_array_equal(run.tables["slow_c"][TSON].values,
                                  np.full(N_MIN, np.nan))


def test_load_run_stores_the_units_per_table():
    """``Run.header_units`` carries each table's declared units by name."""
    run = _tson_run({"slow_k": "K", "slow_c": "deg C"})
    assert run.header_units == {"slow_k": ["", "K"], "slow_c": ["", "deg C"]}
