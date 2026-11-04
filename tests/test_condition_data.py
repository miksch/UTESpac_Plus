"""Absolute-limit test of :func:`utespac.condition_data.qc_table`."""

import numpy as np
import pytest

from utespac.condition_data import qc_table
from utespac.run_config import QCConfig, _DEFAULT_TEMPLATE

T0_DATENUM = 739000.0


def _slow_table(label, values):
    """One day at one sample a minute: below 1 Hz, so the spike test is skipped."""
    hz = 1.0 / 60.0
    t = T0_DATENUM + np.arange(values.size) / 1440.0
    return np.column_stack([t, values]), ["TIMESTAMP", label], hz


@pytest.mark.parametrize("label, unit, base, planted", [
    ("T_Sonic_3", "deg C", 20.0, 100.0),          # internal unit, compared as logged
    ("T_Sonic_3", "K", 293.15, 373.15),
    ("FW_3", "K", 293.15, 373.15),
    ("FW_3", "", 293.15, 373.15),                 # undeclared: K above a median of 250
    ("H2O_3", "mg m-3", 1.0e4, 6.0e4),
    ("H2O_3", "kg m-3", 0.01, 0.06),
    ("CO2_3", "g m-3", 0.7, 1.6),
    ("CO2_3", "ug m-3", 7.0e5, 1.6e6),
])
def test_limit_is_judged_in_the_internal_unit(label, unit, base, planted):
    """With limits Tson/fw [-20, 80] deg C, irgaH2O [0, 50] g m-3 and irgaCO2
    [0, 1500] mg m-3, only the planted sample lies outside once converted,
    and the kept samples are stored as logged."""
    values = np.full(1440, base)
    values[700] = planted
    tbl, labels, hz = _slow_table(label, values)
    info = {"avgPer": 30, **QCConfig().to_info()}
    out, _, _ = qc_table(tbl.copy(), labels, info, dict(_DEFAULT_TEMPLATE), hz,
                         header_units=["", unit])
    expected = values.copy()
    expected[700] = np.nan
    np.testing.assert_array_equal(out[:, 1], expected)
