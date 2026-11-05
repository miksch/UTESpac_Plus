"""utespac.flux.reference.reference_state: the sonic-fallback Tref carries
the Schotanus correction (upstream 874f54b); a slow-response T does not."""

import numpy as np
import xarray as xr

from utespac.flux.reference import reference_state
from utespac.model import TIME_HF, Run, Sensor, Sensors, to_datetime64
from utespac.sonic_temperature import SONIC_HUMIDITY_COEFF

N = 1440                  # one complete day at 1 sample/min -> 48 periods
T0_DATENUM = 738000.0
Q_REF = 10.0              # [g/kg] siteInfo qRef fallback


def _run(columns):
    """A one-table Run at 2.5 m; no RH/H2O sensors, so q_ref = qRef."""
    t64 = to_datetime64(T0_DATENUM + np.arange(N) / N)
    ds = xr.Dataset({name: (TIME_HF, series) for name, series in columns.items()},
                    coords={TIME_HF: t64})
    sensors = Sensors()
    field = {"Ux": "u", "Ts": "Tson", "AirT": "T", "P": "P"}
    for name in columns:
        sensors.append(Sensor(field[name], "X_20Hz", name, 2.5, 0.0, 1))
    labels = ["TIMESTAMP"] + list(columns)
    info = {"avgPer": 30, "siteElevation": 0.0, "qRef": Q_REF}
    return Run(site=info, sensors=sensors, table_names=["X_20Hz"],
               headers={"X_20Hz": (labels, [None] * len(labels))},
               tables={"X_20Hz": ds})


def test_sonic_fallback_tref_is_schotanus_corrected():
    run = _run({"Ux": np.full(N, 2.0), "Ts": np.full(N, 25.0)})
    ref = reference_state(run)
    expected = 298.15 / (1.0 + SONIC_HUMIDITY_COEFF * Q_REF / 1000.0)
    assert np.allclose(ref.T_K, expected)
    assert np.allclose(ref.q, Q_REF / 1000.0)


def test_slow_response_tref_is_uncorrected():
    run = _run({"Ux": np.full(N, 2.0), "Ts": np.full(N, 25.0),
                "AirT": np.full(N, 25.0)})
    ref = reference_state(run)
    assert np.allclose(ref.T_K, 298.15)


def test_barometer_gaps_take_the_reference_pressure():
    P = np.full(N, 101.0)
    gap_periods = [3, 10, 47]
    for k in gap_periods:
        P[k * 30:(k + 1) * 30] = np.nan
    run = _run({"Ux": np.full(N, 2.0), "Ts": np.full(N, 25.0), "P": P})
    ref = reference_state(run)
    assert not np.isnan(ref.P_kPa).any()
    gaps = np.zeros(ref.n, dtype=bool)
    gaps[gap_periods] = True
    assert np.allclose(ref.P_kPa[gaps], ref.P_ref_kPa)
    assert np.allclose(ref.P_kPa[~gaps], 101.0)
