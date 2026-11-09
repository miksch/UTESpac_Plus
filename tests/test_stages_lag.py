"""utespac.stages.flux hands the site's time-lag settings to the engine."""

import numpy as np
import pytest
import xarray as xr

from utespac import stages
from utespac.model import (COMPONENT, HEIGHT, TIME_HF, Run, Sensor, Sensors,
                           to_datetime64)

FS = 1.0                   # [Hz]
N = 86400                  # one complete day -> 48 periods
T0_DATENUM = 738000.0
Z = 3.0                    # [m]
LAG = 5                    # [samples] planted gas delay against w

_FIELD = {"Ux": "u", "Uy": "v", "Uz": "w", "Ts": "Tson",
          "H2O": "irgaH2O", "CO2": "irgaCO2"}


def _run(lag_info):
    """One sonic and IRGA, gases delayed by LAG, rotation the identity."""
    rng = np.random.default_rng(5)
    u = 2.0 + rng.normal(0, 0.3, N)
    v = rng.normal(0, 0.3, N)
    w = rng.normal(0, 0.1, N)
    columns = {"Ux": u, "Uy": v, "Uz": w,
               "Ts": 25.0 + rng.normal(0, 0.2, N),
               "H2O": 10.0 + 0.5 * np.roll(w, LAG) + rng.normal(0, 0.02, N),
               "CO2": 700.0 - 2.0 * np.roll(w, LAG) + rng.normal(0, 0.1, N)}
    t64 = to_datetime64(T0_DATENUM + np.arange(N) / (FS * 86400.0))
    names = {kind: f"{kind}_{Z:g}" for kind in columns}
    ds = xr.Dataset({names[k]: (TIME_HF, x) for k, x in columns.items()},
                    coords={TIME_HF: t64})
    ds.attrs["scan_hz"] = FS
    sensors = Sensors()
    for kind, name in names.items():
        sensors.append(Sensor(_FIELD[kind], "X_1s", name, Z, 0.0, 1, None))
    labels = ["TIMESTAMP"] + list(names.values())
    info = {"avgPer": 30, "siteElevation": 0.0, "qRef": 10.0,
            "detrendingFormat": "constant", **lag_info}
    run = Run(site=info, sensors=sensors, table_names=["X_1s"],
              headers={"X_1s": (labels, [None] * len(labels))},
              tables={"X_1s": ds})
    uvw = np.stack([u, v, w], axis=1)[:, None, :]
    dims = (TIME_HF, HEIGHT, COMPONENT)
    run.rotation = xr.Dataset({"rotated": (dims, uvw), "pf": (dims, uvw)},
                              coords={TIME_HF: t64, HEIGHT: [Z],
                                      COMPONENT: ["u", "v", "w"]})
    return run


def _stored_lag(products, table, key):
    """The stored lag per period at Z, or None when the column is absent."""
    if key not in products[table]:
        return None
    return products[table][key].sel(height=Z).values.tolist()


@pytest.mark.parametrize("lag_info, expected", [
    ({}, None),
    ({"lagSearch": "h2o_co2", "lagMax": 10.0}, [LAG / FS] * 48),
])
def test_flux_stage_applies_the_site_lag_settings(lag_info, expected):
    """lagSearch and lagMax in the site info set the stored gas lags."""
    products = stages.flux(_run(lag_info)).products
    assert _stored_lag(products, "latent_heat", "lag_h2o") == expected
    assert _stored_lag(products, "co2_flux", "lag_co2") == expected
