"""The raw high-frequency products carry every gas-analyser level: a tower
with an EC150-style IRGA on one sonic and a LI-7500 on another gets one
``rhov`` / ``rhoCO2`` column per level, each at its own height (the VAC_ADV
2023 IOP layout; task 2026-09-07 vac-adv-second-irga-level-in-hf-export)."""

import numpy as np
import xarray as xr

from utespac import stages
from utespac.model import TIME_HF, Run, Sensor, Sensors, to_datetime64

FS = 1.0 / 60.0            # [Hz] one sample a minute
N = 1440                   # one complete day -> 48 periods (the reference state wants a day)
T0_DATENUM = 738000.0
UPPER, LOWER = 7.44, 3.07  # [m] IRGASON; CSAT3 + LI-7500

_FIELD = {"Ux": "u", "Uy": "v", "Uz": "w", "Ts": "Tson",
          "H2O": "irgaH2O", "CO2": "irgaCO2", "LiH2O": "LiH2O", "LiCO2": "LiCO2"}


def _run(columns):
    t64 = to_datetime64(T0_DATENUM + np.arange(N) / (FS * 86400.0))
    ds = xr.Dataset({name: (TIME_HF, series) for name, series in columns.items()},
                    coords={TIME_HF: t64})
    ds.attrs["scan_hz"] = FS
    sensors = Sensors()
    for name in columns:
        kind, height = name.split("_")
        sensors.append(Sensor(_FIELD[kind], "X_1min", name, float(height), 0.0, 1))
    labels = ["TIMESTAMP"] + list(columns)
    info = {"avgPer": 30, "siteElevation": 0.0, "qRef": 10.0,
            "saveRawConditionedData": True, "detrendingFormat": "constant"}
    return Run(site=info, sensors=sensors, table_names=["X_1min"],
               headers={"X_1min": (labels, [None] * len(labels))},
               tables={"X_1min": ds})


def _sonic(rng, height):
    hn = f"{height:g}"
    return {f"Ux_{hn}": 2.0 + rng.normal(0, 0.3, N), f"Uy_{hn}": rng.normal(0, 0.3, N),
            f"Uz_{hn}": rng.normal(0, 0.1, N), f"Ts_{hn}": 25.0 + rng.normal(0, 0.2, N)}


def test_two_irga_families_keep_their_own_levels():
    rng = np.random.default_rng(0)
    h2o_upper = 10.0 + rng.normal(0, 0.05, N)          # [g/m3]  IRGASON
    co2_upper = 700.0 + rng.normal(0, 1.0, N)          # [mg/m3]
    h2o_lower = 600.0 + rng.normal(0, 3.0, N)          # [mmol/m3] LI-7500
    co2_lower = 16.0 + rng.normal(0, 0.05, N)          # [mmol/m3]
    cols = {**_sonic(rng, UPPER), **_sonic(rng, LOWER),
            f"H2O_{UPPER}": h2o_upper, f"CO2_{UPPER}": co2_upper,
            f"LiH2O_{LOWER}": h2o_lower, f"LiCO2_{LOWER}": co2_lower}
    run = stages.flux(_run(cols))

    raw = run.raw
    assert raw is not None
    np.testing.assert_array_equal(raw["height_h2o"].values, [UPPER, LOWER])
    np.testing.assert_array_equal(raw["height_co2"].values, [UPPER, LOWER])
    rhov = raw["rhov"].sel(height_h2o=UPPER).values
    np.testing.assert_allclose(rhov, h2o_upper)
    rhov = raw["rhov"].sel(height_h2o=LOWER).values
    np.testing.assert_allclose(rhov, h2o_lower * 0.018)
    np.testing.assert_allclose(raw["rhoCO2"].sel(height_co2=UPPER).values, co2_upper)
    np.testing.assert_allclose(raw["rhoCO2"].sel(height_co2=LOWER).values, co2_lower * 44.0)
    # the averaged products see both levels too
    assert sorted(run.products["latent_heat"]["height"].values) == sorted([UPPER, LOWER])


# ── the written products on this machine ────────────────────────────────────

def _hf_run_pairs():
    import glob
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    for hf in sorted(glob.glob(os.path.join(root, "*", "output", "*_hf_*.nc"))):
        m = re.match(r"(.+)_hf_(.+)$", os.path.basename(hf))
        if m is None:
            continue
        site, tail = m.groups()
        runs = glob.glob(os.path.join(os.path.dirname(hf), f"{site}_*Avg_{tail}"))
        if len(runs) == 1:
            yield hf, runs[0]


def test_hf_files_carry_every_hygrometer_level():
    """For every HF file next to its run file: the HF ``height_h2o`` levels are
    the run file's ``latent_heat`` heights with a finite LE, and likewise for
    ``height_co2`` and ``co2_flux`` (VAC_ADV dropped its 3.07 m LI-7500)."""
    import pytest
    netCDF4 = pytest.importorskip("netCDF4")
    pairs = list(_hf_run_pairs())
    if not pairs:
        pytest.skip("no HF/run file pairs under data/*/output on this machine")
    checks = (("height_h2o", "latent_heat", "LE_wpl_pf"), ("height_co2", "co2_flux", "Fc_wpl_pf"))
    for hf_path, run_path in pairs:
        with netCDF4.Dataset(hf_path) as hf, netCDF4.Dataset(run_path) as run:
            for dim, table, var in checks:
                if dim not in hf.variables or table not in run["products"].groups:
                    continue
                g = run["products"][table]
                z_run = np.asarray(g["height"][:], float)
                finite = np.isfinite(np.asarray(g[var][:], float)).any(axis=0)
                expect = sorted(z_run[finite])
                got = sorted(np.asarray(hf[dim][:], float))
                assert got == expect, f"{hf_path}: {dim} {got} vs {table} levels with {var} {expect}"
