"""Synthetic ``utespac-hf-2`` files for the ec_coherent tests."""

import numpy as np
import xarray as xr

MS = "milliseconds since 1970-01-01 00:00:00"


def make_hf(path, fs=10.0, window_s=300.0, n_records=4, heights=(3.0, 10.0), seed=0,
            with_scalars=True, nan_block=None):
    """Write a small HF file: red-noise winds with a prescribed u'w' covariance, ts, rho_h2o.

    ``nan_block=(record, start, stop)`` blanks that sample range of u_pf in one record.
    Returns (path, truth dict keyed by the ``utespac-hf-2`` names).
    """
    rng = np.random.default_rng(seed)
    n_win = int(round(window_s * fs))
    n = n_win * n_records
    dt = np.timedelta64(int(round(1e9 / fs)), "ns")
    t0 = np.datetime64("2024-06-01T00:00:00", "ns")
    time = t0 + dt * np.arange(1, n + 1)                      # Campbell: first sample one step after midnight
    records = t0 + np.timedelta64(int(window_s * 1e9), "ns") * np.arange(1, n_records + 1)
    nh = len(heights)

    def red(scale):
        x = rng.normal(size=(n, nh))
        for i in range(1, n):
            x[i] = 0.95 * x[i - 1] + x[i]
        return scale * x / x.std(axis=0)

    w = red(0.4)
    u = 5.0 + red(0.8) - 0.5 * w           # negative u'w'
    v = red(0.6)
    Ts = 20.0 + red(0.3) + 0.3 * w
    rhov = 8.0 + red(0.2)
    if nan_block:
        r, a, b = nan_block
        u[r * n_win + a: r * n_win + b, :] = np.nan
    ds = xr.Dataset(
        coords={"time": ("time", time), "height": ("height", np.array(heights, dtype=float)),
                "record": ("record", records)})
    for name, arr in (("u_pf", u), ("v_pf", v), ("w_pf", w), ("ts", Ts)):
        ds[name] = (("time", "height"), arr.astype("f4"))
    if with_scalars:
        ds = ds.assign_coords(height_h2o=("height_h2o", np.array(heights, dtype=float)))
        ds["rho_h2o"] = (("time", "height_h2o"), rhov.astype("f4"))
    ds["ustar_pf"] = (("record", "height"), np.full((n_records, nh), 0.45))
    ds["L"] = (("record", "height"), np.full((n_records, nh), -50.0))
    ds["spike_flag"] = (("record", "height"), np.zeros((n_records, nh)))
    ds["H_ssitc"] = (("record", "height"), np.zeros((n_records, nh)))
    ds.attrs.update(utespac_format="utespac-hf-2", sampling_frequency_hz=fs,
                    flux_averaging_s=window_s, site_id="SYN", detrend="constant",
                    pf_type="GPF", git_commit="test", despiking="none (synthetic)")
    ds.to_netcdf(path, engine="netcdf4",
                 encoding={"time": {"units": MS}, "record": {"units": MS}})
    return str(path), {"u_pf": u, "v_pf": v, "w_pf": w, "ts": Ts, "rho_h2o": rhov,
                       "n_win": n_win}
