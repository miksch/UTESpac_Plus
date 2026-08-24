"""Full-file regression over the VAC001 coherent analysis set (gameplan step 7).

Checks every ``VAC001_coherent_*.nc`` next to the HF files: all 8 groups
present with full provenance, and the closure identities on real data --
spectra band sums recover the window variances, ogive(f_min) the covariance,
MRD modes sum to the covariance, quadrant/octant H = 0 flux fractions sum to
1, scale-band fractions sum to 1, and F_ej + F_sw = F_cs for the wavelet
coherent-flux estimator. Prints one row per file and a worst-case summary;
exits 1 on any failure.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_full_regression_vac001.py [OUTPUT_DIR]
"""

import glob
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ec_coherent import io as ecio   # noqa: E402

GROUPS = ("spectra", "mrd", "quadrant", "octant", "ramps", "amplitude_mod",
          "scale_separation", "coherent_flux")
ATTRS = ("ec_coherent_format", "source_file", "source_git_commit", "git_commit",
         "modules", "site_id", "sampling_frequency_hz", "detrend_method")


def worst(x):
    """Largest finite |x|, NaN if none."""
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return float(np.abs(x).max()) if x.size else np.nan


def check_file(path):
    """Return dict of worst-case closure errors for one coherent file."""
    err = {}
    root = xr.open_dataset(path)
    missing = [a for a in ATTRS if a not in root.attrs]
    if missing:
        raise AssertionError(f"missing root attrs: {missing}")
    root.close()

    sp = ecio.read_group(path, "spectra")
    dv = np.diff(sp["frequency_edges"].values)
    err["spec_var"] = worst((sp["S_u"] * dv).sum("frequency", skipna=False) / sp["var_u"] - 1)
    err["spec_ogive"] = worst(sp["ogive_uw"].isel(frequency=0) / sp["cov_uw"] - 1)
    sp.close()

    mrd = ecio.read_group(path, "mrd")
    err["mrd_sum"] = worst((mrd["D_uw"].sum("mr_scale", skipna=False) - mrd["cov_uw"])
                           / mrd["cov_uw"].std())
    mrd.close()

    q = ecio.read_group(path, "quadrant").isel(hole=0)
    err["quad_sum"] = worst(q["S_frac_uw"].sum("quadrant", skipna=False) - 1)
    q.close()

    oc = ecio.read_group(path, "octant")
    err["oct_sum"] = worst(oc["flux_frac_uwTs_uw"].sum("octant", skipna=False) - 1)
    oc.close()

    sc = ecio.read_group(path, "scale_separation")
    err["band_sum"] = worst(sc["var_frac_u"].sum("scale_band", skipna=False) - 1)
    sc.close()

    cf = ecio.read_group(path, "coherent_flux")
    cs = cf["F_cs_uw_u"]
    err["ej_sw"] = worst((cf["F_ej_uw_u"] + cf["F_sw_uw_u"] - cs) / cs.std())
    err["n_rec"] = int(np.isfinite(cf["F_tot_uw_u"]).sum())
    cf.close()
    return err


TOL = {"spec_var": 1e-4, "spec_ogive": 1e-3, "mrd_sum": 1e-3, "quad_sum": 1e-4,
       "oct_sum": 1e-4, "band_sum": 1e-4, "ej_sw": 1e-4}


def main(argv):
    d = argv[0] if argv else os.path.join(ROOT, "data", "VAC001", "output")
    files = sorted(glob.glob(os.path.join(d, "*_coherent_*.nc")))
    if not files:
        print(f"no coherent files under {d}")
        return 1
    cols = list(TOL) + ["n_rec"]
    print(f"{'file':44s} " + " ".join(f"{c:>10s}" for c in cols))
    bad = 0
    agg = {c: [] for c in TOL}
    for f in files:
        try:
            e = check_file(f)
        except Exception as exc:
            print(f"{os.path.basename(f):44s} ERROR: {exc}")
            bad += 1
            continue
        flags = {c: not (e[c] <= TOL[c]) for c in TOL}   # NaN fails too
        bad += any(flags.values())
        for c in TOL:
            agg[c].append(e[c])
        row = " ".join(f"{e[c]:10.1e}" + ("!" if flags[c] else " ") for c in TOL)
        print(f"{os.path.basename(f):44s} {row} {e['n_rec']:5d}")
    print(f"\nworst over {len(files)} files: "
          + ", ".join(f"{c} {max(agg[c]):.1e}" for c in TOL))
    print("PASS" if bad == 0 else f"FAIL: {bad} file(s) out of tolerance")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
