"""Before/after check of the variable-name refactor on run files.

Compares every product of a ``utespac-run-2`` file (legacy names) with
its ``utespac-run-3`` counterpart (new names) through the alias table in
``utespac.names``: same groups, same heights, same time axis, every
renamed variable numerically identical (``equal_nan``), and the five
columns introduced with the rename equal to their defining products.
Reads the files with netCDF4 directly so the check does not depend on the
reader under test.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/rename_ab_check.py <before_dir> <after_dir> [--glob "*_30minAvg_*.nc"]

Exit status 1 when any file differs; the report lists the differences.
"""

import argparse
import glob
import os
import sys

import netCDF4
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utespac import names  # noqa: E402

NEW_COLUMNS = {
    ("sensible_heat", "H_raw"): ("sensible_heat", "rho_air_ref", "cp_ref", "w_t_air_cov_raw"),
    ("sensible_heat", "H_pf"): ("sensible_heat", "rho_air_ref", "cp_ref", "w_t_air_cov_pf"),
    ("sensible_heat", "H_buoyancy_pf"): ("sensible_heat", "rho_air_ref", "cp_ref", "w_theta_v_cov_pf"),
}


def _products(nc):
    grp = nc.groups.get("products")
    return {} if grp is None else grp.groups


def _read(var):
    arr = np.asarray(var[:], dtype=float)
    return np.where(np.ma.getmaskarray(var[:]), np.nan, arr) if np.ma.isMaskedArray(var[:]) else arr


def compare_file(before, after, report):
    ok = True
    with netCDF4.Dataset(before) as b, netCDF4.Dataset(after) as a:
        fmt_b, fmt_a = b.getncattr("utespac_format"), a.getncattr("utespac_format")
        if fmt_b != "utespac-run-2" or fmt_a != "utespac-run-3":
            report.append(f"{os.path.basename(after)}: formats {fmt_b} -> {fmt_a}")
            return False
        pb, pa = _products(b), _products(a)
        for table, gb in pb.items():
            group = names.GROUPS[table]
            ga = pa.get(group)
            if ga is None:
                report.append(f"{os.path.basename(after)}: group {group} (legacy {table}) missing")
                ok = False
                continue
            if not np.array_equal(_read(gb.variables["time"]), _read(ga.variables["time"])):
                report.append(f"{os.path.basename(after)}: {group} time axis differs")
                ok = False
            if "height" in gb.variables and not np.array_equal(_read(gb.variables["height"]),
                                                                 _read(ga.variables["height"])):
                report.append(f"{os.path.basename(after)}: {group} heights differ")
                ok = False
            for key, vb in gb.variables.items():
                if key in ("time", "height"):
                    continue
                try:
                    new_group, new_name = names.legacy_to_new(table, key)
                except KeyError:
                    report.append(f"{os.path.basename(after)}: no alias for {table}/{key}")
                    ok = False
                    continue
                target = pa.get(new_group)
                va = None if target is None else target.variables.get(new_name)
                if va is None:
                    report.append(f"{os.path.basename(after)}: {new_group}/{new_name} (legacy {table}/{key}) missing")
                    ok = False
                    continue
                xb, xa = _read(vb), _read(va)
                if xb.shape != xa.shape or not np.array_equal(xb, xa, equal_nan=True):
                    nbad = int(np.sum(~np.isclose(xb, xa, equal_nan=True))) if xb.shape == xa.shape else -1
                    report.append(f"{os.path.basename(after)}: {new_group}/{new_name} differs "
                                  f"(legacy {table}/{key}; shape {xb.shape}->{xa.shape}; {nbad} cells)")
                    ok = False
        # derived columns
        for (group, name), (src_group, rho, cp, cov) in NEW_COLUMNS.items():
            g = pa.get(group)
            if g is None or name not in g.variables:
                report.append(f"{os.path.basename(after)}: new column {group}/{name} missing")
                ok = False
                continue
            s = pa[src_group].variables
            expect = _read(s[rho])[:, None] * _read(s[cp])[:, None] * _read(s[cov])
            got = _read(g.variables[name])
            if not np.allclose(expect, got, equal_nan=True, rtol=1e-12, atol=0):
                report.append(f"{os.path.basename(after)}: {group}/{name} != rho cp cov")
                ok = False
        sc, mo = pa.get("scaling"), pa.get("momentum")
        if sc is None or "ustar_pf" not in sc.variables:
            report.append(f"{os.path.basename(after)}: scaling/ustar_pf missing")
            ok = False
        else:
            tau = _read(mo.variables["Tau_pf"])
            expect = np.sqrt(np.where(tau < 0, np.nan, tau))
            if not np.allclose(expect, _read(sc.variables["ustar_pf"]), equal_nan=True, rtol=1e-12, atol=0):
                report.append(f"{os.path.basename(after)}: scaling/ustar_pf != sqrt(Tau_pf)")
                ok = False
        lh = pa.get("latent_heat")
        if lh is not None and "w_h2o_wpl_cov_pf" not in lh.variables:
            report.append(f"{os.path.basename(after)}: latent_heat/w_h2o_wpl_cov_pf missing")
            ok = False
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--glob", default="*_30minAvg_*.nc")
    a = ap.parse_args(argv)
    report, n_ok, n = [], 0, 0
    for fb in sorted(glob.glob(os.path.join(a.before, a.glob))):
        fa = os.path.join(a.after, os.path.basename(fb))
        n += 1
        if not os.path.exists(fa):
            report.append(f"{os.path.basename(fb)}: no after file")
            continue
        n_ok += compare_file(fb, fa, report)
    for line in report:
        print(line)
    print(f"{n_ok}/{n} files identical under the alias map")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
