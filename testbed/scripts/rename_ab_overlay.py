"""Overlay figure: three fluxes before and after the variable-name refactor.

For one site, concatenates the ``utespac-run-2`` (before) and
``utespac-run-3`` (after) run files of one qualifier along time and draws
``w_theta_v_cov_pf``, ``LE_wpl_pf`` and ``Tau_pf`` (legacy ``Thv_wPF``,
``LE_WPL_wPF``, ``tau_PF``) at the highest sonic level, before as a wide
light line and after as a thin dark line, with the maximum absolute
difference printed in each panel. Companion to ``rename_ab_check.py``,
which does the exhaustive numeric comparison.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/rename_ab_overlay.py <before_dir> <after_dir> --site VAC001 \
        --qualifier GPF_ConstDet --out testbed/scratch/rename_ab_VAC001.png
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utespac import names  # noqa: E402

PANELS = [  # (legacy table, legacy key, ylabel)
    ("H", "Thv_wPF", "w'theta_v' [K m s-1]"),
    ("LHflux", "LE_WPL_wPF", "LE (WPL) [W m-2]"),
    ("tau", "tau_PF", "Tau (u*^2) [m2 s-2]"),
]
BEFORE_COLOR, AFTER_COLOR = "#2a78d6", "#eb6834"      # categorical slots 1 and 2


def _read(var):
    x = var[:]
    return np.where(np.ma.getmaskarray(x), np.nan, np.asarray(x, dtype=float))


def _series(files, group, name):
    ts, ys = [], []
    for f in files:
        with netCDF4.Dataset(f) as nc:
            g = nc["products"][group]
            t = _read(g.variables["time"])
            y = _read(g.variables[name])
            if y.ndim == 2:
                hi = int(np.argmax(_read(g.variables["height"])))
                y = y[:, hi]
            ts.append(t)
            ys.append(y)
    t = np.concatenate(ts)
    order = np.argsort(t)
    return t[order], np.concatenate(ys)[order]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--site", default="VAC001")
    ap.add_argument("--qualifier", default="GPF_ConstDet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pat = f"{a.site}_30minAvg_{a.qualifier}_*.nc"
    fb = sorted(glob.glob(os.path.join(a.before, pat)))
    fa = sorted(glob.glob(os.path.join(a.after, pat)))
    if not fb or not fa:
        raise SystemExit(f"no files for {pat}")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(10, 7.5), sharex=True)
    for ax, (table, key, ylabel) in zip(axes, PANELS):
        group, name = names.legacy_to_new(table, key)
        tb, yb = _series(fb, table, key)
        ta, ya = _series(fa, group, name)
        t0 = tb[0]
        days_b = (tb - t0) / 86.4e6
        days_a = (ta - t0) / 86.4e6
        ax.plot(days_b, yb, color=BEFORE_COLOR, lw=3.0, alpha=0.45, label=f"before: {table}/{key}")
        ax.plot(days_a, ya, color=AFTER_COLOR, lw=1.2, label=f"after: {group}/{name}")
        both = np.isfinite(yb) & np.isfinite(ya) if yb.shape == ya.shape else np.zeros(0, bool)
        maxdiff = float(np.nanmax(np.abs(ya - yb))) if yb.shape == ya.shape and both.any() else np.nan
        same_nan = bool(np.array_equal(np.isnan(yb), np.isnan(ya))) if yb.shape == ya.shape else False
        ax.text(0.01, 0.95, f"max |after - before| = {maxdiff:.3g}; NaN pattern identical: {same_nan}",
                transform=ax.transAxes, va="top", fontsize=9, color="#52514e")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel(f"days since first period ({a.site}, {a.qualifier}, highest sonic level)")
    fig.suptitle(f"{a.site}: run-file products before (utespac-run-2) and after (utespac-run-3) the rename")
    fig.tight_layout()
    out = a.out or os.path.join(ROOT, "testbed", "scratch", f"rename_ab_{a.site}_{a.qualifier}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
