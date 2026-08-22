"""Figure: UTESpac VAC001 vs EddyPro — scatter panels plus an H time series.

Companion to compare_vac001_eddypro.py (same loaders and join). Writes
testbed/scratch/vac001_<PF>_vs_eddypro.png.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/plot_vac001_eddypro.py [--pf LPF|GPF]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_vac001_eddypro import load_utespac, load_eddypro, stats, ROOT  # noqa: E402

# single data hue, a second hue for the fit/second series, neutral ink
C_DATA, C_FIT, C_INK, C_GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1"

PANELS = [
    ("H_Thv_wPF", "H",     "H  [W m$^{-2}$]  (UTESpac: Θv'wPF')"),
    ("LE_wPF",    "LE",    "LE  [W m$^{-2}$]  (WPL, wPF')"),
    ("Fc_wPF",    "Fc",    "Fc  [µmol m$^{-2}$ s$^{-1}$]  (WPL, wPF')"),
    ("ustar",     "ustar", "u*  [m s$^{-1}$]"),
    ("sigma_w",   "sigma_w", "σ$_w$  [m s$^{-1}$]"),
    ("TKE",       "TKE",   "TKE  [m$^2$ s$^{-2}$]"),
]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_INK, labelsize=8)
    ax.grid(True, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf", default="LPF", choices=["LPF", "GPF"])
    ap.add_argument("--det", default="LinDet", choices=["LinDet", "ConstDet"])
    args = ap.parse_args()

    ut, z, _ = load_utespac(args.pf, args.det)
    ep, _ = load_eddypro()
    j = ut.join(ep, how="inner", lsuffix="_ut", rsuffix="_ep")

    fig = plt.figure(figsize=(11, 10.5), facecolor="white")
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.42, wspace=0.32)

    for k, (uc, ec, title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        _style(ax)
        if uc not in ut.columns or ec not in ep.columns:
            ax.set_title(f"{title}\n(not available)", fontsize=9, color=C_INK)
            continue
        a = (j[uc] if uc in j.columns else j[f"{uc}_ut"]).values
        b = (j[ec] if ec in j.columns else j[f"{ec}_ep"]).values
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        s = stats(a, b)
        lo = np.nanmin([a.min(), b.min()]); hi = np.nanmax([a.max(), b.max()])
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=C_INK,
                linewidth=1, linestyle="--", zorder=1, label="1:1")
        xx = np.array([lo - pad, hi + pad])
        ax.plot(xx, s["slope"] * xx + s["intercept"], color=C_FIT,
                linewidth=1.6, zorder=2, label="OLS fit")
        ax.scatter(b, a, s=10, color=C_DATA, alpha=0.6, linewidths=0, zorder=3)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=9, color=C_INK)
        ax.set_xlabel("EddyPro", fontsize=8, color=C_INK)
        ax.set_ylabel("UTESpac", fontsize=8, color=C_INK)
        ax.text(0.03, 0.97,
                f"N={s['N']}\nslope {s['slope']:.3f}\nicpt {s['intercept']:.3g}\n"
                f"bias {s['bias']:.3g}\nr² {s['r2']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                color=C_INK)
        if k == 0:
            ax.legend(fontsize=7.5, loc="lower right", frameon=False)

    # time series of H, both sources
    ax = fig.add_subplot(gs[2, :])
    _style(ax)
    ax.plot(j.index, j["H_ep"] if "H_ep" in j.columns else j["H"],
            color=C_FIT, linewidth=1.2, label="EddyPro H")
    ax.plot(j.index, j["H_Thv_wPF"], color=C_DATA, linewidth=1.2,
            label="UTESpac H (Θv'wPF')")
    ax.axhline(0, color=C_INK, linewidth=0.6)
    ax.set_ylabel("H  [W m$^{-2}$]", fontsize=8, color=C_INK)
    ax.set_title(f"VAC001 2023 IOP, z = {z} m — sensible heat flux, 30-min",
                 fontsize=9, color=C_INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.autofmt_xdate(rotation=0, ha="center")

    det_label = {"LinDet": "linear detrend", "ConstDet": "block average"}[args.det]
    fig.suptitle(f"UTESpac ({args.pf}, {det_label}) vs EddyPro (planar fit, SND, WPL, "
                 f"spectral corr.) — {len(j)} joined periods",
                 fontsize=10, color=C_INK, y=0.995)

    out_dir = os.path.join(ROOT, "testbed", "scratch")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"vac001_{args.pf}_{args.det}_vs_eddypro.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"figure written: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
