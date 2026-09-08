"""Energy-balance closure for VAC001: H + LE against Rn - G.

Available energy comes from the user's ``_soil_corr`` slow table
(data/VAC001/raw/slow): ``NETRAD`` (pieced together outside this repo) and
the rebuilt ground heat flux ``ghf_avg`` (plate + storage) with
``G_plate_avg`` as the no-storage alternative. Both are provisional per the
user (2026-08-22), so absolute closure ratios are indicative; the robust
result is the *relative* ranking of the turbulent-flux variants against the
same Rn - G:

  * UTESpac H (Θv'wPF', no Schotanus) + LE(WPL)      -- as computed today
  * UTESpac H - 0.51 T cp/Lv * LE (Schotanus applied post hoc) + LE
  * EddyPro H + LE

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/closure_vac001.py [--pf GPF|LPF] [--det ConstDet|LinDet]

Prints closure statistics (all periods, daytime Rn > 50 W/m2) and writes
testbed/scratch/vac001_closure_<pf>_<det>.png.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_vac001_eddypro import load_utespac, load_eddypro, stats, ROOT, SITE_DIR  # noqa: E402

C1, C2, C3, C_INK, C_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e6e5e1"
SND = 0.51 * 296.0 * 1005.0 / 2.45e6     # W/m2 of H per W/m2 of LE (T ~ 296 K)


def load_slow():
    f = os.path.join(SITE_DIR, "raw", "slow", "VAC_001_slow_logger_vars_2023_soil_corr.csv")
    df = pd.read_csv(f, header=0, skiprows=[1], index_col=0, na_values=["NaN", "NAN"],
                     low_memory=False)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, format="mixed")).round("min")
    cols = ["NETRAD", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT", "ghf_avg", "G_plate_avg",
            "soil_storage_avg", "H", "LE"]
    out = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    # the logger's broken storage columns leak 1e9 values; guard the rebuilt ones too
    for c in ("ghf_avg", "G_plate_avg", "soil_storage_avg", "NETRAD"):
        if c in out:
            out.loc[out[c].abs() > 2000, c] = np.nan
    return out.rename(columns={"H": "H_logger", "LE": "LE_logger"})


def closure_stats(turb, avail):
    m = np.isfinite(turb) & np.isfinite(avail)
    t, a = turb[m], avail[m]
    s = stats(t, a)                                # OLS turb = slope*avail + icpt
    ratio = t.sum() / a.sum() if a.sum() != 0 else np.nan
    forced = (t * a).sum() / (a * a).sum()         # slope through origin
    return dict(N=int(m.sum()), ratio=ratio, slope=s["slope"], icpt=s["intercept"],
                forced=forced, r2=s["r2"], resid=float(np.mean(a - t)))


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=C_INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf", default="GPF", choices=["GPF", "LPF"])
    ap.add_argument("--det", default="ConstDet", choices=["ConstDet", "LinDet"])
    args = ap.parse_args()

    ut, z, _ = load_utespac(args.pf, args.det)
    ep, _ = load_eddypro()
    sl = load_slow()
    j = ut.join(ep[["H", "LE"]].rename(columns={"H": "H_ep", "LE": "LE_ep"}), how="inner") \
          .join(sl, how="inner")
    j["H_ut"] = j["H_buoyancy_pf"]
    j["H_ut_snd"] = j["H_buoyancy_pf"] - SND * j["LE_wpl_pf"]
    j["LE_ut"] = j["LE_wpl_pf"]
    print(f"joined periods: {len(j)}  ({j.index.min()} .. {j.index.max()})")
    print(f"NETRAD finite: {j['NETRAD'].notna().sum()}, ghf_avg finite: {j['ghf_avg'].notna().sum()}, "
          f"G_plate_avg finite: {j['G_plate_avg'].notna().sum()}")
    print("daytime (NETRAD>50) medians: NETRAD %.0f, ghf_avg %.1f, G_plate_avg %.1f, storage %.1f W/m2"
          % tuple(j.loc[j.NETRAD > 50, c].median() for c in
                  ["NETRAD", "ghf_avg", "G_plate_avg", "soil_storage_avg"]))

    variants = [
        ("UTESpac H(Θv'wPF') + LE",            j["H_ut"] + j["LE_ut"]),
        ("UTESpac H − SND(LE) + LE",           j["H_ut_snd"] + j["LE_ut"]),
        ("EddyPro H + LE",                     j["H_ep"] + j["LE_ep"]),
        ("logger EasyFlux H + LE (slow table)", j["H_logger"] + j["LE_logger"]),
    ]
    avail_defs = [("Rn − ghf_avg (plate+storage)", j["NETRAD"] - j["ghf_avg"]),
                  ("Rn − G_plate_avg",            j["NETRAD"] - j["G_plate_avg"]),
                  ("Rn only (no G)",              j["NETRAD"])]

    for sel_name, sel in [("all periods", np.ones(len(j), bool)),
                          ("daytime, Rn > 50 W/m2", (j["NETRAD"] > 50).values)]:
        print(f"\n=== {sel_name} ===")
        print(f"{'turbulent':40s} {'available':32s} {'N':>4s} {'sum ratio':>9s} {'slope0':>7s} "
              f"{'slope':>6s} {'icpt':>7s} {'r2':>6s} {'mean resid':>10s}")
        for an, a in avail_defs:
            for vn, v in variants:
                s = closure_stats(v.values[sel], a.values[sel])
                print(f"{vn:40s} {an:32s} {s['N']:4d} {s['ratio']:9.3f} {s['forced']:7.3f} "
                      f"{s['slope']:6.3f} {s['icpt']:7.1f} {s['r2']:6.3f} {s['resid']:10.1f}")

    # ---- figure: scatter vs Rn - ghf_avg, and mean diurnal composite ----------
    a = j["NETRAD"] - j["ghf_avg"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor="white")
    ax = axes[0]; _style(ax)
    lo, hi = np.nanmin(a) - 20, np.nanmax(a) + 20
    ax.plot([lo, hi], [lo, hi], "--", color=C_INK, linewidth=1, label="1:1")
    for (vn, v), col in zip(variants[:3], (C1, C2, C3)):
        s = closure_stats(v.values, a.values)
        ax.scatter(a, v, s=9, color=col, alpha=0.55, linewidths=0,
                   label=f"{vn}: Σ ratio {s['ratio']:.2f}, slope {s['slope']:.2f}")
    ax.set_xlabel("Rn − G (ghf_avg)  [W m$^{-2}$]  (provisional)", fontsize=8, color=C_INK)
    ax.set_ylabel("H + LE  [W m$^{-2}$]", fontsize=8, color=C_INK)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("30-min periods, whole IOP", fontsize=9, color=C_INK)

    ax = axes[1]; _style(ax)
    hr = j.index.hour + j.index.minute / 60.0
    comp = pd.DataFrame({"Rn-G": a, "Rn": j["NETRAD"], "G": j["ghf_avg"],
                         variants[0][0]: variants[0][1], variants[1][0]: variants[1][1],
                         variants[2][0]: variants[2][1]}).groupby(hr).mean()
    ax.plot(comp.index, comp["Rn-G"], color=C_INK, linewidth=1.6, label="Rn − G")
    ax.plot(comp.index, comp["Rn"], color=C_INK, linewidth=0.8, linestyle=":", label="Rn")
    for (vn, _), col in zip(variants[:3], (C1, C2, C3)):
        ax.plot(comp.index, comp[vn], color=col, linewidth=1.4, label=vn)
    ax.axhline(0, color=C_GRID, linewidth=0.8)
    ax.set_xlabel("hour (local, period end)", fontsize=8, color=C_INK)
    ax.set_ylabel("W m$^{-2}$", fontsize=8, color=C_INK)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("mean diurnal composite", fontsize=9, color=C_INK)
    fig.suptitle(f"VAC001 energy-balance closure — UTESpac {args.pf}/{args.det} vs EddyPro; "
                 f"Rn and G provisional", fontsize=10, color=C_INK)
    out = os.path.join(ROOT, "testbed", "scratch", f"vac001_closure_{args.pf}_{args.det}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nfigure written: {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
