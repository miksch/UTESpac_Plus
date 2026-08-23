"""Planar-fit coefficients for VAC001: UTESpac vs EddyPro, plus the 3D
validation figure (binned (u,v,w) means against the fitted plane).

Reads the LPF 30-min averaged pickles (the same means find_global_pf
regresses on), fits b0,b1,b2 with utespac.pf_coefficients over (a) the
EddyPro planar-fit period 2023-07-06..07-08 and (b) the whole IOP, and
prints them next to EddyPro's B0/B1/B2 from its planar_fit_*.txt. The
figure shows the point cloud with three planes: as fitted, as the legacy
(MATLAB-indexed) code applied it, and EddyPro's.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/pf_vac001_eddypro.py
"""

import glob
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from utespac.pf_coefficients import pf_coefficients      # noqa: E402
from utespac.rotation import pf_matrix as _build_pf_matrix   # noqa: E402
from utespac.testkit import get_header                  # noqa: E402

SITE_DIR = os.path.join(ROOT, "data", "VAC001")
EPOCH = 719529.0
C_DATA, C_FIT, C_WRONG, C_EP, C_INK = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#52514e"


def angles(b1, b2):
    pitch = np.degrees(np.arcsin(-b1 / np.sqrt(1 + b1**2)))
    roll = np.degrees(np.arcsin(b2 / np.sqrt(1 + b2**2)))
    return pitch, roll


def load_means():
    frames = []
    for f in sorted(glob.glob(os.path.join(SITE_DIR, "output", "*_30minAvg_LPF_LinDet_*.pkl"))):
        p = pickle.load(open(f, "rb"))
        tbl, hdr = p["VAC001_20Hz"], p["VAC001_20HzHeader"][0]
        sd, sdh = p["spdAndDir"], p["spdAndDirHeader"]
        t = pd.to_datetime((tbl[:, 0] - EPOCH) * 86400, unit="s").round("min")
        df = pd.DataFrame({"u": tbl[:, hdr.index("Ux_10.85")],
                           "v": tbl[:, hdr.index("Uy_10.85")],
                           "w": tbl[:, hdr.index("Uz_10.85")],
                           "spd": sd[:, sdh.index("10.85m speed")],
                           "dir": sd[:, sdh.index("10.85m direction")]}, index=t)
        frames.append(df)
    return pd.concat(frames).sort_index()


def eddypro_pf():
    files = sorted(glob.glob(os.path.join(SITE_DIR, "eddypro", "*", "*planar_fit*.txt")))
    txt = open(files[-1]).read()
    m = re.search(r"^\s*1\s+0-\s*360\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)", txt, re.M)
    b0, b1, b2 = map(float, m.groups())
    per = re.search(r"Beginning_of_planar_fit_determination_period:\s*(\S+).*?End_of_planar_fit_determination_period:\s*(\S+)", txt, re.S)
    return (b0, b1, b2), per.groups(), os.path.relpath(files[-1], ROOT)


def fit(df, lo=0.5, hi=20.0):
    d = df[(df.spd >= lo) & (df.spd <= hi)].dropna()
    c = pf_coefficients(np.column_stack([d.u, d.v, d.w, d.dir]), [])["degrees_0_to_0"]
    return c, len(d)


def main():
    df = load_means()
    (eb0, eb1, eb2), (p0, p1), ep_file = eddypro_pf()
    ep_mask = (df.index >= p0) & (df.index < pd.Timestamp(p1) + pd.Timedelta(days=1))

    rows = []
    for label, sub, lo, hi in [
        ("UTESpac, EddyPro period, EddyPro filter (u>=0.1, |w|<=1)", df[ep_mask], 0.1, 99),
        ("UTESpac, EddyPro period, UTESpac filter (0.5-20 m/s)", df[ep_mask], 0.5, 20),
        ("UTESpac, whole IOP, UTESpac filter (0.5-20 m/s)", df, 0.5, 20),
    ]:
        c, n = fit(sub, lo, hi)
        rows.append((label, n, *c, *angles(c[1], c[2])))
    rows.insert(0, (f"EddyPro planar_fit ({p0}..{p1})", 714, eb0, eb1, eb2, *angles(eb1, eb2)))
    cw = angles(rows[1][2], rows[1][3])   # legacy code: b0 as b1, b1 as b2
    print(f"EddyPro file: {ep_file}\n")
    print(f"{'fit':60s} {'N':>4s} {'b0':>9s} {'b1':>9s} {'b2':>9s} {'pitch':>7s} {'roll':>7s}")
    for r in rows:
        print(f"{r[0]:60s} {r[1]:4d} {r[2]:9.4f} {r[3]:9.4f} {r[4]:9.4f} {r[5]:7.2f} {r[6]:7.2f}")
    print(f"\nlegacy indexing would apply (b0,b1) as (b1,b2): pitch {cw[0]:.2f}, roll {cw[1]:.2f}")

    # ---- 3D figure: whole-IOP cloud with the three planes ---------------------
    c_fit, _ = fit(df, 0.5, 20)
    d = df[(df.spd >= 0.5) & (df.spd <= 20)].dropna()
    uu, vv = np.meshgrid(np.linspace(d.u.min(), d.u.max(), 12),
                         np.linspace(d.v.min(), d.v.max(), 12))
    fig = plt.figure(figsize=(11, 5), facecolor="white")
    for k, (elev, azim) in enumerate([(18, -60), (4, -150)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(d.u, d.v, d.w, s=6, color=C_DATA, alpha=0.55, linewidths=0, label="30-min means")
        for (b0, b1, b2), col, lab in [
            (c_fit, C_FIT, f"fitted  b0={c_fit[0]:.3f} b1={c_fit[1]:.3f} b2={c_fit[2]:.3f}"),
            ((0.0, c_fit[0], c_fit[1]), C_WRONG, "as legacy code applied (b0,b1 as b1,b2)"),
            ((eb0, eb1, eb2), C_EP, f"EddyPro b0={eb0:.3f} b1={eb1:.3f} b2={eb2:.3f}"),
        ]:
            ax.plot_surface(uu, vv, b0 + b1 * uu + b2 * vv, color=col, alpha=0.22,
                            linewidth=0, shade=False)
            ax.plot([], [], color=col, linewidth=6, alpha=0.5, label=lab)
        ax.set_xlabel("ū [m/s]", fontsize=8); ax.set_ylabel("v̄ [m/s]", fontsize=8)
        ax.set_zlabel("w̄ [m/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)
        if k == 0:
            ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("VAC001 planar fit, whole IOP 30-min means (0.5-20 m/s): point cloud and planes",
                 fontsize=10, color=C_INK)
    out = os.path.join(ROOT, "testbed", "scratch", "vac001_planar_fit_3d.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nfigure written: {os.path.relpath(out, ROOT)}")

    # residual check: mean rotated w with and without b0 removal
    P = _build_pf_matrix(c_fit[1], c_fit[2])
    xyz = np.column_stack([d.u, d.v, d.w])
    w_no_b0 = (P @ xyz.T).T[:, 2]
    w_b0 = (P @ np.column_stack([d.u, d.v, d.w - c_fit[0]]).T).T[:, 2]
    print(f"mean rotated w over the cloud: without b0 removal {w_no_b0.mean():+.4f} m/s, "
          f"with b0 removal {w_b0.mean():+.4f} m/s (b0={c_fit[0]:.4f})")


if __name__ == "__main__":
    main()
