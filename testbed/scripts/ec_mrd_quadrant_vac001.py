"""ec_coherent MRD and quadrant/octant analysis on VAC001.

Runs the ``mrd``, ``quadrant`` and ``octant`` modules over one VAC001 HF file
(default GPF ConstDet 2023-07-06, all 96 records) and draws: (a) record-median
MR cospectra of u'w' and w'Ts' with the detected gap scales; (b) the diurnal
course of the gap timescales; (c) quadrant flux fractions vs hole size H
(Raupach 1981 fig. 6 style, record medians, sign-aware grouping); (d) the
diurnal course of delta_S and eta against UTESpac reference behaviour;
(e) octant flux fractions of u'w' and w'Ts' (record medians, unstable
records); (f) MRD closure and relative sampling error.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_mrd_quadrant_vac001.py [HF_FILE] [--no-run]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ec_coherent import ECConfig, io as ecio                         # noqa: E402
from ec_coherent.cli import run_file                                 # noqa: E402

DEFAULT = os.path.join(ROOT, "data", "VAC001", "output", "VAC001_hf_GPF_ConstDet_2023_07_06.nc")
C_U, C_W, C_REF, C_INK = "#2a78d6", "#eb6834", "#9b59b6", "#52514e"
QCOL = {"Q1": "#c9c5bd", "Q2": "#2a78d6", "Q3": "#9b59b6", "Q4": "#eb6834"}


def main(argv):
    hf_path = next((a for a in argv if not a.startswith("--")), DEFAULT)
    cfg = ECConfig.from_config(modules=("mrd", "quadrant", "octant"))
    out = ecio.output_path(hf_path, cfg.output_suffix)
    if "--no-run" not in argv or not os.path.exists(out):
        out = run_file(hf_path, cfg)
    mr = ecio.read_group(out, "mrd")
    qd = ecio.read_group(out, "quadrant")
    oc = ecio.read_group(out, "octant")

    rec = mr.record.values
    midnight = rec[0].astype("datetime64[D]").astype("datetime64[ns]")
    hours = (rec - midnight).astype("timedelta64[s]").astype(float) / 3600.0
    tau = mr.mr_scale.values

    # ---- MRD closure and summary ----
    for k in ("uw", "wTs"):
        D = mr[f"D_{k}"].values[:, 0, :]
        cov = mr[f"cov_{k}"].values[:, 0]
        fin = np.isfinite(cov)
        resid = np.abs(D[fin].sum(axis=-1) - cov[fin])
        print(f"closure D_{k}: {fin.sum()} records, max |sum(D) - cov| = {resid.max():.3e}")
    gw, gm = mr.gap_wTs.values[:, 0], mr.gap_uw.values[:, 0]
    for name, g in (("gap_wTs", gw), ("gap_uw", gm)):
        f = g[np.isfinite(g)]
        print(f"{name} [s]: detected {f.size}/{len(g)}, median {np.median(f):.0f}, "
              f"IQR {np.percentile(f, 25):.0f}-{np.percentile(f, 75):.0f}" if f.size else
              f"{name}: none detected")

    # ---- quadrant summary ----
    holes = qd.hole.values
    Suw = qd.S_frac_uw.values[:, 0, :, :]              # (record, quadrant, hole)
    print("uw quadrant flux fractions at H=0 (medians): "
          + ", ".join(f"{q} {np.nanmedian(Suw[:, i, 0]):+.2f}"
                      for i, q in enumerate(qd.quadrant.values)))
    print("  Lu & Willmarth 1973 boundary-layer values: Q2 +0.77, Q4 +0.55 (interactions negative)")
    duw = qd.dur_frac_uw.values[:, 0, :, 0]
    print("uw duration fractions at H=0 (medians): "
          + ", ".join(f"{q} {np.nanmedian(duw[:, i]):.3f}" for i, q in enumerate(qd.quadrant.values))
          + "  [Li & Bo 2019 near-neutral: 0.19, 0.296, 0.2, 0.314]")
    for k in ("uw", "wTs"):
        dS = qd[f"delta_S_{k}"].values[:, 0]
        eta = qd[f"eta_{k}"].values[:, 0]
        print(f"{k}: delta_S median {np.nanmedian(dS):+.2f}, eta median {np.nanmedian(eta):.2f}, "
              f"IQR {np.nanpercentile(eta, 25):.2f}-{np.nanpercentile(eta, 75):.2f}")

    # ---- octant summary ----
    cov_wTs = qd.cov_wTs.values[:, 0]
    unstable = cov_wTs > 0
    Fuw = oc.flux_frac_u_w_ts_uw.values[:, 0, :]
    FwT = oc.flux_frac_u_w_ts_wts.values[:, 0, :]
    print(f"octants ({oc.attrs['octant_triplets']}), {unstable.sum()} records with w'Ts' > 0:")
    for name, F in (("u'w' (uwTs)", Fuw), ("w'Ts' (uwTs)", FwT)):
        med = np.nanmedian(F[unstable], axis=0)
        print(f"  {name} fractions: " + ", ".join(f"O{i+1} {v:+.2f}" for i, v in enumerate(med))
              + f"  (O2+O8 = {med[1] + med[7]:+.2f})")
    if "flux_frac_wTsrhov_wTs" in oc:
        Fq = oc.flux_frac_w_ts_rho_h2o_wts.values[:, 0, :]
        med = np.nanmedian(Fq[unstable], axis=0)
        print("  w'Ts' (wTsrhov) fractions: "
              + ", ".join(f"O{i+1} {v:+.2f}" for i, v in enumerate(med)))

    # ---- figure ----
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 12), constrained_layout=True)

    ax = axs[0, 0]
    Duw = mr.D_uw.values[:, 0, :]
    DwT = mr.D_wTs.values[:, 0, :]
    ax.semilogx(tau, np.nanmedian(Duw, axis=0), color=C_U, lw=1.4, label="$D_{uw}$")
    ax.semilogx(tau, np.nanmedian(DwT, axis=0), color=C_W, lw=1.4, label="$D_{wTs}$")
    ax.axhline(0, color=C_INK, lw=0.6)
    for g, c, lab in ((np.nanmedian(gm), C_U, "median gap uw"),
                      (np.nanmedian(gw), C_W, "median gap wTs")):
        if np.isfinite(g):
            ax.axvline(g, color=c, ls="--", lw=1.0, label=lab)
    ax.set_xlabel(r"averaging timescale $\tau$ [s]")
    ax.set_ylabel("MR cospectrum")
    ax.set_title("(a) record-median MR cospectra")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(hours, gw, "o", ms=3, color=C_W, label="gap wTs")
    ax.plot(hours, gm, "s", ms=3, color=C_U, label="gap uw")
    ax.set_yscale("log")
    ax.set_xlabel("hours since 00:00")
    ax.set_ylabel("gap timescale [s]")
    ax.set_title("(b) detected cospectral gap (per record; noisy by construction)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    for i, q in enumerate(qd.quadrant.values):
        ax.plot(holes, np.nanmedian(Suw[:, i, :], axis=0), "o-", ms=3,
                color=QCOL[str(q)], label=str(q))
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("hole size H (rms normalization)")
    ax.set_ylabel("median $S_{i,H}$ of u'w'")
    ax.set_title("(c) stress fractions vs hole size (Raupach 1981 fig. 6 style)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(hours, qd.delta_S_uw.values[:, 0], color=C_U, lw=1.1, label=r"$\Delta S$ uw")
    ax.plot(hours, qd.delta_S_wTs.values[:, 0], color=C_W, lw=1.1, label=r"$\Delta S$ wTs")
    ax.plot(hours, qd.eta_uw.values[:, 0], color=C_U, lw=1.1, ls="--", label=r"$\eta$ uw")
    ax.plot(hours, qd.eta_wTs.values[:, 0], color=C_W, lw=1.1, ls="--", label=r"$\eta$ wTs")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("hours since 00:00")
    ax.set_ylabel(r"$\Delta S$, $\eta$")
    ax.set_title("(d) sign-aware ejection-sweep asymmetry and efficiency")
    ax.legend(fontsize=8, ncol=2)

    ax = axs[2, 0]
    xs = np.arange(8)
    wdt = 0.4
    ax.bar(xs - wdt / 2, np.nanmedian(Fuw[unstable], axis=0), wdt, color=C_U, label="u'w'")
    ax.bar(xs + wdt / 2, np.nanmedian(FwT[unstable], axis=0), wdt, color=C_W, label="w'Ts'")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xticks(xs, [f"O{i+1}" for i in xs])
    ax.set_ylabel("median flux fraction")
    ax.set_title("(e) octant fractions, records with w'Ts' > 0 (Li & Bo 2019 layout)")
    ax.legend(fontsize=8)

    ax = axs[2, 1]
    err = mr.err_uw.values[:, 0, :]
    ok = np.isfinite(err).any(axis=0)                  # m = M has one sample: error undefined
    ax.loglog(tau[ok], np.nanmedian(err[:, ok], axis=0), "o-", ms=3, color=C_U, label="median")
    ax.loglog(tau[ok], np.nanpercentile(err[:, ok], 75, axis=0), ls="--", color=C_U, lw=0.9,
              label="75th pct")
    ax.axhline(0.1, color=C_REF, lw=0.8, ls=":", label="10 %")
    ax.set_xlabel(r"averaging timescale $\tau$ [s]")
    ax.set_ylabel(r"relative sampling error of $D_{uw}$")
    ax.set_title("(f) Howell & Mahrt 1997 eq. 12 sampling error")
    ax.legend(fontsize=8)

    fig.suptitle(os.path.basename(hf_path), fontsize=10)
    fig_path = os.path.join(ROOT, "testbed", "scratch", "ec_mrd_quadrant_vac001.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print("figure:", fig_path)


if __name__ == "__main__":
    main(sys.argv[1:])
