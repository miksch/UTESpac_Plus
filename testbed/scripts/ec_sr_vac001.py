"""ec_coherent structure-function (SR) detector and TKE trigger on VAC001.

Runs the ``ramps`` module over one VAC001 HF file (default GPF ConstDet
2023-07-06, all 96 records) and draws: (a) the kinematic surface-renewal
flux alpha a z/(l+s) against the measured w'Ts' covariance per lag; (b) the
diurnal course of the linearized amplitude and period; (c) the two-lag d/s
split; (d) the lag dependence of S^3(r)/r (the Chen et al. 1997 t_m
diagnostic); (e) TKE-trigger event counts against the u' wavelet detector;
(f) one window's Ts' trace with the TKE triggers and u' detections.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_sr_vac001.py [HF_FILE] [--record N] [--no-run]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ec_coherent import ECConfig, io as ecio, ramps                  # noqa: E402
from ec_coherent import preprocess as pp                             # noqa: E402
from ec_coherent.cli import run_file                                 # noqa: E402

DEFAULT = os.path.join(ROOT, "data", "VAC001", "output", "VAC001_hf_GPF_ConstDet_2023_07_06.nc")
C_T, C_U, C_REF, C_INK = "#9b59b6", "#2a78d6", "#eb6834", "#52514e"
LAG_C = {0.25: "#c7b3e6", 0.5: "#9b59b6", 0.75: "#6a3d91", 1.0: "#3f2361"}


def main(argv):
    hf_path = next((a for a in argv if not a.startswith("--") and not a.isdigit()), DEFAULT)
    rec_show = int(argv[argv.index("--record") + 1]) if "--record" in argv else 21
    cfg = ECConfig.from_config(modules=("ramps",))
    out = ecio.output_path(hf_path, cfg.output_suffix)
    if "--no-run" not in argv or not os.path.exists(out):
        out = run_file(hf_path, cfg)
    ds = ecio.read_group(out, "ramps")
    hf = ecio.open_hf(hf_path)

    rec = ds.record.values
    lags = ds.sr_lag.values
    z = float(ds.height.values[0])
    wT = ds.wT.values[:, 0]
    F = ds.sr_flux_Ts.values[:, 0, :]                      # (record, lag)
    a_lin = ds.sr_a_Ts.values[:, 0, :]
    period = ds.sr_period_Ts.values[:, 0, :]
    d2 = ds.sr_d_Ts.values[:, 0, :]
    s2 = ds.sr_s_Ts.values[:, 0, :]
    il05 = int(np.argmin(np.abs(lags - 0.5)))
    valid = np.isfinite(F)
    r_per_lag = [np.corrcoef(wT[valid[:, il]], F[valid[:, il], il])[0, 1]
                 if valid[:, il].sum() > 2 else np.nan for il in range(len(lags))]
    m05 = valid[:, il05]
    slope05 = np.nanmedian(F[m05, il05] / wT[m05])
    # Castellvi & Snyder 2009 similarity alpha (eq. 3, inertial branch), d = 0 assumed
    ust = hf.ds["ustar"].values
    L = hf.ds["L"].values
    ust = ust[:, 0] if ust.ndim == 2 else ust
    L = L[:, 0] if L.ndim == 2 else L
    zeta = z / L
    alpha_C = ramps.alpha_castellvi(period[:, il05], ust, zeta, z)
    F_C = alpha_C * F[:, il05]
    mC = np.isfinite(F_C) & np.isfinite(wT)
    r_C = np.corrcoef(wT[mC], F_C[mC])[0, 1] if mC.sum() > 2 else np.nan
    slope_C = np.nanmedian(F_C[mC] / wT[mC])
    print(f"{os.path.basename(out)}: SR valid records per lag "
          f"{[int(valid[:, il].sum()) for il in range(len(lags))]} of {len(rec)}; "
          f"r(F_SR, w'Ts') per lag {[f'{r:.2f}' for r in r_per_lag]}; "
          f"median F_SR/w'Ts' at r=0.5 s: {slope05:.2f}; "
          f"a (r=0.5) median {np.nanmedian(a_lin[:, il05]):.2f} K, "
          f"l+s median {np.nanmedian(period[:, il05]):.0f} s, "
          f"two-lag d median {np.nanmedian(d2[:, il05]):.0f} s, s median {np.nanmedian(s2[:, il05]):.0f} s; "
          f"TKE events/30 min median {np.nanmedian(ds.n_events_e.values[:, 0]):.0f} "
          f"(u' wavelet {np.nanmedian(ds.n_events_u.values[:, 0]):.0f})")
    print(f"Castellvi alpha (r=0.5, d=0): median {np.nanmedian(alpha_C):.2f} "
          f"(IQR {np.nanpercentile(alpha_C, 25):.2f}-{np.nanpercentile(alpha_C, 75):.2f}); "
          f"corrected flux: r = {r_C:.2f}, median F_SR,C/w'Ts' = {slope_C:.2f} "
          f"(fixed alpha=1 gave {slope05:.2f})")
    # French et al. 2012 diagnostics: per-lag ratio, sign agreement, q-dominance
    ratio_lag = [np.nanmedian(F[valid[:, il], il] / wT[valid[:, il]]) for il in range(len(lags))]
    sgn, sgn_big = [], []
    for il in range(len(lags)):
        m = valid[:, il] & np.isfinite(wT)
        sgn.append(np.mean(np.sign(a_lin[m, il]) == np.sign(wT[m])))
        mb = m & (np.abs(wT) > 0.01)
        sgn_big.append(np.mean(np.sign(a_lin[mb, il]) == np.sign(wT[mb])))
    S3_05 = ds.sr_S3_rate_Ts.values[:, 0, il05] * lags[il05]
    a_q = np.cbrt(-10.0 * S3_05)
    mq = np.isfinite(a_q) & np.isfinite(a_lin[:, il05])
    print(f"French 2012 checks: median F_SR/w'Ts' by lag {[f'{v:.2f}' for v in ratio_lag]} "
          f"(lags {[float(v) for v in lags]} s); sign(a)==sign(w'Ts') "
          f"{[f'{v * 100:.0f}%' for v in sgn]} (|w'Ts'|>0.01: {[f'{v * 100:.0f}%' for v in sgn_big]}); "
          f"q-only amplitude cbrt(-10 S3): median a_q/a = {np.nanmedian(a_q[mq] / a_lin[mq, il05]):.3f}, "
          f"r = {np.corrcoef(a_q[mq], a_lin[mq, il05])[0, 1]:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    for il, r in enumerate(lags):
        m = valid[:, il]
        ax.plot(wT[m], F[m, il], ".", color=LAG_C.get(float(r), C_T), ms=4,
                label=f"r = {r:g} s (r_corr {r_per_lag[il]:.2f})")
    ax.plot(wT[mC], F_C[mC], "o", mfc="none", mec=C_REF, ms=5,
            label=f"alpha_C corrected, r = 0.5 s (r_corr {r_C:.2f}, med ratio {slope_C:.2f})")
    lim = np.nanmax(np.abs(np.r_[wT[np.isfinite(wT)], F[valid]])) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "-", color=C_INK, lw=0.8)
    ax.axhline(0, color=C_INK, lw=0.4); ax.axvline(0, color=C_INK, lw=0.4)
    ax.set_xlabel("w'Ts' [K m s$^{-1}$]"); ax.set_ylabel("F$_{SR}$ = a z/(l+s) [K m s$^{-1}$]")
    ax.set_title(f"(a) SR flux vs covariance (alpha = {ds.attrs['sr_alpha']:g} and Castellvi 2009 "
                 f"alpha_C, z = {z:g} m)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.plot(rec, a_lin[:, il05], ".", color=C_T, label="a (r = 0.5 s)")
    ax.axhline(0, color=C_INK, lw=0.5)
    ax.set_ylabel("ramp amplitude a [K]")
    ax2 = ax.twinx()
    ax2.plot(rec, period[:, il05], "x", color=C_REF, ms=4, alpha=0.7, label="l+s")
    ax2.plot(rec, ds.mean_spacing_u.values[:, 0], "+", color=C_U, ms=5, alpha=0.7,
             label="u' wavelet spacing")
    ax2.set_ylabel("period [s]")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(b) amplitude and period, diurnal course", fontsize=10)

    ax = axes[0, 2]
    ax.plot(rec, d2[:, il05], ".", color=C_T, label="d (ramp duration)")
    ax.plot(rec, s2[:, il05], ".", color=C_U, label="s (quiet gap)")
    ax.plot(rec, period[:, il05], "-", color=C_REF, lw=0.7, alpha=0.6, label="l+s (linearized)")
    ax.set_ylabel("[s]"); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.set_title("(c) two-lag d/s split (Paw U et al. 2005), r = 0.5 s", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 0]
    S3r = ds.sr_S3_rate_Ts.values[:, 0, :]
    for i in range(len(rec)):
        if np.isfinite(S3r[i]).any():
            ax.plot(lags, np.abs(S3r[i]), "-", color=C_T, alpha=0.15, lw=0.8)
    ax.plot(lags, np.nanmedian(np.abs(S3r), axis=0), "o-", color=C_INK, label="median")
    ax.set_yscale("log")
    ax.set_xlabel("lag r [s]"); ax.set_ylabel("|S$^3$(r)/r| [K$^3$ s$^{-1}$]")
    ax.set_title("(d) S$^3$(r)/r vs lag (rising = below Chen's t$_m$)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    ax.plot(rec, ds.n_events_e.values[:, 0], ".", color=C_REF, label="TKE trigger")
    ax.plot(rec, ds.n_events_u.values[:, 0], ".", color=C_U, label="u' wavelet")
    ax.plot(rec, ds.n_events_Ts.values[:, 0], ".", color=C_T, alpha=0.5, label="Ts' wavelet")
    ax.set_ylabel("events per 30 min"); ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    sf = ds.sweep_frac_e.values[:, 0, :]
    ax.set_title(f"(e) event counts (median sweep fraction {np.nanmedian(sf):.2f})", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    win = next(ecio.iter_windows(hf, variables=("u", "v", "w", "Ts"), records=[rec_show]))
    prep = pp.prepare(win, 0, ("u", "v", "w", "Ts"), method=cfg.preprocess.detrend)
    x = prep.prime["Ts"]
    t = np.arange(x.size) / hf.fs
    seg = (t >= 600) & (t < 1200)
    ax = axes[1, 2]
    ax.plot(t[seg], x[seg], "-", color=C_T, lw=0.7)
    ne = int(ds.n_events_e.values[rec_show, 0])
    te = ds.event_time_e.values[rec_show, 0, :ne]
    for ti in te[(te >= 600) & (te < 1200)]:
        ax.axvline(ti, color=C_REF, lw=1.0, alpha=0.9)
    nu = int(ds.n_events_u.values[rec_show, 0])
    tu = ds.event_time_u.values[rec_show, 0, :nu]
    for ti in tu[(tu >= 600) & (tu < 1200)]:
        ax.axvline(ti, color=C_U, lw=0.6, alpha=0.5, ls=":")
    ax.plot([], [], color=C_REF, label="TKE trigger")
    ax.plot([], [], color=C_U, ls=":", label="u' wavelet detection")
    ax.set_xlabel("t [s] from window start"); ax.set_ylabel("Ts' [K]")
    ax.set_title(f"(f) record {rec_show} ({str(rec[rec_show])[:16]}), "
                 f"{ne} TKE events / {nu} u' events", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"ec_coherent ramps (structure-function + TKE) -- {os.path.basename(hf_path)} "
                 f"(z = {z:g} m, detrend {ds.attrs['detrend_method']}, "
                 f"lags {[float(r) for r in lags]} s, tke_a_s {ds.attrs['tke_a_s']:g} s)",
                 fontsize=10, color=C_INK)
    fig.tight_layout()
    fig_out = os.path.join(ROOT, "testbed", "scratch", "ec_sr_vac001.png")
    fig.savefig(fig_out, dpi=140, bbox_inches="tight")
    print(f"figure written: {os.path.relpath(fig_out, ROOT)}")
    hf.close()


if __name__ == "__main__":
    main(sys.argv[1:])
