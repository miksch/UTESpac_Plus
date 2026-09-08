"""ec_coherent coherent-structure flux fractions on VAC001.

Runs the ``coherent_flux`` module over one VAC001 HF file (default GPF
ConstDet 2023-07-06, all 96 records) and draws: (a) the conditional-average
patterns about the u' events for the day's largest-|w'Ts'| record (the
Thomas & Foken 2007 fig. 2 analog: <w'>, <Ts'> and the products); (b) the
diurnal course of the wavelet-estimator coherent fraction F_cs/F_tot for
w'Ts' and u'w'; (c) the wavelet vs quadrant estimator comparison per hole
size (their §4.2 factor 3-4 finding); (d) the EC relative flux error of
their eq. 12c (their fig. 6: within 4%).

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_coherent_flux_vac001.py [HF_FILE] [--no-run]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ec_coherent import ECConfig, coherent_flux as cflux, io as ecio     # noqa: E402
from ec_coherent import preprocess as pp, ramps                          # noqa: E402
from ec_coherent.cli import run_file                                     # noqa: E402

DEFAULT = os.path.join(ROOT, "data", "VAC001", "output", "VAC001_hf_GPF_ConstDet_2023_07_06.nc")
C_U, C_W, C_REF, C_INK = "#2a78d6", "#eb6834", "#9b59b6", "#52514e"


def patterns(hf, cfg, i_rec):
    """Conditional averages about the u' events of one record (fig. 2 analog)."""
    rc, pc, cf = cfg.ramps, cfg.preprocess, cfg.coherent_flux
    scales = ramps.log_scales(rc.a_min_s, rc.a_max_s, rc.n_scales_per_decade)
    for win in ecio.iter_windows(hf, variables=("u_pf", "v_pf", "w_pf", "ts"), records=[i_rec]):
        prep = pp.prepare(win, 0, ["u_pf", "w_pf", "ts"], method=pc.detrend, tau_s=pc.filter_tau_s)
        pu, pw, pT = (prep.prime[k] for k in ("u_pf", "w_pf", "ts"))
        wT = float(np.nanmean(pw * pT))
        ev = ramps.detect(pu, hf.fs, scales, slope=ramps._slope_for("u_pf", rc.slope_u, wT),
                          peak=rc.peak, edge_scales=rc.edge_scales, D_min_s=rc.D_min_s)
        half = ev.D if cf.window == "duration" else 0.5 * cf.window_s
        lag, cw = cflux.conditional_average(pw, hf.fs, ev.times, half)
        _, cT = cflux.conditional_average(pT, hf.fs, ev.times, half)
        _, cu = cflux.conditional_average(pu, hf.fs, ev.times, half)
        _, cwT = cflux.conditional_average(pw * pT, hf.fs, ev.times, half)
        return dict(lag=lag, cu=cu / pu.std(), cw=cw / pw.std(), cT=cT / pT.std(),
                    cwT=cwT / abs(wT), prod=(cw * cT) / abs(wT), n=ev.n, D=ev.D, wT=wT)
    return None


def turner_sweep(hf, cfg, Ks):
    """Median w'Ts' strong-covariance fraction across records for each threshold K."""
    pc = cfg.preprocess
    fr = {K: [] for K in Ks}
    for win in ecio.iter_windows(hf, variables=("u_pf", "v_pf", "w_pf", "ts")):
        prep = pp.prepare(win, 0, ["w_pf", "ts"], method=pc.detrend, tau_s=pc.filter_tau_s)
        if not (prep.accepted.get("w_pf") and prep.accepted.get("ts")):
            continue
        pw, pT = prep.prime["w_pf"], prep.prime["ts"]
        if not (np.isfinite(pw).all() and np.isfinite(pT).all()):
            continue
        for K in Ks:
            fr[K].append(cflux.turner_fraction(pw, pT, K)["frac"])
    return {K: float(np.nanmedian(v)) if v else np.nan for K, v in fr.items()}


def main(argv):
    hf_path = next((a for a in argv if not a.startswith("--")), DEFAULT)
    cfg = ECConfig.from_config(modules=("coherent_flux",),
                               coherent_flux={"turner": True})   # third estimator on for the comparison
    out = ecio.output_path(hf_path, cfg.output_suffix)
    if "--no-run" not in argv or not os.path.exists(out):
        out = run_file(hf_path, cfg)
    ds = ecio.read_group(out, "coherent_flux")
    hf = ecio.open_hf(hf_path)

    rec = ds.record.values
    midnight = rec[0].astype("datetime64[D]").astype("datetime64[ns]")
    hours = (rec - midnight).astype("timedelta64[s]").astype(float) / 3600.0

    frac_wT = ds.F_frac_wTs_u.values[:, 0]
    frac_uw = ds.F_frac_uw_u.values[:, 0]
    n_u = ds.n_structures_u.values[:, 0]
    # N = 1 degenerates to F_cs = F_tot identically (library note); drop from the statistics
    many = n_u >= 2
    valid_wT = ds.valid_wTs_u.values[:, 0].astype(bool) & many
    valid_uw = ds.valid_uw_u.values[:, 0].astype(bool) & many
    ratio_wT = ds.ratio_wTs_u.values[:, 0]
    holes = ds.hole.values
    quad_wT = ds.F_coh_quad_wTs.values[:, 0, :]
    quad_uw = ds.F_coh_quad_uw.values[:, 0, :]

    print(f"u' events per record: median {np.median(n_u):.0f}; "
          f"half-window D_e median {np.nanmedian(ds.half_window_u.values):.1f} s")
    print(f"single-event records dropped (F_cs = F_tot identity): {np.count_nonzero(n_u == 1)}")
    print(f"F_tot/cov (wTs, u' events): {np.isfinite(ratio_wT).sum()} finite, "
          f"{valid_wT.sum()} inside the 0.8-1.2 gate")
    for name, fr, va in (("w'Ts'", frac_wT, valid_wT), ("u'w'", frac_uw, valid_uw)):
        v = fr[va]
        print(f"F_cs/F_tot {name} (u' events, valid records): median {np.nanmedian(v):.2f}, "
              f"IQR {np.nanpercentile(v, 25):.2f}-{np.nanpercentile(v, 75):.2f}  "
              f"[TF2007 means: momentum 0.16, scalars 0.26; CB93b 0.26/0.40]")
    ej, sw = ds.F_ej_wTs_u.values[:, 0], ds.F_sw_wTs_u.values[:, 0]
    r = np.where(ej != 0, sw / ej, np.nan)[valid_wT]
    print(f"sweep/ejection ratio (wTs): median {np.nanmedian(r):.2f} "
          f"[TF2007 above canopy: < 1]")
    for j, L in enumerate(holes):
        q = quad_wT[valid_wT, j]
        print(f"quadrant estimator wTs, L = {L}: median {np.nanmedian(q):.2f} "
              f"(ratio to wavelet {np.nanmedian(q / frac_wT[valid_wT]):.1f}x)")
    tfr_wT = ds.F_frac_turner_wTs.values[:, 0]
    tfr_uw = ds.F_frac_turner_uw.values[:, 0]
    trat = ds.ratio_turner_wTs.values[:, 0]
    print(f"turner estimator (K = {ds.attrs['turner_K']:.0f}): F_frac wTs median "
          f"{np.nanmedian(tfr_wT[valid_wT]):.2f}, uw {np.nanmedian(tfr_uw[valid_uw]):.2f}; "
          f"reconstruction ratio median {np.nanmedian(trat):.2f}")
    sw_K = turner_sweep(hf, cfg, (1.0, 1.5, 2.0, 3.0, 4.0))
    print("turner K sweep, median F_frac wTs:",
          ", ".join(f"K={K:g} {v:.2f}" for K, v in sw_K.items()),
          "[TL94: K=2 admits background, K=6 weakens events]")
    ferr_wT = ds.flux_error_wTs_u.values[:, 0] / ds.cov_wTs.values[:, 0]
    ferr_uw = ds.flux_error_uw_u.values[:, 0] / ds.cov_uw.values[:, 0]
    print(f"relative EC flux error (eq. 12b/12c): wTs median |err| "
          f"{np.nanmedian(np.abs(ferr_wT[valid_wT])) * 100:.1f}%, "
          f"uw {np.nanmedian(np.abs(ferr_uw[valid_uw])) * 100:.1f}% [TF2007: < 4% mostly]")
    n_Ts = ds.n_structures_ts.values[:, 0]
    print(f"Ts' event set alongside: events in {np.count_nonzero(n_Ts)} of {len(rec)} records "
          f"(median {np.median(n_Ts[n_Ts > 0]) if (n_Ts > 0).any() else float('nan'):.0f} where found)")

    cov_wT = ds.cov_wTs.values[:, 0]
    i_best = int(np.nanargmax(np.abs(np.where(valid_wT, cov_wT, np.nan))))
    pat = patterns(hf, cfg, i_best)

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(pat["lag"], pat["cu"], color=C_U, lw=1.3, label=r"$\langle u'\rangle/\sigma_u$")
    ax.plot(pat["lag"], pat["cw"], color=C_W, lw=1.3, label=r"$\langle w'\rangle/\sigma_w$")
    ax.plot(pat["lag"], pat["cT"], color=C_REF, lw=1.3, label=r"$\langle T_s'\rangle/\sigma_T$")
    ax.plot(pat["lag"], pat["cwT"], color=C_INK, lw=1.0, ls="--",
            label=r"$\langle w'T_s'\rangle/|\overline{w'T_s'}|$")
    ax.plot(pat["lag"], pat["prod"], color=C_INK, lw=1.0,
            label=r"$\langle w'\rangle\langle T_s'\rangle/|\overline{w'T_s'}|$")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.axvline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("lag about the u' event [s]")
    ax.set_title(f"(a) conditional averages, record {i_best} "
                 f"(N = {pat['n']}, $D_e$ = {pat['D']:.1f} s)")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(hours, frac_wT, color=C_W, lw=1.0, alpha=0.45)
    ax.plot(hours[valid_wT], frac_wT[valid_wT], "o", ms=4, color=C_W, label="w'Ts' (valid)")
    ax.plot(hours, frac_uw, color=C_U, lw=1.0, alpha=0.45)
    ax.plot(hours[valid_uw], frac_uw[valid_uw], "s", ms=4, color=C_U, label="u'w' (valid)")
    for y, lab in ((0.26, "TF2007 scalars 0.26"), (0.16, "TF2007 momentum 0.16")):
        ax.axhline(y, color=C_INK, lw=0.7, ls=":")
        ax.text(23.8, y, lab, fontsize=7, ha="right", va="bottom", color=C_INK)
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_ylim(-0.5, 1.2)
    ax.set_xlabel("hours since 2023-07-06 00:00")
    ax.set_ylabel(r"$F_{cs}/F_{tot}$")
    ax.set_title("(b) wavelet-estimator coherent fraction, u' events")
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1, 0]
    wav = np.nanmedian(frac_wT[valid_wT])
    med_q = [np.nanmedian(quad_wT[valid_wT, j]) for j in range(len(holes))]
    med_qu = [np.nanmedian(quad_uw[valid_uw, j]) for j in range(len(holes))]
    ax.plot(holes, med_q, "o-", color=C_W, label="quadrant, w'Ts'")
    ax.plot(holes, med_qu, "s-", color=C_U, label="quadrant, u'w'")
    ax.axhline(wav, color=C_W, ls="--", lw=1.0, label="wavelet, w'Ts'")
    ax.axhline(np.nanmedian(frac_uw[valid_uw]), color=C_U, ls="--", lw=1.0, label="wavelet, u'w'")
    ax.axhline(np.nanmedian(tfr_wT[valid_wT]), color=C_W, ls=":", lw=1.2,
               label="turner, w'Ts' (K = 4)")
    ax.axhline(np.nanmedian(tfr_uw[valid_uw]), color=C_U, ls=":", lw=1.2,
               label="turner, u'w' (K = 4)")
    ax.set_xlabel("hole size L")
    ax.set_ylabel("median coherent flux fraction")
    ax.set_title("(c) estimator comparison (TF2007 §4.2: quadrant 2-4x larger)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    bins = np.linspace(-0.1, 0.1, 41)
    ax.hist(ferr_wT[valid_wT], bins=bins, color=C_W, alpha=0.6, label="w'Ts'")
    ax.hist(ferr_uw[valid_uw], bins=bins, color=C_U, alpha=0.6, label="u'w'")
    for x in (-0.04, 0.04):
        ax.axvline(x, color=C_INK, lw=0.8, ls=":")
    ax.set_xlabel(r"relative flux error $\Delta\overline{x'y'}/\overline{x'y'}$")
    ax.set_ylabel("records")
    ax.set_title("(d) EC flux error from coherent structures (TF2007 fig. 6: |err| < 4%)")
    ax.legend(fontsize=8)

    fig.suptitle(os.path.basename(hf_path), fontsize=10)
    fig_path = os.path.join(ROOT, "testbed", "scratch", "ec_coherent_flux_vac001.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print("figure:", fig_path)
    hf.close()


if __name__ == "__main__":
    main(sys.argv[1:])
