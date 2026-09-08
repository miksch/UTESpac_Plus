"""ec_coherent amplitude modulation and LSM/VLSM separation on VAC001.

Runs the ``ampmod`` and ``scales`` modules over one VAC001 HF file (default
GPF ConstDet 2023-07-06, all 96 records) and draws: (a) the diurnal course of
the single-point AM coefficients R (u_L and w_L modulators); (b) R against
z/L; (c) the cutoff-sensitivity sweep of the deviation register (R vs
lambda_c over more than a decade, Mathis et al. 2009 §7.1.3 repeated on
field data); (d) the record-median pre-multiplied u' spectrum with the
detected cutoffs; (e) the diurnal course of the wavelength-band fractions of
u' variance and of the w'Ts' covariance.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_ampmod_scales_vac001.py [HF_FILE] [--no-run]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ec_coherent import ECConfig, ampmod, io as ecio                 # noqa: E402
from ec_coherent import preprocess as pp, spectra as sp              # noqa: E402
from ec_coherent.cli import run_file                                 # noqa: E402

DEFAULT = os.path.join(ROOT, "data", "VAC001", "output", "VAC001_hf_GPF_ConstDet_2023_07_06.nc")
C_U, C_W, C_REF, C_INK = "#2a78d6", "#eb6834", "#9b59b6", "#52514e"
BAND_C = {"small": "#c9c5bd", "lsm": "#2a78d6", "vlsm": "#eb6834"}
SWEEP_LAMBDA = (200.0, 350.0, 600.0, 1000.0, 1800.0, 3000.0)


def sweep(hf, cfg, lambdas):
    """Median R_uL_uS and R_wL_uS across all records for each fixed cutoff wavelength."""
    med = {"u_pf": [], "w_pf": []}
    pc = cfg.preprocess
    Rs = {lam: {"u_pf": [], "w_pf": []} for lam in lambdas}
    for win in ecio.iter_windows(hf, variables=("u_pf", "v_pf", "w_pf")):
        prep = pp.prepare(win, 0, ["u_pf", "w_pf"], method=pc.detrend, tau_s=pc.filter_tau_s,
                          nan_max_frac=pc.nan_max_frac, taylor_max_ratio=pc.taylor_max_ratio)
        if not (prep.accepted.get("u_pf") and prep.accepted.get("w_pf")):
            continue
        pu, pw = prep.prime["u_pf"], prep.prime["w_pf"]
        if not (np.isfinite(pu).all() and np.isfinite(pw).all()):
            continue
        for lam in lambdas:
            fc = prep.U_mean / lam
            if fc <= win.fs / pu.size:
                continue
            us = pu - ampmod.lowpass_sharp(pu, win.fs, fc)
            env_l = ampmod.lowpass_sharp(ampmod.envelope(us), win.fs, fc)
            Rs[lam]["u_pf"].append(ampmod.am_coefficient(ampmod.lowpass_sharp(pu, win.fs, fc), env_l))
            Rs[lam]["w_pf"].append(ampmod.am_coefficient(ampmod.lowpass_sharp(pw, win.fs, fc), env_l))
    for m in ("u_pf", "w_pf"):
        med[m] = [np.nanmedian(Rs[lam][m]) if Rs[lam][m] else np.nan for lam in lambdas]
    return med


def main(argv):
    hf_path = next((a for a in argv if not a.startswith("--")), DEFAULT)
    cfg = ECConfig.from_config(modules=("ampmod", "scales"))
    out = ecio.output_path(hf_path, cfg.output_suffix)
    if "--no-run" not in argv or not os.path.exists(out):
        out = run_file(hf_path, cfg)
    am = ecio.read_group(out, "amplitude_mod")
    sc = ecio.read_group(out, "scale_separation")
    hf = ecio.open_hf(hf_path)

    rec = am.record.values
    midnight = rec[0].astype("datetime64[D]").astype("datetime64[ns]")
    hours = (rec - midnight).astype("timedelta64[s]").astype(float) / 3600.0
    zeta = am.zeta.values[:, 0]
    Ruu, Rwu = am.R_uL_uS.values[:, 0], am.R_wL_uS.values[:, 0]
    RuT, RwT = am.R_uL_tsS.values[:, 0], am.R_wL_tsS.values[:, 0]
    Rw_wT = am.R_wL_wTsS.values[:, 0]
    lam_c = am.cutoff_lambda.values[:, 0]
    src = am.cutoff_source.values[:, 0]
    n_gap = int((src == 0).sum())
    print(f"cutoff source: spectral_gap {n_gap}, delta_fallback {(src == 1).sum()} of {len(rec)}")
    print(f"lambda_c [m]: median {np.nanmedian(lam_c):.0f}, "
          f"IQR {np.nanpercentile(lam_c, 25):.0f}-{np.nanpercentile(lam_c, 75):.0f}")
    for name, v in (("R_uL_uS", Ruu), ("R_wL_uS", Rwu), ("R_uL_tsS", RuT),
                    ("R_wL_TsS", RwT), ("R_wL_wTsS", Rw_wT)):
        print(f"{name}: median {np.nanmedian(v):+.2f}, IQR "
              f"{np.nanpercentile(v, 25):+.2f} to {np.nanpercentile(v, 75):+.2f}")

    vf = sc.var_frac_u.values[:, 0, :]                # (record, band)
    ff = sc.flux_frac_wTs.values[:, 0, :]
    print("u' variance fractions (median): "
          + ", ".join(f"{b} {np.nanmedian(vf[:, k]):.2f}" for k, b in enumerate(sc.scale_band.values)))
    print("w'Ts' flux fractions (median): "
          + ", ".join(f"{b} {np.nanmedian(ff[:, k]):.2f}" for k, b in enumerate(sc.scale_band.values)))

    med = sweep(hf, cfg, SWEEP_LAMBDA)
    print("cutoff sweep, median R_uL_uS:",
          ", ".join(f"{int(l)} m {r:+.2f}" for l, r in zip(SWEEP_LAMBDA, med["u_pf"])))

    fig, axs = plt.subplots(3, 2, figsize=(12.5, 12), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(hours, Ruu, color=C_U, lw=1.2, label="$R_{u_L,u_S}$")
    ax.plot(hours, Rwu, color=C_W, lw=1.2, label="$R_{w_L,u_S}$")
    ax.plot(hours, Rw_wT, color=C_REF, lw=1.0, label="$R_{w_L,(wTs)_S}$")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("hours since 2023-07-06 00:00")
    ax.set_ylabel("R")
    ax.set_title("(a) single-point AM coefficients, z = 10.85 m")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    m = np.isfinite(zeta)
    ax.scatter(zeta[m], Ruu[m], s=12, color=C_U, label="$R_{u_L,u_S}$")
    ax.scatter(zeta[m], Rwu[m], s=12, color=C_W, label="$R_{w_L,u_S}$")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.axvline(0, color=C_INK, lw=0.6)
    ax.set_xlabel("z/L")
    ax.set_ylabel("R")
    ax.set_title("(b) modulation vs stability (Salesky & Anderson 2018)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(SWEEP_LAMBDA, med["u_pf"], "o-", color=C_U, label="$R_{u_L,u_S}$")
    ax.plot(SWEEP_LAMBDA, med["w_pf"], "s-", color=C_W, label="$R_{w_L,u_S}$")
    ax.set_xscale("log")
    ax.axhline(0, color=C_INK, lw=0.6)
    ax.set_xlabel(r"cutoff $\lambda_c$ [m]")
    ax.set_ylabel("median R over 96 records")
    ax.set_title("(c) cutoff sensitivity (deviation-register validation)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    fs = hf.fs
    n_win = hf.n_per_window
    pmed, Umed = [], []
    pc = cfg.preprocess
    for win in ecio.iter_windows(hf, variables=("u_pf", "v_pf", "w_pf")):
        prep = pp.prepare(win, 0, ["u_pf"], method=pc.detrend, tau_s=pc.filter_tau_s)
        x = prep.prime["u_pf"]
        if np.isfinite(x).all():
            spec = sp.spectrum(x, fs)
            pmed.append(spec.S)
            Umed.append(prep.U_mean)
    spec_f = np.fft.rfftfreq(n_win, d=1.0 / fs)
    Smed = np.nanmedian(np.array(pmed), axis=0)
    edges = sp.log_bins(spec_f, 10)
    pos = spec_f > 0
    fb, Sb, cnt = sp.log_bin(spec_f[pos], np.real(Smed[pos]), edges)
    ok = cnt > 0
    ax.loglog(fb[ok], fb[ok] * Sb[ok], color=C_U, lw=1.4)
    U0 = float(np.nanmedian(Umed))
    for lam, ls, lab in ((np.nanmedian(lam_c), "-", "median cutoff"),
                         (cfg.ampmod.delta_m, "--", r"$\delta$ = 1000 m")):
        ax.axvline(U0 / lam, color=C_REF, ls=ls, lw=1.0, label=lab)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel(r"$f\,S_u$ [m$^2$ s$^{-2}$]")
    ax.set_title("(d) record-median premultiplied u' spectrum")
    ax.legend(fontsize=8)

    for col, (frac, ttl) in enumerate(((vf, "(e) u' variance fractions"),
                                       (ff, "(f) w'Ts' flux fractions"))):
        ax = axs[2, col]
        for k, b in enumerate(sc.scale_band.values):
            ax.plot(hours, frac[:, k], color=BAND_C[str(b)], lw=1.2, label=str(b))
        ax.axhline(0, color=C_INK, lw=0.6)
        ax.set_xlabel("hours since 2023-07-06 00:00")
        ax.set_ylabel("fraction")
        if col == 1:
            ax.set_ylim(-0.6, 1.6)          # fractions blow up where the w'Ts' denominator ~ 0
        ax.set_title(ttl)
        ax.legend(fontsize=8)

    fig.suptitle(os.path.basename(hf_path), fontsize=10)
    fig_path = os.path.join(ROOT, "testbed", "scratch", "ec_ampmod_scales_vac001.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print("figure:", fig_path)
    hf.close()


if __name__ == "__main__":
    main(sys.argv[1:])
