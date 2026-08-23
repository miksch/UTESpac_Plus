"""ec_coherent spectra on VAC001: closure check and Kaimal (1972) neutral overlay.

Runs the ``spectra`` module over one VAC001 HF file (default GPF ConstDet
2023-07-06, all 96 records), writes the analysis netCDF next to it, and
draws the validation figure: (a) closure -- band-summed spectra and ogive(0)
against the window variances/covariances; (b)-(d) pre-multiplied u, w
spectra and -uw cospectrum of the near-neutral records, normalised by u*^2
and plotted against f = nz/U with the Kaimal et al. (1972) eq. 21 curves
(library/writeups/ec_spectra.md) overlaid for 0.01 < f < 4; (e) normalised
ogives of u'w' and w'Ts' for the same records.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_spectra_vac001.py [HF_FILE] [--no-run]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ec_coherent import ECConfig, io as ecio, spectra as spx   # noqa: E402
from ec_coherent.cli import run_file                           # noqa: E402

DEFAULT = os.path.join(ROOT, "data", "VAC001", "output", "VAC001_hf_GPF_ConstDet_2023_07_06.nc")
C_DATA, C_REF, C_INK = "#2a78d6", "#eb6834", "#52514e"
ZL_NEUTRAL = 0.1      # |z/L| below this counts as near-neutral for the overlay (figure choice)


def main(argv):
    hf_path = next((a for a in argv if not a.startswith("--")), DEFAULT)
    cfg = ECConfig.from_config(modules=("spectra",))
    out = ecio.output_path(hf_path, cfg.output_suffix)
    if "--no-run" not in argv or not os.path.exists(out):
        out = run_file(hf_path, cfg)
    ds = ecio.read_group(out, "spectra")
    hf = ecio.open_hf(hf_path)
    z = float(hf.heights[0])
    ustar = np.asarray(hf.ds["ustar"].values[:, 0], dtype=float)
    L = np.asarray(hf.ds["L"].values[:, 0], dtype=float)
    hf.close()

    f = ds.frequency.values
    edges = ds.frequency_edges.values
    width = np.diff(edges)
    U = ds.U_mean.values[:, 0]
    rec_ok = np.isfinite(U) & np.isfinite(ds.var_u.values[:, 0])
    neutral = rec_ok & np.isfinite(L) & (np.abs(z / L) < ZL_NEUTRAL) & np.isfinite(ustar) & (ustar > 0.1)
    print(f"{os.path.basename(out)}: {rec_ok.sum()} records processed, {neutral.sum()} near-neutral "
          f"(|z/L| < {ZL_NEUTRAL}, u* > 0.1)")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax = axes[0, 0]
    for k, c in (("u", C_DATA), ("w", "#1baf7a"), ("Ts", "#9b59b6")):
        S = ds[f"S_{k}"].values[:, 0, :]
        var = ds[f"var_{k}"].values[:, 0]
        band = np.nansum(S * width, axis=1)
        m = rec_ok & (var > 0)
        ax.loglog(var[m], band[m], ".", color=c, ms=5, label=f"{k}: band-sum vs var")
    cov = ds.cov_uw.values[:, 0]
    og0 = ds.ogive_uw.values[:, 0, 0]
    m = rec_ok & (cov < 0)
    ax.loglog(-cov[m], -og0[m], "x", color=C_REF, ms=5, label="-u'w': ogive(f_min) vs -cov")
    lim = ax.get_xlim()
    ax.plot(lim, lim, "-", color=C_INK, lw=0.8)
    ax.set_xlabel("window variance / -covariance")
    ax.set_ylabel("spectral estimate")
    ax.set_title("(a) closure, all records", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    def overlay(ax, key, curve, sign, label):
        if neutral.sum() == 0:
            ax.text(0.5, 0.5, "no near-neutral records", transform=ax.transAxes, ha="center")
            return
        fn_all, y_all = [], []
        for i in np.flatnonzero(neutral):
            S = ds[key].values[i, 0, :]
            fn = f * z / U[i]
            y = sign * fn * S * U[i] / z / ustar[i] ** 2      # n S(n)/u*^2 with n = f [Hz]
            ok = np.isfinite(y) & (y > 0)
            ax.loglog(fn[ok], y[ok], "-", color=C_DATA, alpha=0.25, lw=0.8)
            fn_all.append(fn[ok]); y_all.append(y[ok])
        fn_all = np.concatenate(fn_all); y_all = np.concatenate(y_all)
        b = np.logspace(-3, 1.5, 30)
        idx = np.digitize(fn_all, b)
        med = [np.median(y_all[idx == j]) if (idx == j).sum() >= 3 else np.nan for j in range(1, len(b))]
        ax.loglog(np.sqrt(b[:-1] * b[1:]), med, "-", color=C_DATA, lw=2, label="median of records")
        fk = np.logspace(-2, np.log10(4), 100)
        ax.loglog(fk, spx.kaimal_neutral(fk, curve), "--", color=C_REF, lw=2,
                  label="Kaimal 1972 eq. 21, z/L = 0")
        ax.set_xlim(1e-3, 30); ax.set_ylim(1e-3, 3)
        ax.set_xlabel("f = nz/U"); ax.set_ylabel(label)
        ax.legend(fontsize=7, frameon=False, loc="lower left")

    overlay(axes[0, 1], "S_u", "u", 1.0, "n S_u / u*²"); axes[0, 1].set_title("(b) u spectrum, near-neutral", fontsize=10)
    overlay(axes[0, 2], "S_w", "w", 1.0, "n S_w / u*²"); axes[0, 2].set_title("(c) w spectrum, near-neutral", fontsize=10)
    overlay(axes[1, 0], "Co_uw", "uw", -1.0, "-n Co_uw / u*²"); axes[1, 0].set_title("(d) uw cospectrum, near-neutral", fontsize=10)

    ax = axes[1, 1]
    for i in np.flatnonzero(neutral):
        og = ds.ogive_uw.values[i, 0, :] / ds.cov_uw.values[i, 0]
        ax.semilogx(f, og, "-", color=C_DATA, alpha=0.3, lw=0.8)
        og2 = ds.ogive_wTs.values[i, 0, :] / ds.cov_wTs.values[i, 0]
        ax.semilogx(f, og2, "-", color="#9b59b6", alpha=0.3, lw=0.8)
    ax.axhline(1.0, color=C_INK, lw=0.8)
    ax.plot([], [], color=C_DATA, label="u'w'"); ax.plot([], [], color="#9b59b6", label="w'Ts'")
    ax.set_xlabel("f [Hz]"); ax.set_ylabel("ogive / covariance")
    ax.set_ylim(-0.5, 1.6)
    ax.set_title("(e) normalised ogives, near-neutral", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 2]
    ax.plot(ds.record.values, ds.taylor_ratio.values[:, 0], ".", color=C_DATA, label="σ_M / U")
    ax.axhline(0.5, color=C_REF, lw=1, label="Stull eq. 1.4d limit")
    ax2 = ax.twinx()
    ax2.plot(ds.record.values, ds.w_mean_over_sigma_w.values[:, 0], "x", color="#1baf7a", ms=4,
             label="w̄ / σ_w (residual after PF+yaw)")
    ax.set_ylabel("σ_M / U"); ax2.set_ylabel("w̄ / σ_w")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.set_title("(f) Taylor validity and rotation residual", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")

    fig.suptitle(f"ec_coherent spectra -- {os.path.basename(hf_path)} (z = {z} m, "
                 f"detrend {ds.attrs['detrend_method']}, taper {ds.attrs['taper']})", fontsize=10, color=C_INK)
    fig.tight_layout()
    fig_out = os.path.join(ROOT, "testbed", "scratch", "ec_spectra_vac001.png")
    os.makedirs(os.path.dirname(fig_out), exist_ok=True)
    fig.savefig(fig_out, dpi=140, bbox_inches="tight")
    print(f"figure written: {os.path.relpath(fig_out, ROOT)}")

    # numbers for the record
    for k in ("u", "w", "Ts"):
        S = ds[f"S_{k}"].values[:, 0, :]; var = ds[f"var_{k}"].values[:, 0]
        r = np.nansum(S * width, axis=1)[rec_ok] / var[rec_ok]
        print(f"  band-sum/var {k}: median {np.nanmedian(r):.3f}, range {np.nanmin(r):.3f}-{np.nanmax(r):.3f}")
    r = og0[rec_ok] / cov[rec_ok]
    print(f"  ogive(f_min)/cov uw: median {np.nanmedian(r):.4f}, range {np.nanmin(r):.4f}-{np.nanmax(r):.4f}")
    tr = ds.taylor_ratio.values[:, 0][rec_ok]
    print(f"  taylor ratio sigma_M/U: median {np.nanmedian(tr):.2f}, {np.mean(tr < 0.5) * 100:.0f}% below 0.5")
    wr = ds.w_mean_over_sigma_w.values[:, 0][rec_ok]
    print(f"  |w_mean|/sigma_w: median {np.nanmedian(np.abs(wr)):.3f}, max {np.nanmax(np.abs(wr)):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
