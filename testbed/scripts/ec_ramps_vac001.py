"""ec_coherent wavelet ramp detector on VAC001: scalograms, duration scales, detections.

Runs the ``ramps`` module over one VAC001 HF file (default GPF ConstDet
2023-07-06, all 96 records), writes the analysis netCDF next to it, and
draws: (a) the MHAT wavelet variance of Ts' for every record with the peak
scales marked; (b) the diurnal course of the duration scale D and the event
count for Ts and u; (c) D against the mean spacing; (d) one window's Ts'
trace with the detections (zero-crossing, and RAMP-refined for comparison);
(e) the detection function T_1(a0, b) of that window.

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_ramps_vac001.py [HF_FILE] [--record N] [--no-run]
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


def main(argv):
    hf_path = next((a for a in argv if not a.startswith("--") and not a.isdigit()), DEFAULT)
    rec_show = int(argv[argv.index("--record") + 1]) if "--record" in argv else 30
    cfg = ECConfig.from_config(modules=("ramps",))
    out = ecio.output_path(hf_path, cfg.output_suffix)
    if "--no-run" not in argv or not os.path.exists(out):
        out = run_file(hf_path, cfg)
    ds = ecio.read_group(out, "ramps")
    hf = ecio.open_hf(hf_path)

    rec = ds.record.values
    scales = ds.scale.values
    D_T, D_u = ds.D_Ts.values[:, 0], ds.D_u.values[:, 0]
    n_T, n_u = ds.n_events_Ts.values[:, 0], ds.n_events_u.values[:, 0]
    sp_T, sp_u = ds.mean_spacing_Ts.values[:, 0], ds.mean_spacing_u.values[:, 0]
    ok = np.isfinite(D_T)
    print(f"{os.path.basename(out)}: {ok.sum()} records; D_Ts median {np.nanmedian(D_T):.1f} s "
          f"(range {np.nanmin(D_T):.1f}-{np.nanmax(D_T):.1f}), D_u median {np.nanmedian(D_u):.1f} s; "
          f"events/30 min Ts median {np.nanmedian(n_T):.0f}, u {np.nanmedian(n_u):.0f}; "
          f"mean spacing Ts {np.nanmedian(sp_T):.0f} s, u {np.nanmedian(sp_u):.0f} s; "
          f"slope_Ts negative in {np.mean(ds.slope_Ts.values[:, 0] == -1) * 100:.0f}% of records")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes[0, 0]
    W = ds.W_Ts.values[:, 0, :]
    for i in range(len(rec)):
        if np.isfinite(W[i]).any():
            ax.semilogx(scales, W[i] / np.nanmax(W[i]), "-", color=C_T, alpha=0.15, lw=0.8)
    ax.semilogx(ds.a0_Ts.values[:, 0], np.ones(len(rec)), "|", color=C_REF, ms=12, label="a0 per record")
    ax.set_xlabel("dilation a [s]"); ax.set_ylabel("W_1(a) / max")
    ax.set_title("(a) MHAT wavelet variance of Ts', all records", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    ax.plot(rec, D_T, ".", color=C_T, label="D (Ts)")
    ax.plot(rec, D_u, ".", color=C_U, label="D (u)")
    ax.set_ylabel("duration scale D [s]")
    ax2 = ax.twinx()
    ax2.plot(rec, n_T, "x", color=C_T, ms=4, alpha=0.6, label="n events (Ts)")
    ax2.plot(rec, n_u, "x", color=C_U, ms=4, alpha=0.6, label="n events (u)")
    ax2.set_ylabel("events per 30 min")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(b) duration scale and event count", fontsize=10)

    ax = axes[0, 2]
    ax.plot(D_T, sp_T, ".", color=C_T, label="Ts")
    ax.plot(D_u, sp_u, ".", color=C_U, label="u")
    lim = [0, max(np.nanmax(sp_T), np.nanmax(sp_u)) * 1.05]
    ax.plot(lim, lim, "-", color=C_INK, lw=0.8, label="spacing = D")
    ax.plot(lim, [2 * v for v in lim], "--", color=C_INK, lw=0.8, label="spacing = 2D")
    ax.set_xlabel("D [s]"); ax.set_ylabel("mean spacing [s]")
    ax.set_title("(c) spacing vs duration (CB 1993 II: 29 s vs 12 s)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # one window in detail
    win = next(ecio.iter_windows(hf, variables=("u", "v", "w", "Ts"), records=[rec_show]))
    prep = pp.prepare(win, 0, ("u", "v", "w", "Ts"), method=cfg.preprocess.detrend)
    x = prep.prime["Ts"]
    t = np.arange(x.size) / hf.fs
    sc = ramps.log_scales(cfg.ramps.a_min_s, cfg.ramps.a_max_s, cfg.ramps.n_scales_per_decade)
    slope = ramps._slope_for("Ts", cfg.ramps.slope_Ts, float(np.nanmean(prep.prime["w"] * x)))
    ev = ramps.detect(x, hf.fs, sc, slope=slope, peak=cfg.ramps.peak, edge_scales=cfg.ramps.edge_scales,
                       D_min_s=cfg.ramps.D_min_s)
    evr = ramps.detect(x, hf.fs, sc, slope=slope, peak=cfg.ramps.peak, edge_scales=cfg.ramps.edge_scales,
                       refine="ramp", D_min_s=cfg.ramps.D_min_s)
    T0 = ramps.cwt(x, np.array([ev.a0]), hf.fs)[0]
    seg = (t >= 600) & (t < 900)
    ax = axes[1, 0]
    ax.plot(t[seg], x[seg], "-", color=C_T, lw=0.7)
    for ti in ev.times[(ev.times >= 600) & (ev.times < 900)]:
        ax.axvline(ti, color=C_REF, lw=0.8, alpha=0.8)
    for ti in evr.times[(evr.times >= 600) & (evr.times < 900)]:
        ax.axvline(ti, color=C_INK, lw=0.8, alpha=0.5, ls=":")
    ax.plot([], [], color=C_REF, label="MHAT zero-crossing (landed)")
    ax.plot([], [], color=C_INK, ls=":", label="RAMP-refined (option)")
    ax.set_xlabel("t [s] from window start"); ax.set_ylabel("Ts' [K]")
    ax.set_title(f"(d) record {rec_show} ({str(rec[rec_show])[:16]}), a0 = {ev.a0:.1f} s, D = {ev.D:.1f} s, "
                 f"slope {slope}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1, 1]
    ax.plot(t[seg], T0[seg], "-", color=C_INK, lw=0.8)
    ax.axhline(0, color=C_INK, lw=0.5)
    for ti in ev.times[(ev.times >= 600) & (ev.times < 900)]:
        ax.axvline(ti, color=C_REF, lw=0.8, alpha=0.8)
    ax.set_xlabel("t [s]"); ax.set_ylabel("T_1(a0, b)")
    ax.set_title("(e) detection function at a0, same segment", fontsize=10)

    ax = axes[1, 2]
    ax.semilogx(ev.scales_s, ev.W, "-", color=C_T, label="Ts'")
    ax.axvline(ev.a0, color=C_REF, lw=1, label=f"a0 = {ev.a0:.1f} s")
    xu = prep.prime["u"]
    evu = ramps.detect(xu, hf.fs, sc, slope="positive", peak=cfg.ramps.peak, D_min_s=cfg.ramps.D_min_s)
    ax.semilogx(evu.scales_s, evu.W / evu.W.max() * ev.W.max(), "-", color=C_U, label="u' (rescaled)")
    ax.axvline(evu.a0, color=C_U, lw=1, ls="--", label=f"a0(u) = {evu.a0:.1f} s")
    ax.set_xlabel("dilation a [s]"); ax.set_ylabel("W_1(a)")
    ax.set_title(f"(f) scalograms of record {rec_show}", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"ec_coherent ramps (wavelet) -- {os.path.basename(hf_path)} (z = {float(hf.heights[0])} m, "
                 f"detrend {ds.attrs['detrend_method']}, peak {ds.attrs['peak']}, refine {ds.attrs['refine']})",
                 fontsize=10, color=C_INK)
    fig.tight_layout()
    fig_out = os.path.join(ROOT, "testbed", "scratch", "ec_ramps_vac001.png")
    fig.savefig(fig_out, dpi=140, bbox_inches="tight")
    print(f"figure written: {os.path.relpath(fig_out, ROOT)}")
    hf.close()


if __name__ == "__main__":
    main(sys.argv[1:])
