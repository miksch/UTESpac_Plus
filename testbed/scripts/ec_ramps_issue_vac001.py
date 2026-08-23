"""The two open issues of the wavelet ramp detector, on real VAC001 data, one figure.

(1) Timing: the MHAT zero-crossing at a0 lags the visible microfront; the
RAMP-extremum re-timing (refine="ramp", landed but default-off) puts it on
the front. Shown on a window where Ts detection works (panel d) and as the
pooled per-event lag distribution of the whole day (panel c).

(2) Visibility vs detectability: Ts ramps are visible in the trace of the
strongest-heat-flux window, but its Ts scalogram has no ramp-scale peak, so
the Ts detector returns almost nothing -- while the u detector's events land
on the visible Ts ramps (panels a, b).

Usage (repo root, UTESpac_Plus env)::

    python testbed/scripts/ec_ramps_issue_vac001.py [HF_FILE]
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

DEFAULT = os.path.join(ROOT, "data", "VAC001", "output", "VAC001_hf_GPF_ConstDet_2023_07_06.nc")
C_T, C_U, C_REF, C_INK, C_OK = "#9b59b6", "#2a78d6", "#eb6834", "#52514e", "#1baf7a"


def detect_for(prep, name, fs, cfg, refine="none"):
    x = prep.prime[name]
    wT = float(np.nanmean(prep.prime["w"] * prep.prime["Ts"]))
    slope = ramps._slope_for(name, getattr(cfg.ramps, f"slope_{name}", "auto"), wT)
    sc = ramps.log_scales(cfg.ramps.a_min_s, cfg.ramps.a_max_s, cfg.ramps.n_scales_per_decade)
    return ramps.detect(x, fs, sc, slope=slope, peak=cfg.ramps.peak,
                        edge_scales=cfg.ramps.edge_scales, refine=refine,
                        D_min_s=cfg.ramps.D_min_s), slope


def main(argv):
    hf_path = argv[0] if argv else DEFAULT
    cfg = ECConfig.from_config()
    hf = ecio.open_hf(hf_path)
    fs = hf.fs
    need = ("u", "v", "w", "Ts")

    # ---- pooled per-event lag (zero-crossing minus refined), whole day --------------
    lags = {"Ts": [], "u": []}
    preps = {}
    for win in ecio.iter_windows(hf, variables=need):
        prep = pp.prepare(win, 0, need, method=cfg.preprocess.detrend)
        preps[win.index] = prep
        for name in ("Ts", "u"):
            if not prep.accepted.get(name, False) or not np.isfinite(prep.prime[name]).all():
                continue
            ev, _ = detect_for(prep, name, fs, cfg)
            if ev.n == 0 or not np.isfinite(ev.a0):
                continue
            ref = ramps.refine_times(prep.prime[name], fs, ev.times, ev.a0, "ramp")
            lags[name].append((ev.times - ref) / ev.a0)
    lag_T = np.concatenate(lags["Ts"]) if lags["Ts"] else np.array([])
    lag_u = np.concatenate(lags["u"]) if lags["u"] else np.array([])
    print(f"pooled events: Ts {lag_T.size} (lag median {np.median(lag_T):+.2f} a0), "
          f"u {lag_u.size} (lag median {np.median(lag_u):+.2f} a0)")

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.1], hspace=0.42, wspace=0.22)

    # ---- (a) strongest-heat-flux window: Ts ramps visible, Ts detector fails --------
    rec_a = 21                                # 2023-07-06 11:00, max w'Ts' of the file
    prep_a = preps[rec_a]
    xT, xu = prep_a.prime["Ts"], prep_a.prime["u"]
    t = np.arange(xT.size) / fs
    evT_a, slope_a = detect_for(prep_a, "Ts", fs, cfg)
    evu_a, _ = detect_for(prep_a, "u", fs, cfg, refine="none")
    seg = (t >= 480) & (t < 960)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t[seg], xT[seg], "-", color=C_T, lw=0.8)
    for ti in evT_a.times[(evT_a.times >= 480) & (evT_a.times < 960)]:
        ax.axvline(ti, color=C_REF, lw=2.0, alpha=0.9)
    for ti in evu_a.times[(evu_a.times >= 480) & (evu_a.times < 960)]:
        ax.axvline(ti, color=C_U, lw=0.9, alpha=0.55, ls="--")
    ax.plot([], [], color=C_REF, lw=2, label=f"Ts detections ({evT_a.n} in 30 min, D = {evT_a.D:.0f} s)")
    ax.plot([], [], color=C_U, ls="--", label=f"u detections ({evu_a.n} in 30 min, D = {evu_a.D:.0f} s)")
    ax.set_xlim(480, 960)
    ax.set_xlabel("t [s] from window start")
    ax.set_ylabel("Ts' [K]")
    ax.set_title(f"(a) issue 2 -- record {rec_a} ({str(hf.records[rec_a])[:16]}, the day's largest w'Ts'): "
                 "Ts ramps VISIBLE, Ts detector nearly blind; u events land on them", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)

    # ---- (b) why: the Ts scalogram of that window has no ramp-scale peak ------------
    ax = fig.add_subplot(gs[1, 0])
    WT, Wu = evT_a.W, evu_a.W
    scal = evT_a.scales_s
    ax.semilogx(scal, WT / np.nanmax(WT), "-", color=C_T, lw=1.5, label="W_1(a) Ts'")
    ax.semilogx(scal, Wu / np.nanmax(Wu), "-", color=C_U, lw=1.5, label="W_1(a) u'")
    a_cut = cfg.ramps.D_min_s / ramps.D_G[cfg.ramps.wavelet]
    ax.axvspan(scal[0], a_cut, color=C_INK, alpha=0.10)
    ax.text(np.sqrt(scal[0] * a_cut), 0.05, "excluded\n(D < 6.2 s)", ha="center", fontsize=7, color=C_INK)
    ax.axvline(evT_a.a0, color=C_T, lw=1.2, ls=":", label=f"a0(Ts) = {evT_a.a0:.0f} s -> trend hump")
    ax.axvline(evu_a.a0, color=C_U, lw=1.2, ls=":", label=f"a0(u) = {evu_a.a0:.1f} s -> ramp scale")
    ax.set_xlabel("dilation a [s]")
    ax.set_ylabel("W_1(a) / max")
    ax.set_title(f"(b) record {rec_a} scalograms: no interior Ts peak between\n"
                 "the small-scale shoulder and the trend hump", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    # ---- (c) issue 1 pooled: zero-crossing lag behind the RAMP-refined time ---------
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(-1, 1, 41)
    inside_u = lag_u[np.abs(lag_u) < 1]
    inside_T = lag_T[np.abs(lag_T) < 1]
    ax.hist(inside_u, bins=bins, color=C_U, alpha=0.6,
            label=f"u events (n = {lag_u.size}, {100 * (1 - inside_u.size / lag_u.size):.0f}% at the +-a0 bound)")
    if lag_T.size:
        ax.hist(inside_T, bins=bins, color=C_T, alpha=0.6,
                label=f"Ts events (n = {lag_T.size}, {100 * (1 - inside_T.size / lag_T.size):.0f}% at the bound)")
    ax.axvline(0.35, color=C_REF, lw=1.5, label="+0.35 a0: the ideal-train lag (absent here)")
    ax.axvline(0.0, color=C_INK, lw=0.8)
    med = np.median(lag_u)
    ax.axvline(med, color=C_U, lw=1.5, ls="--", label=f"median (u) = {med:+.2f} a0")
    near = 100 * np.mean(np.abs(lag_u) < 0.1)
    ax.text(0.02, 0.45, f"only {near:.0f}% of u events agree\nwithin 0.1 a0: on real data\nre-timing adds +-a0 scatter,\nnot a systematic bias fix",
            transform=ax.transAxes, fontsize=8, color=C_INK)
    ax.set_xlabel("(zero-crossing time - RAMP-refined time) / a0")
    ax.set_ylabel("events")
    ax.set_title("(c) issue 1 on real data -- all events of the day: no systematic\n"
                 "+0.35 a0 lag; the two timings scatter within +-a0", fontsize=10)
    ax.legend(fontsize=7, frameon=False)

    # ---- (d) issue 1 on one trace: a window where Ts detection works ----------------
    rec_d = 18                                # 09:30, D_Ts = 9.2 s, 99 events
    prep_d = preps[rec_d]
    xd = prep_d.prime["Ts"]
    evd, slope_d = detect_for(prep_d, "Ts", fs, cfg)
    refd = ramps.refine_times(xd, fs, evd.times, evd.a0, "ramp")
    lag_s = float(np.median(evd.times - refd))
    ax = fig.add_subplot(gs[2, :])
    m = (t >= 600) & (t < 780)
    ax.plot(t[m], xd[m], "-", color=C_T, lw=0.9)
    sel = (evd.times >= 600) & (evd.times < 780)
    for ti, tr in zip(evd.times[sel], refd[sel]):
        ax.axvline(ti, color=C_REF, lw=1.6, alpha=0.9)
        ax.axvline(tr, color=C_INK, lw=1.2, ls=":", alpha=0.9)
    ax.plot([], [], color=C_REF, lw=1.6, label="MHAT zero-crossing (landed default)")
    ax.plot([], [], color=C_INK, ls=":", label='RAMP-refined (refine="ramp", option)')
    ax.set_xlim(600, 780)
    ax.set_xlabel("t [s] from window start")
    ax.set_ylabel("Ts' [K]")
    ax.set_title(f"(d) issue 1 -- record {rec_d} ({str(hf.records[rec_d])[:16]}), Ts detection works "
                 f"(a0 = {evd.a0:.1f} s, D = {evd.D:.1f} s, {evd.n} events): zero-crossing lands "
                 f"{lag_s:+.1f} s (median) after the refined time at each sharp drop", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)

    fig.suptitle(f"ec_coherent ramps -- the two DECIDE issues on {os.path.basename(hf_path)} "
                 f"(z = {float(hf.heights[0])} m, detrend {cfg.preprocess.detrend}, D_min = {cfg.ramps.D_min_s} s)",
                 fontsize=11, color=C_INK)
    out = os.path.join(ROOT, "testbed", "scratch", "ec_ramps_issues_vac001.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure written: {os.path.relpath(out, ROOT)}")
    hf.close()


if __name__ == "__main__":
    main(sys.argv[1:])
