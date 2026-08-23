"""Interaction boundary for the global planar fit.

``find_global_pf`` asks a *prompter* for every decision that MATLAB
``findGlobalPF`` took from the keyboard: whether to skip a height, the
direction-bin boundaries, the date barriers, use-all vs day-by-day, which
days to accept, and the final confirmation. Two implementations ship:

* :class:`ScriptedPFSelection` — non-interactive; the answers are given up
  front (defaults: process every height, one 0–360 sector, no date
  barriers, use all data, accept). This is what ``run_utespac`` uses when
  no prompter is passed, and what scripts and tests use.
* :class:`ConsolePFPrompter` — the interactive session with the overview
  and day-by-day figures, used by the ``utespac_main`` CLI.

The core never calls ``input()`` or draws; only the console prompter does.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol

import numpy as np

log = logging.getLogger("utespac")


@dataclass
class PFHeightContext:
    """What the prompter may show for one sonic height (stepped-down series)."""
    z: float
    site_name: str
    direction: np.ndarray
    t_dn: np.ndarray
    u_dn: np.ndarray
    v_dn: np.ndarray
    w_dn: np.ndarray
    spd_dn: np.ndarray


class PFPrompter(Protocol):
    """Decisions ``find_global_pf`` needs, in call order per height."""

    def skip_height(self, z: float) -> bool: ...
    def begin_height(self, ctx: PFHeightContext) -> None: ...
    def bin_boundaries(self, z: float) -> List[float]: ...
    def date_barriers(self, z: float, t_valid: np.ndarray) -> List[float]: ...
    def end_selection(self, z: float) -> None: ...
    def use_all_dates(self, z: float, d_start: float, d_end: float) -> bool: ...
    def begin_day_by_day(self, z: float, bins: List[float]) -> None: ...
    def show_day(self, z: float, k: int, n_days: int, day_coef: Dict, bins: List[float]) -> None: ...
    def use_day(self, z: float, day: float) -> bool: ...
    def show_cumulative(self, z: float, cum_coef: Dict, bins: List[float], n_days: int) -> None: ...
    def end_day_by_day(self, z: float) -> None: ...
    def confirm(self, pf_info: Dict) -> None: ...


@dataclass
class ScriptedPFSelection:
    """Non-interactive planar-fit selection.

    ``bins`` / ``barriers`` map a height (m) to its boundaries; heights not
    listed get ``default_bins`` / ``default_barriers``. ``skip`` lists
    heights to take from the existing ``PFinfo.pkl``. ``use_all`` switches
    the use-all path; ``accept_days`` (height → set of MATLAB day numbers)
    drives the day-by-day path when ``use_all`` is False — ``None`` accepts
    every day.
    """
    bins: Dict[float, List[float]] = field(default_factory=dict)
    barriers: Dict[float, List[float]] = field(default_factory=dict)
    default_bins: List[float] = field(default_factory=list)
    default_barriers: List[float] = field(default_factory=list)
    skip: Iterable[float] = field(default_factory=tuple)
    use_all: bool = True
    accept_days: Optional[Dict[float, Iterable[float]]] = None

    def _key(self, z, table):
        for k in table:
            if abs(float(k) - float(z)) < 0.01:
                return k
        return None

    def skip_height(self, z):
        return any(abs(float(s) - float(z)) < 0.01 for s in self.skip)

    def begin_height(self, ctx):
        pass

    def bin_boundaries(self, z):
        k = self._key(z, self.bins)
        return sorted(set(self.bins[k])) if k is not None else sorted(set(self.default_bins))

    def date_barriers(self, z, t_valid):
        k = self._key(z, self.barriers)
        vals = self.barriers[k] if k is not None else self.default_barriers
        return sorted({v for v in vals if t_valid[0] <= v <= t_valid[-1]})

    def end_selection(self, z):
        pass

    def use_all_dates(self, z, d_start, d_end):
        return self.use_all

    def begin_day_by_day(self, z, bins):
        pass

    def show_day(self, z, k, n_days, day_coef, bins):
        pass

    def use_day(self, z, day):
        if self.accept_days is None:
            return True
        k = self._key(z, self.accept_days)
        if k is None:
            return True
        return any(abs(float(d) - float(day)) < 1e-6 for d in self.accept_days[k])

    def show_cumulative(self, z, cum_coef, bins, n_days):
        pass

    def end_day_by_day(self, z):
        pass

    def confirm(self, pf_info):
        return None


def _matlab_to_datetime(serial):
    from .campbell_date import matlab_datenum_to_datetime
    return matlab_datenum_to_datetime(serial)


def _bin_key(bins, m):
    from .find_global_pf import _bin_key as _bk
    return _bk(bins, m)


class ConsolePFPrompter:
    """Interactive planar-fit selection with the overview and day-by-day figures."""

    def __init__(self, show_figures: bool = True):
        self.show_figures = show_figures
        self._fig = self._ax1 = self._ax2 = None
        self._dbd_fig = self._dbd_axes = None
        self._pitch_cum: List[List[float]] = []
        self._roll_cum: List[List[float]] = []

    # ── helpers ────────────────────────────────────────────────────────────
    def _plt(self):
        import matplotlib
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        return plt

    # ── protocol ───────────────────────────────────────────────────────────
    def skip_height(self, z):
        return input(f"\nSkip {z} m? (1=process, 0=skip): ").strip() == "0"

    def begin_height(self, ctx):
        self._fig = self._ax1 = self._ax2 = None
        if not self.show_figures:
            return
        try:
            plt = self._plt()
            import matplotlib.dates as mdates
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            ax1.plot(ctx.direction, ctx.w_dn, ".", markersize=2)
            ax1.set_xlabel("Wind direction (deg)")
            ax1.set_ylabel("w (m/s)")
            ax1.set_title(f"{ctx.site_name} {ctx.z} m: w vs direction (choose bin boundaries)")
            ax1.set_xlim(0, 360)
            dt = [_matlab_to_datetime(t) for t in ctx.t_dn if np.isfinite(t)]
            spd = ctx.spd_dn[np.isfinite(ctx.t_dn)]
            ax2.plot(dt, spd, ".", markersize=2)
            ax2.set_ylabel("speed (m/s)")
            ax2.set_title("speed vs time (choose date barriers)")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            self._fig, self._ax1, self._ax2 = fig, ax1, ax2
        except Exception as exc:
            log.warning("Could not show the planar-fit figure: %s", exc)

    def bin_boundaries(self, z):
        print(f"\nSelect direction bin boundaries for {z} m.")
        print("  Enter one value per prompt; press Enter with no value when done.")
        bins: List[float] = []
        y_lim = self._ax1.get_ylim() if self._ax1 is not None else (0, 1)
        while True:
            val = input("  Bin boundary (° 0–360, or Enter to finish): ").strip()
            if not val:
                break
            try:
                b = float(val)
            except ValueError:
                continue
            if 0 <= b <= 360:
                bins.append(b)
                if self._ax1 is not None:
                    plt = self._plt()
                    self._ax1.plot([b, b], [y_lim[0], y_lim[1]], "g--", linewidth=2)
                    plt.draw(); plt.pause(0.05)
        return sorted(set(bins))

    def date_barriers(self, z, t_valid):
        all_days = np.unique(np.floor(t_valid))
        print(f"\nData spans {len(all_days)} day(s): "
              f"{_matlab_to_datetime(all_days[0]).strftime('%Y-%m-%d')} to "
              f"{_matlab_to_datetime(all_days[-1]).strftime('%Y-%m-%d')}")
        print("  Optionally split data by date for different PF calculations.")
        print("  Enter MATLAB serial dates; press Enter to skip.")
        barriers: List[float] = []
        ylim = self._ax2.get_ylim() if self._ax2 is not None else (0, 1)
        while True:
            val = input("  Date barrier (MATLAB serial date, or Enter to finish): ").strip()
            if not val:
                break
            try:
                db = float(val)
            except ValueError:
                continue
            if t_valid[0] <= db <= t_valid[-1]:
                barriers.append(db)
                if self._ax2 is not None:
                    plt = self._plt()
                    dt = _matlab_to_datetime(db)
                    self._ax2.plot([dt, dt], [ylim[0], ylim[1]], "g--", linewidth=2)
                    plt.draw(); plt.pause(0.05)
        return sorted(set(barriers))

    def end_selection(self, z):
        if self._fig is not None:
            self._plt().close(self._fig)
            self._fig = self._ax1 = self._ax2 = None

    def use_all_dates(self, z, d_start, d_end):
        return input("  Use all data in range (1) or select day by day (0)?: ").strip() != "0"

    def begin_day_by_day(self, z, bins):
        self._dbd_fig = self._dbd_axes = None
        num_bins = max(1, len(bins))
        self._pitch_cum = [[] for _ in range(num_bins)]
        self._roll_cum = [[] for _ in range(num_bins)]
        if not self.show_figures:
            return
        try:
            plt = self._plt()
            ncols = min(num_bins, 3)
            nrows = int(np.ceil(num_bins / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), squeeze=False)
            axes = axes.ravel()[:num_bins]
            for m, ax in enumerate(axes):
                ax.set_title(_bin_key(bins, m))
                ax.set_xlabel("day #")
                ax.set_ylabel("pitch (x) / roll (d) [deg]")
                ax.grid(True)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            self._dbd_fig, self._dbd_axes = fig, list(axes)
        except Exception as exc:
            log.warning("Could not show the day-by-day figure: %s", exc)

    def show_day(self, z, k, n_days, day_coef, bins):
        if self._dbd_axes is None:
            return
        plt = self._plt()
        for m, ax in enumerate(self._dbd_axes):
            bin_key = _bin_key(bins, m)
            if bin_key in day_coef:
                b1d, b2d = day_coef[bin_key][1], day_coef[bin_key][2]
                p_d = np.degrees(np.arcsin(-b1d / np.sqrt(1 + b1d ** 2)))
                r_d = np.degrees(np.arcsin(b2d / np.sqrt(1 + b2d ** 2)))
                ax.plot(k + 1, p_d, "bx", markersize=8)
                ax.plot(k + 1, r_d, "gd", markersize=8)
                ax.set_xlim(0, n_days + 1)
        plt.draw(); plt.pause(0.05)

    def use_day(self, z, day):
        return input(f"  Use {_matlab_to_datetime(day).strftime('%Y-%m-%d')} "
                     "for cumulative calculation (1=yes, 0=no)?: ").strip() == "1"

    def show_cumulative(self, z, cum_coef, bins, n_days):
        if self._dbd_axes is None:
            return
        plt = self._plt()
        for m, ax in enumerate(self._dbd_axes):
            bin_key = _bin_key(bins, m)
            if bin_key in cum_coef:
                b1c, b2c = cum_coef[bin_key][1], cum_coef[bin_key][2]
                self._pitch_cum[m].append(np.degrees(np.arcsin(-b1c / np.sqrt(1 + b1c ** 2))))
                self._roll_cum[m].append(np.degrees(np.arcsin(b2c / np.sqrt(1 + b2c ** 2))))
                xs = list(range(1, len(self._pitch_cum[m]) + 1))
                for ln in [l for l in ax.lines if getattr(l, "_cum_line", False)]:
                    ln.remove()
                lp, = ax.plot(xs, self._pitch_cum[m], "b.-",
                              label="Pitch" if len(xs) == 1 else "")
                lr, = ax.plot(xs, self._roll_cum[m], "g.-",
                              label="Roll" if len(xs) == 1 else "")
                lp._cum_line = lr._cum_line = True
                if len(xs) == 1:
                    ax.legend(fontsize=8)
                ax.set_xlim(0, n_days + 1)
        plt.draw(); plt.pause(0.05)

    def end_day_by_day(self, z):
        if self._dbd_fig is not None:
            self._plt().close(self._dbd_fig)
            self._dbd_fig = self._dbd_axes = None

    def confirm(self, pf_info):
        from .find_global_pf import format_pf_info
        print("\nVerify Global Planar Fit Coefficients\n")
        print(format_pf_info(pf_info))
        if input("Is this correct? (1=yes, 0=no): ").strip() != "1":
            raise RuntimeError("PF coefficients rejected by user. Check inputs and rerun.")
        if input("Okay to begin analysis? (1=yes, 0=no): ").strip() != "1":
            raise RuntimeError("Program stopped by user.")
