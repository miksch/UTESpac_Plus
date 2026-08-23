"""QC of the raw high-frequency tables (Vickers & Mahrt 1997 style).

:func:`qc_table` is the per-table kernel (absolute limits, iterative
spike removal with consecutive-flag filtering, per-period spike/NaN
fractions); :func:`utespac.stages.condition` runs it over the run model.
"""

from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.interpolate import interp1d

from .averaging import block_average
from .consec_flag_removal import consec_flag_removal
from .strfndw import strfndw


def qc_table(tbl: np.ndarray, header_names: Sequence[str], info: Dict, template: Dict[str, str],
             scan_hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QC one table in place (column 0 is the datenum time axis).

    Returns ``(tbl, spike_flag, nan_flag)``: the conditioned table (limits
    applied, spikes interpolated) and the per-period boolean flags
    ``(n_periods, n_cols)`` — True where the period's spike (NaN) fraction
    reaches ``spikeTest.maxPercent`` (``nanTest.maxPercent``); column 0 is
    always False. Tables slower than 1 Hz skip the spike test.
    """
    spike_def_cfg = info["spikeTest"]["spikeDef"]
    abs_lim_cfg = info["absoluteLimitsTest"]
    n_rows, n_cols = tbl.shape
    is_slow_table = scan_hz < 1  # < 1 Hz — skip spike test, apply limits + NaN flag only

    # Build per-column spike definitions and absolute limits
    spike_def = np.full(n_cols, spike_def_cfg["otherInstrument"])
    min_max = np.full((2, n_cols), np.nan)
    spike_def[0] = np.nan  # timestamp column is never spiked

    for field, pattern in template.items():
        if field not in spike_def_cfg or field not in abs_lim_cfg:
            continue
        if field == "otherInstrument" or not pattern:
            continue
        matched_cols = strfndw(list(header_names), pattern)
        # Ensure Tson is in Celsius
        if field == "Tson":
            for c in matched_cols:
                med = np.nanmedian(tbl[:, c])
                if med > 250:
                    tbl[:, c] -= 273.15
        for c in matched_cols:
            spike_def[c] = spike_def_cfg[field]
            lims = abs_lim_cfg[field]
            min_max[0, c] = lims[0]
            min_max[1, c] = lims[1]

    # Also mark sonic diagnostic column as NaN for spike purposes
    if "sonDiagnostic" in template:
        for c in strfndw(list(header_names), template["sonDiagnostic"]):
            spike_def[c] = np.nan

    # 1. Absolute limits
    for c in range(n_cols):
        if not np.isnan(min_max[0, c]):
            bad = (tbl[:, c] < min_max[0, c]) | (tbl[:, c] > min_max[1, c])
            tbl[bad, c] = np.nan

    # 2. Spike removal (skipped for slow tables — too few samples per window)
    nan_flag = np.isnan(tbl)
    spike_flag = np.zeros((n_rows, n_cols), dtype=bool)

    if not is_slow_table:
        num_days = tbl[-1, 0] - tbl[0, 0]
        frac = info["spikeTest"]["windowSizeFraction"]
        num_bins = int(round(num_days / (info["avgPer"] * frac / 1440))) * 2 + 1
        bp = np.round(np.linspace(0, n_rows, num_bins)).astype(int)

        no_spike_flag = np.zeros(len(bp) - 1, dtype=bool)

        for _run in range(info["spikeTest"]["maxRuns"]):
            if no_spike_flag[:-1].all():
                break
            for k in range(len(bp) - 2):
                if no_spike_flag[k]:
                    continue
                r0, r1 = bp[k], bp[k + 2]
                local = tbl[r0:r1, :].copy()
                local_nan = nan_flag[r0:r1, :]

                stds = np.nanstd(local, axis=0)
                means = np.nanmean(local, axis=0)
                with np.errstate(invalid="ignore"):
                    normed = np.abs((local - means) / np.where(stds == 0, np.nan, stds))

                pot_spikes = np.zeros_like(local, dtype=bool)
                for c in range(n_cols):
                    if not np.isnan(spike_def[c]):
                        pot_spikes[:, c] = normed[:, c] >= spike_def[c]

                local_spikes = consec_flag_removal(
                    pot_spikes, info["spikeTest"]["maxConsecutiveOutliers"]
                )
                spike_flag[r0:r1, :] |= local_spikes

                if not local_spikes.any():
                    no_spike_flag[k] = True
                    continue

                # Linear interpolation over spikes
                for c in range(1, n_cols):
                    bad = local_spikes[:, c] | local_nan[:, c]
                    good = ~bad
                    if good.sum() < 2 or bad.sum() / len(bad) > 0.6:
                        local[:, c] = np.nan
                        continue
                    x_all = np.arange(r1 - r0)
                    x_good = x_all[good]
                    y_good = local[good, c]
                    x_bad = x_all[local_spikes[:, c]]
                    if len(x_bad):
                        f = interp1d(x_good, y_good, bounds_error=False, fill_value=np.nan)
                        local[local_spikes[:, c], c] = f(x_bad)
                tbl[r0:r1, :] = local

    # 3. Per-period flag fractions -> booleans
    t = tbl[:, 0]
    _, mean_spike = block_average(t, spike_flag[:, 1:].astype(float), info["avgPer"])
    mean_spike = np.column_stack([np.zeros(mean_spike.shape[0]), mean_spike])
    mean_spike[mean_spike < info["spikeTest"]["maxPercent"] / 100.0] = 0
    _, mean_nan = block_average(t, nan_flag[:, 1:].astype(float), info["avgPer"])
    mean_nan = np.column_stack([np.zeros(mean_nan.shape[0]), mean_nan])
    mean_nan[mean_nan < info["nanTest"]["maxPercent"] / 100.0] = 0
    return tbl, mean_spike.astype(bool), mean_nan.astype(bool)

