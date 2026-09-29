"""Generic IMU ingest for floating-platform sites.

Two ways an IMU reaches the UTESpac fast files, both ending as extra
``IMU_*`` columns in the sonic's 48-h table (the template keys in
``run.toml`` pick them up; units and axis signs are declared in the
site's ``[imu]`` siteInfo block, not here):

1. **Same logger as the sonic** (the default contract): the IMU channels
   are columns of the fast TOA5 table. Merge :data:`IMU_MAP` (renamed to
   the logger's source names) into the site script's ``columns`` dict,
   exactly like the ``FW_MAP`` pattern in the per-site fast-process scripts.

2. **Separate IMU logger**: read the IMU's own files with
   :func:`load_imu_files` (delimited text with a names row; column
   mapping, optional constant lag), then :func:`align_imu` resamples
   them onto the sonic's uniform grid so a site script can join the
   frames column-wise before writing. Sub-scan misalignment leaks
   platform motion into w' — prefer path 1 whenever the logger allows.

No unit conversion happens here: raw channels pass through verbatim and
``stages.motion`` converts using the site's ``[imu]`` declaration.

Record screening for a separate IMU logger (:func:`flag_spikes`,
:func:`blocks`, :func:`clean_imu`, :func:`hold_exit_jumps`,
:func:`frozen_attitude_mask`, :func:`substitute_accel_tilt`,
:func:`derive_mount_tilt`) works on the logger's own lower-case channel
names, as returned by
:func:`utespac.raw_processing.imu_datalog.parse_datalog`: a ``time``
column (datetime64) plus ``ax``, ``ay``, ``az`` (accelerometer, g),
``gx``, ``gy``, ``gz`` (angular rate, deg/s) and ``roll``, ``pitch``,
``yaw`` (fused attitude, deg).
"""

import glob

import numpy as np
import pandas as pd

from utespac.averaging import epoch_s

# Physical-plausibility limits of :func:`flag_spikes` for a moored floating
# platform, with the accelerometer in g and the gyro in deg/s.
SPIKE_GRAVITY_G = 1.0       # expected specific-force magnitude at rest [g]
SPIKE_ACCEL_TOL_G = 0.30    # allowed deviation of |a| from SPIKE_GRAVITY_G [g]
SPIKE_GYRO_MAX = 100        # allowed |gx|, |gy|, |gz| [deg/s]
SPIKE_ROLL_MAX = 45         # allowed |roll| [deg]
SPIKE_PITCH_MAX = 45        # allowed |pitch| [deg]
SPIKE_YAW_MAX = 180         # allowed |yaw| [deg]

# Records a contiguous block needs before :func:`hold_exit_jumps` uses it.
HOLD_MIN_BLOCK_RECORDS = 1000

# UTESpac IMU column names (run.toml template keys ``imuAx = "IMU_Ax*"``
# etc.) → typical source base names. Override values per logger; drop the
# entries a given IMU does not log (raw-only vs. fused-attitude units are
# declared in siteInfo [imu]).
IMU_MAP = {
    "IMU_Ax":    "Ax",       # accelerometer specific force
    "IMU_Ay":    "Ay",
    "IMU_Az":    "Az",
    "IMU_Gx":    "Gx",       # angular rate
    "IMU_Gy":    "Gy",
    "IMU_Gz":    "Gz",
    "IMU_Roll":  "Roll",     # fused attitude, when the IMU outputs one
    "IMU_Pitch": "Pitch",
    "IMU_Yaw":   "Yaw",
}


def imu_columns(source_map=None, src_suffix=""):
    """``columns`` entries for an IMU riding in the sonic's logger table.

    Parameters
    ----------
    source_map : dict, optional
        {UTESpac IMU name: source column base}; default :data:`IMU_MAP`.
        Keep only the channels the IMU actually logs.
    src_suffix : str, optional
        Suffix on the source names (replicated-instrument numbering).

    Returns
    -------
    dict
        ``{"IMU_Ax": f"Ax{src_suffix}", ...}`` to merge into a site
        script's ``process_table`` ``columns`` dict.
    """
    m = IMU_MAP if source_map is None else source_map
    return {out: f"{src}{src_suffix}" for out, src in m.items()}


def load_imu_files(pattern, columns, time_column=0, time_format=None,
                   read_kwargs=None, lag_s=0.0):
    """Load a separate IMU logger's delimited files into one DataFrame.

    Parameters
    ----------
    pattern : str
        Glob for the IMU files (delimited text with a names row; pass
        ``read_kwargs`` for other layouts).
    columns : dict
        {UTESpac IMU name: source column name}, e.g.
        ``imu_columns({"IMU_Roll": "roll_deg", ...})`` output. Missing
        source columns become NaN with a warning.
    time_column : str or int
        Timestamp column (name, or positional index into the names row).
    time_format : str, optional
        ``pd.to_datetime`` format (default: inferred).
    read_kwargs : dict, optional
        Extra ``pd.read_csv`` options (``sep``, ``skiprows``, ...).
    lag_s : float, optional
        Constant clock offset [s] added to the IMU timestamps to bring
        them onto the sonic's clock (positive = IMU clock runs early).

    Returns
    -------
    pandas.DataFrame
        UTESpac-named IMU channels on a sorted, de-duplicated
        DatetimeIndex.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No IMU files match {pattern}")
    frames = []
    for path in paths:
        df = pd.read_csv(path, **(read_kwargs or {}))
        tcol = df.columns[time_column] if isinstance(time_column, int) else time_column
        t = pd.to_datetime(df[tcol], format=time_format)
        if lag_s:
            t = t + pd.Timedelta(seconds=lag_s)
        out = pd.DataFrame(index=t)
        for name, src in columns.items():
            if src in df.columns:
                out[name] = pd.to_numeric(df[src], errors="coerce").values
            else:
                print(f"  Warning: IMU source column '{src}' missing in "
                      f"{path} — '{name}' filled with NaN.")
                out[name] = np.nan
        frames.append(out)
    full = pd.concat(frames)
    full = full[~full.index.duplicated(keep="first")].sort_index()
    return full


def align_imu(imu, grid, max_gap_s=None):
    """Align a separate-logger IMU frame onto the sonic's sample grid.

    Nearest-timestamp match within ``max_gap_s`` (default: one grid
    interval), NaN beyond it — interpolation across real gaps would
    smear platform motion into the correction.

    Parameters
    ----------
    imu : pandas.DataFrame
        Output of :func:`load_imu_files` (sorted DatetimeIndex).
    grid : pandas.DatetimeIndex
        The sonic table's uniform grid (``common.build_48h_index``).
    max_gap_s : float, optional
        Maximum |IMU stamp − grid stamp| to accept [s].

    Returns
    -------
    pandas.DataFrame
        IMU channels reindexed onto *grid*.
    """
    if max_gap_s is None:
        max_gap_s = (grid[1] - grid[0]) / pd.Timedelta(seconds=1)
    tol = pd.Timedelta(seconds=float(max_gap_s))
    idx = imu.index.get_indexer(grid, method="nearest", tolerance=tol)
    out = pd.DataFrame(index=grid, columns=imu.columns, dtype=float)
    ok = idx >= 0
    out.iloc[ok] = imu.iloc[idx[ok]].values
    return out


def flag_spikes(df, *, gravity=SPIKE_GRAVITY_G, accel_tol=SPIKE_ACCEL_TOL_G,
                gyro_max=SPIKE_GYRO_MAX, roll_max=SPIKE_ROLL_MAX,
                pitch_max=SPIKE_PITCH_MAX, yaw_max=SPIKE_YAW_MAX):
    """Boolean mask of records failing a physical-plausibility screen.

    Serial corruption shows up as isolated records whose gravity vector,
    rotation rate or attitude are impossible for a moored raft.

    Parameters
    ----------
    df : pandas.DataFrame
        IMU records with columns ``ax``, ``ay``, ``az`` [g], ``gx``,
        ``gy``, ``gz`` [deg/s] and ``roll``, ``pitch``, ``yaw`` [deg].
    gravity : float, optional
        Expected specific-force magnitude at rest [g].
    accel_tol : float, optional
        Largest accepted ``abs(|a| - gravity)`` [g].
    gyro_max : float, optional
        Largest accepted absolute angular rate on any axis [deg/s].
    roll_max, pitch_max, yaw_max : float, optional
        Largest accepted absolute roll, pitch and yaw [deg].

    Returns
    -------
    pandas.Series of bool
        True where the record fails any limit, aligned on ``df.index``.
    """
    amag = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    return (
        (np.abs(amag - gravity) > accel_tol)
        | (df[["gx", "gy", "gz"]].abs() > gyro_max).any(axis=1)
        | (df["roll"].abs() > roll_max)
        | (df["pitch"].abs() > pitch_max)
        | (df["yaw"].abs() > yaw_max)
    )


def blocks(times, gap_s):
    """Index ranges of the runs separated by more than *gap_s*.

    Parameters
    ----------
    times : array_like of datetime64
        Record timestamps, in record order.
    gap_s : float
        Timestamp separation [s] that marks a logger outage rather than
        sampling jitter.

    Returns
    -------
    list of tuple of int
        ``(start_index, end_index)`` of each run, both inclusive.
    """
    t = epoch_s(times)
    cut = np.where(np.diff(t) > gap_s)[0]
    return list(zip(np.r_[0, cut + 1], np.r_[cut, len(t) - 1]))


def clean_imu(imu, start, *, inclusive, average_duplicates):
    """IMU records past a deployment start, spike-screened.

    Parameters
    ----------
    imu : pandas.DataFrame or sequence of pandas.DataFrame
        IMU records with a ``time`` column and the channels screened by
        :func:`flag_spikes`. A sequence holds one frame per logger file or
        card; the frames are concatenated after trimming.
    start : scalar or sequence
        First usable time (anything comparable with the ``time`` column,
        e.g. a :class:`pandas.Timestamp` or a date string). A sequence
        pairs element-wise with a sequence *imu*.
    inclusive : bool
        Keep records stamped exactly at *start* (``time >= start``) when
        True, drop them (``time > start``) when False.
    average_duplicates : bool
        When True, the screened records are averaged per duplicate
        timestamp (``groupby("time").mean()``) and then sorted. When False,
        the records are sorted by time (pandas default, unstable sort)
        before screening and duplicate stamps are kept.

    Returns
    -------
    pandas.DataFrame
        Screened records on a fresh ``RangeIndex``, sorted by ``time``.
    """
    if isinstance(imu, pd.DataFrame):
        d = imu[imu["time"] >= start] if inclusive else imu[imu["time"] > start]
    else:
        frames = [f[f["time"] >= s] if inclusive else f[f["time"] > s]
                  for f, s in zip(imu, start)]
        d = pd.concat(frames, ignore_index=True)
    if average_duplicates:
        d = d[~flag_spikes(d)]
        return d.groupby("time", as_index=False).mean().sort_values("time").reset_index(drop=True)
    out = d.sort_values("time")
    return out[~flag_spikes(out)].reset_index(drop=True)


def hold_exit_jumps(df, gap_s, min_records=HOLD_MIN_BLOCK_RECORDS):
    """Yaw change on exit from each held stretch, within one contiguous block.

    The fused yaw stops updating when the gyro reads exactly zero. A hold that
    ends with a large jump would mean real rotation accumulated unseen while
    the channel was frozen; small exits mean the hold is a fusion dead-zone on
    a platform that genuinely did not turn.

    Parameters
    ----------
    df : pandas.DataFrame
        IMU records with columns ``time`` (datetime64) and ``yaw`` [deg],
        sorted by time.
    gap_s : float
        Timestamp separation [s] that splits the record into contiguous
        blocks (see :func:`blocks`).
    min_records : int, optional
        Blocks with ``end_index - start_index`` below this are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per hold exit: stamp ``t`` of the first record after the
        hold, ``hold_s`` of the stretch that just ended and the signed
        ``djump`` [deg] taken on leaving it.
    """
    rows = []
    d = df.reset_index(drop=True)
    for s, e in blocks(d["time"], gap_s):
        if e - s < min_records:
            continue
        sub = d.iloc[s:e + 1]
        yaw = sub["yaw"].to_numpy()
        t = epoch_s(sub["time"])
        idx = np.where(np.r_[True, yaw[1:] != yaw[:-1]])[0]
        if len(idx) < 3:
            continue
        hold = np.diff(np.r_[t[idx], t[-1]])
        jump = (np.r_[np.nan, np.diff(yaw[idx])] + 180) % 360 - 180
        rows.append(pd.DataFrame({"t": sub["time"].to_numpy()[idx],
                                  "hold_s": np.r_[np.nan, hold[:-1]],
                                  "djump": jump}))
    return pd.concat(rows, ignore_index=True).dropna()


def frozen_attitude_mask(imu, min_hold_s):
    """Records inside a roll-and-pitch flat-line lasting at least *min_hold_s*.

    The fused attitude stops updating together with the yaw when the gyro
    reads exactly zero; a long flat-line means the channel carries no tilt
    information over that stretch, whatever value it holds.

    Parameters
    ----------
    imu : pandas.DataFrame
        IMU records with columns ``time`` (datetime64), ``roll`` and
        ``pitch``, sorted by time.
    min_hold_s : float
        Shortest flat-line duration [s] treated as missing attitude.

    Returns
    -------
    mask : numpy.ndarray of bool
        True for records inside a flat-line of at least *min_hold_s*.
    durations : numpy.ndarray
        Duration [s] of every run of identical (roll, pitch), from its first
        to its last stamp, in record order.
    """
    rp = imu[["roll", "pitch"]].to_numpy()
    t = epoch_s(imu["time"])
    same = np.r_[False, np.all(rp[1:] == rp[:-1], axis=1)]
    starts = np.flatnonzero(~same)
    runs = np.r_[starts[1:], len(same)] - starts
    mask = np.zeros(len(same), dtype=bool)
    durations = []
    for s, n in zip(starts, runs):
        dur = t[s + n - 1] - t[s]
        durations.append(dur)
        if dur >= min_hold_s:
            mask[s:s + n] = True
    return mask, np.asarray(durations)


def substitute_accel_tilt(imu, frozen):
    """Replace the fused roll and pitch by the accelerometer tilt where frozen.

    The accelerometer tilt is ``roll = atan2(a_y, a_z)`` and
    ``pitch = atan2(-a_x, hypot(a_y, a_z))``, in degrees.

    Parameters
    ----------
    imu : pandas.DataFrame
        IMU records with columns ``ax``, ``ay``, ``az``, ``roll`` and
        ``pitch``.
    frozen : numpy.ndarray of bool
        Records to substitute, e.g. the mask of
        :func:`frozen_attitude_mask`.

    Returns
    -------
    pandas.DataFrame
        A copy of *imu* with ``roll`` and ``pitch`` replaced on *frozen*.
    """
    out = imu.copy()
    if not frozen.any():
        return out
    roll_a = np.degrees(np.arctan2(out["ay"], out["az"]))
    pitch_a = np.degrees(np.arctan2(-out["ax"], np.hypot(out["ay"], out["az"])))
    out.loc[frozen, "roll"] = roll_a[frozen]
    out.loc[frozen, "pitch"] = pitch_a[frozen]
    return out


def derive_mount_tilt(imu, attitude_signs):
    """Mount roll and pitch [deg] in the platform frame, from the record medians.

    The logger's median roll and pitch are mapped into the platform frame
    by the same per-axis signs that ``IMUInfo.attitudeSigns`` applies per
    sample. For a logger frame that is x-forward, y-right, z-up and a
    platform frame that is y-left, the roll sign flips while the pitch sign
    does not.

    Parameters
    ----------
    imu : pandas.DataFrame
        IMU records with columns ``roll`` and ``pitch`` [deg].
    attitude_signs : sequence of float
        Logger-to-platform signs of (roll, pitch, yaw); only the first two
        are used.

    Returns
    -------
    mount_roll, mount_pitch : float
        Platform-frame mount tilt [deg].
    """
    roll_med = float(imu["roll"].median())
    pitch_med = float(imu["pitch"].median())
    mount_roll = attitude_signs[0] * roll_med
    mount_pitch = attitude_signs[1] * pitch_med
    return mount_roll, mount_pitch
