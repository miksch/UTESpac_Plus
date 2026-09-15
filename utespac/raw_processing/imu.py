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
"""

import glob

import numpy as np
import pandas as pd

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
