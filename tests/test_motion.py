"""Tests for utespac.motion (platform-motion correction kernels) and the
``stages.motion`` stage.

The correctness anchor is the synthetic rigid-body round trip: impose a
known oscillatory roll/pitch/heave on a known wind, synthesize what the
sonic and IMU would measure, and require the correction to recover the
input (task 2026-08-24_floating-platform-imu-motion-correction step 5).
"""

import numpy as np
import pytest
import xarray as xr

from utespac.motion import (G, MotionParams, accel_tilt, complementary_attitude,
                            correct_wind, euler_from_T, euler_T, fill_gaps,
                            integrate, _body_rates)
from utespac.model import TIME_HF, Run, Sensor, Sensors, to_datetime64
from utespac import stages
from utespac.site_config import IMUInfo, SiteInfo

FS = 20.0                 # [Hz]
N = 12000                 # 600 s
T_SEC = np.arange(N) / FS
EDGE = int(50 * FS)       # trim filtfilt / high-pass edges [samples]


# ── platform kinematics (analytic ground truth) ──────────────────────────────

def _platform(f_wave=0.5, a_roll=np.radians(5), a_pitch=np.radians(3),
              heave=0.15, surge=0.10, sway=0.05, yaw0=0.0):
    """Analytic attitude, body rates, platform velocity and accelerometer
    specific force for sinusoidal roll/pitch about the IMU plus
    translational surge/heave."""
    w = 2 * np.pi * f_wave
    roll = a_roll * np.sin(w * T_SEC)
    pitch = a_pitch * np.sin(w * T_SEC + 1.0)
    yaw = np.full(N, yaw0)
    euler_dot = np.column_stack([a_roll * w * np.cos(w * T_SEC),
                                 a_pitch * w * np.cos(w * T_SEC + 1.0),
                                 np.zeros(N)])
    gyro = _body_rates(roll, pitch, euler_dot)
    # translation: position = amp * sin(w t + phase) per component
    amps = np.array([surge, sway, heave])
    phases = np.array([0.4, 2.1, 0.0])
    arg = w * T_SEC[:, None] + phases
    v_plat = amps * w * np.cos(arg)
    a_plat = -amps * w ** 2 * np.sin(arg)
    T = euler_T(roll, pitch, yaw)
    grav = np.array([0.0, 0.0, G])
    acc = np.einsum("nji,nj->ni", T, a_plat + grav)     # T^T (a_e + g)
    att = np.column_stack([roll, pitch, yaw])
    return att, gyro, v_plat, acc, T


def _observed(u_true, T, gyro, v_plat, lever):
    """What the sonic reports: u_obs = T^T (u_true - v_plat) - omega x r."""
    r = np.broadcast_to(np.asarray(lever, float), (N, 3))
    return np.einsum("nji,nj->ni", T, u_true - v_plat) - np.cross(gyro, r)


def _true_wind():
    rng = np.random.default_rng(7)
    base = np.array([3.0, 0.5, 0.05])
    turb = 0.3 * rng.standard_normal((N, 3))
    return base + turb


# ── kernels ──────────────────────────────────────────────────────────────────

def test_euler_T_axes():
    assert np.allclose(euler_T(0.0, 0.0, 0.0), np.eye(3), atol=1e-15)
    # yaw +90 deg carries platform x to earth y
    assert np.allclose(euler_T(0.0, 0.0, np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    # roll +90 deg carries platform y to earth z
    assert np.allclose(euler_T(np.pi / 2, 0.0, 0.0) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    # pitch +90 deg carries platform z to earth x
    assert np.allclose(euler_T(0.0, np.pi / 2, 0.0) @ [0, 0, 1], [1, 0, 0], atol=1e-15)


def test_euler_round_trip():
    rng = np.random.default_rng(1)
    roll = rng.uniform(-1.2, 1.2, 50)
    pitch = rng.uniform(-1.2, 1.2, 50)
    yaw = rng.uniform(-np.pi, np.pi, 50)
    r2, p2, y2 = euler_from_T(euler_T(roll, pitch, yaw))
    np.testing.assert_allclose(r2, roll, atol=1e-12)
    np.testing.assert_allclose(p2, pitch, atol=1e-12)
    np.testing.assert_allclose(y2, yaw, atol=1e-12)


def test_accel_tilt_static():
    roll, pitch = np.radians(4.0), np.radians(-7.0)
    f = euler_T(roll, pitch, 0.3).T @ [0, 0, G]        # yaw does not matter
    r, p = accel_tilt(f[np.newaxis, :])
    assert abs(r[0] - roll) < 1e-12 and abs(p[0] - pitch) < 1e-12


def test_integrate_sine():
    w = 2 * np.pi * 0.5
    got = integrate(np.cos(w * T_SEC), FS)
    # trapezoid amplitude error ~ (w/fs)^2 / 12
    np.testing.assert_allclose(got, np.sin(w * T_SEC) / w, atol=1e-3)


def test_fill_gaps():
    x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
    filled, frac = fill_gaps(x)
    np.testing.assert_allclose(filled, [1, 2, 3, 4, 5, 6])
    assert frac == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fill_gaps(np.full(4, np.nan))


def test_complementary_static_tilt():
    roll, pitch = np.radians(3.0), np.radians(-2.0)
    f = euler_T(roll, pitch, 0.0).T @ [0, 0, G]
    acc = np.broadcast_to(f, (N, 3)).copy()
    att = complementary_attitude(acc, np.zeros((N, 3)), FS, Tcf=20.0)
    core = slice(EDGE, -EDGE)
    np.testing.assert_allclose(att[core, 0], roll, atol=1e-6)
    np.testing.assert_allclose(att[core, 1], pitch, atol=1e-6)
    np.testing.assert_allclose(att[core, 2], 0.0, atol=1e-6)


# ── rigid-body round trips ───────────────────────────────────────────────────

LEVER = (0.1, -0.05, 0.8)


def _params(**kw):
    return MotionParams(fs=FS, lever_arm=LEVER, Tcf=20.0, Ta=20.0, **kw)


def test_round_trip_vendor_attitude():
    """Vendor attitude + gyro + accel recovers the true wind to ~cm/s."""
    att, gyro, v_plat, acc, T = _platform()
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, v_plat, LEVER)
    res = correct_wind(u_obs, _params(yaw_handling="full"), acc=acc, gyro=gyro,
                       attitude=att)
    core = slice(EDGE, -EDGE)
    err = np.abs(res.uvw[core] - u_true[core])
    assert err.max() < 0.02
    # and the contamination was real: uncorrected w is far off
    assert np.abs(u_obs[core, 2] - u_true[core, 2]).max() > 0.3


def test_round_trip_complementary():
    """Raw accel + gyro only (complementary attitude, Anctil 1994 path)."""
    att, gyro, v_plat, acc, T = _platform()
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, v_plat, LEVER)
    res = correct_wind(u_obs, _params(), acc=acc, gyro=gyro)
    core = slice(EDGE, -EDGE)
    # second-order Euler-rate coupling leaves ~0.4 deg; fine for a raft
    np.testing.assert_allclose(res.roll[core], att[core, 0], atol=0.01)
    np.testing.assert_allclose(res.pitch[core], att[core, 1], atol=0.01)
    err = np.abs(res.uvw[core] - u_true[core])
    assert err.max() < 0.05


def test_round_trip_attitude_only():
    """Attitude-only IMU: gyro differentiated from the angles, no
    translational term — tilt and lever arm still come out right when the
    platform does not translate."""
    att, gyro, _, _, T = _platform(heave=0.0, surge=0.0, sway=0.0)
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, np.zeros((N, 3)), LEVER)
    res = correct_wind(u_obs, _params(yaw_handling="full"), attitude=att)
    core = slice(EDGE, -EDGE)
    assert np.abs(res.uvw[core] - u_true[core]).max() < 0.02


def test_lever_arm_pure_rotation_cancels():
    """Zero wind, pure rotation about the IMU: the sonic sees -omega x r
    and the correction must cancel it to zero."""
    att, gyro, _, acc, T = _platform(heave=0.0, surge=0.0, sway=0.0)
    u_obs = _observed(np.zeros((N, 3)), T, gyro, np.zeros((N, 3)), LEVER)
    assert np.abs(u_obs).max() > 0.05          # the lever term is not trivial
    res = correct_wind(u_obs, _params(yaw_handling="full"), acc=acc, gyro=gyro,
                       attitude=att)
    core = slice(EDGE, -EDGE)
    assert np.abs(res.uvw[core]).max() < 5e-3


def test_demean_yaw_keeps_mean_heading_frame():
    """yaw_handling='demean' with a constant heading returns the winds in
    the mean-heading horizontal frame: Rz(-yaw0) applied to the earth
    wind, w untouched."""
    yaw0 = np.radians(35.0)
    att, gyro, v_plat, acc, T = _platform(yaw0=yaw0)
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, v_plat, LEVER)
    res = correct_wind(u_obs, _params(), acc=acc, gyro=gyro, attitude=att)
    expect = np.einsum("ij,nj->ni", euler_T(0.0, 0.0, -yaw0), u_true)
    core = slice(EDGE, -EDGE)
    assert np.abs(res.uvw[core] - expect[core]).max() < 0.02


def test_nan_gaps_filled_and_reported():
    att, gyro, v_plat, acc, T = _platform()
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, v_plat, LEVER)
    acc[500:520, 2] = np.nan
    res = correct_wind(u_obs, _params(yaw_handling="full"), acc=acc, gyro=gyro,
                       attitude=att)
    assert res.imu_nan_frac["acc_z"] == pytest.approx(20 / N)
    core = slice(EDGE, -EDGE)
    assert np.abs(res.uvw[core] - u_true[core]).max() < 0.03


def test_correct_wind_input_validation():
    with pytest.raises(ValueError, match="needs a fused attitude"):
        correct_wind(np.zeros((100, 3)), _params())
    with pytest.raises(ValueError, match="yaw_handling"):
        MotionParams(fs=FS, lever_arm=LEVER, yaw_handling="bogus")
    with pytest.raises(ValueError, match="lever_arm"):
        MotionParams(fs=FS, lever_arm=(1.0, 2.0))


# ── conditioning of the IMU columns ──────────────────────────────────────────

def test_qc_passes_imu_columns():
    """imuYaw 0/360 wraps survive the spike test untouched; gyro limit
    violations are NaN'd; ordinary IMU channels ride through."""
    from utespac.condition_data import qc_table
    from utespac.run_config import QCConfig, _DEFAULT_TEMPLATE

    hz, n = 10.0, 6000                       # 10 min at 10 Hz
    t = T0_DATENUM + np.arange(n) / (86400.0 * hz)
    rng = np.random.default_rng(3)
    yaw = (5.0 * np.arange(n) / hz) % 360.0              # slow spin, wraps
    gx = 2.0 * np.sin(2 * np.pi * 0.5 * np.arange(n) / hz)
    gx[100] = 600.0                                       # beyond [-500, 500]
    roll = 3.0 * np.sin(2 * np.pi * 0.5 * np.arange(n) / hz) \
        + 0.1 * rng.standard_normal(n)
    tbl = np.column_stack([t, yaw, gx, roll])
    labels = ["TIMESTAMP", "IMU_Yaw", "IMU_Gx", "IMU_Roll"]
    info = {"avgPer": 5, **QCConfig().to_info()}
    out, spike, nan = qc_table(tbl.copy(), labels, info, dict(_DEFAULT_TEMPLATE), hz)
    yaw_out = out[:, 1]
    np.testing.assert_array_equal(yaw_out, yaw)           # wraps not "despiked"
    assert np.isnan(out[100, 2]) and not np.isnan(out[99, 2])
    assert np.isfinite(out[:, 3]).mean() > 0.99


# ── site config ──────────────────────────────────────────────────────────────

def test_imu_info_from_mapping():
    site = SiteInfo.from_mapping({
        "imu": {"leverArm": [0.1, 0.0, 0.8], "gyroUnits": "rad/s",
                "Tcf": 30.0, "yawHandling": "zero"}})
    assert isinstance(site.imu, IMUInfo)
    assert site.imu.leverArm == [0.1, 0.0, 0.8]
    assert site.imu.Tcf == 30.0 and site.imu.gyroUnits == "rad/s"
    assert site.imu.accelSigns == [1.0, 1.0, 1.0]      # default filled in


def test_imu_info_validation():
    with pytest.raises(ValueError, match="leverArm"):
        IMUInfo(leverArm=[1.0])
    with pytest.raises(ValueError, match="gyroUnits"):
        IMUInfo(leverArm=[0, 0, 1], gyroUnits="furlong/fortnight")
    with pytest.raises(ValueError, match="accelSigns"):
        IMUInfo(leverArm=[0, 0, 1], accelSigns=[1, 2, 1])
    with pytest.warns(UserWarning, match="unrecognized imu key"):
        SiteInfo.from_mapping({"imu": {"leverArm": [0, 0, 1], "bogusKey": 3}})


# ── the stage ────────────────────────────────────────────────────────────────

T0_DATENUM = 738000.0     # arbitrary day


def _make_run(info, columns):
    """A one-table Run: dict of {column: series} at FS, plus sensors."""
    t64 = to_datetime64(T0_DATENUM + T_SEC / 86400.0)
    ds = xr.Dataset({name: (TIME_HF, series) for name, series in columns.items()},
                    coords={TIME_HF: t64})
    ds.attrs["scan_hz"] = FS
    sensors = Sensors()
    for name in columns:
        if name.startswith("IMU_"):
            field = {"Ax": "imuAx", "Ay": "imuAy", "Az": "imuAz",
                     "Gx": "imuGx", "Gy": "imuGy", "Gz": "imuGz",
                     "Roll": "imuRoll", "Pitch": "imuPitch",
                     "Yaw": "imuYaw"}[name[4:]]
            sensors.append(Sensor(field, "X_20Hz", name, np.nan))
        else:
            field = {"Ux": "u", "Uy": "v", "Uz": "w"}[name.split("_")[0]]
            sensors.append(Sensor(field, "X_20Hz", name, 2.5, 0.0, 1))
    labels = ["TIMESTAMP"] + list(columns)
    return Run(site=info, sensors=sensors, table_names=["X_20Hz"],
               headers={"X_20Hz": (labels, [None] * len(labels))},
               tables={"X_20Hz": ds})


def _stage_columns(att, gyro, u_obs, degrees=True):
    s = np.degrees if degrees else (lambda x: x)
    return {"Ux_2.5": u_obs[:, 0], "Uy_2.5": u_obs[:, 1], "Uz_2.5": u_obs[:, 2],
            "IMU_Roll": s(att[:, 0]), "IMU_Pitch": s(att[:, 1]),
            "IMU_Yaw": s(att[:, 2]),
            "IMU_Gx": s(gyro[:, 0]), "IMU_Gy": s(gyro[:, 1]),
            "IMU_Gz": s(gyro[:, 2])}


def test_stage_corrects_in_place():
    """Attitude + gyro in logger units (deg, deg/s) through the stage
    matches the kernel called with SI inputs."""
    att, gyro, _, _, T = _platform(heave=0.0, surge=0.0, sway=0.0)
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, np.zeros((N, 3)), LEVER)
    info = {"imu": {"leverArm": list(LEVER), "yawHandling": "full"}}
    # the stage writes through to the u_obs buffer, so compute the
    # kernel expectation first
    expect = correct_wind(u_obs, _params(yaw_handling="full"), gyro=gyro,
                          attitude=att)
    run = _make_run(info, _stage_columns(att, gyro, u_obs))
    stages.motion(run)
    ds = run.tables["X_20Hz"]
    got = np.column_stack([ds["Ux_2.5"].values, ds["Uy_2.5"].values,
                           ds["Uz_2.5"].values])
    np.testing.assert_allclose(got, expect.uvw, atol=1e-10)
    core = slice(EDGE, -EDGE)
    assert np.abs(got[core] - u_true[core]).max() < 0.02
    assert run.motion is not None
    assert run.motion.attrs["attitude_source"] == "vendor"
    np.testing.assert_allclose(run.motion["roll"].values, np.degrees(att[:, 0]),
                               atol=1e-10)


def test_stage_noop_without_imu_config():
    att, gyro, _, _, T = _platform(heave=0.0, surge=0.0, sway=0.0)
    u_obs = _observed(_true_wind(), T, gyro, np.zeros((N, 3)), LEVER)
    run = _make_run({}, _stage_columns(att, gyro, u_obs))
    before = run.tables["X_20Hz"]["Uz_2.5"].values.copy()
    stages.motion(run)
    np.testing.assert_array_equal(run.tables["X_20Hz"]["Uz_2.5"].values, before)
    assert run.motion is None


def test_stage_skips_on_incomplete_imu():
    att, gyro, _, _, T = _platform(heave=0.0, surge=0.0, sway=0.0)
    u_obs = _observed(_true_wind(), T, gyro, np.zeros((N, 3)), LEVER)
    cols = _stage_columns(att, gyro, u_obs)
    del cols["IMU_Yaw"], cols["IMU_Gx"], cols["IMU_Gy"], cols["IMU_Gz"]
    run = _make_run({"imu": {"leverArm": list(LEVER)}}, cols)
    before = run.tables["X_20Hz"]["Uz_2.5"].values.copy()
    with pytest.warns(UserWarning, match="triad incomplete|correction skipped"):
        stages.motion(run)
    np.testing.assert_array_equal(run.tables["X_20Hz"]["Uz_2.5"].values, before)
    assert any("skipped" in w for w in run.warnings)


def test_stage_per_sonic_lever_arm():
    """A SonicLevel leverArm overrides the [imu] default."""
    from utespac.site_config import SonicLevel
    att, gyro, _, _, T = _platform(heave=0.0, surge=0.0, sway=0.0)
    u_true = _true_wind()
    u_obs = _observed(u_true, T, gyro, np.zeros((N, 3)), LEVER)
    info = {"imu": {"leverArm": [9.0, 9.0, 9.0], "yawHandling": "full"},
            "sonics": [SonicLevel(height=2.5, orientation=0.0, manufacturer=1,
                                  leverArm=list(LEVER))]}
    run = _make_run(info, _stage_columns(att, gyro, u_obs))
    stages.motion(run)
    ds = run.tables["X_20Hz"]
    got = np.column_stack([ds["Ux_2.5"].values, ds["Uy_2.5"].values,
                           ds["Uz_2.5"].values])
    core = slice(EDGE, -EDGE)
    assert np.abs(got[core] - u_true[core]).max() < 0.02
