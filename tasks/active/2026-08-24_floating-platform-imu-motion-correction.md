# Floating-platform IMU motion correction (2026-08-24)

Tilt/motion correction for a sonic on a moored floating platform with a
colocated IMU. Greenfield: no IMU, motion, or moving-platform code exists
anywhere in the repo (checked 2026-08-24); both processed sites are fixed
land towers.

## Physics

The sonic measures wind in the platform frame. The earth-frame wind is

    u_true = T u_obs + T (Omega x r) + v_plat

where T is the platform-to-earth rotation built from IMU attitude
(roll/pitch/yaw), Omega the angular-rate vector, r the lever arm from the
IMU to the sonic head, and v_plat the platform translational velocity
obtained by integrating the (rotated, high-pass-filtered) accelerometers.
Canonical formulation: Edson et al. (1998). Angular-rate integration and
filter handling updated by Miller et al. (2008). Practical recipe with
filter constants: Landwehr et al. (2015). The moored-buoy special case --
no mean translation, so no GPS complement needed and a complementary
filter closes the attitude at low frequency -- is Anctil et al. (1994).

For a moored raft (not an underway ship) the Prytherch-style
motion-correlated flow-distortion biases are second-order, but the
validation step below still checks for them.

## Where it lands in the code

- New stage `stages.motion(run)` called between `stages.condition` and
  `stages.average` ([pipeline.py:120](../../utespac/pipeline.py#L120)).
  `stages.rotate` reads the raw HF sonic columns straight from
  `run.tables` ([stages.py:161](../../utespac/stages.py#L161)), so the
  corrected u/v/w must be written back into those Datasets before
  averaging; anything later leaves period means and the planar-fit
  regression computed from uncorrected winds.
- Ingest is column-generic: the IMU rides through `process_table` as
  extra columns. Pattern to copy: `FW_MAP` in
  [VAC_ADV_fast_process.py:73](../../raw_processing/VAC_ADV_fast_process.py#L73)
  (a bare `{output: source}` map not tied to a sonic level). Also needs:
  template keys in [run.toml](../../utespac/config/run.toml) (e.g.
  `imuAx = "Ax_*"`), a branch in
  [find_instruments.py](../../utespac/find_instruments.py) (only
  `field == "u"` gets orientation today), qc entries in
  [qc.toml](../../utespac/config/qc.toml) for a new IMU sensor class
  (else the `otherInstrument = 5.0` spike default applies), and a
  `tableNumberOfColumns` bump in the site's `siteInfo.toml`.
- Site geometry (lever arm, IMU-to-sonic mounting angles) goes on
  `SiteInfo` / `SonicLevel` in
  [site_config.py](../../utespac/site_config.py) -- unknown keys warn
  and drop, so the schema must be extended.
- Rotation core stays [rotation.py](../../utespac/rotation.py) (planar
  fit + yaw); see step 4 for how the two interact.

## Build order

Each step ends with the suite green; no step invalidates a previous
step's baseline.

1. **Data contract.** Fix the IMU channel list, units, and sign
   conventions; measure and record the lever arm r and the IMU-to-sonic
   mounting rotation.

   DECIDE: which IMU, and does it output raw accel/gyro only, or a
   fused AHRS attitude? Raw-only means we own a complementary filter
   (Anctil 1994); vendor attitude means we only build T and integrate
   accelerations.
   (default: raw accel + gyro + fused roll/pitch/yaw all logged; use
   vendor attitude, keep raw channels for checks)
   A: default adopted 2026-08-24, hardware still unknown -- the build is
   generic instead: any subset of accel/gyro/attitude triads works
   (vendor attitude preferred via `[imu] useVendorAttitude`, complementary
   filter when only raw channels exist, attitude-only IMUs get rates by
   differentiation); units, axis signs and mounting rotation are declared
   per site in the `[imu]` siteInfo block. Revisit only to fill in the
   actual siteInfo values once the instrument exists.

   DECIDE: clock sync -- same CR6/datalogger scan as the sonic, or a
   separate logger needing lag alignment? Sub-scan misalignment leaks
   motion into w'.
   (default: same logger, same scan table, same rate as the sonic)
   A: default adopted 2026-08-24 as the primary contract (IMU columns in
   the sonic's fast table via `imu_columns()`, the FW_MAP pattern). The
   separate-logger fallback exists anyway: `raw_processing/imu.py`
   `load_imu_files(lag_s=...)` + `align_imu()` (nearest sample within one
   scan, NaN across gaps -- no interpolation that would smear motion).
   Sub-scan lag estimation from w'-accel coherence stays a real-data task.

2. **Ingest.** Site fast-process script with an `IMU_MAP`, template
   keys, `find_instruments` branch, qc.toml entries, `SiteInfo` schema
   fields. Ends with the IMU columns visible on `run.hf` and NaN-safe
   through `condition`.

3. **Motion stage.** `utespac/motion.py` + `stages.motion`: build T per
   sample, correct for lever-arm angular velocity, high-pass and
   integrate accelerations for v_plat, write corrected u/v/w back into
   `run.tables`. Filter constants per Landwehr et al. (2015).

   DECIDE: high-pass cutoff for the acceleration integration (Landwehr
   use ~1/30 Hz class cutoffs at sea; a small lake's wave band is
   faster, so a shorter cutoff may do).
   (default: 20 s complementary/high-pass constant, revisit against
   the wave peak in step 5)
   A: default adopted 2026-08-24: `Tcf = Ta = 20 s`, 4th-order
   Butterworth applied filtfilt (Miller 2008), both site-tunable in the
   `[imu]` block. Must be re-checked against the measured wave peak
   (step 5) before first production use.

4. **Rotation interplay.** After motion correction the winds are already
   earth-frame level, so the planar fit should collapse to near-identity
   plus any real mean streamline tilt.

   DECIDE: keep running planar fit over the corrected winds (catches
   residual IMU mounting bias; PF coefficients become a diagnostic) or
   skip PF for this site and go straight to yaw rotation?
   (default: keep PF on, sector-wise, and treat pitch/roll > ~1 deg in
   the fit as an IMU mounting-calibration flag)
   A: default adopted 2026-08-24; no code change needed -- `stages.motion`
   runs before `average`, so the PF regression and yaw rotation already
   see corrected winds. Yaw handling supports this: the default
   `yawHandling = "demean"` leaves winds in the mean-heading frame, so
   sonic `orientation` and the downstream direction/yaw stages keep their
   fixed-tower meaning (proven by the demean round-trip test).

5. **Validation.** Before/after w spectra and uw/wT cospectra: the wave
   peak in Sww must collapse after correction (Miller 2008 fig-style
   check); w'-vs-platform-acceleration coherence must drop to noise;
   flux impact quantified per period. Synthetic rigid-body unit test:
   impose a known oscillatory tilt/heave on synthetic wind, verify the
   stage recovers the input to round-off. Buoy-vs-tower comparison
   design if a shore tower exists: Flügge et al. (2016).

A flux-value change from the new stage is expected and site-new, so no
`tests/KNOWN_DIVERGENCES.md` row; the synthetic round-trip test is the
correctness anchor instead.

## Status 2026-08-24 (built without data; validation waits on real IMU)

Landed, suite green (268 tests):

- `utespac/motion.py`: kernels -- `euler_T`/`euler_from_T` (ZYX, z-up),
  `accel_tilt`, `complementary_attitude` (Anctil/Miller, 4th-order
  Butterworth filtfilt), `platform_velocity` (rotate, de-gravity,
  HP-integrate-HP), `correct_wind` (Edson eq. 4 with lever-arm and
  translational terms, NaN gap fill with reported fractions,
  yaw_handling demean/full/zero).
- `stages.motion` between `condition` and `average`
  (`pipeline.run_utespac` wired): unit/sign/mounting conversion from the
  site `[imu]` block, per-sonic lever arm (`SonicLevel.leverArm`
  override), corrected u/v/w written back into `run.tables`, attitude +
  platform velocity on `run.motion` for the step-5 diagnostics;
  incomplete IMU -> warn and leave winds untouched.
- Ingest: `imu*` template keys (run.toml + `_DEFAULT_TEMPLATE`), qc.toml
  spike/limits entries for the nine channels (imuYaw excluded from the
  spike test -- 0/360 wraps), `IMUInfo` in site_config (`[imu]` siteInfo
  block: leverArm, units, signs, mount angles, Tcf/Ta, yawHandling),
  `raw_processing/imu.py` (`IMU_MAP`/`imu_columns` same-logger path;
  `load_imu_files` + `align_imu` separate-logger path).
- `tests/test_motion.py`: synthetic rigid-body round trips (vendor,
  complementary, attitude-only, demeaned-yaw, lever-arm cancellation)
  recover the input wind to <= 2-5 cm/s; stage in-place + config tests.
  `tests/test_imu_ingest.py` covers the raw helpers.

Still open (needs the instrument and real data): site fast-process
script + siteInfo `[imu]` values, `tableNumberOfColumns` bump, step-5
spectra/coherence/flux-impact validation, filter constants vs. the
measured wave peak, sub-scan lag check for a separate logger.

## References

All six PDFs received in `library/` 2026-08-24; indexed in
[library/index.md](../../library/index.md), bib entries in
[library/references.bib](../../library/references.bib), DOIs
Crossref-verified and matched against the PDFs, extractions cached in
`library/extracted/` (Anctil1994 is a scan -- render pages to read).

- [@Anctil1994] -- moored discus buoy, the closest analog to a raft:
  no mean translation, complementary filtering of tilt angles.
- [@Edson1998] -- canonical earth-frame transform (the equation above).
- [@Miller2008] -- angular-rate integration and filtering updated;
  motion + mounting offsets underestimated stress by 15% uncorrected.
- [@Landwehr2015] -- practical recipe and filter constants, corrections
  revisited.
- [@Prytherch2015] -- residual motion-correlated flow-distortion bias;
  what step 5 checks for.
- [@Flugge2016] -- buoy-vs-tower comparison, the validation design
  template.

## Code references

- `fluxer` (flux-capacitor) by UofM-CEOS, Python:
  `fluxer.eddycov.flux.wind3D_correct` is a port of S. Miller's MATLAB
  `motion` routine implementing [@Miller2008] -- the implementation to
  lift the algorithm from and to cite in the module docstring.
  Signature confirms the step-3 knobs: `anemometer_pos` (lever arm),
  `Tcf` (complementary-filter period), `Ta` (Butterworth high-pass
  cutoff for the acceleration integration), plus mounting-tilt offsets
  for motion pack and anemometer. The `heading`/`speed` inputs are the
  underway-ship terms a moored raft sets to constants.
  Repo: https://github.com/UofM-CEOS/flux_capacitor
  Docs: https://flux-capacitor.readthedocs.io/en/latest/fluxer.eddycov.html
- AirChem/FluxToolbox was checked 2026-08-24 and dropped: standard
  fixed-tower EC only (rotation, detrending, lag covariance), no
  platform-motion code.
