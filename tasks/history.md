# tasks history -- recently closed

Rolling log of roughly the last 15 closures; older records live only in
archive/. Format: `- [DONE|DROPPED YYYY-MM-DD] <one line> -- <link>`.

- [DONE 2026-08-22] Legacy `siteGill20250723_20250828` / `siteIRGA…` folders
  moved to `data/Gill` and `data/IRGA` (siteInfo at the site root, headers
  in `utespac/`; the IRGA 1-min header gained its `.dat`); no `site*` folder
  with tracked files remains at the repo root; `test_tower_profile` and
  `test_site_config` follow the new paths. GPF regeneration for these
  sites stays on the rotation board line (inputs not on this machine).
- [DONE 2026-08-22] ITC σw/u* stable side beyond z/L = 0.4 (and without a
  latitude) on Pahlow et al. 2001 eq. 14, 1.1 + 0.9 (z/L)^0.6; the last
  `[ASSUMED]` in `calc_ssitc_flags._above_canopy_sigmaw` removed; PDFs of
  Pahlow2001, DeBruin1993, Nieuwstadt1984, Sorbjan1986 received --
  `library/writeups/itc_sigmaw.md`, `tests/test_ssitc_itc.py`, ledger row.
- [DONE 2026-08-22] ITC σw/u* reference sourced: unstable side cited to
  Foken 2008 Table 2.11 (= Foken et al. 2004 Table 9.1, Foken et al. 2012
  Table 4.2; 1.3 / 2.0(−z/L)^(1/8) at 0.032), stable side 0 ≤ z/L ≤ 0.4 on
  Thomas & Foken 2002 (Table 2.12, Coriolis form) with new
  `SiteInfo.latitude` (VAC001 38.300056 from EddyPro metadata); z/L > 0.4
  stays the MATLAB extension marked `[ASSUMED]` (no source on hand) --
  `library/writeups/itc_sigmaw.md`, `tests/test_ssitc_itc.py`.
- [DONE 2026-08-22] Schotanus temperature-flux correction and WPL driver
  (audit findings 3-4): `utespac/sonic_temperature.py` (0.51 coefficient,
  T' = Ts' − 0.51 T̄ q'), `fluxes.py` T_air columns from HF humidity and all
  WPL terms on that w'T' (`matlabCompat` keeps 0.61 + buoyancy flux); four
  sources extracted into `library/writeups/sonic_temperature_flux.md`.
  VAC001 GPF vs EddyPro: H bias +11.1 → −0.03 W/m² (RMSE 15.7 → 0.75), Fc
  slope 0.940 → 0.975 -- record in the audit doc "VAC001 test dataset" and
  [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Minor-fixes batch (audit finding 7): `Vtheta_fw` on the
  level-local humidity, one e_sat formula (Stull 1988 eq. 7.5.2d) with the
  moist denominator in both humidity paths, IRGA q_ref from true dry-air
  density, `find_delta_time` over valid samples, `nandetrend` on the true
  index, `find_eta`/ppm/ddof conventions documented; ITC reference split off
  as a BLOCKED board line -- record in
  [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Slope geometry into `SiteInfo` (`downslopeAspect`,
  `slopeAxis`; French Meadows fallback with a warning when `angle != 0`),
  `calc_snsp_angle` parameterized; keys added to the legacy FM siteInfo
  files -- audit finding 6, `tests/test_site_config.py`, `tests/test_utils.py`.
- [DONE 2026-08-22] Dissipation estimator replaced by the compensated
  inertial-subrange average (lags 0.1-2 s, C2 = 2.0); recovers a synthetic
  Kolmogorov signal's own D_LL epsilon to 1 %, VAC001 07-08 gives
  phi_eps = kappa z eps / u*^3 = 1.15 (IQR 1.07-1.27) near neutral -- audit
  finding 5, `tests/test_utils.py::TestDissipationRate`.
- [DONE 2026-08-22] `get_data` concatenates labeled fields by header label
  (union of columns, `matlab_compat=` keeps the MATLAB block-NaN); `derivedT`
  header trimmed with its data; VAC001 07-12..07-16 outputs regenerated, the
  16-day LPF load now has 53 all-NaN rows (gap periods) instead of 341 --
  `tests/test_get_data.py`, ledger in
  [../tests/KNOWN_DIVERGENCES.md](../tests/KNOWN_DIVERGENCES.md).
- [DONE 2026-08-22] Stand up dopli-style silos (`tasks/`, `testbed/scripts`
  + `scratch`) and the per-site `data/` tree; site discovery by `siteInfo.*`,
  inputs in `data/<SITE>/utespac/`; VAC001 raw/slow/EddyPro landed and
  processed to 48-h inputs -- record in
  [active/2026-08-22_code-audit-and-python-gameplan.md](active/2026-08-22_code-audit-and-python-gameplan.md)
  "VAC001 test dataset".
