# Handoff: after the audit and the VAC001 validation pass

Opened 2026-08-22 at the close of the audit session. State of the tree:
the audit doc
[2026-08-22_code-audit-and-python-gameplan.md](2026-08-22_code-audit-and-python-gameplan.md)
holds the findings, the VAC001-vs-EddyPro results, and the migration plan;
[board.md](../board.md) holds the queue. Landed this session: the `data/`
per-site tree and `tasks/`/`testbed/` silos, the planar-fit rotation fix
(`utespac/sonic_rotation.py`, legacy behaviour behind `info["matlabCompat"]`),
VAC001 processed end to end, and five validation scripts in
`testbed/scripts/`. Suite: 77 passed, 1 skipped.

How to run anything here: repo root, `conda run -n UTESpac_Plus python
testbed/scripts/<script>.py` (conda at
`C:\Users\mdmiksch\AppData\Local\miniconda3\condabin\conda.bat`). The
pipeline on VAC001: `run_vac001.py` (LPF) / `run_vac001_gpf.py` (GPF,
prompts scripted); comparisons: `compare_vac001_eddypro.py --pf GPF --det
ConstDet`, `plot_vac001_eddypro.py`, `pf_vac001_eddypro.py`,
`closure_vac001.py`. A full LPF run of the 16-day record takes ~30 min
(`np.genfromtxt` on 3.4 M rows per file is the cost).

## 1. Schotanus and WPL temperature flux (the one correction the data asked for)

Board: `## utespac-core`, second item (BLOCKED on sources). The VAC001
comparison put a number on finding 3: UTESpac H exceeds EddyPro H by
0.060 × LE (theory 0.062), and applying the term post hoc takes the bias
from +10.4 to −0.6 W/m². Finding 4 (WPL terms driven by w′θv′) is the same
missing quantity leaking into LE and Fc, whose slopes against EddyPro
(0.989, 0.940) should move toward 1 once w′T′ feeds them.

Route: obtain Schotanus et al. 1983, Kaimal & Gaynor 1991, Webb et al.
1980 (ask-for-sources; none are in `library/` yet); write the library
note with the extracted equations and loci; then in `fluxes.py` compute
w′T′ = w′Ts′ − 0.51·T̄·w′q′ where HF humidity exists (fall back to the
mean-humidity rescale otherwise), carry the 0.51 vs 0.61 decision with
its citation, and feed that w′T′ to every WPL term (LE W/m², CO2 flux,
KH2O O2). Behind `matlabCompat` where it breaks parity. Acceptance: re-run
`compare_vac001_eddypro.py --pf GPF --det ConstDet`; expect H bias → ~0
and r² → ~0.99, LE/Fc slopes closer to 1. Also worth noting in the note:
EasyFlux loggers emit `T_SONIC_corr` (the logger-side correction), which
EddyPro consumed here; a config option to use it directly is cheap.

## 2. `get_data` blanking whole days

Board: `## utespac-core`, last item. Small and self-contained: concatenate
by header label instead of matrix shape (or stop trimming all-NaN columns
outside `matlabCompat`). The VAC001 fine-wire outage 07-12..07-17 is the
reproduction case. Do this before anything consumes multi-day pickles.

## 3. Bring the other sites into `data/` and regenerate their GPF products

Board: `## migration`, second item, plus the rotation item's "remaining".
Move `siteGill20250723_20250828` and `siteIRGA20250723_20250828` to
`data/Gill…/` and `data/IRGA…/` (`utespac/` for headers and 48-h files;
`tests/test_tower_profile.py` hardcodes the old paths). Then re-run GPF
with the fixed rotation; every GPF product from before is affected at the
order the VAC001 before/after showed (u* +18 %, LE +20 % there). Extend
`pf_vac001_eddypro.py`'s 3D figure to per-height/sector/date-bin for these
multi-sonic sites (board: the 3D figure item). Add `matlabCompat=True`
runs to the testkit parity comparison so the MATLAB goldens still serve.

## 4. The remaining audit fixes

Board `## utespac-core`: dissipation estimator (finding 5; validate on a
synthetic Kolmogorov signal), slope geometry into `SiteInfo` (finding 6),
and the minor-fixes batch (finding 7). Independent of each other and of
item 1.

## 5. Migration steps 2–5 and ec_coherent

The migration plan in the audit doc ("Migration gameplan") — RunConfig
and logging with dopli-style packaged TOMLs in `utespac/config/`, then the
labeled I/O boundary — starts after items 1–2. ec_coherent Phase 0
(library notes) is unblocked now that the rotation fix is in; its
converter should run only on regenerated pickles (VAC001's are already
post-fix).

## User-side items noted

- `siteVAC001_20230706_20230720/card_convert/` is a byte-identical copy of
  `data/VAC001/raw/fast/` (3.9 GB) and can be deleted; the old folder's
  `siteInfo.py` is superseded by `data/VAC001/siteInfo.toml`.
- VAC001 `tower = 180` is `[ASSUMED]` (no effect on this IOP); sonic
  height kept at 10.85 m against EddyPro's rounded 11.00 m.
- Closure used provisional Rn (assembled outside the repo) and the rebuilt
  `ghf_avg`; `closure_vac001.py` re-runs in seconds if either is revised.
