# UTESpac_Plus

Python port of the UTESpac eddy-covariance pipeline (package `utespac`)
plus `ec_coherent`, the coherent-structure and spectral methods built
on it. The shared instructions live at the user level; this file holds
only what is specific to this repo.

## Kinds of work

- Coding: `utespac/` (flux processing; `raw_processing/` for card
  conversion and the fast and slow tables), `ec_coherent/`, `tests/`,
  and the entry point `utespac_main.py`. The packages sit at the repo
  root, not under `src/`; the coding rule applies to them.
- Analysis: `testbed/scripts/` holds the runnable demos and campaign
  validation scripts; `testbed/scratch/` (gitignored) is the working
  space, and the only place for campaign-specific scripts.
- Research: `library/`. The PDFs, `My Library.bib` and `extracted/` are
  gitignored; `index.md` (abbreviated filename to bibkey, several keys
  on a row where one PDF carries them), `references.bib` and
  `writeups/` are tracked.
- `tasks/` is the active silo (board, history, active, on_hold,
  archive) and is gitignored as a whole: never committed, never
  staged. `UTESpac_MATLAB/` is the original package and `BUGFIXES.txt`
  the record of bugs found while porting it.

## Environment

- Env name: `UTESpac_Plus`, with the package installed editable.
  Invocation as in `machine.md`.
- Paper extraction: the user-level script with `--papers library
  --index library/index.md --out library/extracted`; `--check-index`
  compares the folder with the index in both directions.

## Git

`main` is the release branch and `dev` the integration branch. Feature
branches are `<area>/<slug>` off `dev`, the area matching the log
(`fluxes/`, `raw_processing/`, `meta/`, `docs/`), squash-merged into
`dev`. Unlike the user rule, here nothing lands on `dev` or `main` by a
direct commit, doc-only edits included; opening a PR, merging one and
pushing to either branch need the user's go-ahead. `dev` reaches `main`
as a release merge, not a squash, so `main` keeps each `dev` commit
underneath. The PR description also carries the BUGFIXES entries.

`origin` is `miksch/UTESpac_Plus`; `upstream` is the original
`Diane1306/UTESpac_Plus`. Real site identifiers stay out of tracked
files (the pre-PR diff in the coding rule); `data/fixtures/` stays
local for the same reason.

## Data

`data/` is gitignored except `data/README.md`, each site's
`siteInfo.toml` and the UTESpac header files. `utespac_main.py --root`
and `UTESPAC_DATA_ROOT` point the pipeline at any other data root;
`../vac-proc` keeps its sites that way.
