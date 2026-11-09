# testbed -- runnable space, not a knowledge silo

Dev-facing, mirroring `dopli/testbed/`. Where code gets exercised:
validation runs, comparisons against reference processing, throwaway
working files. Nothing here is a record and nothing here is a task
store. Findings produced by a testbed run are written up in `library/`
notes or the task doc that asked for them, and any open follow-up gets
a line on [tasks/board.md](../tasks/board.md) before the session ends.

## Directories

- [scripts/](scripts/) -- runnable validation entry points (`run_*`,
  `compare_*`, `plot_*`). Tracked. Run from the repo root through the
  `UTESpac_Plus` conda env.
- scratch/ -- assistant-facing working space: one-off probes, logs,
  comparison figures. Gitignored in full.

Site configuration (`siteInfo.toml`, `PFinfo.json`) lives under
`--root` (`<repo>/data` when it holds the site, else
`UTESPAC_DATA_ROOT`); site data lives under a data root set by
`--data-root` or `UTESPAC_DATA_ROOT`, defaulting to the configuration
root (see [data/README.md](../data/README.md)), once a site's data moves
to its own private repo. The `run_*`, `compare_*`, `pf_*`, `closure_*`
and `plot_*` scripts read from the data root; the `ec_*` scripts take a
file path, default `<repo>/data/<SITE>/output` (a `*_hf_*.nc`
high-frequency file there, or the folder itself, holding
`*_coherent_*.nc`, for `ec_full_regression_*`). Testbed runs write
pipeline outputs back under `<data root>/<SITE>/output/` or figures
into scratch/; the `ec_*` scripts write their `_coherent_` file beside
the input file.

The dated `*.md` gameplans at this level predate the `tasks/` silo and
stay here as the ec_coherent planning record (the build-out itself is
tracked in `tasks/active/2026-08-23_ec-coherent-buildout.md`); new plans
go to `tasks/active/`.
