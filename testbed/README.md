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

Site data lives under a data root -- `<repo>/data` by default, or
elsewhere via `UTESPAC_DATA_ROOT`/`--root` (see
[data/README.md](../data/README.md)) once a site's data moves to its
own private repo. Testbed runs read from that root and write pipeline
outputs back under `<root>/<SITE>/output/` or figures into scratch/.

The dated `*.md` gameplans at this level predate the `tasks/` silo and
stay here as the ec_coherent planning record (the build-out itself is
tracked in `tasks/active/2026-08-23_ec-coherent-buildout.md`); new plans
go to `tasks/active/`.
