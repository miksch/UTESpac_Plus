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

Site data stays at the top-level `data/` (see [data/README.md](../data/README.md));
testbed runs read from it and write pipeline outputs back under
`data/<SITE>/output/` or figures into scratch/.

The dated `*.md` gameplans at this level predate the `tasks/` silo and
stay here as the ec_coherent planning record until the package work
begins; new plans go to `tasks/active/`.
