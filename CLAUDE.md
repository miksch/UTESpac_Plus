# UTESpac_Plus

## Git workflow

`main` is the release branch and `dev` is the integration branch. Work
never lands on either by a direct commit.

- Branch from `dev` as `<area>/<short-slug>`, the area matching the log's
  prefixes (`fluxes/`, `raw_processing/`, `meta/`, `docs/`). One branch per
  coherent change.
- Commit and push freely on a feature branch without asking. Opening a PR,
  merging one, and pushing to `dev` or `main` need the user's go-ahead.
- Open the PR once the branch is green, not per commit. Squash-merge into
  `dev`, so `dev` carries one commit per PR and the branch's incremental
  commits do not land.
- `dev` reaches `main` as a release merge, not a squash: `main`'s history
  is the sequence of releases and keeps each `dev` commit underneath.
- Delete the feature branch after the merge, local and remote.

The PR description is the reviewable unit and carries the long form:
symptom, approach, the files and symbols touched, test results, BUGFIXES
entries. Commit messages follow the commit-messages skill — subjects at 72
characters or fewer, bodies at most two short paragraphs — and do not
repeat the PR text.
