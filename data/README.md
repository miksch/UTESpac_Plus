# data -- the per-site data tree

One folder per site. Everything the pipeline reads or writes for a site
lives under it; the repo root holds code only. Only this README, each
site's `siteInfo.toml`, and the UTESpac header files are tracked -- raw,
processed, and reference data stay local (see `.gitignore`).

```
data/
  fixtures/              committed regression fixtures (tracked): one mini
                         site tree per fixture with build_fixture.py and
                         expected/ pins; tests/test_pinned_fixture.py
                         discovers whatever is present and skips otherwise
  <SITE>/
    siteInfo.toml        site facts: sonics, heights, bearings, slope, tables
    scripts/             per-site raw_processing configs (tracked): thin
                         wrappers over raw_processing/ (card convert,
                         fast/slow process) with the site's column maps
                         and date spans
    raw/
      <table>/           as logged: TOA5 .dat / logger CSV, one folder per
                         raw table; several folders when the heights of one
                         UTESpac table are logged in separate files and must
                         be concatenated (e.g. one IRGASON per logger table)
    utespac/             formatted 48-h inputs written by raw_processing:
                         <PREFIX>_<table>_header.dat + <PREFIX>_<table>_<d1>000000_<d2>000000.txt
    output/              pipeline products, netCDF only: <Site>_<avgPer>minAvg_<PF>_<Det>_<date>.nc
                         (the utespac-run-2 run file, utespac.run_io) / *_hf_*.nc (the
                         high-frequency products, utespac.export_hf) / csv/ on request;
                         .pkl files are products of runs before 2026-08-22 (get_data
                         still reads them with fmt="pkl")
    PFinfo.json          global planar-fit coefficients as a record table
                         (utespac.pf_info.PFTable, written by find_global_pf; a
                         PFinfo.pkl left by an earlier run is still read when no
                         JSON exists)
    eddypro/             reference runs (EddyPro project, metadata, outputs)
    <other reference>/   anything else used for validation, named by source
```

Site folder names are the bare site id (`MySite`, not `siteMySite_...`);
date ranges belong to the processed file names, not the folder. The
pipeline discovers a site by the presence of `siteInfo.toml` (or legacy
`siteInfo.py`). Legacy `site*` folders at the repo root still resolve,
but none with tracked files remain.

`raw/` is read-only: `raw_processing` scripts read it and write to
`utespac/`; nothing in the pipeline writes into `raw/`.
