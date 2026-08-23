# data -- the per-site data tree

One folder per site. Everything the pipeline reads or writes for a site
lives under it; the repo root holds code only. Only this README, each
site's `siteInfo.toml`, and the UTESpac header files are tracked -- raw,
processed, and reference data stay local (see `.gitignore`).

```
data/
  <SITE>/
    siteInfo.toml        site facts: sonics, heights, bearings, slope, tables
    raw/
      <table>/           as logged: TOA5 .dat / logger CSV, one folder per
                         raw table; several folders when the heights of one
                         UTESpac table are logged in separate files and must
                         be concatenated (e.g. one IRGASON per logger table)
    utespac/             formatted 48-h inputs written by raw_processing:
                         <PREFIX>_<table>_header.dat + <PREFIX>_<table>_<d1>000000_<d2>000000.txt
    output/              pipeline products: .pkl / csv/ / .nc (labeled netCDF twin of
                         each averaged pickle, utespac.labeled) / *_hf_*.nc (high-
                         frequency netCDF from python -m utespac.export_hf)
    PFinfo.json          global planar-fit coefficients as a record table
                         (utespac.pf_info.PFTable; written by find_global_pf beside
    PFinfo.pkl           the legacy dict; the JSON is read first when both exist)
    eddypro/             reference runs (EddyPro project, metadata, outputs)
    <other reference>/   anything else used for validation, named by source
```

Site folder names are the bare site id (`VAC001`, not `siteVAC001_...`);
date ranges belong to the processed file names, not the folder. The
pipeline discovers a site by the presence of `siteInfo.toml` (or legacy
`siteInfo.py`). Legacy `site*` folders at the repo root still resolve,
but none with tracked files remain; `VAC001` is the site in the repo.

`raw/` is read-only: `raw_processing` scripts read it and write to
`utespac/`; nothing in the pipeline writes into `raw/`.
