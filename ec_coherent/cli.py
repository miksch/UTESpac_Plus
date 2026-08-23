"""Run the ec_coherent modules over one or more ``utespac-hf-1`` files.

    python -m ec_coherent.cli data/VAC001/output/VAC001_hf_GPF_ConstDet_2023_07_06.nc
    python -m ec_coherent.cli <files...> --modules spectra --records 0-5 --config my.toml
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Sequence

from . import ramps, spectra
from .config import ECConfig
from .io import init_output, open_hf, output_path, write_group

log = logging.getLogger("ec_coherent")

MODULES = {"spectra": (spectra.GROUP, spectra.run), "ramps": (ramps.GROUP, ramps.run)}


def _parse_records(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_file(path: str, cfg: ECConfig, records: Optional[Sequence[int]] = None,
             out: Optional[str] = None) -> str:
    """Run ``cfg.modules`` on one HF file; returns the analysis file path."""
    with open_hf(path) as hf:
        out = out or output_path(path, cfg.output_suffix)
        init_output(out, hf, cfg)
        for name in cfg.modules:
            if name not in MODULES:
                raise ValueError(f"module {name!r} not available (have {sorted(MODULES)})")
            group, fn = MODULES[name]
            log.info("%s: %s", hf.site_id, name)
            ds = fn(hf, cfg, records=records)
            write_group(out, group, ds)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ec_coherent", description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="utespac-hf-1 netCDF files")
    ap.add_argument("--modules", help="comma-separated subset of the configured modules")
    ap.add_argument("--records", help="record indices, e.g. 0-5,10")
    ap.add_argument("--config", help="ec_coherent TOML (default: cwd config/ or packaged)")
    ap.add_argument("--out", help="output path (single input only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    kw: Dict = {}
    if args.modules:
        kw["modules"] = tuple(m.strip() for m in args.modules.split(","))
    cfg = ECConfig.from_config(args.config, **kw)
    if args.out and len(args.files) > 1:
        ap.error("--out needs a single input file")
    for f in args.files:
        out = run_file(f, cfg, records=_parse_records(args.records), out=args.out)
        log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
