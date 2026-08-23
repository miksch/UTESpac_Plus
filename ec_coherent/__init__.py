"""ec_coherent -- coherent-structure analysis of UTESpac high-frequency netCDF.

Consumes the ``utespac-hf-1`` files written by :mod:`utespac.export_hf`
and writes one analysis netCDF per input with a group per module. Plan:
``testbed/2026-08-12_ec_coherent_gameplan.md``; science in
``library/writeups/ec_*.md``. Imports from ``utespac`` are allowed, never
the reverse.
"""

from .config import ECConfig
from .io import HFFile, Window, open_hf, iter_windows, write_group

__all__ = ["ECConfig", "HFFile", "Window", "open_hf", "iter_windows", "write_group"]
