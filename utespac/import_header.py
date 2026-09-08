"""importHeader – read a UTESpac header .dat file and its units sibling."""

import os
import re
from typing import List


def _fields(filename: str) -> List[str]:
    """Fields of a single-line, comma-delimited file, unquoted and stripped."""
    with open(filename, "r") as fh:
        line = fh.readline()
    return [f.strip().strip('"').strip("'") for f in line.split(",")]


def units_file(filename: str) -> str:
    """Path of the ``_units.dat`` sibling of a header file."""
    head, tail = os.path.split(filename)
    return os.path.join(head, tail.replace("_header", "_units", 1))


def import_units(filename: str) -> List[str]:
    """Read the declared unit of each column of a header file.

    Parameters
    ----------
    filename : str
        Path of the ``<table>_header.dat`` file, not of the units file.

    Returns
    -------
    list of str
        One entry per header column, in header order; "" where no unit is
        declared and for every column when the ``<table>_units.dat``
        sibling is absent (tables written before the units file existed).
    """
    names = _fields(filename)
    path = units_file(filename)
    if not os.path.exists(path):
        return [""] * len(names)
    units = _fields(path)
    units = units[: len(names)] + [""] * max(0, len(names) - len(units))
    return units


def import_header(filename: str) -> List[List]:
    """Read a single-line, comma-delimited header file.

    Returns a list of two sublists:
        result[0] : list of variable name strings (row 0 in MATLAB)
        result[1] : list of sensor heights (float) extracted from each name,
                    or None when no numeric height is found.
    """
    names = _fields(filename)

    heights = []
    for name in names:
        # The last digit/decimal run in the name is the height (e.g. "Ux_51.5")
        runs = re.findall(r"[\d.]+", name)
        try:
            heights.append(float(runs[-1]) if runs else None)
        except ValueError:
            heights.append(None)

    return [names, heights]
