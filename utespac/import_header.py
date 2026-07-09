"""importHeader – read a UTESpac header .dat file."""

import re
from typing import List


def import_header(filename: str) -> List[List]:
    """Read a single-line, comma-delimited header file.

    Returns a list of two sublists:
        result[0] : list of variable name strings (row 0 in MATLAB)
        result[1] : list of sensor heights (float) extracted from each name,
                    or None when no numeric height is found.
    """
    with open(filename, "r") as fh:
        line = fh.readline()

    # Strip surrounding quotes and whitespace from each field
    fields = [f.strip().strip('"').strip("'") for f in line.split(",")]
    names = fields

    heights = []
    for name in names:
        # The last digit/decimal run in the name is the height (e.g. "Ux_51.5")
        runs = re.findall(r"[\d.]+", name)
        try:
            heights.append(float(runs[-1]) if runs else None)
        except ValueError:
            heights.append(None)

    return [names, heights]
