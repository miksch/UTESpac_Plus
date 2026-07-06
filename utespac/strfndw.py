"""Wildcard string search in a list of strings (strfndw MATLAB equivalent)."""

import re
from typing import List


def strfndw(array: List[str], exp_str: str) -> List[int]:
    """Return 0-based indices of elements in *array* that match *exp_str*.

    Supports ``*`` (zero or more characters) and ``?`` (single character)
    wildcards, exactly like the MATLAB version.

    Returns
    -------
    idx : list of int
        0-based indices of matching elements.
    """
    # Convert wildcard pattern to regex (escape . before expanding * and ?)
    pattern = "^" + exp_str.replace(".", r"\.").replace("*", ".{0,}").replace("?", ".") + "$"
    compiled = re.compile(pattern, re.IGNORECASE)
    return [i for i, s in enumerate(array) if compiled.match(str(s))]
