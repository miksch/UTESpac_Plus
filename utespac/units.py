"""Declared input units, and the conversion to the pipeline's internal ones.

The logger tables declare a unit per column (TOA5 row 2, carried to
``<table>_units.dat`` by :mod:`raw_processing.toa5_tower` and read back by
:func:`utespac.import_header.import_units` into ``Sensor.units``).
:func:`convert` turns a series in a declared unit into the unit the flux
computation works in -- temperature deg C, relative humidity percent,
pressure kPa, vapour density g m-3, CO2 density mg m-3.

Where a table predates the units file the unit is unknown, and the
magnitude heuristics that ``fluxes.m`` used stand in: a temperature whose
median exceeds 250 is Kelvin, a pressure whose median exceeds 200 is hPa,
relative humidity is percent. Each such assumption is logged once per run
and per sensor (``reset_warnings`` at the start of a run).
"""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("utespac")

#: Unit the flux computation works in, per quantity.
INTERNAL = {
    "temperature": "deg C",
    "humidity": "%",
    "pressure": "kPa",
    "h2o_density": "g m-3",
    "co2_density": "mg m-3",
}

# {quantity: {normalized unit: (scale, offset) onto the internal unit}}
_CONVERSIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "temperature": {"degc": (1.0, 0.0), "c": (1.0, 0.0),
                    "k": (1.0, -273.15), "kelvin": (1.0, -273.15)},
    "humidity": {"%": (1.0, 0.0), "percent": (1.0, 0.0),
                 "fraction": (100.0, 0.0), "1": (100.0, 0.0)},
    "pressure": {"kpa": (1.0, 0.0), "hpa": (0.1, 0.0), "mbar": (0.1, 0.0),
                 "mb": (0.1, 0.0), "pa": (0.001, 0.0)},
    "h2o_density": {"gm-3": (1.0, 0.0), "kgm-3": (1000.0, 0.0),
                    "mgm-3": (0.001, 0.0)},
    "co2_density": {"mgm-3": (1.0, 0.0), "gm-3": (1000.0, 0.0),
                    "kgm-3": (1.0e6, 0.0), "ugm-3": (0.001, 0.0)},
}


def normalize(unit: Optional[str]) -> str:
    """Canonical spelling of a declared unit (case, spacing, ``g/m^3`` forms)."""
    s = "" if unit is None else str(unit).strip().lower()
    for sign in ("°", "º"):
        s = s.replace(sign, "deg")
    s = s.replace("^", "").replace(" ", "").replace("_", "")
    s = s.replace("/m3", "m-3").replace("/m2", "m-2")
    return s


def _degC_or_kelvin(values: np.ndarray) -> np.ndarray:
    """deg C from a series that may be Kelvin (median above 250)."""
    return values - 273.15 if np.nanmedian(values) > 250 else values


def _kelvin_or_degC(values: np.ndarray) -> np.ndarray:
    """K from a series that may be deg C (median below 200)."""
    return values + 273.15 if np.nanmedian(values) < 200 else values


def _kPa_or_hPa(values: np.ndarray) -> np.ndarray:
    """kPa from a series that may be hPa (median above 200)."""
    return values / 10.0 if np.nanmedian(values) > 200 else values


def _as_is(values: np.ndarray) -> np.ndarray:
    return values


# {(quantity, normalized target): (heuristic, what it assumes)}
_FALLBACKS: Dict[Tuple[str, str], Tuple[Callable, str]] = {
    ("temperature", "degc"): (_degC_or_kelvin, "read as deg C, as K above a median of 250"),
    ("temperature", "k"): (_kelvin_or_degC, "read as K, as deg C below a median of 200"),
    ("humidity", "%"): (_as_is, "read as percent"),
    ("pressure", "kpa"): (_kPa_or_hPa, "read as kPa, as hPa above a median of 200"),
    ("h2o_density", "gm-3"): (_as_is, "read as g m-3"),
    ("co2_density", "mgm-3"): (_as_is, "read as mg m-3"),
}

_warned = set()


def reset_warnings() -> None:
    """Forget which fallback assumptions have been logged; call once per run."""
    _warned.clear()


def _name(sensor) -> str:
    if sensor is None:
        return "<unnamed>"
    column = getattr(sensor, "column", None)
    if column is None:
        return str(sensor)
    return f"{getattr(sensor, 'table', '?')}:{column}"


def convert(values, unit: Optional[str], quantity: str, target: Optional[str] = None,
            *, sensor=None, warn: bool = True) -> np.ndarray:
    """Convert a series to the unit the pipeline works in.

    Parameters
    ----------
    values : array_like
        The series as the logger recorded it.
    unit : str or None
        Unit declared for the column (``Sensor.units``); "" or None when the
        table carries no units file.
    quantity : {'temperature', 'humidity', 'pressure', 'h2o_density', 'co2_density'}
        What the series measures.
    target : str, optional
        Unit to convert to; defaults to ``INTERNAL[quantity]`` (the flux
        computation also asks for temperature in ``'K'``).
    sensor : Sensor or str, optional
        Named in the fallback warning, and the key it is deduplicated by.
    warn : bool, optional
        Log the fallback assumption (default True).

    Returns
    -------
    ndarray
        A new float array in *target*. With no declared unit the magnitude
        heuristic for ``(quantity, target)`` is applied instead and the
        assumption is logged once per run and per sensor.

    Raises
    ------
    ValueError
        For an unknown *quantity* or *target*, or a declared unit that is
        not a known unit of that quantity.
    """
    if quantity not in _CONVERSIONS:
        raise ValueError(f"unknown quantity {quantity!r}")
    table = _CONVERSIONS[quantity]
    tgt = normalize(target if target is not None else INTERNAL[quantity])
    if tgt not in table:
        raise ValueError(f"{quantity}: cannot convert to {target!r}")

    out = np.array(values, dtype=float)
    declared = normalize(unit)

    if not declared:
        fallback, assumption = _FALLBACKS.get((quantity, tgt), (_as_is, "read as given"))
        key = (_name(sensor), quantity, tgt)
        if warn and key not in _warned:
            _warned.add(key)
            log.warning(f"{_name(sensor)}: no unit declared for {quantity}; {assumption}.")
        return fallback(out)

    if declared not in table:
        raise ValueError(f"{quantity}: unknown unit {unit!r} "
                         f"(known: {', '.join(sorted(table))})")
    if declared == tgt:
        return out

    scale, offset = table[declared]          # declared -> internal
    scale_t, offset_t = table[tgt]           # target   -> internal
    return (out * scale + offset - offset_t) / scale_t
