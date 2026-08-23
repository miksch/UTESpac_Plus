"""Packaged processing config for UTESpac.

One commented TOML per stage ships with the package — ``run.toml`` (run-level
and output options plus the sensor-name templates), ``qc.toml`` (the
Vickers & Mahrt conditioning tests), ``pf.toml`` (planar fit) and
``flux.toml`` (flux-stage knobs). The dataclasses in
:mod:`utespac.run_config` are the schema and carry the embedded defaults; a
TOML mirrors them with comments and overrides them when present, so a
partial file is fine. Site facts (heights, bearings, slope geometry,
elevation, latitude) stay in each site's ``siteInfo.toml`` and never live
here.

Resolution order, shared by every ``from_config``: per-call keyword >
explicit dataclass/dict > ``config/<name>.toml`` in the working directory >
packaged TOML > dataclass default. Mirrors ``dopli.config``.
"""

import tomllib
from importlib.resources import files
from pathlib import Path


def load_toml(name):
    """Parsed packaged TOML ``name`` (with or without ``.toml``), nested dict."""
    if not name.endswith(".toml"):
        name += ".toml"
    resource = files(__package__) / name
    if not resource.is_file():
        raise FileNotFoundError(f"no packaged utespac config {name!r}")
    with resource.open("rb") as fh:
        return tomllib.load(fh)


def flatten_sections(raw):
    """Merge one level of ``[section]`` tables into a flat ``{key: value}``.

    Sections are cosmetic grouping; a key must appear under exactly one
    section. Nested tables below a section are kept as dicts.
    """
    flat = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    return flat


def resolve(name, config=None, flatten=True):
    """Resolve stage config ``name`` to a dict.

    ``config`` is ``None`` (auto-load: ``config/<name>.toml`` in the cwd wins
    over the packaged file), a path to a TOML file, or a dict (returned as a
    shallow copy). ``flatten=False`` keeps the ``[section]`` nesting, for
    files whose sections carry meaning (``qc.toml``, ``run.toml``).
    """
    if config is None:
        cwd_override = Path.cwd() / "config" / f"{name}.toml"
        if cwd_override.is_file():
            with open(cwd_override, "rb") as fh:
                raw = tomllib.load(fh)
        else:
            raw = load_toml(name)
    elif isinstance(config, (str, Path)):
        with open(config, "rb") as fh:
            raw = tomllib.load(fh)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raise TypeError("config must be None, a path to a TOML file, or a dict; "
                        f"got {type(config).__name__}")
    return flatten_sections(raw) if flatten else raw
