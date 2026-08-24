"""The UTESpac processing pipeline as a function with a structured result.

``run_utespac(config, site, dates)`` drives find_files → find_instruments →
(find_global_pf) → per date: load_data → find_serial_date → the stages of
:mod:`utespac.stages` on a labeled :class:`~utespac.model.Run`
(``load_run`` → ``condition`` → ``motion`` → ``average`` → ``wind`` →
``rotate`` → ``flux`` → ``save_data``: the run netCDF, the HF netCDF, CSV), and returns a
:class:`RunResult` instead of printing. Interaction for the global planar
fit goes through a prompter (:mod:`utespac.prompts`); with none given the
selection is scripted (single sector, all dates).

The stages read the run facts from the legacy ``info`` dict
(``Run.site``); ``RunConfig.to_info()`` renders it here and ``find_files``
adds the site facts, so a legacy ``info`` dict is accepted too
(``run_utespac(info=..., template=...)``).
"""

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .run_config import RunConfig
from .prompts import PFPrompter, ScriptedPFSelection
from .model import Run
from .pf_info import PFTable
from . import stages

log = logging.getLogger("utespac")


@dataclass
class DateResult:
    """Outcome of one date row."""
    index: int                      # 1-based row in the date matrix
    files: List[Optional[str]]
    status: str                     # "ok" | "error"
    paths: Dict[str, str] = field(default_factory=dict)   # written products by kind
    error: Optional[str] = None
    run: Optional[Run] = None       # the labeled run, when run_utespac(keep_runs=True)


@dataclass
class RunResult:
    site: str
    info: Dict[str, Any]            # the resolved legacy info dict (site facts applied)
    table_names: List[str]
    pf_info: Optional[Dict]
    dates: List[DateResult]

    @property
    def ok(self) -> bool:
        return all(d.status == "ok" for d in self.dates)

    @property
    def n_ok(self) -> int:
        return sum(d.status == "ok" for d in self.dates)


def run_utespac(config: Optional[RunConfig] = None, *, site: str, dates="all",
                prompter: Optional[PFPrompter] = None,
                info: Optional[Dict[str, Any]] = None,
                template: Optional[Dict[str, str]] = None,
                keep_runs: bool = False) -> RunResult:
    """Run the pipeline for one site.

    Parameters
    ----------
    config : RunConfig
        Run configuration (``RunConfig.from_config(rootFolder=...)``). Either
        ``config`` or a legacy ``info`` dict must be given.
    site : str
        Site folder name (or bare legacy id) under ``rootFolder``.
    dates : "all" | int | list[int]
        Date rows to process (1-based), as ``find_files`` understands them.
    prompter : PFPrompter, optional
        Decisions for the global planar fit; default ``ScriptedPFSelection()``.
    info, template : legacy inputs
        A fully built ``info`` dict and sensor templates, for callers that
        have not moved to ``RunConfig``.
    keep_runs : bool
        Keep each date's labeled :class:`~utespac.model.Run` on its
        ``DateResult`` (high-frequency data included — memory per date).
    """
    if config is None and info is None:
        raise ValueError("run_utespac needs a RunConfig or a legacy info dict")
    if config is not None:
        if info is not None:
            raise ValueError("pass either config or info, not both")
        info = config.to_info()
        template = dict(config.template) if template is None else template
    elif template is None:
        raise ValueError("a legacy info dict needs its template dict as well")
    if prompter is None:
        prompter = ScriptedPFSelection()

    from .find_files import find_files
    from .find_instruments import find_instruments
    from .find_global_pf import find_global_pf
    from .load_data import load_data
    from .find_serial_date import find_serial_date
    from .save_data import save_data

    headers, data_files, table_names, info = find_files(info, site=site, dates=dates)
    sensor_info = find_instruments(headers, template, info)

    pf_info = None
    if info["PF"]["globalCalculation"] == "global":
        pf_info = find_global_pf(info, template, sensor_info, prompter=prompter)

    results: List[DateResult] = []
    for i, row in enumerate(data_files):
        try:
            data, data_info = load_data(row, i + 1, len(data_files), info, table_names)
            data, data_info, info = find_serial_date(data, data_info, info)
            # the labeled run (utespac.model / utespac.stages)
            run = stages.load_run(info, data, headers, table_names, sensor_info, data_info,
                                  pf_table=PFTable.from_legacy(pf_info) if pf_info else None)
            stages.condition(run, template)
            stages.motion(run)
            stages.average(run)
            stages.wind(run)
            stages.rotate(run)
            stages.flux(run)
            paths = save_data(run)
            results.append(DateResult(i + 1, list(row), "ok", paths=paths,
                                      run=run if keep_runs else None))
        except Exception as exc:   # one bad date must not stop the run
            log.error("Problem with date row %d: %s\n%s", i + 1, exc, traceback.format_exc())
            results.append(DateResult(i + 1, list(row), "error", error=repr(exc)))

    log.info("Processing complete: %d of %d date rows written.",
             sum(r.status == "ok" for r in results), len(results))
    return RunResult(site=info["siteFolder"], info=info, table_names=table_names,
                     pf_info=pf_info, dates=results)
