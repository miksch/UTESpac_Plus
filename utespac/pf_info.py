"""Global planar-fit coefficients as a labeled record table (``PFinfo.json``).

The legacy ``PFinfo.pkl`` is a nested dict whose keys encode the data
(``cm_1085`` → ``day_739404to739419`` → ``degrees_0_to_0`` → ``[b0, b1, b2]``).
:class:`PFTable` carries the same content as explicit records — height,
date window, wind-direction sector, coefficients — and converts both ways,
so ``find_global_pf`` can persist the table and the rotation stage can keep
reading the dict until it is migrated.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from .campbell_date import MATLAB_EPOCH

FORMAT = "utespac-pfinfo-1"
_DAY_RE = re.compile(r"day_(\d+)to(\d+)")
_SECTOR_RE = re.compile(r"degrees_([\d.]+)_to_([\d.]+)")


def _datenum_to_date(d: int) -> date:
    return (datetime(1970, 1, 1) + timedelta(days=int(d) - int(MATLAB_EPOCH))).date()


def _date_to_datenum(d: date) -> int:
    return (d - date(1970, 1, 1)).days + int(MATLAB_EPOCH)


@dataclass(frozen=True)
class PFRecord:
    """One planar fit: a sonic height, a date window (inclusive days) and a
    wind-direction sector; ``sector_lo == sector_hi`` means all directions."""
    height: float          # [m]
    date_start: str        # ISO date
    date_end: str          # ISO date
    sector_lo: float       # [deg]
    sector_hi: float       # [deg]
    b0: float
    b1: float
    b2: float


@dataclass
class PFTable:
    """Planar-fit coefficients with explicit dimensions."""
    records: List[PFRecord] = field(default_factory=list)
    site: Optional[str] = None
    info_string: Optional[list] = None   # legacy ``infoString`` columns, kept verbatim

    @property
    def heights(self) -> List[float]:
        return sorted({r.height for r in self.records})

    # ── legacy dict ─────────────────────────────────────────────────────────
    @classmethod
    def from_legacy(cls, pf_info: Dict, site: Optional[str] = None) -> "PFTable":
        """Build from the ``cm_/day_/degrees_`` dict written by ``find_global_pf``."""
        records = []
        for cm_key, windows in pf_info.items():
            if cm_key == "infoString":
                continue
            height = int(cm_key.split("_", 1)[1]) / 100.0
            for day_key, sectors in windows.items():
                m = _DAY_RE.fullmatch(day_key)
                if not m:
                    raise ValueError(f"unrecognised PFinfo date key {day_key!r}")
                d0, d1 = _datenum_to_date(m.group(1)), _datenum_to_date(m.group(2))
                for sec_key, coef in sectors.items():
                    s = _SECTOR_RE.fullmatch(sec_key)
                    if not s:
                        raise ValueError(f"unrecognised PFinfo sector key {sec_key!r}")
                    b0, b1, b2 = (float(c) for c in np.asarray(coef, dtype=float)[:3])
                    records.append(PFRecord(height, d0.isoformat(), d1.isoformat(),
                                            float(s.group(1)), float(s.group(2)), b0, b1, b2))
        return cls(records, site=site, info_string=pf_info.get("infoString"))

    def to_legacy(self) -> Dict:
        """The nested dict the rotation stage reads (``infoString`` included)."""
        out: Dict = {}
        for r in self.records:
            cm_key = f"cm_{round(r.height * 100)}"
            day_key = (f"day_{_date_to_datenum(date.fromisoformat(r.date_start))}"
                       f"to{_date_to_datenum(date.fromisoformat(r.date_end))}")
            sec_key = f"degrees_{r.sector_lo:g}_to_{r.sector_hi:g}"
            out.setdefault(cm_key, {}).setdefault(day_key, {})[sec_key] = \
                np.array([r.b0, r.b1, r.b2], dtype=float)
        if self.info_string is not None:
            out["infoString"] = self.info_string
        else:
            from .find_global_pf import _build_info_string
            out["infoString"] = _build_info_string(out)
        return out

    # ── lookup ──────────────────────────────────────────────────────────────
    def coefficients(self, height: float, when, direction: Optional[float] = None,
                     tol: float = 0.01) -> Optional[PFRecord]:
        """Record covering *height*, the day of *when* (datetime64, datetime or
        datenum) and *direction* [deg]; None when nothing matches."""
        if isinstance(when, (int, float, np.floating)):
            day = _datenum_to_date(int(np.floor(when)))
        elif isinstance(when, np.datetime64):
            day = when.astype("datetime64[D]").astype(date)
        else:
            day = when.date() if isinstance(when, datetime) else when
        for r in self.records:
            if abs(r.height - height) > tol:
                continue
            if not (date.fromisoformat(r.date_start) <= day <= date.fromisoformat(r.date_end)):
                continue
            if r.sector_lo == r.sector_hi or direction is None:
                return r
            if r.sector_hi > r.sector_lo:
                if r.sector_lo < direction < r.sector_hi:
                    return r
            elif direction > r.sector_lo or direction < r.sector_hi:
                return r
        return None

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict:
        d = {"format": FORMAT, "site": self.site,
             "records": [asdict(r) for r in self.records]}
        if self.info_string is not None:
            d["info_string"] = self.info_string
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "PFTable":
        if d.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} document: format={d.get('format')!r}")
        return cls([PFRecord(**r) for r in d["records"]], site=d.get("site"),
                   info_string=d.get("info_string"))

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path) -> "PFTable":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
