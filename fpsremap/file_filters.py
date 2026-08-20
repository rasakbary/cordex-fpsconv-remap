"""
Optional filters for choosing how a model's files are merged.

The filters:

  --complete-years: keep only calendar years that are fully covered. Useful when there are stray months in the begining/end of 
  a model time slice and you don't want to include them in the final merged file (i.e., you want to have only complete years). 

  --far-future: in the archive rcp85 scenario has two timeslices 'near future (2040-2049)' and 'end of century (2090-2099)', so this 
  filter is only for my use case which focused only on 'end of century' scenario. keep only the later of two blocks of years.

  --drop-reversed: for very few cases drop files whose end date comes before their start date.

Everything works from filenames alone and needs the usual CORDEX date range
at the end of the name:

    <anything>_YYYYMMDD-YYYYMMDD.nc

For a more general case use consider that: these types of names will NOT be recognised `_01011996_31_03_1996`, `_199601-199612`, 
`_1996010100-1996123123` and `_19960101_19961231`. If your files look like that, change `_DATE_RANGE_RE` and  `_parse_date`.
"""

from __future__ import annotations

import calendar
import glob
import os
import re
from dataclasses import dataclass
from datetime import date

from .logging_utils import log

_DATE_RANGE_RE = re.compile(r"_(\d{8})-(\d{8})(?:\D|$)")

YEARLY_STARTS = {"0101"}
MONTHLY_STARTS = {f"{m:02d}01" for m in range(1, 13)}


@dataclass
class FileSpan:
    path: str
    start: date | None
    end: date | None
    d0: str
    d1: str

    @property
    def year(self) -> int | None:
        return self.start.year if self.start else None

    @property
    def label(self) -> str:
        return f"{self.d0}-{self.d1}" if self.d0 else os.path.basename(self.path)


def _parse_date(s: str) -> date | None:
    """Parse YYYYMMDD, clamping impossible days to month end.
    consider that 360-day-calendar models produce filenames like `20010230`.
    """
    try:
        y, m, d = int(s[:4]), int(s[4:6]), int(s[6:8])
        try:
            return date(y, m, d)
        except ValueError:
            return date(y, m, min(d, calendar.monthrange(y, m)[1]))
    except Exception:                                    # noqa: BLE001
        return None


def parse_span(path: str) -> FileSpan:
    """Extract the date range from a CORDEX filename."""
    m = _DATE_RANGE_RE.search(os.path.basename(path))
    if not m:
        return FileSpan(path, None, None, "", "")
    d0, d1 = m.group(1), m.group(2)
    return FileSpan(path, _parse_date(d0), _parse_date(d1), d0, d1)


def find_files(folder: str, pattern_prefix: str) -> list[str]:
    """All .nc files in `folder` whose basename starts with `pattern_prefix`.
    """
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*.nc"))
        if os.path.basename(f).startswith(pattern_prefix)
    )


# Filters
def drop_reversed_spans(files: list[str]) -> list[str]:
    """Drop files whose filename end-date precedes its start-date."""
    kept = []
    for fp in files:
        sp = parse_span(fp)
        if sp.d0 and sp.d1 and sp.d1 < sp.d0:
            log(f"   [SKIP] reversed date range {sp.label}: {os.path.basename(fp)}")
            continue
        kept.append(fp)
    return kept


def keep_complete_calendar_years(files: list[str]) -> list[str]:
    """Keep only files belonging to fully covered calendar years."""
    by_year: dict[int, list[FileSpan]] = {}
    unparseable: list[str] = []
    undated: list[str] = []

    for fp in files:
        sp = parse_span(fp)
        if sp.start is None or sp.end is None:
            undated.append(os.path.basename(fp))
            unparseable.append(fp)
            continue
        if sp.start.year != sp.end.year:
            log(f"   [WARN] file spans multiple calendar years, keeping without "
                f"year-grouping: {os.path.basename(fp)}")
            unparseable.append(fp)
            continue
        by_year.setdefault(sp.start.year, []).append(sp)

    if undated:
        log(f"   [WARN] {len(undated)} file(s) carry no recognisable "
            f"_YYYYMMDD-YYYYMMDD range in their name. The complete-year check "
            f"cannot be applied to them; they are kept unchanged:")
        for name in undated[:5]:
            log(f"            {name}")
        if len(undated) > 5:
            log(f"            ... and {len(undated) - 5} more")

    kept = list(unparseable)
    for year in sorted(by_year):
        spans = by_year[year]
        counts: dict[int, int] = {}
        for sp in spans:
            for m in range(sp.start.month, sp.end.month + 1):
                counts[m] = counts.get(m, 0) + 1

        if counts == {m: 1 for m in range(1, 13)}:
            kept.extend(sp.path for sp in spans)
            continue

        missing = [m for m in range(1, 13) if m not in counts]
        duplicated = sorted(m for m, n in counts.items() if n > 1)
        why = []
        if missing:
            why.append(f"missing month(s) {missing}")
        if duplicated:
            why.append(f"month(s) {duplicated} covered more than once")
        names = ", ".join(os.path.basename(sp.path) for sp in spans)
        log(f"   [DROP] incomplete year {year} ({len(spans)} file(s), "
            f"{'; '.join(why)}): {names}")
    return sorted(kept)


def keep_end_of_century(files: list[str], threshold_year: int = 2070) -> list[str]:
    """Keep files whose midpoint year is at or after `threshold_year`."""
    kept = []
    for fp in files:
        sp = parse_span(fp)
        if sp.start is None:
            continue
        y0 = sp.start.year
        y1 = sp.end.year if sp.end else y0
        if (y0 + y1) / 2.0 >= threshold_year:
            kept.append(fp)
    dropped = len(files) - len(kept)
    if dropped:
        log(f"   end-of-century filter (midpoint year >= {threshold_year}): "
            f"kept {len(kept)}, dropped {dropped}")
    return sorted(kept)


def keep_far_future_by_gap(files: list[str], min_gap_years: int = 10) -> list[str]:
    """Split on the largest year gap and keep the later block."""
    spans = []
    for fp in sorted(files):
        sp = parse_span(fp)
        if sp.start is None:
            log(f"   [WARN] cannot parse date range from: {os.path.basename(fp)}")
            continue
        spans.append(sp)
    if not spans:
        return []

    spans.sort(key=lambda s: (s.year, s.d0))
    years = sorted({s.year for s in spans})
    if len(years) < 2:
        return [s.path for s in spans]

    gap, left, right = max(
        ((b - a, a, b) for a, b in zip(years[:-1], years[1:])), key=lambda t: t[0]
    )

    if gap < min_gap_years:
        log(f"   [INFO] no clear near/far split (largest gap {gap}y < "
            f"{min_gap_years}y). Keeping ALL {len(spans)} files.")
        return [s.path for s in spans]

    far = [s for s in spans if s.year >= right]
    near = [s for s in spans if s.year < right]
    log(f"   far-future split at start_year >= {right} "
        f"(largest gap {gap}y, between {left} and {right})")
    if near:
        log(f"      near-future {min(s.year for s in near)}-"
            f"{max((s.end.year if s.end else s.year) for s in near)}: "
            f"{len(near)} file(s) dropped")
    log(f"      far-future  {min(s.year for s in far)}-"
        f"{max((s.end.year if s.end else s.year) for s in far)}: "
        f"{len(far)} file(s) kept")

    far.sort(key=lambda s: (s.year, s.d0))
    return [s.path for s in far]


def diagnose(files: list[str], period: str = "historical",
             min_gap_years: int = 10) -> list[str]:
    findings: list[str] = []
    if not files:
        return findings

    spans = [parse_span(f) for f in files]

    undated = [s for s in spans if s.start is None]
    if undated:
        findings.append(
            f"{len(undated)} file(s) have no recognisable "
            f"_YYYYMMDD-YYYYMMDD range in their name, so no date-based check "
            f"applies to them (e.g. {os.path.basename(undated[0].path)})"
        )

    reversed_ = [s for s in spans if s.d0 and s.d1 and s.d1 < s.d0]
    if reversed_:
        findings.append(
            f"{len(reversed_)} file(s) have a reversed date range "
            f"(end before start), which usually means a corrupt or duplicated "
            f"stub: {', '.join(os.path.basename(s.path) for s in reversed_[:3])}"
            + (" ..." if len(reversed_) > 3 else "")
            + "  [--drop-reversed removes these]"
        )

    dated = [s for s in spans
             if s.start is not None and s.end is not None
             and s.start.year == s.end.year
             and s.end.month >= s.start.month]
    by_year: dict[int, list[FileSpan]] = {}
    for s in dated:
        by_year.setdefault(s.start.year, []).append(s)

    partial = []
    for year in sorted(by_year):
        months = set()
        for s in by_year[year]:
            months.update(range(s.start.month, s.end.month + 1))
        if not months or months == set(range(1, 13)):
            continue
        covered = sorted(months)
        partial.append(f"{year} (months {covered[0]}-{covered[-1]}, "
                       f"{len(covered)} of 12)")
    if partial:
        findings.append(
            f"{len(partial)} partially covered calendar year(s): "
            + "; ".join(partial[:4]) + (" ..." if len(partial) > 4 else "")
            + "  [--complete-years removes these]"
        )

    # Disjoint time blocks - the near/far future case.
    years = sorted({s.start.year for s in spans if s.start is not None})
    if len(years) > 1:
        gap, left, right = max(
            ((b - a, a, b) for a, b in zip(years[:-1], years[1:])),
            key=lambda t: t[0],
        )
        if gap >= min_gap_years:
            findings.append(
                f"the record is split into two blocks with a {gap}-year gap "
                f"between {left} and {right}; merging both gives one file with "
                f"a gap in the middle  [--far-future keeps only the later block]"
            )

    return findings


def apply_filters(files: list[str], *, period: str = "historical",
                  complete_years: bool = False,
                  far_future: bool = False,
                  drop_reversed: bool = False,
                  rcp85_eoc_threshold: int | None = None,
                  min_gap_years: int = 10) -> list[str]:
    """Apply whichever filters were explicitly requested.

    With no flags set this returns the input unchanged - the default is to use
    every file that was found.
      1. drop corrupt (reversed-span) files
      2. restrict to the wanted time block (rcp85 only)
      3. drop partial years, so a partially transferred block is caught
    """
    if not files:
        return []

    if drop_reversed:
        files = drop_reversed_spans(files)

    if far_future and period == "rcp85":
        if rcp85_eoc_threshold is not None:
            files = keep_end_of_century(files, threshold_year=rcp85_eoc_threshold)
        else:
            files = keep_far_future_by_gap(files, min_gap_years=min_gap_years)

    if complete_years:
        files = keep_complete_calendar_years(files)

    return sorted(files)


def summarise(files: list[str]) -> str:
    """One-line coverage summary for logs and reports."""
    if not files:
        return "no files"
    spans = [parse_span(f) for f in files]
    years = [s.year for s in spans if s.year is not None]
    if not years:
        return f"{len(files)} file(s), date range unknown"
    ends = [(s.end.year if s.end else s.year) for s in spans if s.year is not None]
    return f"{len(files)} file(s), {min(years)}-{max(ends)}"
