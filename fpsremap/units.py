"""
Check and convert the units of the data variable.

1. For each model units are read from every file separately. Note that a few models label one file as a flux in kg m-2 s-1 and 
   the next as mm/h.

2. For some model few files hold a wrong unit label; those are listed in `units_override` in the registry. This need to be confirmed only after 
   checking the data itself. 

"""

from __future__ import annotations

from .nc_metadata import read_var_units
from .logging_utils import log


TARGET_UNITS = {"pr": "mm/h", "tas": "K"}

_PR_FLUX = {"kg m-2 s-1", "kgm-2s-1", "kg/m2/s", "kg m**-2 s**-1"}
_PR_ACCUM = {"kg m-2", "kgm-2", "kg/m2"}
_PR_RATE = {"mm/h", "mm h-1", "mm/hr", "mm hr-1"}
_TAS_KELVIN = {"k", "kelvin", "degk", "degrees_k"}
_TAS_CELSIUS = {"degc", "degree_c", "celsius", "°c", "degrees_c"}


def normalise(units: str) -> str:
    """Collapse whitespace so 'kg  m-2   s-1' compares equal to 'kg m-2 s-1'."""
    return " ".join(str(units).strip().split())


def pr_factor(units: str, dt_hours: float | None = 1.0) -> float:
    """Multiplicative factor converting `pr` in `units` to mm/h."""
    u = normalise(units)
    ul = u.lower()

    if ul in _PR_FLUX:
        # kg m-2 s-1 == mm/s -> mm/h
        return 3600.0

    if ul in _PR_ACCUM:
        if dt_hours is None:
            raise ValueError(
                f"pr units {u!r} is an accumulation per timestep, but the "
                f"timestep could not be determined. Cannot convert safely."
            )
        if abs(dt_hours - 1.0) > 1e-6:
            raise ValueError(
                f"pr units {u!r} is an accumulation over a {dt_hours:g}-hour "
                f"timestep. The mm/h conversion used here assumes hourly "
                f"data. Divide by {dt_hours:g} explicitly."
            )
        return 1.0

    if ul in _PR_RATE:
        return 1.0

    raise ValueError(f"Unsupported pr units: {units!r}")


def tas_factor(units: str) -> tuple[float, str]:
    """`tas` is kept in its native units; returns (factor, target_units)."""
    u = normalise(units)
    ul = u.lower()
    if ul in _TAS_KELVIN:
        return 1.0, "K"
    if ul in _TAS_CELSIUS:
        return 1.0, u
    raise ValueError(f"Unsupported tas units: {units!r} (expected K)")


def factor_and_target(units: str, variable: str,
                      dt_hours: float | None = 1.0) -> tuple[float, str]:
    """Return (scale_factor, target_units) for one file's units string."""
    if variable == "pr":
        return pr_factor(units, dt_hours), "mm/h"
    if variable == "tas":
        return tas_factor(units)
    raise ValueError(
        f"Unsupported variable: {variable!r}. Add it to fpsremap/units.py "
        f"with an explicit conversion rule."
    )


def effective_units(nc_file: str, variable: str,
                    override: str | None = None) -> tuple[str, str]:
    """Read a file's units, applying a registry override if one exists.

    Returns (effective_units, note).
    """
    raw = read_var_units(nc_file, variable)
    if override:
        return override, f"'{raw}' [OVERRIDE -> '{override}']"
    return raw, f"'{raw}'"


def plan_conversions(files: list[str], variable: str,
                     override: str | None = None,
                     dt_hours: float | None = 1.0) -> list[dict]:
    """Decide, per file, whether and how to convert."""
    plan = []
    targets = set()
    for idx, fp in enumerate(files):
        eff, note = effective_units(fp, variable, override)
        factor, target = factor_and_target(eff, variable, dt_hours)
        targets.add(target)
        plan.append({
            "index": idx,
            "path": fp,
            "raw_units": eff,
            "note": note,
            "factor": factor,
            "target_units": target,
            "needs_scaling": factor != 1.0,
        })

    if len(targets) > 1:
        raise ValueError(
            f"Inconsistent target units across files: {sorted(targets)}. "
            f"These cannot be merged into one dataset."
        )
    return plan


def log_plan(plan: list[dict], variable: str) -> None:
    n_scaled = sum(1 for p in plan if p["needs_scaling"])
    log(f"   units: checked {len(plan)} file(s) individually; "
        f"{n_scaled} need scaling, {len(plan) - n_scaled} pass through")
    seen: set[tuple[str, float]] = set()
    for p in plan:
        key = (p["raw_units"], p["factor"])
        if key in seen:
            continue
        seen.add(key)
        log(f"      {p['note']}  ->  x{p['factor']:g}  ->  '{p['target_units']}'")
