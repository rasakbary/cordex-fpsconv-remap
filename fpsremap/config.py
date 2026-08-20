"""
Read the YAML settings and fill in the paths.

Every path, model name and grid definition lives in config/*.yaml.

This module turns the YAML into plain Python objects and substitutes the
{archive} / {output} / {scratch} / {VARIABLE} / {PERIOD} placeholders.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "config")

VALID_ENSEMBLES = ("cpm", "rcm")

def _load_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            f"Expected it under {CONFIG_DIR}. If you moved the config folder, "
            f"pass --config-dir."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


# Paths
@dataclass
class Paths:
    archive: str
    output: str
    scratch: str
    srcgrids: str
    perfile: str
    reports: str
    fixed: str
    clean_intermediate: bool
    force_rewrite_srcgrids: bool
    deflate_level: int

    def expand(self, template: str, **kwargs: str) -> str:
        """Substitute {archive}/{output}/{scratch} plus any extra placeholders."""
        out = template.format(
            archive=self.archive,
            output=self.output,
            scratch=self.scratch,
            **kwargs,
        )
        return os.path.normpath(out) if out else out

    def ensure_dirs(self) -> None:
        """Create the working directories. Called once before any CDO work."""
        for d in (self.output, self.srcgrids, self.perfile, self.reports, self.fixed):
            os.makedirs(d, exist_ok=True)


def load_paths(config_dir: str = CONFIG_DIR) -> Paths:
    raw = _load_yaml(os.path.join(config_dir, "paths.yaml"))
    roots = raw.get("roots", {})
    work = raw.get("work", {})
    opts = raw.get("options", {})

    for key in ("archive", "output", "scratch"):
        if not roots.get(key):
            raise ValueError(f"paths.yaml: roots.{key} is required")

    output = roots["output"]

    def _under_output(value: str, default: str) -> str:
        v = value or default
        return v if os.path.isabs(v) else os.path.join(output, v)

    return Paths(
        archive=roots["archive"],
        output=output,
        scratch=roots["scratch"],
        srcgrids=_under_output(work.get("srcgrids"), "_srcgrids"),
        perfile=_under_output(work.get("perfile"), "_perfile_units"),
        reports=_under_output(work.get("reports"), "_reports"),
        fixed=_under_output(work.get("fixed"), "_fixed"),
        clean_intermediate=bool(opts.get("clean_intermediate", True)),
        force_rewrite_srcgrids=bool(opts.get("force_rewrite_srcgrids", True)),
        deflate_level=int(opts.get("deflate_level", 5)),
    )


# Target grids
@dataclass
class TargetGrid:
    name: str
    description: str
    gridtype: str
    xsize: int
    ysize: int
    xfirst: float
    xinc: float
    yfirst: float
    yinc: float

    def to_cdo_spec(self) -> str:
        """As a CDO grid description."""
        return textwrap.dedent(
            f"""\
            gridtype  = {self.gridtype}
            xsize     = {self.xsize}
            ysize     = {self.ysize}
            xfirst    = {self.xfirst}
            xinc      = {self.xinc}
            yfirst    = {self.yfirst}
            yinc      = {self.yinc}
            """
        )

    def write(self, path: str, force: bool = False) -> str:
        """Write the grid description file if missing (or if force=True)."""
        if force or not os.path.isfile(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_cdo_spec())
        return path

    @property
    def lon_range(self) -> tuple[float, float]:
        return self.xfirst, self.xfirst + (self.xsize - 1) * self.xinc

    @property
    def lat_range(self) -> tuple[float, float]:
        return self.yfirst, self.yfirst + (self.ysize - 1) * self.yinc


def load_grids(config_dir: str = CONFIG_DIR) -> tuple[dict[str, TargetGrid], dict[str, str]]:
    raw = _load_yaml(os.path.join(config_dir, "grids.yaml"))
    grids: dict[str, TargetGrid] = {}
    for name, spec in (raw.get("grids") or {}).items():
        grids[name] = TargetGrid(
            name=name,
            description=str(spec.get("description", "")).strip(),
            gridtype=str(spec.get("gridtype", "lonlat")),
            xsize=int(spec["xsize"]),
            ysize=int(spec["ysize"]),
            xfirst=float(spec["xfirst"]),
            xinc=float(spec["xinc"]),
            yfirst=float(spec["yfirst"]),
            yinc=float(spec["yinc"]),
        )
    defaults = dict(raw.get("defaults") or {})
    return grids, defaults


@dataclass
class ModelEntry:
    abbr: str
    enabled: bool = True
    domain: str = ""
    notes: str = ""
    raw: dict = field(default_factory=dict)

    def is_unavailable(self, variable: str, period: str) -> bool:
        for item in self.raw.get("unavailable") or []:
            if item.get("variable") == variable and item.get("period") == period:
                return True
        return False

    def pole_override(self) -> dict | None:
        """Rotated-pole parameters when the file has no grid_mapping."""
        po = self.raw.get("pole_override")
        return dict(po) if po else None

    def resolve_pattern(self, variable: str, period: str) -> str:
        """Resolve the model's filename pattern for (variable, period)."""
        pats = self.raw.get("patterns")
        if isinstance(pats, dict):
            by_var = pats.get(variable)
            if not isinstance(by_var, dict):
                raise KeyError(
                    f"Model {self.abbr}: no patterns defined for variable '{variable}'"
                )
            pattern = by_var.get(period)
            if not pattern:
                raise KeyError(
                    f"Model {self.abbr}: no pattern for variable '{variable}', "
                    f"period '{period}'"
                )
            return str(pattern)

        if period in self.raw:
            pattern = self.raw[period]
        elif "default" in self.raw:
            pattern = self.raw["default"]
        else:
            raise KeyError(
                f"Model {self.abbr}: no 'patterns', 'default' or '{period}' key"
            )

        pattern = str(pattern).replace("{VARIABLE}", variable).replace("{PERIOD}", period)
        if "{" in pattern or "}" in pattern:
            raise ValueError(
                f"Model {self.abbr}: unresolved placeholder in pattern: {pattern}"
            )
        return pattern

    def input_dir(self, registry: "Registry", paths: Paths,
                  variable: str, period: str) -> str:
        """Directory holding this model's files for (variable, period)."""
        override = (self.raw.get("dir_override") or {}).get(variable, {}).get(period)
        if override:
            return paths.expand(override, VARIABLE=variable, PERIOD=period)

        if registry.input_dir_template:
            return paths.expand(
                registry.input_dir_template, VARIABLE=variable, PERIOD=period
            )

        try:
            tmpl = registry.input_dirs[variable][period][self.domain]
        except KeyError as e:
            raise KeyError(
                f"Model {self.abbr}: no input directory for "
                f"variable={variable}, period={period}, domain={self.domain!r}. "
                f"Check `input_dirs` in the registry YAML."
            ) from e
        return paths.expand(tmpl, VARIABLE=variable, PERIOD=period)


@dataclass
class Registry:
    ensemble: str
    models: dict[str, ModelEntry]
    units_override: dict[tuple[str, str, str], str]
    input_dir_template: str = ""                       
    input_dirs: dict = field(default_factory=dict)     
    domain: str = ""

    def select(self, only: str | None = None,
               exclude: tuple[str, ...] = ()) -> list[ModelEntry]:
        """Models to process."""
        if only:
            if only not in self.models:
                raise KeyError(
                    f"Unknown model {only!r} for ensemble {self.ensemble}. "
                    f"Known: {', '.join(sorted(self.models))}"
                )
            return [self.models[only]]
        return [
            m for name, m in self.models.items()
            if m.enabled and name not in exclude
        ]

    def units_for(self, abbr: str, variable: str, period: str) -> str | None:
        return self.units_override.get((abbr, variable, period.lower()))


def load_registry(ensemble: str, config_dir: str = CONFIG_DIR) -> Registry:
    ensemble = ensemble.lower()
    if ensemble not in VALID_ENSEMBLES:
        raise ValueError(f"ensemble must be one of {VALID_ENSEMBLES}, got {ensemble!r}")

    raw = _load_yaml(os.path.join(config_dir, f"models_{ensemble}.yaml"))

    models: dict[str, ModelEntry] = {}
    for abbr, spec in (raw.get("models") or {}).items():
        spec = spec or {}
        models[abbr] = ModelEntry(
            abbr=abbr,
            enabled=bool(spec.get("enabled", True)),
            domain=str(spec.get("domain", raw.get("domain", ""))),
            notes=str(spec.get("notes", "")).strip(),
            raw=spec,
        )

    overrides: dict[tuple[str, str, str], str] = {}
    for item in raw.get("units_override") or []:
        key = (item["model"], item["variable"], str(item["period"]).lower())
        overrides[key] = str(item["units"])

    return Registry(
        ensemble=ensemble,
        models=models,
        units_override=overrides,
        input_dir_template=str(raw.get("input_dir", "")),
        input_dirs=dict(raw.get("input_dirs") or {}),
        domain=str(raw.get("domain", "")),
    )


@dataclass
class Settings:
    paths: Paths
    registry: Registry
    grid: TargetGrid
    variable: str
    period: str

    @property
    def period_tag(self) -> str:
        """Capitalised period used in output filenames (Historical / Rcp85)."""
        return "Historical" if self.period == "historical" else "Rcp85"

    def target_grid_file(self) -> str:
        """Path to the CDO target-grid description file, written on demand."""
        path = os.path.join(self.paths.scratch, f"target_grid_{self.grid.name}.txt")
        return self.grid.write(path, force=False)


def build_settings(ensemble: str, variable: str, period: str,
                   target_grid: str | None = None,
                   config_dir: str = CONFIG_DIR) -> Settings:
    paths = load_paths(config_dir)
    registry = load_registry(ensemble, config_dir)
    grids, defaults = load_grids(config_dir)

    grid_name = target_grid or defaults.get(ensemble.lower())
    if not grid_name:
        raise ValueError(
            f"No target grid for ensemble {ensemble!r}. Set defaults.{ensemble} "
            f"in grids.yaml or pass --target-grid."
        )
    if grid_name not in grids:
        raise KeyError(
            f"Unknown target grid {grid_name!r}. Known: {', '.join(sorted(grids))}"
        )

    return Settings(
        paths=paths,
        registry=registry,
        grid=grids[grid_name],
        variable=variable,
        period=period,
    )
