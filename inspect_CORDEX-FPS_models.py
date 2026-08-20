#!/usr/bin/env python3
"""
inspect_CORDEX-FPS_models.py - list the model files inside the archive and inspect to find 
what grid each model uses, and how remap.py will handle it. 

Two modes, chosen with --mode:

1) list: which files each model has, the years they cover, and any files in
the folder that match no model pattern.

2) dump: opens one file per model and reports its dimensions, variables, attributes, grid mapping, coordinates, 
whether cell corners are present, the time axis and the units. It also says which of the four grid cases the model 
falls into, which is important to know.

3) all: both above modes.

Use as following:
# what CPM precipitation files are there for the historical period?
python inspect_CORDEX-FPS_models.py --ensemble cpm --variable pr --period historical

# full metadata for the RCMs, also written as JSON
python inspect_CORDEX-FPS_models.py --ensemble rcm --variable pr --period rcp85 --mode dump --json

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from fpsremap import __version__, nc_metadata, file_filters
from fpsremap.config import CONFIG_DIR, build_settings, load_paths, load_registry

try:
    from netCDF4 import Dataset
except ImportError:                                      
    sys.exit("netCDF4 is required. Install with: conda install -c conda-forge netcdf4")


# CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Inventory and inspect CORDEX-FPSCONV NetCDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n", 1)[-1],
    )
    p.add_argument("--ensemble", choices=["cpm", "rcm"], default="cpm",
                   help="Which registry to read (default: cpm).")
    p.add_argument("--variable", default="pr",
                   help="Variable to inspect (default: pr).")
    p.add_argument("--period", choices=["historical", "rcp85"], default="historical",
                   help="Period to inspect (default: historical).")
    p.add_argument("--mode", choices=["list", "dump", "all"], default="list",
                   help="What to report (default: list).")
    p.add_argument("--model", default=None,
                   help="Restrict to one model abbreviation.")
    p.add_argument("--exclude", default="",
                   help="Comma-separated model abbreviations to skip.")
    p.add_argument("--sample", choices=["first", "last", "all"], default="first",
                   help="Which file(s) to open in --mode dump (default: first). "
                        "'all' is slow but catches files that differ mid-series.")
    p.add_argument("--complete-years", action="store_true",
                   help="Show the files that remain after keeping only fully "
                        "covered calendar years.")
    p.add_argument("--far-future", action="store_true",
                   help="Show the files that remain after keeping only the later "
                        "of two disjoint rcp85 time blocks.")
    p.add_argument("--drop-reversed", action="store_true",
                   help="Show the files that remain after dropping reversed date "
                        "ranges.")
    p.add_argument("--json", action="store_true",
                   help="Also write a machine-readable JSON report.")
    p.add_argument("--stdout", action="store_true",
                   help="Print to the terminal instead of writing report files.")
    p.add_argument("--output", default=None,
                   help="Explicit output path for the text report.")
    p.add_argument("--config-dir", default=CONFIG_DIR,
                   help="Directory holding the YAML configuration.")
    return p.parse_args(argv)


class Report:
    """Accumulates text so it can be printed and/or written in one go."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str = ""):
        self.lines.append(msg)

    def rule(self, char: str = "=", width: int = 60):
        self.lines.append(char * width)

    def header(self, title: str, char: str = "="):
        self.rule(char)
        self.lines.append(title)
        self.rule(char)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def _summarise_array(arr) -> str:
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return "all non-finite"
    v = a[finite]
    return (f"min={v.min():.4f} max={v.max():.4f} "
            f"mean={v.mean():.4f} n_nonfinite={int((~finite).sum())}")


def predict_remap_path(nc_file: str, variable: str,
                       pole_override: dict | None) -> tuple[str, str]:
    """Predict which rout the remap.py will take. Returns (path_id, explanation)."""
    try:
        gm_var, gm_name = nc_metadata.get_grid_mapping(nc_file, variable)
    except Exception as e:                               # noqa: BLE001
        return "ERROR", f"could not read grid mapping: {e}"

    if nc_metadata.detect_lonlat_vertices(nc_file)["has_vertices"]:
        return ("A", "native 2D lon/lat vertices present -> direct remapcon "
                     "(most accurate, no grid reconstruction)")

    if gm_name == "rotated_latitude_longitude":
        bnds = nc_metadata.detect_rotated_axis_bounds(nc_file)
        extra = ("using native rlon_bnds/rlat_bnds"
                 if bnds["has_bounds"] else "corners inferred from cell centres")
        return "B", f"rotated pole -> curvilinear source grid ({extra})"

    if not gm_name and nc_metadata.has_rotated_axes(nc_file):
        if pole_override:
            return ("B", "rlon/rlat present but NO grid_mapping -> curvilinear "
                         "source grid using the registry's pole_override "
                         "(validated against the file's 2D lon/lat)")
        return ("B", "rlon/rlat present but NO grid_mapping and no pole_override "
                     "in the registry -> will assume the EURO-CORDEX default pole. "
                     "Check the fit score in the remap log.")

    if gm_name == "lambert_conformal_conic":
        return "C", "Lambert conformal conic -> projection source grid (PROJ derives corners)"

    return ("-", f"no usable grid description (grid_mapping={gm_name!r}): no cell "
                 f"corners, no rotated pole, no Lambert projection. remap.py will "
                 f"not work on this model.")


# Per-file inspection
def inspect_file(nc_path: str, variable: str,
                 pole_override: dict | None = None) -> dict:
    """Collect everything worth knowing about one file."""
    info: dict = {"path": nc_path, "basename": os.path.basename(nc_path)}
    try:
        info["size_bytes"] = os.path.getsize(nc_path)
    except OSError:
        info["size_bytes"] = 0

    try:
        with Dataset(nc_path, "r") as ds:
            info["global_attributes"] = {
                a: nc_metadata.attr_to_text(ds.getncattr(a)) for a in ds.ncattrs()
            }
            info["dimensions"] = {
                d: {"size": len(dim), "unlimited": dim.isunlimited()}
                for d, dim in ds.dimensions.items()
            }
            info["variables"] = {
                name: {
                    "dtype": str(v.dtype),
                    "dimensions": list(v.dimensions),
                    "shape": list(v.shape),
                    "attributes": {a: nc_metadata.attr_to_text(v.getncattr(a))
                                   for a in v.ncattrs()},
                }
                for name, v in ds.variables.items()
            }

            data_var = variable if variable in ds.variables else \
                nc_metadata.find_data_variable(nc_path, variable)
            info["data_variable"] = data_var

            rlon = rlat = ""
            try:
                rlon, rlat = nc_metadata.find_rotated_axes(ds)
            except KeyError:
                pass
            xname = yname = ""
            try:
                xname, yname = nc_metadata.find_projection_axes(ds)
            except KeyError:
                pass
            lon2d, lat2d = nc_metadata.find_lonlat_2d(ds)

            info["coords"] = {
                "rotated_axes": [rlon, rlat] if rlon else [],
                "projection_axes": [xname, yname] if xname else [],
                "lonlat_2d": [lon2d, lat2d] if lon2d else [],
            }

            if lon2d and lat2d:
                info["lon_summary"] = _summarise_array(ds.variables[lon2d][:])
                info["lat_summary"] = _summarise_array(ds.variables[lat2d][:])
            if rlon and rlat:
                info["grid_shape"] = [int(ds.variables[rlat].size),
                                      int(ds.variables[rlon].size)]

        gm_var, gm_name = nc_metadata.get_grid_mapping(nc_path, data_var or variable)
        info["grid_mapping"] = {"variable": gm_var, "name": gm_name}
        if gm_var:
            with Dataset(nc_path, "r") as ds:
                if gm_var in ds.variables:
                    gv = ds.variables[gm_var]
                    info["grid_mapping"]["attributes"] = {
                        a: nc_metadata.attr_to_text(gv.getncattr(a)) for a in gv.ncattrs()
                    }

        info["vertices"] = nc_metadata.detect_lonlat_vertices(nc_path)
        info["axis_bounds"] = nc_metadata.detect_rotated_axis_bounds(nc_path)
        info["time"] = nc_metadata.describe_time_axis(nc_path)

        if data_var:
            try:
                info["units"] = nc_metadata.read_var_units(nc_path, data_var)
            except Exception:                            
                info["units"] = ""

        path_id, why = predict_remap_path(nc_path, data_var or variable, pole_override)
        info["remap_path"] = {"id": path_id, "explanation": why}

        # Attribute-type quirks that would crash CDO.
        from fpsremap.fix_attributes import needs_fixing
        needs, reasons = needs_fixing(nc_path, data_var or variable)
        info["needs_fixing"] = {"needed": needs, "reasons": reasons}

        info["ok"] = True
    except Exception as e:                               
        info["ok"] = False
        info["error"] = f"{type(e).__name__}: {e}"
    return info


# Modes
def mode_list(args, rep: Report) -> dict:
    settings = build_settings(args.ensemble, args.variable, args.period,
                              config_dir=args.config_dir)
    registry, paths = settings.registry, settings.paths
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    rep.header(f"FILE INVENTORY  |  ensemble={args.ensemble.upper()}  "
               f"variable={args.variable}  period={args.period}")
    rep(f"archive root : {paths.archive}")
    rep(f"target grid  : {settings.grid.name}  "
        f"({settings.grid.xsize}x{settings.grid.ysize} @ {settings.grid.xinc} deg)")
    active = [n for n, on in (("--complete-years", args.complete_years),
                              ("--far-future", args.far_future),
                              ("--drop-reversed", args.drop_reversed)) if on]
    rep(f"filters      : {' '.join(active) if active else 'none - showing the raw archive'}")
    rep()

    result: dict = {"models": {}, "directories": {}}
    seen_dirs: dict[str, set[str]] = {}

    for entry in registry.select(only=args.model, exclude=exclude):
        rep("-" * 60)
        rep(f"MODEL  {entry.abbr}"
            + (f"   [domain {entry.domain}]" if entry.domain else ""))
        if entry.notes:
            rep(f"  note: {entry.notes}")

        if entry.is_unavailable(args.variable, args.period):
            rep(f"  SKIPPED: marked unavailable for {args.variable}/{args.period}")
            result["models"][entry.abbr] = {"status": "unavailable"}
            rep()
            continue

        try:
            pattern = entry.resolve_pattern(args.variable, args.period)
            folder = entry.input_dir(registry, paths, args.variable, args.period)
        except (KeyError, ValueError) as e:
            rep(f"  CONFIG ERROR: {e}")
            result["models"][entry.abbr] = {"status": "config_error", "error": str(e)}
            rep()
            continue

        rep(f"  folder : {folder}")
        rep(f"  pattern: {pattern}*")

        if not os.path.isdir(folder):
            rep("  MISSING DIRECTORY")
            result["models"][entry.abbr] = {"status": "missing_dir", "folder": folder}
            rep()
            continue

        files = file_filters.find_files(folder, pattern)
        seen_dirs.setdefault(folder, set()).update(files)

        raw_n = len(files)
        for finding in file_filters.diagnose(files, period=args.period):
            rep(f"  NOTE   : {finding}")

        any_filter = args.complete_years or args.far_future or args.drop_reversed
        if any_filter and files:
            files = file_filters.apply_filters(
                files,
                period=args.period,
                complete_years=args.complete_years,
                far_future=args.far_future,
                drop_reversed=args.drop_reversed,
            )

        rep(f"  files  : {file_filters.summarise(files)}"
            + (f"   (from {raw_n} before filtering)"
               if any_filter and raw_n != len(files) else ""))
        for f in files:
            sp = file_filters.parse_span(f)
            rep(f"      {sp.label:>19}  {os.path.basename(f)}")
        if not files:
            rep("      NO FILES MATCHED")
        rep()

        result["models"][entry.abbr] = {
            "status": "ok" if files else "no_files",
            "folder": folder, "pattern": pattern,
            "n_files": len(files), "n_files_raw": raw_n,
            "coverage": file_filters.summarise(files),
            "files": [os.path.basename(f) for f in files],
        }

    rep("-" * 60)
    rep("UNMATCHED FILES")
    any_unmatched = False
    for folder, claimed in sorted(seen_dirs.items()):
        if not os.path.isdir(folder):
            continue
        all_nc = set(file_filters.find_files(folder, ""))
        unmatched = sorted(all_nc - claimed)
        result["directories"][folder] = {
            "n_total": len(all_nc), "n_unmatched": len(unmatched),
            "unmatched": [os.path.basename(f) for f in unmatched],
        }
        if unmatched:
            any_unmatched = True
            rep(f"  {folder}   ({len(unmatched)} of {len(all_nc)})")
            for f in unmatched:
                rep(f"      {os.path.basename(f)}")
    if not any_unmatched:
        rep("  none - every file in the scanned folders matched to a model pattern")
    rep()
    return result


def mode_dump(args, rep: Report) -> dict:
    settings = build_settings(args.ensemble, args.variable, args.period,
                              config_dir=args.config_dir)
    registry, paths = settings.registry, settings.paths
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    rep.header(f"METADATA DUMP  |  ensemble={args.ensemble.upper()}  "
               f"variable={args.variable}  period={args.period}")
    rep(f"Which file is being inspected: {args.sample}")
    rep()

    result: dict = {"models": {}}
    path_tally: dict[str, list[str]] = {}

    for entry in registry.select(only=args.model, exclude=exclude):
        rep("=" * 60)
        rep(f"MODEL  {entry.abbr}")
        rep("=" * 60)

        if entry.is_unavailable(args.variable, args.period):
            rep(f"  marked unavailable for {args.variable}/{args.period}\n")
            continue
        try:
            pattern = entry.resolve_pattern(args.variable, args.period)
            folder = entry.input_dir(registry, paths, args.variable, args.period)
        except (KeyError, ValueError) as e:
            rep(f"  CONFIG ERROR: {e}\n")
            continue

        files = file_filters.find_files(folder, pattern)
        if not files:
            rep(f"  no files in {folder}\n")
            result["models"][entry.abbr] = {"status": "no_files"}
            continue

        chosen = {"first": files[:1], "last": files[-1:], "all": files}[args.sample]
        rep(f"  {len(files)} file(s) found; inspecting {len(chosen)}")
        rep()

        infos = []
        for fp in chosen:
            info = inspect_file(fp, args.variable, entry.pole_override())
            infos.append(info)
            _render_file_info(rep, info)

        result["models"][entry.abbr] = {"status": "ok", "files": infos}
        if infos and infos[0].get("ok"):
            pid = infos[0]["remap_path"]["id"]
            path_tally.setdefault(pid, []).append(entry.abbr)

    # The most important part of the report goes last, where it is read.
    rep("=" * 60)
    rep("REMAP PATH SUMMARY")
    rep("=" * 60)
    labels = {
        "A": "corners stated in the file -> remap directly  (most accurate)",
        "B": "rotated pole               -> curvilinear grid description",
        "C": "Lambert conformal conic    -> projection grid description",
        "-": "none of the above          -> remap.py will stop on this model",
        "ERROR": "could not be determined",
    }
    for pid in ("A", "B", "C", "-", "ERROR"):
        if pid in path_tally:
            rep(f"  path {pid}: {labels[pid]}")
            rep(f"           {', '.join(sorted(path_tally[pid]))}")
    if "-" in path_tally:
        rep()
        rep("  The models listed under '-' have no grid description CDO can use.")
        rep("  remap.py will stop on them. Either add a writer for their projection")
        rep("  in fpsremap/grids.py, or set enabled: false.")
    rep()
    result["remap_path_summary"] = path_tally
    return result


def _render_file_info(rep: Report, info: dict) -> None:
    rep(f"  --- {info['basename']}  ({_fmt_bytes(info.get('size_bytes', 0))})")
    if not info.get("ok"):
        rep(f"      UNREADABLE: {info.get('error')}")
        rep()
        return

    gm = info.get("grid_mapping", {})
    rep(f"      grid_mapping : {gm.get('name') or '(none)'}"
        f"{'  [var: ' + gm['variable'] + ']' if gm.get('variable') else ''}")
    for k, v in (gm.get("attributes") or {}).items():
        if k != "grid_mapping_name":
            rep(f"          {k} = {v}")

    c = info.get("coords", {})
    rep(f"      coordinates  : rotated={c.get('rotated_axes') or '-'}  "
        f"projected={c.get('projection_axes') or '-'}  "
        f"lonlat2d={c.get('lonlat_2d') or '-'}")
    if info.get("grid_shape"):
        ny, nx = info["grid_shape"]
        rep(f"      native grid  : {nx} x {ny} cells")
    if info.get("lon_summary"):
        rep(f"      lon          : {info['lon_summary']}")
        rep(f"      lat          : {info['lat_summary']}")

    v = info.get("vertices", {})
    b = info.get("axis_bounds", {})
    rep(f"      cell corners : "
        + ("2D vertices present" if v.get("has_vertices")
           else "1D axis bounds present" if b.get("has_bounds")
           else "NONE - must be reconstructed"))

    t = info.get("time", {})
    if t.get("ok"):
        rep(f"      time         : {t['n']} steps, {nc_metadata.dt_label(t.get('dt_hours'))}, "
            f"calendar={t.get('calendar')}, units={t.get('units')}")
    elif t.get("note"):
        rep(f"      time         : {t['note']}")

    if info.get("units") is not None:
        rep(f"      units        : {info.get('units') or '(none)'}")

    ns = info.get("needs_fixing", {})
    if ns.get("needed"):
        rep("      ATTRIBUTE PROBLEMS (remap.py will repair these automatically):")
        for r in ns.get("reasons", []):
            rep(f"          - {r}")

    rp = info.get("remap_path", {})
    rep(f"      REMAP PATH {rp.get('id')}: {rp.get('explanation')}")

    ga = info.get("global_attributes", {})
    for key in ("driving_model_id", "driving_experiment_name", "model_id",
                "institute_id", "experiment_id", "frequency", "CORDEX_domain"):
        if key in ga:
            rep(f"      {key:<24}: {ga[key]}")
    rep()


# Main
def main(argv=None) -> int:
    args = parse_args(argv)
    rep = Report()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep(f"# fpsremap {__version__}  |  inspect_CORDEX-FPS_models.py  |  generated {stamp}")
    rep(f"# command: {' '.join(sys.argv)}")
    rep()

    payload: dict = {
        "generated": stamp, "fpsremap_version": __version__,
        "command": " ".join(sys.argv), "args": vars(args),
    }

    try:
        if args.mode in ("list", "all"):
            payload["list"] = mode_list(args, rep)
        if args.mode in ("dump", "all"):
            payload["dump"] = mode_dump(args, rep)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    text = rep.text()

    if args.stdout:
        print(text)
        return 0

    paths = load_paths(args.config_dir)
    os.makedirs(paths.reports, exist_ok=True)
    stem = f"inspect_{args.ensemble}_{args.variable}_{args.period}_{args.mode}"
    txt_path = args.output or os.path.join(paths.reports, f"{stem}.txt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {txt_path}")

    if args.json:
        json_path = os.path.splitext(txt_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
