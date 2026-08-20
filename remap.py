#!/usr/bin/env python3
"""
remap.py - merge a model's files and remap them onto a regular lon/lat grid.

What happens to each model
  0. find its files ------> from the pattern and folder in the registry

  1. report anything unusual ------> partial years, gaps, reversed dates

  2. repair broken attributes ------> e.g., that would make CDO fail

  3. convert units per file ------> before merging, using each file's own units

  4. merge ------> into one file on the model's own grid

  5. describe the grid to CDO ------> this happens only when the file lacks cell corners

  6. remap ------> conservatively, onto the target grid

  7. set the units attribute ------> only for pr

  8. compress ------> NetCDF-4, deflate 5, one map per chunk

  9. put the attributes back ------> the CORDEX original ones, plus a fpsremap_* record

How the grid is described to CDO (steps 5 and 6):
Conservative remapping needs the four corners of every source cell, and some files store only centres, 
so the corners have to be worked out. In the archive the code was developed for, three cases were possible:

  A  the file states its corners ------> remapping happen straight away (2D lon/lat with `bounds`).

  B  rotated pole, or rlon/rlat ------> write a curvilinear grid description;
     with no grid_mapping at all ------> the pole convention is worked out and checked against the file's own lon/lat.

  C  Lambert conformal conic ------> write a projection grid description and let PROJ work out the corners.

  
Run inspect_CORDEX-FPS_models.py --mode dump first to see which case each model falls into.

Use as following: 
# one model
python remap.py --ensemble cpm --variable pr --period historical --model BTU

# every model, one after another
python remap.py --ensemble cpm --variable pr --period historical

# Parallel run for few models at a time, each with its own log file
mkdir -p logs
parallel -j 2 'ionice -c2 -n7 nice -n10 python -u remap.py --ensemble cpm
        --variable pr --period historical --model {} > logs/{}.log 2>&1' ::: BTU CNRM CMCC DWD

Running several models at once is done with GNU parallel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from fpsremap import __version__, cdo, nc_metadata, grids, fix_attributes, file_filters, units
from fpsremap.config import CONFIG_DIR, build_settings
from fpsremap.logging_utils import log, log_header, set_prefix


# CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Merge + conservatively remap CORDEX-FPSCONV output to a "
                    "regular lon/lat grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Use as following:", 1)[-1],
    )
    p.add_argument("--ensemble", choices=["cpm", "rcm"], required=True,
                   help="Which registry and default target grid to use.")
    p.add_argument("--variable", default="pr",
                   help="Variable to process (default: pr). pr is converted to mm/h; "
                        "tas is kept in native units.")
    p.add_argument("--period", choices=["historical", "rcp85"], default="historical",
                   help="Period to process (default: historical).")
    p.add_argument("--model", default=None,
                   help="Process only this model. Overrides `enabled: false` in the "
                        "registry, so a disabled model can be re-run without editing YAML.")
    p.add_argument("--exclude", default="",
                   help="Comma-separated model abbreviations to skip.")
    p.add_argument("--target-grid", default=None,
                   help="Target grid name from grids.yaml (default: the ensemble default).")

    g = p.add_argument_group(
        "optional file filters",
        "By default every file found is merged. These discard part of the "
        "archive and are off unless you ask for them.",)
    g.add_argument("--complete-years", action="store_true",
                   help="Keep only fully covered calendar years. Use when the "
                        "analysis needs whole years.")
    g.add_argument("--far-future", action="store_true",
                   help="For rcp85 only: when the record splits into two blocks "
                        "separated by a long gap, keep the later one.")
    g.add_argument("--drop-reversed", action="store_true",
                   help="Drop files whose filename end date precedes its start "
                        "date (usually corrupt or duplicated stubs).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report every step and every CDO command without running "
                        "them or writing output. Validates configuration and file "
                        "discovery for a whole ensemble quickly.")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Keep merged-native and pre-compression files for debugging.")
    p.add_argument("--overwrite", action="store_true",
                   help="Reprocess models whose final output already exists.")
    p.add_argument("--config-dir", default=CONFIG_DIR,
                   help="Directory holding the YAML configuration.")
    return p.parse_args(argv)


# Per-model processing
def process_model(entry, settings, args) -> dict:
    """Run every processing step for one model. Returns a status dict."""
    abbr = entry.abbr
    set_prefix(abbr)
    t_start = time.perf_counter()

    variable, period = settings.variable, settings.period
    paths, registry = settings.paths, settings.registry
    tag = settings.period_tag

    log_header(f"MODEL {abbr}  |  {variable}  |  {period}  |  "
               f"target grid {settings.grid.name}")

    if entry.notes:
        log(f"   note: {entry.notes}")
    if entry.is_unavailable(variable, period):
        log(f"   SKIP: marked unavailable for {variable}/{period} in the registry")
        return {"model": abbr, "status": "unavailable"}

    # 0. discover
    pattern = entry.resolve_pattern(variable, period)
    folder = entry.input_dir(registry, paths, variable, period)
    log(f"   folder : {folder}")
    log(f"   pattern: {pattern}*")

    if not os.path.isdir(folder):
        log("   SKIP: input directory does not exist")
        return {"model": abbr, "status": "missing_dir", "folder": folder}

    files = file_filters.find_files(folder, pattern)
    if not files:
        log("   SKIP: no files matched the pattern")
        return {"model": abbr, "status": "no_files"}
    log(f"   discovered {file_filters.summarise(files)}")

    # 1. report anything unusual, then apply any requested filters
    # By default nothing is removed and all files will be used. The findings are reported so you can
    # decide whether they matter for your analysis.
    for finding in file_filters.diagnose(files, period=period):
        log(f"   [NOTE] {finding}")

    if args.complete_years or args.far_future or args.drop_reversed:
        files = file_filters.apply_filters(
            files,
            period=period,
            complete_years=args.complete_years,
            far_future=args.far_future,
            drop_reversed=args.drop_reversed,
        )
        if not files:
            log("   SKIP: the requested filters removed every file")
            return {"model": abbr, "status": "no_files_after_filters"}
        log(f"   after filtering: {file_filters.summarise(files)}")
    else:
        log("   using every file found (no filters requested)")

    # output paths
    out_final = os.path.join(paths.output, f"{abbr}_{variable}_{tag}_merged.nc")
    if os.path.isfile(out_final) and not args.overwrite:
        log(f"   SKIP: output already exists (use --overwrite to redo)")
        log(f"         {out_final}")
        return {"model": abbr, "status": "exists", "output": out_final}

    merged_native = os.path.join(paths.output, f"{abbr}_{variable}_{tag}_merged_native.nc")
    regridded_raw = os.path.join(paths.output, f"{abbr}_{variable}_{tag}_regrid_rawunits.nc")
    finalized_tmp = os.path.join(paths.output, f"{abbr}_{variable}_{tag}_finalized.tmp.nc")
    perfile_dir = os.path.join(paths.perfile, f"{abbr}_{variable}_{tag}")
    fix_dir = os.path.join(paths.fixed, f"{abbr}_{variable}_{tag}")

    sample = files[0]
    temporaries: list[str] = []

    # 2. fix attribute in case of problamtic ones for cdo
    if not args.dry_run:
        os.makedirs(fix_dir, exist_ok=True)
    files, created = fix_attributes.fix_if_needed(
        files, variable, fix_dir, dry_run=args.dry_run
    )
    temporaries.extend(created)
    if created:
        sample = files[0]

    # 3. per-file unit conversion
    time_info = nc_metadata.describe_time_axis(sample)
    dt_hours = time_info.get("dt_hours")
    log(f"   time step: {nc_metadata.dt_label(dt_hours)} "
        f"(calendar {time_info.get('calendar') or 'unknown'})")

    override = registry.units_for(abbr, variable, period)
    if override:
        log(f"   units OVERRIDE from registry: forcing '{override}' "
            f"(the file's own label is known to be wrong)")

    plan = units.plan_conversions(files, variable, override=override, dt_hours=dt_hours)
    units.log_plan(plan, variable)
    target_units = plan[0]["target_units"]

    merge_inputs: list[str] = []
    if any(p["needs_scaling"] for p in plan):
        if not args.dry_run:
            os.makedirs(perfile_dir, exist_ok=True)
        for p in plan:
            if not p["needs_scaling"]:
                merge_inputs.append(p["path"])
                continue
            stem = os.path.splitext(os.path.basename(p["path"]))[0]
            out_fp = os.path.join(perfile_dir, f"units_{p['index']:04d}_{stem}.nc")
            cdo.scale_variable(p["path"], out_fp, variable, p["factor"],
                               label=f"({file_filters.parse_span(p['path']).label})",
                               dry_run=args.dry_run)
            merge_inputs.append(out_fp)
            temporaries.append(out_fp)
    else:
        merge_inputs = [p["path"] for p in plan]

    # 4. merge
    cdo.mergetime(merge_inputs, merged_native, dry_run=args.dry_run)

    # 5/6. remap
    target_grid_file = settings.target_grid_file()
    gm_var, gm_name = nc_metadata.get_grid_mapping(sample, variable)
    log(f"   grid_mapping: name={gm_name or '(none)'} var={gm_var or '(none)'}")

    remap_path, srcgrid_file, grid_info = "", None, {}

    if nc_metadata.detect_lonlat_vertices(sample)["has_vertices"]:
        remap_path = "A"
        log("   path A: native 2D lon/lat vertices -> direct remapcon")

    elif gm_name == "rotated_latitude_longitude" or (
            not gm_name and nc_metadata.has_rotated_axes(sample)):
        remap_path = "B"
        pole_override = entry.pole_override()
        if not gm_name:
            if pole_override is None:
                pole_override = dict(grids.EURO_CORDEX_POLE)
                log("   path B: rlon/rlat present but NO grid_mapping; assuming the "
                    "EURO-CORDEX default pole (validated against the file's lon/lat)")
            else:
                log("   path B: rlon/rlat present but NO grid_mapping; using the "
                    "registry's pole_override")
        else:
            pole_override = None
            log("   path B: rotated pole -> curvilinear source grid")

        srcgrid_file = os.path.join(paths.srcgrids, f"srcgrid_{abbr}_rotcurv.txt")
        if args.dry_run:
            log(f"      DRY-RUN would write source grid {srcgrid_file}")
        elif paths.force_rewrite_srcgrids or not os.path.isfile(srcgrid_file):
            os.makedirs(paths.srcgrids, exist_ok=True)
            grid_info = grids.write_rotated_curvilinear_srcgrid(
                sample, srcgrid_file, variable, pole_override=pole_override
            )
            log(f"      wrote {srcgrid_file}")
        else:
            log(f"      reusing existing source grid {srcgrid_file}")

    elif gm_name == "lambert_conformal_conic":
        remap_path = "C"
        log("   path C: Lambert conformal conic -> projection source grid")
        srcgrid_file = os.path.join(paths.srcgrids, f"srcgrid_{abbr}_lambert.txt")
        if args.dry_run:
            log(f"      DRY-RUN would write source grid {srcgrid_file}")
        elif paths.force_rewrite_srcgrids or not os.path.isfile(srcgrid_file):
            os.makedirs(paths.srcgrids, exist_ok=True)
            grid_info = grids.write_lambert_srcgrid(sample, srcgrid_file, variable)
            log(f"      wrote {srcgrid_file}")
        else:
            log(f"      reusing existing source grid {srcgrid_file}")

    else:
        # No usable grid description. Stop remapping.
        raise RuntimeError(
            f"{abbr}: cannot describe the source grid to CDO.\n"
            f"    grid_mapping : {gm_name or '(none)'}\n"
            f"    cell corners : none\n"
            f"    rotated axes : {'yes' if nc_metadata.has_rotated_axes(sample) else 'no'}\n"
            f"    sample file  : {os.path.basename(sample)}\n"
            f"  The file has no 2D lon/lat bounds, no rotated pole and no Lambert "
            f"projection, so the cell corners that conservative remapping needs "
            f"cannot be worked out. Inspect it with:\n"
            f"    python inspect_CORDEX-FPS_models.py --ensemble "
            f"{settings.registry.ensemble} --variable {variable} --period {period} "
            f"--mode dump --model {abbr} --stdout\n"
            f"  and either add a grid writer for its projection in "
            f"fpsremap/grids.py, or set `enabled: false` for this model."
        )

    cdo.remapcon(merged_native, regridded_raw, variable, target_grid_file,
                 srcgrid_file=srcgrid_file, label=f"path {remap_path}",
                 dry_run=args.dry_run)

    if not args.dry_run and not os.path.isfile(regridded_raw):
        raise RuntimeError(f"{abbr}: remap produced no output at {regridded_raw}")

    # 7. units attribute
    # Values were already scaled per file in step 3; this only fixes the label, which CDO's expr= does not update.
    if variable == "pr":
        cdo.set_units(regridded_raw, finalized_tmp, variable, target_units,
                      dry_run=args.dry_run)
        compress_src = finalized_tmp
        temporaries.append(finalized_tmp)
    else:
        log(f"   units: {variable} kept in native units ({target_units}); no change")
        compress_src = regridded_raw

    # 8. compress
    cdo.write_compressed(compress_src, out_final, paths.deflate_level,
                         ny=settings.grid.ysize, nx=settings.grid.xsize,
                         dry_run=args.dry_run)

    # 9. put the original attributes back
    if not args.dry_run:
        cdo.copy_global_attributes(sample, out_final)
        cdo.tag_processing_history(out_final, {
            "version": __version__,
            "ensemble": settings.registry.ensemble,
            "model": abbr,
            "variable": variable,
            "period": period,
            "target_grid": (f"{settings.grid.name} {settings.grid.xsize}x"
                            f"{settings.grid.ysize} @ {settings.grid.xinc} deg"),
            "remap_method": f"cdo remapcon (path {remap_path})",
            "source_files": str(len(files)),
            "coverage": file_filters.summarise(files),
            "units": target_units,
        })
        log("   wrote attributes back (CORDEX originals + fpsremap_* record)")

    # cleanup
    clean = paths.clean_intermediate and not args.keep_intermediate
    if not args.dry_run:
        cdo.remove_files([merged_native, regridded_raw], enabled=clean)
        cdo.remove_files(temporaries, enabled=clean)
        cdo.remove_dir_if_empty(perfile_dir)
        cdo.remove_dir_if_empty(fix_dir)
        if not clean:
            log("   kept intermediates (--keep-intermediate)")

    dt_min = (time.perf_counter() - t_start) / 60.0
    log(f"   DONE in {dt_min:.2f} min -> {out_final}")
    return {
        "model": abbr, "status": "ok", "output": out_final,
        "remap_path": remap_path, "n_files": len(files),
        "coverage": file_filters.summarise(files), "minutes": round(dt_min, 2),
        "grid_info": grid_info,
    }


# Main
def main(argv=None) -> int:
    args = parse_args(argv)
    set_prefix("")

    try:
        settings = build_settings(args.ensemble, args.variable, args.period,
                                  target_grid=args.target_grid,
                                  config_dir=args.config_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"CONFIGURATION ERROR: {e}", file=sys.stderr)
        return 2

    log_header(f"fpsremap {__version__}  |  remap.py  |  "
               f"{args.ensemble.upper()} {args.variable} {args.period}")
    log(f"archive     : {settings.paths.archive}")
    log(f"output      : {settings.paths.output}")
    log(f"target grid : {settings.grid.name}  {settings.grid.xsize}x"
        f"{settings.grid.ysize} @ {settings.grid.xinc} deg")
    log(f"              lon {settings.grid.lon_range[0]:.4f}..{settings.grid.lon_range[1]:.4f}  "
        f"lat {settings.grid.lat_range[0]:.4f}..{settings.grid.lat_range[1]:.4f}")
    active = [n for n, on in (("--complete-years", args.complete_years),
                              ("--far-future", args.far_future),
                              ("--drop-reversed", args.drop_reversed)) if on]
    log(f"filters     : {' '.join(active) if active else 'none - using every file found'}")

    if args.dry_run:
        log("MODE        : DRY RUN - no CDO commands are executed, nothing is written")
    else:
        try:
            log(f"cdo         : {cdo.check_cdo_available()}")
        except cdo.CdoError as e:
            print(f"\n{e}", file=sys.stderr)
            return 2
        settings.paths.ensure_dirs()
        settings.target_grid_file()

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    try:
        models = settings.registry.select(only=args.model, exclude=exclude)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    log(f"models      : {len(models)} -> {', '.join(m.abbr for m in models)}")

    results, failures = [], []
    for entry in models:
        try:
            results.append(process_model(entry, settings, args))
        except Exception as e:                           
            set_prefix(entry.abbr)
            log(f"   FAILED: {type(e).__name__}: {e}")
            failures.append({"model": entry.abbr, "status": "failed", "error": str(e)})

    set_prefix("")
    log_header("SUMMARY")
    for r in results + failures:
        extra = ""
        if r["status"] == "ok":
            extra = (f"path {r['remap_path']}  {r['n_files']} files  "
                     f"{r['coverage']}  {r['minutes']} min")
        elif r["status"] == "failed":
            extra = r["error"]
        log(f"   {r['model']:<16} {r['status']:<24} {extra}")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    log("")
    log(f"   {n_ok} succeeded, {len(failures)} failed, "
        f"{len(results) - n_ok} skipped")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
