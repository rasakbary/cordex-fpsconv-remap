"""
Run CDO commands and copy NetCDF attributes onto the result.

`copy_global_attributes` puts the source file's global attributes back on the
output, because CDO writes only a short header and the CORDEX information
about the run would otherwise be lost.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from netCDF4 import Dataset

from .logging_utils import log

# Attributes CDO/NCO write themselves; never copy these over a fresh output.
_TOOL_ATTRS = {"history", "CDO", "CDI", "NCO"}


class CdoError(RuntimeError):
    """A CDO command returned a non-zero exit status."""


def check_cdo_available() -> str:
    """Return the CDO version string, or raise with a message."""
    exe = shutil.which("cdo")
    if not exe:
        raise CdoError(
            "`cdo` was not found on PATH. Install it with:\n"
            "    conda install -c conda-forge cdo\n"
            "or load your site's CDO module before running."
        )
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=30)
        first = (out.stdout or out.stderr).splitlines()
        return first[0].strip() if first else "cdo (version unknown)"
    except Exception:                                    
        return "cdo (version unknown)"


def run_cdo(cmd: list[str], description: str, *, step: str = "",
            dry_run: bool = False) -> float:
    """Run one CDO command. Returns elapsed seconds. Raises CdoError on failure."""
    tag = f"[{step}] " if step else ""
    log(f"   {tag}{description}")

    if dry_run:
        log(f"      DRY-RUN: {' '.join(cmd)}")
        return 0.0

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0

    if proc.returncode != 0:
        log(f"      FAILED after {dt:.1f}s")
        log(f"      command: {' '.join(cmd)}")
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines()[-15:]:
                log(f"      cdo: {line}")
        raise CdoError(f"{description} failed (exit {proc.returncode})")

    log(f"      ok in {dt:.1f}s")
    return dt


# Common CDO operations
def mergetime(files: list[str], out_nc: str, *, dry_run: bool = False) -> float:
    return run_cdo(["cdo", "-O", "mergetime", *files, out_nc],
                   f"mergetime {len(files)} file(s) -> {os.path.basename(out_nc)}",
                   step="1/4", dry_run=dry_run)


def scale_variable(in_nc: str, out_nc: str, var: str, factor: float,
                   *, label: str = "", dry_run: bool = False) -> float:
    return run_cdo(["cdo", "-O", f"expr,{var}={var}*{factor:g}", in_nc, out_nc],
                   f"scale {var} x{factor:g} {label}".strip(), dry_run=dry_run)


def select_variable(in_nc: str, out_nc: str, var: str,
                    *, dry_run: bool = False) -> float:
    return run_cdo(["cdo", "-O", f"-selname,{var}", in_nc, out_nc],
                   f"select {var} -> {os.path.basename(out_nc)}", dry_run=dry_run)


def remapcon(in_nc: str, out_nc: str, var: str, target_grid_file: str,
             srcgrid_file: str | None = None, *, label: str = "",
             dry_run: bool = False) -> float:
    """Conservative remap, optionally forcing the source grid with -setgrid.

    `-setgrid` is applied first (it only rewrites metadata), then `-selname` reduces the file 
    to the single variable, and only then the `remapcon` run.
    """
    cmd = ["cdo", "-O", f"remapcon,{target_grid_file}", f"-selname,{var}"]
    if srcgrid_file:
        cmd.append(f"-setgrid,{srcgrid_file}")
    cmd += [in_nc, out_nc]
    what = f"remapcon ({label})" if label else "remapcon"
    return run_cdo(cmd, f"{what} -> {os.path.basename(out_nc)}",
                   step="2/4", dry_run=dry_run)


def set_units(in_nc: str, out_nc: str, var: str, units: str,
              *, dry_run: bool = False) -> float:
    return run_cdo(["cdo", "-O", f"setattribute,{var}@units={units}", in_nc, out_nc],
                   f"set {var} units = {units}", step="3/4", dry_run=dry_run)


def write_compressed(in_nc: str, out_nc: str, deflate_level: int = 5,
                     *, ny: int | None = None, nx: int | None = None,
                     dry_run: bool = False) -> float:
    """Write the final output as compressed NetCDF-4.

    `-f nc4 -z zip_<n>` applies deflate plus the shuffle filter and auto-chunks
    each record to one full horizontal field, i.e. (time=1, lat=ny, lon=nx).
    Note that this layout is right for this project because downstream analysis reads
    whole maps per timestep, not long time series at single points. Otherwise the chunk method is important
    and need to be correctly used.
    """
    shape = f", chunk 1x{ny}x{nx}" if ny and nx else ""
    return run_cdo(
        ["cdo", "-O", "-f", "nc4", "-z", f"zip_{deflate_level}", "copy", in_nc, out_nc],
        f"compress (deflate {deflate_level}, shuffle{shape}) -> {os.path.basename(out_nc)}",
        step="4/4", dry_run=dry_run,
    )


def copy_global_attributes(src_nc: str, dst_nc: str) -> None:
    """Carry the source file's global attributes onto the remapped output.

    CDO writes a minimal header, so without this the output loses its CORDEX
    record of the run (driving model, experiment, institute, contact, comments ...). 
    """
    with Dataset(src_nc, "r") as src, Dataset(dst_nc, "r+") as dst:
        dst_attrs = set(dst.ncattrs())
        for a in sorted(src.ncattrs()):
            s_val = src.getncattr(a)
            if a in _TOOL_ATTRS and a in dst_attrs:
                if a == "history":
                    d_val = dst.getncattr("history")
                    if str(s_val).strip() and str(s_val).strip() not in str(d_val):
                        dst.setncattr("history",
                                      str(s_val).rstrip() + "\n" + str(d_val).lstrip())
                continue
            if a not in dst_attrs:
                dst.setncattr(a, s_val)
            elif dst.getncattr(a) != s_val:
                alt = f"source_{a}"
                if alt not in dst_attrs:
                    dst.setncattr(alt, s_val)
        if "fpsremap_source_file" not in dst_attrs:
            dst.setncattr("fpsremap_source_file", os.path.basename(src_nc))


def tag_processing_history(nc_file: str, entries: dict[str, str]) -> None:
    """Record how this file was produced, as `fpsremap_*` global attributes.
    """
    with Dataset(nc_file, "r+") as ds:
        for k, v in entries.items():
            ds.setncattr(f"fpsremap_{k}", str(v))


def remove_files(paths, *, enabled: bool = True) -> None:
    """Delete intermediates."""
    if not enabled:
        return
    for p in paths:
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except OSError as e:                             
            log(f"   [WARN] could not remove {p}: {e}")


def remove_dir_if_empty(path: str) -> None:
    try:
        if path and os.path.isdir(path):
            os.rmdir(path)
    except OSError:
        pass
