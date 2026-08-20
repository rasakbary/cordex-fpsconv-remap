"""
Read grid, coordinate and time information out of a NetCDF file..
"""

from __future__ import annotations

import re

import numpy as np
from netCDF4 import Dataset


def coerce_attr_float(val, default: float | None = None) -> float:
    """Coerce a NetCDF attribute to float, tolerating values that break float().

    cases this handles in the FPSCONV archive:
      - Fortran-style precision suffixes stored as text: '-162.f', '6371000.d'
      - 1-element arrays instead of scalars
      - byte strings
    """
    if val is None:
        if default is not None:
            return float(default)
        raise ValueError("attribute is None and no default given")

    if isinstance(val, bytes):
        val = val.decode(errors="replace")

    if isinstance(val, np.ndarray):
        if val.size == 0:
            if default is not None:
                return float(default)
            raise ValueError("empty array attribute")
        val = val.ravel()[0]

    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    s = str(val).strip()
    s = re.sub(r"[fFdD]$", "", s)   # '-162.f' -> '-162.'
    return float(s)


def attr_to_text(val) -> str:
    """Flatten any string-like attribute (bytes, np.str_, arrays) to plain str."""
    if isinstance(val, bytes):
        return val.decode(errors="replace")
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return attr_to_text(val.item())
        return " ".join(attr_to_text(x) for x in val.ravel())
    return str(val)


# Grid mapping
def get_grid_mapping(nc_file: str, var: str) -> tuple[str, str]:
    """Return (grid_mapping_varname, grid_mapping_name_lowercase)."""
    with Dataset(nc_file, "r") as ds:
        if var not in ds.variables:
            return "", ""
        gm_var = attr_to_text(getattr(ds.variables[var], "grid_mapping", "")).strip()
        # handles 'rotated pole' with a space instead of an underscore.
        gm_var_fixed = gm_var.replace(" ", "_")
        if gm_var_fixed in ds.variables:
            gm_var = gm_var_fixed
        if not gm_var or gm_var not in ds.variables:
            return gm_var, ""
        gm_name = attr_to_text(
            getattr(ds.variables[gm_var], "grid_mapping_name", "")
        ).strip().lower()
        return gm_var, gm_name


# Coordinate axis
def find_var_by_standard_name(ds: Dataset, std_name: str,
                              ndim: int | None = None) -> str:
    target = std_name.strip().lower()
    for name, v in ds.variables.items():
        sn = attr_to_text(getattr(v, "standard_name", "")).strip().lower()
        if sn == target and (ndim is None or v.ndim == ndim):
            return name
    return ""


def find_rotated_axes(ds: Dataset) -> tuple[str, str]:
    """1D rotated axes. Raises KeyError if the file is not rotated-pole."""
    rlon = find_var_by_standard_name(ds, "grid_longitude", ndim=1)
    rlat = find_var_by_standard_name(ds, "grid_latitude", ndim=1)
    if not rlon and "rlon" in ds.variables and ds.variables["rlon"].ndim == 1:
        rlon = "rlon"
    if not rlat and "rlat" in ds.variables and ds.variables["rlat"].ndim == 1:
        rlat = "rlat"
    if not rlon or not rlat:
        raise KeyError(
            "Could not locate rotated 1D axes (grid_longitude/grid_latitude or rlon/rlat)."
        )
    return rlon, rlat


def find_projection_axes(ds: Dataset) -> tuple[str, str]:
    """1D projected axes. Raises KeyError if the file is not projected."""
    xname = find_var_by_standard_name(ds, "projection_x_coordinate", ndim=1)
    yname = find_var_by_standard_name(ds, "projection_y_coordinate", ndim=1)
    if not xname and "x" in ds.variables and ds.variables["x"].ndim == 1:
        xname = "x"
    if not yname and "y" in ds.variables and ds.variables["y"].ndim == 1:
        yname = "y"
    if not xname or not yname:
        raise KeyError("Could not locate projection axes (projection_x/y_coordinate or x/y).")
    return xname, yname


def find_lonlat_2d(ds: Dataset) -> tuple[str, str]:
    """2D geographic coordinates. Returns ("", "") when absent."""
    lon = find_var_by_standard_name(ds, "longitude", ndim=2)
    lat = find_var_by_standard_name(ds, "latitude", ndim=2)
    if not lon and "lon" in ds.variables and ds.variables["lon"].ndim == 2:
        lon = "lon"
    if not lat and "lat" in ds.variables and ds.variables["lat"].ndim == 2:
        lat = "lat"
    if not lon or not lat:
        return "", ""
    return lon, lat


def has_rotated_axes(nc_file: str) -> bool:
    """True when rlon/rlat exist, regardless of whether grid_mapping does."""
    with Dataset(nc_file, "r") as ds:
        try:
            find_rotated_axes(ds)
            return True
        except KeyError:
            return False


# Cell-corner detection
def detect_lonlat_vertices(nc_file: str) -> dict:
    """Detect 2D lon/lat carrying a `bounds` attribute -> (ny, nx, 4) vertices.

    This is the best case for conservative remapping: the file already provide
    the cell corners, so CDO can compute overlap areas without us providing CDO the corners. 
    When this returns has_vertices=True the remapper skips grid reconstruction step.
    """
    info = {"has_vertices": False, "lon_name": "", "lat_name": "",
            "lon_bnds": "", "lat_bnds": ""}
    with Dataset(nc_file, "r") as ds:
        lon, lat = find_lonlat_2d(ds)
        if not lon or not lat:
            return info
        lb = attr_to_text(getattr(ds.variables[lon], "bounds", "")).strip()
        ab = attr_to_text(getattr(ds.variables[lat], "bounds", "")).strip()
        if lb and ab and lb in ds.variables and ab in ds.variables:
            lbv, abv = ds.variables[lb], ds.variables[ab]
            if (lbv.ndim == 3 and abv.ndim == 3
                    and lbv.shape[-1] == 4 and abv.shape[-1] == 4):
                info.update({"has_vertices": True, "lon_name": lon, "lat_name": lat,
                             "lon_bnds": lb, "lat_bnds": ab})
    return info


def detect_rotated_axis_bounds(nc_file: str) -> dict:
    """Detect native 1D rlon_bnds/rlat_bnds of shape (n, 2).

    Second-best case: the model states its own cell edges in rotated space, so
    we can rotate exact edges to geographic coordinates.
    """
    info = {"has_bounds": False, "rlon_bnds": "", "rlat_bnds": ""}
    with Dataset(nc_file, "r") as ds:
        try:
            rlon, rlat = find_rotated_axes(ds)
        except KeyError:
            return info
        rb = attr_to_text(getattr(ds.variables[rlon], "bounds", "")).strip()
        ab = attr_to_text(getattr(ds.variables[rlat], "bounds", "")).strip()
        if (rb and ab and rb in ds.variables and ab in ds.variables
                and ds.variables[rb].ndim == 2 and ds.variables[ab].ndim == 2):
            info.update({"has_bounds": True, "rlon_bnds": rb, "rlat_bnds": ab})
    return info


# Time axis
def guess_time_var(ds: Dataset) -> str:
    if "time" in ds.variables:
        return "time"
    for name, v in ds.variables.items():
        if attr_to_text(getattr(v, "standard_name", "")).strip().lower() == "time":
            return name
        if "since" in attr_to_text(getattr(v, "units", "")).lower():
            return name
    return ""


def describe_time_axis(nc_file: str) -> dict:
    """Summarise the time axis: length, calendar, units, span, timestep.

    The timestep is taken as the median difference.
    """
    out = {"ok": False, "name": "", "n": 0, "units": "", "calendar": "",
           "first": None, "last": None, "dt_hours": None, "note": ""}
    try:
        with Dataset(nc_file, "r") as ds:
            tname = guess_time_var(ds)
            if not tname:
                out["note"] = "no time variable found"
                return out
            tv = ds.variables[tname]
            vals = np.asarray(tv[:], dtype=float)
            out.update({
                "name": tname,
                "n": int(vals.size),
                "units": attr_to_text(getattr(tv, "units", "")),
                "calendar": attr_to_text(getattr(tv, "calendar", "standard")),
            })
            if vals.size:
                out["first"] = float(vals[0])
                out["last"] = float(vals[-1])
            if vals.size > 1:
                step = float(np.median(np.diff(vals)))
                u = out["units"].lower()
                if u.startswith("second"):
                    out["dt_hours"] = step / 3600.0
                elif u.startswith("minute"):
                    out["dt_hours"] = step / 60.0
                elif u.startswith("hour"):
                    out["dt_hours"] = step
                elif u.startswith("day"):
                    out["dt_hours"] = step * 24.0
                else:
                    out["note"] = f"unrecognised time units: {out['units']!r}"
            out["ok"] = True
    except Exception as e:                      # noqa: BLE001 - report, never abort a scan
        out["note"] = f"{type(e).__name__}: {e}"
    return out


def dt_label(dt_hours: float | None) -> str:
    if dt_hours is None:
        return "unknown"
    if abs(dt_hours - 1.0) < 1e-6:
        return "1hr"
    if abs(dt_hours - 3.0) < 1e-6:
        return "3hr"
    if abs(dt_hours - 6.0) < 1e-6:
        return "6hr"
    if abs(dt_hours - 24.0) < 1e-6:
        return "daily"
    return f"{dt_hours:g}h"


# Variable units
def read_var_units(nc_file: str, var: str) -> str:
    with Dataset(nc_file, "r") as ds:
        if var not in ds.variables:
            raise KeyError(f"Variable '{var}' not in {nc_file}")
        return attr_to_text(getattr(ds.variables[var], "units", "")).strip()


def find_data_variable(nc_file: str, preferred: str = "") -> str:
    
    with Dataset(nc_file, "r") as ds:
        if preferred and preferred in ds.variables:
            return preferred
        for name, v in ds.variables.items():
            if v.ndim >= 3 and any(d.lower() == "time" for d in v.dimensions):
                return name
    return ""
