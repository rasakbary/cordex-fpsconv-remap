"""
Build a CDO grid description file for a model's own horizontal grid.

Why this is needed
------------------
Conservative remapping works out how much each source cell overlaps each
target cell, so CDO needs the four corners of every source cell. Some of the
files only store cell centres, and CDO can't implement remapcon and give the message:

    Source grid cell corner coordinates missing!

So basically the reason is: A rotated-pole file stores three things:
the rotated axes rlon/rlat, a grid_mapping variable holding the pole
position, and 2D lon/lat arrays that the data variable points at with a
`coordinates = "lon lat"` attribute. CDO sees those 2D lon/lat arrays and
treats the grid as curvilinear, meaning an arbitrary list of cell centres. A
curvilinear grid with no bounds cannot be conservatively remapped, so CDO
stops. Even though the regular rotated axes it would need to work the
corners out itself are just in the same file.

What this script does
---------------------
It writes an explicit grid description and push it to CDO with -setgrid,
replacing the grid CDO inferred. Corners come from the best source available:

  1. Corners already in the file: 2D lon/lat carrying a `bounds` attribute
     that points at an (ny, nx, 4) array. Nothing is computed. The caller
     skips this module and calls remapcon directly.

  2. Cell edges already in the file: rlon_bnds / rlat_bnds, shape (n, 2).
     Those exact edges are converted from rotated to geographic coordinates.

  3. Edges taken as the midpoint between neighbouring cell centres, with half
     a cell added at each end. For an axis whose spacing is constant - true
     of every rotated grid we have - the midpoint is exactly the cell edge. 

Another thing worth noting is about the rotated pole:
---------------------------------------------------------
Converting rlon/rlat to lon/lat needs the position of the rotated north pole.
Two things go wrong with the value stored in the file:

  - some files give the south pole instead of the north pole, which is the
    same point 180 degrees away in longitude;
  - some files number rlon in the opposite direction.

Either one can put the domain thousands of kilometres away, and nothing in the metadata 
says which convention was used. Here the code converts the grid using all six
combinations of (pole longitude, pole longitude +/- 180) and (rlon, -rlon),
and compares each result against the 2D lon/lat arrays the file already
contains. Whichever combination that reproduces them is used, and the size of the
mismatch goes into the log; for an exact match this value must be near zero.

"""

from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from .nc_metadata import (
    coerce_attr_float,
    find_lonlat_2d,
    find_projection_axes,
    find_rotated_axes,
    get_grid_mapping,
)
from .logging_utils import log

# Default rotated pole for the EURO-CORDEX domain, used only when a file has
# rlon/rlat but no grid_mapping variable at all. 
EURO_CORDEX_POLE = {
    "grid_north_pole_longitude": -162.0,
    "grid_north_pole_latitude": 39.25,
    "north_pole_grid_longitude": 0.0,
}


# Longitude helpers
def wrap_lon(lon_deg):
    """Wrap longitudes into [-180, 180)."""
    return (np.asarray(lon_deg, dtype=float) + 180.0) % 360.0 - 180.0


def unwrap_lon_around(lon, lon0):
    """Shift `lon` into the branch nearest `lon0`."""
    return lon0 + ((np.asarray(lon, dtype=float) - lon0 + 180.0) % 360.0 - 180.0)


def axis_spacing(a) -> tuple[bool, float, float]:
    """Is the spacing along an axis constant? Returns (uniform, step, spread).

    `spread` is (largest step - smallest step) divided by the median step, so
    it can be compared across axes with different resolutions.
    """
    a = np.asarray(a, dtype=float).ravel()
    if a.size < 3:
        return True, 0.0, 0.0
    d = np.diff(a)
    step = float(np.median(d))
    if step == 0:
        return False, 0.0, float("inf")
    spread = float(d.max() - d.min()) / abs(step)
    return spread <= 1e-3, step, spread


def axis_edges(a) -> np.ndarray:
    """n cell centres -> n+1 cell edges (midpoints, half-cell extrapolated ends)."""
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    if n < 2:
        raise ValueError("Need at least 2 points to compute edges.")
    edges = np.empty(n + 1, dtype=float)
    edges[1:-1] = 0.5 * (a[:-1] + a[1:])
    edges[0] = a[0] - (edges[1] - a[0])
    edges[-1] = a[-1] + (a[-1] - edges[-2])
    return edges


# Rotated pole transform
def rotated_to_geo(rlon_deg, rlat_deg, pole_lon_deg, pole_lat_deg,
                   north_pole_grid_longitude_deg: float = 0.0,
                   *, flip_rlon: bool = False):
    """Rotated lon/lat -> geographic lon/lat (CF rotated_latitude_longitude).

    Returns (lon, lat) in degrees, lon wrapped to [-180, 180).
    """
    rlon_arr = np.asarray(rlon_deg, dtype=float)
    if flip_rlon:
        rlon_arr = -rlon_arr

    lam_r = np.deg2rad(rlon_arr + float(north_pole_grid_longitude_deg))
    phi_r = np.deg2rad(np.asarray(rlat_deg, dtype=float))
    lam_p = np.deg2rad(float(pole_lon_deg))
    phi_p = np.deg2rad(float(pole_lat_deg))

    sin_phi = (np.sin(phi_p) * np.sin(phi_r)
               + np.cos(phi_p) * np.cos(phi_r) * np.cos(lam_r))
    phi = np.arcsin(np.clip(sin_phi, -1.0, 1.0))

    y = -np.cos(phi_r) * np.sin(lam_r)
    x = (np.cos(phi_p) * np.sin(phi_r)
         - np.sin(phi_p) * np.cos(phi_r) * np.cos(lam_r))
    lam = lam_p + np.arctan2(y, x)

    return wrap_lon(np.rad2deg(lam)), np.rad2deg(phi)


def _rmse_deg(dlon, dlat) -> float:
    """RMSE of an angular difference, wrapping the longitude component."""
    dlon = (np.asarray(dlon, dtype=float) + 180.0) % 360.0 - 180.0
    return float(np.sqrt(np.nanmean(dlon ** 2) + np.nanmean(np.asarray(dlat) ** 2)))


def choose_rotated_fix(rlon2d, rlat2d, pole_lon, pole_lat, npg_lon,
                       lon_ref=None, lat_ref=None) -> tuple[float, bool, float]:
    """Pick the (pole_lon, flip_rlon) combination that reproduces the file.

    Returns (pole_lon_used, flip_rlon, score). Lower score is better; a score
    near 0 with a reference present means an exact match.
    """
    best = (None, None, None)
    for plon in (pole_lon, pole_lon + 180.0, pole_lon - 180.0):
        for flip in (False, True):
            lon_c, lat_c = rotated_to_geo(
                rlon2d, rlat2d, plon, pole_lat,
                north_pole_grid_longitude_deg=npg_lon, flip_rlon=flip,
            )
            lon_c = wrap_lon(lon_c)

            if lon_ref is not None and lat_ref is not None:
                score = _rmse_deg(lon_c - lon_ref, lat_c - lat_ref)
            else:

                lon_min, lon_max = float(np.nanmin(lon_c)), float(np.nanmax(lon_c))
                lat_min, lat_max = float(np.nanmin(lat_c)), float(np.nanmax(lat_c))
                score = 0.0
                if lon_max < -30 or lon_min > 60 or lat_max < 30 or lat_min > 70:
                    score += 1e6

            if best[2] is None or score < best[2]:
                best = (plon, flip, score)

    return float(best[0]), bool(best[1]), float(best[2])


# Corner ordering and file writing
def order_vertices_sw_se_ne_nw(lon4, lat4, lon_center):
    """Reorder 4 corners per cell into CDO's expected SW, SE, NE, NW sequence.

    Note that CDO does not check corner ordering; so a wrongly ordered
    cell silently produces a near-zero overlap area.
    """
    lon4u = unwrap_lon_around(lon4, np.asarray(lon_center)[..., None])

    idx_lat = np.argsort(lat4, axis=-1)
    s_idx, n_idx = idx_lat[..., :2], idx_lat[..., 2:]

    s_ord = np.argsort(np.take_along_axis(lon4u, s_idx, axis=-1), axis=-1)
    sw_idx = np.take_along_axis(s_idx, s_ord[..., :1], axis=-1)
    se_idx = np.take_along_axis(s_idx, s_ord[..., 1:], axis=-1)

    n_ord = np.argsort(np.take_along_axis(lon4u, n_idx, axis=-1), axis=-1)
    nw_idx = np.take_along_axis(n_idx, n_ord[..., :1], axis=-1)
    ne_idx = np.take_along_axis(n_idx, n_ord[..., 1:], axis=-1)

    idx = np.concatenate([sw_idx, se_idx, ne_idx, nw_idx], axis=-1)
    return np.take_along_axis(lon4u, idx, axis=-1), np.take_along_axis(lat4, idx, axis=-1)


def _write_vals_block(f, key: str, vals, per_line: int = 8, fmt: str = "{:.10f}"):
    """Write `key = v v v ...` wrapped over several aligned lines."""
    vals = np.asarray(vals, dtype=float).ravel()
    prefix = f"{key} = "
    cont = " " * len(prefix)
    for i in range(0, len(vals), per_line):
        chunk = " ".join(fmt.format(v) for v in vals[i:i + per_line])
        f.write((prefix if i == 0 else cont) + chunk + "\n")


# Rotated -> curvilinear source grid
def write_rotated_curvilinear_srcgrid(nc_file: str, out_gridfile: str,
                                      var: str = "pr",
                                      pole_override: dict | None = None) -> dict:
    """Write a `gridtype = curvilinear` CDO grid file for a rotated-pole model.
    Returns a small dict describing what was done, for logging and reports.
    """
    with Dataset(nc_file, "r") as ds:
        gm_var, gm_name = get_grid_mapping(nc_file, var)

        if pole_override is not None:
            pole_lon = float(pole_override["grid_north_pole_longitude"])
            pole_lat = float(pole_override["grid_north_pole_latitude"])
            npg_lon = float(pole_override.get("north_pole_grid_longitude", 0.0))
            pole_source = "override"
        else:
            if gm_name != "rotated_latitude_longitude":
                raise ValueError(
                    f"Not a rotated_latitude_longitude grid: {gm_name!r} in {nc_file}"
                )
            if not gm_var or gm_var not in ds.variables:
                raise KeyError(f"Grid mapping variable {gm_var!r} not found in {nc_file}")
            gm = ds.variables[gm_var]
            pole_lon = coerce_attr_float(getattr(gm, "grid_north_pole_longitude"))
            pole_lat = coerce_attr_float(getattr(gm, "grid_north_pole_latitude"))
            npg_lon = coerce_attr_float(
                getattr(gm, "north_pole_grid_longitude", 0.0), default=0.0
            )
            pole_source = "file"

        rlon_name, rlat_name = find_rotated_axes(ds)
        rlon = np.asarray(ds.variables[rlon_name][:], dtype=float)
        rlat = np.asarray(ds.variables[rlat_name][:], dtype=float)
        xsize, ysize = int(rlon.size), int(rlat.size)
        rlon2d, rlat2d = np.meshgrid(rlon, rlat)

        # Reference lon/lat from the file.
        lon_ref_name, lat_ref_name = find_lonlat_2d(ds)
        lon_ref = (wrap_lon(np.asarray(ds.variables[lon_ref_name][:], dtype=float))
                   if lon_ref_name else None)
        lat_ref = (np.asarray(ds.variables[lat_ref_name][:], dtype=float)
                   if lat_ref_name else None)

        pole_lon_used, flip_rlon, score = choose_rotated_fix(
            rlon2d, rlat2d, pole_lon, pole_lat, npg_lon, lon_ref, lat_ref
        )
        log(f"   rotated-pole fix: pole_lon={pole_lon_used:.6f} "
            f"(from {pole_source}), flip_rlon={flip_rlon}, score={score:.6g}")
        if lon_ref is None and score >= 1e5:
            log("   [WARN] no 2D lon/lat in file AND no candidate landed over Europe. "
                "Verify the output before using it.")

        # cell centres:from the file's own lon/lat
        if lon_ref is not None and lat_ref is not None:
            lon2d, lat2d = lon_ref, lat_ref
            centre_source = "file 2D lon/lat"
        else:
            lon2d, lat2d = rotated_to_geo(
                rlon2d, rlat2d, pole_lon_used, pole_lat,
                north_pole_grid_longitude_deg=npg_lon, flip_rlon=flip_rlon,
            )
            lon2d = wrap_lon(lon2d)
            centre_source = "computed from rlon/rlat"

        lat_min, lat_max = float(np.nanmin(lat2d)), float(np.nanmax(lat2d))
        lon_min, lon_max = float(np.nanmin(lon2d)), float(np.nanmax(lon2d))
        log(f"   geo extent: lon [{lon_min:.2f}, {lon_max:.2f}]  "
            f"lat [{lat_min:.2f}, {lat_max:.2f}]  (centres from {centre_source})")
        if not (-90.0 <= lat_min <= 90.0 and -90.0 <= lat_max <= 90.0):
            raise RuntimeError(
                "Computed latitudes outside [-90, 90]; the rotated->geo transform "
                "is wrong. Refusing to write a corrupt source grid."
            )

        # Cell corners, taken from the best source available.
        c_lon = c_lat = None
        corner_source = ""

        # 1: native 2D lon/lat vertices.
        if lon_ref_name and lat_ref_name:
            lon_b = str(getattr(ds.variables[lon_ref_name], "bounds", "")).strip()
            lat_b = str(getattr(ds.variables[lat_ref_name], "bounds", "")).strip()
            if lon_b and lat_b and lon_b in ds.variables and lat_b in ds.variables:
                lbv, abv = ds.variables[lon_b], ds.variables[lat_b]
                if (lbv.ndim == 3 and abv.ndim == 3
                        and lbv.shape[-1] == 4 and abv.shape[-1] == 4):
                    c_lon = wrap_lon(np.asarray(lbv[:], dtype=float))
                    c_lat = np.asarray(abv[:], dtype=float)
                    corner_source = "native 2D lon/lat vertices"

        # 2: native 1D rotated-axis bounds.
        if c_lon is None:
            rlon_b = str(getattr(ds.variables[rlon_name], "bounds", "")).strip()
            rlat_b = str(getattr(ds.variables[rlat_name], "bounds", "")).strip()
            if (rlon_b and rlat_b and rlon_b in ds.variables and rlat_b in ds.variables
                    and ds.variables[rlon_b].ndim == 2
                    and ds.variables[rlat_b].ndim == 2):
                rlon_bnds = np.asarray(ds.variables[rlon_b][:], dtype=float)  # (nx, 2)
                rlat_bnds = np.asarray(ds.variables[rlat_b][:], dtype=float)  # (ny, 2)
                west = np.tile(rlon_bnds[:, 0], (ysize, 1))
                east = np.tile(rlon_bnds[:, 1], (ysize, 1))
                south = np.tile(rlat_bnds[:, 0][:, None], (1, xsize))
                north = np.tile(rlat_bnds[:, 1][:, None], (1, xsize))
                c_rlon = np.stack([west, east, east, west], axis=-1)
                c_rlat = np.stack([south, south, north, north], axis=-1)
                c_lon, c_lat = rotated_to_geo(
                    c_rlon, c_rlat, pole_lon_used, pole_lat,
                    north_pole_grid_longitude_deg=npg_lon, flip_rlon=flip_rlon,
                )
                c_lon = wrap_lon(c_lon)
                corner_source = "native rlon_bnds/rlat_bnds"

        # Last option: we work the edges out from the cell centres. For an axis
        # with constant spacing the midpoint between two centres is the cell edge, 
        # but that only holds if the spacing is really constant.
        if c_lon is None:
            ok_x, step_x, spread_x = axis_spacing(rlon)
            ok_y, step_y, spread_y = axis_spacing(rlat)
            log(f"   axis spacing: rlon step {step_x:.6f} (spread {spread_x:.1e}), "
                f"rlat step {step_y:.6f} (spread {spread_y:.1e})")
            if not (ok_x and ok_y):
                log("   [WARN] the spacing varies by more than 0.1% of a step, so "
                    "cell edges taken as midpoints between centres are only "
                    "approximate. Check this model's output.")
            rlon_e, rlat_e = axis_edges(rlon), axis_edges(rlat)
            # Respect descending axes: keep west < east and south < north.
            if rlon.size >= 2 and rlon[1] >= rlon[0]:
                west_e, east_e = rlon_e[:-1], rlon_e[1:]
            else:
                west_e, east_e = rlon_e[1:], rlon_e[:-1]
            if rlat.size >= 2 and rlat[1] >= rlat[0]:
                south_e, north_e = rlat_e[:-1], rlat_e[1:]
            else:
                south_e, north_e = rlat_e[1:], rlat_e[:-1]

            west = np.tile(west_e, (ysize, 1))
            east = np.tile(east_e, (ysize, 1))
            south = np.tile(south_e[:, None], (1, xsize))
            north = np.tile(north_e[:, None], (1, xsize))
            c_rlon = np.stack([west, east, east, west], axis=-1)
            c_rlat = np.stack([south, south, north, north], axis=-1)
            c_lon, c_lat = rotated_to_geo(
                c_rlon, c_rlat, pole_lon_used, pole_lat,
                north_pole_grid_longitude_deg=npg_lon, flip_rlon=flip_rlon,
            )
            c_lon = wrap_lon(c_lon)
            corner_source = "inferred edges (midpoints between centres)"

        log(f"   cell corners from: {corner_source}")

        c_lon = unwrap_lon_around(c_lon, np.asarray(lon2d)[..., None])
        c_lon, c_lat = order_vertices_sw_se_ne_nw(c_lon, c_lat, lon2d)

    _write_curvilinear_file(out_gridfile, lon2d, lat2d, c_lon, c_lat, xsize, ysize)

    return {
        "gridtype": "curvilinear",
        "xsize": xsize,
        "ysize": ysize,
        "pole_lon_used": pole_lon_used,
        "pole_lat": pole_lat,
        "flip_rlon": flip_rlon,
        "fit_score": score,
        "pole_source": pole_source,
        "centre_source": centre_source,
        "corner_source": corner_source,
    }


def _write_curvilinear_file(out_gridfile, lon2d, lat2d, c_lon, c_lat, xsize, ysize):
    with open(out_gridfile, "w", encoding="utf-8") as f:
        f.write("gridtype  = curvilinear\n")
        f.write(f"xsize     = {xsize}\n")
        f.write(f"ysize     = {ysize}\n")
        f.write(f"gridsize  = {xsize * ysize}\n")
        f.write("xname     = lon\n")
        f.write("yname     = lat\n")
        f.write('xunits    = "degrees_east"\n')
        f.write('yunits    = "degrees_north"\n')
        _write_vals_block(f, "xvals", np.asarray(lon2d).ravel(order="C"))
        _write_vals_block(f, "yvals", np.asarray(lat2d).ravel(order="C"))
        _write_vals_block(f, "xbounds",
                          np.asarray(c_lon).reshape(-1, 4, order="C").ravel(order="C"))
        _write_vals_block(f, "ybounds",
                          np.asarray(c_lat).reshape(-1, 4, order="C").ravel(order="C"))


# Lambert conformal conic -> projection source grid
def clean_proj4(p: str) -> str:
    p = p.strip()
    if p.lower().startswith(("proj ", "proj\t")):
        p = p[5:].strip()
    return p


def write_lambert_srcgrid(nc_file: str, out_gridfile: str, var: str = "pr") -> dict:
    """Write a `gridtype = projection` CDO grid file for a Lambert model.

    For Lambart cases we do NOT reconstruct corners ourselves: we give CDO the 
    projection definition and a regular x/y axis, and let PROJ derive the geographic
    corners.
    """
    with Dataset(nc_file, "r") as ds:
        gm_var, gm_name = get_grid_mapping(nc_file, var)
        if gm_name != "lambert_conformal_conic":
            raise ValueError(f"Not a lambert_conformal_conic grid: {gm_name!r}")
        if not gm_var or gm_var not in ds.variables:
            raise KeyError(f"Grid mapping variable {gm_var!r} not found in {nc_file}")

        xname, yname = find_projection_axes(ds)
        x = np.asarray(ds.variables[xname][:], dtype=float)
        y = np.asarray(ds.variables[yname][:], dtype=float)

        # CDO expects metres; several models store kilometres.
        xunits = str(getattr(ds.variables[xname], "units", "")).strip().lower()
        yunits = str(getattr(ds.variables[yname], "units", "")).strip().lower()
        if xunits in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            x = x * 1000.0
        if yunits in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
            y = y * 1000.0

        xsize, ysize = int(x.size), int(y.size)
        dx = float(np.median(np.diff(x)))
        dy = float(np.median(np.diff(y)))
        if not np.allclose(np.diff(x), dx, atol=1e-6):
            raise ValueError(f"{xname} is not uniformly spaced; cannot use xfirst/xinc.")
        if not np.allclose(np.diff(y), dy, atol=1e-6):
            raise ValueError(f"{yname} is not uniformly spaced; cannot use yfirst/yinc.")

        gm = ds.variables[gm_var]
        attrs = {}
        for k in ("standard_parallel", "longitude_of_central_meridian",
                  "latitude_of_projection_origin", "false_easting", "false_northing",
                  "earth_radius", "semi_major_axis", "semi_minor_axis",
                  "inverse_flattening"):
            if hasattr(gm, k):
                attrs[k] = getattr(gm, k)

        proj_params = ""
        for key in ("proj4", "proj4_params", "proj4_param"):
            if hasattr(gm, key):
                proj_params = clean_proj4(str(getattr(gm, key)))
                break

    lines = [
        "gridtype  = projection",
        f"xsize     = {xsize}",
        f"ysize     = {ysize}",
        "xname     = x",
        "yname     = y",
        'xunits    = "meter"',
        'yunits    = "meter"',
        f"xfirst    = {float(x[0]):.10f}",
        f"xinc      = {dx:.10f}",
        f"yfirst    = {float(y[0]):.10f}",
        f"yinc      = {dy:.10f}",
        "grid_mapping_name = lambert_conformal_conic",
    ]
    for k, v in attrs.items():
        arr = np.asarray(v).ravel()
        if arr.size == 1:
            lines.append(f"{k} = {float(arr[0]):.15g}")
        else:
            lines.append(f"{k} = " + " ".join(f"{float(a):.15g}" for a in arr))
    if proj_params:
        lines.append(f'proj_params = "{proj_params}"')

    with open(out_gridfile, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "gridtype": "projection",
        "xsize": xsize,
        "ysize": ysize,
        "xinc_m": dx,
        "yinc_m": dy,
        "proj_params": proj_params,
    }
