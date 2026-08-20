"""
Repair NetCDF attributes that have the wrong type and make CDO fail.

Some few models are unusal and store attributes as text where a number is required:
1. grid_mapping = "rotated pole", with a space instead of an underscore, so no variable of that name 
exists and the grid mapping is invisible;
2. _FillValue as the 7-character string "1e+20  " rather than a float;
3. missing_value the same.

For these cases the netCDF4 library reads these without complaint, so the files look fine.
However, CDO will fail when writing output, with a message like:
Error (cdf_put_att_text): NetCDF: Not a valid data type or _FillValue type mismatch

_FillValue cannot be changed afterwards; the NetCDF library fixes it when
the variable is created. So we need to rebuild the file with a
numeric fill value supplied at creation.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from netCDF4 import Dataset

from .nc_metadata import attr_to_text
from .logging_utils import log


_COPY_CHUNK_STEPS = 240        # this chunk is like 10 days at hourly resolution


def _parse_to_number(val, target_dtype):
    """Parse a textual attribute into a scalar of `target_dtype`."""
    arr = np.asarray(val)
    if arr.dtype.kind in ("i", "u", "f"):
        return arr.astype(target_dtype).ravel()[0]
    s = attr_to_text(val).strip()
    try:
        return np.array([float(s)], dtype=target_dtype)[0]
    except Exception:                                    
        return np.array([1.0e20], dtype=target_dtype)[0]


def _is_text_like(val) -> bool:
    if isinstance(val, (str, bytes, np.bytes_, np.str_)):
        return True
    return isinstance(val, np.ndarray) and val.dtype.kind in ("O", "U", "S")


def needs_fixing(nc_file: str, variable: str) -> tuple[bool, list[str]]:
    """Returns (needs_repair and list_of_reasons)."""
    reasons: list[str] = []
    try:
        with Dataset(nc_file, "r") as ds:
            if variable not in ds.variables:
                return False, []
            v = ds.variables[variable]

            gm = attr_to_text(getattr(v, "grid_mapping", "")).strip()
            if gm and " " in gm:
                reasons.append(f"grid_mapping contains a space: {gm!r}")

            for attr in ("_FillValue", "missing_value"):
                if attr in v.ncattrs() and _is_text_like(v.getncattr(attr)):
                    reasons.append(
                        f"{attr} is stored as text: "
                        f"{attr_to_text(v.getncattr(attr))!r}"
                    )
    except Exception as e:                               # noqa: BLE001
        reasons.append(f"could not read metadata ({type(e).__name__}: {e})")
        return True, reasons

    return bool(reasons), reasons


def fix_file(src_path: str, dst_path: str, variable: str) -> str:
    """Rebuild one NetCDF file with correctly typed attributes."""
    log(f"   sanitising {os.path.basename(src_path)}")
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing_value.*",
                                category=UserWarning)
        _rebuild(src_path, dst_path, variable)

    return dst_path


def _rebuild(src_path: str, dst_path: str, variable: str) -> None:
    with Dataset(src_path, "r") as src, \
            Dataset(dst_path, "w", format="NETCDF4") as dst:

        # global attributes
        for aname in src.ncattrs():
            aval = src.getncattr(aname)
            if _is_text_like(aval):
                aval = attr_to_text(aval)
            try:
                dst.setncattr(aname, aval)
            except Exception as e:                       # noqa: BLE001
                log(f"      [WARN] skipped global attr {aname}: {e}")

        # dimensions
        for dname, dim in src.dimensions.items():
            dst.createDimension(dname, None if dim.isunlimited() else len(dim))

        # variables
        for vname, src_v in src.variables.items():
            attrs = {a: src_v.getncattr(a) for a in src_v.ncattrs()}

            # _FillValue must be passed at creation time; it cannot be set later.
            fill_arg = None
            if "_FillValue" in attrs:
                fill_arg = _parse_to_number(attrs.pop("_FillValue"), src_v.dtype)

            use_zlib = (vname == variable and src_v.ndim >= 2)
            chunksizes = None
            if use_zlib and "time" in src_v.dimensions:
                chunksizes = tuple(
                    1 if d == "time" else len(src.dimensions[d])
                    for d in src_v.dimensions
                )

            new_v = dst.createVariable(
                vname, src_v.dtype, dimensions=src_v.dimensions,
                zlib=use_zlib, complevel=1 if use_zlib else 0,
                shuffle=use_zlib, chunksizes=chunksizes, fill_value=fill_arg,
            )

            for aname, aval in attrs.items():
                if (vname == variable and aname == "grid_mapping"
                        and isinstance(aval, str) and " " in aval.strip()):
                    fixed = aval.strip().replace(" ", "_")
                    aval = "rotated_pole" if "rotated_pole" in src.variables else fixed

                if aname == "missing_value" and np.asarray(aval).dtype != src_v.dtype:
                    aval = _parse_to_number(aval, src_v.dtype)

                if _is_text_like(aval) and aname != "grid_mapping":
                    aval = attr_to_text(aval)

                try:
                    new_v.setncattr(aname, aval)
                except Exception as e:                   # noqa: BLE001
                    log(f"      [WARN] skipped {vname}.{aname}: {e}")

            # data
            if src_v.ndim == 0:
                new_v[...] = src_v[...]
            elif vname == variable and "time" in src_v.dimensions:
                t_idx = src_v.dimensions.index("time")
                nt = src_v.shape[t_idx]
                for i in range(0, nt, _COPY_CHUNK_STEPS):
                    sl = [slice(None)] * src_v.ndim
                    sl[t_idx] = slice(i, min(i + _COPY_CHUNK_STEPS, nt))
                    new_v[tuple(sl)] = src_v[tuple(sl)]
            else:
                new_v[...] = src_v[...]


def fix_if_needed(files: list[str], variable: str, out_dir: str,
                       *, dry_run: bool = False) -> tuple[list[str], list[str]]:

    resolved: list[str] = []
    created: list[str] = []
    checked_first = False

    for fp in files:
        needs, reasons = needs_fixing(fp, variable)
        if not needs:
            resolved.append(fp)
            continue

        if not checked_first:
            log("   attributes with the wrong type found; repairing before merge:")
            for r in reasons:
                log(f"      - {r}")
            checked_first = True

        dst = os.path.join(out_dir, f"fixed_{os.path.basename(fp)}")
        if dry_run:
            log(f"   DRY-RUN would sanitise {os.path.basename(fp)}")
            resolved.append(fp)
            continue

        fix_file(fp, dst, variable)
        resolved.append(dst)
        created.append(dst)

    return resolved, created
