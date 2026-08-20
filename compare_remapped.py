#!/usr/bin/env python3
"""
compare_remapped.py - check a remapped file against its source.

To check the target grid against the source grid, here we pick random points from each quadrant of the 
domain in the target grid and matched to the nearest source cell. By this we try to see if the values
actually landed in the right place.

Use as following:
python compare_remapped.py --variable pr --outdir ./check_BTU 
--test /out/BTU_pr_Historical_merged.nc --source /archive/ALP-3/1hr/pr/historical/pr_ALP-3_..._CLMcom-BTU-*.nc

"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fpsremap import units

try:
    from netCDF4 import Dataset, num2date
except ImportError:                                      
    sys.exit("netCDF4 is required. Install with: conda install -c conda-forge netcdf4")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False



def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Check a remapped file against its source file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples\n", 1)[-1],
    )
    p.add_argument("--test", required=True,
                   help="The merged, remapped file to check.")
    p.add_argument("--source", nargs="+", required=True,
                   help="The native-grid files.")
    p.add_argument("--points-per-panel", type=int, default=200,
                   help="Points drawn from each of the four quadrants (default: 200).")
    p.add_argument("--n-times", type=int, default=500,
                   help="Timesteps sampled across the record (default: 500). ")
    p.add_argument("--source-units", default=None,
                   help="Force the units of the source files, for the few whose "
                        "`units` attribute is wrong. Mirrors `units_override` in "
                        "the model registry, e.g. --source-units 'kg m-2 s-1'.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for point selection, so a run repeats (default: 0).")
    p.add_argument("--variable", default="pr", help="Variable name (default: pr).")
    p.add_argument("--outdir", default="./remap_compare", help="Where to write results.")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG output.")
    return p.parse_args(argv)


# Grid alignment
def read_axes(nc_path: str) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, "r") as ds:
        for lon_name in ("lon", "longitude", "x"):
            if lon_name in ds.variables and ds.variables[lon_name].ndim == 1:
                break
        else:
            raise KeyError(f"{nc_path}: no 1D longitude axis found")
        for lat_name in ("lat", "latitude", "y"):
            if lat_name in ds.variables and ds.variables[lat_name].ndim == 1:
                break
        else:
            raise KeyError(f"{nc_path}: no 1D latitude axis found")
        return (np.asarray(ds.variables[lon_name][:], dtype=float),
                np.asarray(ds.variables[lat_name][:], dtype=float))


# check the source files
def read_source_lonlat(nc_path: str) -> tuple[np.ndarray, np.ndarray]:
    """2D lon/lat of a native-grid file."""
    with Dataset(nc_path, "r") as ds:
        lon = lat = None
        for name, v in ds.variables.items():
            sn = str(getattr(v, "standard_name", "")).strip().lower()
            if v.ndim == 2 and (sn == "longitude" or name in ("lon", "longitude")):
                lon = np.asarray(v[:], dtype=float)
            if v.ndim == 2 and (sn == "latitude" or name in ("lat", "latitude")):
                lat = np.asarray(v[:], dtype=float)
        if lon is None or lat is None:
            raise KeyError(f"{nc_path}: no 2D lon/lat found; is this a native-grid file?")
    return lon, lat


def pick_quadrant_points(lon1d, lat1d, per_panel: int, seed: int = 0):
    """Choose random target cells from each quadrant of the target grid.

    Returns a list of dicts with the panel name and the cell's index and position.
    """
    rng = np.random.default_rng(seed)
    ny, nx = lat1d.size, lon1d.size
    mid_x, mid_y = nx // 2, ny // 2
    quadrants = {
        "NW": (range(mid_y, ny), range(0, mid_x)),
        "NE": (range(mid_y, ny), range(mid_x, nx)),
        "SW": (range(0, mid_y), range(0, mid_x)),
        "SE": (range(0, mid_y), range(mid_x, nx)),
    }
    points = []
    for name, (jr, ir) in quadrants.items():
        js, its = np.array(list(jr)), np.array(list(ir))
        n = min(per_panel, js.size * its.size)
        flat = rng.choice(js.size * its.size, size=n, replace=False)
        for f in flat:
            j, i = js[f // its.size], its[f % its.size]
            points.append({"panel": name, "j": int(j), "i": int(i),
                           "lon": float(lon1d[i]), "lat": float(lat1d[j])})
    return points


def match_to_source(points, src_lon2d, src_lat2d):
    """Find the source cell nearest each target point, and its distance difference."""
    slon = src_lon2d.ravel()
    slat = src_lat2d.ravel()
    ny, nx = src_lon2d.shape
    coslat = np.cos(np.deg2rad(np.mean(src_lat2d)))

    for start in range(0, len(points), 64):          
        block = points[start:start + 64]
        tl = np.array([p["lon"] for p in block])[:, None]
        tt = np.array([p["lat"] for p in block])[:, None]
        d2 = ((slon[None, :] - tl) * coslat) ** 2 + (slat[None, :] - tt) ** 2
        k = np.argmin(d2, axis=1)
        for p, kk, dd in zip(block, k, np.sqrt(d2[np.arange(len(block)), k])):
            p["sj"], p["si"] = int(kk // nx), int(kk % nx)
            p["dist_km"] = float(dd) * 111.0
    return points


def _decode_times(ds, tname="time"):
    v = ds.variables[tname]
    return num2date(v[:], units=v.units,
                    calendar=getattr(v, "calendar", "standard"),
                    only_use_cftime_datetimes=True)


def source_unit_factors(source_files, variable, override=None):
    """The conversion factor for EACH source file, read from that file.

    Note that units must be read per file becuase few cases in this archive label one
    year as a flux in kg m-2 s-1 and the next as mm/h.
    """
    factors, seen = {}, {}
    for fp in source_files:
        if override:
            u = override
        else:
            with Dataset(fp, "r") as ds:
                u = str(getattr(ds.variables[variable], "units", "")).strip()
        f, target = units.factor_and_target(u, variable)
        factors[fp] = f
        seen.setdefault((u, f, target), []).append(os.path.basename(fp))
    return factors, seen


def gather_source_series(source_files, variable, points, n_times, factors):
    """Read the sampled points in the source, at a subset of timesteps."""
    sj = np.array([p["sj"] for p in points])
    si = np.array([p["si"] for p in points])

    index = []                                       
    for fp in source_files:
        with Dataset(fp, "r") as ds:
            nt = ds.variables[variable].shape[0]
        index += [(fp, k) for k in range(nt)]

    take = np.linspace(0, len(index) - 1, min(n_times, len(index))).astype(int)
    take = sorted(set(take.tolist()))

    values, stamps = [], []
    by_file: dict[str, list[int]] = {}
    for t in take:
        fp, k = index[t]
        by_file.setdefault(fp, []).append(k)

    for fp in source_files:
        if fp not in by_file:
            continue
        factor = factors[fp]                         
        with Dataset(fp, "r") as ds:
            var = ds.variables[variable]
            times = _decode_times(ds)
            for k in by_file[fp]:
                m = np.ma.filled(np.asarray(var[k, :, :], dtype=float), np.nan)
                values.append(m[sj, si] * factor)
                stamps.append(str(times[k]))
        print(f"    source: {len(values)}/{len(take)} timesteps", end="\r", flush=True)
    print(" " * 60, end="\r")
    return np.array(values), stamps


def gather_test_series(test_file, variable, points, stamps):
    """The same points and the same timestamps, in the remapped file."""
    tj = np.array([p["j"] for p in points])
    ti = np.array([p["i"] for p in points])
    with Dataset(test_file, "r") as ds:
        var = ds.variables[variable]
        lookup = {str(t): k for k, t in enumerate(_decode_times(ds))}
        rows, keep = [], []
        for n, st in enumerate(stamps):
            k = lookup.get(st)
            if k is None:
                continue
            m = np.ma.filled(np.asarray(var[k, :, :], dtype=float), np.nan)
            rows.append(m[tj, ti])
            keep.append(n)
    return np.array(rows), keep


def summarise_panels(points, src, test):
    """One row per quadrant."""
    panels = ["NW", "NE", "SW", "SE"]
    rows = []
    for name in panels + ["ALL"]:
        sel = np.array([True] * len(points)) if name == "ALL" else \
              np.array([p["panel"] == name for p in points])
        a, b = src[:, sel].ravel(), test[:, sel].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if a.size == 0:
            rows.append((name, 0, *[float("nan")] * 6))
            continue
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        bias = float(b.mean() - a.mean())
        rel = 100.0 * bias / a.mean() if a.mean() != 0 else float("nan")
        rows.append((name, int(sel.sum()), float(a.mean()), float(b.mean()),
                     bias, rel, corr, float(np.abs(b - a).max())))
    return rows



def point_check_against_source(args) -> int:
    """Compare sampled target cells with the nearest source cells."""
    missing = [f for f in args.source if not os.path.isfile(f)]
    if missing:
        print(f"ERROR: source file(s) not found: {missing[:3]}", file=sys.stderr)
        return 2
    source_files = sorted(args.source)

    lon_t, lat_t = read_axes(args.test)
    src_lon, src_lat = read_source_lonlat(source_files[0])
    print(f"target grid : {lon_t.size} x {lat_t.size}")
    print(f"source grid : {src_lon.shape[1]} x {src_lon.shape[0]}  "
          f"({len(source_files)} file(s))")

    
    try:
        factors, seen = source_unit_factors(source_files, args.variable,
                                            args.source_units)
    except (ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print("units       : read from each file separately")
    for (u, f, target), names in sorted(seen.items()):
        print(f"   '{u}' x{f:g} -> '{target}'   ({len(names)} file(s))")
    if len(seen) > 1:
        print("   note: this model labels its files inconsistently, which is why "
              "the factor is taken per file")

    points = pick_quadrant_points(lon_t, lat_t, args.points_per_panel, args.seed)
    points = match_to_source(points, src_lon, src_lat)

    d = np.array([p["dist_km"] for p in points])
    print(f"nearest source cell: median {np.median(d):.2f} km, worst {d.max():.2f} km")
    far = d > 10.0
    if far.any():
        print(f"  [WARN] {int(far.sum())} point(s) are more than 10 km from any "
              f"source cell and are dropped - the target may reach outside the source")
        points = [p for p, f in zip(points, far) if not f]
    if not points:
        print("ERROR: no target point has a nearby source cell. The two grids do "
              "not overlap, which means the remap went VERY VEERY wrong!!",
              file=sys.stderr)
        return 2

    print(f"\nreading {args.n_times} timesteps at {len(points)} points")
    src, stamps = gather_source_series(source_files, args.variable, points,
                                       args.n_times, factors)
    test, keep = gather_test_series(args.test, args.variable, points, stamps)
    if not keep:
        print("ERROR: no timestamp in the source matched one in the remapped file.",
              file=sys.stderr)
        return 2
    src = src[keep]
    print(f"matched {len(keep)} timesteps in both files")

    rows = summarise_panels(points, src, test)

    width = 88
    lines = ["", "=" * width,
             f"RANDOM CHECK AGAINST THE SOURCE  |  {args.points_per_panel} points per "
             f"quadrant, {len(keep)} timesteps", "=" * width,
             f"{'panel':<8}{'points':>8}{'mean src':>12}{'mean out':>12}"
             f"{'bias':>12}{'rel bias':>11}{'corr':>9}{'max diff':>12}",
             "-" * width]
    for name, n, ms, mt, bias, rel, corr, mx in rows:
        if name == "ALL":
            lines.append("-" * width)
        lines.append(f"{name:<8}{n:>8}{ms:>12.5f}{mt:>12.5f}"
                     f"{bias:>12.5f}{rel:>10.2f}%{corr:>9.4f}{mx:>12.4f}")
    lines.append("=" * width)

    _, _, _, _, _, rel_all, corr_all, _ = rows[-1]
    lines.append("")
    if np.isfinite(corr_all) and corr_all > 0.99:
        lines.append(f"PASS: correlation {corr_all:.4f}.")
    elif np.isfinite(corr_all) and corr_all > 0.9:
        lines.append(f"CHECK: correlation {corr_all:.4f}. Expected above 0.99 when "
                     f"source and target resolutions are similar. Worth a look at "
                     f"the scatter plot.")
    else:
        lines.append(f"FAIL: correlation {corr_all:.4f}. The remapped values do not "
                     f"track the source. Suspect the grid description - a mirrored "
                     f"axis, the wrong pole, or an offset.")
    if np.isfinite(rel_all) and abs(rel_all) > 5:
        lines.append(f"FAIL: mean is off by {rel_all:+.1f}%. Check the unit "
                     f"conversion first, then the grid.")

    text = "\n".join(lines)
    print(text)

    out_txt = os.path.join(args.outdir, "point_check.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"test   : {args.test}\nsource : {len(source_files)} file(s), "
                f"first {source_files[0]}\n{text}\n")
    with open(os.path.join(args.outdir, "point_check.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "target_j", "target_i", "lon", "lat",
                    "source_j", "source_i", "dist_km", "mean_source", "mean_test"])
        for n, p in enumerate(points):
            w.writerow([p["panel"], p["j"], p["i"], f"{p['lon']:.4f}", f"{p['lat']:.4f}",
                        p["sj"], p["si"], f"{p['dist_km']:.3f}",
                        f"{np.nanmean(src[:, n]):.6f}", f"{np.nanmean(test[:, n]):.6f}"])
    print(f"\nWrote {out_txt}")

    if not args.no_plots and HAVE_MPL:
        plot_point_check(points, src, test, args.outdir, args.variable)
    return 0


def plot_point_check(points, src, test, outdir, variable):
    """One scatter per quadrant: source value against remapped value."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    for ax, name in zip(axes.ravel(), ("NW", "NE", "SW", "SE")):
        sel = np.array([p["panel"] == name for p in points])
        a, b = src[:, sel].ravel(), test[:, sel].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        ax.scatter(a[ok], b[ok], s=15, alpha=0.25, edgecolors="none")
        hi = float(np.nanmax([a[ok].max(initial=0), b[ok].max(initial=0)])) or 1.0
        ax.plot([0, hi], [0, hi], lw=1, color="k")
        ax.set_title(f"{name}  ({int(sel.sum())} points)", fontsize=14)
        ax.set_xlabel(f"source {variable}", fontsize=13)
        ax.set_ylabel(f"remapped {variable}", fontsize=13)
    fig.suptitle("Comparing source grid against target grids", fontsize=14)
    fig.savefig(os.path.join(outdir, "point_check_scatter.png"), dpi=130)
    plt.close(fig)
    print(f"Wrote {os.path.join(outdir, 'point_check_scatter.png')}")


# Main
def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.test):
        print(f"ERROR: file not found: {args.test}", file=sys.stderr)
        return 2
    os.makedirs(args.outdir, exist_ok=True)
    return point_check_against_source(args)


if __name__ == "__main__":
    raise SystemExit(main())
