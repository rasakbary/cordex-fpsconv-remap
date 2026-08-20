# Remapping the CORDEX-FPSCONV model output

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22043353.svg)](https://doi.org/10.5281/zenodo.22043353)

Scripts for remapping CORDEX-FPSCONV model output onto a common regular latitude/longitude grid. The scripts within 
this repository were developed on convection-permitting models (CPMs) and their regional climate models (RCMs).

The remapping itself is done by [CDO](https://code.mpimet.mpg.de/projects/cdo).
These scripts handle the parts around it: finding each model's files, working
out how to describe its native grid to CDO, harmonising units, and merging the
result into one file per model.

## What's here

| Script | Purpose |
|---|---|
| `inspect_CORDEX-FPS_models.py` | List the available files and show the grid and coordinate metadata of each model |
| `remap.py` | Merge a model's files, convert units, remap to the target grid, compress |
| `compare_remapped.py` | Check a remapped file against its source files |

`fpsremap/` holds the shared code used by all three scripts. Adjust to your archieve by editing the `config/*.yaml` which holds the paths, model names and grid definitions.

## Why remapping

Each institute ran its own model on its own grid, so the files cannot be
compared cell by cell until they are on a common grid. Three things to know about the
CORDEX-FPSCONV models:

- **Grids vary.** Files use rotated-pole coordinates, Lambert conformal
  projections, or plain 2D latitude/longitude.
- **Cell corners are usually missing.** Conservative remapping needs the four
  corners of each source cell, and most files store only cell centres. For
  rotated and projected grids CDO cannot derive them, so they have to be
  reconstructed.
- **Conventions are inconsistent.** Some files store the rotated pole 180° from
  what the standard formula expects, or reverse the sign of `rlon`. Units for
  precipitation appear as `kg m-2 s-1`, `kg m-2` and `mm/h`, and sometimes varying
  between files of the same model. A few files store `_FillValue` as text,
  which makes CDO fail when writing output.


## Which remapping method is used

CDO offers several interpolation methods — `remapcon` (first-order conservative), 
`remapbil` (bilinear), `remapnn` (nearest neighbour), and some more. You can find the 
full desciption of each methods in the [CDO User Guide](https://zenodo.org/records/7112925).

These scripts use `remapcon`, first-order conservative remapping (Jones, 1999),
for all variables and both ensembles. This method preserves the areal mean precipitation 
and is therefore appropriate for our analysis (e.g., areal mean precipitation). In conservative 
remapping the grid-cell value is a weighted average of the source cells, weighted by how much 
area they share with the target cell. And therefore it preserves the total mass over the domain.


## Requirements

- Python 3.10+
- CDO on your `PATH` (not a Python package)
- `numpy`, `netCDF4`, `PyYAML`; `matplotlib` 

```bash
conda env create -f environment.yml && conda activate fpsremap
```

## Setup

Edit `config/paths.yaml`:

```yaml
roots:
  archive: /path-to-archive-directory/CORDEX-FPSCONV   # input, read only
  output:  /path-to-output-directory/rakbary          # where results go
  scratch: /home/rakbary                               # target-grid description file
```

## Usage

```bash
# To start check the folder what files are there? (filenames only)
python inspect_CORDEX-FPS_models.py --ensemble cpm --variable pr --period historical

# dump nc files and read important information
python inspect_CORDEX-FPS_models.py --ensemble cpm --variable pr --period historical --mode dump

# try one model e.g., BTU
python remap.py --ensemble cpm --variable pr --period historical --model BTU

# Check the result against its source files
python compare_remapped.py --variable pr --outdir ./check_BTU \
    --test <output>/BTU_pr_Historical_merged.nc \
    --source <archive>/ALP-3/1hr/pr/historical/pr_ALP-3_..._CLMcom-BTU-*.nc
```

`--ensemble cpm|rcm` selects the model ensemble and target grid; `--period
historical|rcp85` selects the time slice.

For the full ensemble, run several models at once with GNU parallel.

```bash
mkdir -p logs
parallel -j 2 'ionice -c2 -n7 nice -n10 python -u remap.py \
    --ensemble cpm --variable pr --period historical --model {} \
    > logs/{}.log 2>&1' ::: BTU CNRM CMCC DWD ETH ICTP KIT
```

### What `remap.py` does to each model

Find and read the files → report anything unusual about them → repair malformed
attributes if present → convert units per file → merge → describe the source
grid to CDO → remap → compress → restore the original CORDEX attributes.

Units are converted **before** merging, using each file's own `units`
attribute, because some models label the unit in their files inconsistently.

### A note on incomplete time slices

**Some models here ship a stray month or two at the start or end of
their time slice**, alongside the complete yearly or monthly files.

By default `remap.py` merges **every files it finds** and changes nothing. You can control this
by providing it the file filtering argument. 

### Describing the source grid to CDO

| | Situation | What happens |
|---|---|---|
| A | File states its cell corners (2D lon/lat with `bounds`) | Remap directly; nothing is reconstructed |
| B | Rotated pole, or `rlon`/`rlat` with no grid mapping | Write a curvilinear grid description, with the pole convention checked against the file's own lon/lat |
| C | Lambert conformal conic | Write a projection grid description and let PROJ derive the corners |

In case B the corners come from the file's own `bounds` if present, then
`rlon_bnds`/`rlat_bnds`, then midpoints between cell centres. The log records
which was used.

Every model in this archive falls into one of the three. A file matching none
of them is skipped with an error naming what was missing.

`inspect_CORDEX-FPS_models.py --mode dump` reports which case each model falls into, which needs to be 
checked before remapping.

## Target grids

Defined in `config/grids.yaml`.

| Name | Resolution | Size | For |
|---|---|---|---|
| `gar_cpm_0275` | 0.0275° (~3 km) | 437 × 183 | ALP-3 CPM ensemble |
| `gar_rcm_011` | 0.11° (~12 km) | 136 × 92 | Driving RCM ensemble |

Each is close to the native resolution of the models that use it.

## Configuration

`config/models_cpm.yaml` and `config/models_rcm.yaml` list the models. Set
`enabled: false` to skip any model; `--model ABBR` overrides that for a single run.

- `units_override` forces the units for a given model, variable and period —
  only use it when you have actually checked the data and confirmed the label is wrong.
- `unavailable` marks models that do not exist in the archive.


## Output

One file per model: `{MODEL}_{variable}_{Period}_merged.nc`, on the target grid,
with `pr` in mm/h and `tas` in its native units. Original CORDEX global
attributes are carried over, plus `fpsremap_*` attributes recording the target
grid, method and version.

Files are written as NetCDF-4 with deflate level 5. Hourly precipitation
compresses roughly 5×, which makes a huge difference to storage and to transfer over a 
network filesystem; the cost when reading is negligible.

Chunking is set as one complete map per timestep, which is suitable when reading the whole map. 


## References

- Jones, P. W. (1999): *First- and second-order conservative remapping schemes
  for grids in spherical coordinates.* Monthly Weather Review, 127, 2204–2210.

## Citation

If this code contributes to a publication, please cite the repository.

> Akbary, R. (2026). *cordex-fpsconv-remap: remapping CORDEX-FPSCONV model output onto a common grid* (Version v1.0.2). Zenodo.
> https://doi.org/10.5281/zenodo.22043353

## Licence

MIT — see [LICENSE](LICENSE).
