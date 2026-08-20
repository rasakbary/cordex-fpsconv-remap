"""
fpsremap - shared code for remapping the CORDEX-FPS model
output, both the convection-permitting models (CPMs) and the regional
climate models (RCMs).

This is a library. The scripts you run are in the folder above:

    inspect_CORDEX-FPS_models.py   list the files and show their grids
    remap.py                       merge, convert units, remap, compress
    compare_remapped.py            compare a result against a reference

Modules:

    config          read the YAML settings and fill in the paths
    nc_metadata     read grid, coordinate and time information from a file
    grids           rotated-pole and Lambert maths, and the grid
                    description files that CDO needs
    cdo             run CDO commands and copy attributes onto the output
    units           check and convert units
    file_filters    optional filters for choosing which files to use
    fix_attributes  repair attributes that have the wrong type
"""

__version__ = "1.0.0"
