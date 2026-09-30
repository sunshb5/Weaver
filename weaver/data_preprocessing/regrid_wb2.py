"""Regrid WeatherBench2 fields onto the release grid."""

import os
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm
from weaver.data_preprocessing import regridding
from weaver.data_preprocessing.variables import (
    SOURCE_VARIABLES,
    STATIC_VARIABLES,
)

# Fields to regrid; constants are stored as ``*.nc`` files.
VARS = [f"{name}.nc" if name in STATIC_VARIABLES else name
        for name in SOURCE_VARIABLES]

def parse_args():
    """Parse regridding arguments."""
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--ddeg_out', type=float, default=1.40625, help='Output grid spacing in degrees.')
    parser.add_argument('--start_year', type=int, default=2008, help='Start year (inclusive).')
    parser.add_argument('--end_year', type=int, default=2018, help='End year (inclusive).')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for reading datasets (default=100).')

    return parser.parse_args()

def main():
    """Regrid constants and yearly time-varying fields."""
    # Parse arguments.
    args = parse_args()

    # Read parameters.
    root_dir = args.root_dir
    save_dir = args.save_dir
    ddeg_out = args.ddeg_out
    start_year = args.start_year
    end_year = args.end_year
    chunk_size = args.chunk_size

    # Build the year range.
    years = list(range(start_year, end_year + 1))
    # Create the output directory.
    os.makedirs(save_dir, exist_ok=True)

    # Build target cell-center coordinates.
    lat_start = -90 + ddeg_out / 2
    lat_stop = 90 - ddeg_out / 2
    new_lat = np.linspace(lat_start, lat_stop, num=int(180/ddeg_out), endpoint=True)
    new_lon = np.linspace(0, 360, num=int(360//ddeg_out), endpoint=False)

    # Initialize lazily from the first input grid.
    regridder = None
    # Build input paths.
    var_dirs = [os.path.join(root_dir, v) for v in VARS]

    # Process each field.
    for dir in tqdm(var_dirs, desc='vars', position=0):
        # Extract the field name.
        var_name = os.path.basename(dir)

        # Constants are stored as ``*.nc`` files.
        if var_name.endswith('.nc'):
            # Open constants with a consistent dimension order.
            ds_in = xr.open_dataset(dir).transpose(..., 'latitude', 'longitude')

            # Construct the regridder from the first grid.
            if regridder is None:
                # Read source coordinates.
                old_lon = ds_in.coords['longitude'].data
                old_lat = ds_in.coords['latitude'].data
                # Build source and target grids.
                source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
                target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
                # Create the conservative regridder.
                regridder = regridding.ConservativeRegridder(source_grid, target_grid)

            # Regrid and save the result.
            ds_out = regridder.regrid_dataset(ds_in)
            ds_out.to_netcdf(os.path.join(save_dir, var_name))
            ds_in.close()

        else:
            # Time-varying fields use one subdirectory.
            os.makedirs(os.path.join(save_dir, var_name), exist_ok=True)
            # Process each year.
            for year in tqdm(years, desc='years', position=1, leave=False):
                # Open the year with chunked reads.
                ds_in = xr.open_dataset(os.path.join(dir, f'{year}.nc'), chunks={'time': chunk_size})

                # Construct the regridder from the first grid.
                if regridder is None:
                    # Read source coordinates.
                    old_lon = ds_in.coords['longitude'].data
                    old_lat = ds_in.coords['latitude'].data
                    # Build source and target grids.
                    source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
                    target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
                    # Create the conservative regridder.
                    regridder = regridding.ConservativeRegridder(source_grid, target_grid)

                # Regrid, normalize dimension order, and save.
                ds_out = regridder.regrid_dataset(ds_in).transpose(..., 'latitude', 'longitude')
                ds_out.to_netcdf(os.path.join(save_dir, var_name, f'{year}.nc'))
                ds_in.close()

if __name__ == "__main__":
    main()
