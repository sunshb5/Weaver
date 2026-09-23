"""Download WeatherBench2 ERA5 data from Google Cloud Storage."""

import argparse
import xarray as xr
import os
from tqdm import tqdm
from weaver.data_preprocessing.variables import (
    SOURCE_VARIABLES,
    find_data_variable,
)

def main():
    """Parse arguments and download the selected dataset."""
    # Build the argument parser.
    parser = argparse.ArgumentParser()

    # Dataset URI or a WeatherBench2 dataset name.
    parser.add_argument("--file", type=str, required=True,
                        help="Zarr URI, or a name below gs://weatherbench2/datasets/era5/")
    # Output directory.
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2018)

    # Parse arguments.
    args = parser.parse_args()

    # Read parameters.
    file = args.file
    save_dir = args.save_dir

    # Create the output directory.
    os.makedirs(save_dir, exist_ok=True)
    # Open the Zarr dataset from GCS.
    uri = file if "://" in file else 'gs://weatherbench2/datasets/era5/' + file
    ds = xr.open_zarr(uri)

    # Select the requested year range.
    years = list(range(args.start_year, args.end_year + 1))
    variables = SOURCE_VARIABLES
    source_names = {}
    source_to_canonical = {}
    for variable in variables:
        source_name = find_data_variable(ds, variable).name
        previous = source_to_canonical.get(source_name)
        if previous is not None and previous != variable:
            raise ValueError(
                f"source variable {source_name!r} resolves to both "
                f"{previous!r} and {variable!r}; provide canonical WeatherBench2 names"
            )
        source_names[variable] = source_name
        source_to_canonical[source_name] = variable

    # Download each variable.
    for var in tqdm(variables, desc="variables", position=0):
        # Normalize common short ERA5 names (t2m/u10/z/...) to the
        # canonical names used by the model and all downstream files.
        source_name = source_names[var]
        ds_var = ds[[source_name]].rename({source_name: var})
        if len(ds_var.dims) < 3: # constant variables
            # Save constant fields as one file.
            ds_var.to_netcdf(os.path.join(save_dir, f'{var}.nc'))
        else:
            # Save time-varying fields by year.
            save_dir_var = os.path.join(save_dir, var)
            os.makedirs(save_dir_var, exist_ok=True)
            # Export each year.
            for year in tqdm(years, desc="years", position=1, leave=False):
                # Select the current year.
                ds_var_year = ds_var.sel(time=str(year))
                # Write NetCDF.
                ds_var_year.to_netcdf(os.path.join(save_dir_var, f'{year}.nc'))


if __name__ == "__main__":
    main()
