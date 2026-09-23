"""Convert yearly NetCDF fields into one-step HDF5 samples."""

import os
import argparse
import numpy as np
import xarray as xr
import h5py
import tempfile
from tqdm import tqdm
from weaver.data_preprocessing.variables import (
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SINGLE_LEVEL_VARIABLES,
    SOURCE_VARIABLES,
    STATIC_VARIABLES,
)

# Fields to export; keep this aligned with the release schema.
VARS = SOURCE_VARIABLES


def atomic_save_npy(path, array):
    """Publish a NumPy file atomically when split jobs run concurrently."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as handle:
        temporary = handle.name
        np.save(handle, array)
    os.replace(temporary, path)


def create_one_step_dataset(root_dir, save_dir, split, years, list_vars, chunk_size=None):
    """Create one HDF5 file per timestamp for the selected split."""
    # Create the split directory.
    save_dir_split = os.path.join(save_dir, split)
    os.makedirs(save_dir_split, exist_ok=True)

    # Classify variables by storage layout.
    list_constant_vars = [v for v in list_vars if v in STATIC_VARIABLES]
    list_single_vars = [v for v in list_vars if v in SINGLE_LEVEL_VARIABLES]
    list_pressure_vars = [v for v in list_vars if v in PRESSURE_VARIABLES]
    expected_fields = set(list_constant_vars + list_single_vars)
    expected_fields.update(
        f'{name}_{level}'
        for name in list_pressure_vars
        for level in PRESSURE_LEVELS
    )

    # Read coordinates and fields in the same sorted order.  Sorting only the
    # coordinate arrays would silently mis-register a field when an upstream
    # NetCDF file stores latitude from north to south.
    with xr.open_dataset(os.path.join(root_dir, f'{list_constant_vars[0]}.nc')) as ds_constant:
        ds_constant = ds_constant.sortby('latitude').sortby('longitude')
        lat = ds_constant.latitude.to_numpy()
        lon = ds_constant.longitude.to_numpy()
    atomic_save_npy(os.path.join(save_dir, 'lat.npy'), lat)
    atomic_save_npy(os.path.join(save_dir, 'lon.npy'), lon)

    constant_fields = {}
    for var in list_constant_vars:
        constant_path = os.path.join(root_dir, f'{var}.nc')
        with xr.open_dataset(constant_path) as constant_ds:
            constant_ds = constant_ds.sortby('latitude').sortby('longitude')
            field = constant_ds[var].to_numpy()
        constant_fields[var] = field.reshape(field.shape[-2:]).astype(np.float32)

    # Process one year at a time.
    for year in tqdm(years, desc='years', position=0):
        # Use one field to determine the number of timestamps.
        ds_sample = xr.open_dataset(os.path.join(root_dir, list_single_vars[0], f'{year}.nc'))
        # Compute the number of chunks.
        if chunk_size is not None:
            n_chunks = (len(ds_sample.time) + chunk_size - 1) // chunk_size
        else:
            n_chunks = 1
            chunk_size = len(ds_sample.time)
        ds_sample.close()

        # File index within the year.
        idx_in_year = 0

        # Open all fields once to avoid repeated I/O.
        ds_dict = {}
        for var in (list_single_vars + list_pressure_vars):
            dataset = xr.open_dataset(os.path.join(root_dir, var, f'{year}.nc'))
            dataset = dataset.sortby('latitude').sortby('longitude')
            if var in list_pressure_vars:
                available_levels = tuple(int(level) for level in dataset.level.values)
                if available_levels != tuple(PRESSURE_LEVELS):
                    raise ValueError(
                        f'{var}/{year}.nc has pressure levels {available_levels}; '
                        f'expected {tuple(PRESSURE_LEVELS)}'
                    )
            ds_dict[var] = dataset

        # Process one chunk at a time.
        for chunk_id in tqdm(range(n_chunks), desc='chunks', position=1, leave=False):
            # Store the current chunk as NumPy arrays.
            dict_np = {}
            list_time_stamps = None
            # Convert xarray fields to NumPy arrays.
            for var in (list_single_vars + list_pressure_vars):
                # Select the chunk time slice.
                ds = ds_dict[var].isel(time=slice(chunk_id*chunk_size, (chunk_id+1)*chunk_size))
                # Record timestamps once per chunk.
                if list_time_stamps is None:
                    list_time_stamps = ds.time.values
                if var in list_single_vars:
                    # Single-level field: (N, H, W).
                    dict_np[var] = ds[var].values
                else:
                    # Split pressure-level fields by level.
                    available_levels = ds.level.values
                    ds_np = ds[var].values
                    for i, level in enumerate(available_levels):
                        dict_np[f'{var}_{int(level)}'] = ds_np[:, i]

            # Write one HDF5 file per timestamp.
            for i in tqdm(range(len(list_time_stamps)), desc='time stamps', position=2, leave=False):
                # Build the HDF5 groups.
                data_dict = {
                    'input': {'time': str(list_time_stamps[i])}
                }
                # Add dynamic fields.
                for var in dict_np.keys():
                    data_dict['input'][var] = dict_np[var][i]
                # Add constant fields.
                for var in list_constant_vars:
                    data_dict['input'][var] = constant_fields[var]

                if set(data_dict['input']) - {'time'} != expected_fields:
                    raise RuntimeError('generated HDF5 sample does not contain the canonical fields')

                # Publish atomically so interrupted writes are not mistaken for
                # complete samples.
                output_path = os.path.join(save_dir_split, f'{year}_{idx_in_year:04}.h5')
                temporary_path = output_path + f'.tmp.{os.getpid()}'
                with h5py.File(temporary_path, 'w', libver='latest') as f:
                    for main_key, sub_dict in data_dict.items():
                        # Create the top-level group.
                        group = f.create_group(main_key)

                        # Write each field.
                        for sub_key, array in sub_dict.items():
                            if sub_key != 'time':
                                # Store numeric fields as uncompressed float32.
                                group.create_dataset(sub_key, data=array, compression=None, dtype=np.float32)
                            else:
                                # Store timestamps as uncompressed strings.
                                group.create_dataset(sub_key, data=array, compression=None)

                os.replace(temporary_path, output_path)

                # Advance the yearly index.
                idx_in_year += 1

        for dataset in ds_dict.values():
            dataset.close()


def parse_args():
    """Parse the NetCDF-to-HDF5 conversion arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=2008, help='Start year (inclusive).')
    parser.add_argument('--end_year', type=int, default=2016, help='End year (inclusive).')
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset (train, val, test).")
    parser.add_argument("--chunk_size", type=int, default=10, help="Chunk size for reading datasets (default=10).")

    return parser.parse_args()


def main():
    """Parse arguments and create the selected split."""
    # Parse arguments.
    args = parse_args()

    # Build the dataset.
    create_one_step_dataset(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        split=args.split,
        years=list(range(args.start_year, args.end_year + 1)),
        list_vars=VARS,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
