"""Compute normalization statistics for raw fields or field differences."""

import os
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from weaver.data_preprocessing.variables import (
    PRESSURE_LEVELS,
    PRESSURE_VARIABLES,
    SINGLE_LEVEL_VARIABLES,
    SOURCE_VARIABLES,
    STATIC_VARIABLES,
)

# Match the canonical field schema used by HDF5 export.
VARS = SOURCE_VARIABLES

def parse_args():
    """Parse normalization-statistics arguments."""
    parser = argparse.ArgumentParser(description='Regridding NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save regridded files.')
    parser.add_argument('--start_year', type=int, default=2008, help='Start year (inclusive).')
    parser.add_argument('--end_year', type=int, default=2016, help='End year (inclusive).')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size for reading datasets (default=100).')
    parser.add_argument('--lead_time', type=int, choices=(-1, 6, 12, 24), default=-1,
                        help='-1 for state statistics; otherwise increment lead time in hours.')
    parser.add_argument('--data_frequency', type=int, default=6, help='Data frequency in hours (default=6).')  # this depends on the dataset

    return parser.parse_args()

def main():
    """Compute and save per-variable means and standard deviations."""
    # Parse arguments.
    args = parse_args()

    # Read parameters.
    root_dir = args.root_dir
    save_dir = args.save_dir
    start_year = args.start_year
    end_year = args.end_year
    chunk_size = args.chunk_size
    lead_time = args.lead_time
    data_freq = args.data_frequency

    # Build the year range.
    years = list(range(start_year, end_year + 1))
    # Create the output directory.
    os.makedirs(save_dir, exist_ok=True)

    # Static ERA5 files live at the data root; classify field types.
    list_constant_vars = [v for v in VARS if v in STATIC_VARIABLES]
    list_single_vars = [v for v in VARS if v in SINGLE_LEVEL_VARIABLES]
    list_pressure_vars = [v for v in VARS if v in PRESSURE_VARIABLES]

    required_paths = [os.path.join(root_dir, f'{var}.nc') for var in list_constant_vars]
    required_paths += [
        os.path.join(root_dir, var, f'{year}.nc')
        for var in list_single_vars + list_pressure_vars
        for year in years
    ]
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            f"missing {len(missing_paths)} source files; first entries: {missing_paths[:5]}"
        )

    # Select output names for raw or difference statistics.
    mean_file_name = f"normalize_diff_mean_{lead_time}.npz" if lead_time != -1 else "normalize_mean.npz"
    std_file_name = f"normalize_diff_std_{lead_time}.npz" if lead_time != -1 else "normalize_std.npz"

    # Always rebuild from the requested year range.  Loading a finalized
    # scalar archive and appending new chunk statistics would silently corrupt
    # a repeated run (the default release directory already contains archives).
    normalize_mean = {}
    normalize_std = {}
    for var in list_single_vars:
        normalize_mean[var] = []
        normalize_std[var] = []
    for var in list_pressure_vars:
        for level in PRESSURE_LEVELS:
            normalize_mean[f'{var}_{level}'] = []
            normalize_std[f'{var}_{level}'] = []

    # A negative lead time means raw-field statistics.
    steps = lead_time // data_freq if lead_time != -1 else None

    # Process single-level and pressure-level fields.
    all_vars = list_single_vars + list_pressure_vars
    print(f"\nSingle-level variables: {list_single_vars}")
    print(f"Pressure-level variables: {list_pressure_vars}")
    print(f"Constant variables: {list_constant_vars}")
    print(f"Total variables: {len(all_vars)}")

    for var in tqdm(all_vars, desc='variables', position=0):
        # Skip missing variable directories.
        var_dir = os.path.join(root_dir, var)
        if not os.path.exists(var_dir):
            print(f"[skip] directory not found: {var_dir}")
            continue
        # Skip variables without the first year.
        sample_path = os.path.join(root_dir, var, f'{years[0]}.nc')
        if not os.path.exists(sample_path):
            print(f"[skip] file not found: {sample_path}")
            continue
        for year in tqdm(years, desc='years', position=1, leave=False):
            # Open the current variable/year.
            path = os.path.join(root_dir, var, f'{year}.nc')
            if not os.path.exists(path):
                print(f"[skip] file not found: {path}")
                continue
            ds = xr.open_dataset(path)

            # Chunk the time dimension to control memory use.
            current_chunk_size = chunk_size if chunk_size is not None else len(ds.time)
            n_chunks = (len(ds.time) + current_chunk_size - 1) // current_chunk_size

            # Materialize each chunk before reducing it.
            for chunk_id in tqdm(range(n_chunks), desc='chunks', position=2, leave=False):
                # Select the current time slice.
                chunk_start = chunk_id * current_chunk_size
                # Include the preceding lead-time samples so differences at a
                # chunk boundary are counted exactly once.  Without this
                # overlap, every chunk boundary silently drops one valid
                # 6/12/24-hour training target.
                read_start = max(0, chunk_start - (steps or 0))
                read_end = min(
                    len(ds.time), (chunk_id + 1) * current_chunk_size
                )
                ds_small = ds.isel(time=slice(read_start, read_end))
                if var in SINGLE_LEVEL_VARIABLES:
                    # Single-level field: (N, H, W).
                    ds_np = ds_small[var].values # N, H, W
                    # Form differences when requested.
                    if steps is not None:
                        ds_np = ds_np[steps:] - ds_np[:-steps]
                    # Store chunk statistics.
                    normalize_mean[var].append(np.nanmean(ds_np))
                    normalize_std[var].append(np.nanstd(ds_np))
                else:
                    # Pressure-level field: (N, levels, H, W).
                    ds_np = ds_small[var].values # N, Levels, H, W
                    levels_in_ds = tuple(int(level) for level in ds.level.values)
                    if levels_in_ds != tuple(PRESSURE_LEVELS):
                        raise ValueError(
                            f'{path} has pressure levels {levels_in_ds}; '
                            f'expected {tuple(PRESSURE_LEVELS)}'
                        )
                    # Reduce each pressure level independently.
                    for i, level in enumerate(levels_in_ds):
                        ds_np_lev = ds_np[:, i]
                        # Form differences when requested.
                        if steps is not None:
                            ds_np_lev = ds_np_lev[steps:] - ds_np_lev[:-steps]
                        # Store level statistics.
                        normalize_mean[f'{var}_{level}'].append(np.nanmean(ds_np_lev))
                        normalize_std[f'{var}_{level}'].append(np.nanstd(ds_np_lev))
            ds.close()

        # Merge chunk statistics with the law of total variance.
        if var in SINGLE_LEVEL_VARIABLES:
            mean_over_files, std_over_files = np.array(normalize_mean[var]), np.array(normalize_std[var])
            # var(X) = E[var(X|Y)] + var(E[X|Y]).
            variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
            std = np.sqrt(variance)
            # The global mean is the mean of chunk means.
            mean = mean_over_files.mean()
            # Store one-element arrays for npz compatibility.
            normalize_mean[var] = mean.reshape([1])
            normalize_std[var] = std.reshape([1])

            # Save incremental progress.
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
        else:
            # Apply the same merge per pressure level.
            for l in PRESSURE_LEVELS:
                var_lev = f'{var}_{l}'
                mean_over_files, std_over_files = np.array(normalize_mean[var_lev]), np.array(normalize_std[var_lev])
                # Law of total variance.
                variance = (std_over_files**2).mean() + (mean_over_files**2).mean() - mean_over_files.mean()**2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean_over_files.mean()
                normalize_mean[var_lev] = mean.reshape([1])
                normalize_std[var_lev] = std.reshape([1])

            # Save incremental progress.
            np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
            np.savez(os.path.join(save_dir, std_file_name), **normalize_std)

    # Process constant fields.
    for var in list_constant_vars:
        path = os.path.join(root_dir, f'{var}.nc')
        if not os.path.exists(path):
            print(f"[skip] constant file not found: {path}")
            continue
        if lead_time != -1:
            # Constant differences are zero.
            normalize_mean[var] = [0.0]
            # A unit scale keeps the identically-zero normalized increment
            # finite; the model explicitly fixes static-variable increments
            # to zero during training and inference.
            normalize_std[var] = [1.0]
        else:
            # Compute raw constant statistics.
            with xr.open_dataset(path) as ds:
                ds_np = ds[var].values
            normalize_mean[var] = ds_np.mean().reshape([1])
            normalize_std[var] = ds_np.std().reshape([1])

    # Always write both output archives.
    if not normalize_mean:
        print("[warning] no variables were processed; writing empty archives")
    np.savez(os.path.join(save_dir, mean_file_name), **normalize_mean)
    np.savez(os.path.join(save_dir, std_file_name), **normalize_std)
    print(f"Saved: {mean_file_name} ({len(normalize_mean)} variables)")
    print(f"Saved: {std_file_name} ({len(normalize_std)} variables)")


if __name__ == "__main__":
    main()
