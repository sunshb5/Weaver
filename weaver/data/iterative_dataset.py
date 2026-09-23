"""Iterative ERA5 datasets for randomized training and fixed lead-time tests.

Files use ``{year}_{idx:04d}.h5`` and are spaced six hours apart by default.
"""

import os
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from glob import glob


def get_data_given_path(path, variables):
    """Read selected fields and return an array shaped ``(V, H, W)``."""
    with h5py.File(path, 'r') as f:
        # Read only requested fields and the timestamp.
        data = {
            main_key: {
                sub_key: np.array(value)
                for sub_key, value in group.items()
                if sub_key in variables + ['time']
            }
            for main_key, group in f.items()
            if main_key in ['input']
        }

    # Preserve the configured variable order.
    x = [data['input'][v] for v in variables]
    return np.stack(x, axis=0)


def get_out_path(root_dir, year, inp_file_idx, steps):
    """Return a future file path, handling year boundaries."""
    out_file_idx = inp_file_idx + steps
    out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.h5')

    if not os.path.exists(out_path):
        # The target crosses a year boundary; find the last current-year file.
        for i in range(steps):
            out_file_idx = inp_file_idx + i
            out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.h5')
            if os.path.exists(out_path):
                max_step_forward = i

        # Resolve the remaining steps in the next year.
        remaining_steps = steps - max_step_forward
        next_year = year + 1
        out_path = os.path.join(root_dir, f'{next_year}_{remaining_steps - 1:04}.h5')

    return out_path


# ============================================================================
# Training set: one input and randomized multi-step targets.
# ============================================================================
class ERA5MultiStepRandomizedDataset(Dataset):
    """Randomize the interval used to build iterative training targets."""

    def __init__(
        self,
        root_dir,
        variables,
        inp_transform,
        out_transform_dict,
        steps,
        list_intervals=[6, 12, 24],
        data_freq=6,
    ):
        super().__init__()

        # Every interval must align with the file frequency.
        for l in list_intervals:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform_dict = out_transform_dict
        self.steps = steps
        self.list_intervals = list_intervals
        self.data_freq = data_freq

        # Collect and sort all HDF5 files.
        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)

        # Drop files without a complete future target.
        max_files_needed = steps * max(list_intervals) // data_freq
        self.inp_file_paths = file_paths[:-max_files_needed]
        self.file_paths = file_paths

    def __len__(self):
        return len(self.inp_file_paths)

    def __getitem__(self, index):
        # 1. Read the input field.
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)

        # 2. Sample an interval for augmentation.
        chosen_interval = np.random.choice(self.list_intervals)

        # Parse year and index from names such as ``2020_0015.h5``.
        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)

        outs = []    # Absolute targets used to form the next difference.
        diffs = []   # Normalized target differences.
        last_out = inp_data

        # 3. Build iterative targets.
        for step in range(1, self.steps + 1):
            # Convert the interval to a file offset.
            file_steps = (step * chosen_interval) // self.data_freq
            out_path = get_out_path(self.root_dir, year, inp_file_idx, steps=file_steps)
            out = get_data_given_path(out_path, self.variables)

            # Predict changes rather than absolute fields.
            diff = out - last_out
            diff = torch.from_numpy(diff)
            # Normalize with the interval-specific transform.
            diffs.append(self.out_transform_dict[chosen_interval](diff))

            outs.append(out)
            last_out = out

        # 4. Assemble the sample.
        inp_data = torch.from_numpy(inp_data)
        diffs = torch.stack(diffs, dim=0)  # (T, V, H, W)

        # Return transform statistics for inverse scaling during inference.
        out_transform_mean = torch.from_numpy(self.out_transform_dict[chosen_interval].mean)
        out_transform_std = torch.from_numpy(self.out_transform_dict[chosen_interval].std)

        # Scale the interval feature by ten.
        list_intervals = np.array([chosen_interval] * self.steps)
        list_intervals = torch.from_numpy(list_intervals).to(dtype=inp_data.dtype) / 10.0

        return (
            self.inp_transform(inp_data),  # (V, H, W)
            diffs,                          # (T, V, H, W)
            out_transform_mean,             # (V,)
            out_transform_std,              # (V,)
            list_intervals,                 # Scaled intervals, (T,)
            self.variables,
        )


# ============================================================================
# Validation/test set: one input and fixed lead-time targets.
# ============================================================================
class ERA5MultiLeadtimeDataset(Dataset):
    """Build fixed-lead-time targets for validation and testing."""

    def __init__(
        self,
        root_dir,
        variables,
        transform,
        list_lead_times,
        data_freq=6,
    ):
        super().__init__()

        for l in list_lead_times:
            assert l % data_freq == 0

        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.list_lead_times = list_lead_times
        self.data_freq = data_freq

        file_paths = glob(os.path.join(root_dir, '*.h5'))
        file_paths = sorted(file_paths)

        # Find the largest requested lead time in file steps.
        max_lead_time = max(*list_lead_times) if len(list_lead_times) > 1 else list_lead_times[0]
        max_steps = max_lead_time // data_freq

        # Drop files without a target at the maximum lead time.
        self.inp_file_paths = file_paths[:-max_steps]
        self.file_paths = file_paths

    def __len__(self):
        return len(self.inp_file_paths)

    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)

        year, inp_file_idx = os.path.basename(inp_path).split('.')[0].split('_')
        year, inp_file_idx = int(year), int(inp_file_idx)

        dict_out = {}

        # Read each requested future field.
        for lead_time in self.list_lead_times:
            file_steps = lead_time // self.data_freq
            out_path = get_out_path(self.root_dir, year, inp_file_idx, steps=file_steps)
            dict_out[lead_time] = get_data_given_path(out_path, self.variables)

        inp_data = torch.from_numpy(inp_data)
        dict_out = {lead_time: torch.from_numpy(out) for lead_time, out in dict_out.items()}

        return (
            self.transform(inp_data),  # (V, H, W)
            {lead_time: self.transform(out) for lead_time, out in dict_out.items()},
            # Targets: {lead_time: (V, H, W), ...}
            self.variables,
        )
