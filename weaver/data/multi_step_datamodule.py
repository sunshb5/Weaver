"""Lightning data module for iterative WEAVER forecasts."""

import os
from typing import Optional, Sequence, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from lightning import LightningDataModule

from weaver.data.iterative_dataset import ERA5MultiStepRandomizedDataset, ERA5MultiLeadtimeDataset


def collate_fn_train(
    batch,
) -> Tuple[torch.tensor, torch.tensor, Sequence[str], Sequence[str]]:
    """Stack randomized multi-step samples into a training batch."""
    # Inputs: (B, V, H, W).
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W
    # Targets: (B, T, V, H, W).
    out = torch.stack([batch[i][1] for i in range(len(batch))]) # B, T, V, H, W
    # Inverse-transform means: (B, V).
    out_transform_mean = torch.stack([batch[i][2] for i in range(len(batch))]) # B, V
    # Inverse-transform standard deviations: (B, V).
    out_transform_std = torch.stack([batch[i][3] for i in range(len(batch))]) # B, V
    # Scaled intervals: (B, T).
    interval = torch.stack([batch[i][4] for i in range(len(batch))]) # B, T
    # Variable names are shared across the batch.
    variables = batch[0][5]
    return inp, out, out_transform_mean, out_transform_std, interval, variables


def collate_fn_val(batch):
    """Collate fixed-lead-time validation/test samples."""
    # Inputs: (B, V, H, W).
    inp = torch.stack([batch[i][0] for i in range(len(batch))]) # B, V, H, W

    # Each sample stores targets keyed by lead time.
    out_dicts = [batch[i][1] for i in range(len(batch))]
    # Use the lead times from the first sample.
    list_lead_times = out_dicts[0].keys()
    # Stack targets independently for each lead time.
    out = {}
    for lead_time in list_lead_times:
        out[lead_time] = torch.stack([out_dicts[i][lead_time] for i in range(len(batch))])

    # Variable names are shared across the batch.
    variables = batch[0][2]

    return inp, out, variables


class MultiStepDataRandomizedModule(LightningDataModule):
    """Data module for randomized training and fixed-lead-time evaluation."""

    def __init__(
        self,
        root_dir,
        variables,
        list_train_intervals,
        steps,
        val_lead_times,
        data_freq=6,
        batch_size=64,
        val_batch_size=64,
        num_workers=0,
        pin_memory=False,
        statistics_dir=None,
    ):
        """Initialize datasets, transforms, and loader settings."""
        super().__init__()

        # Keep configuration in Lightning's hyperparameter namespace.
        self.save_hyperparameters(logger=False)

        # HDF5 may live on a large external filesystem while the small,
        # training-period-only statistics ship with the code release.
        self.statistics_dir = statistics_dir or root_dir

        # Load global input normalization statistics.
        normalize_mean = dict(np.load(os.path.join(self.statistics_dir, "normalize_mean.npz")))
        # Preserve the configured variable order.
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(self.statistics_dir, "normalize_std.npz")))
        # Preserve the configured variable order.
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        # Input transform: (x - mean) / std.
        self.transforms = transforms.Normalize(normalize_mean, normalize_std)

        # Load interval-specific difference statistics.
        out_transforms = {}
        for l in list_train_intervals:
            # Load the standard deviation for this interval.
            normalize_diff_std = dict(np.load(os.path.join(
                self.statistics_dir, f"normalize_diff_std_{l}.npz"
            )))
            # Preserve the configured variable order.
            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
            # Difference transform: x / std.
            out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
        self.out_transforms = out_transforms

        # Datasets are created lazily in setup().
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_lat_lon(self):
        """Return latitude and longitude arrays."""
        lat = np.load(os.path.join(self.statistics_dir, "lat.npy"))
        lon = np.load(os.path.join(self.statistics_dir, "lon.npy"))
        return lat, lon
    
    def get_transforms(self):
        """Return input and interval-specific output transforms."""
        return self.transforms, self.out_transforms

    def setup(self, stage: Optional[str] = None):
        """Create train, validation, and test datasets on first use."""
        # Lazily initialize datasets once.
        if not self.data_train and not self.data_val and not self.data_test:
            # Training uses randomized intervals.
            self.data_train = ERA5MultiStepRandomizedDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'train'),
                variables=self.hparams.variables,
                inp_transform=self.transforms,
                out_transform_dict=self.out_transforms,
                steps=self.hparams.steps,
                list_intervals=self.hparams.list_train_intervals,
                data_freq=self.hparams.data_freq,
            )

            # Validation uses fixed lead times.
            self.data_val = ERA5MultiLeadtimeDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'val'),
                variables=self.hparams.variables,
                transform=self.transforms,
                list_lead_times=self.hparams.val_lead_times,
                data_freq=self.hparams.data_freq
            )

            # Test has the same structure and lead times.
            self.data_test = ERA5MultiLeadtimeDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'test'),
                variables=self.hparams.variables,
                transform=self.transforms,
                list_lead_times=self.hparams.val_lead_times,
                data_freq=self.hparams.data_freq
            )

    def train_dataloader(self):
        """Build the training data loader."""
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_train,
        )

    def val_dataloader(self):
        """Build the validation data loader."""
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_val,
        )

    def test_dataloader(self):
        """Build the test data loader."""
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_fn_val,
        )
