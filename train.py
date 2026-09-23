"""Train WEAVER with PyTorch Lightning and FSDP.

Usage: ``python train.py --config configs/weaver.yaml``
"""
import os
import torch

# Enable Tensor Core matmul kernels where available.
torch.set_float32_matmul_precision('medium')

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from weaver.data.multi_step_datamodule import MultiStepDataRandomizedModule
from weaver.models.forecast_module import WeatherForecastModule
from weaver.models.backbone import (
    Block,
    RegionalPriorMoEBlock,
    CrossCoupledAttentionBlock,
    CrossCoupledAttentionGatherBlock,
    DualStreamPatchEmbedding,
)


class CustomCLI(LightningCLI):
    """LightningCLI with project-specific callback and FSDP handling."""

    def _instantiate_trainer(self, config, callbacks):
        """Normalize callbacks and configure the optional FSDP strategy."""
        key = "callbacks"
        if key in config:
            # Normalize callbacks to a list.
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            # Add user callbacks.
            config[key].extend(callbacks)
            # Add trainer defaults.
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            # Add the config callback unless running a fast debug job.
            if self.save_config_callback and not config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                config[key].append(config_callback)
        else:
            # Warn if the trainer does not expose callbacks.
            rank_zero_warn(
                f"The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will"
                " not be included."
            )

        # Configure FSDP. Activation checkpointing is controlled by
        # WEAVER_FSDP_CKPT; freeze-backbone runs use a compatible wrapping set.
        if config['strategy'] == 'fsdp':
            fsdp_activation_checkpointing = os.environ.get("WEAVER_FSDP_CKPT", "true").lower() in ("1", "true", "yes")
            # Frozen CCA blocks are kept in the parent FSDP unit to avoid the
            # unallocated-storage path in torch 2.1.2.
            fsdp_freeze_backbone = os.environ.get("WEAVER_FREEZE_BACKBONE", "false").lower() in ("1", "true", "yes")
            auto_wrap = {Block, RegionalPriorMoEBlock, DualStreamPatchEmbedding}
            if not fsdp_freeze_backbone:
                auto_wrap.update({CrossCoupledAttentionBlock, CrossCoupledAttentionGatherBlock})
            fsdp_kwargs = dict(
                sharding_strategy="FULL_SHARD",
                auto_wrap_policy=auto_wrap,
                # Required when frozen and trainable parameters are mixed.
                use_orig_params=True,
            )
            if fsdp_activation_checkpointing:
                fsdp_kwargs['activation_checkpointing_policy'] = {RegionalPriorMoEBlock, Block}
            fsdp_strategy = FSDPStrategy(**fsdp_kwargs)
            config['strategy'] = fsdp_strategy

        # Instantiate the trainer from the normalized configuration.
        return self.trainer_class(**config)


class ClearCacheCallback(Callback):
    """Clear the CUDA cache before fitting and validation."""
    def on_fit_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
        if trainer.is_global_zero:
            print(f"[ClearCache] CUDA cache emptied at fit start")

    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()
        if trainer.is_global_zero:
            print(f"[ClearCache] CUDA cache emptied before validation")


def main():
    """Parse the configuration, build the components, and start training."""
    cli = CustomCLI(
        model_class=WeatherForecastModule,
        datamodule_class=MultiStepDataRandomizedModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    # Create the output directory.
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # Set the geographic grid used by positional features.
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    # Attach data transforms.
    cli.model.set_transforms(*cli.datamodule.get_transforms())

    # Configure train intervals and validation lead times.
    cli.model.set_base_intervals_and_lead_times(
        cli.datamodule.hparams.list_train_intervals,
        cli.datamodule.hparams.val_lead_times,
    )

    # Clear CUDA state after checkpoint restoration and before validation.
    cli.trainer.callbacks.append(ClearCacheCallback())

    # Train from scratch by default; resuming must be explicit:
    #   WEAVER_RESUME=auto  -> <default_root_dir>/checkpoints/last.ckpt
    #   WEAVER_RESUME=/path/to/checkpoint.ckpt -> explicit checkpoint
    resume = os.environ.get("WEAVER_RESUME", "").strip()
    if not resume or resume.lower() in {"0", "false", "none", "no"}:
        ckpt_resume_path = None
    elif resume.lower() == "auto":
        ckpt_resume_path = os.path.join(
            cli.trainer.default_root_dir, "checkpoints", "last.ckpt"
        )
        if not os.path.isfile(ckpt_resume_path):
            raise FileNotFoundError(
                f"WEAVER_RESUME=auto, but checkpoint does not exist: {ckpt_resume_path}"
            )
    else:
        ckpt_resume_path = os.path.abspath(os.path.expanduser(resume))
        if not os.path.isfile(ckpt_resume_path):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {ckpt_resume_path}")

    if cli.trainer.is_global_zero:
        mode = f"resume from {ckpt_resume_path}" if ckpt_resume_path else "train from scratch"
        print(f"[WEAVER] {mode}")

    # Start training; a non-None path resumes optimizer state.
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)


if __name__ == "__main__":
    main()
