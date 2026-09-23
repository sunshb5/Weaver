"""Lightning callback that writes dynamic train/validation loss history."""

import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class LossLogger(Callback):
    """Write one readable loss table with dynamic validation lead-time columns."""

    def __init__(self, save_dir: str = "results"):
        super().__init__()
        self.save_dir = save_dir
        self.train_mses = []
        self.val_losses = {}        # {epoch: {"6": 0.123, "24": ...}}
        self.current_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        train_mse = trainer.callback_metrics.get("train/w_mse_aggregate")
        if train_mse is not None:
            self.train_mses.append(train_mse.item())
            print(f"Epoch {self.current_epoch} - Train MSE: {train_mse.item():.6f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        # Discover validation metrics for all configured lead times.
        val_keys = sorted(
            [k for k in metrics.keys()
             if k.startswith("val/w_mse_aggregate_") and k.endswith("_hrs_ensemble_mean")],
            key=lambda k: int(k.split("aggregate_")[1].split("_hrs")[0]),
        )
        if not val_keys:
            return
        self.val_losses[self.current_epoch] = {}
        parts = []
        for k in val_keys:
            lead = k.split("aggregate_")[1].split("_hrs")[0]
            self.val_losses[self.current_epoch][lead] = metrics[k].item()
            parts.append(f"Val Loss {lead}h: {metrics[k].item():.6f}")
        print(f"Epoch {self.current_epoch} - " + ", ".join(parts))
        self._save_losses()
        self.current_epoch += 1

    def _save_losses(self):
        os.makedirs(self.save_dir, exist_ok=True)
        filepath = os.path.join(self.save_dir, "loss_history.txt")
        # Keep all lead times seen so far in numeric order.
        all_leads = sorted({int(l) for ep in self.val_losses.values() for l in ep.keys()})

        with open(filepath, "w") as f:
            f.write("epoch,train_mse," + ",".join(f"val_loss_{lead}h" for lead in all_leads) + "\n")
            for epoch in range(len(self.train_mses)):
                mse = self.train_mses[epoch]
                row = [str(epoch), str(mse)]
                for lead in all_leads:
                    row.append(str(self.val_losses.get(epoch, {}).get(str(lead), "")))
                f.write(",".join(row) + "\n")
        print(f"Loss history saved to: {filepath}")
