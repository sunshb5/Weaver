

from typing import Any, List, Tuple, Dict

import torch
from lightning import LightningModule
from torchvision.transforms import transforms

from weaver.models.backbone import WeaverBackbone
from weaver.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from weaver.utils.metrics import (
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from weaver.utils.data_utils import CONSTANTS, WEIGHT_DICT


class WeatherForecastModule(LightningModule):
    
    def __init__(
        self,
        net: WeaverBackbone,
        weighted_loss: bool = True,
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        pretrained_path: str = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.save_hyperparameters(ignore=['net'])
        self.net = net

        
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path)

        
        
        if freeze_backbone:
            n_trainable = 0
            for name, p in self.named_parameters():
                trainable = (".moe." in name) and \
                            ("region_bias" not in name) and \
                            ("bias_logit_scale" not in name)
                p.requires_grad = trainable
                if trainable:
                    n_trainable += 1
            
            print(f"[FreezeBackbone] frozen. trainable params: {n_trainable}")

    def load_pretrained_weights(self, pretrained_path):
        
        
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(
                pretrained_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        state_dict = checkpoint["state_dict"]
        
        
        
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[Pretrained] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[Pretrained] missing keys: {missing[:5]}")
        if unexpected:
            print(f"[Pretrained] unexpected keys: {unexpected[:5]}")
    def set_base_intervals_and_lead_times(self, list_train_intervals, val_lead_times):
        
        self.val_lead_times = val_lead_times
        self.list_train_intervals = list_train_intervals

    def set_lat_lon(self, lat, lon):
        
        self.lat = lat
        self.lon = lon

    def set_transforms(self, inp_transform, diff_transform):
        
        self.inp_transform = inp_transform
        
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)

        self.diff_transform = diff_transform
        
        self.reverse_diff_transform = {
            k: self.get_reverse_transform(v) for k, v in diff_transform.items()
        }

    def get_reverse_transform(self, transform):
        
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)
    
    def replace_constant(self, yhat, out_variables):
        
        for i in range(yhat.shape[1]):
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = 0.0  
        return yhat

    def pad(self, x: torch.Tensor):
        
        h = x.shape[-2]
        
        if h % self.net.patch_size != 0:
            pad_size = self.net.patch_size - h % self.net.patch_size
            
            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0
        return padded_x, pad_size

    def forward(self, x: torch.Tensor, variables, interval) -> torch.Tensor:
        
        padded_x, pad_size = self.pad(x)  
        output = self.net(padded_x, variables, interval)[:, :, pad_size:]  
        return output

    def forward_train(self, x: torch.Tensor, variables, interval_tensors, mean_diff_transform, std_diff_transform):
        
        norm_diffs = []
        
        mean_diff_transform = mean_diff_transform.unsqueeze(-1).unsqueeze(-1)
        std_diff_transform = std_diff_transform.unsqueeze(-1).unsqueeze(-1)
        n_steps = interval_tensors.shape[-1]  

        
        for i in range(n_steps):
            
            norm_pred_diff = self(x, variables, interval_tensors[:, i])
            
            norm_pred_diff = self.replace_constant(norm_pred_diff, variables)
            norm_diffs.append(norm_pred_diff)
            
            raw_pred_diff = norm_pred_diff * std_diff_transform + mean_diff_transform
            
            pred = self.reverse_inp_transform(x) + raw_pred_diff
            
            x = self.inp_transform(pred)
        return norm_diffs

    def training_step(self, batch: Any, batch_idx: int):
        
        x, gt_diff, mean_diff_transform, std_diff_transform, interval_tensors, variables = batch
        
        pred_diff = self.forward_train(x, variables, interval_tensors, mean_diff_transform, std_diff_transform)
        
        pred_diff = torch.stack(pred_diff, dim=1).flatten(0, 1)
        gt_diff = gt_diff.flatten(0, 1)  
        
        loss_dict = lat_weighted_mse(
            pred_diff,
            gt_diff,
            variables,
            self.lat,
            weighted=self.hparams.weighted_loss,
            weight_dict=WEIGHT_DICT
        )

        
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=x.shape[0],
            )

        
        main_loss = loss_dict[f"w_mse_aggregate"]

        
        self.log(
            "train/total_loss",
            main_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=x.shape[0],
        )

        
        return main_loss
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        
        self.evaluate(batch, self.val_lead_times, "val")

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        
        self.evaluate(batch, self.val_lead_times, "test")

    def forward_validation(self, x: torch.Tensor, variables, interval, steps):
        
        
        interval_tensor = torch.Tensor([interval]).to(device=x.device, dtype=x.dtype) / 10.0
        interval_tensor = interval_tensor.repeat(x.shape[0])  

        
        for _ in range(steps):
            
            pred_diff = self(x, variables, interval_tensor)
            
            pred_diff = self.replace_constant(pred_diff, variables)
            
            pred_diff = self.reverse_diff_transform[interval](pred_diff)
            
            pred = self.reverse_inp_transform(x) + pred_diff
            
            x = self.inp_transform(pred)
        return x
    
    def evaluate(
        self, batch: Tuple[torch.Tensor, Dict, List[str], List[str]],
        val_lead_times: List[int],
        stage: str
    ):
        
        x, dict_y, variables = batch

        def get_loss_dict(y, yhat, list_metrics, postfix):
            
            all_loss_dicts = []
            for metric in list_metrics:
                loss_dict = metric(
                    yhat,
                    y,
                    self.reverse_inp_transform,
                    variables,
                    lat=self.lat,
                    log_postfix=postfix,
                    weighted=self.hparams.weighted_loss,
                    weight_dict=WEIGHT_DICT
                )
                all_loss_dicts.append(loss_dict)

            
            final_loss_dict = {}
            for d in all_loss_dicts:
                final_loss_dict.update(d)

            
            final_loss_dict = {f"{stage}/{k}": v for k, v in final_loss_dict.items()}
            return final_loss_dict

        for target_lead_time in val_lead_times:
            all_norm_preds = []
            
            for base_interval in self.list_train_intervals:
                
                if target_lead_time % base_interval == 0:
                    steps = target_lead_time // base_interval  
                    
                    norm_pred = self.forward_validation(x, variables, base_interval, steps)
                    
                    base_loss_dict = get_loss_dict(
                        dict_y[target_lead_time],
                        norm_pred,
                        list_metrics=[lat_weighted_mse_val, lat_weighted_rmse],
                        postfix=f"{target_lead_time}_hrs_base_{base_interval}"
                    )
                    all_norm_preds.append(norm_pred)

                    
                    self.log_dict(
                        base_loss_dict,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=x.shape[0],
                    )

            
            mean_norm_pred = torch.stack(all_norm_preds, dim=0).mean(0)
            
            ensemble_loss_dict = get_loss_dict(
                dict_y[target_lead_time],
                mean_norm_pred,
                list_metrics=[lat_weighted_mse_val, lat_weighted_rmse],
                postfix=f"{target_lead_time}_hrs_ensemble_mean"
            )

            
            self.log_dict(
                ensemble_loss_dict,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

    def configure_optimizers(self):
        
        decay = []      
        no_decay = []   
        for name, m in self.named_parameters():
            
            if not m.requires_grad:
                continue
            
            if "channel_embed" in name or "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,  
                },
            ]
        )

        
        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,   
            self.hparams.max_epochs * n_steps,       
            self.hparams.warmup_start_lr,            
            self.hparams.eta_min,                    
        )
        
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
