#!/usr/bin/env python3
"""Evaluate a WEAVER checkpoint with latitude-weighted RMSE and ACC.

The evaluator caches rollouts shared by multiple lead times and can resume
from a JSON progress checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
import yaml
from torch.utils.data import DataLoader

# Add the project root for source-checkout execution.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from weaver.utils.release_paths import default_checkpoint_path, default_config_path, release_root

RELEASE_ROOT = release_root()
DEFAULT_CONFIG = default_config_path()
DEFAULT_CHECKPOINT = default_checkpoint_path()


def resolve_config_path(value, config_dir):
    """Resolve beside-config paths, then fall back to the release root."""

    path = Path(value)
    if path.is_absolute():
        return path
    config_dir = Path(config_dir).resolve()
    beside_config = config_dir / path
    if beside_config.exists():
        return beside_config
    project_path = RELEASE_ROOT / path
    if project_path.exists():
        return project_path
    if config_dir.name == "configs":
        return project_path
    return beside_config

from weaver.data.multi_step_datamodule import (
    MultiStepDataRandomizedModule,
    collate_fn_val,
)
from weaver.models.forecast_module import WeatherForecastModule


def parse_config(config_path: str):
    """Load YAML and resolve simple ``${section.key}`` references."""
    with open(config_path, "r") as f:
        content = f.read()

    cfg = yaml.safe_load(content)
    raw = yaml.safe_load(content)

    def resolve(d, root):
        if isinstance(d, dict):
            return {k: resolve(v, root) for k, v in d.items()}
        if isinstance(d, list):
            return [resolve(v, root) for v in d]
        if isinstance(d, str) and d.startswith("${") and d.endswith("}"):
            path = d[2:-1].split(".")
            val = root
            for p in path:
                val = val[p]
            return val
        return d

    return resolve(raw, cfg)


def load_model(cfg: dict, ckpt_path: str, device: str = "cuda", config_dir=None):
    """Instantiate the configured network and load a strict checkpoint."""
    import importlib
    config_dir = Path(config_dir or cfg.get("_config_dir", PROJECT_ROOT)).resolve()
    net_cfg = cfg["model"]["net"]["init_args"]
    for key in ("region_map_path", "region_prior_init_path"):
        value = net_cfg.get(key)
        if value and not Path(value).is_absolute():
            net_cfg[key] = str(resolve_config_path(value, config_dir))
    class_path = cfg["model"]["net"]["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    net_cls = getattr(importlib.import_module(module_name), class_name)
    net = net_cls(**net_cfg)

    # Evaluation does not use training-only pretrained/freeze options.
    skip = {"net", "pretrained_path", "freeze_backbone"}
    model_kwargs = {k: v for k, v in cfg["model"].items() if k not in skip}
    model = WeatherForecastModule(net, **model_kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key[len("model."):] if key.startswith("model.") else key
        if normalized_key in normalized_state_dict:
            raise RuntimeError(
                "Checkpoint contains duplicate parameters after prefix normalization: "
                f"{normalized_key!r}"
            )
        normalized_state_dict[normalized_key] = value
    state_dict = normalized_state_dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={missing[:20]} (total {len(missing)}), "
            f"unexpected={unexpected[:20]} (total {len(unexpected)})"
        )

    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()
    return model


def build_datamodule(cfg: dict, lead_times=None):
    """Build the data module, optionally overriding evaluation lead times."""
    data_cfg = dict(cfg["data"])
    statistics_dir = data_cfg.get("statistics_dir")
    config_dir = Path(cfg.get("_config_dir", PROJECT_ROOT)).resolve()
    root_dir = data_cfg.get("root_dir")
    if root_dir and not Path(root_dir).is_absolute():
        data_cfg["root_dir"] = str(resolve_config_path(root_dir, config_dir))
    if statistics_dir and not Path(statistics_dir).is_absolute():
        data_cfg["statistics_dir"] = str(resolve_config_path(statistics_dir, config_dir))
    if lead_times is not None:
        data_cfg["val_lead_times"] = lead_times
    return MultiStepDataRandomizedModule(**data_cfg)


def load_climatology(clim_path: str, variables: list, device: str = "cuda"):
    """Load climatology in the configured variable order."""
    clim_dict = dict(np.load(clim_path, allow_pickle=False))
    clim = np.stack([clim_dict[v] for v in variables], axis=0)  # [V, H, W]
    return torch.from_numpy(clim).to(device)


# ── Rollout cache ────────────────────────────────────────────────────────────


def _rollout_with_cache(
    model: WeatherForecastModule,
    inp: torch.Tensor,
    variables: list,
    interval: int,
    max_steps: int,
    needed_lead_times: set,
) -> dict:
    """Roll out one interval and cache predictions at requested lead times."""
    interval_tensor = (
        torch.Tensor([interval])
        .to(device=inp.device, dtype=inp.dtype)
        / 10.0
    )
    interval_tensor = interval_tensor.repeat(inp.shape[0])

    x = inp.clone()
    results = {}
    with autocast():
        for s in range(max_steps):
            pred_diff = model(x, variables, interval_tensor)
            pred_diff = model.replace_constant(pred_diff, variables)
            pred_diff = model.reverse_diff_transform[interval](pred_diff)
            pred = model.reverse_inp_transform(x) + pred_diff
            x = model.inp_transform(pred)

            step_lead = (s + 1) * interval
            if step_lead in needed_lead_times:
                results[step_lead] = x.clone()
    return results


# ── Resumable evaluation ─────────────────────────────────────────────────────


def save_checkpoint(path: str, accum: dict, last_batch_idx: int):
    """Save intermediate evaluation state to a JSON checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    serializable = json.loads(json.dumps(accum, default=float))
    payload = {
        "last_batch_idx": last_batch_idx,
        "accum": serializable,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    print(f"  [checkpoint saved] batch {last_batch_idx} → {path}")


def load_checkpoint(path: str):
    """Load ``(accum, last_batch_idx)`` from a JSON checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path) as f:
        payload = json.load(f)
    return payload["accum"], payload["last_batch_idx"]


# ── Main evaluation loop ────────────────────────────────────────────────────


def run_evaluation(
    model: WeatherForecastModule,
    datamodule: MultiStepDataRandomizedModule,
    device: str = "cuda",
    clim_path: str = None,
    variables: list = None,
    batch_size: int = None,
    lead_times: list = None,
    resume_from: str = None,
    checkpoint_path: str = None,
    max_batches: int = None,
):
    """Compute RMSE and ACC for each variable and requested lead time."""
    datamodule.setup("test")
    bs = batch_size if batch_size is not None else datamodule.hparams.val_batch_size
    n_test = len(datamodule.data_test)
    eval_lead_times = lead_times or datamodule.hparams.val_lead_times
    list_intervals = datamodule.hparams.list_train_intervals

    loader = DataLoader(
        datamodule.data_test,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn_val,
    )
    n_total_batches = len(loader)
    if max_batches is not None:
        if max_batches <= 0:
            raise ValueError("max_batches must be positive")
        n_total_batches = min(n_total_batches, max_batches)

    lat = datamodule.get_lat_lon()[0]
    all_variables = datamodule.hparams.variables
    reverse_transform = model.reverse_inp_transform

    # Select variables to report.
    if variables is not None:
        eval_vars = [v for v in variables if v in all_variables]
        eval_indices = [all_variables.index(v) for v in eval_vars]
    else:
        eval_vars = all_variables
        eval_indices = list(range(len(all_variables)))

    # Load climatology for ACC when available.
    clim_t = None
    if clim_path and os.path.exists(clim_path):
        clim_t = load_climatology(clim_path, all_variables, device)
        print(f"Loaded climatology from: {clim_path}  shape={clim_t.shape}")
    elif clim_path:
        print(f"[WARN] Climatology file not found: {clim_path}, skipping ACC")

    # Compute latitude weights once.
    w_lat_np = np.cos(np.deg2rad(lat))
    w_lat_np = w_lat_np / w_lat_np.mean()

    # ── Restore or initialize accumulators ───────────────────────────────
    resume_batch_idx = -1
    if resume_from:
        try:
            loaded_accum, resume_batch_idx = load_checkpoint(resume_from)
            # JSON serializes integer keys as strings; restore them here.
            accum = {}
            for lt_str, lt_data in loaded_accum.items():
                lt = int(lt_str)
                accum[lt] = {}
                for v, v_data in lt_data.items():
                    accum[lt][v] = v_data
            print(f"Resumed from checkpoint (batch {resume_batch_idx}): {resume_from}")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"[WARN] Cannot load checkpoint '{resume_from}': {e}")
            print("Starting from scratch.")
            resume_from = None

    if resume_from is None or resume_batch_idx < 0:
        # Initialize accumulators.
        accum = {}
        for lt in eval_lead_times:
            accum[lt] = {}
            for v in eval_vars:
                acc = {"rmse_sum": 0.0, "n": 0}
                if clim_t is not None:
                    acc.update({
                        "sum_pred": 0.0, "sum_y": 0.0, "sum_w": 0.0,
                        "sum_w_pred": 0.0, "sum_w_y": 0.0,
                        "sum_w_pred_y": 0.0, "sum_w_pred2": 0.0, "sum_w_y2": 0.0,
                        "n_elements": 0,
                    })
                accum[lt][v] = acc
        resume_batch_idx = -1
        print(f"Starting new evaluation on {n_test} test samples ({n_total_batches} batches)")

    # ── Skip completed batches ────────────────────────────────────────────
    skip_batches = resume_batch_idx + 1
    if skip_batches > 0:
        if skip_batches >= n_total_batches:
            print(f"[WARN] resume batch {resume_batch_idx} >= total {n_total_batches}, nothing to do.")
            has_acc = clim_t is not None
            return _finalize_results(accum, eval_lead_times, eval_vars, has_acc)

        from itertools import islice
        loader_iter = islice(iter(loader), skip_batches, None)
        print(f"Resuming from batch {skip_batches}/{n_total_batches} (skipping {skip_batches})")
    else:
        loader_iter = iter(loader)

    # ── Checkpoint path ───────────────────────────────────────────────────
    if checkpoint_path is None and resume_from is not None:
        checkpoint_path = resume_from

    # ── Main loop ─────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch_idx, (inp, targets, _vars) in enumerate(loader_iter, start=skip_batches):
            if batch_idx >= n_total_batches:
                break
            inp = inp.to(device)
            B = inp.shape[0]
            w_lat_t = torch.from_numpy(w_lat_np).to(
                dtype=inp.dtype, device=device
            ).unsqueeze(0).unsqueeze(-1)

            # Roll out each interval only to its longest requested lead time.
            batch_preds = {lt: [] for lt in eval_lead_times}

            for interval in list_intervals:
                valid_lead_times = {
                    lt for lt in eval_lead_times if lt % interval == 0
                }
                if not valid_lead_times:
                    continue
                max_lt = max(valid_lead_times)
                max_steps = max_lt // interval

                cached = _rollout_with_cache(
                    model, inp, all_variables,
                    interval, max_steps, valid_lead_times,
                )
                for lt in valid_lead_times:
                    batch_preds[lt].append(cached[lt])
                del cached
                torch.cuda.empty_cache()

            # Average all valid interval predictions.
            for lt in eval_lead_times:
                if not batch_preds[lt]:
                    continue
                ensemble_pred = torch.stack(batch_preds[lt], dim=0).mean(0)

                y = targets[lt].to(device)
                pred_phys = reverse_transform(ensemble_pred)
                y_phys = reverse_transform(y)
                error = (pred_phys - y_phys) ** 2

                for vi in eval_indices:
                    var = all_variables[vi]
                    b = inp.shape[0]

                    # RMSE.
                    per_sample_mse = (error[:, vi] * w_lat_t).mean(dim=(-2, -1))
                    per_sample_rmse = torch.sqrt(per_sample_mse)
                    accum[lt][var]["rmse_sum"] += per_sample_rmse.sum().item()
                    accum[lt][var]["n"] += b

                    # ACC.
                    if clim_t is not None:
                        pred_anom = pred_phys[:, vi] - clim_t[vi]
                        y_anom = y_phys[:, vi] - clim_t[vi]
                        d_acc = accum[lt][var]
                        d_acc["sum_pred"] += pred_anom.sum().item()
                        d_acc["sum_y"] += y_anom.sum().item()
                        d_acc["sum_w"] += (w_lat_t * torch.ones_like(pred_anom)).sum().item()
                        d_acc["sum_w_pred"] += (w_lat_t * pred_anom).sum().item()
                        d_acc["sum_w_y"] += (w_lat_t * y_anom).sum().item()
                        d_acc["sum_w_pred_y"] += (w_lat_t * pred_anom * y_anom).sum().item()
                        d_acc["sum_w_pred2"] += (w_lat_t * pred_anom ** 2).sum().item()
                        d_acc["sum_w_y2"] += (w_lat_t * y_anom ** 2).sum().item()
                        d_acc["n_elements"] += pred_anom.numel()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{n_total_batches} batches")

            # Save progress every ten batches.
            if checkpoint_path and (batch_idx + 1) % 10 == 0:
                save_checkpoint(checkpoint_path, accum, batch_idx)

    # Finalize results.
    has_acc = clim_t is not None
    return _finalize_results(accum, eval_lead_times, eval_vars, has_acc)


def _finalize_results(accum, eval_lead_times, eval_vars, has_acc: bool):
    """Finalize RMSE and ACC from accumulated statistics."""
    results = []
    for lt in sorted(accum.keys()):
        entry = {"lead_time": lt, "metrics": []}
        for var in eval_vars:
            d = accum[lt][var]
            s, n = d["rmse_sum"], d["n"]
            rmse = float(s / n) if n > 0 else float("nan")

            metric = {"variable": var, "rmse": round(rmse, 6)}

            if has_acc:
                # ACC uses the latitude-weighted global mean.  Since the
                # latitude weights are not spatially uniform, an unweighted
                # mean here would not describe the same weighted population
                # as the covariance terms below.
                p_mean = d["sum_w_pred"] / d["sum_w"]
                y_mean = d["sum_w_y"] / d["sum_w"]

                numerator = (d["sum_w_pred_y"]
                             - y_mean * d["sum_w_pred"]
                             - p_mean * d["sum_w_y"]
                             + p_mean * y_mean * d["sum_w"])
                denom_p = (d["sum_w_pred2"]
                           - 2 * p_mean * d["sum_w_pred"]
                           + p_mean ** 2 * d["sum_w"])
                denom_y = (d["sum_w_y2"]
                           - 2 * y_mean * d["sum_w_y"]
                           + y_mean ** 2 * d["sum_w"])

                acc_val = numerator / np.sqrt(denom_p * denom_y) if (denom_p * denom_y) > 0 else float("nan")
                metric["acc"] = round(acc_val, 6)

            entry["metrics"].append(metric)
        results.append(entry)
    return results, has_acc


# ── Result formatting ────────────────────────────────────────────────────────


def print_results(results, has_acc: bool):
    """Print a compact metrics table."""
    variables = [m["variable"] for m in results[0]["metrics"]]
    lead_times = [r["lead_time"] for r in results]

    print("\n" + "=" * 110)
    print("TEST EVALUATION RESULTS")
    print("=" * 110)

    col_w = 32
    metric_w = 11
    print(f"\n{'Variable':<{col_w}}", end="")
    for lt in lead_times:
        print(f"  RMSE ({lt:>3d}h)", end="")
    print()

    print("-" * (col_w + 15 * len(lead_times)))
    for vi, var in enumerate(variables):
        print(f"{var:<{col_w}}", end="")
        for ri in range(len(lead_times)):
            rmse = results[ri]["metrics"][vi]["rmse"]
            print(f"  {rmse:>{metric_w}.4f}", end="")
        print()

    print("\n" + "-" * 80)
    print(f"{'Mean RMSE (all vars)':<{col_w}}", end="")
    for ri in range(len(lead_times)):
        rmse_vals = [results[ri]["metrics"][vi]["rmse"] for vi in range(len(variables))]
        print(f"  {np.mean(rmse_vals):>{metric_w}.4f}", end="")
    print()

    if has_acc:
        print("\n" + "=" * 110)
        print("ACC (Anomaly Correlation Coefficient, higher is better)")
        print(f"\n{'Variable':<{col_w}}", end="")
        for lt in lead_times:
            print(f"  ACC ({lt:>3d}h)", end="")
        print()

        print("-" * (col_w + 15 * len(lead_times)))
        for vi, var in enumerate(variables):
            print(f"{var:<{col_w}}", end="")
            for ri in range(len(lead_times)):
                acc = results[ri]["metrics"][vi].get("acc", float("nan"))
                print(f"  {acc:>{metric_w}.4f}", end="")
            print()

        print("\n" + "-" * 80)
        print(f"{'Mean ACC (all vars)':<{col_w}}", end="")
        for ri in range(len(lead_times)):
            acc_vals = [results[ri]["metrics"][vi].get("acc", float("nan"))
                        for vi in range(len(variables))]
            valid = [a for a in acc_vals if not np.isnan(a)]
            print(f"  {np.mean(valid):>{metric_w}.4f}" if valid else "       N/A", end="")
        print()

    key_vars = ["2m_temperature", "10m_u_component_of_wind",
                "10m_v_component_of_wind", "mean_sea_level_pressure",
                "geopotential_500", "temperature_500"]
    present_keys = [v for v in key_vars if v in variables]
    if present_keys:
        print("\n" + "=" * 110)
        cols = ["RMSE"] + (["ACC"] if has_acc else [])
        print("Key variables")
        header = f"{'Variable':<{col_w}}"
        for col in cols:
            for lt in lead_times:
                header += f"  {col} ({lt:>3d}h)"
        print(header)
        print("-" * len(header))

        for var in present_keys:
            vi = variables.index(var)
            print(f"{var:<{col_w}}", end="")
            for ri in range(len(lead_times)):
                print(f"  {results[ri]['metrics'][vi]['rmse']:>{metric_w}.4f}", end="")
            if has_acc:
                for ri in range(len(lead_times)):
                    acc = results[ri]["metrics"][vi].get("acc", float("nan"))
                    print(f"  {acc:>{metric_w}.4f}", end="")
            print()

    print("=" * 110 + "\n")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WEAVER checkpoint on test set (supports resume)"
    )
    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG),
                        help="Path to training config YAML")
    parser.add_argument("--ckpt", default=str(DEFAULT_CHECKPOINT),
                        help="Path to checkpoint")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to save JSON metrics (optional)")
    parser.add_argument(
        "--clim", default=None,
        help="Path to climatology npz file (default: assets/statistics beside --config; set to '' to skip ACC)",
    )
    parser.add_argument("--lead-times", nargs="+", type=int, default=None,
                        help="Lead times to evaluate (default: config val_lead_times)")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--data-root", default=None,
                        help="Override data root_dir")
    parser.add_argument("--variables", nargs="*", default=None,
                        help="Variables to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: config val_batch_size)")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from checkpoint file path")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit test batches for a smoke test; omit for paper metrics")
    args = parser.parse_args()

    # --- 1. Load configuration ---
    print(f"Loading config: {args.config}")
    cfg = parse_config(args.config)
    cfg["_config_dir"] = str(Path(args.config).resolve().parent)
    if args.data_root:
        cfg["data"]["root_dir"] = args.data_root

    # --- 2. Load model and configure acceleration ---
    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(
        cfg, args.ckpt, device=args.device,
        config_dir=cfg["_config_dir"],
    )

    # Enable cuDNN autotuning and Tensor Core matmul.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # --- 3. Build data module ---
    lead_times = args.lead_times if args.lead_times else cfg["data"]["val_lead_times"]
    print(f"Lead times: {lead_times}")
    print(f"Data root: {cfg['data']['root_dir']}")
    datamodule = build_datamodule(cfg, lead_times=lead_times)

    # --- 4. Set model metadata ---
    model.set_lat_lon(*datamodule.get_lat_lon())
    model.set_transforms(*datamodule.get_transforms())
    model.set_base_intervals_and_lead_times(
        datamodule.hparams.list_train_intervals,
        lead_times,
    )

    # --- 5. Resolve climatology ---
    clim_path = (
        args.clim if args.clim is not None
        else str(resolve_config_path(
            "assets/statistics/clim.npz", cfg["_config_dir"]
        ))
    )

    # --- 6. Run evaluation ---
    datamodule.setup("test")
    n_test = len(datamodule.data_test)
    print(f"\nRunning evaluation on {n_test} test samples...")

    ckpt_path = args.resume_from
    if ckpt_path is None and args.output:
        ckpt_path = args.output + ".checkpoint"

    results, has_acc = run_evaluation(
        model, datamodule, device=args.device, clim_path=clim_path,
        variables=args.variables, batch_size=args.batch_size,
        lead_times=lead_times,
        resume_from=args.resume_from,
        checkpoint_path=ckpt_path,
        max_batches=args.max_batches,
    )

    # --- 7. Write output ---
    print_results(results, has_acc)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
