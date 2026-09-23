"""Latitude-weighted weather-forecast metrics."""

import numpy as np
import torch


def lat_weighted_mse(pred, y, vars, lat, weighted=False, weight_dict=None):
    """Compute latitude-weighted MSE with optional variable weights."""
    # Pointwise squared error.
    error = (pred - y) ** 2

    # Normalize cosine latitude weights to mean one.
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    # Per-variable losses.
    for i, var in enumerate(vars):
        loss_dict[f"w_mse_{var}"] = (error[:, i] * w_lat).mean()

    # Aggregate across variables.
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in vars]).to(device=error.device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(vars)).to(device=error.device).view(1, -1, 1, 1) / len(vars)

    loss_dict["w_mse_aggregate"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    """Compute validation MSE in normalized space without gradients."""
    error = (pred - y) ** 2  # [B, V, H, W]

    # Latitude weights.
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    # Validation does not require gradients.
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    # Aggregate loss.
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in vars]).to(device=error.device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(vars)).to(device=error.device).view(1, -1, 1, 1) / len(vars)

    loss_dict[f"w_mse_aggregate_{log_postfix}"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    """Compute physical-unit latitude-weighted RMSE."""
    # Convert to physical units.
    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2

    # Latitude weights.
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            # Average spatially, then take the root and batch mean.
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    return loss_dict

def lat_weighted_crps(pred: torch.Tensor, y: torch.Tensor, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    """Compute latitude-weighted CRPS for ensemble forecasts."""
    # The ensemble adds one dimension relative to the target.
    assert len(pred.shape) == len(y.shape) + 1
    # pred: [B, N, V, H, W]; y: [B, V, H, W].
    pred = transform(pred)
    y = transform(y)

    H, N = pred.shape[-2], pred.shape[1]

    # Latitude weights.
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device)

    def crps_var(pred_var: torch.Tensor, y_var: torch.Tensor):
        """Compute CRPS for one variable."""
        with torch.no_grad():
            error_term = torch.abs(pred_var - y_var.unsqueeze(1))  # [B, N, H, W]
            error_term = error_term * w_lat.view(1, 1, H, 1)
            error_term = torch.mean(error_term)

        # Ensemble spread term.
        with torch.no_grad():
            spread_term = torch.abs(pred_var.unsqueeze(2) - pred_var.unsqueeze(1))
            spread_term = spread_term * w_lat.view(1, 1, 1, H, 1)
            spread_term = spread_term.mean(dim=(-2, -1))
            spread_term = spread_term.sum(dim=(1, 2)) / (2 * N * (N - 1))
            spread_term = spread_term.mean()

        return error_term - spread_term

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"w_crps_{var}_{log_postfix}"] = crps_var(pred[:, :, i], y[:, i])

    return loss_dict

def lat_weighted_spread_skill_ratio(pred: torch.Tensor, y: torch.Tensor, transform, vars, lat, log_postfix, weighted=False, weight_dict=None):
    """Compute the latitude-weighted ensemble spread-skill ratio."""
    assert len(pred.shape) == len(y.shape) + 1
    # pred: [B, N, V, H, W]; y: [B, V, H, W].

    # Ensemble-mean RMSE is the skill denominator.
    rmse_dict = lat_weighted_rmse(pred.mean(dim=1), y, transform, vars, lat, log_postfix, weighted, weight_dict)

    pred = transform(pred)
    y = transform(y)

    H = pred.shape[-2]

    # Latitude weights.
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device)

    # Compute ensemble spread.
    var = torch.var(pred, dim=1)
    var = var * w_lat.view(1, 1, H, 1)
    spread = var.mean(dim=(-2, -1)).sqrt().mean(dim=0)

    # SSR = spread / RMSE.
    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"w_ssr_{var}_{log_postfix}"] = spread[i] / rmse_dict[f"w_rmse_{var}_{log_postfix}"]

    return loss_dict

def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """Compute latitude-weighted anomaly correlation coefficients."""
    # Convert to physical units.
    pred = transform(pred)
    y = transform(y)

    # Latitude weights.
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)

    # Remove climatology to obtain anomalies.
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            # Remove the batch mean.
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            # Weighted covariance divided by weighted standard deviations.
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    return loss_dict
