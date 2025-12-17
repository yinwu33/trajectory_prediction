import torch
import torch.nn.functional as F


def ADE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute Average Displacement Error (ADE).

    Args:
        pred: [B, T, 2] predicted trajectories
        gt: [B, T, 2] ground truth trajectories

    Returns:
        ADE: [B] average displacement error
    """
    ade = torch.norm(pred - gt, dim=-1).mean(dim=-1)  # [B]
    return ade


def FDE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute Final Displacement Error (FDE).

    Args:
        pred: [B, T, 2] predicted trajectories
        gt: [B, T, 2] ground truth trajectories

    Returns:
        FDE: [B] final displacement error
    """
    fde = torch.norm(pred[:, -1, :] - gt[:, -1, :], dim=-1)  # [B]
    return fde


def minADE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute Minimum Average Displacement Error (minADE).

    Args:
        pred: [B, K, T, 2] predicted trajectories
        gt: [B, T, 2] ground truth trajectories

    Returns:
        minADE: [B] minimum ADE over K predictions
    """
    B, K, T, _ = pred.shape
    gt_expanded = gt.unsqueeze(1).expand(B, K, T, 2)
    ade = torch.norm(pred - gt_expanded, dim=-1).mean(dim=-1)  # [B, K]
    min_ade, _ = ade.min(dim=-1)  # [B]
    return min_ade


def minFDE(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute Minimum Final Displacement Error (minFDE).

    Args:
        pred: [B, K, T, 2] predicted trajectories
        gt: [B, T, 2] ground truth trajectories

    Returns:
        minFDE: [B] minimum FDE over K predictions
    """
    B, K, T, _ = pred.shape
    gt_final = gt[:, -1, :].unsqueeze(1).expand(B, K, 2)  # [B, K, 2]
    pred_final = pred[:, :, -1, :]  # [B, K, 2]
    fde = torch.norm(pred_final - gt_final, dim=-1)  # [B, K]
    min_fde, _ = fde.min(dim=-1)  # [B]
    return min_fde
