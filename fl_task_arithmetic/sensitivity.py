"""
Sensitivity scoring for gradient mask calibration in sparse fine-tuning.
Uses gradient magnitude to identify least-sensitive parameters.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from torch.utils.data import DataLoader


def compute_sensitivity_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute sensitivity scores based on gradient magnitude."""
    model.train()
    model.to(device)

    sensitivity_scores: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            sensitivity_scores[name] = torch.zeros_like(param.data)

    criterion = nn.CrossEntropyLoss()
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        if isinstance(batch, dict):
            images = batch["img"].to(device)
            labels = batch["fine_label"].to(device)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                sensitivity_scores[name] += param.grad.abs()

        batch_count += 1

    for name in sensitivity_scores:
        sensitivity_scores[name] /= max(batch_count, 1)

    return sensitivity_scores


def calibrate_gradient_masks(
    model: nn.Module,
    dataloader: DataLoader,
    sparsity_ratio: float,
    num_calibration_rounds: int,
    device: torch.device,
    num_batches_per_round: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Iteratively calibrate binary masks by freezing least-sensitive parameters."""
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"

    print(f"Starting mask calibration with {num_calibration_rounds} rounds...")
    print(f"Sparsity ratio: {sparsity_ratio:.2%} (freezing {sparsity_ratio:.2%} of parameters)")

    masks: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            masks[name] = torch.ones_like(param.data)

    for round_idx in range(num_calibration_rounds):
        print(f"\nCalibration round {round_idx + 1}/{num_calibration_rounds}")

        sensitivity_scores = compute_sensitivity_scores(
            model, dataloader, device, num_batches_per_round
        )

        all_scores = []
        for name in sensitivity_scores:
            active_scores = sensitivity_scores[name][masks[name] == 1]
            all_scores.append(active_scores.flatten())

        if len(all_scores) == 0:
            print("Warning: No active parameters remaining!")
            break

        all_scores = torch.cat(all_scores)
        threshold = torch.quantile(all_scores, sparsity_ratio)
        print(f"  Sensitivity threshold: {threshold:.6f}")

        total_params = 0
        frozen_params = 0
        for name in masks:
            param_mask = sensitivity_scores[name] > threshold
            masks[name] = param_mask.float()
            total_params += masks[name].numel()
            frozen_params += (masks[name] == 0).sum().item()

        actual_sparsity = frozen_params / total_params if total_params > 0 else 0
        print(f"  Frozen parameters: {frozen_params}/{total_params} ({actual_sparsity:.2%})")

    print("\nMask calibration complete!")
    return masks


def get_masked_parameters(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """Create an iterator of (mask, parameter) tuples for SparseSGDM."""
    masked_params = []
    for name, param in model.named_parameters():
        if name in masks and param.requires_grad:
            masked_params.append((masks[name], param))
    return masked_params
