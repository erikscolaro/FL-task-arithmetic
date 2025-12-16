"""
Sensitivity scoring for gradient mask calibration in sparse fine-tuning.

This module identifies the least-sensitive parameters (those with low gradient magnitude)
to determine which weights can be frozen during fine-tuning.
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
    """
    Compute sensitivity scores based on gradient magnitude.
    
    Parameters with higher gradient magnitude are more sensitive to updates.
    We freeze parameters with LOW sensitivity scores (below threshold).
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for calibration data
        device: Device to run computation on
        num_batches: Number of batches to use (None = use all)
        
    Returns:
        Dictionary mapping parameter names to sensitivity scores (same shape as params)
    """
    model.train()
    model.to(device)
    
    # Initialize score accumulator
    sensitivity_scores = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            sensitivity_scores[name] = torch.zeros_like(param.data)
    
    criterion = nn.CrossEntropyLoss()
    
    # Accumulate gradients over batches
    batch_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
            
        # Extract data (support both dict and tuple formats)
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
        
        # Accumulate absolute gradient magnitudes
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                sensitivity_scores[name] += param.grad.abs()
        
        batch_count += 1
    
    # Average over batches
    for name in sensitivity_scores:
        sensitivity_scores[name] /= batch_count
    
    return sensitivity_scores


def calibrate_gradient_masks(
    model: nn.Module,
    dataloader: DataLoader,
    sparsity_ratio: float,
    num_calibration_rounds: int,
    device: torch.device,
    num_batches_per_round: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Calibrate gradient masks using iterative sensitivity-based pruning.
    
    Multi-round calibration process:
    1. Compute sensitivity scores (gradient magnitude)
    2. Identify least sensitive parameters (bottom sparsity_ratio percentile)
    3. Create binary masks (1 = update, 0 = freeze)
    4. Repeat for multiple rounds to refine the masks
    
    Args:
        model: PyTorch model to calibrate masks for
        dataloader: DataLoader for calibration data
        sparsity_ratio: Fraction of parameters to freeze (0.0 to 1.0)
                       e.g., 0.9 means freeze 90% of parameters
        num_calibration_rounds: Number of iterative calibration rounds
        device: Device to run computation on
        num_batches_per_round: Number of batches to use per round (None = all)
        
    Returns:
        Dictionary mapping parameter names to binary masks (1 = update, 0 = freeze)
    """
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"
    
    print(f"Starting mask calibration with {num_calibration_rounds} rounds...")
    print(f"Sparsity ratio: {sparsity_ratio:.2%} (freezing {sparsity_ratio:.2%} of parameters)")
    
    # Initialize masks (all ones = all parameters active initially)
    masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            masks[name] = torch.ones_like(param.data)
    
    # Iterative calibration
    for round_idx in range(num_calibration_rounds):
        print(f"\nCalibration round {round_idx + 1}/{num_calibration_rounds}")
        
        # Compute sensitivity scores for current model state
        sensitivity_scores = compute_sensitivity_scores(model, dataloader, device, num_batches_per_round)
        
        # Flatten all scores and masks to compute global threshold
        all_scores = []
        for name in sensitivity_scores:
            # Only consider currently active parameters (mask == 1)
            active_scores = sensitivity_scores[name][masks[name] == 1]
            all_scores.append(active_scores.flatten())
        
        if len(all_scores) == 0:
            print("Warning: No active parameters remaining!")
            break
            
        all_scores = torch.cat(all_scores)
        
        # Compute threshold for this round
        # We want to freeze the bottom sparsity_ratio% of parameters
        threshold = torch.quantile(all_scores, sparsity_ratio)
        
        print(f"  Sensitivity threshold: {threshold:.6f}")
        
        # Update masks: freeze parameters with sensitivity below threshold
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
    """
    Create an iterator of (mask, parameter) tuples for use with SparseSGDM.
    
    Args:
        model: PyTorch model
        masks: Dictionary of masks from calibrate_gradient_masks
        
    Returns:
        List of (mask, parameter) tuples
    """
    masked_params = []
    for name, param in model.named_parameters():
        if name in masks and param.requires_grad:
            masked_params.append((masks[name], param))
    return masked_params
