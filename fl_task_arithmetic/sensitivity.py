"""
Sensitivity scoring for gradient mask calibration in sparse fine-tuning.
Uses squared gradient magnitude (Fisher Information approximation) to identify 
least-sensitive parameters, following the TaLoS approach.

Reference: TaLoS - Task-Localized Sparse Fine-tuning (ICLR 2025)
https://github.com/iurada/talos-task-arithmetic
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from torch.utils.data import DataLoader


def compute_sensitivity_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    num_batches: Optional[int] = None,
    R: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Compute sensitivity scores based on squared gradient magnitude.
    
    Following TaLoS, we use grad^2 as the sensitivity metric (Fisher Information
    approximation). This measures how much the loss changes when perturbing each
    parameter.
    
    Args:
        model: The model to compute sensitivity for
        dataloader: DataLoader for calibration data
        device: Device to run on
        masks: Optional existing masks (only compute scores for active params)
        num_batches: Limit number of batches for efficiency
        R: Number of samples per input (TaLoS uses stochastic sampling)
    """
    model.train()
    model.to(device)

    sensitivity_scores: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            sensitivity_scores[name] = torch.zeros_like(param.data, device='cpu')

    criterion = nn.CrossEntropyLoss()
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and num_batches >= 0 and batch_idx >= num_batches:
            break

        if isinstance(batch, dict):
            images = batch["img"].to(device)
            labels = batch["fine_label"].to(device)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

        # TaLoS approach: sample R times from the model's output distribution
        for _ in range(R):
            model.zero_grad()
            outputs = model(images)
            
            # Standard cross-entropy with ground truth labels
            loss = criterion(outputs, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # TaLoS uses grad^2 (squared gradient) as sensitivity metric
                    sensitivity_scores[name] += param.grad.data.pow(2).detach().cpu()

        batch_count += 1

    # No need to average - we want cumulative sensitivity
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
    Iteratively calibrate binary masks using TaLoS-style progressive sparsity.
    
    TaLoS uses iterative pruning with exponentially increasing sparsity:
    At each round r, target_sparsity = final_sparsity^((r+1)/num_rounds)
    
    This allows the model to gradually identify which parameters are truly
    important, as the gradient landscape changes after each pruning round.
    
    Args:
        model: Model to calibrate masks for
        dataloader: DataLoader for calibration data
        sparsity_ratio: Final target sparsity (fraction of params to freeze, e.g., 0.9 = 90% frozen)
        num_calibration_rounds: Number of iterative pruning rounds
        device: Device to run on
        num_batches_per_round: Limit batches per round for efficiency
    
    Returns:
        Dictionary mapping parameter names to binary masks (1 = active, 0 = frozen)
    """
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"

    # In TaLoS, sparsity is defined as "fraction to KEEP", we define it as "fraction to FREEZE"
    # So we need to convert: keep_ratio = 1 - sparsity_ratio
    keep_ratio = 1.0 - sparsity_ratio
    
    print(f"Starting TaLoS-style mask calibration with {num_calibration_rounds} rounds...")
    print(f"Final sparsity: {sparsity_ratio:.2%} frozen, {keep_ratio:.2%} active")

    # Initialize all masks to 1 (all parameters active)
    masks: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            masks[name] = torch.ones_like(param.data, device='cpu')

    for round_idx in range(num_calibration_rounds):
        # TaLoS progressive sparsity: sparse = keep_ratio^((round+1)/num_rounds)
        # This gradually decreases the number of active parameters
        round_keep_ratio = keep_ratio ** ((round_idx + 1) / num_calibration_rounds)
        round_sparsity = 1.0 - round_keep_ratio
        
        print(f"\nCalibration round {round_idx + 1}/{num_calibration_rounds}")
        print(f"  Target: keep {round_keep_ratio:.2%}, freeze {round_sparsity:.2%}")

        # Compute sensitivity scores (grad^2)
        sensitivity_scores = compute_sensitivity_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            masks=masks,
            num_batches=num_batches_per_round,
        )

        # Collect ALL scores globally (TaLoS style)
        all_scores = torch.cat([sensitivity_scores[name].flatten() for name in sensitivity_scores])
        total_params = all_scores.numel()

        if total_params == 0:
            print("  Warning: No parameters found!")
            break

        # Debug: distribution of scores
        num_zeros = (all_scores == 0).sum().item()
        num_nonzeros = (all_scores > 0).sum().item()
        
        # If all gradients are zero, skip freezing to avoid masking everything
        if num_nonzeros == 0:
            print("  Warning: all gradients are zero; skipping pruning for this round.")
            continue
            
        print(f"  Non-zero scores: min={all_scores[all_scores > 0].min():.2e}, "
              f"max={all_scores.max():.2e}, mean={all_scores[all_scores > 0].mean():.2e}")

        # TaLoS: k = int((1.0 - sparsity) * global_scores.numel())
        # where sparsity = fraction to KEEP
        # We have round_keep_ratio = fraction to keep
        k = int(round_keep_ratio * total_params)
        
        if k < 1:
            print("  Warning: k < 1, skipping round")
            continue
            
        # TaLoS uses kthvalue to find threshold
        # k-th smallest value = threshold below which we freeze
        # But we want top-k largest, so we use (total - k + 1)-th smallest
        k_for_kth = total_params - k + 1
        k_for_kth = max(1, min(k_for_kth, total_params))
        
        threshold, _ = torch.kthvalue(all_scores, k_for_kth)
        print(f"  Threshold (k={k_for_kth}/{total_params}): {threshold:.6f}")

        # TaLoS: mask = where(score <= threshold, 0, 1)
        # Parameters with score <= threshold are frozen
        # Parameters with score > threshold are kept
        frozen_params = 0
        for name in masks:
            score = sensitivity_scores[name]
            # score > threshold -> keep (mask=1), score <= threshold -> freeze (mask=0)
            new_mask = torch.where(score > threshold, 
                                   torch.ones_like(score), 
                                   torch.zeros_like(score))
            masks[name] = new_mask
            frozen_params += (new_mask == 0).sum().item()

        actual_sparsity = frozen_params / total_params if total_params > 0 else 0
        print(f"  Result: {frozen_params}/{total_params} frozen ({actual_sparsity:.2%})")

    print("\nMask calibration complete!")
    final_frozen = sum((m == 0).sum().item() for m in masks.values())
    final_total = sum(m.numel() for m in masks.values())
    print(f"Final: {final_frozen}/{final_total} parameters frozen ({final_frozen/final_total:.2%})")
    if final_frozen == 0:
        print("Warning: no parameters were frozen; check gradients or data if sparsity was expected.")
    
    return masks


def get_masked_parameters(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """Create an iterator of (mask, parameter) tuples for SparseSGDM."""
    masked_params = []
    for name, param in model.named_parameters():
        if name in masks and param.requires_grad:
            masked_params.append((masks[name], param))
    return masked_params
