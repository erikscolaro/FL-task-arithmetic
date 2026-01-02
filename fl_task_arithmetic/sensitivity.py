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
from torch.func import functional_call, vmap, grad


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict

def compute_sensitivity_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    num_batches: Optional[int] = None,
    R: int = 1,
) -> Dict[str, torch.Tensor]:
    
    # --- MICRO-BATCH CONFIGURATION ---
    # Try 4. If you get OOM, go down to 2. If it holds, go up to 8.
    MICRO_BATCH_SIZE = 4
    # --------------------------------

    model.train()
    model.to(device)
    params_all = {k: v for k, v in model.named_parameters() if v.requires_grad}
    params_diff = {}
    params_fixed = {}
    
    for k, v in params_all.items():
        if masks is not None and k in masks and (masks[k] == 0).all():
            params_fixed[k] = v
        else:
            params_diff[k] = v
    #Add all non-trainable params to fixed
    params_fixed.update({k: v for k, v in model.named_parameters() if not v.requires_grad})
    buffers = {k: v for k, v in model.named_buffers()}
    
    sensitivity_scores = {k: torch.zeros_like(v, device='cpu') for k, v in params_diff.items()}

    # Stateless function for vmap
    def compute_loss_stateless(p_diff, p_fixed, bufs, img, lbl):
        all_params = {**p_diff, **p_fixed}
        img = img.unsqueeze(0)
        lbl = lbl.unsqueeze(0)
        out = torch.func.functional_call(model, (all_params, bufs), img)
        return torch.nn.functional.cross_entropy(out, lbl)

    # vmap vectorizes the computation
    compute_per_sample_grads = torch.vmap(
        torch.func.grad(compute_loss_stateless, argnums=0),
        in_dims=(None, None, None, 0, 0)
    )

    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and num_batches >= 0 and batch_idx >= num_batches:
            break

        if isinstance(batch, dict):
            images_full = batch["img"].to(device)
            labels_full = batch["fine_label"].to(device)
        else:
            images_full, labels_full = batch
            images_full, labels_full = images_full.to(device), labels_full.to(device)

        # TaLoS loop (R sampling)
        for _ in range(R):
            
            # --- MICRO-BATCHING LOOP ---
            # We split the large batch (e.g., 32) into micro-batches (e.g., 4)
            total_samples = images_full.shape[0]
            
            for i in range(0, total_samples, MICRO_BATCH_SIZE):
                # Select the slice (chunk)
                img_chunk = images_full[i : i + MICRO_BATCH_SIZE]
                lbl_chunk = labels_full[i : i + MICRO_BATCH_SIZE]
                
                # Compute gradients with vmap ONLY on this chunk
                # Here we only use memory for 4 images, not 32!
                chunk_grads = compute_per_sample_grads(
                    params_diff, params_fixed, buffers, img_chunk, lbl_chunk
                )

                # Accumulate squared gradients
                for name, g_sample in chunk_grads.items():
                    # g_sample has shape [Micro_Batch, Params]
                    sensitivity_scores[name] += g_sample.pow(2).sum(dim=0).detach().cpu()
                
                # Clean up intermediate references to free VRAM immediately
                del chunk_grads, img_chunk, lbl_chunk
            
            # --- END MICRO-BATCHING ---

        batch_count += 1
        
        # Clear cache after each full batch to avoid fragmentation
        torch.cuda.empty_cache()

        if batch_count % 10 == 0:
            print(f"  Computed sensitivity for {batch_count} batches...")
    
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
            
            # --- FIX ---
            if name not in sensitivity_scores:
                # Parameter was fully frozen in previous rounds and excluded from computation.
                # It remains frozen (mask is already all 0s).
                frozen_params += masks[name].numel()
                continue
            # --- FIX ---
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


def calibrate_gradient_masks_most_sensitive(
    model: nn.Module,
    dataloader: DataLoader,
    sparsity_ratio: float,
    num_calibration_rounds: int,
    device: torch.device,
    num_batches_per_round: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"
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
            

        k = max(1, min(k, total_params))
        
        threshold, _ = torch.kthvalue(all_scores, k)
        print(f"  Threshold (k={k}/{total_params}): {threshold:.6f}")

        # Parameters with score >= threshold are frozen
        # Parameters with score < threshold are kept
        frozen_params = 0
        for name in masks:
            
            # --- FIX ---
            if name not in sensitivity_scores:
                # Parameter was fully frozen in previous rounds and excluded from computation.
                # It remains frozen (mask is already all 0s).
                frozen_params += masks[name].numel()
                continue
            # --- FIX ---
            score = sensitivity_scores[name]
            # THIS IS WHAT'S 
            new_mask = torch.where(score < threshold, 
                                   torch.ones_like(score), 
                                   torch.zeros_like(score))
            masks[name] = new_mask
            frozen_params += (new_mask == 0).sum().item()

        actual_sparsity = frozen_params / total_params if total_params > 0 else 0
        print(f"  Result: {frozen_params}/{total_params} frozen ({actual_sparsity:.2%})")

    print("\nMask most sensitive calibration complete!")
    final_frozen = sum((m == 0).sum().item() for m in masks.values())
    final_total = sum(m.numel() for m in masks.values())
    print(f"Final: {final_frozen}/{final_total} parameters frozen ({final_frozen/final_total:.2%})")
    if final_frozen == 0:
        print("Warning: no parameters were frozen; check gradients or data if sparsity was expected.")
    return masks


def calibrate_gradient_masks_with_randomness(
    model: nn.Module,
    sparsity_ratio: float
) -> Dict[str, torch.Tensor]:
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"
    keep_ratio = 1.0 - sparsity_ratio
    print(f"Final sparsity: {sparsity_ratio:.2%} frozen, {keep_ratio:.2%} active")

    # Initialize all masks to 1 (all parameters active)
    masks: Dict[str, torch.Tensor] = {}
    total_params = 0
    param_infos = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_infos.append((name, param))
            total_params += param.numel()

    num_zero = int(total_params * sparsity_ratio)
    global_mask = torch.ones(total_params, device="cpu")
    offset = 0
    
    if num_zero > 0:
        zero_indices = torch.randperm(total_params)[:num_zero]
        global_mask[zero_indices] = 0.0
    
    for name, param in param_infos:
        numel = param.numel()
        masks[name] = global_mask[offset : offset + numel].view_as(param).clone()
        offset += numel


    print("\nMask most sensitive calibration complete!")
    final_frozen = sum((m == 0).sum().item() for m in masks.values())
    final_total = sum(m.numel() for m in masks.values())
    print(f"Final: {final_frozen}/{final_total} parameters frozen ({final_frozen/final_total:.2%})")
    if final_frozen == 0:
        print("Warning: no parameters were frozen; check gradients or data if sparsity was expected.")
    return masks



def calibrate_gradient_masks_with_lowest_magnitudes(
    model: nn.Module,
    sparsity_ratio: float,
) -> Dict[str, torch.Tensor]:
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"
    keep_ratio = 1.0 - sparsity_ratio
    print(f"Final sparsity: {sparsity_ratio:.2%} frozen, {keep_ratio:.2%} active")

    # Initialize all masks to 1 (all parameters active)
    masks: Dict[str, torch.Tensor] = {}
    total_params = 0
    param_infos = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_infos.append((name, param))
            total_params += param.numel()

    num_zero = int(total_params * sparsity_ratio)
    
    all_magnitudes = torch.cat([
        param.detach().abs().reshape(-1).cpu()
        for _, param in param_infos
    ])

    if num_zero > 0:
        threshold, _ = torch.kthvalue(all_magnitudes, num_zero)
    else:
        threshold = -float("inf")

    # ones for params with highest magnitude
    masks: Dict[str, torch.Tensor] = {}
    for name, param in param_infos:
        mask = (param.detach().abs() > threshold).float()
        masks[name] = mask


    print("\nMask most sensitive calibration complete!")
    final_frozen = sum((m == 0).sum().item() for m in masks.values())
    final_total = sum(m.numel() for m in masks.values())
    print(f"Final: {final_frozen}/{final_total} parameters frozen ({final_frozen/final_total:.2%})")
    if final_frozen == 0:
        print("Warning: no parameters were frozen; check gradients or data if sparsity was expected.")
    return masks


def calibrate_gradient_masks_with_highest_magnitudes(
    model: nn.Module,
    sparsity_ratio: float,
) -> Dict[str, torch.Tensor]:
    assert 0.0 <= sparsity_ratio < 1.0, "sparsity_ratio must be in [0, 1)"
    keep_ratio = 1.0 - sparsity_ratio
    print(f"Final sparsity: {sparsity_ratio:.2%} frozen, {keep_ratio:.2%} active")

    # Initialize all masks to 1 (all parameters active)
    masks: Dict[str, torch.Tensor] = {}
    total_params = 0
    param_infos = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_infos.append((name, param))
            total_params += param.numel()

    num_zero = int(total_params * (1 - sparsity_ratio))

    all_magnitudes = torch.cat([
        param.detach().abs().reshape(-1).cpu()
        for _, param in param_infos
    ])

    if num_zero > 0:
        threshold, _ = torch.kthvalue(all_magnitudes, num_zero)
    else:
        threshold = -float("inf")

    # ones for params with lowest magnitude

    masks: Dict[str, torch.Tensor] = {}
    for name, param in param_infos:
        mask = (param.detach().abs() < threshold).float()
        masks[name] = mask


    print("\nMask most sensitive calibration complete!")
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
