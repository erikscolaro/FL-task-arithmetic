"""FL-task-arithmetic: A Flower / PyTorch app."""

from typing import Literal, cast, Optional, Dict
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import wandb
from fl_task_arithmetic.strategy import CustomFedAvg, get_evaluate_fn
from fl_task_arithmetic.task import CustomDino, load_server_test_data
from fl_task_arithmetic.sensitivity import calibrate_gradient_masks
from datetime import datetime
import os
from utilities.wandb_utils import load_model_from_wandb, save_model_to_wandb
from torch.utils.data import DataLoader

# ------------------------------
# ServerApp Initialization
# ------------------------------
app = ServerApp()


# ------------------------------
# Main Entry Point
# ------------------------------
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # ------------------------------
    # Read Configuration from Context
    # ------------------------------
    fraction_train: float = context.run_config["fraction-train"]  # type:ignore
    fraction_evaluate: float = context.run_config["fraction-evaluate"]  # type:ignore
    num_rounds: int = context.run_config["num-server-rounds"]  # type:ignore
    lr: float = context.run_config["lr"]  # type:ignore

    # ------------------------------
    # Read Wandb (Weights & Biases) Configuration
    # ------------------------------

    entity = str(context.run_config["entity"])
    project = str(context.run_config["project"])
    group = str(context.run_config["group"])
    notes = str(context.run_config["notes"])
    resume = cast(Literal["allow", "must", "never"], (context.run_config["resume"]))
    run_id = int(context.run_config["run_id"])

    print(
        f"Wandb config:\n"
        f"\tentity={entity}\n"
        f"\tproject={project}\n"
        f"\tgroup={group}\n"
        f"\tnotes={notes}\n"
        f"\tresume={resume}\n"
        f"\trun_id={run_id}"
    )

    # ------------------------------
    # Initialize Wandb API and Run
    # ------------------------------
    run_id_str = f"{project}-{group}-{run_id}-server"

    # Initialize Wandb run for logging
    run = wandb.init(
        entity=entity,
        project=project,
        group=group,
        name="server",
        id=run_id_str,
        notes=notes,
        resume=resume,
        mode="online",
    )

    wandb_api = wandb.Api()
    last_round = 0
    run_online = wandb_api.run(f"{entity}/{project}/{project}-{group}-{run_id}-server")

    # ------------------------------
    # Model Initialization and Recovery
    # ------------------------------
    global_model = CustomDino(num_classes=100)

    if resume == "never":
        # If not resuming, start from scratch and delete old artifacts
        print("Resume disabled. Starting from scratch and removing old artifacts.")
        for artifact in run_online.logged_artifacts():
            artifact.delete()
        last_round = 0
        start_backbone = cast(torch.nn.Module, torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=True))
        global_model = CustomDino(num_classes=100, backbone=start_backbone)

    else:
        # Try to load the last model checkpoint from Wandb
        try:
            history = run_online.history(keys=["round"], pandas=False)
            if len(history) > 0:
                last_round = history[-1]["round"]
                print(
                    f"Found previous run on cloud. Resuming from round {last_round}."
                )
                load_model_from_wandb(run=run, model=global_model)
                # updating context
                context.run_config["server-round"] = last_round
                context.run_config["num-server-rounds"] += last_round
                num_rounds += last_round
            else:
                print("No history found. Starting from scratch.")
                save_model_to_wandb(run=run, model=global_model)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch.")
            save_model_to_wandb(run=run, model=global_model)

    # ------------------------------
    # Calibrate Gradient Masks (if using sparse fine-tuning)
    # ------------------------------
    use_sparse = context.run_config.get("use-sparse-finetuning", False)
    masks: Optional[Dict[str, torch.Tensor]] = None
    
    if use_sparse:
        print("\n=== Server-side Mask Calibration ===")
        sparsity_ratio = context.run_config.get("sparsity-ratio", 0.0)
        num_calibration_rounds = context.run_config.get("num-calibration-rounds", 1)
        num_batches_calibration = context.run_config.get("num-batches-calibration", 10)
        
        print(f"Sparsity ratio: {sparsity_ratio}")
        print(f"Calibration rounds: {num_calibration_rounds}")
        print(f"Batches per round: {num_batches_calibration}")
        
        # Try to load masks from wandb if resuming
        masks_artifact_name = f"masks-{group}-{run_id}"
        if resume != "never" and last_round > 0:
            try:
                print("Attempting to load masks from wandb...")
                masks_artifact = run.use_artifact(f"{masks_artifact_name}:latest")
                masks_dir = masks_artifact.download()
                masks = torch.load(f"{masks_dir}/masks.pth")
                print(f"  Masks loaded from wandb artifact!")
                num_frozen = sum((m == 0).sum().item() for m in masks.values()) # type: ignore
                num_total = sum(m.numel() for m in masks.values()) # type: ignore
                print(f"  Loaded masks: {num_frozen}/{num_total} params frozen ({100*num_frozen/num_total:.1f}%)")
            except Exception as e:
                print(f"Could not load masks from wandb: {e}")
                print("Will calibrate new masks...")
                masks = None
        
        # Calibrate masks if not loaded from wandb
        if masks is None:
            calibration_dataset = load_server_test_data()
            if calibration_dataset is not None:
                calibration_loader = DataLoader(
                    dataset=calibration_dataset,  # type: ignore
                    batch_size=context.run_config["client-batch-size"],  # type: ignore
                    shuffle=True,
                )
                
                # IMPORTANT: Unfreeze backbone for calibration
                print("Unfreezing backbone for mask calibration...")
                global_model.unfreeze_backbone()
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                masks = calibrate_gradient_masks(
                    model=global_model,
                    dataloader=calibration_loader,
                    sparsity_ratio=sparsity_ratio, # type: ignore
                    num_calibration_rounds=num_calibration_rounds, # type: ignore
                    device=device,
                    num_batches_per_round=num_batches_calibration, # type: ignore
                )
                print(f"✓ Masks calibrated successfully on server!")
                
                # Save masks to wandb
                try:
                    print("Saving masks to wandb...")
                    masks_artifact = wandb.Artifact(
                        name=masks_artifact_name,
                        type="masks",
                        description=f"Gradient masks with sparsity ratio {sparsity_ratio}",
                        metadata={
                            "sparsity_ratio": sparsity_ratio,
                            "num_calibration_rounds": num_calibration_rounds,
                            "num_batches_calibration": num_batches_calibration,
                        }
                    )
                    masks_path = "masks.pth"
                    torch.save(masks, masks_path)
                    masks_artifact.add_file(masks_path)
                    run.log_artifact(masks_artifact)
                    os.remove(masks_path)  # Clean up local file
                    print(f"✓ Masks saved to wandb artifact: {masks_artifact_name}")
                except Exception as e:
                    print(f"Warning: Could not save masks to wandb: {e}")
            else:
                print("Warning: Could not load calibration data, sparse fine-tuning disabled")
                use_sparse = False

    # ------------------------------
    # Prepare Model State for Federated Learning
    # ------------------------------
    arrays = ArrayRecord(global_model.state_dict())
    print("correcly created the state dict for the global model")

    # ------------------------------
    # Start Federated Training
    # ------------------------------
    strategy = CustomFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        last_round=last_round,
        masks=masks  # Pass pre-calibrated masks
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds - last_round,
        evaluate_fn=get_evaluate_fn(
            run,
            global_model,
            context,
            last_round
        ),
    )

    run.finish()
