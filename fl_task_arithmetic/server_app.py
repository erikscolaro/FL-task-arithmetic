"""FL-task-arithmetic: A Flower / PyTorch app."""

from typing import Literal, cast
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import wandb
from fl_task_arithmetic.strategy import CustomFedAvg, get_evaluate_fn
from fl_task_arithmetic.task import CustomDino
from datetime import datetime
import os
from utilities.wandb_utils import load_model_from_wandb, save_model_to_wandb

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
            last_round = history[-1]["round"]
            print(
                f"I find a previous run on the cloud. I'll resume from round {last_round}."
            )
            load_model_from_wandb(run=run, model=global_model)
            # updating context
            context.run_config["server-round"] = last_round
            context.run_config["num-server-rounds"] += last_round
            num_rounds += last_round
        except Exception:
            print("Starting from scratch.")
            # Optionally, initialize with a pretrained backbone
            # start_backbone = cast(nn.Module, torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=True))
            # global_model = CustomDino(num_classes=100, backbone=start_backbone)
            save_model_to_wandb(run=run, model=global_model)

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
        last_round=last_round
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds - last_round,
        evaluate_fn=get_evaluate_fn(
            run,
            global_model,
            context
        ),
    )

    run.finish()
