"""FL-task-arithmetic: A Flower / PyTorch app."""

from typing import cast
import torch
import torch.nn as nn
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_task_arithmetic.task import CustomDino, Net
from datetime import datetime
import os

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]  # type: ignore[call-operator]
    num_rounds: int = context.run_config["num-server-rounds"]  # type: ignore[call-operator]
    lr: float = context.run_config["lr"]  # type: ignore[call-operator]

    # Load global model
    start_backbone = cast(nn.Module, torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=True))
    print("loaded predefined model")
    global_model = CustomDino(num_classes=100, backbone=start_backbone)
    print("Corretly instantiated the global initial model ")
    arrays = ArrayRecord(global_model.state_dict())
    print("correcly created the state dict for the global model")

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(state_dict, f"results/{timestamp}-final_model.pt")
