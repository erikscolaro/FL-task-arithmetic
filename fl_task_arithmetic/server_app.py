"""FL-task-arithmetic: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_task_arithmetic.task import Net
from datetime import datetime
import os

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"] # type: ignore[call-operator]
    num_rounds: int = context.run_config["num-server-rounds"] # type: ignore[call-operator]
    lr: float = context.run_config["lr"] # type: ignore[call-operator]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

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
