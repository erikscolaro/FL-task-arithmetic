
"""FL-task-arithmetic: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_task_arithmetic.task import CustomDino, load_data
from fl_task_arithmetic.task import test as test_fn
from fl_task_arithmetic.task import train as train_fn
from fl_task_arithmetic.task import train_sparse as train_sparse_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Determine if we should use sparse fine-tuning
    use_sparse = context.run_config.get("use-sparse-finetuning", False)
    mask_calibration_type = context.run_config.get("mask-calibration-type", 0)
    
    # Load the model and initialize it with the received weights
    model = CustomDino(num_classes=100)
    
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict()) # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions, context) # type: ignore

    # Call the appropriate training function
    if use_sparse:
        # Extract masks from server message if available
        masks = None
        if "masks" in msg.content:
            masks_dict = msg.content["masks"].to_torch_state_dict() # type: ignore
            masks = {name: tensor for name, tensor in masks_dict.items()}
            print(f"[Client {partition_id}] Received {len(masks)} masks from server")
        
        print(f"[Client {partition_id}] Using sparse fine-tuning")
        train_loss = train_sparse_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
            masks=masks,  # Pass pre-calibrated masks from server
        )
    else:
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            msg.content["config"]["lr"],
            device,
        )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset), # type: ignore
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = CustomDino(num_classes=100)
    
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict()) # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions, context) # type: ignore

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset), # type: ignore
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
