"""FL-task-arithmetic: A Flower / PyTorch app."""

from typing import Optional, cast, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.preprocessor import Divider
from flwr.app import Context
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop

from fl_task_arithmetic.model import CustomDino


fds = None  # Cache FederatedDataset

# DINO transforms (224x224 images for CustomDino model)
dino_transforms = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [dino_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, context: Context):
    """
    Load a partition of the CIFAR100 dataset.

    CIFAR100 initially has the following splits:
        {"train": 50k images, "test": 10k images}.

    We further split the training set according to the value of 
    'val-ratio-of-train' defined in pyproject.toml. For example, if 
    val-ratio-of-train = 0.1, the resulting splits are:
        {"train": 45k images, "val": 5k images, "test": 10k images}.

    The training split is then partitioned among clients using PathologicalPartitioner,
    that takes from the toml a parameter called 'num-classes-per-partition'. 
    If this number is equal to the number of classes in the dataset, the partitioning is IID.
    Otherwise, the partitions are non-IID and in the extreme case (=1) each partition has samples
    from only one class (extreme non-IID)

    Each client receives a partition. For instance, with 10 clients, 
    each receives 45k / 10 = 4.5k images. From the client's point of 
    view, this is a local dataset that has not been split yet.

    We further split the client's partition using 
    'client-test-ratio-of-partition', which defines the percentage 
    of images used as the test set. For example, if this value is 0.2, 
    the client's partition is split as follows:
        {"train": 3.4k images, "test": 900 images}.
    """

    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="fine_label",
            num_classes_per_partition=context.run_config["num-classes-per-partition"], # type: ignore
            class_assignment_mode="deterministic",
            shuffle=True, # Randomize the order of samples after the partition.
            seed=42,
        )
        # train split -> train + validation split
        preprocessor = Divider(
            divide_config={
                "train": 1.0 - context.run_config["val-ratio-of-train"],  # type: ignore[call-operator]
                "valid": context.run_config["val-ratio-of-train"],  # type: ignore[call-operator]
            },
            divide_split="train",
            drop_remaining_splits=False,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar100",
            preprocessor=preprocessor,
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide on each node the partition received into train and test as specified in context
    partition_splits = partition.train_test_split(test_size=context.run_config["client-test-ratio-of-partition"], seed=42)  # type: ignore[call-operator]
    # Construct dataloaders
    partition_splits = partition_splits.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_splits["train"],  # type: ignore[call-operator]
        batch_size=context.run_config["client-batch-size"],  # type: ignore[call-operator]
        shuffle=True,
    )
    testloader = DataLoader(
        partition_splits["test"],  # type: ignore[call-operator]
        batch_size=context.run_config["client-batch-size"],  # type: ignore[call-operator]
    )
    return trainloader, testloader


def load_server_test_data():
    """
    Load the centralized test dataset for server-side evaluation.
    
    This function loads the test split directly from HuggingFace datasets,
    independent of the federated partitioning used by clients. This is 
    necessary because the server runs in a separate process and doesn't
    have access to the client's FederatedDataset instance.
    """
    from datasets import load_dataset
    
    # Load CIFAR-100 test split directly (not partitioned)
    dataset = load_dataset("uoft-cs/cifar100", split="test")
    
    # Apply DINO transforms using set_transform (applies on-the-fly)
    def apply_transforms(batch):
        batch["img"] = [dino_transforms(img) for img in batch["img"]]
        return batch
    
    dataset.set_transform(apply_transforms)  # type: ignore[union-attr]
    return dataset

def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["fine_label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def train_sparse(
    net,
    trainloader,
    epochs,
    lr,
    device,
    masks: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Train the model using sparse fine-tuning with pre-calibrated gradient masks.
    
    The masks should be pre-calibrated on the server before federated training.
    This function uses SparseSGDM with the provided masks to perform sparse fine-tuning.
    
    Args:
        net: Model to train (should be CustomDino with frozen classifier)
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        masks: Pre-calibrated gradient masks from server (dict mapping param names to masks)
        
    Returns:
        Average training loss
    """
    from fl_task_arithmetic.sensitivity import get_masked_parameters
    from fl_task_arithmetic.sparseSGDM import SparseSGDM
    
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Use pre-calibrated masks if provided
    if masks is not None and len(masks) > 0:
        print(f"\n=== Sparse Fine-tuning with Pre-calibrated Masks ===")
        
        # Move masks to CPU if needed (SparseSGDM expects CPU masks)
        masks_cpu = {name: mask.cpu() if mask.device != torch.device('cpu') else mask 
                     for name, mask in masks.items()}
        
        # Prepare masked parameters for SparseSGDM
        masked_params = get_masked_parameters(net, masks_cpu)
        
        # Use SparseSGDM with masks
        optimizer = SparseSGDM(
            net.parameters(),
            masks=masked_params,
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        
        num_frozen = sum((m == 0).sum().item() for m in masks_cpu.values())
        num_total = sum(m.numel() for m in masks_cpu.values())
        print(f"Using masks: {num_frozen}/{num_total} params frozen ({100*num_frozen/num_total:.1f}%)")
    else:
        # No masks provided, use standard Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print(f"\n=== Standard Fine-tuning (no masks provided) ===")
    
    # Step 2: Fine-tune with masked gradients
    net.train()
    running_loss = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["fine_label"].to(device)
            
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(trainloader)
        running_loss += avg_epoch_loss
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
    
    avg_trainloss = running_loss / epochs
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["fine_label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
