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


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

# DINO transforms (for use with CustomDino model)
dino_transforms = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        # DINO was trained with ImageNet statistics, but we use CIFAR-100 for fine-tuning
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    # Use dino_transforms for DINO model, pytorch_transforms for simple CNN
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
            class_assignment_mode="random",  # deterministic to comply with eventual checkpoints
            shuffle=True, # Randomize the order of samples after the partition.
            seed=42, #TODO: should we modify the default value?
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
    if fds is not None:
        return fds.load_split("test")
    else:
        return None

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
    sparsity_ratio: float = 0.0,
    num_calibration_rounds: int = 1,
    num_batches_calibration: Optional[int] = 10,
):
    """
    Train the model using sparse fine-tuning with gradient masking.
    
    Two-step sparse fine-tuning process:
    1. Calibrate gradient masks by identifying least sensitive parameters (low gradient magnitude)
    2. Fine-tune using SparseSGDM with the calibrated masks
    
    Args:
        net: Model to train (should be CustomDino with frozen classifier)
        trainloader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        sparsity_ratio: Fraction of parameters to freeze (0.0 to 1.0)
        num_calibration_rounds: Number of mask calibration rounds
        num_batches_calibration: Batches to use per calibration round
        
    Returns:
        Average training loss
    """
    from fl_task_arithmetic.sensitivity import calibrate_gradient_masks, get_masked_parameters
    from fl_task_arithmetic.sparseSGDM import SparseSGDM
    
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Step 1: Calibrate gradient masks if sparsity > 0
    if sparsity_ratio > 0.0:
        print(f"\n=== Step 1: Calibrating Gradient Masks ===")
        masks = calibrate_gradient_masks(
            model=net,
            dataloader=trainloader,
            sparsity_ratio=sparsity_ratio,
            num_calibration_rounds=num_calibration_rounds,
            device=device,
            num_batches_per_round=num_batches_calibration,
        )
        
        # Prepare masked parameters for SparseSGDM
        masked_params = get_masked_parameters(net, masks)
        
        # Use SparseSGDM with masks
        optimizer = SparseSGDM(
            net.parameters(),
            masks=masked_params,
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        print(f"\n=== Step 2: Sparse Fine-tuning with SparseSGDM ===")
    else:
        # No sparsity, use standard Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        print(f"\n=== Standard Fine-tuning (no sparsity) ===")
    
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
