"""FL-task-arithmetic: A Flower / PyTorch app."""

from typing import Optional, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.preprocessor import Divider
from flwr.app import Context
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop

class CustomDino(nn.Module):
    def __init__(self, num_classes: int = 100, backbone: Optional[nn.Module] = None):
        super().__init__()
        if backbone is None:
            # Carica DINO senza pretrained e rimuove la head
            backbone = cast(nn.Module, torch.hub.load(
                "facebookresearch/dino:main", "dino_vits16", pretrained=True
            ))
        self.backbone: nn.Module = backbone
        self.classifier = nn.Linear(384, num_classes)  # 384 = output CLS token DINO ViT-S/16

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)        # [batch, 384]
        logits = self.classifier(features) # [batch, num_classes]
        return logits #, features



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

# parameters for normalization are avaiable online and are standard for the cifar100, no need to recompute them.
pytorch_transforms = Compose([ToTensor(), Normalize( (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

dino_transforms = Compose(
    [
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        # here, mean and std dev of cifar 100 not dino, because training phase
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]] #TODO: modifiy with dino transfomrs when switching to to dino
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
