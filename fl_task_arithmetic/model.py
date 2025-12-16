from typing import Optional, cast
from torch import nn
import torch
from pathlib import Path
import wandb
import os


def _classifier_cache_dirs():
    """Return candidate cache directories for the classifier weights."""
    here = Path(__file__).resolve().parent
    return [
        here / "trained",
        Path(__file__).resolve().parent.parent / "utilities" / "trained",
    ]


def load_icarl_classifier_from_wandb(
    project: str = "fl-task-arithmetic",
    entity: str = "aml-fl-project",
    artifact_name: str = "nearest_centroid_classifier",
    artifact_version: str = "latest",
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """
    Load the iCaRL nearest centroid classifier from Weights & Biases.
    Falls back to cached/local copies if download is unavailable.
    """
    # Resolve cache directory
    if cache_dir is None:
        cache_dirs = _classifier_cache_dirs()
    else:
        cache_dirs = [Path(cache_dir)]

    # Try cache locations
    for cdir in cache_dirs:
        cdir.mkdir(parents=True, exist_ok=True)
        local_path = cdir / "nearest_centroid_classifier.pth"
        if local_path.exists():
            print(f"Loading iCaRL classifier from local cache: {local_path}")
            classifier = nn.Linear(384, 100)
            classifier.load_state_dict(torch.load(local_path, map_location="cpu"))
            for param in classifier.parameters():
                param.requires_grad = False
            return classifier

    # Otherwise, download from wandb into first cache dir
    target_dir = cache_dirs[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / "nearest_centroid_classifier.pth"
    print(f"Downloading iCaRL classifier from W&B: {entity}/{project}/{artifact_name}:{artifact_version}")
    try:
        api = wandb.Api()
        artifact_path = f"{entity}/{project}/{artifact_name}:{artifact_version}"
        artifact = api.artifact(artifact_path)
        artifact_dir = artifact.download(root=str(target_dir))

        pth_files = list(Path(artifact_dir).glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth file found in artifact {artifact_path}")

        classifier = nn.Linear(384, 100)
        classifier.load_state_dict(torch.load(pth_files[0], map_location="cpu"))

        import shutil
        shutil.copy(pth_files[0], local_path)
        print(f"Cached classifier to: {local_path}")

        for param in classifier.parameters():
            param.requires_grad = False
        return classifier

    except Exception as e:
        print(f"Warning: Failed to download from W&B: {e}")
        print("Attempting to load from local trained/ directory...")
        # Try fallbacks explicitly
        for cdir in cache_dirs:
            fallback = cdir / "nearest_centroid_classifier.pth"
            if fallback.exists():
                classifier = nn.Linear(384, 100)
                classifier.load_state_dict(torch.load(fallback, map_location="cpu"))
                for param in classifier.parameters():
                    param.requires_grad = False
                return classifier
        raise FileNotFoundError("Could not load classifier from W&B or local cache paths")


class CustomDino(nn.Module):
    """DINO ViT-S/16 with frozen iCaRL nearest centroid classifier."""

    def __init__(
        self,
        num_classes: int = 100,
        backbone: Optional[nn.Module] = None,
        load_classifier_from_wandb: bool = True,
        classifier_state_dict: Optional[dict] = None,
    ):
        super().__init__()

        # Load DINO backbone
        if backbone is None:
            backbone = cast(
                nn.Module,
                torch.hub.load(
                    "facebookresearch/dino:main", "dino_vits16", pretrained=True
                ),
            )
        self.backbone: nn.Module = backbone
        self.num_classes = num_classes

        # Load iCaRL classifier
        if classifier_state_dict is not None:
            self.classifier = nn.Linear(384, num_classes)
            self.classifier.load_state_dict(classifier_state_dict)
        elif load_classifier_from_wandb:
            self.classifier = load_icarl_classifier_from_wandb()
        else:
            # fallback to first cache dir
            local_path = _classifier_cache_dirs()[0] / "nearest_centroid_classifier.pth"
            self.classifier = nn.Linear(384, num_classes)
            self.classifier.load_state_dict(torch.load(local_path, map_location="cpu"))

        for param in self.classifier.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
