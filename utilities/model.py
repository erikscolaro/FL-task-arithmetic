from typing import Optional, cast
from torch import nn
import torch
from pathlib import Path

class CustomDino(nn.Module):
    def __init__(self, num_classes: int = 100, backbone: Optional[nn.Module] = None):
        super().__init__()
        if backbone is None:
            backbone = cast(
                nn.Module,
                torch.hub.load(
                    "facebookresearch/dino:main", "dino_vits16", pretrained=True
                ),
            )
        self.backbone: nn.Module = backbone
        self.num_classes = num_classes
        self.classifier = nn.Linear(384, num_classes)
        self.classifier.load_state_dict(torch.load(Path(__file__).resolve().parent / "./trained/nearest_centroid_classifier.pth"))
        for param in self.classifier.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    
    