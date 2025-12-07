from typing import Optional
import torch.nn as nn
from typing import cast
import torch


class Dino(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone: nn.Module = cast(nn.Module, torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16", pretrained=True
        ))
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor):
        return self.backbone(x)