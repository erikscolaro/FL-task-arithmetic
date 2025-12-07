import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dino_backbone import Dino
from models.icarl_head import get_trained_icarl_classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DinoIcarlModel(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.dino_backbone = Dino
        self.icarl_head = get_trained_icarl_classifier(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.dino_backbone(x)
        # No gradients for pretrained icarl head
        with torch.no_grad():
            out = self.icarl_head(features)
        
        return out