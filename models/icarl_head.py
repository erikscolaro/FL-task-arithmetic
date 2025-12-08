import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dino_backbone import Dino
import wandb
from utilities.wandb_utils import load_checkpoint_from_wandb


RUN_ID = "run-1-icarl_cifar100"
ENTITY = "aml-fl-project"
PROJECT = "fl-task-arithmetic"
GROUP = "icarl-cifar100"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_EXEMPLARS_VECTORS = 1000


class IcarlHead(nn.Module):
    def __init__(self, feature_dim: int = 384, num_classes: int = 100):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor):
        return self.classifier(features)

class IcarlModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Dino()
        self.classifier = IcarlHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class Icarl(nn.Module):
    def __init__(
            self,
            feature_dim: int = 384,
            num_classes: int = 100,
            memory_size= TOTAL_EXEMPLARS_VECTORS,
            device='cuda'
        ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.is_old_model_usable = False

        self.model = IcarlModel(
            num_classes=num_classes,
        ).to(self.device)
        self.old_model = IcarlModel(
            num_classes=num_classes,
        ).to(self.device)

        self.exemplar_sets = []
        self.exemplar_means = []


    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def reduce_exemplar_sets(self, m):
        """
        Step 2: Shrink stored exemplars to fit memory budget.
        m = memory_size / num_classes_seen_so_far
        """
        print(f"Reducing exemplars to {m} per class...")
        for y in range(len(self.exemplar_sets)):
            self.exemplar_sets[y] = self.exemplar_sets[y][:m]

    def construct_exemplar_sets(self, images, m, transform, class_id):
        """
        Step 3: Select new exemplars using Herding (nearest to mean).
        """
        print(f"Constructing {m} exemplars vectors per class number {class_id}")
        self.model.eval()

        # Compute mean of the class
        with torch.no_grad():
            # Extract features
            # Note: We need a loader to process 'images' (which is a list/tensor of raw images)
            # For efficiency in this script, we assume 'images' fits in VRAM or we batch it.
            # Simplified:
            img_tensor = torch.stack(images).to(self.device)
            _, features = self.model(img_tensor)
            features = F.normalize(features, p=2, dim=1)
            class_mean = torch.mean(features, dim=0)

            # Herding Selection
            exemplar_set = []
            exemplar_features = []

            # We assume features are [N, D]
            # We iterate m times to pick m samples
            for k in range(m):
                S = torch.sum(torch.stack(exemplar_features), dim=0) if len(exemplar_features) > 0 else torch.zeros(self.feature_dim).to(self.device)
                # Objective: minimize || class_mean - (S + phi(x)) / k   ||
                phi = features # [N, D]
                mu = class_mean # [D]
                # Distance for all candidates
                dists = torch.norm(mu - ((S + phi)/k), dim=1)
                # Pick best that isn't already chosen (simple way: set dist to inf)
                # In strict implementation, we remove the index.
                best_idx = torch.argmin(dists).item()
                exemplar_set.append(images[best_idx])
                exemplar_features.append(features[best_idx])
                # Mask this index so it's not picked again
                features[best_idx] = features[best_idx] + 10000 # Hacky mask

            self.exemplar_sets.append(exemplar_set)

    def classify_nme(self, x):
        """
        Step 4: Classification using Nearest Mean of Exemplars.
        Strict Implementation of Algorithm 1 & Eq. 2
        """
        self.model.eval()
        with torch.no_grad():
            # 1. Get features of the image to classify
            _, query_features = self.model(x.to(self.device))
            # Normalize query features (Section 2.1)
            query_features = F.normalize(query_features, p=2, dim=1)

            # 2. Compute Prototypes (Means of Exemplars)
            means = []
            for y in range(len(self.exemplar_sets)):
                # Get all exemplars for class y
                ex_imgs = torch.stack(self.exemplar_sets[y]).to(self.device)
                # Extract features for exemplars
                _, ex_feats = self.model(ex_imgs)
                # Normalize exemplar features BEFORE averaging (Section 2.1)
                ex_feats = F.normalize(ex_feats, p=2, dim=1)
                # Compute the mean
                class_mean = torch.mean(ex_feats, dim=0)
                # Re-normalize the mean vector itself (Section 2.1: "averages are also re-normalized")
                class_mean = F.normalize(class_mean.unsqueeze(0), p=2, dim=1).squeeze(0)
                means.append(class_mean)

            if len(means) == 0: return torch.zeros(x.size(0))
            means = torch.stack(means) # [Num_Classes_Seen, Feature_Dim]
            # 3. Find Nearest Prototype (Algorithm 1)
            # "y* = argmin || phi(x) - mu_y ||"
            dists = torch.cdist(query_features, means) # [Batch, Num_Classes]
            preds = torch.argmin(dists, dim=1)

        return preds



def get_trained_icarl_classifier(device=DEVICE) -> nn.Module | None:
    try:
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            group=GROUP,
            name="iCaRL_CIFAR100",
            id=RUN_ID,
            resume="allow",
            mode="online",
        )

        icarl = Icarl(
            num_classes=100,
            memory_size=TOTAL_EXEMPLARS_VECTORS,
            device=device
        )

        checkpoint = load_checkpoint_from_wandb(
            run,
            icarl,
            "model.pth"
        )
        
        checkpoint_dict, artifact = checkpoint
        icarl.load_state_dict(checkpoint_dict['model'])
        return icarl.model.classifier
    except Exception as e:
        print(f"Error loading iCaRL classifier: {e}")
        return None