import wandb
import torch
from torch.nn import Module
from pathlib import Path

def save_model_to_wandb(
    run: wandb.Run, model: Module, group: str, filename: str = "model.pth"
) -> None:
    """Save a PyTorch model to WandB as an artifact."""

    # 1. Save model locally
    torch.save(model.state_dict(), filename)

    # 2. Create artifact
    artifact_name = f"{group}-checkpoints"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model"
    )

    artifact.add_file(filename)

    # 3. Log artifact to the existing run
    run.log_artifact(artifact)

    print(f"Model saved to WandB as artifact '{artifact_name}'.")

def load_model_from_wandb(
    run: wandb.Run,
    model: Module,
    group: str,
    version: str = "latest"
) -> None:
    """Download the latest model artifact and load it into `model`."""
    try:
        artifact_name = f"{group}-checkpoints"
        artifact = run.use_artifact(f"{artifact_name}:{version}", type="model")
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / "model.pth"
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from: {model_path}")
    except Exception as e :
        print(f"Model checkpoint not found on WandB. {e}")
