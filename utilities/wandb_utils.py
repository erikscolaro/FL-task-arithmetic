import wandb
import torch
from torch.nn import Module
from pathlib import Path

def save_model_to_wandb(
    run: wandb.Run,
    model: Module, 
    filename: str = "model.pth",
    metadata: dict = None,
) -> None:
    """Save a PyTorch model to WandB as an artifact."""

    # 1. Save model locally
    torch.save(model.state_dict(), filename)

    # 2. Create artifact
    artifact_name = f"{run.group}-models" if hasattr(run, "group") and run.group else "models"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata or {}
    )

    artifact.add_file(filename)

    # 3. Log artifact to the existing run
    run.log_artifact(artifact)

    print(f"Model saved to WandB as artifact '{artifact_name}'.")

def load_model_from_wandb(
    run: wandb.Run,
    model: Module, 
    filename: str = "model.pth",
    version: str = "latest"
) -> wandb.Artifact | None:
    """Download the latest model artifact and load it into `model`."""
    try:
        artifact_name = f"{run.group}-models" if hasattr(run, "group") and run.group else "models"
        artifact = run.use_artifact(f"{artifact_name}:{version}", type="model")
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / filename
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, model.device if hasattr(model, 'device') else None))
        print(f"Successfully loaded model from: {model_path}")
        return artifact
    except Exception as e :
        print(e)
        print(f"Model checkpoint not found on WandB. {e}")



def save_checkpoint_to_wandb(
    run: wandb.Run,
    checkpoint: dict, 
    filename: str = "model.pth",
    metadata: dict = None,
) -> None:
    """Save a PyTorch model to WandB as an artifact."""

    # 1. Save model locally
    torch.save(checkpoint, filename)

    # 2. Create artifact
    artifact_name = f"{run.group}-checkpoints" if hasattr(run, "group") and run.group else "checkpoints"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata or {}
    )

    artifact.add_file(filename)

    # 3. Log artifact to the existing run
    run.log_artifact(artifact)

    print(f"Model saved to WandB as artifact '{artifact_name}'.")


def load_checkpoint_from_wandb(
    run: wandb.Run,
    model: Module, 
    filename: str = "model.pth",
    version: str = "latest"
) -> tuple[dict, wandb.Artifact] | None:
    """Download the latest model artifact and load it into `model`."""
    try:
        artifact_name = f"{run.group}-checkpoints" if hasattr(run, "group") and run.group else "checkpoints"
        artifact = run.use_artifact(f"{artifact_name}:{version}", type="model")
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / filename
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, model.device if hasattr(model, 'device') else None, weights_only=False)
        print(f"Successfully loaded model from: {model_path}")
        return checkpoint, artifact
    except Exception as e :
        print(e)
        print(f"Model checkpoint not found on WandB. {e}")
