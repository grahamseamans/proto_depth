import torch
from pathlib import Path
import urllib.request
import kaolin.io.obj
from kaolin.math.quat import quat_unit, rot33_from_quat


def download_models(models_dir: Path) -> None:
    """Download common 3D test models if they don't exist."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    model_urls = {
        "bunny": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
        "spot": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/spot.obj",
        "armadillo": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj",
    }
    for name, url in model_urls.items():
        path = models_dir / f"{name}.obj"
        if not path.exists():
            print(f"Downloading {name} model...")
            urllib.request.urlretrieve(url, path)


def load_mesh(
    models_dir: str | Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a mesh from the models directory. Downloads if not found."""
    models_dir = Path(models_dir)
    if not models_dir.exists() or not any(models_dir.glob("*.obj")):
        download_models(models_dir)
    model_files = {
        "bunny": "bunny.obj",
        "spot": "spot.obj",
        "armadillo": "armadillo.obj",
    }
    for name, filename in model_files.items():
        path = models_dir / filename
        if path.exists():
            print(f"Loading {name} model...")
            mesh = kaolin.io.obj.import_mesh(str(path))
            vertices = mesh.vertices.to(dtype=torch.float32, device=device)
            faces = mesh.faces.to(dtype=torch.int64, device=device)
            return vertices, faces
    raise FileNotFoundError(f"Failed to load any models from {models_dir}")


def transform_vertices(
    vertices: torch.Tensor,
    position: torch.Tensor,
    rotation: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Transform mesh vertices based on position, rotation (quaternion), and scale.
    Uses Kaolin's quaternion utilities for robust, differentiable transforms.
    """
    # Ensure rotation is a 1D tensor of length 4
    if rotation.shape[-1] != 4:
        raise ValueError("Rotation must be a quaternion of shape (4,)")

    # Normalize quaternion to ensure valid rotation
    q = quat_unit(rotation)
    # Convert quaternion to rotation matrix
    R = rot33_from_quat(q.unsqueeze(0)).squeeze(0)  # [3,3]

    # Scale the rotation matrix
    R = R * scale

    # Apply rotation and translation
    transformed = vertices @ R.T + position.unsqueeze(0)
    return transformed
