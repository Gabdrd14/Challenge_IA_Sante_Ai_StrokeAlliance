from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import DynUNet


class AVCInferenceModel(torch.nn.Module):
    """Architecture 3D U-Net minimale utilisée pour charger un checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        strides = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            kernel_size=[(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=[16, 32, 64, 128, 256],
            norm_name="instance",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def load_inference_model(
    checkpoint: Path,
    device: torch.device,
) -> AVCInferenceModel:
    """Construit le réseau et charge les poids d'un checkpoint Lightning."""
    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    if not isinstance(state_dict, dict):
        raise ValueError("Le checkpoint ne contient pas de state_dict exploitable.")

    model = AVCInferenceModel().to(device)
    model_state_dict = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state_dict:
        raise ValueError("Aucun poids du réseau n'a été trouvé dans le checkpoint.")
    model.model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    return model
