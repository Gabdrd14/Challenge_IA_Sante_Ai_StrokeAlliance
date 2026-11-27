import pathlib
import sys

import cv2
import numpy as np

from .config import PREDICTION_THRESHOLD
from .data import convert_pvdataset_to_npy


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    for mask_slice in mask:
        if np.sum(mask_slice) == 0:
            continue
        num_regions, labelled_img, stats, _ = cv2.connectedComponentsWithStats(mask_slice)
        for label in range(1, num_regions):
            area = stats[label, cv2.CC_STAT_AREA]
            if area <= 3:
                mask_slice[labelled_img == label] = 0
    return mask


class SlicePredictor:
    """Prédicteur différé utilisant le 3D U-Net entraîné."""

    def __init__(self):
        self.model = None
        self.device = None
        self.checkpoint = None

    def _load_model(self, checkpoint: pathlib.Path):
        import torch

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        bundled_root = getattr(sys, "_MEIPASS", None)
        package_root = (
            pathlib.Path(bundled_root)
            if bundled_root
            else pathlib.Path(__file__).resolve().parents[2]
        )
        sys.path.insert(0, str(package_root))
        from projet_master.model_inference import load_inference_model

        self.model = load_inference_model(checkpoint, self.device)
        self.checkpoint = checkpoint

    def predict(self, model_volume: np.ndarray, checkpoint: pathlib.Path) -> np.ndarray:
        """Prédit un volume converti au format d'entraînement (D, H, W)."""
        import torch

        if self.model is None or self.checkpoint != checkpoint:
            self._load_model(checkpoint)

        model_volume = np.asarray(model_volume, dtype=np.float32)
        if model_volume.ndim != 3:
            raise ValueError(
                f"Volume modèle attendu sous la forme (D,H,W), reçu {model_volume.shape}."
            )

        normalized = (model_volume - model_volume.mean()) / (
            model_volume.std() + 1e-8
        )

        with torch.inference_mode():
            batch = (
                torch.from_numpy(normalized.copy())
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )
            logits = self.model(batch)
            if logits.ndim != 5 or logits.shape[1] != 1:
                raise RuntimeError(
                    f"Sortie inattendue du modèle 3D : {tuple(logits.shape)}"
                )
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        return (probabilities >= PREDICTION_THRESHOLD).astype(np.uint8)


def predict_pvdataset_volume(
    predictor: "SlicePredictor",
    file_path: str,
    scan_id,
    checkpoint: pathlib.Path,
) -> np.ndarray:
    """Exécute exactement le même pipeline pour le viewer et l'étude entière."""
    model_volume = convert_pvdataset_to_npy(file_path, scan_id)
    return predictor.predict(model_volume, checkpoint)
