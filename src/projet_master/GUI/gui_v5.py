
import os
import pathlib
import sys
import logging
import unicodedata
import zipfile
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
import brkraw
from roifile import ImagejRoi, ROI_TYPE, roiwrite, roiread
import pandas as pd
from PySide6.QtCore import QDateTime, QEvent, QPoint, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QCursor,
    QIcon,
    QImage,
    QKeyEvent,
    QBrush,
    QColor,
    QPixmap,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QComboBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QDialog,
    QProgressBar,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
if sys.platform == "darwin":
    CONFIG_DIRECTORY = (
        pathlib.Path.home() / "Library" / "Application Support" / "StrokeAlliance"
    )
elif os.name == "nt":
    CONFIG_DIRECTORY = pathlib.Path(
        os.environ.get("APPDATA", pathlib.Path.home() / "AppData" / "Roaming")
    ) / "StrokeAlliance"
else:
    CONFIG_DIRECTORY = pathlib.Path(
        os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")
    ) / "stroke-alliance"
MODEL_DIRECTORY_FILE = CONFIG_DIRECTORY / "model_directory.txt"
MASK_DRAWING_METHOD_FILE = CONFIG_DIRECTORY / "mask_drawing_method.txt"
LOG_FILE = CONFIG_DIRECTORY / "guiv2.log"
PREDICTION_THRESHOLD = 0.5
MASK_ERASER_DIAMETER = 0.90
VOXEL_WIDTH_MM = 0.07
VOXEL_HEIGHT_MM = 0.07
VOXEL_DEPTH_MM = 0.5

VOXEL_AREA_MM2 = VOXEL_WIDTH_MM * VOXEL_HEIGHT_MM
VOXEL_VOLUME_MM3 = VOXEL_AREA_MM2 * VOXEL_DEPTH_MM

VOXEL_WIDTH_MM_RAT = 0.15
VOXEL_HEIGHT_MM_RAT = 0.15
VOXEL_DEPTH_MM_RAT = 0.9

VOXEL_AREA_MM2_RAT = VOXEL_WIDTH_MM_RAT * VOXEL_HEIGHT_MM_RAT
VOXEL_VOLUME_MM3_RAT = VOXEL_AREA_MM2_RAT * VOXEL_DEPTH_MM_RAT

LOGO_PATH = pathlib.Path(__file__).resolve().parent / "logo.png"
ETAPLAB_LOGO_PATH = pathlib.Path(__file__).resolve().parent / "etaplablogo.png"


def replace_erased_contour_segment(
    mask: np.ndarray,
    outline_mask: np.ndarray,
    erased_outline: np.ndarray,
    stroke_points: np.ndarray,
) -> bool:
    """Remplace une portion effacée du contour par le tracé de l'utilisateur."""
    if not len(stroke_points):
        return False

    height, width = mask.shape
    stroke_points = np.asarray(stroke_points, dtype=np.int32)
    stroke_points[:, 0] = np.clip(stroke_points[:, 0], 0, width - 1)
    stroke_points[:, 1] = np.clip(stroke_points[:, 1], 0, height - 1)
    stroke = np.zeros(mask.shape, dtype=np.uint8)
    if len(stroke_points) == 1:
        cv2.circle(stroke, tuple(stroke_points[0]), 3, 1, thickness=-1)
    else:
        cv2.polylines(
            stroke,
            [stroke_points.reshape(-1, 1, 2)],
            False,
            1,
            thickness=3,
            lineType=cv2.LINE_8,
        )
    nearby_stroke = cv2.dilate(
        stroke,
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)

    contours, _ = cv2.findContours(
        np.ascontiguousarray(outline_mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    component_count, component_labels = cv2.connectedComponents(
        np.ascontiguousarray(mask, dtype=np.uint8),
        connectivity=8,
    )

    for contour in contours:
        points = contour[:, 0, :]
        hidden = erased_outline[points[:, 1], points[:, 0]]
        if not hidden.any() or hidden.all():
            continue

        closed_hidden = cv2.morphologyEx(
            hidden.astype(np.uint8).reshape(-1, 1),
            cv2.MORPH_CLOSE,
            np.ones((5, 1), dtype=np.uint8),
        ).ravel().astype(bool)
        run_starts = np.flatnonzero(closed_hidden & ~np.roll(closed_hidden, 1))
        for run_start in run_starts:
            index = int(run_start)
            while closed_hidden[index]:
                index = (index + 1) % len(points)

            before_index = (int(run_start) - 1) % len(points)
            before = points[before_index]
            after = points[index]
            if not (
                nearby_stroke[before[1], before[0]]
                and nearby_stroke[after[1], after[0]]
            ):
                continue

            forward_distance = (
                np.linalg.norm(stroke_points[0] - before)
                + np.linalg.norm(stroke_points[-1] - after)
            )
            reverse_distance = (
                np.linalg.norm(stroke_points[-1] - before)
                + np.linalg.norm(stroke_points[0] - after)
            )
            bridge_points = (
                stroke_points
                if forward_distance <= reverse_distance
                else stroke_points[::-1]
            )

            remaining_indices = [index]
            while remaining_indices[-1] != before_index:
                remaining_indices.append(
                    (remaining_indices[-1] + 1) % len(points)
                )
            polygon = np.concatenate(
                (
                    points[remaining_indices],
                    bridge_points,
                )
            )
            replacement = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(replacement, [polygon.reshape(-1, 1, 2)], 1)

            component_index = int(component_labels[before[1], before[0]])
            if not component_index or component_index >= component_count:
                continue
            component = component_labels == component_index
            mask[component] = 0
            mask[replacement.astype(bool)] = 1
            outline_mask[component] = 0
            outline_mask[replacement.astype(bool)] = 1
            erased_outline[component] = False
            erased_outline[replacement.astype(bool)] = False
            return True

    return False


def configure_logging() -> logging.Logger:
    """Configure le journal applicatif dans un dossier accessible en écriture."""
    logger = logging.getLogger("stroke_alliance")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    try:
        CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError:
        handler = logging.NullHandler()
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )

    logger.addHandler(handler)
    return logger


LOGGER = configure_logging()


def find_dataset_files(dir_path: str) -> list[str]:
    """Recherche les fichiers .PvDatasets dans un dossier."""
    if not dir_path or not os.path.isdir(dir_path):
        return []

    return sorted(
        os.path.join(dir_path, entry.name)
        for entry in os.scandir(dir_path)
        if entry.name.lower().endswith(".pvdatasets")
        and entry.is_file()
    )


def load_data(file_path: str, scan_id=None):
    """
    Charge les données IRM d'un scan dans un fichier PvDataset.

    Returns:
        data: tableau IRM ou None
        selected_scan: informations du scan sélectionné ou None
        all_scans: informations de tous les scans
    """
    loader = brkraw.load(file_path)
    all_scans = loader.info(scope="scan", as_dict=True)

    if not all_scans:
        return None, None, all_scans

    selected_scan_id = scan_id if scan_id in all_scans else next(iter(all_scans))
    scan_info = all_scans[selected_scan_id]
    return np.asarray(loader.get_dataobj(selected_scan_id)), scan_info, all_scans


def scan_label(scan_id, scan_info: dict) -> str:
    """Construit un libellé lisible en conservant l'identifiant du scan."""
    protocol = scan_info.get("Protocol", "Protocol inconnu")
    return f"{scan_id} - {protocol}"


def default_scan_index(all_scans: dict) -> int:
    """Privilégie T2 haute résolution, puis T2 star rapide, puis le premier scan."""
    scans = list(all_scans.items())
    normalized_protocols = []
    for _, scan_info in scans:
        protocol = unicodedata.normalize(
            "NFKD",
            str(scan_info.get("Protocol", ""))
            .casefold()
            .replace("*", "star"),
        )
        normalized_protocols.append(
            "".join(
                character
                for character in protocol
                if character.isalnum()
                and not unicodedata.combining(character)
            )
        )

    for index, protocol in enumerate(normalized_protocols):
        if "t2hauteresolution" in protocol:
            return index
    for index, protocol in enumerate(normalized_protocols):
        if "t2star" in protocol and "rapide" in protocol:
            return index
    return 0


def model_scan_index(all_scans: dict, model_path: pathlib.Path | None) -> int:
    """Sélectionne le canal IRM correspondant au modèle choisi."""
    if model_path is None:
        return default_scan_index(all_scans)

    if "rat" in str(model_path).casefold():
        expected = ("t2turborare",)
    elif "hem" in str(model_path).casefold():
        expected = ("t2starrapide", "t2star")
    elif "souris" in str(model_path).casefold():
        expected = ("t2hauteresolution",)
    else:
        return default_scan_index(all_scans)

    for index, (_, scan_info) in enumerate(all_scans.items()):
        protocol = unicodedata.normalize(
            "NFKD",
            str(scan_info.get("Protocol", ""))
            .casefold()
            .replace("*", "star"),
        )
        protocol = "".join(
            character
            for character in protocol
            if character.isalnum()
            and not unicodedata.combining(character)
        )
        if any(pattern in protocol for pattern in expected):
            return index

    return default_scan_index(all_scans)


def model_scan_id(all_scans: dict, model_path: pathlib.Path | None):
    """Retourne l'identifiant Bruker du scan choisi pour un modèle."""
    scan_index = model_scan_index(all_scans, model_path)
    scan_ids = list(all_scans)
    if 0 <= scan_index < len(scan_ids):
        return scan_ids[scan_index]
    return None


def convert_pvdataset_to_npy(file_path: str, scan_id) -> np.ndarray:
    """Convertit le scan sélectionné au format modèle (D, H, W)."""
    loader = brkraw.load(file_path)
    scans = loader.info(scope="scan", as_dict=True)
    if scan_id not in scans:
        raise ValueError(f"Analyse sélectionnée introuvable : {scan_id}.")

    raw_volume = np.asarray(loader.get_dataobj(scan_id))
    raw_volume = np.squeeze(raw_volume)
    while raw_volume.ndim > 3:
        raw_volume = raw_volume[..., 0]
    if raw_volume.ndim != 3:
        raise ValueError(
            f"L'analyse sélectionnée doit être un volume 3D, reçu {raw_volume.shape}."
        )

    # Les volumes Bruker sont généralement renvoyés en (H, W, D) ;
    # le modèle entraîné utilise le format (D, H, W) produit par loader.py.
    
    if ((raw_volume.shape[0] == 256 and raw_volume.shape[1] == 256) or 
       (raw_volume.shape[0] == 266 and raw_volume.shape[1] == 226)):
        
        model_volume = np.transpose(raw_volume, (2, 1, 0))
    
    
    elif ((raw_volume.shape[1] == 256 and raw_volume.shape[2] == 256) or
         (raw_volume.shape[1] == 226 and raw_volume.shape[2] == 266)):
        
        model_volume = raw_volume
         
    else:
        raise ValueError(
            "Dimensions incompatibles avec le 3D U-Net : "
            f"{raw_volume.shape}. Le modèle attend des coupes 256x256."
        )

    return model_volume.astype(np.float32, copy=False)


def save_prediction_rois(
    prediction_masks: np.ndarray,
    output_path: pathlib.Path,
    allow_empty: bool = False,
) -> pathlib.Path:
    """Enregistre les contours prédits comme ROI ImageJ dans un ZIP choisi."""
    masks = np.asarray(prediction_masks)
    if masks.ndim != 3:
        raise ValueError(f"Masques attendus sous la forme (D,H,W), reçu {masks.shape}.")

    rois = []
    for slice_index, slice_mask in enumerate(masks):
        binary_mask = np.asarray(slice_mask > 0, dtype=np.uint8)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        valid_contours = []
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            points = contour[:, 0, :].astype(np.float32)
            valid_contours.append(points)

        if not valid_contours:
            continue

        path_commands = []
        all_points = np.concatenate(valid_contours)
        for points in valid_contours:
            contour_commands = [0.0, *points[0]]
            for point in points[1:]:
                contour_commands.extend([1.0, *point])
            contour_commands.append(4.0)
            path_commands.extend(contour_commands)

        roi = ImagejRoi(
            roitype=ROI_TYPE.RECT,
            name=f"{slice_index + 1:03d}",
            position=slice_index + 1,
            left=int(np.floor(all_points[:, 0].min())),
            top=int(np.floor(all_points[:, 1].min())),
            right=int(np.ceil(all_points[:, 0].max())) + 1,
            bottom=int(np.ceil(all_points[:, 1].max())) + 1,
            shape_roi_size=len(path_commands),
            multi_coordinates=np.asarray(path_commands, dtype=np.float32),
        )
        rois.append(roi)

    if not rois:
        if allow_empty:
            output_path = pathlib.Path(output_path)
            if output_path.suffix.lower() != ".zip":
                output_path = output_path.with_suffix(".zip")
            with zipfile.ZipFile(output_path, "w"):
                pass
            return output_path
        raise ValueError("Aucun contour prédit à exporter en ROI.")
    output_path = pathlib.Path(output_path)
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")
    output_path.unlink(missing_ok=True)
    roiwrite(output_path, rois)
    return output_path


def save_modified_roi_slice(
    prediction_masks: np.ndarray,
    output_path: pathlib.Path,
    slice_index: int,
) -> pathlib.Path:
    """Remplace uniquement une coupe dans un ZIP ROI existant."""
    output_path = pathlib.Path(output_path)
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")

    existing_rois = roiread(str(output_path)) if output_path.is_file() else []
    retained_rois = []
    for roi in existing_rois:
        try:
            roi_slice = int(getattr(roi, "position", 0) or 0)
        except (TypeError, ValueError):
            roi_slice = 0
        if roi_slice <= 0:
            name = str(getattr(roi, "name", ""))
            digits = "".join(character for character in name if character.isdigit())
            roi_slice = int(digits) if digits else 0
        if roi_slice != slice_index + 1:
            retained_rois.append(roi)
    existing_rois = retained_rois

    model_masks = np.transpose(np.asarray(prediction_masks), (2, 1, 0))
    if not 0 <= slice_index < model_masks.shape[0]:
        raise ValueError(f"Index de coupe invalide : {slice_index + 1}.")

    binary_mask = np.asarray(model_masks[slice_index] > 0, dtype=np.uint8)
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    valid_contours = [
        contour[:, 0, :].astype(np.float32)
        for contour in contours
        if contour.shape[0] >= 3
    ]

    if valid_contours:
        all_points = np.concatenate(valid_contours)
        path_commands = []
        for points in valid_contours:
            contour_commands = [0.0, *points[0]]
            for point in points[1:]:
                contour_commands.extend([1.0, *point])
            contour_commands.append(4.0)
            path_commands.extend(contour_commands)
        existing_rois.append(
            ImagejRoi(
                roitype=ROI_TYPE.RECT,
                name=f"{slice_index + 1:03d}",
                position=slice_index + 1,
                left=int(np.floor(all_points[:, 0].min())),
                top=int(np.floor(all_points[:, 1].min())),
                right=int(np.ceil(all_points[:, 0].max())) + 1,
                bottom=int(np.ceil(all_points[:, 1].max())) + 1,
                shape_roi_size=len(path_commands),
                multi_coordinates=np.asarray(path_commands, dtype=np.float32),
            )
        )

    existing_rois.sort(
        key=lambda roi: int(getattr(roi, "position", 0) or 0)
    )
    if existing_rois:
        output_path.unlink(missing_ok=True)
        roiwrite(output_path, existing_rois)
    else:
        with zipfile.ZipFile(output_path, "w"):
            pass
    return output_path


def available_checkpoints(model_directory: pathlib.Path) -> list[pathlib.Path]:
    """Liste les checkpoints 3D U-Net disponibles dans le dossier configuré."""
    checkpoints = sorted(model_directory.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé dans {model_directory}."
        )
    return checkpoints


def saved_model_directory() -> pathlib.Path | None:
    """Lit le dossier de modèles configuré s'il existe encore."""
    if not MODEL_DIRECTORY_FILE.is_file():
        return None
    configured_path = MODEL_DIRECTORY_FILE.read_text(encoding="utf-8").strip()
    if not configured_path:
        return None
    model_directory = pathlib.Path(configured_path).expanduser()
    return model_directory if model_directory.is_dir() else None


def save_model_directory(model_directory: pathlib.Path) -> None:
    """Enregistre le dossier de modèles pour les prochains lancements."""
    CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MODEL_DIRECTORY_FILE.write_text(
        str(model_directory.resolve()),
        encoding="utf-8",
    )


def saved_mask_drawing_method() -> bool:
    """Lit la préférence de dessin ImageJ, désactivée par défaut."""
    if not MASK_DRAWING_METHOD_FILE.is_file():
        return False
    try:
        value = MASK_DRAWING_METHOD_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        LOGGER.exception("Impossible de lire la préférence de dessin du masque.")
        return False
    if value not in {"true", "false"}:
        LOGGER.warning("Préférence de dessin du masque invalide: %r", value)
        return False
    return value == "true"


def save_mask_drawing_method(enabled: bool) -> None:
    """Enregistre la préférence de dessin du masque pour les prochains lancements."""
    CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    MASK_DRAWING_METHOD_FILE.write_text(
        "true" if enabled else "false",
        encoding="utf-8",
    )

def file_generator(folder_path: str , output_folder: str ,nb_slice : int) -> None:
    """Générateur du fichier excel des résultats sur l'ensemble des sujets du dossier racine """
    name_file = f"results_{folder_path}_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}.xlsx"
    """Structure du fichier excel : id , surface slice , surface slice +1, surface slice + n , volume total"""
    
    """fonction parcoura chaque roifile dans le dossier output de la prediction totale du dossier"""
    # Construire dynamiquement les colonnes de surface :
    # - une colonne "surface_slice_i" pour chaque coupe comprise entre 1 et nb_slice
    # - conserver "id" comme première colonne
    # - conserver "volume_total" comme dernière colonne
    # colonnes_slice = [f"surface_slice_{i + 1}" for i in range(nb_slice)]
    # colonnes = ["id", *colonnes_slice, "volume_total"]
    # dataframe = pd.DataFrame(columns=colonnes)

    # Parcourir chaque fichier ROI du dossier contenant les prédictions totales.
    # Pour chaque ROI :
    # - identifier le sujet dans le nom du fichier ou son dossier parent ;
    # - lire les coordonnées des polygones et leur position de coupe ;
    # - calculer la surface de chaque coupe ;
    # - ajouter une ligne du tableau pour le sujet ;
    # - sommer les surfaces des coupes pour obtenir le volume total.
    # dossier_pred_total = pathlib.Path(output_folder)
    # roi_files = sorted(dossier_pred_total.rglob("*.zip"))
    # for roi_file in roi_files:
    #     ...


    if os.path.exists(f"{output_folder}/{name_file}"):
        i = 1
        while True:
            new_name_file = f"results_{folder_path}_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}_{i}.xlsx"
            if not os.path.exists(new_name_file):
                name_file = new_name_file
                break
            i += 1
    
    


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


def prepare_volume(data: np.ndarray) -> np.ndarray:
    """Prépare un tableau 3D pour la navigation entre les coupes."""
    data = np.squeeze(data)

    while data.ndim > 3:
        data = data[..., 0]

    if data.ndim == 2:
        data = data[:, :, np.newaxis]

    if data.ndim != 3:
        raise ValueError(
            f"Unsupported MRI array shape: {data.shape}"
        )

    return data


def slice_to_pixmap(
    slice_data: np.ndarray,
    contrast: int = 100,
    intensity: int = 100,
    saturation: int = 0,
) -> QPixmap:
    """Convertit une coupe IRM en niveaux de gris avec les réglages demandés."""
    slice_data = np.asarray(slice_data)
    if slice_data.ndim != 2:
        raise ValueError(f"Expected a 2D MRI slice, got {slice_data.shape}")

    finite = np.isfinite(slice_data)
    if not finite.any():
        raise ValueError("MRI slice contains no finite intensity values.")

    values = slice_data[finite].astype(np.float64, copy=False)
    low, high = np.percentile(values, (1.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0

    center = (low + high) / 2.0
    display_range = (high - low) * (100.0 / max(contrast, 1))
    # La saturation resserre ou élargit la plage visible autour de son centre.
    display_range *= max(0.01, 1.0 - saturation / 100.0)
    low = center - display_range / 2.0
    high = center + display_range / 2.0
    scaled = (slice_data.astype(np.float64, copy=False) - low) * (
        255.0 / (high - low)
    )
    normalized = np.zeros(slice_data.shape, dtype=np.uint8)
    normalized[finite] = np.clip(scaled[finite], 0, 255).round().astype(np.uint8)
    if intensity != 100:
        adjusted = (
            (normalized[finite].astype(np.float64) - 128.0)
            * (intensity / 100.0)
            + 128.0
        )
        normalized[finite] = np.clip(adjusted, 0, 255).round().astype(np.uint8)
    raw = np.ascontiguousarray(normalized)

    height, width = raw.shape
    image = QImage(
        raw.data,
        width,
        height,
        raw.strides[0],
        QImage.Format.Format_Grayscale8,
    ).copy()  # Détache le QImage de la mémoire NumPy.
    return QPixmap.fromImage(image)


@dataclass
class ContourPointEdit:
    slice_index: int
    contour: np.ndarray
    point_index: int
    marker_index: int
    current_position: tuple[int, int]
    component_mask: np.ndarray
    original_mask: np.ndarray
    original_outline: np.ndarray
    original_erased: np.ndarray
    original_total_voxels: int
    original_slice_voxels: int


class MRIImageView(QGraphicsView):
    """Vue IRM à résolution native avec zoom et déplacement à la souris."""

    edit_path_finished = Signal(object, bool)
    contour_region_selection_requested = Signal(object)
    contour_control_point_added = Signal(object)
    contour_control_point_removed = Signal(int)
    contour_point_drag_started = Signal(object, int, int)
    contour_point_dragged = Signal(object)
    contour_point_drag_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        #self._pixmap_item.setTransformationMode(
            #Qt.TransformationMode.SmoothTransformation
        #)
        self._prediction_item: QGraphicsPathItem | None = None
        self._has_image = False
        self._user_zoomed = False
        self._edit_mode: str | None = None
        self._legacy_polygon_drawing = False
        self._contour_edit_mode = False
        self._active_contour: np.ndarray | None = None
        self._active_visible_indices = np.array([], dtype=np.intp)
        self._control_points: list[tuple[int, int]] = []
        self._control_point_items: list[QGraphicsEllipseItem] = []
        self._last_auto_added_control_point: tuple[int, int] | None = None
        self._contour_drag_active = False
        self._drag_marker_index = -1
        self._contour_drag_origin_scene: QPointF | None = None
        self._contour_drag_origin_global: QPointF | None = None
        self._contour_drag_scale_x = 1.0
        self._contour_drag_scale_y = 1.0
        self._pan_with_right_button = False
        self._pan_active = False
        self._pan_waiting_for_release = False
        self._pan_button: Qt.MouseButton | None = None
        self._pan_last_position = QPoint()
        self._edit_points = []
        self._edit_item = None
        self._drawing_active = False
        self._last_pointer_position = QPoint()
        self._auto_scroll_timer = QTimer(self)
        self._auto_scroll_timer.setInterval(30)
        self._auto_scroll_timer.timeout.connect(self._auto_scroll_while_drawing)

        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMinimumSize(300, 300)
        self.viewport().setMouseTracking(True)
        self.setToolTip(
            "Molette : zoom | Glisser un point du contour rouge pour l'ajuster | "
            "Glisser ailleurs : déplacer l'image"
        )

    def set_mri_pixmap(self, pixmap: QPixmap):
        """Définit une coupe native en conservant le zoom si possible."""
        had_image = self._has_image
        was_zoomed = self._user_zoomed
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._has_image = not pixmap.isNull()
        if not had_image or not was_zoomed:
            self.fit_image()
        else:
            # Conserve le zoom et le déplacement de l'utilisateur pendant
            # le défilement des coupes.
            self._user_zoomed = True

    def set_prediction_overlay(
        self,
        mask: np.ndarray | None,
        erased_outline: np.ndarray | None = None,
    ):
        """Affiche le contour du masque en vectoriel, sans épaississement au zoom."""
        if self._prediction_item is not None:
            self._scene.removeItem(self._prediction_item)
            self._prediction_item = None
        if mask is None:
            return

        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"Masque 2D attendu, reçu {mask.shape}.")
        if erased_outline is None:
            erased_outline = np.zeros(mask.shape, dtype=bool)
        else:
            erased_outline = np.asarray(erased_outline, dtype=bool)
            if erased_outline.shape != mask.shape:
                raise ValueError(
                    "Le masque d'effacement du contour doit avoir la même "
                    f"forme que le masque, reçu {erased_outline.shape} et {mask.shape}."
                )
        if not mask.any():
            return

        contours, _ = cv2.findContours(
            np.ascontiguousarray(mask, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        path = QPainterPath()
        for contour in contours:
            points = contour[:, 0, :]
            if len(points) == 0:
                continue
            hidden = erased_outline[points[:, 1], points[:, 0]]
            visible_indices = np.flatnonzero(~hidden)
            if not hidden.any():
                path.moveTo(float(points[0, 0]), float(points[0, 1]))
                for point in points[1:]:
                    path.lineTo(float(point[0]), float(point[1]))
                path.closeSubpath()
                continue

            if not len(visible_indices):
                continue
            starts = []
            for index in visible_indices:
                previous = (index - 1) % len(points)
                if hidden[previous]:
                    starts.append(index)
            if not starts:
                starts.append(int(visible_indices[0]))
            for start in starts:
                path.moveTo(float(points[start, 0]), float(points[start, 1]))
                index = (start + 1) % len(points)
                while not hidden[index] and index != start:
                    path.lineTo(float(points[index, 0]), float(points[index, 1]))
                    index = (index + 1) % len(points)

        self._prediction_item = QGraphicsPathItem(path)
        pen = QPen(Qt.GlobalColor.red, 1.5)
        pen.setCosmetic(True)
        self._prediction_item.setPen(pen)
        self._scene.addItem(self._prediction_item)

    def set_contour_edit_mode(self, enabled: bool):
        self._contour_edit_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self._draw_control_points()
            self.setToolTip(
                "Cliquer sur une zone pour la sélectionner | "
                "Glisser un point du contour pour l'ajuster | "
                "Double-clic sur le contour ou dans la zone : ajouter un point | "
                "Clic droit sur un point : le supprimer | "
                "Clic droit + glisser : déplacer l'image"
            )
        else:
            self.viewport().unsetCursor()
            self._remove_control_point_items()
            self._contour_drag_active = False
            self._drag_marker_index = -1
            self._contour_drag_origin_scene = None
            self._contour_drag_origin_global = None
            self.setDragMode(
                QGraphicsView.DragMode.NoDrag
                if self._edit_mode is not None
                else QGraphicsView.DragMode.ScrollHandDrag
            )
            self.setToolTip(
                "Molette : zoom | Glisser ailleurs : déplacer l'image"
            )

    def set_active_contour(
        self,
        contour: np.ndarray | None,
        control_points: list[tuple[int, int]],
        erased_outline: np.ndarray | None = None,
    ):
        self._active_contour = (
            None
            if contour is None
            else np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2).copy()
        )
        self._control_points = list(control_points)
        if self._active_contour is None:
            self._active_visible_indices = np.array([], dtype=np.intp)
        elif erased_outline is None:
            self._active_visible_indices = np.arange(
                len(self._active_contour),
                dtype=np.intp,
            )
        else:
            points = self._active_contour[:, 0, :]
            hidden = np.asarray(erased_outline, dtype=bool)[
                points[:, 1],
                points[:, 0],
            ]
            self._active_visible_indices = np.flatnonzero(~hidden)
        if self._contour_edit_mode:
            self._draw_control_points()

    def _remove_control_point_items(self):
        for item in self._control_point_items:
            self._scene.removeItem(item)
        self._control_point_items = []

    def _draw_control_points(self):
        self._remove_control_point_items()
        if not self._contour_edit_mode or self._active_contour is None:
            return
        for x, y in self._control_points:
            item = QGraphicsEllipseItem(-3, -3, 6, 6)
            item.setBrush(QBrush(QColor(255, 255, 255, 220)))
            pen = QPen(Qt.GlobalColor.yellow, 1.5)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setZValue(3)
            item.setPos(float(x), float(y))
            self._scene.addItem(item)
            self._control_point_items.append(item)

    def _control_point_index_at(
        self,
        position: QPoint,
        hit_radius: float = 10.0,
    ) -> int:
        hit_radius_squared = hit_radius**2
        closest_index = -1
        closest_distance = hit_radius_squared
        for index, (x, y) in enumerate(self._control_points):
            marker_position = self.mapFromScene(QPointF(float(x), float(y)))
            delta = marker_position - position
            distance_squared = delta.x() ** 2 + delta.y() ** 2
            if distance_squared <= closest_distance:
                closest_index = index
                closest_distance = distance_squared
        return closest_index

    def move_control_point(self, index: int, x: int, y: int):
        if not 0 <= index < len(self._control_points):
            return
        self._control_points[index] = (x, y)
        if index < len(self._control_point_items):
            self._control_point_items[index].setPos(float(x), float(y))

    def _contour_scene_position(self, global_position: QPointF) -> QPointF:
        if (
            self._contour_drag_origin_scene is None
            or self._contour_drag_origin_global is None
        ):
            raise RuntimeError("Le glissement de contour n'a pas été initialisé.")
        # Live volume updates can resize the view; global deltas keep the scene
        # anchor stable while the viewport is being repositioned.
        delta = global_position - self._contour_drag_origin_global
        return QPointF(
            self._contour_drag_origin_scene.x()
            + delta.x() / self._contour_drag_scale_x,
            self._contour_drag_origin_scene.y()
            + delta.y() / self._contour_drag_scale_y,
        )

    def _nearest_active_contour_point(
        self,
        position: QPointF,
        tolerance: float | None = None,
        include_erased: bool = False,
        excluded_points: set[tuple[int, int]] | None = None,
    ) -> int | None:
        if self._active_contour is None:
            return None
        visible_indices = (
            np.arange(len(self._active_contour), dtype=np.intp)
            if include_erased
            else self._active_visible_indices
        )
        if not len(visible_indices):
            return None
        scale = max(abs(self.transform().m11()), 1e-6)
        if tolerance is None:
            tolerance = 10.0 / scale
        query = np.array([position.x(), position.y()])
        points = self._active_contour[
            visible_indices,
            0,
            :,
        ].astype(float)
        if excluded_points:
            available = np.array(
                [
                    (int(point[0]), int(point[1])) not in excluded_points
                    for point in points
                ],
                dtype=bool,
            )
            points = points[available]
            visible_indices = visible_indices[available]
            if not len(visible_indices):
                return None
        delta = points - query
        distances_squared = np.einsum("ij,ij->i", delta, delta)
        nearest_index = int(np.argmin(distances_squared))
        if distances_squared[nearest_index] > tolerance**2:
            return None
        return int(visible_indices[nearest_index])

    def set_edit_mode(
        self,
        mode: str | None,
        pan_with_right_button: bool = False,
        legacy_polygon_drawing: bool = False,
    ):
        """Active le dessin ou l'effacement libre du masque."""
        if (
            mode != self._edit_mode
            or legacy_polygon_drawing != self._legacy_polygon_drawing
        ) and self._edit_item is not None:
            self._scene.removeItem(self._edit_item)
            self._edit_item = None
        if (
            mode != self._edit_mode
            or legacy_polygon_drawing != self._legacy_polygon_drawing
        ):
            self._edit_points = []
        self._edit_mode = mode
        self._legacy_polygon_drawing = legacy_polygon_drawing
        enabled = mode is not None
        self._pan_with_right_button = enabled and pan_with_right_button
        self._pan_active = False
        self._pan_waiting_for_release = False
        self._pan_button = None
        self._drawing_active = False
        self._auto_scroll_timer.stop()
        if not enabled:
            tooltip = (
                "Molette : zoom | Glisser un point du contour rouge pour l'ajuster | "
                "Glisser ailleurs : déplacer l'image"
            )
        elif legacy_polygon_drawing:
            tooltip = (
                "Tracer une zone à main levée puis relâcher pour "
                f"{'l’effacer' if mode == 'crop' else 'l’ajouter'} "
                f"(trait de {MASK_ERASER_DIAMETER:g} px) | "
                "Clic droit + glisser : déplacer"
            )
        elif mode == "crop":
            tooltip = (
                "Tracer au clic gauche pour gommer uniquement le contour rouge "
                "(sans modifier le masque ni le volume) | "
                "Clic droit + glisser : déplacer"
            )
        else:
            tooltip = (
                "Relier les extrémités d'un contour effacé pour le restaurer, ou "
                "dessiner une forme fermée en relâchant près du départ."
            )
        self.setToolTip(tooltip)
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if enabled
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if not enabled and self._edit_item is not None:
            self._scene.removeItem(self._edit_item)
            self._edit_item = None
        if not enabled:
            self._edit_points = []

    def _update_edit_preview(self, close_path: bool = False):
        if not self._edit_points:
            return
        if self._edit_item is not None:
            if self._legacy_polygon_drawing:
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                path.closeSubpath()
                pen = QPen(Qt.GlobalColor.yellow, MASK_ERASER_DIAMETER)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                self._edit_item.setPen(pen)
            elif self._edit_mode == "crop":
                path = QPainterPath(self._edit_points[0])
                if len(self._edit_points) == 1:
                    radius = MASK_ERASER_DIAMETER / 2
                    point = self._edit_points[0]
                    path.addEllipse(
                        QRectF(
                            point.x() - radius,
                            point.y() - radius,
                            MASK_ERASER_DIAMETER,
                            MASK_ERASER_DIAMETER,
                        )
                    )
                    pen = QPen(Qt.GlobalColor.yellow, 1)
                    pen.setCosmetic(True)
                    self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                    self._edit_item.setPen(pen)
                else:
                    for point in self._edit_points[1:]:
                        path.lineTo(point)
                    pen = QPen(Qt.GlobalColor.yellow, MASK_ERASER_DIAMETER)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                    self._edit_item.setPen(pen)
            else:
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                if close_path:
                    path.closeSubpath()
            self._edit_item.setPath(path)

    def _auto_scroll_while_drawing(self):
        if (
            not self._drawing_active
            or not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        ):
            self._auto_scroll_timer.stop()
            return

        margin = 20
        speed = 8
        position = self._last_pointer_position
        viewport_rect = self.viewport().rect()
        horizontal_delta = (
            -speed
            if position.x() < margin
            else speed
            if position.x() >= viewport_rect.width() - margin
            else 0
        )
        vertical_delta = (
            -speed
            if position.y() < margin
            else speed
            if position.y() >= viewport_rect.height() - margin
            else 0
        )
        if horizontal_delta:
            scrollbar = self.horizontalScrollBar()
            scrollbar.setValue(scrollbar.value() + horizontal_delta)
        if vertical_delta:
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() + vertical_delta)
        if horizontal_delta or vertical_delta:
            current = self.mapToScene(position)
            if self._edit_points[-1] != current:
                self._edit_points.append(current)
                self._update_edit_preview()

    def mousePressEvent(self, event):
        if self._contour_edit_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_auto_added_control_point = None
            position = event.position().toPoint()
            marker_index = self._control_point_index_at(
                position,
                hit_radius=7.0
                if event.button() == Qt.MouseButton.RightButton
                else 10.0,
            )
            if (
                event.button() == Qt.MouseButton.RightButton
                and marker_index >= 0
            ):
                self.contour_control_point_removed.emit(marker_index)
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                scene_position = self.mapToScene(position)
                point_index = self._nearest_active_contour_point(
                    QPointF(*self._control_points[marker_index])
                    if marker_index >= 0
                    else scene_position,
                    tolerance=float("inf") if marker_index >= 0 else None,
                )
                if point_index is not None and self._active_contour is not None:
                    if marker_index < 0:
                        point = self._active_contour[point_index, 0]
                        control_point = (int(point[0]), int(point[1]))
                        was_control_point = (
                            control_point in self._control_points
                        )
                        self.contour_control_point_added.emit(
                            QPointF(float(point[0]), float(point[1]))
                        )
                        if control_point in self._control_points:
                            marker_index = self._control_points.index(
                                control_point
                            )
                            if not was_control_point:
                                self._last_auto_added_control_point = (
                                    control_point
                                )
                    point = self._active_contour[point_index, 0]
                    drag_origin = (
                        self._control_points[marker_index]
                        if marker_index >= 0
                        else (int(point[0]), int(point[1]))
                    )
                    self._contour_drag_origin_scene = QPointF(
                        float(drag_origin[0]),
                        float(drag_origin[1]),
                    )
                    self._contour_drag_origin_global = event.globalPosition()
                    transform = self.transform()
                    self._contour_drag_scale_x = max(
                        abs(transform.m11()),
                        1e-6,
                    )
                    self._contour_drag_scale_y = max(
                        abs(transform.m22()),
                        1e-6,
                    )
                    self._contour_drag_active = True
                    self._drag_marker_index = marker_index
                    self.contour_point_drag_started.emit(
                        self._active_contour.copy(),
                        point_index,
                        marker_index,
                    )
                else:
                    self.contour_region_selection_requested.emit(
                        scene_position
                    )
                event.accept()
                return
        if self._contour_edit_mode or (
            self._edit_mode is not None and self._pan_with_right_button
        ):
            pan_button = Qt.MouseButton.RightButton
        elif self._edit_mode is None:
            pan_button = Qt.MouseButton.LeftButton
        else:
            pan_button = None
        if pan_button is not None and event.button() == pan_button:
            if self._edit_item is not None:
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
            self._edit_points = []
            self._pan_active = True
            self._pan_waiting_for_release = True
            self._pan_button = pan_button
            self._drawing_active = False
            self._auto_scroll_timer.stop()
            self._pan_last_position = event.position().toPoint()
            event.accept()
            return
        if self._pan_waiting_for_release:
            event.accept()
            return
        if self._edit_mode and event.button() == Qt.MouseButton.LeftButton:
            current = self.mapToScene(event.position().toPoint())
            if self._edit_mode == "crop":
                self._edit_points = [current]
            elif not self._edit_points:
                self._edit_points = [current]
                path = QPainterPath(current)
                self._edit_item = QGraphicsPathItem(path)
                if self._legacy_polygon_drawing:
                    pen = QPen(
                        Qt.GlobalColor.yellow,
                        MASK_ERASER_DIAMETER,
                    )
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                else:
                    pen = QPen(Qt.GlobalColor.yellow, 3)
                    pen.setCosmetic(True)
                self._edit_item.setPen(pen)
                self._scene.addItem(self._edit_item)
            elif self._edit_points[-1] != current:
                self._edit_points.append(current)
                self._update_edit_preview()
            self._drawing_active = True
            self._last_pointer_position = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._contour_drag_active:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.contour_point_dragged.emit(
                    self._contour_scene_position(event.globalPosition())
                )
            event.accept()
            return
        if self._pan_active:
            if self._pan_button is not None and event.buttons() & self._pan_button:
                position = event.position().toPoint()
                delta = position - self._pan_last_position
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - delta.x()
                )
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - delta.y()
                )
                self._pan_last_position = position
            event.accept()
            return
        if self._pan_waiting_for_release:
            event.accept()
            return
        if (
            self._edit_mode
            and self._edit_points
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._drawing_active = True
            self._last_pointer_position = event.position().toPoint()
            current = self.mapToScene(self._last_pointer_position)
            if not self._edit_points[-1] == current:
                self._edit_points.append(current)
                if self._edit_mode == "crop" and self._edit_item is None:
                    self._edit_item = QGraphicsPathItem()
                    pen = QPen(
                        Qt.GlobalColor.yellow,
                        MASK_ERASER_DIAMETER,
                    )
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                    self._edit_item.setPen(pen)
                    self._scene.addItem(self._edit_item)
                self._update_edit_preview()
            margin = 20
            viewport_rect = self.viewport().rect()
            if (
                self._last_pointer_position.x() < margin
                or self._last_pointer_position.x() >= viewport_rect.width() - margin
                or self._last_pointer_position.y() < margin
                or self._last_pointer_position.y() >= viewport_rect.height() - margin
            ):
                if not self._auto_scroll_timer.isActive():
                    self._auto_scroll_timer.start()
            else:
                self._auto_scroll_timer.stop()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._contour_drag_active
            and event.button() == Qt.MouseButton.LeftButton
        ):
            if (
                self._contour_drag_origin_global is not None
                and event.globalPosition()
                != self._contour_drag_origin_global
            ):
                self.contour_point_dragged.emit(
                    self._contour_scene_position(event.globalPosition())
                )
            self.contour_point_drag_finished.emit()
            self._contour_drag_active = False
            self._drag_marker_index = -1
            self._contour_drag_origin_scene = None
            self._contour_drag_origin_global = None
            event.accept()
            return
        if self._pan_active:
            if (
                event.button() == self._pan_button
                or event.buttons() == Qt.MouseButton.NoButton
            ):
                self._pan_active = False
                self._pan_waiting_for_release = False
                self._pan_button = None
                self._pan_last_position = QPoint()
            event.accept()
            return
        if self._pan_waiting_for_release:
            if event.buttons() == Qt.MouseButton.NoButton:
                self._pan_waiting_for_release = False
            event.accept()
            return
        if (
            self._edit_mode
            and self._edit_points
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drawing_active = False
            self._auto_scroll_timer.stop()
            release_position = event.position().toPoint()
            start_position = self.mapFromScene(self._edit_points[0])
            release_point = self.mapToScene(release_position)
            if self._edit_points[-1] != release_point:
                self._edit_points.append(release_point)
            if self._legacy_polygon_drawing:
                self._update_edit_preview(close_path=True)
                path = (
                    self._edit_item.path()
                    if self._edit_item is not None
                    else QPainterPath()
                )
                if self._edit_item is not None:
                    self._scene.removeItem(self._edit_item)
                    self._edit_item = None
                valid_polygon = len(self._edit_points) >= 3
                self._edit_points = []
                if valid_polygon:
                    self.edit_path_finished.emit(path, True)
                event.accept()
                return
            if self._edit_mode == "crop":
                self._update_edit_preview()
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                if self._edit_item is not None:
                    self._scene.removeItem(self._edit_item)
                    self._edit_item = None
                self._edit_points = []
                self.edit_path_finished.emit(path, False)
                event.accept()
                return
            close_delta = release_position - start_position
            closes_path = (
                len(self._edit_points) >= 3
                and close_delta.x() ** 2 + close_delta.y() ** 2 <= 16
            )

            if closes_path:
                self._update_edit_preview(close_path=True)
                if self._edit_item is None:
                    event.accept()
                    return
                path = self._edit_item.path()
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
                self._edit_points = []
                self.edit_path_finished.emit(path, True)
            elif self._edit_mode == "add" and self._edit_item is not None:
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
                self._edit_points = []
                self.edit_path_finished.emit(path, False)
            elif self._edit_item is not None:
                self._update_edit_preview()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit_image(self):
        """Ajuste l'image entière à la vue sans créer d'image rééchantillonnée."""
        if not self._has_image:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._user_zoomed = False

    def wheelEvent(self, event):
        if not self._has_image:
            super().wheelEvent(event)
            return
        factor = 1.20 if event.angleDelta().y() > 0 else 1.0 / 1.20
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        # Évite des niveaux de zoom trop petits ou trop élevés.
        if 0.03 <= new_scale <= 50.0:
            self.scale(factor, factor)
            self._user_zoomed = True
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if (
            self._contour_edit_mode
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_position = self.mapToScene(event.position().toPoint())
            duplicate_press_point = False
            if (
                self._active_contour is not None
                and self._last_auto_added_control_point is not None
            ):
                point_index = self._nearest_active_contour_point(
                    scene_position,
                    include_erased=True,
                )
                if point_index is not None:
                    point = self._active_contour[point_index, 0]
                    duplicate_press_point = (
                        int(point[0]),
                        int(point[1]),
                    ) == self._last_auto_added_control_point
            if duplicate_press_point:
                self._last_auto_added_control_point = None
            else:
                self.contour_control_point_added.emit(scene_position)
                self._last_auto_added_control_point = None
            event.accept()
            return
        self.fit_image()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Conserve l'image ajustée tant que l'utilisateur ne zoome pas manuellement.
        if self._has_image and not self._user_zoomed:
            self.fit_image()


class MainWindow(QWidget):
    """Fenêtre principale : sélection du dossier et liste des datasets."""

    def __init__(self):
        super().__init__()
        

        self.setWindowTitle("MRI Dataset Browser")
        self.resize(700, 500)
        self.setAcceptDrops(True)
        # Icone du logo dans la barre de titre et sur la barre des tâches
        self.icon = QIcon(str(LOGO_PATH))
        self.setWindowIcon(self.icon)

        # Conserve les références vers toutes les fenêtres de visualisation ouvertes.
        self.viewers = []
        self.dataset_paths = []
        self.rois_saving_dir_path = None
        self.model_checkpoint = None

        layout = QVBoxLayout(self)

        self.browse_button = QPushButton("Parcourir un dossier de datasets")
        self.browse_button.clicked.connect(self.browse_folder)
        layout.addWidget(self.browse_button)

        self.model_directory_button = QPushButton("Spécifier le dossier des modèles")
        self.model_directory_button.clicked.connect(self.choose_model_directory)
        layout.addWidget(self.model_directory_button)
        self.model_warning_label = QLabel()
        self.model_warning_label.setWordWrap(True)
        self.model_warning_label.setStyleSheet(
            "color: #9c0006; background-color: #ffc7ce; padding: 6px;"
        )
        layout.addWidget(self.model_warning_label)

        self.model_directory_label = QLabel()
        self.model_directory_label.setWordWrap(True)
        self._refresh_model_directory_label()
        layout.addWidget(self.model_directory_label)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        layout.addWidget(self.folder_label)

        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(
            lambda item: self.open_selected_file()
        )
        layout.addWidget(self.file_list)

        self.open_button = QPushButton("Ouvrir le dataset sélectionné")
        self.open_button.clicked.connect(self.open_selected_file)
        layout.addWidget(self.open_button)
        
        self.open_button_2 = QPushButton("Ouvrir le dataset sélectionné + ROI")
        self.open_button_2.clicked.connect(self.open_selected_file_with_roi)
        layout.addWidget(self.open_button_2)

        self.status_label = QLabel(
            "Select or drop a folder containing .PvDatasets files."
        )
        layout.addWidget(self.status_label)

    def browse_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select a dataset folder",
        )

        if dir_path:
            self.load_folder(dir_path)

    def _refresh_model_directory_label(self):
        model_directory = saved_model_directory()
        if model_directory is None:
            self.model_warning_label.setText(
                "⚠ Aucun dossier de modèle n'est configuré. "
                "Spécifiez un dossier contenant au moins un fichier .ckpt "
                "avant d'ouvrir un dataset."
            )
            self.model_warning_label.show()
            self.model_directory_label.setText(
                "Dossier des modèles : non configuré"
            )
        else:
            self.model_warning_label.clear()
            self.model_warning_label.hide()
            self.model_directory_label.setText(
                f"Dossier des modèles : {model_directory}"
            )

    def choose_model_directory(self) -> pathlib.Path | None:
        """Choisit et enregistre le dossier contenant les checkpoints."""
        current_directory = saved_model_directory()
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "Choisir le dossier contenant les modèles",
            str(current_directory or PROJECT_ROOT),
        )
        if not selected_directory:
            return None

        model_directory = pathlib.Path(selected_directory)
        try:
            available_checkpoints(model_directory)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Modèles introuvables", str(exc))
            return None

        save_model_directory(model_directory)
        self._refresh_model_directory_label()
        self.status_label.setText("Dossier des modèles enregistré.")
        return model_directory

    def choose_study_model(self, model_directory: pathlib.Path) -> pathlib.Path | None:
        """Ouvre une fenêtre listant les dossiers modèles puis leurs checkpoints."""
        model_directory = pathlib.Path(model_directory)
        try:
            checkpoints = available_checkpoints(model_directory)
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Modèles introuvables", str(exc))
            return None

        dialog = QDialog(self)
        dialog.setWindowTitle("Choisir le modèle pour l'étude")
        dialog.setMinimumSize(650, 430)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Dossiers modèles disponibles"))
        folder_list = QListWidget()
        for folder in sorted(
            {
                checkpoint.parent
                for checkpoint in checkpoints
            },
            key=lambda path: str(path).casefold(),
        ):
            label = (
                str(folder.relative_to(model_directory))
                if folder != model_directory
                else model_directory.name
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(folder))
            folder_list.addItem(item)
        layout.addWidget(folder_list)

        layout.addWidget(QLabel("Modèles disponibles dans le dossier sélectionné"))
        checkpoint_list = QListWidget()
        layout.addWidget(checkpoint_list)

        def refresh_checkpoints(folder: pathlib.Path) -> None:
            checkpoint_list.clear()
            folder_checkpoints = sorted(
                (
                    checkpoint
                    for checkpoint in checkpoints
                    if checkpoint.parent == folder
                ),
                key=lambda path: path.name.casefold(),
            )
            for checkpoint in folder_checkpoints:
                item = QListWidgetItem(checkpoint.name)
                item.setData(Qt.ItemDataRole.UserRole, str(checkpoint))
                checkpoint_list.addItem(item)
            if checkpoint_list.count():
                checkpoint_list.setCurrentRow(0)

        def folder_selected(item: QListWidgetItem) -> None:
            refresh_checkpoints(pathlib.Path(item.data(Qt.ItemDataRole.UserRole)))

        folder_list.itemSelectionChanged.connect(
            lambda: folder_selected(folder_list.currentItem())
            if folder_list.currentItem() is not None
            else None
        )
        folder_list.itemDoubleClicked.connect(folder_selected)
        if folder_list.count():
            folder_list.setCurrentRow(0)
            refresh_checkpoints(
                pathlib.Path(
                    folder_list.item(0).data(Qt.ItemDataRole.UserRole)
                )
            )

        selected_model: list[pathlib.Path | None] = [None]
        buttons_layout = QHBoxLayout()
        cancel_button = QPushButton("Annuler")
        validate_button = QPushButton("Utiliser ce modèle")
        validate_button.setEnabled(checkpoint_list.count() > 0)

        def validate_selection() -> None:
            item = checkpoint_list.currentItem()
            if item is None:
                return
            selected_model[0] = pathlib.Path(
                item.data(Qt.ItemDataRole.UserRole)
            )
            dialog.accept()

        def update_validation_state() -> None:
            validate_button.setEnabled(checkpoint_list.currentItem() is not None)

        cancel_button.clicked.connect(dialog.reject)
        validate_button.clicked.connect(validate_selection)
        checkpoint_list.itemSelectionChanged.connect(update_validation_state)
        checkpoint_list.itemDoubleClicked.connect(
            lambda _item: validate_selection()
        )
        buttons_layout.addWidget(cancel_button)
        buttons_layout.addWidget(validate_button)
        layout.addLayout(buttons_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return selected_model[0]
        return None
    
    

    def load_folder(self, dir_path: str):
        """Remplit la liste avec les datasets du dossier sélectionné."""
        files = find_dataset_files(dir_path)

        self.file_list.clear()
        self.dataset_paths = []
        self.folder_label.setText(dir_path)

        if not files:
            self.status_label.setText(
                "No .PvDatasets files found."
            )
            QMessageBox.warning(
                self,
                "No datasets",
                "The selected folder contains no .PvDatasets files.",
            )
            return

        self.dataset_paths = files

        for file_path in files:
            self.file_list.addItem(os.path.basename(file_path))

        self.file_list.setCurrentRow(0)
        self.status_label.setText(
            f"{len(files)} dataset(s) found. Double-click to open."
        )
        
        self.open_study_analysis_dialog()

    def open_study_analysis_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Analyse de l'étude")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setMinimumSize(520, 180)
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        label = QLabel("Voulez-vous analyser l'étude entière ?")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        yes_button = QPushButton("Oui")
        no_button = QPushButton("Non")
        yes_button.clicked.connect(lambda: self.open_rois_saving_dialog(dialog))
        no_button.clicked.connect(dialog.reject)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(yes_button)
        buttons_layout.addWidget(no_button)
        layout.addLayout(buttons_layout)
        dialog.exec()

    def open_rois_saving_dialog(self, dialog):
        dialog.accept()
        QTimer.singleShot(0, self._choose_rois_saving_directory)

    def _choose_rois_saving_directory(self):
        rois_saving_dir_path = QFileDialog.getExistingDirectory(
            self,
            "Enregistrement des ROI",
            os.getcwd(),
        )
        if not rois_saving_dir_path:
            return
        self.rois_saving_dir_path = rois_saving_dir_path
        self.analyze_study()

    def write_prediction_error_report(
        self,
        errors: list[tuple[str, str]],
        empty_predictions: list[tuple[str, str]] | None = None,
    ) -> pathlib.Path:
        """Écrit la liste des fichiers non prédits dans le dossier de sortie."""
        output_directory = pathlib.Path(self.rois_saving_dir_path)
        output_directory.mkdir(parents=True, exist_ok=True)
        report_path = output_directory / "erreurs_prediction.txt"
        empty_predictions = empty_predictions or []
        lines = [
            "Rapport de prediction",
            f"Date: {QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}",
            f"Nombre d'erreurs: {len(errors)}",
            f"Nombre sans contour predit: {len(empty_predictions)}",
            "",
            "Erreurs:",
        ]
        if errors:
            lines.extend(
                f"{file_path} | {reason}"
                for file_path, reason in errors
            )
        else:
            lines.append("Aucune erreur.")
        lines.extend(["", "Sans contour predit:"])
        if empty_predictions:
            lines.extend(
                f"{file_path} | {reason}"
                for file_path, reason in empty_predictions
            )
        else:
            lines.append("Aucune prediction sans contour.")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return report_path

    def analyze_study(self):
        if not self.dataset_paths:
            QMessageBox.warning(self, "Dossier vide", "Choisissez un dossier d'une étude avant de lancer l'analyse")
            return
        if not self.rois_saving_dir_path:
            QMessageBox.warning(self, "Dossier non sélectionné", "Choisissez un dossier pour enregistrer les ROI avant de lancer l'analyse")
            return
        model_directory = saved_model_directory()
        if model_directory is None:
            model_directory = self.choose_model_directory()
            if model_directory is None:
                return
        selected_model = self.choose_study_model(model_directory)
        if selected_model is None:
            QMessageBox.information(
                self,
                "Modèle non sélectionné",
                "Aucune analyse n'a été lancée.",
            )
            return
        self.model_checkpoint = selected_model
        if not self.model_checkpoint.is_file():
            QMessageBox.warning(
                self,
                "Modèle introuvable",
                "Le fichier sélectionné n'existe pas.",
            )
            return
        
        prediction_errors: list[tuple[str, str]] = []
        empty_predictions: list[tuple[str, str]] = []
        roi_generated_count = 0
        empty_prediction_count = 0
        self.predictor = SlicePredictor()
        progress = QDialog(self)
        progress.setWindowTitle("Chargement")
        progress.setModal(True)
        progress_layout = QVBoxLayout(progress)
        progress_label = QLabel("Analyse de l'étude en cours...")
        progress_label.setWordWrap(True)
        progress_layout.addWidget(progress_label)
        progress_bar = QProgressBar(progress)
        progress_bar.setRange(0, len(self.dataset_paths))
        progress_bar.setValue(0)
        progress_bar.setFormat("%v / %m fichiers")
        progress_layout.addWidget(progress_bar)
        error_count_label = QLabel("Erreurs : 0")
        progress_layout.addWidget(error_count_label)
        empty_prediction_label = QLabel("Sans contour prédit : 0")
        progress_layout.addWidget(empty_prediction_label)
        roi_count_label = QLabel("ROI générés : 0")
        progress_layout.addWidget(roi_count_label)
        progress.setMinimumSize(520, 220)
        progress.resize(520, 220)
        progress.open()
        progress.raise_()
        progress.activateWindow()
        QApplication.processEvents()

        def update_error_count() -> None:
            error_count_label.setText(f"Erreurs : {len(prediction_errors)}")
            QApplication.processEvents()

        def update_success_counts() -> None:
            empty_prediction_label.setText(
                f"Sans contour prédit : {empty_prediction_count}"
            )
            roi_count_label.setText(f"ROI générés : {roi_generated_count}")
            QApplication.processEvents()

        try:
            for index, file_path in enumerate(self.dataset_paths):
                progress_label.setText(
                    f"Analyse {index + 1}/{len(self.dataset_paths)} : "
                    f"{os.path.basename(file_path)}"
                )
                QApplication.processEvents()

                try:
                    _, _, all_scans = load_data(file_path)
                    scan_id = model_scan_id(
                        all_scans,
                        self.model_checkpoint,
                    )
                    model_volume = convert_pvdataset_to_npy(file_path, scan_id)
                    model_mask = self.predictor.predict(
                        model_volume,
                        self.model_checkpoint,
                    )
                except Exception as exc:
                    prediction_errors.append(
                        (file_path, f"Erreur de chargement ou de prédiction: {exc}")
                    )
                    progress_bar.setValue(progress_bar.value() + 1)
                    update_error_count()
                    QApplication.processEvents()
                    continue

                try:
                    progress_label.setText(
                        f"Enregistrement des ROI {index + 1}/{len(self.dataset_paths)}"
                    )
                    QApplication.processEvents()
                    save_prediction_rois(
                        model_mask,
                        os.path.join(
                            self.rois_saving_dir_path,
                            os.path.basename(file_path),
                        ),
                    )
                    roi_generated_count += 1
                    update_success_counts()
                except Exception as exc:
                    if "Aucun contour prédit" in str(exc):
                        empty_prediction_count += 1
                        empty_predictions.append(
                            (file_path, "Aucun contour prédit")
                        )
                    else:
                        prediction_errors.append(
                            (file_path, f"Erreur d'enregistrement des ROI: {exc}")
                        )
                    update_error_count()
                    update_success_counts()

                progress_bar.setValue(progress_bar.value() + 1)
                QApplication.processEvents()

            self.warning_files = []
            self.write_prediction_error_report(
                prediction_errors,
                empty_predictions,
            )
        finally:
            progress.close()

    def open_warnings(self):
        dialog = QDialog(self)
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        self.warning_file_list = QListWidget()
        for file in self.warning_files:
            self.warning_file_list.addItem(os.path.basename(file))
        self.warning_file_list.setCurrentRow(0)
        self.warning_file_list.itemDoubleClicked.connect(lambda item: self.open_selected_file_and_roi(item))
        layout.addWidget(self.warning_file_list)
        dialog.show()

    def open_selected_file_and_roi(self, item):
        row = self.warning_file_list.row(item)
        file = self.warning_files[row]
        self.open_selected_file_with_roi(file)

    def _resolve_roi_zip(
        self,
        file_path: str | pathlib.Path,
        roi_directory: pathlib.Path,
    ) -> pathlib.Path | None:
        """Associe un PvDataset à son ZIP ROI dans le dossier choisi."""
        dataset_name = pathlib.Path(file_path).name
        dataset_stem = pathlib.Path(dataset_name).stem
        candidates = (
            roi_directory / f"{dataset_stem}.zip",
            roi_directory / f"{dataset_name}.zip",
            roi_directory / dataset_name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        for zip_path in sorted(
            roi_directory.rglob("*.zip"),
            key=lambda path: path.name.casefold(),
        ):
            if zip_path.stem.casefold() == dataset_stem.casefold():
                return zip_path
        return None

    def open_selected_file_with_roi(self, file_path: str | pathlib.Path | None = None):
        """Ouvre le dataset sélectionné avec son fichier ROI déjà prédit."""
        # QPushButton.clicked emits a bool; for the button, ignore it and use
        # the currently selected row.
        if isinstance(file_path, bool):
            file_path = None
        if file_path is None:
            row = self.file_list.currentRow()
            if row < 0 or row >= len(self.dataset_paths):
                QMessageBox.information(
                    self,
                    "No selection",
                    "Please select a dataset first.",
                )
                return
            file_path = self.dataset_paths[row]

        roi_directory = QFileDialog.getExistingDirectory(
            self,
            "Choisir le dossier contenant les ROI prédites",
            str(self.rois_saving_dir_path or os.getcwd()),
        )
        if not roi_directory:
            return
        roi_directory = pathlib.Path(roi_directory)
        roi_zip_path = self._resolve_roi_zip(file_path, roi_directory)
        if roi_zip_path is None:
            QMessageBox.warning(
                self,
                "ROI introuvables",
                "Aucun fichier ZIP ROI correspondant n'a été trouvé pour le dataset sélectionné.",
            )
            return

        try:
            model_directory = saved_model_directory() or PROJECT_ROOT

            viewer = MRIViewer(
                file_path,
                model_directory,
                self.dataset_paths,
                roi_zip_path=roi_zip_path,
                roi_directory=roi_directory,
            )
            self.viewers.append(viewer)
            viewer.destroyed.connect(
                lambda obj=None, window=viewer:
                    self._remove_viewer(window)
            )
            viewer.show()
            viewer.raise_()
            viewer.activateWindow()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Erreur d'ouverture",
                f"Impossible d'ouvrir le dataset avec ses ROI :\n{exc}",
            )

    def open_selected_file(self):
        """Ouvre le dataset sélectionné dans une fenêtre séparée."""
        row = self.file_list.currentRow()

        if row < 0 or row >= len(self.dataset_paths):
            QMessageBox.information(
                self,
                "No selection",
                "Please select a dataset first.",
            )
            return

        file_path = self.dataset_paths[row]

        try:
            model_directory = saved_model_directory()
            if model_directory is None:
                model_directory = self.choose_model_directory()
                if model_directory is None:
                    return

            # Sans parent : il s'agit d'une fenêtre principale indépendante.
            viewer = MRIViewer(file_path, model_directory, self.dataset_paths)
            self.viewers.append(viewer)

            # Supprime la référence lorsque la fenêtre est détruite.
            viewer.destroyed.connect(
                lambda obj=None, window=viewer:
                    self._remove_viewer(window)
            )

            viewer.show()
            viewer.raise_()
            viewer.activateWindow()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Viewer error",
                f"Could not open the MRI viewer:\n{exc}",
            )

    def _remove_viewer(self, viewer):
        if viewer in self.viewers:
            self.viewers.remove(viewer)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()

        if not urls:
            event.ignore()
            return

        dir_path = urls[0].toLocalFile()

        if os.path.isdir(dir_path):
            self.load_folder(dir_path)
            event.acceptProposedAction()
        else:
            QMessageBox.warning(
                self,
                "Invalid selection",
                "Please drop a folder, not an individual file.",
            )
            event.ignore()


class MRIViewer(QWidget):
    """Fenêtre indépendante pour parcourir les coupes IRM."""

    def __init__(
        self,
        file_path: str,
        model_directory: pathlib.Path,
        dataset_paths: list[str] | None = None,
        roi_zip_path: pathlib.Path | None = None,
        roi_directory: pathlib.Path | None = None,
    ):
        super().__init__()

        self.file_path = file_path
        self.model_directory = model_directory
        self.roi_zip_path = roi_zip_path
        self.roi_directory = roi_directory
        self.roi_only = roi_zip_path is not None
        self.dataset_paths = dataset_paths or [file_path]
        self.file_index = next(
            (
                index
                for index, dataset_path in enumerate(self.dataset_paths)
                if dataset_path == file_path
            ),
            0,
        )
        self.volume = None
        self.prediction_masks = None
        self.original_prediction_masks = None
        self.prediction_outline_masks = None
        self.erased_outline_masks = None
        self.all_scans = {}
        self.scan_ids = []
        self.predictor = SlicePredictor()
        self.model_paths = []
        self.mask_edit_mode = None
        self.mask_drawing_method = (
            "guiv2" if saved_mask_drawing_method() else "gui_v4"
        )
        self.contour_edit_active = False
        self._active_contour_slice: int | None = None
        self._active_contour: np.ndarray | None = None
        self._active_contour_component: np.ndarray | None = None
        self._active_contour_seed: QPointF | None = None
        self._active_control_points: list[tuple[int, int]] = []
        self._control_points_by_slice: dict[int, list[tuple[int, int]]] = {}
        self._contour_edit_state: ContourPointEdit | None = None
        self.mask_history = []
        self.outline_history = []
        self.erased_outline_history = []
        self.mask_history_index = -1
        self.prediction_cache = {}
        self.current_prediction_key = None

        self.setWindowTitle(
            f"MRI Viewer - {os.path.basename(file_path)}"
        )
        self.setWindowIcon(QIcon(str(LOGO_PATH)))
        
        self.resize(1100, 750)

        main_layout = QVBoxLayout(self)

        etaplab_logo = QLabel()
        etaplab_pixmap = QPixmap(str(ETAPLAB_LOGO_PATH))
        if not etaplab_pixmap.isNull():
            etaplab_logo.setPixmap(
                etaplab_pixmap.scaledToHeight(
                    38,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            etaplab_logo.setToolTip("ETAP Lab")
        else:
            LOGGER.warning("Logo ETAP Lab introuvable: %s", ETAPLAB_LOGO_PATH)

        toolbar = QVBoxLayout()
        toolbar.setSpacing(4)

        data_group = QGroupBox("Données")
        data_layout = QHBoxLayout(data_group)
        data_layout.setContentsMargins(6, 4, 6, 4)
        data_layout.setSpacing(4)
        self.analysis_combo = QComboBox()
        self.analysis_combo.setToolTip("Choisir le type d'analyse présent dans le PvDataset")
        self.analysis_combo.setFixedWidth(210)
        self.analysis_combo.setFixedHeight(30)
        self.analysis_combo.currentIndexChanged.connect(self.change_scan)
        data_layout.addWidget(self.analysis_combo)

        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Choisir le checkpoint du modèle")
        self.model_combo.setFixedWidth(210)
        self.model_combo.setFixedHeight(30)
        self.model_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_result_list()
        )
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        data_layout.addWidget(self.model_combo)
        if self.roi_only:
            self.model_combo.hide()

        prediction_group = QGroupBox(
            "ROI existantes" if self.roi_only else "Prédiction et export"
        )
        prediction_layout = QHBoxLayout(prediction_group)
        prediction_layout.setContentsMargins(6, 4, 6, 4)
        prediction_layout.setSpacing(4)
        self.analyze_button = QPushButton("Analyser toutes les coupes")
        self.analyze_button.setFixedWidth(190)
        self.analyze_button.setFixedHeight(30)
        self.analyze_button.setStyleSheet(
            """
            QPushButton {
                color: palette(button-text);
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QPushButton[predictionReady="true"] {
                background-color: #35E86B;
                color: #111111;
                border: 1px solid #35E86B;
            }
            QPushButton:disabled {
                color: #888888;
                background-color: #d6d6d6;
                border: 1px solid #c2c2c2;
            }
            """
        )
        self.analyze_button.clicked.connect(self.analyze_volume)
        prediction_layout.addWidget(self.analyze_button)
        if self.roi_only:
            self.analyze_button.hide()

        self.save_roi_button = QPushButton("Sauvegarder les ROI")
        if self.roi_only:
            self.save_roi_button.setText("Sauvegarder les modifications ROI")
            self.save_roi_button.setFixedWidth(260)
        else:
            self.save_roi_button.setFixedWidth(150)
        self.save_roi_button.setToolTip(
            "Exporter les contours prédits dans un fichier ROI ZIP"
        )
        self.save_roi_button.setFixedHeight(30)
        self.save_roi_button.setEnabled(False)
        self.save_roi_button.clicked.connect(self.save_prediction_as_rois)
        prediction_layout.addWidget(self.save_roi_button)

        top_controls_layout = QHBoxLayout()
        top_controls_layout.setSpacing(4)
        top_controls_layout.addWidget(data_group)
        top_controls_layout.addWidget(prediction_group)
        top_controls_layout.addStretch()
        top_controls_layout.addWidget(
            etaplab_logo,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )
        toolbar.addLayout(top_controls_layout)

        edit_group = QGroupBox("Édition du masque")
        edit_layout = QHBoxLayout(edit_group)
        edit_layout.setContentsMargins(6, 4, 6, 4)
        edit_layout.setSpacing(4)

        self.legacy_drawing_checkbox = QCheckBox("Mode ImageJ")
        self.legacy_drawing_checkbox.setChecked(
            self.mask_drawing_method == "guiv2"
        )
        self.legacy_drawing_checkbox.setToolTip(
            "Active le dessin libre de guiv2.py : tracer une zone à main levée "
            "puis la remplir ou l'effacer au relâchement. L'aperçu garde "
            "l'épaisseur de trait de gui_v4.py."
        )
        self.legacy_drawing_checkbox.setFixedHeight(30)
        self.legacy_drawing_checkbox.setFixedWidth(
            self.legacy_drawing_checkbox.sizeHint().width()
        )
        self.legacy_drawing_checkbox.toggled.connect(
            self.set_mask_drawing_method
        )
        edit_layout.addWidget(self.legacy_drawing_checkbox)
      
        self.add_mask_button = QPushButton("Ajouter une zone")
        self.add_mask_button.setToolTip(
            "Dessinez une zone fermée qui rejoint les extrémités du contour effacé. "
            "Le masque et son volume seront recalculés."
        )
        self.add_mask_button.setEnabled(False)
        self.add_mask_button.setCheckable(True)
        self.add_mask_button.setAttribute(
            Qt.WidgetAttribute.WA_MacShowFocusRect, False
        )
        self.add_mask_button.setStyleSheet(
            """
            QPushButton {
                color: palette(button-text);
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QPushButton:pressed:!checked {
                background-color: palette(button);
                border-color: palette(dark);
                border-style: inset;
            }
            QPushButton:checked,
            QPushButton:checked:hover,
            QPushButton:checked:pressed,
            QPushButton:checked:focus {
                background-color: #35E86B;
                color: #111111;
                border: 1px solid #35E86B;
                border-radius: 6px;
            }
            QPushButton:disabled {
                color: #888888;
                background-color: #d6d6d6;
                border: 1px solid #c2c2c2;
            }
            """
        )
        self.add_mask_button.setFixedWidth(140)
        self.add_mask_button.setFixedHeight(30)
        self.add_mask_button.clicked.connect(
            lambda checked: self.set_mask_edit_mode("add" if checked else None)
        )
        edit_layout.addWidget(self.add_mask_button)

        self.crop_mask_button = QPushButton("Gommer une zone")
        self.crop_mask_button.setToolTip(
            "Effacer uniquement le contour rouge, sans modifier le masque ni le volume. "
            "Le mode reste actif pour les autres coupes ; une coupe déjà vide n'a "
            "plus de contour à gommer. Recliquez ici ou appuyez sur Q pour quitter."
        )
        self.crop_mask_button.setEnabled(False)
        self.crop_mask_button.setCheckable(True)
        self.crop_mask_button.setAttribute(
            Qt.WidgetAttribute.WA_MacShowFocusRect, False
        )
        self.crop_mask_button.setStyleSheet(
            """
            QPushButton {
                color: palette(button-text);
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QPushButton:pressed:!checked {
                background-color: palette(button);
                border-color: palette(dark);
                border-style: inset;
            }
            QPushButton:checked,
            QPushButton:checked:hover,
            QPushButton:checked:pressed,
            QPushButton:checked:focus {
                background-color: #35E86B;
                color: #111111;
                border: 1px solid #35E86B;
                border-radius: 6px;
            }
            QPushButton:disabled {
                color: #888888;
                background-color: #d6d6d6;
                border: 1px solid #c2c2c2;
            }
            """
        )
        self.crop_mask_button.setFixedWidth(150)
        self.crop_mask_button.setFixedHeight(30)
        self.crop_mask_button.clicked.connect(
            lambda checked: self.set_mask_edit_mode("crop" if checked else None)
        )
        edit_layout.addWidget(self.crop_mask_button)

        self.edit_zone_button = QPushButton("Éditer la zone")
        self.edit_zone_button.setToolTip(
            "Activez le mode, cliquez sur une zone, puis déplacez ses points. "
            "Double-clic : ajouter un point | Clic droit sur un point : le supprimer | "
            "Q : quitter le mode."
        )
        self.edit_zone_button.setEnabled(False)
        self.edit_zone_button.setCheckable(True)
        self.edit_zone_button.setFixedWidth(125)
        self.edit_zone_button.setFixedHeight(30)
        self.edit_zone_button.setStyleSheet(
            """
            QPushButton:checked {
                background-color: #35E86B;
                color: #111111;
                border: 1px solid #35E86B;
                border-radius: 6px;
            }
            QPushButton:disabled {
                color: #888888;
                background-color: #d6d6d6;
                border: 1px solid #c2c2c2;
            }
            """
        )
        self.edit_zone_button.toggled.connect(self.set_contour_edit_mode)
        edit_layout.addWidget(self.edit_zone_button)

        self.undo_button = QPushButton("⮐")
        self.undo_button.setToolTip("Annuler la dernière modification")
        self.undo_button.setEnabled(False)
        self.undo_button.setFixedWidth(36)
        self.undo_button.setFixedHeight(30)
        self.undo_button.clicked.connect(self.undo_mask_edit)
        edit_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("⮑")
        self.redo_button.setToolTip("Rétablir la modification annulée")
        self.redo_button.setEnabled(False)
        self.redo_button.setFixedWidth(36)
        self.redo_button.setFixedHeight(30)
        self.redo_button.clicked.connect(self.redo_mask_edit)
        edit_layout.addWidget(self.redo_button)

        self.reset_mask_button = QPushButton("Réinitialiser les modifications")
        self.reset_mask_button.setEnabled(False)
        self.reset_mask_button.setFixedWidth(215)
        self.reset_mask_button.setFixedHeight(30)
        self.reset_mask_button.clicked.connect(self.reset_mask_edits)
        edit_layout.addWidget(self.reset_mask_button)
        toolbar.addWidget(edit_group)

        main_layout.addLayout(toolbar)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.image_label = MRIImageView()
        self.image_label.edit_path_finished.connect(
            self.apply_mask_edit_path
        )
        self.image_label.contour_region_selection_requested.connect(
            self.select_contour_for_edit
        )
        self.image_label.contour_control_point_added.connect(
            self.add_contour_control_point
        )
        self.image_label.contour_control_point_removed.connect(
            self.remove_contour_control_point
        )
        self.image_label.contour_point_drag_started.connect(
            self._begin_contour_point_edit
        )
        self.image_label.contour_point_dragged.connect(
            self._preview_contour_point_edit
        )
        self.image_label.contour_point_drag_finished.connect(
            self._finish_contour_point_edit
        )
        image_layout.addWidget(self.image_label, stretch=1)

        self.slice_label = QLabel()
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.slice_label)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self.show_slice)
        image_layout.addWidget(self.slice_slider)

        viewer_controls_layout = QHBoxLayout()
        viewer_controls_layout.addStretch()
        self.fit_button = QPushButton("Réinitialiser le zoom")
        self.fit_button.setFixedWidth(150)
        self.fit_button.setFixedHeight(30)
        self.fit_button.setToolTip("Fit the entire MRI slice in the viewer")
        self.fit_button.clicked.connect(self.image_fit)
        viewer_controls_layout.addWidget(self.fit_button)
        image_layout.addLayout(viewer_controls_layout)

        # Le viewer et les résultats occupent la même ligne.
        content_layout = QHBoxLayout()
        self.previous_button = QPushButton("<")
        self.previous_button.setFixedWidth(50)
        self.previous_button.setFixedHeight(30)

        self.previous_button.setToolTip("Coupe précédente")
        self.previous_button.clicked.connect(self.show_previous_slice)
        content_layout.addWidget(self.previous_button)
        content_layout.addWidget(image_panel, stretch=19)
        self.next_button = QPushButton(">")
        self.next_button.setFixedWidth(50)
        self.next_button.setFixedHeight(30)

        self.next_button.setToolTip("Coupe suivante")
        self.next_button.clicked.connect(self.show_next_slice)
        content_layout.addWidget(self.next_button)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        self.result_label = QLabel("Résultat de la prédiction")
        self.result_volume_label = QLabel("Volume total : -")
        self.result_volume_label.setStyleSheet("color: #35E86B;")
        file_navigation_layout = QHBoxLayout()
        self.previous_file_button = QPushButton("Sujet précédent")
        self.previous_file_button.setFixedWidth(125)
        self.previous_file_button.setFixedHeight(30)

        self.previous_file_button.clicked.connect(
            lambda: self.navigate_file(-1)
        )
        self.next_file_button = QPushButton("Sujet suivant")
        self.next_file_button.setFixedWidth(100)
        self.next_file_button.setFixedHeight(30)

        self.next_file_button.clicked.connect(
            lambda: self.navigate_file(1)
        )
        file_navigation_layout.addWidget(self.previous_file_button)
        file_navigation_layout.addWidget(self.next_file_button)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(
            ["Coupe", "Voxels", "Aire (mm²)"]
        )
        self.result_table.resizeColumnsToContents()
        self.result_table.setMinimumWidth(0)
        self.result_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.result_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.result_table.currentCellChanged.connect(
            lambda row, *_: self._select_result_slice(row)
        )
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_volume_label)
        result_layout.addLayout(file_navigation_layout)
        result_layout.addWidget(self.result_table)
        result_layout.addStretch()
        display_controls = QGroupBox("Réglages d'affichage")
        display_controls_layout = QVBoxLayout(display_controls)
        self.contrast_label = QLabel("Contraste : 100 %")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._update_display_controls)
        display_controls_layout.addWidget(self.contrast_label)
        display_controls_layout.addWidget(self.contrast_slider)

        self.intensity_label = QLabel("Intensité : 100 %")
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.valueChanged.connect(self._update_display_controls)
        display_controls_layout.addWidget(self.intensity_label)
        display_controls_layout.addWidget(self.intensity_slider)

        self.saturation_label = QLabel("Saturation : 0 %")
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(self._update_display_controls)
        display_controls_layout.addWidget(self.saturation_label)
        display_controls_layout.addWidget(self.saturation_slider)

        self.reset_display_button = QPushButton("Réinitialiser l'affichage")
        self.reset_display_button.clicked.connect(self.reset_display_controls)
        self.reset_display_button.setFixedHeight(30)

        display_controls_layout.addWidget(self.reset_display_button)
        result_layout.addWidget(display_controls)

        self.file_position_label = QLabel()
        self.file_position_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        result_layout.addWidget(self.file_position_label)
        content_layout.addWidget(result_panel, stretch=6)
        main_layout.addLayout(content_layout, stretch=1)

        self.load_models()
        self._update_file_navigation()
        self.load_file()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    def _update_display_controls(self):
        """Actualise les réglages visuels appliqués à toutes les coupes."""
        self.contrast_label.setText(f"Contraste : {self.contrast_slider.value()} %")
        self.intensity_label.setText(f"Intensité : {self.intensity_slider.value()} %")
        self.saturation_label.setText(f"Saturation : {self.saturation_slider.value()} %")
        if self.volume is not None:
            self.show_slice(self.slice_slider.value())

    def reset_display_controls(self):
        """Restaure les réglages d'affichage par défaut."""
        self.contrast_slider.setValue(100)
        self.intensity_slider.setValue(100)
        self.saturation_slider.setValue(0)

    def _prediction_cache_key(self, file_path: str, scan_id):
        model_path = self.model_combo.currentData()
        if scan_id is None or not isinstance(model_path, pathlib.Path):
            return None
        return (
            str(pathlib.Path(file_path).resolve()),
            repr(scan_id),
            str(model_path.resolve()),
        )

    def _set_prediction_ready(self, ready: bool):
        self.analyze_button.setProperty("predictionReady", ready)
        self.analyze_button.style().unpolish(self.analyze_button)
        self.analyze_button.style().polish(self.analyze_button)
        self.analyze_button.setToolTip(
            "Prédiction disponible en mémoire pour ce sujet et cette analyse."
            if ready
            else "Lancer la prédiction sur toutes les coupes."
        )

    def _cache_current_prediction(self):
        if self.current_prediction_key is None or self.prediction_masks is None:
            return
        self.prediction_cache[self.current_prediction_key] = {
            "prediction_masks": self.prediction_masks,
            "original_prediction_masks": self.original_prediction_masks,
            "prediction_outline_masks": self.prediction_outline_masks,
            "erased_outline_masks": self.erased_outline_masks,
            "mask_history": self.mask_history,
            "outline_history": self.outline_history,
            "erased_outline_history": self.erased_outline_history,
            "mask_history_index": self.mask_history_index,
        }

    def _clear_current_prediction(self):
        if self.contour_edit_active:
            self.set_contour_edit_mode(False)
        self._finish_contour_point_edit()
        if self.roi_only:
            self._auto_save_roi_modifications()
        self._clear_selected_contour()
        self._control_points_by_slice.clear()
        self.prediction_masks = None
        self.original_prediction_masks = None
        self.prediction_outline_masks = None
        self.erased_outline_masks = None
        self.current_prediction_key = None
        self.reset_mask_history()
        self.result_table.setRowCount(0)
        self.result_label.setText("Résultat de la prédiction")
        self.result_volume_label.setText("Volume total : -")
        self.save_roi_button.setEnabled(False)
        self.crop_mask_button.setEnabled(False)
        self.add_mask_button.setEnabled(False)
        self.edit_zone_button.setEnabled(False)
        self.set_mask_edit_mode(None)
        self._set_prediction_ready(False)

    def _restore_prediction_cache(self, file_path: str, scan_id) -> bool:
        cache_key = self._prediction_cache_key(file_path, scan_id)
        cached_prediction = self.prediction_cache.get(cache_key)
        if cached_prediction is None:
            self._clear_current_prediction()
            return False

        self.prediction_masks = cached_prediction["prediction_masks"]
        self.original_prediction_masks = cached_prediction[
            "original_prediction_masks"
        ]
        self.prediction_outline_masks = cached_prediction[
            "prediction_outline_masks"
        ]
        self.erased_outline_masks = cached_prediction["erased_outline_masks"]
        self.mask_history = cached_prediction["mask_history"]
        self.outline_history = cached_prediction["outline_history"]
        self.erased_outline_history = cached_prediction[
            "erased_outline_history"
        ]
        self.mask_history_index = cached_prediction["mask_history_index"]
        self.current_prediction_key = cache_key
        self.save_roi_button.setEnabled(True)
        self.crop_mask_button.setEnabled(True)
        self.add_mask_button.setEnabled(True)
        self.edit_zone_button.setEnabled(True)
        self._set_prediction_ready(True)
        self._refresh_result_list()
        self.update_history_buttons()
        self.show_slice(self.slice_slider.value())
        return True

    def _update_file_navigation(self):
        """Met à jour les boutons et la position du fichier courant."""
        file_count = len(self.dataset_paths)
        self.previous_file_button.setEnabled(self.file_index > 0)
        self.next_file_button.setEnabled(self.file_index < file_count - 1)
        self.file_position_label.setText(
            f"Sujet {self.file_index + 1} / {file_count}"
        )

    def navigate_file(self, offset: int):
        """Charge l'IRM précédente ou suivante du dossier."""
        new_index = self.file_index + offset
        if not 0 <= new_index < len(self.dataset_paths):
            return

        if self.contour_edit_active:
            self.set_contour_edit_mode(False)
        self._finish_contour_point_edit()
        if self.roi_only:
            self._auto_save_roi_modifications()
        self._clear_selected_contour()
        self._cache_current_prediction()
        self.file_index = new_index
        self.file_path = self.dataset_paths[self.file_index]
        self.setWindowTitle(
            f"MRI Viewer - {os.path.basename(self.file_path)}"
        )
        LOGGER.info(
            "Navigation vers l'IRM %d/%d: %s",
            self.file_index + 1,
            len(self.dataset_paths),
            self.file_path,
        )
        if self.roi_only and self.roi_directory is not None:
            dataset_name = pathlib.Path(self.file_path).name
            dataset_stem = pathlib.Path(dataset_name).stem
            candidates = (
                self.roi_directory / f"{dataset_stem}.zip",
                self.roi_directory / f"{dataset_name}.zip",
            )
            self.roi_zip_path = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.is_file()
                ),
                None,
            )
            if self.roi_zip_path is None:
                self.roi_zip_path = next(
                    (
                        zip_path
                        for zip_path in sorted(
                            self.roi_directory.rglob("*.zip"),
                            key=lambda path: path.name.casefold(),
                        )
                        if zip_path.stem.casefold() == dataset_stem.casefold()
                    ),
                    None,
                )
        self._update_file_navigation()
        self.load_file()

    def _selected_voxel_constants(self):
        model_path = self.model_combo.currentData()
        model_name = str(model_path) if model_path is not None else ""
        #print(model_name)
        if "RAT" in model_name:
            return VOXEL_AREA_MM2_RAT, VOXEL_VOLUME_MM3_RAT
        return VOXEL_AREA_MM2, VOXEL_VOLUME_MM3

    def _on_model_changed(self, _index: int):
        if not self.all_scans:
            return
        model_path = self.model_combo.currentData()
        scan_index = model_scan_index(self.all_scans, model_path)
        if scan_index != self.analysis_combo.currentIndex():
            self.analysis_combo.setCurrentIndex(scan_index)
        self._refresh_result_list()

    def load_models(self):
        """Remplit le sélecteur avec les checkpoints du dossier configuré."""
        if self.roi_only:
            self.model_combo.clear()
            self.model_combo.hide()
            return
        try:
            self.model_paths = available_checkpoints(self.model_directory)
        except FileNotFoundError as exc:
            self.model_combo.clear()
            self.model_combo.addItem("Aucun modèle disponible")
            self.model_combo.setEnabled(False)
            self.analyze_button.setEnabled(False)
            raise exc

        self.model_paths.sort(
            key=lambda checkpoint: (
                not any(
                    "ia_souris" in part.casefold()
                    for part in checkpoint.relative_to(
                        self.model_directory
                    ).parts
                ),
                str(checkpoint).casefold(),
            )
        )
        self.model_combo.clear()
        for checkpoint in self.model_paths:
            relative_checkpoint = checkpoint.relative_to(self.model_directory)
            model_name = (
                relative_checkpoint.parts[0]
                if len(relative_checkpoint.parts) > 1
                else self.model_directory.name
            )
            self.model_combo.addItem(
                model_name,
                checkpoint,
            )
        self.model_combo.setCurrentIndex(0)

    def _load_roi_masks(self) -> None:
        """Charge les contours d'un ZIP ROI dans les masques éditables du viewer."""
        if self.volume is None or self.roi_zip_path is None:
            self._clear_current_prediction()
            QMessageBox.warning(
                self,
                "ROI introuvables",
                f"Aucun fichier ROI correspondant trouvé pour :\n"
                f"{os.path.basename(self.file_path)}",
            )
            return

        height, width, depth = self.volume.shape
        masks = np.zeros((width, height, depth), dtype=np.uint8)

        rois = roiread(str(self.roi_zip_path))
        for roi in rois:
            position = getattr(roi, "position", 0)
            try:
                position = int(position)
            except (TypeError, ValueError):
                position = 0
            if position <= 0:
                name = getattr(roi, "name", "")
                name_number = "".join(
                    character
                    for character in str(name).split("-")[0]
                    if character.isdigit()
                )
                if not name_number:
                    continue
                try:
                    position = int(name_number)
                except (TypeError, ValueError):
                    continue

            slice_index = position - 1
            if not 0 <= slice_index < depth:
                continue

            coordinates = roi.coordinates()
            if coordinates is None:
                continue
            points = np.asarray(coordinates, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 2 or len(points) < 3:
                continue

            displayed_mask = np.zeros((height, width), dtype=np.uint8)
            polygon = np.rint(points[:, :2]).astype(np.int32)
            cv2.fillPoly(displayed_mask, [polygon], 1)
            masks[:, :, slice_index] = np.transpose(displayed_mask)

        self.prediction_masks = masks
        self.original_prediction_masks = masks.copy()
        self.prediction_outline_masks = masks.copy()
        self.erased_outline_masks = np.zeros_like(masks, dtype=bool)
        self.current_prediction_key = None
        self.reset_mask_history()
        self.save_roi_button.setEnabled(True)
        self.crop_mask_button.setEnabled(True)
        self.add_mask_button.setEnabled(True)
        self.edit_zone_button.setEnabled(True)
        self._set_prediction_ready(True)
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())

    def load_file(self):
        """Charge le scan IRM sélectionné."""
        if self.contour_edit_active:
            self.set_contour_edit_mode(False)
        self._finish_contour_point_edit()
        self._clear_selected_contour()
        self._control_points_by_slice.clear()
        self._cache_current_prediction()
        try:
            _, _, all_scans = load_data(self.file_path)
            self.all_scans = all_scans
            self.scan_ids = list(all_scans)
            self.analysis_combo.blockSignals(True)
            try:
                self.analysis_combo.clear()
                for scan_id, scan_info in all_scans.items():
                    self.analysis_combo.addItem(
                        scan_label(scan_id, scan_info), scan_id
                    )
                if self.scan_ids:
                    self.analysis_combo.setCurrentIndex(
                        model_scan_index(
                            all_scans,
                            self.model_combo.currentData(),
                        )
                    )
            finally:
                self.analysis_combo.blockSignals(False)

            selected_scan_id = self.analysis_combo.currentData()
            data, _, _ = load_data(self.file_path, selected_scan_id)
            if data is None:
                self.slice_label.setText("No MRI volume available")
                return

            self.volume = prepare_volume(data)
            number_of_slices = self.volume.shape[2]

            self.slice_slider.setRange(0, number_of_slices - 1)
            self.slice_slider.setEnabled(number_of_slices > 1)

            # Commence à la première coupe afin que le curseur et les flèches
            # aient une origine claire.
            self.slice_slider.setValue(0)
            if not self._restore_prediction_cache(self.file_path, selected_scan_id):
                self.show_slice(0)
            if self.roi_only:
                self._load_roi_masks()
            self._update_navigation_buttons()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Loading error",
                f"Could not load this dataset:\n{exc}",
            )

    def change_scan(self, index: int):
        """Charge le scan sélectionné dans la liste des analyses."""
        if index < 0 or index >= len(self.scan_ids):
            return
        if self.contour_edit_active:
            self.set_contour_edit_mode(False)
        self._finish_contour_point_edit()
        self._clear_selected_contour()
        self._control_points_by_slice.clear()
        self._cache_current_prediction()
        scan_id = self.analysis_combo.itemData(index)
        try:
            data, _, _ = load_data(self.file_path, scan_id)
            if data is None:
                raise ValueError("Cette analyse ne contient pas de volume exploitable.")
            self.volume = prepare_volume(data)
            self.slice_slider.setRange(0, self.volume.shape[2] - 1)
            self.slice_slider.setValue(0)
            if not self._restore_prediction_cache(self.file_path, scan_id):
                self.show_slice(0)
            if self.roi_only:
                self._load_roi_masks()
            self._update_navigation_buttons()
        except Exception as exc:
            QMessageBox.critical(self, "Analyse impossible", f"Impossible de charger cette analyse:\n{exc}")

    def analyze_volume(self):
        """Exécute le modèle choisi sur toutes les coupes du scan sélectionné."""
        if self.volume is None:
            return
        if self.contour_edit_active:
            self.set_contour_edit_mode(False)
        self._finish_contour_point_edit()
        self._clear_selected_contour()
        self._control_points_by_slice.clear()
        self.analyze_button.setEnabled(False)
        self.slice_label.setText("Analyse en cours...")
        QApplication.processEvents()
        try:
            scan_id = self.analysis_combo.currentData()
            model_volume = convert_pvdataset_to_npy(self.file_path, scan_id)
            model_path = self.model_combo.currentData()
            if not isinstance(model_path, pathlib.Path):
                raise ValueError("Aucun checkpoint de modèle sélectionné.")
            model_mask = self.predictor.predict(model_volume, model_path)
            # Conserve le volume original du viewer et reconvertit uniquement
            # le masque. Cette transposition inverse celle de loader.py ;
            # l'axe des coupes garde le même ordre, sans inversion.
            self.prediction_masks = np.transpose(model_mask, (2, 1, 0))
            self.original_prediction_masks = self.prediction_masks.copy()
            self.current_prediction_key = self._prediction_cache_key(
                self.file_path,
                scan_id,
            )
            self._refresh_result_list()
            self.reset_mask_history()
            self.save_roi_button.setEnabled(True)
            self.crop_mask_button.setEnabled(True)
            self.add_mask_button.setEnabled(True)
            self.edit_zone_button.setEnabled(True)
            self._set_prediction_ready(True)
            self.update_history_buttons()
            self.show_slice(self.slice_slider.value())
            self._cache_current_prediction()
        except Exception as exc:
            QMessageBox.critical(self, "Erreur d'analyse", f"Prédiction impossible:\n{exc}")
        finally:
            self.analyze_button.setEnabled(True)

    def _auto_save_roi_modifications(self) -> None:
        """Réécrit le ZIP ROI correspondant après chaque modification du masque."""
        if not self.roi_only or self.roi_zip_path is None or self.prediction_masks is None:
            return
        try:
            save_modified_roi_slice(
                self.prediction_masks,
                self.roi_zip_path,
                self.slice_slider.value(),
            )
        except Exception as exc:
            LOGGER.error(
                "Impossible de réécrire le ZIP ROI %s: %s",
                self.roi_zip_path,
                exc,
            )
            QMessageBox.critical(
                self,
                "Erreur de sauvegarde",
                f"Impossible de réécrire le fichier ROI existant :\n{exc}",
            )

    def set_mask_edit_mode(self, mode: str | None):
        """Définit le mode de dessin libre actif pour le masque courant."""
        if mode is not None and self.contour_edit_active:
            self.set_contour_edit_mode(False)
        self.mask_edit_mode = mode
        self.image_label.set_edit_mode(
            mode,
            pan_with_right_button=mode is not None,
            legacy_polygon_drawing=self.mask_drawing_method == "guiv2",
        )
        if mode == "crop":
            self.add_mask_button.setChecked(False)
        elif mode == "add":
            self.crop_mask_button.setChecked(False)
        else:
            self.crop_mask_button.setChecked(False)
            self.add_mask_button.setChecked(False)

    def set_contour_edit_mode(self, enabled: bool):
        if enabled and self.prediction_masks is None:
            enabled = False
        if self.edit_zone_button.isChecked() != enabled:
            previous_block_state = self.edit_zone_button.blockSignals(True)
            self.edit_zone_button.setChecked(enabled)
            self.edit_zone_button.blockSignals(previous_block_state)

        if enabled:
            self.set_mask_edit_mode(None)
            self.contour_edit_active = True
            self.image_label.set_contour_edit_mode(True)
            if (
                self._active_contour_slice == self.slice_slider.value()
                and self._active_contour_component is not None
            ):
                erased = self.erased_outline_masks[
                    :, :, self.slice_slider.value()
                ].T
                self.image_label.set_active_contour(
                    self._active_contour,
                    self._active_control_points,
                    erased,
                )
            self.slice_label.setText(
                "Cliquez sur la zone à modifier. Glissez un point pour le "
                "déplacer; double-clic pour en ajouter un; clic droit pour "
                "en retirer un. Q quitte le mode."
            )
            return

        self._finish_contour_point_edit()
        self.contour_edit_active = False
        self.image_label.set_contour_edit_mode(False)

    def _clear_selected_contour(self):
        self._active_contour_slice = None
        self._active_contour = None
        self._active_contour_component = None
        self._active_contour_seed = None
        self._active_control_points = []
        self.image_label.set_active_contour(None, [])

    def select_contour_for_edit(
        self,
        position: QPointF,
        preserve_control_points: bool = False,
    ):
        if (
            not self.contour_edit_active
            or self.prediction_masks is None
            or self.erased_outline_masks is None
        ):
            return

        slice_index = self.slice_slider.value()
        displayed_mask = np.ascontiguousarray(
            self.prediction_masks[:, :, slice_index].T,
            dtype=np.uint8,
        )
        x = int(round(position.x()))
        y = int(round(position.y()))
        height, width = displayed_mask.shape
        if not (0 <= x < width and 0 <= y < height):
            return
        _, labels = cv2.connectedComponents(
            (displayed_mask > 0).astype(np.uint8),
            connectivity=8,
        )
        component_id = int(labels[y, x])
        if component_id == 0:
            y0, y1 = max(0, y - 1), min(height, y + 2)
            x0, x1 = max(0, x - 1), min(width, x + 2)
            nearby = np.argwhere(labels[y0:y1, x0:x1] > 0)
            if not len(nearby):
                return
            distances = (nearby[:, 0] + y0 - y) ** 2 + (
                nearby[:, 1] + x0 - x
            ) ** 2
            nearest = nearby[int(np.argmin(distances))]
            component_id = int(labels[nearest[0] + y0, nearest[1] + x0])

        component = labels == component_id
        same_component = (
            self._active_contour_slice == slice_index
            and self._active_contour_component is not None
            and np.array_equal(component, self._active_contour_component)
        )
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return
        contour = max(contours, key=cv2.contourArea)
        erased = np.ascontiguousarray(
            self.erased_outline_masks[:, :, slice_index].T,
            dtype=bool,
        )
        if preserve_control_points or same_component:
            control_points = list(self._active_control_points)
        else:
            saved_points = self._control_points_by_slice.get(slice_index)
            if saved_points is not None:
                contour_points = contour[:, 0, :].astype(float)
                control_points = []
                for saved_point in saved_points:
                    distances = np.sum(
                        (contour_points - np.asarray(saved_point)) ** 2,
                        axis=1,
                    )
                    nearest_index = int(np.argmin(distances))
                    if distances[nearest_index] <= 4.0:
                        nearest_point = contour_points[nearest_index]
                        px, py = int(nearest_point[0]), int(nearest_point[1])
                        if not erased[py, px]:
                            control_points.append((px, py))
            else:
                control_points = []
            if not control_points:
                defaults = cv2.approxPolyDP(
                    contour,
                    3.0,
                    True,
                ).reshape(-1, 2)
                if len(defaults) < 3:
                    defaults = contour.reshape(-1, 2)
                control_points = [
                    (int(point[0]), int(point[1]))
                    for point in defaults
                    if not erased[int(point[1]), int(point[0])]
                ]
        self._active_contour_slice = slice_index
        self._active_contour = contour.copy()
        self._active_contour_component = component
        self._active_contour_seed = QPointF(x, y)
        self._active_control_points = list(control_points)
        self._control_points_by_slice[slice_index] = list(control_points)
        self.image_label.set_active_contour(
            contour,
            self._active_control_points,
            erased,
        )
        self.slice_label.setText(
            "Glissez un point du contour pour le déplacer. Double-clic : "
            "ajouter un point; clic droit sur un point : le supprimer."
        )

    def add_contour_control_point(self, position: QPointF):
        if (
            not self.contour_edit_active
            or self.prediction_masks is None
            or self.erased_outline_masks is None
        ):
            return
        slice_index = self.slice_slider.value()
        if (
            self._active_contour is None
            or self._active_contour_slice != slice_index
        ):
            self.select_contour_for_edit(position)
        if (
            self._active_contour is None
            or self._active_contour_slice != slice_index
            or self._active_contour_component is None
        ):
            self.slice_label.setText(
                "Sélectionnez une zone, puis double-cliquez sur son contour "
                "ou à l'intérieur pour ajouter un point."
            )
            return

        displayed_mask = self.prediction_masks[:, :, slice_index].T
        x = int(round(position.x()))
        y = int(round(position.y()))
        height, width = displayed_mask.shape
        if not (0 <= x < width and 0 <= y < height):
            return
        if (
            displayed_mask[y, x]
            and not self._active_contour_component[y, x]
        ):
            self.select_contour_for_edit(position)
            if (
                self._active_contour is None
                or self._active_contour_slice != slice_index
                or self._active_contour_component is None
            ):
                return

        inside_component = bool(self._active_contour_component[y, x])
        point_index = self.image_label._nearest_active_contour_point(
            position,
            tolerance=float("inf") if inside_component else None,
            include_erased=True,
            excluded_points=set(self._active_control_points),
        )
        if point_index is None:
            self.slice_label.setText(
                "Aucun point disponible ici. Double-cliquez le contour ou "
                "l'intérieur de la zone sélectionnée."
            )
            return
        point = self._active_contour[point_index, 0]
        control_point = (int(point[0]), int(point[1]))
        self._active_control_points.append(control_point)
        self._control_points_by_slice[slice_index] = list(
            self._active_control_points
        )
        erased = self.erased_outline_masks[
            :, :, slice_index
        ].T
        self.image_label.set_active_contour(
            self._active_contour,
            self._active_control_points,
            erased,
        )
        self.slice_label.setText(
            "Point ajouté. Faites-le glisser pour modifier le contour; "
            "double-cliquez à l'intérieur de la zone pour en ajouter un autre."
        )

    def remove_contour_control_point(self, point_index: int):
        if (
            not self.contour_edit_active
            or self._active_contour_slice is None
            or not 0 <= point_index < len(self._active_control_points)
        ):
            return
        del self._active_control_points[point_index]
        if self._active_contour_slice is not None:
            self._control_points_by_slice[self._active_contour_slice] = list(
                self._active_control_points
            )
        erased = self.erased_outline_masks[
            :, :, self._active_contour_slice
        ].T
        self.image_label.set_active_contour(
            self._active_contour,
            self._active_control_points,
            erased,
        )

    def _begin_contour_point_edit(
        self,
        contour: np.ndarray,
        point_index: int,
        marker_index: int,
    ) -> None:
        if (
            not self.contour_edit_active
            or self.prediction_masks is None
            or self.prediction_outline_masks is None
            or self.erased_outline_masks is None
        ):
            return
        if self._contour_edit_state is not None:
            self._finish_contour_point_edit()

        slice_index = self.slice_slider.value()
        displayed_mask = np.ascontiguousarray(
            self.prediction_masks[:, :, slice_index].T,
            dtype=np.uint8,
        )
        contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2).copy()
        if not 0 <= point_index < len(contour):
            return
        height, width = displayed_mask.shape
        x, y = contour[point_index, 0]
        if not (0 <= x < width and 0 <= y < height):
            return

        _, labels = cv2.connectedComponents(
            (displayed_mask > 0).astype(np.uint8),
            connectivity=8,
        )
        component_id = int(labels[y, x])
        if component_id == 0:
            y0, y1 = max(0, y - 1), min(height, y + 2)
            x0, x1 = max(0, x - 1), min(width, x + 2)
            nearby = np.argwhere(labels[y0:y1, x0:x1] > 0)
            if not len(nearby):
                return
            distances = (nearby[:, 0] + y0 - y) ** 2 + (
                nearby[:, 1] + x0 - x
            ) ** 2
            nearest = nearby[int(np.argmin(distances))]
            component_id = int(labels[nearest[0] + y0, nearest[1] + x0])

        self._contour_edit_state = ContourPointEdit(
            slice_index=slice_index,
            contour=contour,
            point_index=point_index,
            marker_index=marker_index,
            current_position=(int(x), int(y)),
            component_mask=labels == component_id,
            original_mask=displayed_mask,
            original_outline=np.ascontiguousarray(
                self.prediction_outline_masks[:, :, slice_index].T,
                dtype=np.uint8,
            ),
            original_erased=np.ascontiguousarray(
                self.erased_outline_masks[:, :, slice_index].T,
                dtype=bool,
            ),
            original_total_voxels=int(
                self.prediction_masks.astype(bool).sum()
            ),
            original_slice_voxels=int(displayed_mask.astype(bool).sum()),
        )

    def _preview_contour_point_edit(self, position: QPointF) -> None:
        state = self._contour_edit_state
        if state is None or self.prediction_masks is None:
            return

        height, width = state.original_mask.shape
        x = min(max(int(round(position.x())), 0), width - 1)
        y = min(max(int(round(position.y())), 0), height - 1)
        new_contour = state.contour.copy()
        new_contour[state.point_index, 0] = (x, y)
        state.current_position = (x, y)
        contour_size = len(state.contour)
        original_point = state.contour[state.point_index, 0]
        movement = float(
            np.hypot(x - int(original_point[0]), y - int(original_point[1]))
        )
        if movement and contour_size > 2:
            span = min(
                max(1, int(np.ceil(movement))),
                max(1, (contour_size - 1) // 2),
            )
            segment_indices = (
                state.point_index - span + np.arange(2 * span + 1)
            ) % contour_size
            ordered_contour = state.contour[
                (state.point_index - span + np.arange(contour_size))
                % contour_size
            ]
            new_contour = np.concatenate(
                (
                    ordered_contour[:1],
                    np.asarray([[[x, y]]], dtype=np.int32),
                    ordered_contour[
                        2 * span :
                    ],
                ),
                axis=0,
            )
            local_points = np.concatenate(
                (
                    state.contour[segment_indices],
                    np.asarray([[[x, y]]], dtype=np.int32),
                ),
                axis=0,
            )[:, 0, :]
        else:
            local_points = new_contour[[state.point_index], 0, :]
        x0 = max(int(local_points[:, 0].min()) - 1, 0)
        y0 = max(int(local_points[:, 1].min()) - 1, 0)
        x1 = min(int(local_points[:, 0].max()) + 2, width)
        y1 = min(int(local_points[:, 1].max()) + 2, height)
        patch_shape = (y1 - y0, x1 - x0)
        original_fill = np.zeros(patch_shape, dtype=np.uint8)
        new_fill = np.zeros(patch_shape, dtype=np.uint8)
        offset = (-x0, -y0)
        cv2.fillPoly(original_fill, [state.contour], 1, offset=offset)
        cv2.fillPoly(new_fill, [new_contour], 1, offset=offset)
        added_region = (new_fill > 0) & (original_fill == 0)
        removed_region = (
            (original_fill > 0)
            & (new_fill == 0)
            & state.component_mask[y0:y1, x0:x1]
        )
        displayed_mask = state.original_mask.copy()
        mask_patch = displayed_mask[y0:y1, x0:x1]
        mask_patch[added_region] = 1
        mask_patch[removed_region] = 0
        displayed_outline = state.original_outline.copy()
        outline_patch = displayed_outline[y0:y1, x0:x1]
        outline_patch[added_region] = 1
        outline_patch[removed_region] = 0
        displayed_erased = state.original_erased.copy()
        erased_patch = displayed_erased[y0:y1, x0:x1]
        erased_patch[added_region | removed_region] = False

        slice_index = state.slice_index
        self.prediction_masks[:, :, slice_index] = displayed_mask.T
        self.prediction_outline_masks[:, :, slice_index] = (
            displayed_outline.T
        )
        self.erased_outline_masks[:, :, slice_index] = displayed_erased.T
        if (
            state.marker_index >= 0
            and state.marker_index < len(self._active_control_points)
        ):
            self._active_control_points[state.marker_index] = (x, y)
            self._control_points_by_slice[slice_index] = list(
                self._active_control_points
            )
            self.image_label.move_control_point(state.marker_index, x, y)
        slice_voxels = int(displayed_mask.astype(bool).sum())
        total_voxels = (
            state.original_total_voxels
            + slice_voxels
            - state.original_slice_voxels
        )
        self._update_live_volume_preview(
            slice_index,
            slice_voxels,
            total_voxels,
        )
        self.image_label.set_prediction_overlay(
            displayed_outline,
            displayed_erased,
        )

    def _update_live_volume_preview(
        self,
        slice_index: int,
        slice_voxels: int,
        total_voxels: int,
    ):
        voxel_area_mm2, voxel_volume_mm3 = self._selected_voxel_constants()
        self.result_label.setText(
            f"Résultat de la prédiction ({total_voxels} voxels)"
        )
        self.result_volume_label.setText(
            f"Volume total : {total_voxels * voxel_volume_mm3:.4f} mm³"
        )
        if slice_index < self.result_table.rowCount():
            self.result_table.setItem(
                slice_index,
                1,
                QTableWidgetItem(str(slice_voxels)),
            )
            self.result_table.setItem(
                slice_index,
                2,
                QTableWidgetItem(
                    f"{slice_voxels * voxel_area_mm2:.4f}"
                ),
            )

    def _finish_contour_point_edit(self) -> None:
        state = self._contour_edit_state
        if state is None or self.prediction_masks is None:
            self._contour_edit_state = None
            return
        self._contour_edit_state = None
        slice_index = state.slice_index
        current_mask = self.prediction_masks[:, :, slice_index]
        current_outline = self.prediction_outline_masks[:, :, slice_index]
        current_erased = self.erased_outline_masks[:, :, slice_index]
        if (
            np.array_equal(current_mask, state.original_mask.T)
            and np.array_equal(current_outline, state.original_outline.T)
            and np.array_equal(current_erased, state.original_erased.T)
        ):
            return

        if not self.mask_history:
            self.reset_mask_history()
        self.mask_history = self.mask_history[: self.mask_history_index + 1]
        self.outline_history = self.outline_history[
            : self.mask_history_index + 1
        ]
        self.erased_outline_history = self.erased_outline_history[
            : self.mask_history_index + 1
        ]
        self.mask_history.append(self.prediction_masks.copy())
        self.outline_history.append(self.prediction_outline_masks.copy())
        self.erased_outline_history.append(self.erased_outline_masks.copy())
        self.mask_history_index += 1
        self.update_history_buttons()
        self._refresh_result_list()
        self._cache_current_prediction()
        if self.roi_only:
            self._auto_save_roi_modifications()
        if (
            self.contour_edit_active
            and self._active_contour_slice == slice_index
            and self.slice_slider.value() == slice_index
        ):
            self.select_contour_for_edit(
                QPointF(*state.current_position),
                preserve_control_points=True,
            )
            if (
                self._active_contour is not None
                and 0 <= state.marker_index < len(self._active_control_points)
            ):
                contour_points = self._active_contour[:, 0, :]
                distances = np.sum(
                    (
                        contour_points
                        - np.asarray(state.current_position)
                    )
                    ** 2,
                    axis=1,
                )
                nearest_point = contour_points[int(np.argmin(distances))]
                self._active_control_points[state.marker_index] = (
                    int(nearest_point[0]),
                    int(nearest_point[1]),
                )
                self._control_points_by_slice[slice_index] = list(
                    self._active_control_points
                )
                erased = self.erased_outline_masks[
                    :, :, slice_index
                ].T
                self.image_label.set_active_contour(
                    self._active_contour,
                    self._active_control_points,
                    erased,
                )

    def set_mask_drawing_method(self, legacy_polygon_drawing: bool):
        """Sélectionne le dessin polygonal libre de guiv2.py ou celui de gui_v4.py."""
        self.mask_drawing_method = (
            "guiv2" if legacy_polygon_drawing else "gui_v4"
        )
        self.image_label.set_edit_mode(
            self.mask_edit_mode,
            pan_with_right_button=self.mask_edit_mode is not None,
            legacy_polygon_drawing=legacy_polygon_drawing,
        )
        try:
            save_mask_drawing_method(legacy_polygon_drawing)
        except OSError as exc:
            LOGGER.exception("Impossible d'enregistrer la préférence de dessin.")
            QMessageBox.warning(
                self,
                "Préférence non enregistrée",
                f"Le choix du mode de dessin ne pourra pas être conservé :\n{exc}",
            )

    def eventFilter(self, watched, event):
        if (
            (
                self.mask_edit_mode in ("add", "crop")
                or self.contour_edit_active
            )
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and event.key() == Qt.Key.Key_Q
            and isinstance(watched, QWidget)
            and (watched is self or self.isAncestorOf(watched))
        ):
            if self.contour_edit_active:
                self.edit_zone_button.setChecked(False)
            else:
                self.add_mask_button.setChecked(False)
                self.crop_mask_button.setChecked(False)
                self.set_mask_edit_mode(None)
            return True
        return super().eventFilter(watched, event)

    def _refresh_result_list(self):
        """Affiche les voxels et l'aire prédite de chaque coupe."""
        voxel_area_mm2, voxel_volume_mm3 = self._selected_voxel_constants()
        self.result_table.setRowCount(0)
        if self.prediction_masks is None:
            self.result_label.setText("Résultat de la prédiction")
            self.result_volume_label.setText("Volume total : -")
            return

        positive_voxels = self.prediction_masks.astype(bool).sum(axis=(0, 1))
        total_positive = int(positive_voxels.sum())
        self.result_label.setText(
            f"Résultat de la prédiction ({total_positive} voxels)"
        )
        self.result_volume_label.setText(
            f"Volume total : {total_positive * voxel_volume_mm3:.4f} mm³"
        )
        self.result_table.setRowCount(len(positive_voxels))
        for index, count in enumerate(positive_voxels):
            for column, value in enumerate(
                (
                    str(index + 1),
                    str(int(count)),
                    f"{count * voxel_area_mm2:.4f}",
                )
            ):
                item = QTableWidgetItem(value)
                self.result_table.setItem(index, column, item)
        current_slice = self.slice_slider.value()
        if current_slice < self.result_table.rowCount():
            self.result_table.setCurrentCell(current_slice, 0)

    def _select_result_slice(self, index: int):
        """Navigue vers la coupe sélectionnée dans le panneau de résultats."""
        if (
            index < 0
            or self.volume is None
            or index >= self.slice_slider.maximum() + 1
        ):
            return
        self.slice_slider.setValue(index)

    def reset_mask_history(self):
        """Supprime l'historique lorsqu'une nouvelle prédiction ou analyse est chargée."""
        self.mask_history = []
        if self.prediction_masks is None:
            self.prediction_outline_masks = None
            self.erased_outline_masks = None
            self.outline_history = []
            self.erased_outline_history = []
        else:
            self.prediction_outline_masks = self.prediction_masks.copy()
            self.erased_outline_masks = np.zeros_like(
                self.prediction_masks,
                dtype=bool,
            )
            self.outline_history = [self.prediction_outline_masks.copy()]
            self.erased_outline_history = [self.erased_outline_masks.copy()]
            self.mask_history = [self.prediction_masks.copy()]
            self.mask_history_index = 0
        if self.prediction_masks is None:
            self.mask_history_index = -1
        self.update_history_buttons()

    def update_history_buttons(self):
        self.undo_button.setEnabled(self.mask_history_index > 0)
        self.redo_button.setEnabled(
            0 <= self.mask_history_index < len(self.mask_history) - 1
        )
        slice_index = self.slice_slider.value()
        self.reset_mask_button.setEnabled(
            self.original_prediction_masks is not None
            and self.prediction_masks is not None
            and (
                not np.array_equal(
                    self.prediction_masks[:, :, slice_index],
                    self.original_prediction_masks[:, :, slice_index],
                )
                or (
                    self.erased_outline_masks is not None
                    and bool(self.erased_outline_masks[:, :, slice_index].any())
                )
                or (
                    self.prediction_outline_masks is not None
                    and not np.array_equal(
                        self.prediction_outline_masks[:, :, slice_index],
                        self.original_prediction_masks[:, :, slice_index],
                    )
                )
            )
        )

    def undo_mask_edit(self):
        self._finish_contour_point_edit()
        if self.mask_history_index <= 0:
            return
        self.mask_history_index -= 1
        self.prediction_masks = self.mask_history[self.mask_history_index].copy()
        self.prediction_outline_masks = self.outline_history[
            self.mask_history_index
        ].copy()
        self.erased_outline_masks = self.erased_outline_history[
            self.mask_history_index
        ].copy()
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())
        self._cache_current_prediction()
        if self.roi_only:
            self._auto_save_roi_modifications()

    def redo_mask_edit(self):
        self._finish_contour_point_edit()
        if self.mask_history_index >= len(self.mask_history) - 1:
            return
        self.mask_history_index += 1
        self.prediction_masks = self.mask_history[self.mask_history_index].copy()
        self.prediction_outline_masks = self.outline_history[
            self.mask_history_index
        ].copy()
        self.erased_outline_masks = self.erased_outline_history[
            self.mask_history_index
        ].copy()
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())
        self._cache_current_prediction()
        if self.roi_only:
            self._auto_save_roi_modifications()

    def reset_mask_edits(self):
        """Restaure uniquement la coupe courante à la prédiction originale."""
        self._finish_contour_point_edit()
        if self.original_prediction_masks is None or self.prediction_masks is None:
            return
        slice_index = self.slice_slider.value()
        if not self.reset_mask_button.isEnabled():
            return

        self.prediction_masks[:, :, slice_index] = (
            self.original_prediction_masks[:, :, slice_index]
        )
        self.prediction_outline_masks[:, :, slice_index] = (
            self.original_prediction_masks[:, :, slice_index]
        )
        self.erased_outline_masks[:, :, slice_index] = False

        self.mask_history = self.mask_history[: self.mask_history_index + 1]
        self.outline_history = self.outline_history[
            : self.mask_history_index + 1
        ]
        self.erased_outline_history = self.erased_outline_history[
            : self.mask_history_index + 1
        ]
        self.mask_history.append(self.prediction_masks.copy())
        self.outline_history.append(self.prediction_outline_masks.copy())
        self.erased_outline_history.append(self.erased_outline_masks.copy())
        self.mask_history_index += 1
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())
        self._cache_current_prediction()
        if self.roi_only:
            self._auto_save_roi_modifications()

    def apply_mask_edit_path(self, path: QPainterPath, closes_path: bool):
        """Efface un trait ou ajoute une zone sur la coupe affichée."""
        if self.prediction_masks is None or self.mask_edit_mode is None:
            return

        displayed_mask = self.prediction_masks[:, :, self.slice_slider.value()]
        displayed_mask = np.ascontiguousarray(np.transpose(displayed_mask))
        height, width = displayed_mask.shape
        legacy_drawing = self.mask_drawing_method == "guiv2"
        if legacy_drawing:
            polygon = np.array(
                [
                    [
                        max(
                            0,
                            min(
                                width - 1,
                                int(round(path.elementAt(index).x)),
                            ),
                        ),
                        max(
                            0,
                            min(
                                height - 1,
                                int(round(path.elementAt(index).y)),
                            ),
                        ),
                    ]
                    for index in range(path.elementCount())
                ],
                dtype=np.int32,
            )
            if len(polygon) < 3:
                return
            edit_area = np.zeros(displayed_mask.shape, dtype=np.uint8)
            cv2.fillPoly(edit_area, [polygon], 1)
            edit_area = edit_area.astype(bool)
        else:
            path_image = QImage(
                width,
                height,
                QImage.Format.Format_RGB32,
            )
            path_image.fill(Qt.GlobalColor.black)
            painter = QPainter(path_image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            if self.mask_edit_mode == "crop":
                pen = QPen(Qt.GlobalColor.white, MASK_ERASER_DIAMETER)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
                if path.elementCount() == 1:
                    point = path.currentPosition()
                    radius = MASK_ERASER_DIAMETER / 2
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(Qt.GlobalColor.white)
                    painter.drawEllipse(
                        QRectF(
                            point.x() - radius,
                            point.y() - radius,
                            MASK_ERASER_DIAMETER,
                            MASK_ERASER_DIAMETER,
                        )
                    )
            elif not closes_path:
                pen = QPen(Qt.GlobalColor.white, MASK_ERASER_DIAMETER)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(Qt.GlobalColor.white)
                painter.drawPath(path)
            painter.end()
            grayscale_path = path_image.convertToFormat(
                QImage.Format.Format_Grayscale8
            )
            path_pixels = np.frombuffer(
                grayscale_path.constBits(),
                dtype=np.uint8,
                count=grayscale_path.sizeInBytes(),
            ).reshape(height, grayscale_path.bytesPerLine())[:, :width]
            edit_area = (
                path_pixels > 0
                if self.mask_edit_mode == "crop"
                else path_pixels >= 128
            )
        if not edit_area.any():
            return

        slice_index = self.slice_slider.value()
        displayed_outline = np.ascontiguousarray(
            np.transpose(self.prediction_outline_masks[:, :, slice_index])
        )
        displayed_erased_outline = np.ascontiguousarray(
            np.transpose(self.erased_outline_masks[:, :, slice_index])
        )
        if legacy_drawing:
            target_value = self.mask_edit_mode == "add"
            if not np.any(edit_area & (displayed_mask != target_value)):
                return
        elif self.mask_edit_mode == "crop":
            outline = np.zeros(displayed_outline.shape, dtype=np.uint8)
            contours, _ = cv2.findContours(
                np.ascontiguousarray(displayed_outline, dtype=np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                self.slice_label.setText(
                    f"Coupe {slice_index + 1} / {self.volume.shape[2]} "
                    "(aucun contour rouge à gommer)"
                )
                return
            cv2.drawContours(outline, contours, -1, 1, 1)
            erased_stroke = cv2.dilate(
                edit_area.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
            erased_area = erased_stroke & outline.astype(bool)
            outline_changed = np.any(
                erased_area & ~displayed_erased_outline
            )
            if not outline_changed:
                return
        else:
            if not closes_path:
                stroke_points = np.array(
                    [
                        [
                            round(path.elementAt(index).x),
                            round(path.elementAt(index).y),
                        ]
                        for index in range(path.elementCount())
                    ],
                    dtype=np.int32,
                )
                if not replace_erased_contour_segment(
                    displayed_mask,
                    displayed_outline,
                    displayed_erased_outline,
                    stroke_points,
                ):
                    return
            elif not np.any(
                edit_area & (displayed_mask == 0)
            ) and not np.any(edit_area & displayed_erased_outline):
                return

        if not self.mask_history:
            self.mask_history = [self.prediction_masks.copy()]
            self.outline_history = [self.prediction_outline_masks.copy()]
            self.erased_outline_history = [self.erased_outline_masks.copy()]
            self.mask_history_index = 0

        if legacy_drawing:
            target_value = 1 if self.mask_edit_mode == "add" else 0
            displayed_mask[edit_area] = target_value
            displayed_outline[edit_area] = target_value
            displayed_erased_outline[edit_area] = False
            self.prediction_masks[:, :, slice_index] = np.transpose(
                displayed_mask
            )
            self.prediction_outline_masks[:, :, slice_index] = np.transpose(
                displayed_outline
            )
            self.erased_outline_masks[:, :, slice_index] = np.transpose(
                displayed_erased_outline
            )
        elif self.mask_edit_mode == "crop":
            displayed_erased_outline |= erased_area
            remaining_outline = outline.astype(bool) & ~displayed_erased_outline
            if not remaining_outline.any():
                displayed_mask[:] = 0
                displayed_outline[:] = 0
                displayed_erased_outline[:] = False
            else:
                component_count, component_labels = cv2.connectedComponents(
                    np.ascontiguousarray(displayed_mask, dtype=np.uint8),
                    connectivity=8,
                )
                for component_index in range(1, component_count):
                    component = component_labels == component_index
                    contours, _ = cv2.findContours(
                        np.ascontiguousarray(component, dtype=np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE,
                    )
                    boundary = np.concatenate(
                        [contour[:, 0, :] for contour in contours],
                        axis=0,
                    )
                    if np.all(
                        displayed_erased_outline[
                            boundary[:, 1],
                            boundary[:, 0],
                        ]
                    ):
                        displayed_mask[component] = 0
                        displayed_outline[component] = 0
                        displayed_erased_outline[component] = False

            self.prediction_masks[:, :, slice_index] = np.transpose(
                displayed_mask
            )
            self.prediction_outline_masks[:, :, slice_index] = np.transpose(
                displayed_outline
            )
            self.erased_outline_masks[:, :, slice_index] = np.transpose(
                displayed_erased_outline
            )
        else:
            if closes_path:
                displayed_mask[edit_area] = 1
                displayed_outline[edit_area] = 1
                displayed_erased_outline[edit_area] = False
            self.prediction_outline_masks[:, :, slice_index] = np.transpose(
                displayed_outline
            )
            self.erased_outline_masks[:, :, slice_index] = np.transpose(
                displayed_erased_outline
            )

        if self.mask_edit_mode == "add":
            self.prediction_masks[:, :, slice_index] = np.transpose(
                displayed_mask
            )
        self.mask_history = self.mask_history[: self.mask_history_index + 1]
        self.outline_history = self.outline_history[
            : self.mask_history_index + 1
        ]
        self.erased_outline_history = self.erased_outline_history[
            : self.mask_history_index + 1
        ]
        self.mask_history.append(self.prediction_masks.copy())
        self.outline_history.append(self.prediction_outline_masks.copy())
        self.erased_outline_history.append(self.erased_outline_masks.copy())
        self.mask_history_index += 1
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())
        self._cache_current_prediction()
        if self.roi_only:
            self._auto_save_roi_modifications()

    def save_prediction_as_rois(self):
        """Exporte la prédiction courante vers un ZIP choisi par l'utilisateur."""
        self._finish_contour_point_edit()
        if self.prediction_masks is None:
            QMessageBox.information(
                self,
                "Aucune prédiction",
                "Lancez une analyse avant de sauvegarder les ROI.",
            )
            return

        try:
            if self.roi_only:
                for slice_index in range(self.prediction_masks.shape[2]):
                    if not np.array_equal(
                        self.prediction_masks[:, :, slice_index],
                        self.original_prediction_masks[:, :, slice_index],
                    ):
                        save_modified_roi_slice(
                            self.prediction_masks,
                            self.roi_zip_path,
                            slice_index,
                        )
                output_path = self.roi_zip_path
                QMessageBox.information(
                    self,
                    "ROI sauvegardées",
                    f"Les modifications ont été enregistrées dans :\n{output_path}",
                )
                self._cache_current_prediction()
                return

            suggested_name = pathlib.Path(self.file_path).with_suffix(".zip").name
            selected_path, _ = QFileDialog.getSaveFileName(
                self,
                "Choisir l'emplacement des ROI",
                str(pathlib.Path(self.file_path).with_name(suggested_name)),
                "Fichier ZIP (*.zip)",
            )
            if not selected_path:
                return

            # L'export utilise le format modèle (D,H,W) en conservant la
            # position originale de chaque coupe et le repère des ROI.
            model_masks = np.transpose(self.prediction_masks, (2, 1, 0))
            output_path = save_prediction_rois(model_masks, pathlib.Path(selected_path))
            QMessageBox.information(
                self,
                "ROI sauvegardées",
                f"Les ROI ont été sauvegardées dans :\n{output_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Erreur d'export",
                f"Impossible de sauvegarder les ROI :\n{exc}",
            )

    def show_previous_slice(self):
        self.slice_slider.setValue(max(0, self.slice_slider.value() - 1))

    def show_next_slice(self):
        self.slice_slider.setValue(
            min(self.slice_slider.maximum(), self.slice_slider.value() + 1)
        )

    def _update_navigation_buttons(self):
        value = self.slice_slider.value()
        self.previous_button.setEnabled(value > self.slice_slider.minimum())
        self.next_button.setEnabled(value < self.slice_slider.maximum())

    def show_slice(self, index: int):
        """Affiche la coupe courante sans créer de pixmap redimensionné."""
        if self.volume is None:
            return
        if (
            self._contour_edit_state is not None
            and self._contour_edit_state.slice_index != index
        ):
            self._finish_contour_point_edit()
        if (
            self._active_contour_slice is not None
            and self._active_contour_slice != index
        ):
            self._clear_selected_contour()

        slice_data = self.volume[:, :, index]
        # Rotation de 90 degrés pour respecter l'orientation du scanner.
        # Transposition pour respecter l'orientation du scanner.
        slice_data = np.transpose(slice_data)
        slice_data = np.rot90(slice_data, k=4)

        
        pixmap = slice_to_pixmap(
            slice_data,
            self.contrast_slider.value(),
            self.intensity_slider.value(),
            self.saturation_slider.value(),
        )
        self.image_label.set_mri_pixmap(pixmap)
        if self.prediction_masks is not None:
            outline_mask = self.prediction_outline_masks[:, :, index]
            outline_mask = np.rot90(np.transpose(outline_mask), k=4)
            erased_outline = self.erased_outline_masks[:, :, index]
            erased_outline = np.rot90(np.transpose(erased_outline), k=4)
            self.image_label.set_prediction_overlay(
                outline_mask,
                erased_outline,
            )
        else:
            self.image_label.set_prediction_overlay(None)
        if (
            self.contour_edit_active
            and self._contour_edit_state is None
            and self._active_contour_slice == index
            and self._active_contour_seed is not None
        ):
            self.select_contour_for_edit(
                self._active_contour_seed,
                preserve_control_points=True,
            )
        self._update_navigation_buttons()
        if index < self.result_table.rowCount():
            self.result_table.blockSignals(True)
            self.result_table.setCurrentCell(index, 0)
            self.result_table.blockSignals(False)
        self.slice_label.setText(
            f"Coupe {index + 1} / {self.volume.shape[2]}"
        )
        self.update_history_buttons()

    def image_fit(self):
        """Réinitialise le zoom et affiche toute la coupe IRM."""
        self.image_label.fit_image()

    def closeEvent(self, event):
        if self.contour_edit_active:
            self.set_contour_edit_mode(False)
        else:
            self._finish_contour_point_edit()
        if self.roi_only:
            self._auto_save_roi_modifications()
        super().closeEvent(event)
        

if __name__ == "__main__":
    app = QApplication([])

    app.setStyleSheet(
        """
        QPushButton {
            border: 1px solid #a6a6a6;
            border-radius: 6px;
        }
        QPushButton:pressed:enabled {
            background-color: #d6d6d6;
            border-color: #777777;
            border-style: inset;
        }
        QPushButton:disabled {
            color: #888888;
            background-color: #c8c8c8;
            border: 1px solid #c2c2c2;
        }
        """
    )

    window = MainWindow()
    window.show()

    app.exec()
