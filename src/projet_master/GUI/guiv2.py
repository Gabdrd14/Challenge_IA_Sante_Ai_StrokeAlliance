
import os
import pathlib
import sys
import logging
from logging.handlers import RotatingFileHandler
import cv2
import numpy as np
import brkraw
from roifile import ImagejRoi, ROI_TYPE, roiwrite

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QSlider,
    QComboBox,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPolygonItem,
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
LOG_FILE = CONFIG_DIRECTORY / "guiv2.log"
PREDICTION_THRESHOLD = 0.5
VOXEL_WIDTH_MM = 0.07
VOXEL_HEIGHT_MM = 0.07
VOXEL_DEPTH_MM = 0.5
VOXEL_AREA_MM2 = VOXEL_WIDTH_MM * VOXEL_HEIGHT_MM
VOXEL_VOLUME_MM3 = VOXEL_AREA_MM2 * VOXEL_DEPTH_MM
LOGO_PATH = pathlib.Path(__file__).resolve().parent / "logo.png"
ETAPLAB_LOGO_PATH = pathlib.Path(__file__).resolve().parent / "etaplablogo.png"


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
    """Sélectionne par défaut le scan 2 parmi les T2 haute résolution."""
    high_resolution_scans = [
        (index, scan_id, scan_info)
        for index, (scan_id, scan_info) in enumerate(all_scans.items())
        if "t2_haute_resolution" in str(scan_info.get("Protocol", "")).lower()
    ]
    if not high_resolution_scans:
        return 0

    for index, scan_id, _ in high_resolution_scans:
        if str(scan_id) == "2":
            return index

    return high_resolution_scans[min(1, len(high_resolution_scans) - 1)][0]


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
    if raw_volume.shape[0] == 256 and raw_volume.shape[1] == 256:
        model_volume = np.transpose(raw_volume, (2, 1, 0))
    elif raw_volume.shape[1] == 256 and raw_volume.shape[2] == 256:
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
) -> pathlib.Path:
    """Enregistre les contours prédits comme ROI ImageJ dans un ZIP choisi."""
    masks = np.asarray(prediction_masks)
    if masks.ndim != 3:
        raise ValueError(f"Masques attendus sous la forme (D,H,W), reçu {masks.shape}.")

    rois = []
    for slice_index, mask in enumerate(masks):
        binary_mask = (mask > 0).astype(np.uint8)
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
        if not valid_contours:
            continue

        path_commands = []
        for contour in valid_contours:
            path_commands.extend([0.0, *contour[0]])
            for point in contour[1:]:
                path_commands.extend([1.0, *point])
            path_commands.append(4.0)

        all_points = np.concatenate(valid_contours)
        roi = ImagejRoi(
            roitype=ROI_TYPE.RECT,
            name=f"{slice_index + 1:03d}",
            position=slice_index,
            left=int(np.floor(all_points[:, 0].min())),
            top=int(np.floor(all_points[:, 1].min())),
            right=int(np.ceil(all_points[:, 0].max())) + 1,
            bottom=int(np.ceil(all_points[:, 1].max())) + 1,
            shape_roi_size=len(path_commands),
            multi_coordinates=np.asarray(path_commands, dtype=np.float32),
        )
        rois.append(roi)

    if not rois:
        raise ValueError("Aucun contour prédit à exporter en ROI.")
    output_path = pathlib.Path(output_path)
    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")
    roiwrite(output_path, rois)
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


def add_prediction_overlay(pixmap: QPixmap, mask: np.ndarray) -> QPixmap:
    """Dessine le contour rouge de prédiction utilisé par test_model.py."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Masque 2D attendu, reçu {mask.shape}.")

    if not mask.any():
        return pixmap

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    contour = mask & ~interior

    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[contour] = (255, 0, 0, 255)
    rgba = np.ascontiguousarray(rgba)
    overlay = QImage(
        rgba.data,
        mask.shape[1],
        mask.shape[0],
        rgba.strides[0],
        QImage.Format.Format_RGBA8888,
    ).copy()

    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.drawImage(0, 0, overlay)
    painter.end()
    return result


class MRIImageView(QGraphicsView):
    """Vue IRM à résolution native avec zoom et déplacement à la souris."""

    edit_path_finished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        self._has_image = False
        self._user_zoomed = False
        self._edit_mode = False
        self._edit_points = []
        self._edit_item = None

        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMinimumSize(300, 300)
        self.setToolTip(
            "Mouse wheel: zoom | Left-click and drag: pan | Double-click: Reset zoom"
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

    def set_edit_mode(self, enabled: bool):
        """Active le dessin libre à la place du déplacement pour éditer le masque."""
        self._edit_mode = enabled
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

    def mousePressEvent(self, event):
        if self._edit_mode and event.button() == Qt.MouseButton.LeftButton:
            self._edit_points = [
                self.mapToScene(event.position().toPoint())
            ]
            if self._edit_item is not None:
                self._scene.removeItem(self._edit_item)
            self._edit_item = QGraphicsPolygonItem()
            self._edit_item.setPen(QPen(Qt.GlobalColor.yellow, 2))
            self._scene.addItem(self._edit_item)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._edit_mode and self._edit_points:
            current = self.mapToScene(event.position().toPoint())
            if not self._edit_points[-1] == current:
                self._edit_points.append(current)
            self._edit_item.setPolygon(QPolygonF(self._edit_points))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._edit_mode
            and self._edit_points
            and event.button() == Qt.MouseButton.LeftButton
        ):
            current = self.mapToScene(event.position().toPoint())
            self._edit_points.append(current)
            points = self._edit_points
            self._edit_points = []
            if self._edit_item is not None:
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
            if len(points) >= 3:
                self.edit_path_finished.emit(points)
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
    ):
        super().__init__()

        self.file_path = file_path
        self.model_directory = model_directory
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
        self.all_scans = {}
        self.scan_ids = []
        self.predictor = SlicePredictor()
        self.model_paths = []
        self.mask_edit_mode = None
        self.mask_history = []
        self.mask_history_index = -1

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
        data_layout.addWidget(self.model_combo)

        prediction_group = QGroupBox("Prédiction et export")
        prediction_layout = QHBoxLayout(prediction_group)
        prediction_layout.setContentsMargins(6, 4, 6, 4)
        prediction_layout.setSpacing(4)
        self.analyze_button = QPushButton("Analyser toutes les coupes")
        self.analyze_button.setFixedWidth(190)
        self.analyze_button.setFixedHeight(30)
        self.analyze_button.clicked.connect(self.analyze_volume)
        prediction_layout.addWidget(self.analyze_button)

        self.save_roi_button = QPushButton("Sauvegarder les ROI")
        self.save_roi_button.setToolTip(
            "Exporter les contours prédits dans un fichier ROI ZIP"
        )
        self.save_roi_button.setFixedWidth(150)
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
        self.crop_mask_button = QPushButton("Rogner le masque")
        self.crop_mask_button.setEnabled(False)
        self.crop_mask_button.setCheckable(True)
        self.crop_mask_button.setFixedWidth(150)
        self.crop_mask_button.setFixedHeight(30)
        self.crop_mask_button.clicked.connect(
            lambda checked: self.set_mask_edit_mode("crop" if checked else None)
        )
        edit_layout.addWidget(self.crop_mask_button)

        self.add_mask_button = QPushButton("Ajouter une zone")
        self.add_mask_button.setEnabled(False)
        self.add_mask_button.setCheckable(True)
        self.add_mask_button.setFixedWidth(140)
        self.add_mask_button.setFixedHeight(30)
        self.add_mask_button.clicked.connect(
            lambda checked: self.set_mask_edit_mode("add" if checked else None)
        )
        edit_layout.addWidget(self.add_mask_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setFixedWidth(80)
        self.undo_button.setFixedHeight(30)
        self.undo_button.clicked.connect(self.undo_mask_edit)
        edit_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.setEnabled(False)
        self.redo_button.setFixedWidth(80)
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
        self.fit_button = QPushButton("Reset zoom")
        self.fit_button.setFixedWidth(100)
        self.fit_button.setFixedHeight(26)
        self.fit_button.setToolTip("Fit the entire MRI slice in the viewer")
        self.fit_button.clicked.connect(self.image_fit)
        viewer_controls_layout.addWidget(self.fit_button)
        image_layout.addLayout(viewer_controls_layout)

        # Le viewer et les résultats occupent la même ligne.
        content_layout = QHBoxLayout()
        self.previous_button = QPushButton("<")
        self.previous_button.setToolTip("Coupe précédente")
        self.previous_button.clicked.connect(self.show_previous_slice)
        content_layout.addWidget(self.previous_button)
        content_layout.addWidget(image_panel, stretch=19)
        self.next_button = QPushButton(">")
        self.next_button.setToolTip("Coupe suivante")
        self.next_button.clicked.connect(self.show_next_slice)
        content_layout.addWidget(self.next_button)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        self.result_label = QLabel("Résultat de la prédiction")
        self.result_volume_label = QLabel("Volume total : -")
        file_navigation_layout = QHBoxLayout()
        self.previous_file_button = QPushButton("Sujet précédent")
        self.previous_file_button.clicked.connect(
            lambda: self.navigate_file(-1)
        )
        self.next_file_button = QPushButton("Sujet suivant")
        self.next_file_button.clicked.connect(
            lambda: self.navigate_file(1)
        )
        file_navigation_layout.addWidget(self.previous_file_button)
        file_navigation_layout.addWidget(self.next_file_button)
        self.result_table = QTableWidget(0, 3)
        self.result_table.setHorizontalHeaderLabels(
            ["Coupe", "Voxels", "Aire (mm²)"]
        )
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
        self._update_file_navigation()
        self.load_file()

    def load_models(self):
        """Remplit le sélecteur avec les checkpoints du dossier configuré."""
        try:
            self.model_paths = available_checkpoints(self.model_directory)
        except FileNotFoundError as exc:
            self.model_combo.clear()
            self.model_combo.addItem("Aucun modèle disponible")
            self.model_combo.setEnabled(False)
            self.analyze_button.setEnabled(False)
            raise exc

        self.model_combo.clear()
        for checkpoint in self.model_paths:
            self.model_combo.addItem(
                str(checkpoint.relative_to(self.model_directory)),
                checkpoint,
            )
        self.model_combo.setCurrentIndex(
            max(
                range(len(self.model_paths)),
                key=lambda index: self.model_paths[index].stat().st_mtime,
            )
        )

    def load_file(self):
        """Charge le scan IRM sélectionné."""
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
                        default_scan_index(all_scans)
                    )
            finally:
                self.analysis_combo.blockSignals(False)

            selected_scan_id = self.analysis_combo.currentData()
            data, _, _ = load_data(self.file_path, selected_scan_id)
            if data is None:
                self.slice_label.setText("No MRI volume available")
                return

            self.volume = prepare_volume(data)
            self.prediction_masks = None
            self.original_prediction_masks = None
            self.result_table.setRowCount(0)
            self.result_label.setText("Résultat de la prédiction")
            self.result_volume_label.setText("Volume total : -")
            self.reset_mask_history()
            self.save_roi_button.setEnabled(False)
            self.set_mask_edit_mode(None)
            self.crop_mask_button.setEnabled(False)
            self.add_mask_button.setEnabled(False)

            number_of_slices = self.volume.shape[2]

            self.slice_slider.setRange(0, number_of_slices - 1)
            self.slice_slider.setEnabled(number_of_slices > 1)

            # Commence à la première coupe afin que le curseur et les flèches
            # aient une origine claire.
            self.slice_slider.setValue(0)
            self.show_slice(0)
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
        scan_id = self.analysis_combo.itemData(index)
        try:
            data, _, _ = load_data(self.file_path, scan_id)
            if data is None:
                raise ValueError("Cette analyse ne contient pas de volume exploitable.")
            self.volume = prepare_volume(data)
            self.prediction_masks = None
            self.original_prediction_masks = None
            self.result_table.setRowCount(0)
            self.result_label.setText("Résultat de la prédiction")
            self.result_volume_label.setText("Volume total : -")
            self.reset_mask_history()
            self.save_roi_button.setEnabled(False)
            self.set_mask_edit_mode(None)
            self.crop_mask_button.setEnabled(False)
            self.add_mask_button.setEnabled(False)
            self.slice_slider.setRange(0, self.volume.shape[2] - 1)
            self.slice_slider.setValue(0)
            self.show_slice(0)
            self._update_navigation_buttons()
        except Exception as exc:
            QMessageBox.critical(self, "Analyse impossible", f"Impossible de charger cette analyse:\n{exc}")

    def analyze_volume(self):
        """Exécute le modèle choisi sur toutes les coupes du scan sélectionné."""
        if self.volume is None:
            return
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
            self._refresh_result_list()
            self.reset_mask_history()
            self.save_roi_button.setEnabled(True)
            self.crop_mask_button.setEnabled(True)
            self.add_mask_button.setEnabled(True)
            self.update_history_buttons()
            self.show_slice(self.slice_slider.value())
        except Exception as exc:
            QMessageBox.critical(self, "Erreur d'analyse", f"Prédiction impossible:\n{exc}")
        finally:
            self.analyze_button.setEnabled(True)

    def set_mask_edit_mode(self, mode: str | None):
        """Définit le mode de dessin libre actif pour le masque courant."""
        self.mask_edit_mode = mode
        self.image_label.set_edit_mode(mode is not None)
        if mode == "crop":
            self.add_mask_button.setChecked(False)
        elif mode == "add":
            self.crop_mask_button.setChecked(False)

    def _refresh_result_list(self):
        """Affiche les voxels et l'aire prédite de chaque coupe."""
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
            f"Volume total : {total_positive * VOXEL_VOLUME_MM3:.4f} mm³"
        )
        self.result_table.setRowCount(len(positive_voxels))
        for index, count in enumerate(positive_voxels):
            self.result_table.setItem(
                index, 0, QTableWidgetItem(str(index + 1))
            )
            self.result_table.setItem(
                index, 1, QTableWidgetItem(str(int(count)))
            )
            self.result_table.setItem(
                index,
                2,
                QTableWidgetItem(f"{count * VOXEL_AREA_MM2:.4f}"),
            )
        self.result_table.resizeColumnsToContents()

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
        self.mask_history_index = -1
        self.update_history_buttons()

    def update_history_buttons(self):
        self.undo_button.setEnabled(self.mask_history_index > 0)
        self.redo_button.setEnabled(
            0 <= self.mask_history_index < len(self.mask_history) - 1
        )
        self.reset_mask_button.setEnabled(
            self.original_prediction_masks is not None
            and self.prediction_masks is not None
            and not np.array_equal(
                self.prediction_masks,
                self.original_prediction_masks,
            )
        )

    def _record_mask_state(self):
        """Sauvegarde une copie du masque avant d'appliquer une modification."""
        if self.prediction_masks is None:
            return
        self.mask_history = self.mask_history[: self.mask_history_index + 1]
        self.mask_history.append(self.prediction_masks.copy())
        self.mask_history_index += 1
        self.update_history_buttons()

    def undo_mask_edit(self):
        if self.mask_history_index <= 0:
            return
        self.mask_history_index -= 1
        self.prediction_masks = self.mask_history[self.mask_history_index].copy()
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())

    def redo_mask_edit(self):
        if self.mask_history_index >= len(self.mask_history) - 1:
            return
        self.mask_history_index += 1
        self.prediction_masks = self.mask_history[self.mask_history_index].copy()
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())

    def reset_mask_edits(self):
        """Restaure la prédiction originale en supprimant les modifications."""
        if self.original_prediction_masks is None:
            return
        self.prediction_masks = self.original_prediction_masks.copy()
        self.mask_history = [self.prediction_masks.copy()]
        self.mask_history_index = 0
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())

    def apply_mask_edit_path(self, points):
        """Applique un polygone libre au masque de la coupe affichée."""
        if self.prediction_masks is None or self.mask_edit_mode is None:
            return

        displayed_mask = self.prediction_masks[:, :, self.slice_slider.value()]
        displayed_mask = np.ascontiguousarray(np.transpose(displayed_mask))
        height, width = displayed_mask.shape
        polygon = np.array(
            [
                [
                    max(0, min(width - 1, int(round(point.x())))),
                    max(0, min(height - 1, int(round(point.y())))),
                ]
                for point in points
            ],
            dtype=np.int32,
        )
        if len(polygon) < 3:
            return

        if not self.mask_history:
            self.mask_history = [self.prediction_masks.copy()]
            self.mask_history_index = 0
        else:
            self._record_mask_state()

        edit_value = 0 if self.mask_edit_mode == "crop" else 1
        cv2.fillPoly(displayed_mask, [polygon], edit_value)

        # L'image affichée est la transposée de la coupe stockée par le viewer.
        self.prediction_masks[:, :, self.slice_slider.value()] = np.transpose(
            displayed_mask
        )
        self.mask_history = self.mask_history[: self.mask_history_index + 1]
        self.mask_history.append(self.prediction_masks.copy())
        self.mask_history_index += 1
        self.update_history_buttons()
        self._refresh_result_list()
        self.show_slice(self.slice_slider.value())

    def save_prediction_as_rois(self):
        """Exporte la prédiction courante vers un ZIP choisi par l'utilisateur."""
        if self.prediction_masks is None:
            QMessageBox.information(
                self,
                "Aucune prédiction",
                "Lancez une analyse avant de sauvegarder les ROI.",
            )
            return

        try:
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
        if self.prediction_masks is not None:
            mask = self.prediction_masks[:, :, index]
            mask = np.transpose(mask)
            mask = np.rot90(mask, k=4)
            pixmap = add_prediction_overlay(pixmap, mask)
        self.image_label.set_mri_pixmap(pixmap)
        self._update_navigation_buttons()
        if index < self.result_table.rowCount():
            self.result_table.blockSignals(True)
            self.result_table.setCurrentCell(index, 0)
            self.result_table.blockSignals(False)
        self.slice_label.setText(
            f"Coupe {index + 1} / {self.volume.shape[2]}"
        )

    def image_fit(self):
        """Réinitialise le zoom et affiche toute la coupe IRM."""
        self.image_label.fit_image()
        

       
       
        
            
            
            
                    
   
    


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec()
