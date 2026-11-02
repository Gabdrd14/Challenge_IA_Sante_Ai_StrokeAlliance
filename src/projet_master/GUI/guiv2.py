
import os
import pathlib
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
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPolygonItem,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL_DIRECTORY_FILE = PROJECT_ROOT / "model_directory.txt"
PREDICTION_THRESHOLD = 0.5
LOGO_PATH = pathlib.Path(__file__).resolve().parent / "logo.png"

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
        import importlib.util

        model_path = PROJECT_ROOT / "segment_stroke_3DUnet.py"
        module_spec = importlib.util.spec_from_file_location(
            "projet_master_3d_unet",
            model_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Impossible de charger le modèle {model_path}.")
        model_module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(model_module)
        AVCSegModel = model_module.AVCSegModel

        self.model = AVCSegModel.load_from_checkpoint(
            str(checkpoint),
            map_location=self.device,
        ).to(self.device)
        self.model.eval()
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
    window: tuple[float, float] | None = None,
) -> QPixmap:
    """Convertit une coupe IRM pour l'affichage sans modifier les intensités.

    Une fenêtre calculée sur le volume complet doit être fournie afin que le
    contraste reste stable entre les coupes. Seule la copie d'affichage est
    convertie en niveaux de gris 8 bits ; le volume original reste inchangé.
    """
    slice_data = np.asarray(slice_data)
    if slice_data.ndim != 2:
        raise ValueError(f"Expected a 2D MRI slice, got {slice_data.shape}")

    finite = np.isfinite(slice_data)
    if not finite.any():
        raise ValueError("MRI slice contains no finite intensity values.")

    values = slice_data[finite].astype(np.float64, copy=False)
    if window is None:
        # Valeur de secours pour une utilisation isolée. Le viewer fournit
        # une fenêtre calculée sur l'ensemble du volume.
        low, high = np.percentile(values, (1.0, 99.5))
    else:
        low, high = window

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0

    # Application de la fenêtre sur une copie destinée à l'affichage.
    # Aucun filtrage ni interpolation n'est appliqué aux pixels natifs.
    scaled = (slice_data.astype(np.float64, copy=False) - low) * (255.0 / (high - low))
    normalized = np.zeros(slice_data.shape, dtype=np.uint8)
    normalized[finite] = np.clip(scaled[finite], 0, 255).round().astype(np.uint8)
    normalized = np.ascontiguousarray(normalized)

    height, width = normalized.shape
    image = QImage(
        normalized.data,
        width,
        height,
        normalized.strides[0],
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
        self.resize(650, 500)
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
            viewer = MRIViewer(file_path, model_directory)
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

    def __init__(self, file_path: str, model_directory: pathlib.Path):
        super().__init__()

        self.file_path = file_path
        self.model_directory = model_directory
        self.volume = None
        self.prediction_masks = None
        self.all_scans = {}
        self.scan_ids = []
        self.predictor = SlicePredictor()
        self.model_paths = []
        self.display_window = None
        self.mask_edit_mode = None
        self.mask_history = []
        self.mask_history_index = -1

        self.setWindowTitle(
            f"MRI Viewer - {os.path.basename(file_path)}"
        )
        self.resize(1100, 750)

        main_layout = QVBoxLayout(self)

        # Un libellé contenant le chemin peut être ajouté ici si nécessaire.
        
        
        toolbar = QVBoxLayout()

        data_group = QGroupBox("Données")
        data_layout = QHBoxLayout(data_group)
        self.analysis_combo = QComboBox()
        self.analysis_combo.setToolTip("Choisir le type d'analyse présent dans le PvDataset")
        self.analysis_combo.currentIndexChanged.connect(self.change_scan)
        data_layout.addWidget(self.analysis_combo, stretch=1)

        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Choisir le checkpoint du modèle")
        data_layout.addWidget(self.model_combo, stretch=1)
        toolbar.addWidget(data_group)

        prediction_group = QGroupBox("Prédiction et export")
        prediction_layout = QHBoxLayout(prediction_group)
        self.analyze_button = QPushButton("Analyser toutes les coupes")
        self.analyze_button.clicked.connect(self.analyze_volume)
        prediction_layout.addWidget(self.analyze_button)

        self.save_roi_button = QPushButton("Sauvegarder les ROI")
        self.save_roi_button.setToolTip(
            "Exporter les contours prédits dans un fichier ROI ZIP"
        )
        self.save_roi_button.setEnabled(False)
        self.save_roi_button.clicked.connect(self.save_prediction_as_rois)
        prediction_layout.addWidget(self.save_roi_button)
        toolbar.addWidget(prediction_group)

        edit_group = QGroupBox("Édition du masque")
        edit_layout = QHBoxLayout(edit_group)
        self.crop_mask_button = QPushButton("Rogner le masque")
        self.crop_mask_button.setEnabled(False)
        self.crop_mask_button.setCheckable(True)
        self.crop_mask_button.clicked.connect(
            lambda checked: self.set_mask_edit_mode("crop" if checked else None)
        )
        edit_layout.addWidget(self.crop_mask_button)

        self.add_mask_button = QPushButton("Ajouter une zone")
        self.add_mask_button.setEnabled(False)
        self.add_mask_button.setCheckable(True)
        self.add_mask_button.clicked.connect(
            lambda checked: self.set_mask_edit_mode("add" if checked else None)
        )
        edit_layout.addWidget(self.add_mask_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_mask_edit)
        edit_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo")
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self.redo_mask_edit)
        edit_layout.addWidget(self.redo_button)
        toolbar.addWidget(edit_group)

        navigation_group = QGroupBox("Navigation")
        navigation_layout = QHBoxLayout(navigation_group)
        self.previous_button = QPushButton("<")
        self.previous_button.setToolTip("Coupe précédente")
        self.previous_button.clicked.connect(self.show_previous_slice)
        navigation_layout.addWidget(self.previous_button)

        self.next_button = QPushButton(">")
        self.next_button.setToolTip("Coupe suivante")
        self.next_button.clicked.connect(self.show_next_slice)
        navigation_layout.addWidget(self.next_button)
        

        self.fit_button = QPushButton("Reset zoom")
        self.fit_button.setToolTip("Fit the entire MRI slice in the viewer")
        self.fit_button.clicked.connect(self.image_fit)
        navigation_layout.addWidget(self.fit_button)
        toolbar.addWidget(navigation_group)
        main_layout.addLayout(toolbar)

        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)

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

        main_layout.addWidget(image_panel)

        self.load_models()
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
            data, _, all_scans = load_data(self.file_path)
            self.all_scans = all_scans
            self.scan_ids = list(all_scans)
            self.analysis_combo.blockSignals(True)
            self.analysis_combo.clear()
            for scan_id, scan_info in all_scans.items():
                self.analysis_combo.addItem(scan_label(scan_id, scan_info), scan_id)
            self.analysis_combo.blockSignals(False)

            if data is None:
                self.slice_label.setText("No MRI volume available")
                return

            self.volume = prepare_volume(data)
            self.prediction_masks = None
            self.reset_mask_history()
            self.save_roi_button.setEnabled(False)
            self.set_mask_edit_mode(None)
            self.crop_mask_button.setEnabled(False)
            self.add_mask_button.setEnabled(False)

            # Calcule une fenêtre de contraste robuste sur le volume complet
            # pour conserver une luminosité stable entre les coupes.
            finite_values = self.volume[np.isfinite(self.volume)]
            if finite_values.size:
                low, high = np.percentile(finite_values, (1.0, 99.5))
                if high <= low:
                    low, high = float(finite_values.min()), float(finite_values.max())
                self.display_window = (float(low), float(high))

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
            self.reset_mask_history()
            self.save_roi_button.setEnabled(False)
            self.set_mask_edit_mode(None)
            self.crop_mask_button.setEnabled(False)
            self.add_mask_button.setEnabled(False)
            self.display_window = None
            finite_values = self.volume[np.isfinite(self.volume)]
            if finite_values.size:
                low, high = np.percentile(finite_values, (1.0, 99.5))
                self.display_window = (float(low), float(high))
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
        self.show_slice(self.slice_slider.value())

    def redo_mask_edit(self):
        if self.mask_history_index >= len(self.mask_history) - 1:
            return
        self.mask_history_index += 1
        self.prediction_masks = self.mask_history[self.mask_history_index].copy()
        self.update_history_buttons()
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

        
        pixmap = slice_to_pixmap(slice_data, self.display_window)
        if self.prediction_masks is not None:
            mask = self.prediction_masks[:, :, index]
            mask = np.transpose(mask)
            mask = np.rot90(mask, k=4)
            pixmap = add_prediction_overlay(pixmap, mask)
        self.image_label.set_mri_pixmap(pixmap)
        self._update_navigation_buttons()
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
