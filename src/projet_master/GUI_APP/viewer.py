import os
import pathlib

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QIcon, QImage, QKeyEvent, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import (
    ETAPLAB_LOGO_PATH,
    LOGGER,
    LOGO_PATH,
    MASK_ERASER_DIAMETER,
    VOXEL_AREA_MM2,
    VOXEL_AREA_MM2_RAT,
    VOXEL_VOLUME_MM3,
    VOXEL_VOLUME_MM3_RAT,
)
from .data import load_data, model_scan_index, prepare_volume, scan_label
from .image_view import ContourPointEdit, MRIImageView, slice_to_pixmap
from .masks import replace_erased_contour_segment
from .prediction import SlicePredictor, predict_pvdataset_volume
from .roi import roiread, save_modified_roi_slice, save_prediction_rois
from .settings import (
    available_checkpoints,
    save_mask_drawing_method,
    saved_mask_drawing_method,
)


class MRIViewer(QWidget):
    """Fenêtre indépendante pour parcourir les coupes IRM."""

    def __init__(
        self,
        file_path: str,
        model_directory: pathlib.Path,
        dataset_paths: list[str] | None = None,
        roi_zip_path: pathlib.Path | None = None,
        roi_directory: pathlib.Path | None = None,
        model_species: str = "souris",
        study_prediction_masks: dict[str, np.ndarray] | None = None,
    ):
        super().__init__()

        self.file_path = file_path
        self.model_directory = model_directory
        self.roi_zip_path = roi_zip_path
        self.roi_directory = roi_directory
        self.roi_only = roi_zip_path is not None
        self.model_species = model_species
        self.study_prediction_masks = study_prediction_masks or {}
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

        species_label = QLabel("Type :")
        data_layout.addWidget(species_label)
        self.souris_button = QPushButton("Souris")
        self.rat_button = QPushButton("Rat")
        self.souris_button.setCheckable(True)
        self.rat_button.setCheckable(True)
        self.souris_button.setFixedWidth(70)
        self.rat_button.setFixedWidth(70)
        self.souris_button.setChecked(self.model_species == "souris")
        self.rat_button.setChecked(self.model_species == "rat")

        def select_species(species: str) -> None:
            if species == self.model_species:
                return
            self.model_species = species
            self.souris_button.setChecked(species == "souris")
            self.rat_button.setChecked(species == "rat")
            self._on_species_changed()

        self.souris_button.clicked.connect(lambda: select_species("souris"))
        self.rat_button.clicked.connect(lambda: select_species("rat"))
        data_layout.addWidget(self.souris_button)
        data_layout.addWidget(self.rat_button)

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
        if (
            getattr(self, "model_species", "souris") == "rat"
            or "rat" in pathlib.Path(model_name).stem.casefold()
        ):
            return VOXEL_AREA_MM2_RAT, VOXEL_VOLUME_MM3_RAT
        return VOXEL_AREA_MM2, VOXEL_VOLUME_MM3

    def _on_model_changed(self, _index: int):
        if not self.all_scans:
            return
        model_path = self.model_combo.currentData()
        scan_index = model_scan_index(
            self.all_scans,
            model_path,
            getattr(self, "model_species", "souris"),
        )
        if scan_index != self.analysis_combo.currentIndex():
            self.analysis_combo.setCurrentIndex(scan_index)
        self._refresh_result_list()

    def _on_species_changed(self) -> None:
        """Met à jour les constantes voxel et le canal IRM par défaut."""
        if self.all_scans:
            model_path = self.model_combo.currentData()
            scan_index = model_scan_index(
                self.all_scans,
                model_path,
                self.model_species,
            )
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

        study_prediction_mask = self.study_prediction_masks.get(
            str(pathlib.Path(self.file_path).resolve())
        )
        if study_prediction_mask is not None:
            model_masks = np.asarray(
                study_prediction_mask,
                dtype=np.uint8,
            )
            if model_masks.ndim == 3:
                masks = np.transpose(model_masks, (2, 1, 0))
                if masks.shape == (self.volume.shape[1], self.volume.shape[0], self.volume.shape[2]):
                    self.prediction_masks = masks
                    self.original_prediction_masks = masks.copy()
                    self.prediction_outline_masks = masks.copy()
                    self.erased_outline_masks = np.zeros_like(
                        masks,
                        dtype=bool,
                    )
                    self.current_prediction_key = None
                    self.reset_mask_history()
                    self.original_prediction_masks = masks.copy()
                    self.prediction_outline_masks = masks.copy()
                    self.erased_outline_masks = np.zeros_like(
                        masks,
                        dtype=bool,
                    )
                    self.save_roi_button.setEnabled(True)
                    self.crop_mask_button.setEnabled(True)
                    self.add_mask_button.setEnabled(True)
                    self.edit_zone_button.setEnabled(True)
                    self._set_prediction_ready(True)
                    self._refresh_result_list()
                    self.show_slice(self.slice_slider.value())
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
                            getattr(self, "model_species", "souris"),
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
            if self.roi_only:
                self._load_roi_masks()
            elif not self._restore_prediction_cache(self.file_path, selected_scan_id):
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
            if self.roi_only:
                self._load_roi_masks()
            elif not self._restore_prediction_cache(self.file_path, scan_id):
                self.show_slice(0)
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
            model_path = self.model_combo.currentData()
            if not isinstance(model_path, pathlib.Path):
                raise ValueError("Aucun checkpoint de modèle sélectionné.")
            model_mask = predict_pvdataset_volume(
                self.predictor,
                self.file_path,
                scan_id,
                model_path,
            )
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
