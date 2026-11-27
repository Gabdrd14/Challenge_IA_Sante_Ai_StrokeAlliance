import os
import pathlib

import numpy as np
from PySide6.QtCore import QDateTime, QTimer, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .config import LOGO_PATH, PROJECT_ROOT
from .data import find_dataset_files, load_data, scan_label
from .prediction import SlicePredictor, predict_pvdataset_volume, postprocess_mask
from .roi import save_prediction_rois
from .settings import (
    available_checkpoints,
    save_model_directory,
    saved_model_directory,
)
from .viewer import MRIViewer


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
        self.study_scan_protocol = None
        self.study_scan_id = None
        self.study_prediction_masks: dict[str, np.ndarray] = {}

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

    def choose_study_scan(self) -> tuple[object, str] | None:
        """Ouvre une fenêtre listant les canaux du premier dataset de l'étude."""
        if not self.dataset_paths:
            return None

        _, _, all_scans = load_data(self.dataset_paths[0])
        if not all_scans:
            QMessageBox.warning(
                self,
                "Aucun canal",
                "Le premier PvDataset ne contient aucun canal analysable.",
            )
            return None

        dialog = QDialog(self)
        dialog.setWindowTitle("Choisir le channel pour l'étude")
        dialog.setMinimumSize(520, 360)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Canaux disponibles dans le premier dataset"))

        scan_list = QListWidget()
        for scan_id, scan_info in all_scans.items():
            item = QListWidgetItem(
                scan_label(scan_id, scan_info)
            )
            item.setData(Qt.ItemDataRole.UserRole, scan_id)
            scan_list.addItem(item)
        layout.addWidget(scan_list)

        selected: list[tuple[object, str] | None] = [None]
        buttons = QHBoxLayout()
        cancel_button = QPushButton("Annuler")
        validate_button = QPushButton("Utiliser ce channel")

        def validate_selection() -> None:
            item = scan_list.currentItem()
            if item is None:
                return
            scan_id = item.data(Qt.ItemDataRole.UserRole)
            protocol = str(all_scans[scan_id].get("Protocol", ""))
            selected[0] = (scan_id, protocol)
            dialog.accept()

        cancel_button.clicked.connect(dialog.reject)
        validate_button.clicked.connect(validate_selection)
        scan_list.itemDoubleClicked.connect(
            lambda _item: validate_selection()
        )
        buttons.addWidget(cancel_button)
        buttons.addWidget(validate_button)
        layout.addLayout(buttons)

        if scan_list.count():
            scan_list.setCurrentRow(0)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return selected[0]
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

        selected_scan = self.choose_study_scan()
        if selected_scan is None:
            QMessageBox.information(
                self,
                "Channel non sélectionné",
                "Aucune analyse n'a été lancée.",
            )
            return
        self.study_scan_protocol = selected_scan[1]
        self.study_scan_id = selected_scan[0]
        
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
                    scan_id = next(
                        (
                            scan_id
                            for scan_id, scan_info in all_scans.items()
                            if str(scan_info.get("Protocol", ""))
                            == self.study_scan_protocol
                        ),
                        None,
                    )
                    if scan_id is None:
                        raise ValueError(
                            f"Channel introuvable: {self.study_scan_protocol}"
                        )
                    model_mask = predict_pvdataset_volume(
                        self.predictor,
                        file_path,
                        scan_id,
                        self.model_checkpoint,
                    )
                    model_mask = postprocess_mask(model_mask)
                    self.study_prediction_masks[
                        str(pathlib.Path(file_path).resolve())
                    ] = model_mask.copy()
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
                model_species=getattr(self, "model_species", "souris"),
                study_prediction_masks=self.study_prediction_masks,
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
            viewer = MRIViewer(
                file_path,
                model_directory,
                self.dataset_paths,
                model_species=getattr(self, "model_species", "souris"),
            )
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
