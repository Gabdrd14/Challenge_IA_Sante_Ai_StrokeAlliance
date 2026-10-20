import os
import cv2
import torch
import brkraw
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QLayout, QLabel, QApplication, QVBoxLayout, QPushButton, QDialog, QErrorMessage, QFileDialog

#MODEL = torch.load("model.pt")
PROTOCOL = "T2_haute_resolution_Turbo"

def load_data(file_path: str) -> torch.Tensor | None:
    loader = brkraw.load(file_path)
    info = loader.info(scope="scan", as_dict=True)
    slices = None
    for scan_id, scan in info.items():
        if scan.get("Protocol") == PROTOCOL:
            slices = loader.get_dataobj(scan_id)
            break
    if slices is None:
        return None
    cv2.normalize(slices, slices, 0, 255, cv2.NORM_MINMAX)
    return torch.tensor(slices)

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.layout: QLayout = QVBoxLayout(self)

        self.browse_button = QPushButton("Parcourir")
        self.browse_button.clicked.connect(self.open_browsing_dialog)
        self.layout.addWidget(self.browse_button)

        self.background_text = QLabel("Glissez-déposez un fichier ici, ou parcourez vos fichiers.")
        self.background_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.background_text.setStyleSheet("""
            color: #888;
            font-size: 18px;
            border-radius: 10px;
            padding: 40px;
        """)
        self.layout.addWidget(self.background_text)

        self.setAcceptDrops(True)
        self.resize(600, 600)
        self.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            dir_path = urls[0].toLocalFile()
            if self.check_dir(dir_path):
                self.background_text.hide()
                self.open_confirm_dialog(dir_path)
            else:
                self.show_error_msg(f"Mauvais dossier : [{dir_path}].")
        else:
            event.ignore()

    def check_dir(self, dir_path: str) -> bool:
        if not dir_path:
            return False
        if not os.path.isdir(dir_path):
            return False
        for entry in os.scandir(dir_path):
            if not entry.path.endswith(".PvDatasets"):
                return False
        return True

    def show_error_msg(self, error_msg: str) -> None:
        error_dialog = QErrorMessage(self)
        error_dialog.showMessage(error_msg)

    def predict(self, dir_path: str) -> torch.Tensor | None:
        for file in os.scandir(dir_path):
            data = load_data(file.path)
            if data is None:
                self.show_error_msg(f"Protocole {PROTOCOL} non disponible pour le fichier [{file.path}].")
                return None
        #return MODEL(data)
            print("predicted")

    def open_confirm_dialog(self, dir_path: str):
        dialog = ConfirmDialog(self, dir_path)
        dialog.accepted.connect(lambda: self.predict(dir_path))
        dialog.rejected.connect(self.background_text.show)
        dialog.show()

    def open_browsing_dialog(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self)
        if not self.check_dir(dir_path):
            self.show_error_msg(f"Mauvais dossier : [{dir_path}]")
            return None
        print(dir_path)
        for entry in os.scandir(dir_path):
            print(entry.path)
            

class ConfirmDialog(QDialog):

    def __init__(self, parent: MainWindow, file_path: str):
        super().__init__(parent)

        self.setWindowTitle("Confirmation")
        self.resize(300, 150)
        layout = QVBoxLayout(self)

        label = QLabel(f"Traiter ce dossier de fichiers : [{file_path}] ?")
        layout.addWidget(label)

        confirm_button = QPushButton("Confirmer")
        other_button = QPushButton("Choisir un autre dossier")
        confirm_button.clicked.connect(self.accept)
        other_button.clicked.connect(self.reject)
        layout.addWidget(confirm_button)
        layout.addWidget(other_button)


if __name__ == "__main__":
    from tempfile import mkstemp
    dir_path = "./fake_dir"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for i in range(5):
        fd, temp = mkstemp(suffix=f"_fake_file_{i+1}.PvDatasets", dir=dir_path)
        os.close(fd)
    app = QApplication([])
    w = MainWindow()
    app.exec()
    for entry in os.scandir(dir_path):
        os.remove(entry.path)
    os.rmdir(dir_path)