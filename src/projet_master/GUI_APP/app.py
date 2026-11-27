import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

if not __package__:
    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root))

from projet_master.GUI_APP.main_window import MainWindow


def main() -> None:
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


if __name__ == "__main__":
    main()
