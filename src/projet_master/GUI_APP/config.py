import logging
import os
import pathlib
import sys
from logging.handlers import RotatingFileHandler


PACKAGE_DIRECTORY = pathlib.Path(__file__).resolve().parent
RESOURCE_DIRECTORY = PACKAGE_DIRECTORY / "assets"
PROJECT_ROOT = PACKAGE_DIRECTORY.parents[2]
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

LOGO_PATH = RESOURCE_DIRECTORY / "logo.png"
ETAPLAB_LOGO_PATH = RESOURCE_DIRECTORY / "etaplablogo.png"


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
