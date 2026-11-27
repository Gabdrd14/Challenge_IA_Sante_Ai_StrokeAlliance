import pathlib

from .config import (
    CONFIG_DIRECTORY,
    LOGGER,
    MASK_DRAWING_METHOD_FILE,
    MODEL_DIRECTORY_FILE,
)


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
