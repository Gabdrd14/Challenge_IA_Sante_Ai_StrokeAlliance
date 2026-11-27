import pathlib
import zipfile

import cv2
import numpy as np
from roifile import ImagejRoi, ROI_TYPE, roiread, roiwrite


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
