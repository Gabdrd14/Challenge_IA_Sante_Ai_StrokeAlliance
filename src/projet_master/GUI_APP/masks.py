import cv2
import numpy as np


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
