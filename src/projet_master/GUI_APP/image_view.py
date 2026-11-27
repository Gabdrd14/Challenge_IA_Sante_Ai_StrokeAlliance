from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QPixmap,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
)

from .config import MASK_ERASER_DIAMETER


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


@dataclass
class ContourPointEdit:
    slice_index: int
    contour: np.ndarray
    point_index: int
    marker_index: int
    current_position: tuple[int, int]
    component_mask: np.ndarray
    original_mask: np.ndarray
    original_outline: np.ndarray
    original_erased: np.ndarray
    original_total_voxels: int
    original_slice_voxels: int


class MRIImageView(QGraphicsView):
    """Vue IRM à résolution native avec zoom et déplacement à la souris."""

    edit_path_finished = Signal(object, bool)
    contour_region_selection_requested = Signal(object)
    contour_control_point_added = Signal(object)
    contour_control_point_removed = Signal(int)
    contour_point_drag_started = Signal(object, int, int)
    contour_point_dragged = Signal(object)
    contour_point_drag_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = self._scene.addPixmap(QPixmap())
        #self._pixmap_item.setTransformationMode(
            #Qt.TransformationMode.SmoothTransformation
        #)
        self._prediction_item: QGraphicsPathItem | None = None
        self._has_image = False
        self._user_zoomed = False
        self._edit_mode: str | None = None
        self._legacy_polygon_drawing = False
        self._contour_edit_mode = False
        self._active_contour: np.ndarray | None = None
        self._active_visible_indices = np.array([], dtype=np.intp)
        self._control_points: list[tuple[int, int]] = []
        self._control_point_items: list[QGraphicsEllipseItem] = []
        self._last_auto_added_control_point: tuple[int, int] | None = None
        self._contour_drag_active = False
        self._drag_marker_index = -1
        self._contour_drag_origin_scene: QPointF | None = None
        self._contour_drag_origin_global: QPointF | None = None
        self._contour_drag_scale_x = 1.0
        self._contour_drag_scale_y = 1.0
        self._pan_with_right_button = False
        self._pan_active = False
        self._pan_waiting_for_release = False
        self._pan_button: Qt.MouseButton | None = None
        self._pan_last_position = QPoint()
        self._edit_points = []
        self._edit_item = None
        self._drawing_active = False
        self._last_pointer_position = QPoint()
        self._auto_scroll_timer = QTimer(self)
        self._auto_scroll_timer.setInterval(30)
        self._auto_scroll_timer.timeout.connect(self._auto_scroll_while_drawing)

        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMinimumSize(300, 300)
        self.viewport().setMouseTracking(True)
        self.setToolTip(
            "Molette : zoom | Glisser un point du contour rouge pour l'ajuster | "
            "Glisser ailleurs : déplacer l'image"
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

    def set_prediction_overlay(
        self,
        mask: np.ndarray | None,
        erased_outline: np.ndarray | None = None,
    ):
        """Affiche le contour du masque en vectoriel, sans épaississement au zoom."""
        if self._prediction_item is not None:
            self._scene.removeItem(self._prediction_item)
            self._prediction_item = None
        if mask is None:
            return

        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError(f"Masque 2D attendu, reçu {mask.shape}.")
        if erased_outline is None:
            erased_outline = np.zeros(mask.shape, dtype=bool)
        else:
            erased_outline = np.asarray(erased_outline, dtype=bool)
            if erased_outline.shape != mask.shape:
                raise ValueError(
                    "Le masque d'effacement du contour doit avoir la même "
                    f"forme que le masque, reçu {erased_outline.shape} et {mask.shape}."
                )
        if not mask.any():
            return

        contours, _ = cv2.findContours(
            np.ascontiguousarray(mask, dtype=np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        path = QPainterPath()
        for contour in contours:
            points = contour[:, 0, :]
            if len(points) == 0:
                continue
            hidden = erased_outline[points[:, 1], points[:, 0]]
            visible_indices = np.flatnonzero(~hidden)
            if not hidden.any():
                path.moveTo(float(points[0, 0]), float(points[0, 1]))
                for point in points[1:]:
                    path.lineTo(float(point[0]), float(point[1]))
                path.closeSubpath()
                continue

            if not len(visible_indices):
                continue
            starts = []
            for index in visible_indices:
                previous = (index - 1) % len(points)
                if hidden[previous]:
                    starts.append(index)
            if not starts:
                starts.append(int(visible_indices[0]))
            for start in starts:
                path.moveTo(float(points[start, 0]), float(points[start, 1]))
                index = (start + 1) % len(points)
                while not hidden[index] and index != start:
                    path.lineTo(float(points[index, 0]), float(points[index, 1]))
                    index = (index + 1) % len(points)

        self._prediction_item = QGraphicsPathItem(path)
        pen = QPen(Qt.GlobalColor.red, 1.5)
        pen.setCosmetic(True)
        self._prediction_item.setPen(pen)
        self._scene.addItem(self._prediction_item)

    def set_contour_edit_mode(self, enabled: bool):
        self._contour_edit_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self._draw_control_points()
            self.setToolTip(
                "Cliquer sur une zone pour la sélectionner | "
                "Glisser un point du contour pour l'ajuster | "
                "Double-clic sur le contour ou dans la zone : ajouter un point | "
                "Clic droit sur un point : le supprimer | "
                "Clic droit + glisser : déplacer l'image"
            )
        else:
            self.viewport().unsetCursor()
            self._remove_control_point_items()
            self._contour_drag_active = False
            self._drag_marker_index = -1
            self._contour_drag_origin_scene = None
            self._contour_drag_origin_global = None
            self.setDragMode(
                QGraphicsView.DragMode.NoDrag
                if self._edit_mode is not None
                else QGraphicsView.DragMode.ScrollHandDrag
            )
            self.setToolTip(
                "Molette : zoom | Glisser ailleurs : déplacer l'image"
            )

    def set_active_contour(
        self,
        contour: np.ndarray | None,
        control_points: list[tuple[int, int]],
        erased_outline: np.ndarray | None = None,
    ):
        self._active_contour = (
            None
            if contour is None
            else np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2).copy()
        )
        self._control_points = list(control_points)
        if self._active_contour is None:
            self._active_visible_indices = np.array([], dtype=np.intp)
        elif erased_outline is None:
            self._active_visible_indices = np.arange(
                len(self._active_contour),
                dtype=np.intp,
            )
        else:
            points = self._active_contour[:, 0, :]
            hidden = np.asarray(erased_outline, dtype=bool)[
                points[:, 1],
                points[:, 0],
            ]
            self._active_visible_indices = np.flatnonzero(~hidden)
        if self._contour_edit_mode:
            self._draw_control_points()

    def _remove_control_point_items(self):
        for item in self._control_point_items:
            self._scene.removeItem(item)
        self._control_point_items = []

    def _draw_control_points(self):
        self._remove_control_point_items()
        if not self._contour_edit_mode or self._active_contour is None:
            return
        for x, y in self._control_points:
            item = QGraphicsEllipseItem(-3, -3, 6, 6)
            item.setBrush(QBrush(QColor(255, 255, 255, 220)))
            pen = QPen(Qt.GlobalColor.yellow, 1.5)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                True,
            )
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            item.setZValue(3)
            item.setPos(float(x), float(y))
            self._scene.addItem(item)
            self._control_point_items.append(item)

    def _control_point_index_at(
        self,
        position: QPoint,
        hit_radius: float = 10.0,
    ) -> int:
        hit_radius_squared = hit_radius**2
        closest_index = -1
        closest_distance = hit_radius_squared
        for index, (x, y) in enumerate(self._control_points):
            marker_position = self.mapFromScene(QPointF(float(x), float(y)))
            delta = marker_position - position
            distance_squared = delta.x() ** 2 + delta.y() ** 2
            if distance_squared <= closest_distance:
                closest_index = index
                closest_distance = distance_squared
        return closest_index

    def move_control_point(self, index: int, x: int, y: int):
        if not 0 <= index < len(self._control_points):
            return
        self._control_points[index] = (x, y)
        if index < len(self._control_point_items):
            self._control_point_items[index].setPos(float(x), float(y))

    def _contour_scene_position(self, global_position: QPointF) -> QPointF:
        if (
            self._contour_drag_origin_scene is None
            or self._contour_drag_origin_global is None
        ):
            raise RuntimeError("Le glissement de contour n'a pas été initialisé.")
        # Live volume updates can resize the view; global deltas keep the scene
        # anchor stable while the viewport is being repositioned.
        delta = global_position - self._contour_drag_origin_global
        return QPointF(
            self._contour_drag_origin_scene.x()
            + delta.x() / self._contour_drag_scale_x,
            self._contour_drag_origin_scene.y()
            + delta.y() / self._contour_drag_scale_y,
        )

    def _nearest_active_contour_point(
        self,
        position: QPointF,
        tolerance: float | None = None,
        include_erased: bool = False,
        excluded_points: set[tuple[int, int]] | None = None,
    ) -> int | None:
        if self._active_contour is None:
            return None
        visible_indices = (
            np.arange(len(self._active_contour), dtype=np.intp)
            if include_erased
            else self._active_visible_indices
        )
        if not len(visible_indices):
            return None
        scale = max(abs(self.transform().m11()), 1e-6)
        if tolerance is None:
            tolerance = 10.0 / scale
        query = np.array([position.x(), position.y()])
        points = self._active_contour[
            visible_indices,
            0,
            :,
        ].astype(float)
        if excluded_points:
            available = np.array(
                [
                    (int(point[0]), int(point[1])) not in excluded_points
                    for point in points
                ],
                dtype=bool,
            )
            points = points[available]
            visible_indices = visible_indices[available]
            if not len(visible_indices):
                return None
        delta = points - query
        distances_squared = np.einsum("ij,ij->i", delta, delta)
        nearest_index = int(np.argmin(distances_squared))
        if distances_squared[nearest_index] > tolerance**2:
            return None
        return int(visible_indices[nearest_index])

    def set_edit_mode(
        self,
        mode: str | None,
        pan_with_right_button: bool = False,
        legacy_polygon_drawing: bool = False,
    ):
        """Active le dessin ou l'effacement libre du masque."""
        if (
            mode != self._edit_mode
            or legacy_polygon_drawing != self._legacy_polygon_drawing
        ) and self._edit_item is not None:
            self._scene.removeItem(self._edit_item)
            self._edit_item = None
        if (
            mode != self._edit_mode
            or legacy_polygon_drawing != self._legacy_polygon_drawing
        ):
            self._edit_points = []
        self._edit_mode = mode
        self._legacy_polygon_drawing = legacy_polygon_drawing
        enabled = mode is not None
        self._pan_with_right_button = enabled and pan_with_right_button
        self._pan_active = False
        self._pan_waiting_for_release = False
        self._pan_button = None
        self._drawing_active = False
        self._auto_scroll_timer.stop()
        if not enabled:
            tooltip = (
                "Molette : zoom | Glisser un point du contour rouge pour l'ajuster | "
                "Glisser ailleurs : déplacer l'image"
            )
        elif legacy_polygon_drawing:
            tooltip = (
                "Tracer une zone à main levée puis relâcher pour "
                f"{'l’effacer' if mode == 'crop' else 'l’ajouter'} "
                f"(trait de {MASK_ERASER_DIAMETER:g} px) | "
                "Clic droit + glisser : déplacer"
            )
        elif mode == "crop":
            tooltip = (
                "Tracer au clic gauche pour gommer uniquement le contour rouge "
                "(sans modifier le masque ni le volume) | "
                "Clic droit + glisser : déplacer"
            )
        else:
            tooltip = (
                "Relier les extrémités d'un contour effacé pour le restaurer, ou "
                "dessiner une forme fermée en relâchant près du départ."
            )
        self.setToolTip(tooltip)
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

    def _update_edit_preview(self, close_path: bool = False):
        if not self._edit_points:
            return
        if self._edit_item is not None:
            if self._legacy_polygon_drawing:
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                path.closeSubpath()
                pen = QPen(Qt.GlobalColor.yellow, MASK_ERASER_DIAMETER)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                self._edit_item.setPen(pen)
            elif self._edit_mode == "crop":
                path = QPainterPath(self._edit_points[0])
                if len(self._edit_points) == 1:
                    radius = MASK_ERASER_DIAMETER / 2
                    point = self._edit_points[0]
                    path.addEllipse(
                        QRectF(
                            point.x() - radius,
                            point.y() - radius,
                            MASK_ERASER_DIAMETER,
                            MASK_ERASER_DIAMETER,
                        )
                    )
                    pen = QPen(Qt.GlobalColor.yellow, 1)
                    pen.setCosmetic(True)
                    self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                    self._edit_item.setPen(pen)
                else:
                    for point in self._edit_points[1:]:
                        path.lineTo(point)
                    pen = QPen(Qt.GlobalColor.yellow, MASK_ERASER_DIAMETER)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                    self._edit_item.setPen(pen)
            else:
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                if close_path:
                    path.closeSubpath()
            self._edit_item.setPath(path)

    def _auto_scroll_while_drawing(self):
        if (
            not self._drawing_active
            or not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        ):
            self._auto_scroll_timer.stop()
            return

        margin = 20
        speed = 8
        position = self._last_pointer_position
        viewport_rect = self.viewport().rect()
        horizontal_delta = (
            -speed
            if position.x() < margin
            else speed
            if position.x() >= viewport_rect.width() - margin
            else 0
        )
        vertical_delta = (
            -speed
            if position.y() < margin
            else speed
            if position.y() >= viewport_rect.height() - margin
            else 0
        )
        if horizontal_delta:
            scrollbar = self.horizontalScrollBar()
            scrollbar.setValue(scrollbar.value() + horizontal_delta)
        if vertical_delta:
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.value() + vertical_delta)
        if horizontal_delta or vertical_delta:
            current = self.mapToScene(position)
            if self._edit_points[-1] != current:
                self._edit_points.append(current)
                self._update_edit_preview()

    def mousePressEvent(self, event):
        if self._contour_edit_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self._last_auto_added_control_point = None
            position = event.position().toPoint()
            marker_index = self._control_point_index_at(
                position,
                hit_radius=7.0
                if event.button() == Qt.MouseButton.RightButton
                else 10.0,
            )
            if (
                event.button() == Qt.MouseButton.RightButton
                and marker_index >= 0
            ):
                self.contour_control_point_removed.emit(marker_index)
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                scene_position = self.mapToScene(position)
                point_index = self._nearest_active_contour_point(
                    QPointF(*self._control_points[marker_index])
                    if marker_index >= 0
                    else scene_position,
                    tolerance=float("inf") if marker_index >= 0 else None,
                )
                if point_index is not None and self._active_contour is not None:
                    if marker_index < 0:
                        point = self._active_contour[point_index, 0]
                        control_point = (int(point[0]), int(point[1]))
                        was_control_point = (
                            control_point in self._control_points
                        )
                        self.contour_control_point_added.emit(
                            QPointF(float(point[0]), float(point[1]))
                        )
                        if control_point in self._control_points:
                            marker_index = self._control_points.index(
                                control_point
                            )
                            if not was_control_point:
                                self._last_auto_added_control_point = (
                                    control_point
                                )
                    point = self._active_contour[point_index, 0]
                    drag_origin = (
                        self._control_points[marker_index]
                        if marker_index >= 0
                        else (int(point[0]), int(point[1]))
                    )
                    self._contour_drag_origin_scene = QPointF(
                        float(drag_origin[0]),
                        float(drag_origin[1]),
                    )
                    self._contour_drag_origin_global = event.globalPosition()
                    transform = self.transform()
                    self._contour_drag_scale_x = max(
                        abs(transform.m11()),
                        1e-6,
                    )
                    self._contour_drag_scale_y = max(
                        abs(transform.m22()),
                        1e-6,
                    )
                    self._contour_drag_active = True
                    self._drag_marker_index = marker_index
                    self.contour_point_drag_started.emit(
                        self._active_contour.copy(),
                        point_index,
                        marker_index,
                    )
                else:
                    self.contour_region_selection_requested.emit(
                        scene_position
                    )
                event.accept()
                return
        if self._contour_edit_mode or (
            self._edit_mode is not None and self._pan_with_right_button
        ):
            pan_button = Qt.MouseButton.RightButton
        elif self._edit_mode is None:
            pan_button = Qt.MouseButton.LeftButton
        else:
            pan_button = None
        if pan_button is not None and event.button() == pan_button:
            if self._edit_item is not None:
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
            self._edit_points = []
            self._pan_active = True
            self._pan_waiting_for_release = True
            self._pan_button = pan_button
            self._drawing_active = False
            self._auto_scroll_timer.stop()
            self._pan_last_position = event.position().toPoint()
            event.accept()
            return
        if self._pan_waiting_for_release:
            event.accept()
            return
        if self._edit_mode and event.button() == Qt.MouseButton.LeftButton:
            current = self.mapToScene(event.position().toPoint())
            if self._edit_mode == "crop":
                self._edit_points = [current]
            elif not self._edit_points:
                self._edit_points = [current]
                path = QPainterPath(current)
                self._edit_item = QGraphicsPathItem(path)
                if self._legacy_polygon_drawing:
                    pen = QPen(
                        Qt.GlobalColor.yellow,
                        MASK_ERASER_DIAMETER,
                    )
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                else:
                    pen = QPen(Qt.GlobalColor.yellow, 3)
                    pen.setCosmetic(True)
                self._edit_item.setPen(pen)
                self._scene.addItem(self._edit_item)
            elif self._edit_points[-1] != current:
                self._edit_points.append(current)
                self._update_edit_preview()
            self._drawing_active = True
            self._last_pointer_position = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._contour_drag_active:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.contour_point_dragged.emit(
                    self._contour_scene_position(event.globalPosition())
                )
            event.accept()
            return
        if self._pan_active:
            if self._pan_button is not None and event.buttons() & self._pan_button:
                position = event.position().toPoint()
                delta = position - self._pan_last_position
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - delta.x()
                )
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - delta.y()
                )
                self._pan_last_position = position
            event.accept()
            return
        if self._pan_waiting_for_release:
            event.accept()
            return
        if (
            self._edit_mode
            and self._edit_points
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._drawing_active = True
            self._last_pointer_position = event.position().toPoint()
            current = self.mapToScene(self._last_pointer_position)
            if not self._edit_points[-1] == current:
                self._edit_points.append(current)
                if self._edit_mode == "crop" and self._edit_item is None:
                    self._edit_item = QGraphicsPathItem()
                    pen = QPen(
                        Qt.GlobalColor.yellow,
                        MASK_ERASER_DIAMETER,
                    )
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    self._edit_item.setBrush(Qt.BrushStyle.NoBrush)
                    self._edit_item.setPen(pen)
                    self._scene.addItem(self._edit_item)
                self._update_edit_preview()
            margin = 20
            viewport_rect = self.viewport().rect()
            if (
                self._last_pointer_position.x() < margin
                or self._last_pointer_position.x() >= viewport_rect.width() - margin
                or self._last_pointer_position.y() < margin
                or self._last_pointer_position.y() >= viewport_rect.height() - margin
            ):
                if not self._auto_scroll_timer.isActive():
                    self._auto_scroll_timer.start()
            else:
                self._auto_scroll_timer.stop()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            self._contour_drag_active
            and event.button() == Qt.MouseButton.LeftButton
        ):
            if (
                self._contour_drag_origin_global is not None
                and event.globalPosition()
                != self._contour_drag_origin_global
            ):
                self.contour_point_dragged.emit(
                    self._contour_scene_position(event.globalPosition())
                )
            self.contour_point_drag_finished.emit()
            self._contour_drag_active = False
            self._drag_marker_index = -1
            self._contour_drag_origin_scene = None
            self._contour_drag_origin_global = None
            event.accept()
            return
        if self._pan_active:
            if (
                event.button() == self._pan_button
                or event.buttons() == Qt.MouseButton.NoButton
            ):
                self._pan_active = False
                self._pan_waiting_for_release = False
                self._pan_button = None
                self._pan_last_position = QPoint()
            event.accept()
            return
        if self._pan_waiting_for_release:
            if event.buttons() == Qt.MouseButton.NoButton:
                self._pan_waiting_for_release = False
            event.accept()
            return
        if (
            self._edit_mode
            and self._edit_points
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._drawing_active = False
            self._auto_scroll_timer.stop()
            release_position = event.position().toPoint()
            start_position = self.mapFromScene(self._edit_points[0])
            release_point = self.mapToScene(release_position)
            if self._edit_points[-1] != release_point:
                self._edit_points.append(release_point)
            if self._legacy_polygon_drawing:
                self._update_edit_preview(close_path=True)
                path = (
                    self._edit_item.path()
                    if self._edit_item is not None
                    else QPainterPath()
                )
                if self._edit_item is not None:
                    self._scene.removeItem(self._edit_item)
                    self._edit_item = None
                valid_polygon = len(self._edit_points) >= 3
                self._edit_points = []
                if valid_polygon:
                    self.edit_path_finished.emit(path, True)
                event.accept()
                return
            if self._edit_mode == "crop":
                self._update_edit_preview()
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                if self._edit_item is not None:
                    self._scene.removeItem(self._edit_item)
                    self._edit_item = None
                self._edit_points = []
                self.edit_path_finished.emit(path, False)
                event.accept()
                return
            close_delta = release_position - start_position
            closes_path = (
                len(self._edit_points) >= 3
                and close_delta.x() ** 2 + close_delta.y() ** 2 <= 16
            )

            if closes_path:
                self._update_edit_preview(close_path=True)
                if self._edit_item is None:
                    event.accept()
                    return
                path = self._edit_item.path()
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
                self._edit_points = []
                self.edit_path_finished.emit(path, True)
            elif self._edit_mode == "add" and self._edit_item is not None:
                path = QPainterPath(self._edit_points[0])
                for point in self._edit_points[1:]:
                    path.lineTo(point)
                self._scene.removeItem(self._edit_item)
                self._edit_item = None
                self._edit_points = []
                self.edit_path_finished.emit(path, False)
            elif self._edit_item is not None:
                self._update_edit_preview()
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
        if (
            self._contour_edit_mode
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_position = self.mapToScene(event.position().toPoint())
            duplicate_press_point = False
            if (
                self._active_contour is not None
                and self._last_auto_added_control_point is not None
            ):
                point_index = self._nearest_active_contour_point(
                    scene_position,
                    include_erased=True,
                )
                if point_index is not None:
                    point = self._active_contour[point_index, 0]
                    duplicate_press_point = (
                        int(point[0]),
                        int(point[1]),
                    ) == self._last_auto_added_control_point
            if duplicate_press_point:
                self._last_auto_added_control_point = None
            else:
                self.contour_control_point_added.emit(scene_position)
                self._last_auto_added_control_point = None
            event.accept()
            return
        self.fit_image()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Conserve l'image ajustée tant que l'utilisateur ne zoome pas manuellement.
        if self._has_image and not self._user_zoomed:
            self.fit_image()
