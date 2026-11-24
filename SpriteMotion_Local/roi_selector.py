"""Visual selector with zoom, scrollbars and image-coordinate ROI for SpriteMotion.

The subject box is used as a tracking hint; its coordinates stay in original
frame space regardless of how the selection image is displayed.
"""
from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QScrollBar, QWidget, QDialogButtonBox,
)


def clamp_scene_point(pt: QPointF, width: int, height: int) -> QPointF:
    return QPointF(max(0.0, min(float(width), pt.x())),
                   max(0.0, min(float(height), pt.y())))


def roi_from_points(start: QPointF, end: QPointF, width: int, height: int):
    """Convert a dragged scene-space box into a bounded integer pixel ROI."""
    a = clamp_scene_point(start, width, height)
    b = clamp_scene_point(end, width, height)
    x0 = max(0, min(width, int(min(a.x(), b.x()))))
    y0 = max(0, min(height, int(min(a.y(), b.y()))))
    x1 = max(0, min(width, int(max(a.x(), b.x()) + 0.9999)))
    y1 = max(0, min(height, int(max(a.y(), b.y()) + 0.9999)))
    return x0, y0, max(0, x1-x0), max(0, y1-y0)


class SelectableImageView(QGraphicsView):
    selection_changed = Signal(object)
    zoom_changed = Signal(int)

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.picture = self.scene_obj.addPixmap(pixmap)
        self.w, self.h = pixmap.width(), pixmap.height()
        self.scene_obj.setSceneRect(QRectF(0, 0, self.w, self.h))
        self.selection_rect: QGraphicsRectItem = self.scene_obj.addRect(QRectF())
        self.selection_rect.setPen(QPen(QColor('#ffce4d'), 2, Qt.PenStyle.DashLine))
        self.selection_rect.setBrush(QBrush(QColor(255, 205, 65, 32)))
        self.selection_rect.setZValue(10)
        self.selection_rect.hide()
        self.roi = None
        self.drag_start = None
        self.fit_mode = True
        self.pan_mode = False
        self._zoom_factor = 1.0
        self.setBackgroundBrush(QBrush(QColor('#171c29')))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(260, 230)
        self.setToolTip('Arraste com o botão esquerdo para marcar o personagem. Ctrl + roda do mouse para zoom; roda normal para rolar; Espaço + arrastar para mover.')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode and self.w and self.h:
            self.fit_image()

    def fit_image(self):
        if not self.w or not self.h:
            return
        self.fit_mode = True
        self.resetTransform()
        # Fit entire source image instead of showing cropped original pixels.
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()
        self.zoom_changed.emit(round(100*self._zoom_factor))

    def actual_size(self):
        self.fit_mode = False
        self.resetTransform()
        self._zoom_factor = 1.0
        self.zoom_changed.emit(100)

    def change_zoom(self, factor: float):
        new_zoom = self.transform().m11()*factor
        if new_zoom < 0.08 or new_zoom > 16.0:
            return
        self.fit_mode = False
        self.scale(factor, factor)
        self._zoom_factor = self.transform().m11()
        self.zoom_changed.emit(round(100*self._zoom_factor))

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            steps = event.angleDelta().y()/120.0
            if steps:
                self.change_zoom(1.18 ** steps)
            event.accept()
            return
        # The default view scrolls vertically, horizontally with Shift.
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.pan_mode = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.pan_mode = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.pan_mode:
            self.drag_start = clamp_scene_point(self.mapToScene(event.position().toPoint()), self.w, self.h)
            self.selection_rect.show()
            self.selection_rect.setRect(QRectF(self.drag_start, self.drag_start))
            self.roi = None
            self.selection_changed.emit(None)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            dest = clamp_scene_point(self.mapToScene(event.position().toPoint()), self.w, self.h)
            self.selection_rect.setRect(QRectF(self.drag_start, dest).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drag_start is not None:
            dest = clamp_scene_point(self.mapToScene(event.position().toPoint()), self.w, self.h)
            roi = roi_from_points(self.drag_start, dest, self.w, self.h)
            self.drag_start = None
            if roi[2] < 4 or roi[3] < 4:
                self.selection_rect.hide()
                self.roi = None
            else:
                self.roi = roi
                x, y, w, h = roi
                self.selection_rect.setRect(QRectF(x, y, w, h))
            self.selection_changed.emit(self.roi)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def clear_selection(self):
        self.roi = None
        self.drag_start = None
        self.selection_rect.hide()
        self.selection_changed.emit(None)


class SubjectSelectionDialog(QDialog):
    """Modal source-resolution ROI selector. Cancel returns None."""
    def __init__(self, frame_bgr, filename='', parent=None):
        super().__init__(parent)
        self.setWindowTitle('SpriteMotion | Marque o personagem')
        self.setWindowFlags(Qt.WindowType.Window |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.WindowCloseButtonHint)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self.view = SelectableImageView(QPixmap.fromImage(img), self)
        self.setMinimumSize(500, 390)
        self.resize(960, 680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        lead = QLabel(f'Vídeo: {filename}\nArraste um retângulo ao redor do personagem inteiro, incluindo cabeça e pés.')
        lead.setWordWrap(True)
        layout.addWidget(lead)
        layout.addWidget(self.view, 1)
        controls = QHBoxLayout()
        zoomout = QPushButton('− Afastar')
        zoomout.clicked.connect(lambda: self.view.change_zoom(1/1.25))
        controls.addWidget(zoomout)
        zoomin = QPushButton('＋ Aproximar')
        zoomin.clicked.connect(lambda: self.view.change_zoom(1.25))
        controls.addWidget(zoomin)
        fitbtn = QPushButton('Ajustar à janela')
        fitbtn.clicked.connect(self.view.fit_image)
        controls.addWidget(fitbtn)
        actualbtn = QPushButton('100%')
        actualbtn.clicked.connect(self.view.actual_size)
        controls.addWidget(actualbtn)
        self.percent = QLabel('Zoom: automático')
        self.view.zoom_changed.connect(lambda value: self.percent.setText(f'Zoom: {value}%'))
        controls.addWidget(self.percent)
        controls.addStretch()
        resetbtn = QPushButton('Limpar seleção')
        resetbtn.clicked.connect(self.view.clear_selection)
        controls.addWidget(resetbtn)
        layout.addLayout(controls)
        row = QHBoxLayout()
        self.status = QLabel('Role para navegar. Ctrl + roda dá zoom. Espaço + arrastar move a imagem.')
        self.status.setWordWrap(True)
        row.addWidget(self.status, 1)
        self.cancel_btn = QPushButton('Ignorar marcação')
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.cancel_btn)
        self.confirm_btn = QPushButton('Confirmar personagem')
        self.confirm_btn.setObjectName('primary')
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.accept)
        row.addWidget(self.confirm_btn)
        layout.addLayout(row)
        self.view.selection_changed.connect(self.on_selection_changed)
        QTimer.singleShot(0, self.view.fit_image)

    def on_selection_changed(self, roi):
        self.confirm_btn.setEnabled(roi is not None)
        if roi is not None:
            x, y, w, h = roi
            self.status.setText(f'Área selecionada: x={x}, y={y}, largura={w}, altura={h} px')
        else:
            self.status.setText('Arraste na imagem para selecionar. Use os controles de zoom ou a rolagem.')

    def selected_roi(self):
        return self.view.roi

    def accept(self):
        if self.view.roi is None:
            return
        super().accept()
