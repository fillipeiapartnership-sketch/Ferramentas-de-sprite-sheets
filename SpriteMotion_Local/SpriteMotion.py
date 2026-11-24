"""SpriteMotion Local: video > unique transparent poses > grid > animation preview."""
from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot, QTimer, QSize, QRectF, QPointF, QUrl, QEvent
from PySide6.QtGui import QColor, QPainter, QPixmap, QImage, QIcon, QLinearGradient, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QSpinBox, QComboBox, QCheckBox, QSlider, QMessageBox, QProgressBar, QFrame,
    QDialog, QRadioButton, QAbstractItemView, QLineEdit, QInputDialog, QScrollArea)

from sprite_engine import (Frame, Group, process_video, process_additional,
    reapply_cleanup, reprocess_stable_existing, load_project, export_group,
    geometry, render_cell, suggest_cycle_for_group, suggest_cycle_window_for_group)
from smart_retouch import select_retouch, apply_retouch_selection
from gif_export import export_animation_gif
from roi_selector import SubjectSelectionDialog


STYLE = """
QWidget { background:#101523; color:#e8edf7; font-family:'Segoe UI',Arial; font-size:12px; }
QMainWindow { background:#0d1420; }
QFrame#panel { background:#171f30; border:1px solid #29364b; border-radius:12px; }
QLabel#heading { font-size:17px; font-weight:700; color:#f1f5ff; }
QLabel#sub { color:#95a8c2; }
QPushButton { background:#2b374c; color:#e9eefb; border:1px solid #46536c; border-radius:7px;
              padding:8px 12px; font-weight:600; }
QPushButton:hover { background:#384a63; } QPushButton:disabled{color:#64758e; background:#1d2637;}
QPushButton#primary { background:#5568ef; border:1px solid #7181ff; color:white; }
QPushButton#primary:hover { background:#687aff; }
QListWidget { background:#0e1625; border:1px solid #34415b; border-radius:8px; padding:5px; }
QListWidget::item { color:#dfecfd; padding:6px; border-radius:5px; }
QListWidget::item:selected { background:#314b82; }
QComboBox,QSpinBox,QLineEdit { background:#101b2c; border:1px solid #45546e;
    border-radius:6px; padding:6px; color:#fff; }
QComboBox QAbstractItemView { background:#202a3d; color:#fff; selection-background-color:#344b84; }
QProgressBar{background:#101a2a;border:1px solid #344257;border-radius:5px;min-height:12px;text-align:center;}
QProgressBar::chunk{background:#7581fc;border-radius:4px;}
QSlider::groove:horizontal{background:#344459;height:5px;border-radius:3px;}
QSlider::handle:horizontal{background:#8497ff;width:13px;margin:-5px 0;border-radius:6px;}
QCheckBox { spacing:7px; }
QScrollArea { background:#171f30; border:0; }
"""


def qpix(img: Image.Image) -> QPixmap:
    rgba = img.convert('RGBA')
    data = rgba.tobytes('raw','RGBA')
    q = QImage(data, rgba.width, rgba.height, 4*rgba.width, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(q.copy())


def preview_pixmap(pm: QPixmap, size: QSize, checker=False):
    w, h = max(1, size.width()), max(1, size.height())
    out = QPixmap(w, h)
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    gradient = QLinearGradient(0,0,w,h)
    gradient.setColorAt(0,QColor('#25344c'))
    gradient.setColorAt(1,QColor('#34435c'))
    painter.fillRect(0,0,w,h,gradient)
    if checker:
        # Diagnostic transparency background ONLY, never baked into exported PNGs.
        step=18
        for iy in range(0,h,step):
            for ix in range(0,w,step):
                painter.fillRect(ix,iy,min(step,w-ix),min(step,h-iy),
                  QColor('#d5d9e0' if ((ix//step+iy//step)%2) else '#798295'))
    scaled = pm.scaled(max(1,w-24),max(1,h-24), Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
    painter.drawPixmap((w-scaled.width())//2,(h-scaled.height())//2,scaled)
    painter.end()
    return out


class SpriteHoverPreview(QFrame):
    """Lupa temporária, não interativa, ao passar o mouse no thumbnail."""
    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setObjectName('hover_popup')
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setStyleSheet('QFrame#hover_popup {background:#172137; border:2px solid #899df1; border-radius:9px;}')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        self.title = QLabel('')
        self.title.setStyleSheet('font-weight:700; color:#f2f6ff;')
        layout.addWidget(self.title)
        self.picture = QLabel()
        self.picture.setFixedSize(340, 370)
        layout.addWidget(self.picture)
        self.footer = QLabel('Prévia ampliada · dois cliques para corrigir')
        self.footer.setStyleSheet('color:#adc5e0; font-size:11px;')
        layout.addWidget(self.footer)
        self.adjustSize()

    def show_frame(self, frame, path, position):
        try:
            with Image.open(path) as source:
                rgba = source.convert('RGBA')
                if path == frame.sprite:
                    alpha = rgba.getchannel('A')
                    bbox = alpha.getbbox()
                    if bbox:
                        margin = 8
                        rgba = rgba.crop((max(0, bbox[0]-margin), max(0, bbox[1]-margin),
                                          min(rgba.width, bbox[2]+margin), min(rgba.height, bbox[3]+margin)))
                pm = qpix(rgba)
            self.picture.setPixmap(preview_pixmap(pm, self.picture.size(), checker=True))
            self.title.setText(f'QUADRO {frame.index+1:04d}  |  LUPA TEMPORÁRIA')
            screen = QGuiApplication.screenAt(position) or QGuiApplication.primaryScreen()
            area = screen.availableGeometry() if screen else None
            self.adjustSize()
            px = position.x() + 25
            py = position.y() + 20
            if area is not None:
                if px + self.width() > area.right():
                    px = position.x() - self.width() - 25
                if py + self.height() > area.bottom():
                    py = area.bottom() - self.height()
                px = max(area.left(), px)
                py = max(area.top(), py)
            self.move(px, py)
            self.show()
        except (OSError, ValueError):
            self.hide()


class Job(QObject):
    progress = Signal(int,str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, func):
        super().__init__()
        self.func=func
        self.aborted=False

    @Slot()
    def run(self):
        try:
            result=self.func(self.progress.emit, lambda: self.aborted)
            self.succeeded.emit(result)
        except Exception as e:
            self.failed.emit(f'{type(e).__name__}: {e}\n\n{traceback.format_exc()[-4000:]}')


class BrushCanvas(QWidget):
    edited = Signal(str)
    selection_changed = Signal(bool, str)
    def __init__(self, original_path, sprite_path):
        super().__init__()
        self.original=np.array(Image.open(original_path).convert('RGBA'))
        self.cut=np.array(Image.open(sprite_path).convert('RGBA'))
        self.brush=12
        self.erase=False
        self.mode='auto'
        self.tolerance=19
        self.smart_radius=28
        self.pressed=False
        self.history=[]
        self.last=None
        self.canvas_rect=QRectF()
        self._pixmap_cache=None
        self.pending_selection=None
        self._selection_pixmap=None
        self.zoom=6
        self.min_zoom=1
        self.max_zoom=20
        self.setMouseTracking(True)
        self.update_zoom()

    def update_zoom(self):
        h, w = self.cut.shape[:2]
        zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))
        self.zoom = zoom
        self.setMinimumSize(w*zoom, h*zoom)
        self.resize(w*zoom, h*zoom)
        self.update()

    def set_zoom(self, zoom):
        self.zoom = int(zoom)
        self.update_zoom()

    def zoom_in(self):
        self.set_zoom(min(self.max_zoom, self.zoom + 1))

    def zoom_out(self):
        self.set_zoom(max(self.min_zoom, self.zoom - 1))

    def set_brush(self, r): self.brush=r
    def set_erase(self, erase):
        self.set_mode('brush_erase' if erase else 'brush_restore')

    def cancel_selection(self, message='Seleção descartada. Nenhum pixel foi alterado.'):
        self.pending_selection=None
        self._selection_pixmap=None
        self.selection_changed.emit(False, message)
        self.update()

    def set_mode(self, mode):
        if self.mode != str(mode):
            self.cancel_selection('Ferramenta alterada; seleção pendente descartada.')
        self.mode = str(mode)

    def set_tolerance(self, value):
        if self.tolerance != int(value):
            self.cancel_selection('Tolerância alterada. Clique novamente para selecionar.')
        self.tolerance = int(value)

    def set_smart_radius(self, value):
        if self.smart_radius != int(value):
            self.cancel_selection('Raio alterado. Clique novamente para selecionar.')
        self.smart_radius = int(value)

    def confirm_selection(self):
        if self.pending_selection is None:
            self.edited.emit('Selecione primeiro a mancha com um clique.')
            return
        try:
            updated, message, changed = apply_retouch_selection(self.cut, self.pending_selection)
        except Exception as error:
            self.edited.emit(f'Não foi possível aplicar a seleção: {error}')
            return
        if changed:
            self.history.append(self.cut.copy())
            if len(self.history)>20: self.history.pop(0)
            self.cut=updated
            self._pixmap_cache=None
        self.cancel_selection('Seleção confirmada.')
        self.edited.emit(message)
    def _pixel_at(self, pos):
        rect=self.canvas_rect
        if not rect.contains(pos) or rect.width() <= 0 or rect.height() <= 0:
            return None
        x=int((pos.x()-rect.x()) * self.cut.shape[1] / rect.width())
        y=int((pos.y()-rect.y()) * self.cut.shape[0] / rect.height())
        return min(max(x, 0), self.cut.shape[1]-1), min(max(y, 0), self.cut.shape[0]-1)

    def undo(self):
        if self.pending_selection is not None:
            self.cancel_selection()
            return
        if self.history:
            self.cut=self.history.pop()
            self._pixmap_cache=None
            self.update()

    def mousePressEvent(self,e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self.mode in ('auto', 'erase', 'color'):
            point = self._pixel_at(e.position())
            if point is None:
                self.edited.emit('Clique dentro do sprite para selecionar a mancha.')
                return
            try:
                selected, message = select_retouch(
                    self.cut, point[0], point[1], self.tolerance,
                    self.smart_radius, self.mode)
            except Exception as error:
                selected, message = None, 'Falha ao selecionar mancha: ' + str(error)
            self.pending_selection=selected
            self._selection_pixmap=None
            self.selection_changed.emit(selected is not None, message)
            self.edited.emit(message)
            self.update()
            return
        if self.pending_selection is not None:
            self.cancel_selection()
        self.history.append(self.cut.copy())
        if len(self.history)>20: self.history.pop(0)
        self.pressed=True
        self.last=None
        self.draw_at(e.position())

    def mouseMoveEvent(self,e):
        if self.pressed and self.mode in ('brush_restore', 'brush_erase'):
            self.draw_at(e.position())

    def mouseReleaseEvent(self,e):
        if e.button()==Qt.MouseButton.LeftButton:
            self.pressed=False;self.last=None

    def draw_at(self,point):
        position=self._pixel_at(point)
        if position is None:
            self.last=None;return
        x,y=position
        mark=np.zeros(self.cut.shape[:2],dtype=np.uint8)
        r=max(1,self.brush)
        if self.last is None:
            cv2.circle(mark,(x,y),r,255,-1,cv2.LINE_AA)
        else:
            cv2.line(mark,self.last,(x,y),255,2*r,cv2.LINE_AA)
        touched=mark>0
        current=self.cut[:,:,3]
        if self.mode == 'brush_erase':
            current[touched]=np.minimum(current[touched], 255-mark[touched])
        else:
            current[touched]=np.maximum(current[touched], mark[touched])
            self.cut[:,:,:3][touched]=self.original[:,:,:3][touched]
        self.last=(x,y)
        self._pixmap_cache=None
        self.update()

    def paintEvent(self,e):
        p=QPainter(self)
        p.setClipRect(e.rect())
        # Paint only the visible part when a high zoom makes the canvas huge.
        step=max(12, 8*self.zoom)
        visible=e.rect()
        xbegin=max(0,(visible.left()//step)*step)
        ybegin=max(0,(visible.top()//step)*step)
        xend=min(self.width(),visible.right()+step+1)
        yend=min(self.height(),visible.bottom()+step+1)
        for iy in range(ybegin,yend,step):
            for ix in range(xbegin,xend,step):
                p.fillRect(ix,iy,step,step,QColor('#d4d8df' if (ix//step+iy//step)%2 else '#78849b'))
        h,w=self.cut.shape[:2]
        dw,dh=w*self.zoom,h*self.zoom
        self.canvas_rect=QRectF(0,0,dw,dh)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if self._pixmap_cache is None:
            self._pixmap_cache=qpix(Image.fromarray(self.cut,'RGBA'))
        p.drawPixmap(self.canvas_rect,self._pixmap_cache,QRectF(0,0,w,h))
        if self.pending_selection is not None:
            if self._selection_pixmap is None:
                selection=self.pending_selection
                x0,y0,x1,y1=selection.bounds
                overlay=np.zeros_like(self.cut)
                patch=overlay[y0:y1,x0:x1]
                if selection.action == 'erase':
                    patch[:,:,:3]=[255,90,55]
                else:
                    patch[:,:,:3]=[32,230,244]
                patch[:,:,3][selection.mask]=185
                self._selection_pixmap=qpix(Image.fromarray(overlay,'RGBA'))
            p.drawPixmap(self.canvas_rect,self._selection_pixmap,QRectF(0,0,w,h))
        p.end()


class MaskDialog(QDialog):
    def __init__(self, frame:Frame, parent=None):
        super().__init__(parent)
        # Ordinary titlebar controls, including maximize, restore and minimize.
        # A modal QDialog is still used, so the user must save or cancel edits.
        self.setWindowFlags(Qt.WindowType.Window |
                            Qt.WindowType.WindowMinMaxButtonsHint |
                            Qt.WindowType.WindowCloseButtonHint)
        self.setWindowTitle(f'Retoque do sprite | quadro {frame.index+1}')
        self.resize(1050,800)
        self.frame=frame
        lay=QVBoxLayout(self)
        info=QLabel('Clique na mancha para selecioná-la. Confira o destaque e clique em Confirmar remoção / correção.\n'
                    'Use maximizar/minimizar na barra de título. A rolagem horizontal e vertical aparece quando o zoom ultrapassa a janela.')
        info.setObjectName('sub');info.setWordWrap(True);lay.addWidget(info)
        self.canvas=BrushCanvas(frame.original,frame.sprite)
        self.scroll=QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setWidget(self.canvas)
        lay.addWidget(self.scroll,1)
        tools=QHBoxLayout()
        tools.addWidget(QLabel('Ferramenta'))
        self.tool=QComboBox()
        self.tool.addItem('Clique inteligente (recomendado)', 'auto')
        self.tool.addItem('Clique: apagar para transparência', 'erase')
        self.tool.addItem('Clique: corrigir cor sem cortar', 'color')
        self.tool.addItem('Pincel: apagar manualmente', 'brush_erase')
        self.tool.addItem('Pincel: recuperar personagem', 'brush_restore')
        self.tool.currentIndexChanged.connect(lambda:self.canvas.set_mode(self.tool.currentData()))
        tools.addWidget(self.tool,2)
        tools.addWidget(QLabel('Tolerância'))
        self.tolerance=QSpinBox();self.tolerance.setRange(3,80);self.tolerance.setValue(19)
        self.tolerance.valueChanged.connect(self.canvas.set_tolerance)
        self.tolerance.setToolTip('Menor = seleciona pixels de cor mais semelhante. Comece em 19.')
        tools.addWidget(self.tolerance)
        tools.addWidget(QLabel('Raio'))
        self.radius=QSpinBox();self.radius.setRange(3,160);self.radius.setValue(28)
        self.radius.valueChanged.connect(self.canvas.set_smart_radius)
        self.radius.setToolTip('Limite em pixels em torno do ponto clicado; impede remoção de regiões distantes.')
        tools.addWidget(self.radius)
        tools.addWidget(QLabel('Pincel'))
        brush=QSpinBox();brush.setRange(1,120);brush.setValue(12)
        brush.valueChanged.connect(self.canvas.set_brush)
        tools.addWidget(brush)
        lay.addLayout(tools)
        zoom_tools=QHBoxLayout()
        zoom_tools.addWidget(QLabel('Zoom'))
        zoom=QSpinBox();zoom.setRange(self.canvas.min_zoom,self.canvas.max_zoom);zoom.setValue(self.canvas.zoom)
        zoom.valueChanged.connect(self.apply_zoom); self.zoom_box = zoom
        zoom_tools.addWidget(zoom)
        zoom_out=QPushButton('－')
        zoom_out.clicked.connect(lambda:self.apply_zoom(max(self.canvas.min_zoom,self.canvas.zoom-1)))
        zoom_tools.addWidget(zoom_out)
        zoom_in=QPushButton('＋')
        zoom_in.clicked.connect(lambda:self.apply_zoom(min(self.canvas.max_zoom,self.canvas.zoom+1)))
        zoom_tools.addWidget(zoom_in)
        fit=QPushButton('Ajustar à janela');fit.clicked.connect(self.fit_to_window)
        zoom_tools.addWidget(fit)
        undo=QPushButton('Desfazer');undo.clicked.connect(self.undo)
        zoom_tools.addWidget(undo)
        zoom_tools.addStretch()
        lay.addLayout(zoom_tools)
        selection_row=QHBoxLayout()
        self.selection_state=QLabel('Nenhuma mancha selecionada. O clique apenas destaca os pixels.')
        self.selection_state.setObjectName('sub')
        self.selection_state.setWordWrap(True)
        selection_row.addWidget(self.selection_state,1)
        self.confirm_button=QPushButton('Confirmar remoção / correção')
        self.confirm_button.setObjectName('primary')
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.canvas.confirm_selection)
        selection_row.addWidget(self.confirm_button)
        self.cancel_selection_button=QPushButton('Descartar seleção')
        self.cancel_selection_button.setEnabled(False)
        self.cancel_selection_button.clicked.connect(lambda:self.canvas.cancel_selection())
        selection_row.addWidget(self.cancel_selection_button)
        lay.addLayout(selection_row)
        self.canvas.selection_changed.connect(self.on_selection_changed)
        self.result_label=QLabel('O clique identifica uma área, mas não apaga nada até sua confirmação manual.')
        self.result_label.setObjectName('sub');self.result_label.setWordWrap(True)
        self.canvas.edited.connect(self.result_label.setText)
        lay.addWidget(self.result_label)
        bottom=QHBoxLayout();bottom.addStretch()
        cancel=QPushButton('Cancelar');cancel.clicked.connect(self.reject)
        save=QPushButton('Salvar correção');save.setObjectName('primary');save.clicked.connect(self.accept)
        bottom.addWidget(cancel);bottom.addWidget(save);lay.addLayout(bottom)
        self.fit_to_window()

    def on_selection_changed(self, exists, message):
        self.confirm_button.setEnabled(bool(exists))
        self.cancel_selection_button.setEnabled(bool(exists))
        self.selection_state.setText(message)
        if exists and self.canvas.pending_selection is not None:
            selected=self.canvas.pending_selection
            action='APAGAR' if selected.action == 'erase' else 'CORRIGIR COR'
            self.confirm_button.setText(f'Confirmar {action.lower()} ({selected.count} px)')
        else:
            self.confirm_button.setText('Confirmar remoção / correção')

    def undo(self):
        pending=self.canvas.pending_selection is not None
        self.canvas.undo()
        self.result_label.setText('Seleção descartada.' if pending else 'Última correção desfeita.')

    def apply_zoom(self, value):
        if hasattr(self, 'zoom_box') and self.zoom_box.value() != int(value):
            self.zoom_box.blockSignals(True)
            self.zoom_box.setValue(int(value))
            self.zoom_box.blockSignals(False)
        old_width = max(1,self.canvas.width())
        old_height = max(1,self.canvas.height())
        cx = (self.scroll.horizontalScrollBar().value() + self.scroll.viewport().width()/2)/old_width
        cy = (self.scroll.verticalScrollBar().value() + self.scroll.viewport().height()/2)/old_height
        self.canvas.set_zoom(int(value))
        self.scroll.horizontalScrollBar().setValue(round(cx*self.canvas.width()-self.scroll.viewport().width()/2))
        self.scroll.verticalScrollBar().setValue(round(cy*self.canvas.height()-self.scroll.viewport().height()/2))

    def fit_to_window(self):
        area = self.scroll.viewport().size()
        h, w = self.canvas.cut.shape[:2]
        if w <= 0 or h <= 0:
            return
        zoom = max(1, min(self.canvas.max_zoom, min(max(1, area.width()//w), max(1, area.height()//h))))
        self.apply_zoom(zoom)

    def accept(self):
        if self.canvas.pending_selection is not None:
            choice=QMessageBox.question(self, 'Seleção pendente',
                'Existe uma seleção destacada que ainda não foi confirmada.\n'
                'Descartar somente essa seleção e salvar as outras correções?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if choice != QMessageBox.StandardButton.Yes:
                return
            self.canvas.cancel_selection()
        Image.fromarray(self.canvas.cut,'RGBA').save(self.frame.sprite)
        super().accept()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SpriteMotion Local v15 | Vídeo para Sprite Sheet')
        self.resize(1510,890)
        self.groups=[]
        self.current=None
        self.thread=None
        self.job=None
        # Worker results are only delivered AFTER QThread.finished, so that no
        # modal dialog or extra user action can trap a live worker thread.
        self._job_callback=None
        self._pending_result=None
        self._pending_error=None
        self.last_export_folder=None
        self.anim_index=0
        self.anim_cache={}
        self.preview_geom=None
        self.preview_size=QSize(420,450)
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.output_root = self.load_storage_folder()
        self.build()

    def hide_hover(self):
        if hasattr(self, 'hover_popup'):
            self.hover_popup.hide()
        self.hover_key=None

    def eventFilter(self, obj, event):
        if hasattr(self, 'frame_list') and obj is self.frame_list.viewport():
            typ=event.type()
            if typ == QEvent.Type.MouseMove:
                item=self.frame_list.itemAt(event.position().toPoint())
                if item is None or self.current is None:
                    self.hide_hover()
                else:
                    idx=item.data(Qt.ItemDataRole.UserRole)
                    if idx is not None and 0 <= idx < len(self.current.frames):
                        fr=self.current.frames[idx]
                        src=fr.sprite if fr.sprite and Path(fr.sprite).is_file() else fr.original
                        key=(id(self.current),idx,src)
                        if key != self.hover_key:
                            self.hover_key=key
                            self.hover_popup.show_frame(fr,src,event.globalPosition().toPoint())
                    else:
                        self.hide_hover()
            elif typ in (QEvent.Type.Leave, QEvent.Type.Wheel,
                         QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
                self.hide_hover()
        return super().eventFilter(obj,event)

    def panel(self,layout=None):
        box=QFrame();box.setObjectName('panel')
        if layout is not None: box.setLayout(layout)
        return box

    def build(self):
        root=QWidget();self.setCentralWidget(root)
        all_layout=QVBoxLayout(root)
        all_layout.setContentsMargins(14,12,14,12)
        all_layout.setSpacing(9)
        header=QHBoxLayout()
        title=QLabel('SpriteMotion  /  LOCAL v15');title.setObjectName('heading')
        header.addWidget(title)
        info=QLabel('Vídeos → rastreamento opcional do personagem → transparência → sprites únicos → animação')
        info.setObjectName('sub');header.addWidget(info)
        header.addStretch()
        self.import_button=QPushButton('＋ Importar vídeos');self.import_button.setObjectName('primary')
        self.import_button.clicked.connect(self.import_videos)
        header.addWidget(self.import_button)
        self.open_project_button=QPushButton('Abrir projeto anterior')
        self.open_project_button.clicked.connect(self.open_project)
        header.addWidget(self.open_project_button)
        all_layout.addLayout(header)
        # User-controlled storage, default next to the extracted program. No Documents output.
        storage_row=QHBoxLayout()
        storage_row.addWidget(QLabel('Pasta dos projetos'))
        self.storage_input=QLineEdit(str(self.output_root))
        self.storage_input.setReadOnly(True)
        self.storage_input.setToolTip('Frames originais, recortes e projetos serão salvos nesta pasta. Não altera projetos já criados.')
        storage_row.addWidget(self.storage_input,1)
        self.storage_button=QPushButton('Escolher pasta...')
        self.storage_button.clicked.connect(self.choose_storage_folder)
        storage_row.addWidget(self.storage_button)
        all_layout.addLayout(storage_row)
        self.split=QSplitter(Qt.Orientation.Horizontal)
        all_layout.addWidget(self.split,1)

        # Group list / per-video settings
        left_lay=QVBoxLayout();left_lay.setContentsMargins(13,13,13,13)
        left_lay.addWidget(self.heading('GRUPOS DE MOVIMENTOS'))
        self.groups_list=QListWidget();self.groups_list.currentRowChanged.connect(self.change_group)
        self.groups_list.itemChanged.connect(self.rename_group)
        self.groups_list.setMinimumHeight(115)
        left_lay.addWidget(self.groups_list,1)
        split_button=QPushButton('Duplicar seleção em novo grupo')
        split_button.clicked.connect(self.clone_group)
        left_lay.addWidget(split_button)
        remove_group=QPushButton('Tirar grupo da lista')
        remove_group.clicked.connect(self.remove_group)
        left_lay.addWidget(remove_group)
        left_lay.addWidget(self.heading('CONFIGURAÇÕES DE ANÁLISE'))
        left_lay.addWidget(QLabel('Modelo de remoção de fundo'))
        self.model=QComboBox()
        self.model.addItem('Anime / personagens 2D', 'isnet-anime')
        self.model.addItem('Geral / 3D / personagens', 'isnet-general-use')
        self.model.addItem('Geral detalhado (mais pesado)', 'birefnet-general-lite')
        left_lay.addWidget(self.model)
        self.guide_box=QCheckBox('Ajudar com retângulo no personagem')
        self.guide_box.setChecked(True)
        self.guide_box.setToolTip('O retângulo identifica o personagem, sem recortar a imagem em movimento. A escala e as coordenadas dos sprites não dependem dele.')
        left_lay.addWidget(self.guide_box)
        left_lay.addWidget(self.sub('Quando ativado, a ferramenta abre o primeiro frame para você desenhar um retângulo no personagem.'))
        left_lay.addWidget(QLabel('Remover quadros repetidos'))
        self.sensitivity=QComboBox()
        self.sensitivity.addItem('0 · Só quase idênticos',0)
        self.sensitivity.addItem('1 · Muito conservador',1)
        self.sensitivity.addItem('2 · Conservador (recomendado)',2)
        self.sensitivity.addItem('3 · Moderado',3)
        self.sensitivity.addItem('4 · Agressivo',4)
        self.sensitivity.setCurrentIndex(2)
        left_lay.addWidget(self.sensitivity)
        self.cycle_button=QPushButton('Identificar ciclo contínuo sem tremor')
        self.cycle_button.setToolTip('Evita saltos causados pela mistura de poses de várias repetições. Reusa quadros já extraídos.')
        self.cycle_button.clicked.connect(self.reselect_cycle)
        left_lay.addWidget(self.cycle_button)
        left_lay.addWidget(QLabel('Ajuste da borda (em pixels)'))
        self.edge=QSpinBox();self.edge.setRange(-4,4);self.edge.setValue(0)
        self.edge.setToolTip('0 preserva a máscara. Negativo contrai, positivo expande.')
        left_lay.addWidget(self.edge)
        left_lay.addWidget(self.heading('LIMPEZA DO FUNDO VERDE'))
        left_lay.addWidget(QLabel('Detecção do cenário verde'))
        self.cleanup_mode=QComboBox()
        self.cleanup_mode.addItem('Automático (recomendado)', 'auto')
        self.cleanup_mode.addItem('Forçar limpeza verde', 'green')
        self.cleanup_mode.addItem('Fundo preto uniforme / escuro', 'dark')
        self.cleanup_mode.addItem('Desligado', 'off')
        left_lay.addWidget(self.cleanup_mode)
        left_lay.addWidget(QLabel('Força da remoção dos halos (%)'))
        self.cleanup_strength=QSpinBox()
        self.cleanup_strength.setRange(0,100);self.cleanup_strength.setValue(85)
        self.cleanup_strength.setToolTip('Comece em 85. Se um acessório verde for afetado, reduza para 50–65.')
        left_lay.addWidget(self.cleanup_strength)
        left_lay.addWidget(QLabel('Faixa de limpeza na borda (pixels)'))
        self.cleanup_radius=QSpinBox()
        self.cleanup_radius.setRange(1,32);self.cleanup_radius.setValue(10)
        self.cleanup_radius.setToolTip('Normalmente 8–12 px; evite valores altos se a roupa for verde.')
        left_lay.addWidget(self.cleanup_radius)
        self.reclean_button=QPushButton('♻ Refazer limpeza dos sprites prontos')
        self.reclean_button.setToolTip('Usa máscaras já calculadas. Não roda a IA novamente.')
        self.reclean_button.clicked.connect(self.reclean_current)
        left_lay.addWidget(self.reclean_button)
        left_lay.addWidget(self.heading('CORREÇÃO DE TREMOR (V6–V9)'))
        self.stable_reprocess_button=QPushButton('Reprocessar máscaras sem recorte móvel')
        self.stable_reprocess_button.setToolTip('Recria a transparência com o frame inteiro, sem mover o recorte conforme o retângulo. Faz backup antes de substituir PNGs já corrigidos.')
        self.stable_reprocess_button.clicked.connect(self.reprocess_stable_current)
        left_lay.addWidget(self.stable_reprocess_button)
        left_lay.addWidget(self.sub('Para projetos antigos: selecione os sprites e use este botão. Os PNGs antigos ficam guardados em backup no projeto.'))
        left_lay.addWidget(self.sub('Modo automático só atua quando encontra cenário verde nas bordas. Para recortes antigos, abra o projeto anterior.'))
        left_lay.addWidget(self.sub('O vídeo é processado localmente. O primeiro uso de cada modelo requer baixar seus pesos.'))
        left_panel=self.panel(left_lay)
        left_panel.setMinimumWidth(238)
        scroll=QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(left_panel)
        self.split.addWidget(scroll)

        # frame browser
        mid_lay=QVBoxLayout();mid_lay.setContentsMargins(13,13,13,13)
        mid_lay.addWidget(self.heading('FRAMES E SPRITES PRINCIPAIS'))
        filter_row=QHBoxLayout()
        self.all_frames=QCheckBox('Mostrar todos os quadros')
        self.all_frames.toggled.connect(self.refresh_frames)
        filter_row.addWidget(self.all_frames)
        filter_row.addStretch()
        self.frame_counter=self.sub('Nenhum vídeo carregado')
        filter_row.addWidget(self.frame_counter)
        mid_lay.addLayout(filter_row)
        self.frame_list=QListWidget()
        self.frame_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.frame_list.setIconSize(QSize(101,101))
        self.frame_list.setGridSize(QSize(129,144))
        self.frame_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.frame_list.setMovement(QListWidget.Movement.Static)
        self.frame_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.frame_list.itemChanged.connect(self.frame_changed)
        self.frame_list.itemDoubleClicked.connect(self.edit_frame)
        self.frame_list.setMouseTracking(True)
        self.frame_list.viewport().setMouseTracking(True)
        self.frame_list.viewport().installEventFilter(self)
        self.hover_popup=SpriteHoverPreview()
        self.hover_key=None
        mid_lay.addWidget(self.frame_list,1)
        ops=QHBoxLayout()
        select_all=QPushButton('Selecionar visíveis');select_all.clicked.connect(lambda:self.select_visible(True))
        none=QPushButton('Desmarcar visíveis');none.clicked.connect(lambda:self.select_visible(False))
        ops.addWidget(select_all);ops.addWidget(none)
        mid_lay.addLayout(ops)
        self.process_more=QPushButton('Processar quadros selecionados')
        self.process_more.setObjectName('primary');self.process_more.clicked.connect(self.process_missing)
        mid_lay.addWidget(self.process_more)
        mid_lay.addWidget(self.sub('Passe o mouse para ampliar o sprite temporariamente. Dê dois cliques para corrigir manchas com seleção e confirmação manual.'))
        self.split.addWidget(self.panel(mid_lay))

        # preview / export
        right_lay=QVBoxLayout();right_lay.setContentsMargins(13,13,13,13)
        right_lay.addWidget(self.heading('PRÉVIA DA ANIMAÇÃO'))
        self.canvas_label=QLabel('Importe um vídeo para começar')
        self.canvas_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas_label.setMinimumSize(290,275)
        self.canvas_label.setStyleSheet('background:#2b3b54;border:1px solid #41566d;border-radius:10px;')
        right_lay.addWidget(self.canvas_label,1)
        self.preview_checker=QCheckBox('Mostrar transparência em xadrez (somente prévia)')
        self.preview_checker.toggled.connect(self.show_preview_frame)
        right_lay.addWidget(self.preview_checker)
        animation=QHBoxLayout()
        self.play=QPushButton('▶ Reproduzir');self.play.clicked.connect(self.toggle_preview)
        animation.addWidget(self.play)
        animation.addWidget(QLabel('FPS'))
        self.fps=QSpinBox();self.fps.setRange(1,60);self.fps.setValue(12)
        self.fps.valueChanged.connect(self.restart_timer)
        animation.addWidget(self.fps)
        right_lay.addLayout(animation)
        self.preview_count=self.sub('Nenhuma pose selecionada')
        right_lay.addWidget(self.preview_count)
        right_lay.addWidget(self.heading('EXPORTAÇÃO DO SPRITE SHEET'))
        dims=QGridLayout()
        dims.addWidget(QLabel('Largura do quadro'),0,0)
        dims.addWidget(QLabel('Altura do quadro'),0,1)
        self.cellw=QSpinBox();self.cellw.setRange(32,2048);self.cellw.setSingleStep(32);self.cellw.setValue(384)
        self.cellh=QSpinBox();self.cellh.setRange(32,2048);self.cellh.setSingleStep(32);self.cellh.setValue(512)
        dims.addWidget(self.cellw,1,0);dims.addWidget(self.cellh,1,1)
        dims.addWidget(QLabel('Colunas'),2,0);dims.addWidget(QLabel('Espaçamento'),2,1)
        self.columns=QSpinBox();self.columns.setRange(1,30);self.columns.setValue(6)
        self.spacing=QSpinBox();self.spacing.setRange(0,64);self.spacing.setValue(0)
        dims.addWidget(self.columns,3,0);dims.addWidget(self.spacing,3,1)
        right_lay.addLayout(dims)
        self.auto_fit=QCheckBox('Ajustar tamanhos diferentes automaticamente')
        self.auto_fit.setChecked(True)
        self.auto_fit.setToolTip('Mantém proporções. Quando as imagens têm resoluções diferentes, '
            'calcula uma escala por resolução, sem ampliar/reduzir cada pose individualmente.')
        self.auto_fit.toggled.connect(self.invalidate_preview)
        right_lay.addWidget(self.auto_fit)
        right_lay.addWidget(self.sub('Mesma escala dentro de cada sequência; grids iguais e personagem sem distorção.'))
        self.center=QCheckBox('Estabilizar tronco (para vídeos com deslocamento)')
        self.center.setChecked(False)
        self.center.setToolTip('Desativado (recomendado): conserva posições relativas dos quadros originais. Ative apenas quando a câmera ou personagem se deslocarem.')
        self.center.toggled.connect(self.invalidate_preview)
        right_lay.addWidget(self.center)
        self.cellw.valueChanged.connect(self.invalidate_preview)
        self.cellh.valueChanged.connect(self.invalidate_preview)
        self.export=QPushButton('⬇ Exportar sprites PNG + atlas');self.export.setObjectName('primary')
        self.export.clicked.connect(self.export_current)
        self.export.setText('⬇ Exportar PNG + atlas + GIF')
        self.gif_enabled=QCheckBox('Criar GIF animado com sprites corrigidos')
        self.gif_enabled.setChecked(True)
        self.gif_enabled.setToolTip('Usa somente os quadros selecionados, as correções salvas, a escala do grid e o FPS da prévia. GIF tem transparência de 1 bit e até 256 cores.')
        right_lay.addWidget(self.gif_enabled)
        right_lay.addWidget(self.export)
        self.export_gif_button=QPushButton('Exportar apenas GIF')
        self.export_gif_button.clicked.connect(self.export_gif_only)
        right_lay.addWidget(self.export_gif_button)
        self.last_export_info=self.sub('Nenhuma exportação nesta sessão.')
        right_lay.addWidget(self.last_export_info)
        self.open_export_folder_button=QPushButton('📂 Abrir pasta exportada')
        self.open_export_folder_button.setEnabled(False)
        self.open_export_folder_button.clicked.connect(self.open_export_folder)
        right_lay.addWidget(self.open_export_folder_button)
        right_lay.addWidget(self.sub('PNG RGBA para jogos; GIF em loop para visualizar/compartilhar a animação.'))
        self.split.addWidget(self.panel(right_lay))
        self.split.setSizes([300,740,450])

        bottom=QHBoxLayout()
        self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0)
        bottom.addWidget(self.progress,1)
        self.status=self.sub('Pronto para importar MP4, MOV, AVI ou WEBM. Você pode guiar a ferramenta marcando o personagem.')
        bottom.addWidget(self.status,2)
        self.cancel=QPushButton('Cancelar operação');self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancel_job);bottom.addWidget(self.cancel)
        all_layout.addLayout(bottom)

    def heading(self,s):
        t=QLabel(s);t.setStyleSheet('font-weight:700;letter-spacing:.7px;color:#c7d4ff;');return t

    def sub(self,s):
        t=QLabel(s);t.setObjectName('sub');t.setWordWrap(True);return t

    def closeEvent(self, event):
        self.hide_hover()
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, 'Processamento em andamento',
                'Cancele a operação e aguarde o encerramento antes de fechar o SpriteMotion.')
            event.ignore()
            return
        self.timer.stop()
        super().closeEvent(event)

    def choose_subject_roi(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            QMessageBox.warning(self, 'Não foi possível abrir o vídeo',
                                f'Falha ao abrir {Path(video_path).name} para marcar o personagem.')
            return None
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            QMessageBox.warning(self, 'Sem frame inicial',
                                f'Não foi possível ler o primeiro frame de {Path(video_path).name}.')
            return None
        try:
            dialog = SubjectSelectionDialog(frame, Path(video_path).name, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                return dialog.selected_roi()
        except Exception as e:
            QMessageBox.warning(self, 'Seleção não disponível',
                                'Não foi possível abrir o seletor de personagem.\n\n' + str(e))
        return None

    def load_storage_folder(self):
        """Default to app extraction directory; optional adjacent settings file."""
        default = Path(__file__).resolve().parent
        settings = default/'configuracoes.json'
        if settings.is_file():
            try:
                saved = json.loads(settings.read_text(encoding='utf-8')).get('project_folder', '')
                if saved and Path(saved).is_dir():
                    return Path(saved).resolve()
            except (OSError, ValueError, TypeError):
                pass
        return default

    def choose_storage_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Escolher local de armazenamento', str(self.output_root))
        if not folder:
            return
        self.output_root = Path(folder).resolve()
        self.storage_input.setText(str(self.output_root))
        config_path = Path(__file__).resolve().parent/'configuracoes.json'
        try:
            config_path.write_text(json.dumps({'project_folder': str(self.output_root)}, ensure_ascii=False, indent=2), encoding='utf-8')
            self.status.setText(f'Os próximos projetos serão salvos em {self.output_root}')
        except OSError:
            QMessageBox.information(self, 'Pasta alterada nesta sessão',
                'A pasta foi escolhida, mas não foi possível salvar a preferência no diretório do programa. '
                'Será necessário selecioná-la novamente quando reiniciar.')

    def work_dir(self):
        return self.output_root

    def start_job(self, task, on_success):
        if self.thread is not None:
            return
        self.import_button.setEnabled(False)
        self.open_project_button.setEnabled(False)
        self.storage_button.setEnabled(False)
        self.export_gif_button.setEnabled(False)
        self.reclean_button.setEnabled(False)
        self.stable_reprocess_button.setEnabled(False)
        self.cycle_button.setEnabled(False)
        self.process_more.setEnabled(False)
        self.export.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setValue(0)
        self._job_callback=on_success
        self._pending_result=None
        self._pending_error=None
        thread=QThread(self)
        job=Job(task)
        job.moveToThread(thread)
        thread.started.connect(job.run)
        job.progress.connect(self.set_progress)
        # Store the result on GUI thread; do NOT show a popup during
        # succeeded emission while QThread still needs to exit.
        job.succeeded.connect(self._record_job_success)
        job.failed.connect(self._record_job_failure)
        job.succeeded.connect(thread.quit)
        job.failed.connect(thread.quit)
        job.succeeded.connect(job.deleteLater)
        job.failed.connect(job.deleteLater)
        thread.finished.connect(self.job_finished)
        thread.finished.connect(thread.deleteLater)
        self.thread=thread
        self.job=job
        thread.start()

    @Slot(object)
    def _record_job_success(self, result):
        self._pending_result=result
        self._pending_error=None

    @Slot(str)
    def _record_job_failure(self, message):
        self._pending_error=message

    @Slot(int, str)
    def set_progress(self,p,message):
        self.progress.setValue(p)
        self.status.setText(message)

    @Slot()
    def job_finished(self):
        # QThread.finished is emitted only after the worker has stopped.
        self.thread=None
        self.job=None
        self.import_button.setEnabled(True)
        self.open_project_button.setEnabled(True)
        self.storage_button.setEnabled(True)
        self.export_gif_button.setEnabled(True)
        self.reclean_button.setEnabled(True)
        self.stable_reprocess_button.setEnabled(True)
        self.cycle_button.setEnabled(True)
        self.process_more.setEnabled(True)
        self.export.setEnabled(True)
        self.cancel.setEnabled(False)
        # Queue UI handling to the next event-loop pass to allow repaint.
        QTimer.singleShot(0,self._deliver_job_result)

    @Slot()
    def _deliver_job_result(self):
        callback=self._job_callback
        error=self._pending_error
        result=self._pending_result
        self._job_callback=None
        self._pending_error=None
        self._pending_result=None
        if error is not None:
            self.job_failed(error)
        elif callback is not None:
            try:
                callback(result)
            except Exception:
                self.job_failed('Falha ao atualizar a interface após a operação:\n'+traceback.format_exc())

    def cancel_job(self):
        if self.job:
            self.job.aborted=True
            self.status.setText('Cancelando após a etapa atual...')

    def job_failed(self,message):
        self.status.setText('Operação interrompida. Veja o erro.')
        # Open asynchronously: even failures must not block the GUI thread.
        dialog=QMessageBox(self)
        dialog.setWindowTitle('Erro no SpriteMotion')
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setText('A operação não foi concluída.')
        dialog.setDetailedText(message)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose,True)
        dialog.open()

    def open_export_folder(self):
        folder=self.last_export_folder
        if not folder or not Path(folder).is_dir():
            self.status.setText('Pasta exportada não encontrada. Faça uma nova exportação.')
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(folder).resolve()))):
            self.status.setText(f'Não foi possível abrir a pasta automaticamente: {folder}')

    def _set_export_complete(self, folder, message):
        self.last_export_folder=Path(folder)
        self.open_export_folder_button.setEnabled(True)
        self.last_export_info.setText(message+'\nPasta: '+str(folder))
        self.status.setText(message)
        self.progress.setValue(100)

    def import_videos(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Importar vídeos','','Vídeos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;Todos (*.*)')
        if not paths:return
        model=self.model.currentData()
        sensitivity=self.sensitivity.currentData()
        edge=self.edge.value()
        cm, cs, cr=self.cleanup_mode.currentData(),self.cleanup_strength.value(),self.cleanup_radius.value()
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        root=self.work_dir()
        roi_map = {}
        if self.guide_box.isChecked():
            for path in paths:
                roi = self.choose_subject_roi(path)
                if roi is not None:
                    roi_map[path] = roi
        def task(report, stopped):
            results=[]
            for i,path in enumerate(paths):
                if stopped():raise InterruptedError('Cancelado.')
                folder=root/f'{Path(path).stem}_{timestamp}_{i+1}'
                def progress(p,msg):report(round((i*100+p)/len(paths)),f'{i+1}/{len(paths)} · {msg}')
                group=process_video(path,str(folder),model,sensitivity,edge,progress,stopped,cm,cs,cr, roi_map.get(path))
                results.append(group)
            return results
        def done(results):
            for g in results:
                self.groups.append(g)
                item=QListWidgetItem(g.name)
                item.setFlags(item.flags()|Qt.ItemFlag.ItemIsEditable)
                self.groups_list.addItem(item)
            if results:self.groups_list.setCurrentRow(len(self.groups)-len(results))
            self.status.setText(f'{len(results)} grupo(s) importado(s). Os originais também foram preservados.')
        self.start_job(task,done)

    def open_project(self):
        filename,_=QFileDialog.getOpenFileName(self,'Abrir projeto SpriteMotion',
                    str(self.work_dir()),'Projetos SpriteMotion (projeto.json)')
        if not filename:return
        try:
            group=load_project(filename)
            self.groups.append(group)
            item=QListWidgetItem(group.name)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsEditable)
            self.groups_list.addItem(item)
            self.groups_list.setCurrentRow(len(self.groups)-1)
            self.status.setText(f'Projeto recuperado: {len(group.frames)} quadros. Você pode refazer a limpeza do fundo.')
        except Exception as e:
            QMessageBox.critical(self,'Não foi possível abrir o projeto',str(e))

    def reprocess_stable_current(self):
        g=self.current
        if g is None:return
        indexes=[i for i in self.selected_indexes() if g.frames[i].sprite and Path(g.frames[i].sprite).is_file()]
        if not indexes:
            QMessageBox.information(self,'Sem sprites','Selecione sprites já processados para corrigir o recorte.')
            return
        answer=QMessageBox.question(self,'Corrigir tremor do retângulo',
            f'Reprocessar {len(indexes)} sprites usando o frame inteiro?\n\n'
            'Isso substitui os PNGs selecionados, inclusive os retoques manuais.\n'
            'O SpriteMotion fará uma cópia de segurança dos PNGs e máscaras anteriores\n'
            'em uma subpasta do projeto para você poder recuperá-los.\n\n'
            'A IA local será executada novamente, o que pode levar alguns minutos.',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        if answer!=QMessageBox.StandardButton.Yes:return
        model=self.model.currentData();edge=self.edge.value()
        cm,cs,cr=self.cleanup_mode.currentData(),self.cleanup_strength.value(),self.cleanup_radius.value()
        self.timer.stop();self.play.setText('▶ Reproduzir')
        def task(report,stopped):
            return reprocess_stable_existing(g,indexes,model,edge,report,stopped,cm,cs,cr)
        def done(backup):
            self.invalidate_preview();self.refresh_frames();self.show_preview_frame()
            self.status.setText(f'Recorte estável refeito em {len(indexes)} sprites. Backup: {backup}')
        self.start_job(task,done)

    def reclean_current(self):
        g=self.current
        if g is None:return
        indexes=[i for i,fr in enumerate(g.frames) if fr.sprite and Path(fr.sprite).is_file()]
        if not indexes:
            QMessageBox.information(self,'Sem sprites','Não há sprites processados neste grupo.')
            return
        mode=self.cleanup_mode.currentData()
        strength=self.cleanup_strength.value();radius=self.cleanup_radius.value()
        answer=QMessageBox.question(self,'Refazer limpeza do fundo',
             f'Refazer a limpeza de {len(indexes)} PNGs prontos do grupo {g.name}?\n\n'
             'O recorte da IA será reutilizado, quando disponível.\n'
             'Correções feitas com pincel após a última limpeza podem ser substituídas.\n'
             'Os frames originais não serão modificados.',
             QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)
        if answer!=QMessageBox.StandardButton.Yes:return
        self.timer.stop();self.play.setText('▶ Reproduzir')
        def task(report,stopped):
            return reapply_cleanup(g,indexes,mode,strength,radius,report,stopped)
        def done(indices):
            self.invalidate_preview();self.refresh_frames();self.show_preview_frame()
            self.status.setText(f'Limpeza do fundo refeita em {len(indices)} sprites.')
        self.start_job(task,done)

    def reselect_cycle(self):
        g=self.current
        if g is None:return
        def task(report,stopped):
            report(10,'Procurando repetições de movimento...')
            window=suggest_cycle_window_for_group(g)
            return window
        def done(window):
            if not window:
                QMessageBox.information(self,'Ciclo não identificado',
                    'Não foi detectada uma repetição confiável. A seleção atual foi mantida. '
                    'Para movimentos sem repetição, selecione as poses manualmente.')
                return
            start,period=window
            if QMessageBox.question(self,'Um ciclo identificado',
                f'Foram encontrados aproximadamente {period} quadros por ciclo.\n'
                f'Melhor segmento contínuo: {start+1} a {start+period}.\n\n'
                'Selecionar esse ciclo sem saltar quadros? Os demais permanecerão no projeto.',
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:
                return
            for i,fr in enumerate(g.frames):fr.selected=(start<=i<start+period)
            self.save_project_selection()
            self.invalidate_preview();self.refresh_frames();self.show_preview_frame()
            missing=sum(not g.frames[i].sprite or not Path(g.frames[i].sprite).exists()
                        for i in range(start,start+period))
            QMessageBox.information(self,'Ciclo selecionado',
                f'{period} quadros consecutivos ({start+1} a {start+period}) escolhidos.\n'
                f'{missing} ainda precisam ter o fundo removido.\n\n'
                'Caso haja quadros sem transparência, clique em "Processar quadros selecionados".')
        self.start_job(task,done)

    def save_project_selection(self):
        g=self.current
        if g is None:return
        target=Path(g.folder)/'projeto.json'
        if not target.is_file():return
        try:
            data=__import__('json').loads(target.read_text(encoding='utf-8'))
            if len(g.frames) != int(data.get('frames',len(g.frames))):return
            data['selected_indexes']=[fr.index for fr in g.frames if fr.selected]
            target.write_text(__import__('json').dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        except (ValueError,OSError,TypeError):pass

    def selected_indexes(self):
        if self.current is None:return []
        return [i for i,f in enumerate(self.current.frames) if f.selected]

    def ready_indexes(self):
        if self.current is None:return []
        return [i for i in self.selected_indexes() if self.current.frames[i].sprite
                and Path(self.current.frames[i].sprite).exists()]

    def change_group(self,row):
        self.hide_hover()
        self.timer.stop();self.play.setText('▶ Reproduzir')
        self.current=self.groups[row] if 0<=row<len(self.groups) else None
        self.anim_index=0
        self.invalidate_preview()
        if self.current:
            self.fps.setValue(max(1,min(30,round(self.current.fps))))
        self.refresh_frames()
        self.show_preview_frame()

    def rename_group(self,item):
        row=self.groups_list.row(item)
        if 0<=row<len(self.groups) and item.text().strip():
            self.groups[row].name=item.text().strip()

    def remove_group(self):
        row=self.groups_list.currentRow()
        if row<0:return
        del self.groups[row]
        self.groups_list.takeItem(row)
        self.change_group(self.groups_list.currentRow())

    def clone_group(self):
        if self.current is None:return
        inds=self.selected_indexes()
        if not inds:return
        name,ok=QInputDialog.getText(self,'Novo grupo','Nome do movimento (ex.: Caminhada, Ataque, Pulo)')
        if not ok or not name.strip():return
        group=copy.copy(self.current)
        group.name=name.strip()
        group.frames=[copy.copy(self.current.frames[i]) for i in inds]
        self.groups.append(group)
        item=QListWidgetItem(group.name);item.setFlags(item.flags()|Qt.ItemFlag.ItemIsEditable)
        self.groups_list.addItem(item)
        self.groups_list.setCurrentRow(len(self.groups)-1)

    def thumbnail(self,path):
        try:
            with Image.open(path) as im:
                small=im.convert('RGBA')
                small.thumbnail((102,102),Image.Resampling.LANCZOS)
                p=Image.new('RGBA',(102,102),(44,60,84,255))
                p.alpha_composite(small,((102-small.width)//2,(102-small.height)//2))
                return QIcon(qpix(p))
        except Exception:return QIcon()

    def refresh_frames(self):
        self.hide_hover()
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        if self.current is None:
            self.frame_counter.setText('Nenhum vídeo carregado')
            self.frame_list.blockSignals(False)
            return
        show_all=self.all_frames.isChecked()
        total=0
        for ix,frame in enumerate(self.current.frames):
            if not show_all and not frame.selected:continue
            total+=1
            src=frame.sprite if frame.sprite and Path(frame.sprite).exists() else frame.original
            label=f'{frame.index+1:04}'
            item=QListWidgetItem(self.thumbnail(src),label)
            item.setData(Qt.ItemDataRole.UserRole,ix)
            item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if frame.selected else Qt.CheckState.Unchecked)
            item.setToolTip(f'Quadro {frame.index+1} · {frame.timestamp:.2f}s\n{frame.reason or "Pose selecionada"}' +
                            ('\nTransparência pronta' if frame.sprite else '\nAinda não processado'))
            self.frame_list.addItem(item)
        self.frame_list.blockSignals(False)
        nsel=len(self.selected_indexes());nready=len(self.ready_indexes())
        self.frame_counter.setText(f'{nsel} selecionados · {nready} transparentes / {len(self.current.frames)} originais')
        self.preview_count.setText(f'{nready} sprites prontos para prévia e exportação')

    def frame_changed(self,item):
        if self.current is None:return
        idx=item.data(Qt.ItemDataRole.UserRole)
        if idx is None:return
        fr=self.current.frames[idx]
        fr.selected=item.checkState()==Qt.CheckState.Checked
        self.save_project_selection()
        if fr.selected and not fr.sprite:
            item.setToolTip(item.toolTip()+'\nClique em Processar quadros selecionados.')
        self.invalidate_preview()
        self.frame_counter.setText(f'{len(self.selected_indexes())} selecionados · {len(self.ready_indexes())} transparentes / {len(self.current.frames)} originais')
        self.preview_count.setText(f'{len(self.ready_indexes())} sprites prontos para prévia e exportação')

    def select_visible(self,value):
        self.frame_list.blockSignals(True)
        for i in range(self.frame_list.count()):
            item=self.frame_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
            ix=item.data(Qt.ItemDataRole.UserRole)
            if self.current is not None and ix is not None:self.current.frames[ix].selected=value
        self.frame_list.blockSignals(False)
        self.save_project_selection()
        self.invalidate_preview();self.refresh_frames()

    def process_missing(self):
        g=self.current
        if not g:return
        indexes=[i for i in self.selected_indexes() if not g.frames[i].sprite or
                 not Path(g.frames[i].sprite).exists()]
        if not indexes:
            QMessageBox.information(self,'Quadros','Todos os quadros selecionados já têm transparência.')
            return
        model=self.model.currentData();edge=self.edge.value()
        cm,cs,cr=self.cleanup_mode.currentData(),self.cleanup_strength.value(),self.cleanup_radius.value()
        def task(report,stopped):
            return process_additional(g,indexes,model,edge,report,stopped,cm,cs,cr)
        def done(results):
            for idx,path in results:
                g.frames[idx].sprite=path;g.frames[idx].selected=True;g.frames[idx].reason='seleção manual'
            self.invalidate_preview();self.refresh_frames();self.show_preview_frame()
            self.status.setText(f'{len(results)} quadro(s) extra(s) processado(s).')
        self.start_job(task,done)

    def edit_frame(self,item):
        self.hide_hover()
        if not self.current:return
        idx=item.data(Qt.ItemDataRole.UserRole)
        frame=self.current.frames[idx]
        if not frame.sprite or not Path(frame.sprite).exists():
            QMessageBox.information(self,'Refino manual','Selecione este quadro e clique em Processar quadros selecionados.')
            return
        self.timer.stop();self.play.setText('▶ Reproduzir')
        if MaskDialog(frame,self).exec()==QDialog.DialogCode.Accepted:
            self.invalidate_preview();self.refresh_frames();self.show_preview_frame()
            self.status.setText(f'Correção do quadro {frame.index+1} salva.')

    def invalidate_preview(self,*args):
        self.anim_cache.clear()
        self.preview_geom=None
        self.anim_index=0
        self.show_preview_frame()

    def show_preview_frame(self):
        if not hasattr(self,'canvas_label'):return
        g=self.current
        ready=self.ready_indexes()
        if not g or not ready:
            self.canvas_label.setText('Nenhum sprite transparente selecionado')
            return
        try:
            if self.preview_geom is None:
                self.preview_geom=geometry([g.frames[i].sprite for i in ready],
                                           self.cellw.value(),self.cellh.value(),self.center.isChecked(),
                                           self.auto_fit.isChecked())
            index=ready[self.anim_index%len(ready)]
            key=(id(g),index)
            if key not in self.anim_cache:
                self.anim_cache[key]=qpix(render_cell(g.frames[index].sprite,self.preview_geom))
                if len(self.anim_cache)>48:
                    self.anim_cache.pop(next(iter(self.anim_cache)))
            sz=QSize(max(20,self.canvas_label.width()),max(20,self.canvas_label.height()))
            self.canvas_label.setPixmap(preview_pixmap(self.anim_cache[key],sz,self.preview_checker.isChecked()))
        except Exception as e:
            self.canvas_label.setText(f'Prévia indisponível: {e}')

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.preview_geom is not None:self.show_preview_frame()

    def next_frame(self):
        ready=self.ready_indexes()
        if ready:
            self.anim_index=(self.anim_index+1)%len(ready)
            self.show_preview_frame()

    def toggle_preview(self):
        if self.timer.isActive():
            self.timer.stop();self.play.setText('▶ Reproduzir');return
        if not self.ready_indexes():return
        self.timer.start(max(16,round(1000/self.fps.value())))
        self.play.setText('⏸ Pausar')

    def restart_timer(self):
        if self.timer.isActive():self.timer.start(max(16,round(1000/self.fps.value())))

    def _export_indexes(self):
        g=self.current
        if g is None:
            return None
        indices=self.selected_indexes()
        if not indices:
            QMessageBox.information(self, 'Exportação', 'Selecione alguns sprites primeiro.')
            return None
        if any(not g.frames[i].sprite or not Path(g.frames[i].sprite).exists() for i in indices):
            QMessageBox.warning(self, 'Falta transparência',
                'Existem quadros selecionados ainda não processados. Clique em Processar quadros selecionados.')
            return None
        return indices

    def _choose_export_dir(self):
        suggested = self.work_dir()/'Exportacoes'
        folder = QFileDialog.getExistingDirectory(self, 'Selecione a pasta de exportação',
                                                   str(suggested if suggested.is_dir() else self.work_dir()))
        if not folder:
            return None
        safe_name=''.join(ch if ch.isalnum() or ch in '-_ ' else '_' for ch in self.current.name).strip() or 'Animacao'
        return Path(folder)/safe_name

    def export_gif_only(self):
        indices=self._export_indexes()
        if indices is None:return
        dest=self._choose_export_dir()
        if dest is None:return
        g=self.current
        fps,w,h=self.fps.value(),self.cellw.value(),self.cellh.value()
        center,auto_fit=self.center.isChecked(),self.auto_fit.isChecked()
        def task(report,stopped):
            path=export_animation_gif(g,dest/'animacao.gif',indices,fps,w,h,center,auto_fit,report,stopped)
            return str(path)
        def done(path):
            self._set_export_complete(Path(path).parent,
                f'GIF criado: {len(indices)} sprites corrigidos, {fps} FPS. ' + Path(path).name)
        self.start_job(task,done)

    def export_current(self):
        indices=self._export_indexes()
        if indices is None:return
        dest=self._choose_export_dir()
        if dest is None:return
        g=self.current
        w,h,col,sp=self.cellw.value(),self.cellh.value(),self.columns.value(),self.spacing.value()
        center=self.center.isChecked()
        auto_fit=self.auto_fit.isChecked()
        gif_enabled=self.gif_enabled.isChecked()
        fps=self.fps.value()
        def task(report,stopped):
            # Both paths use the same selected, corrected PNG sequence and geometry.
            def progress(p,msg):
                if stopped():raise InterruptedError('Exportação cancelada.')
                report(round(p*0.75) if gif_enabled else p,msg)
            result=export_group(g,str(dest),indices,w,h,col,sp,center,progress,auto_fit)
            gif_path=None
            gif_warning=None
            if gif_enabled:
                try:
                    gif_path=export_animation_gif(g,dest/'animacao.gif',indices,fps,w,h,center,auto_fit,
                        lambda p,msg:report(75+round(p*.25),msg), stopped)
                except InterruptedError:
                    raise
                except Exception as error:
                    gif_warning=str(error)
            return result,str(dest),str(gif_path) if gif_path else None,gif_warning
        def done(result):
            (count,pages),folder,gif_path,gif_warning=result
            msg=f'{count} PNGs em {pages} folha(s) + atlas.json.'
            if gif_path:
                msg+=f' GIF criado a {fps} FPS.'
            if gif_warning:
                msg+=' GIF não criado: '+gif_warning
            self._set_export_complete(folder,msg)
        self.start_job(task,done)


def main():
    app=QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    win=Main();win.show()
    sys.exit(app.exec())


if __name__=='__main__':
    main()
