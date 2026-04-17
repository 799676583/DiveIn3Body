"""AstralWeaver full-screen orbital screensaver."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency at runtime
    cv2 = None
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ARRAY,
    GL_COLOR_BUFFER_BIT,
    GL_FLOAT,
    GL_LINEAR,
    GL_LINE_SMOOTH,
    GL_LINE_SMOOTH_HINT,
    GL_LINE_STRIP,
    GL_MODULATE,
    GL_NICEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_QUADS,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_ARRAY,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glColorPointer,
    glColor4f,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glEnd,
    glGenTextures,
    glHint,
    glLineWidth,
    glTexCoord2f,
    glTexEnvi,
    glTexImage2D,
    glTexParameteri,
    glVertex2f,
    glVertexPointer,
    glViewport,
)
from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPoint, QPointF, QRect, QRectF, Qt, QPropertyAnimation, QStandardPaths, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QCursor, QImage, QLinearGradient, QMouseEvent, QPainter, QPen, QRadialGradient
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


APP_VERSION = "2.2.1"
APP_NAME = "AstralWeaver"
LEGACY_APP_DIR = "ThreeBodyScreensaver"
APP_DIR_NAME = "AstralWeaver"
G = 1.0
SOFTENING = 0.09
FRAME_MS = 20
DEFAULT_DT = 0.008
DEFAULT_STEPS = 1
DEFAULT_TAIL = 10000
DEFAULT_COLLISION_RADIUS = 0.02
DEFAULT_ESCAPE_DISTANCE = 9.0
DEFAULT_COLLISION_DETECTION_ENABLED = False
POSITION_LIMIT = 8.0
MASS_MIN = 0.01
MASS_MAX = 100.0
FADE_SECONDS = 3.0
COLLISION_FLASH_SECONDS = 0.24
INTERACTION_CHECK_INTERVAL = 1.2
STABLE_BINARY_SECONDS = 60.0
MAX_ACCELERATION = 42.0
MAX_SPEED = 4.8
MAX_PHYSICS_DT = 0.004
MAX_SUBSTEPS_PER_FRAME = 96
TAIL_DRAW_SEGMENTS = 10000


def preset_file_path() -> Path:
    app_data = os.getenv("APPDATA")
    if app_data:
        new_path = Path(app_data) / APP_DIR_NAME / "saved_presets.json"
        legacy_path = Path(app_data) / LEGACY_APP_DIR / "saved_presets.json"
    else:
        new_path = Path.home() / ".astral_weaver" / "saved_presets.json"
        legacy_path = Path.home() / ".three_body_screensaver" / "saved_presets.json"
    if legacy_path.exists() and not new_path.exists():
        return legacy_path
    return new_path


PRESET_FILE = preset_file_path()


def settings_file_path() -> Path:
    app_data = os.getenv("APPDATA")
    if app_data:
        new_path = Path(app_data) / APP_DIR_NAME / "settings.json"
        legacy_path = Path(app_data) / LEGACY_APP_DIR / "settings.json"
    else:
        new_path = Path.home() / ".astral_weaver" / "settings.json"
        legacy_path = Path.home() / ".three_body_screensaver" / "settings.json"
    if legacy_path.exists() and not new_path.exists():
        return legacy_path
    return new_path


SETTINGS_FILE = settings_file_path()

APP_BG = "#02040b"
TEXT = "#eef4ff"
MUTED = "#a4afc4"
ACCENT = "#5eead4"
GLASS = (
    "qlineargradient("
    "x1:0, y1:0, x2:1, y2:1, "
    "stop:0 rgba(36, 48, 68, 0.93), "
    "stop:0.38 rgba(26, 36, 52, 0.91), "
    "stop:1 rgba(14, 20, 31, 0.95))"
)
GLASS_BORDER = "rgba(255, 255, 255, 0.26)"

BODY_COLORS = (
    "#ff596d",
    "#48e0cf",
    "#ffd166",
    "#8cb4ff",
    "#f783ff",
    "#7ee081",
    "#ff9f5a",
    "#c6a7ff",
    "#6fe7ff",
    "#ffdf7e",
)

SPACE_PALETTE = (
    "#fff8d6",
    "#ffe9a3",
    "#ffd166",
    "#ffb86b",
    "#ff8f70",
    "#ff596d",
    "#ff6fae",
    "#f783ff",
    "#d7a7ff",
    "#b794ff",
    "#8cb4ff",
    "#6ea8ff",
    "#6fe7ff",
    "#48e0cf",
    "#5eead4",
    "#7ee081",
    "#b5f48a",
    "#d8ff9a",
    "#ffffff",
    "#dff7ff",
    "#b8e7ff",
    "#a8c7ff",
    "#c6a7ff",
    "#ffc3d8",
    "#ff9f5a",
    "#ffdf7e",
    "#a8ffe6",
    "#9ef7ff",
    "#a0ffbd",
    "#f5f3c4",
    "#e7dcff",
    "#ffd9a8",
)


FIGURE_EIGHT_POSITIONS = np.array(
    [[-0.97000436, 0.24308753, 0.0], [0.97000436, -0.24308753, 0.0], [0.0, 0.0, 0.0]],
    dtype=float,
)
FIGURE_EIGHT_VELOCITIES = np.array(
    [[0.466203685, 0.43236573, 0.0], [0.466203685, 0.43236573, 0.0], [-0.93240737, -0.86473146, 0.0]],
    dtype=float,
)


@dataclass(frozen=True)
class ParameterSpec:
    label: str
    default: float
    minimum: float
    maximum: float
    resolution: float
    positive: bool = False


@dataclass
class BodyState:
    x: float
    y: float
    angle: float
    speed: float
    mass: float
    color: Optional[str] = None
    name: Optional[str] = None

    def velocity(self) -> Tuple[float, float]:
        theta = math.radians(self.angle)
        return self.speed * math.cos(theta), self.speed * math.sin(theta)


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    color = QColor(hex_color)
    color.setAlpha(alpha)
    return color


def normalized_color(color: object) -> Optional[str]:
    if not isinstance(color, str):
        return None
    q = QColor(color)
    if not q.isValid():
        return None
    return q.name()


def state_from_xy_velocity(x: float, y: float, vx: float, vy: float, mass: float = 1.0) -> BodyState:
    angle = math.degrees(math.atan2(vy, vx)) % 360.0 if abs(vx) + abs(vy) > 1e-12 else 0.0
    return BodyState(x=x, y=y, angle=angle, speed=math.hypot(vx, vy), mass=mass)


def random_body_state(index: int = 0, count: int = 3) -> BodyState:
    radius = random.uniform(0.9, 2.8)
    theta = 2.0 * math.pi * index / max(count, 1) + random.uniform(-0.34, 0.34)
    x = math.cos(theta) * radius + random.uniform(-0.25, 0.25)
    y = math.sin(theta) * radius + random.uniform(-0.25, 0.25)
    tangent = (math.degrees(theta) + random.choice((-1, 1)) * 90.0 + random.uniform(-24.0, 24.0)) % 360.0
    return BodyState(x=x, y=y, angle=tangent, speed=random.uniform(0.28, 1.05), mass=random.uniform(0.55, 1.55))


def random_body_states(count: int) -> List[BodyState]:
    count = max(2, min(10, count))
    direction = random.choice((-1.0, 1.0))
    base_radius = random.uniform(1.9 + 0.13 * count, 2.8 + 0.2 * count)
    total_mass = random.uniform(0.55, 0.95) * count
    states: List[BodyState] = []
    for index in range(count):
        theta = 2.0 * math.pi * index / count + random.uniform(-0.11, 0.11)
        radius = base_radius * random.uniform(0.9, 1.16)
        x = math.cos(theta) * radius + random.uniform(-0.04, 0.04)
        y = math.sin(theta) * radius + random.uniform(-0.04, 0.04)
        tangent = (math.degrees(theta) + direction * 90.0 + random.uniform(-9.0, 9.0)) % 360.0
        orbital_speed = 0.39 * math.sqrt(total_mass / max(radius, 1.0))
        speed = orbital_speed * random.uniform(0.88, 1.08)
        states.append(BodyState(x=x, y=y, angle=tangent, speed=speed, mass=random.uniform(0.5, 1.18)))
    return states


def builtin_presets() -> Dict[str, Dict[str, object]]:
    radius = 1.0
    omega = math.sqrt(1.0 / math.sqrt(3.0))
    lagrange: List[BodyState] = []
    for theta in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6):
        lagrange.append(state_from_xy_velocity(radius * math.cos(theta), radius * math.sin(theta), -omega * radius * math.sin(theta), omega * radius * math.cos(theta)))

    euler_omega = math.sqrt(1.25)
    return {
        "builtin:figure_eight": {
            "name": "经典 8 字",
            "bodies": [
                state_from_xy_velocity(float(FIGURE_EIGHT_POSITIONS[i, 0]), float(FIGURE_EIGHT_POSITIONS[i, 1]), float(FIGURE_EIGHT_VELOCITIES[i, 0]), float(FIGURE_EIGHT_VELOCITIES[i, 1]))
                for i in range(3)
            ],
        },
        "builtin:lagrange": {"name": "拉格朗日三角", "bodies": lagrange},
        "builtin:euler": {
            "name": "欧拉共线",
            "bodies": [
                state_from_xy_velocity(-1.0, 0.0, 0.0, -euler_omega),
                state_from_xy_velocity(0.0, 0.0, 0.0, 0.0),
                state_from_xy_velocity(1.0, 0.0, 0.0, euler_omega),
            ],
        },
    }


def body_to_dict(body: BodyState) -> Dict[str, object]:
    data: Dict[str, object] = {"x": body.x, "y": body.y, "angle": body.angle, "speed": body.speed, "mass": body.mass}
    if body.color:
        data["color"] = body.color
    if body.name:
        data["name"] = body.name
    return data


def body_from_dict(raw: Dict[str, object]) -> BodyState:
    return BodyState(
        x=float(raw.get("x", 0.0)),
        y=float(raw.get("y", 0.0)),
        angle=float(raw.get("angle", 0.0)) % 360.0,
        speed=max(0.0, float(raw.get("speed", 0.8))),
        mass=min(MASS_MAX, max(MASS_MIN, float(raw.get("mass", 1.0)))),
        color=normalized_color(raw.get("color")),
        name=str(raw.get("name")).strip() if isinstance(raw.get("name"), str) and str(raw.get("name")).strip() else None,
    )


class GlassFrame(QFrame):
    def __init__(self, radius: int = 24, padding: int = 14, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("glassFrame")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(
            f"""
            QFrame#glassFrame {{
                background: {GLASS};
                border: 1px solid {GLASS_BORDER};
                border-radius: {radius}px;
            }}
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(46)
        shadow.setOffset(0, 14)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.setGraphicsEffect(shadow)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(padding, padding, padding, padding)
        self.layout.setSpacing(10)


class LockableParameterLabel(QLabel):
    doubleClicked = Signal()

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ParameterControl(QWidget):
    changed = Signal()

    def __init__(self, spec: ParameterSpec, label_width: int = 44, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._updating = False
        self._locked = False
        self._inactive = False
        self._lock_style_mode = "default"
        self.label = LockableParameterLabel(spec.label)
        self.label.setFixedWidth(label_width)
        self.label.setToolTip("双击锁定/解锁；锁定后随机时保持不变")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps())
        self.entry = QLineEdit()
        self.entry.setFixedWidth(68)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.entry)

        self.slider.valueChanged.connect(self._slider_changed)
        self.entry.editingFinished.connect(self._entry_changed)
        self.label.doubleClicked.connect(self.toggle_locked)
        self._update_lock_style()
        self.set_value(spec.default, emit=False)

    def value(self) -> float:
        return self.spec.minimum + self.slider.value() * self.spec.resolution

    def is_locked(self) -> bool:
        return self._locked

    def set_locked(self, locked: bool) -> None:
        self._locked = locked
        self._update_lock_style()

    def set_inactive(self, inactive: bool) -> None:
        self._inactive = inactive
        self._update_lock_style()

    def set_lock_style_mode(self, mode: str) -> None:
        self._lock_style_mode = mode
        self._update_lock_style()

    def toggle_locked(self) -> None:
        self.set_locked(not self._locked)

    def set_value(self, value: float, emit: bool = True) -> None:
        if self.spec.positive and value <= 0:
            value = self.spec.minimum
        value = min(max(value, self.spec.minimum), self.spec.maximum)
        step = int(round((value - self.spec.minimum) / self.spec.resolution))
        self._updating = True
        self.slider.setValue(step)
        self.entry.setText(self._format_value(self.value()))
        self._updating = False
        if emit:
            self.changed.emit()

    def _steps(self) -> int:
        return max(1, int(round((self.spec.maximum - self.spec.minimum) / self.spec.resolution)))

    def _format_value(self, value: float) -> str:
        if self.spec.resolution >= 1:
            return str(int(round(value)))
        if self.spec.resolution >= 0.01:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.5f}".rstrip("0").rstrip(".")

    def _slider_changed(self, _value: int) -> None:
        if self._updating:
            return
        self.entry.setText(self._format_value(self.value()))
        self.changed.emit()

    def _entry_changed(self) -> None:
        if self._updating:
            return
        try:
            value = float(self.entry.text().strip())
        except ValueError:
            self.entry.setText(self._format_value(self.value()))
            return
        self.set_value(value)

    def _update_lock_style(self) -> None:
        if self._inactive:
            self.label.setStyleSheet(
                """
                color: rgba(164, 175, 196, 0.48);
                background: transparent;
                border: 0;
                padding-left: 0;
                """
            )
            self.label.setToolTip("当前已禁用。双击参数名启用")
        elif self._lock_style_mode == "inline":
            if self._locked:
                self.label.setStyleSheet(
                    """
                    color: rgba(239, 246, 255, 0.96);
                    background: transparent;
                    border: 0;
                    padding-left: 0;
                    font-weight: 600;
                    """
                )
                self.label.setToolTip("已锁定：随机时保持不变。双击解锁")
            else:
                self.label.setStyleSheet(
                    """
                    color: rgba(216, 225, 239, 0.78);
                    background: transparent;
                    border: 0;
                    padding-left: 0;
                    font-weight: 600;
                    """
                )
                self.label.setToolTip("双击锁定；锁定后随机时保持不变")
        elif self._locked:
            self.label.setStyleSheet(
                f"""
                color: {ACCENT};
                background: rgba(94, 234, 212, 0.14);
                border: 1px solid rgba(94, 234, 212, 0.42);
                border-radius: 6px;
                padding-left: 4px;
                """
            )
            self.label.setToolTip("已锁定：随机时保持不变。双击解锁")
        else:
            self.label.setStyleSheet("color: rgba(238, 244, 255, 0.86); background: transparent; border: 0; padding-left: 0;")
            self.label.setToolTip("双击锁定；锁定后随机时保持不变")


class SpaceColorPopup(QFrame):
    colorSelected = Signal(str)
    visibilityChanged = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setObjectName("spaceColorPopup")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._buttons: Dict[str, QPushButton] = {}
        self.setStyleSheet(
            f"""
            QFrame#spaceColorPopup {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(34, 45, 64, 0.95),
                    stop:1 rgba(12, 18, 30, 0.96));
                border: 1px solid rgba(255, 255, 255, 0.24);
                border-radius: 14px;
            }}
            QLabel#paletteTitle {{
                color: {TEXT};
                font-size: 10pt;
                font-weight: 700;
            }}
            QLabel#paletteHint {{
                color: rgba(164, 175, 196, 0.82);
                font-size: 8pt;
            }}
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        title = QLabel("星光色")
        title.setObjectName("paletteTitle")
        hint = QLabel("点击色块应用")
        hint.setObjectName("paletteHint")
        layout.addWidget(title)
        layout.addWidget(hint)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(7)
        for index, color in enumerate(SPACE_PALETTE):
            button = QPushButton()
            button.setFixedSize(25, 25)
            button.setCursor(Qt.PointingHandCursor)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _checked=False, value=color: self._select(value))
            self._buttons[color] = button
            grid.addWidget(button, index // 8, index % 8)
        layout.addLayout(grid)
        self.set_current_color(SPACE_PALETTE[0])

    def show_for(self, anchor: QWidget, current_color: str) -> None:
        self.set_current_color(current_color)
        self.adjustSize()
        point = anchor.mapToGlobal(anchor.rect().bottomLeft())
        self.move(point.x() - 4, point.y() + 8)
        self.show()

    def set_current_color(self, current_color: str) -> None:
        current = normalized_color(current_color)
        for color, button in self._buttons.items():
            selected = normalized_color(color) == current
            button.setStyleSheet(self._button_style(color, selected))

    def _button_style(self, color: str, selected: bool) -> str:
        border = "2px solid rgba(255, 255, 255, 0.96)" if selected else "1px solid rgba(255, 255, 255, 0.32)"
        size_adjust = "margin: 0px;" if selected else "margin: 1px;"
        return (
            "QPushButton {"
            f"background: {color};"
            f"border: {border};"
            "border-radius: 7px;"
            f"{size_adjust}"
            "}"
            "QPushButton:hover {"
            "border: 2px solid rgba(255, 255, 255, 0.90);"
            "margin: 0px;"
            "}"
        )

    def _select(self, color: str) -> None:
        self.colorSelected.emit(color)
        self.hide()

    def showEvent(self, event) -> None:
        self.visibilityChanged.emit(True)
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.visibilityChanged.emit(False)
        super().hideEvent(event)


class PositionEditorPopup(QFrame):
    positionSelected = Signal(float, float)
    visibilityChanged = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setObjectName("positionEditorPopup")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(
            f"""
            QFrame#positionEditorPopup {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(34, 45, 64, 0.95),
                    stop:1 rgba(12, 18, 30, 0.96));
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 14px;
            }}
            QLabel#coordTitle {{
                color: {TEXT};
                font-size: 10pt;
                font-weight: 700;
            }}
            QLabel#coordLabel {{
                color: rgba(164, 175, 196, 0.88);
                font-size: 8.5pt;
            }}
            QLineEdit#coordEntry {{
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(188,210,238,0.08);
                border-radius: 8px;
                padding: 4px 6px;
                min-width: 68px;
            }}
            QLineEdit#coordEntry:focus {{
                border: 1px solid rgba(94,234,212,0.16);
            }}
            QPushButton#coordApply {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(190,210,238,0.08);
                border-radius: 8px;
                padding: 5px 12px;
            }}
            QPushButton#coordApply:hover {{
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(190,210,238,0.12);
            }}
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        title = QLabel("位置")
        title.setObjectName("coordTitle")
        layout.addWidget(title)

        form = QHBoxLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self.x_label = QLabel("x")
        self.x_label.setObjectName("coordLabel")
        self.x_edit = QLineEdit()
        self.x_edit.setObjectName("coordEntry")
        self.y_label = QLabel("y")
        self.y_label.setObjectName("coordLabel")
        self.y_edit = QLineEdit()
        self.y_edit.setObjectName("coordEntry")
        form.addWidget(self.x_label)
        form.addWidget(self.x_edit)
        form.addWidget(self.y_label)
        form.addWidget(self.y_edit)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        self.apply_button = QPushButton("应用")
        self.apply_button.setObjectName("coordApply")
        button_row.addWidget(self.apply_button)
        layout.addLayout(button_row)

        self.apply_button.clicked.connect(self._submit)
        self.x_edit.returnPressed.connect(self._submit)
        self.y_edit.returnPressed.connect(self._submit)

    def show_for(self, global_point: QPoint, x: float, y: float) -> None:
        self.x_edit.setText(f"{x:.2f}".rstrip("0").rstrip("."))
        self.y_edit.setText(f"{y:.2f}".rstrip("0").rstrip("."))
        self.adjustSize()
        self.move(global_point + QPoint(12, 12))
        self.show()
        self.raise_()
        self.x_edit.setFocus()
        self.x_edit.selectAll()

    def _submit(self) -> None:
        try:
            x_value = float(self.x_edit.text().strip())
            y_value = float(self.y_edit.text().strip())
        except ValueError:
            return
        x_value = min(POSITION_LIMIT, max(-POSITION_LIMIT, x_value))
        y_value = min(POSITION_LIMIT, max(-POSITION_LIMIT, y_value))
        self.positionSelected.emit(x_value, y_value)
        self.hide()

    def showEvent(self, event) -> None:
        self.visibilityChanged.emit(True)
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.visibilityChanged.emit(False)
        super().hideEvent(event)


class VelocityDial(QWidget):
    angleChanged = Signal(float)

    def __init__(self, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.color = color
        self.angle = 0.0
        self.setFixedSize(90, 90)
        self.setCursor(Qt.PointingHandCursor)

    def set_angle(self, angle: float) -> None:
        self.angle = angle % 360.0
        self.update()

    def set_color(self, color: str) -> None:
        self.color = color
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = min(self.width(), self.height()) * 0.39
        glow = QRadialGradient(center, radius * 1.15)
        glow.setColorAt(0.0, QColor(255, 255, 255, 25))
        glow.setColorAt(0.75, QColor(120, 136, 170, 35))
        glow.setColorAt(1.0, QColor(255, 255, 255, 4))
        painter.setBrush(glow)
        painter.setPen(QPen(QColor(255, 255, 255, 58), 1.0))
        painter.drawEllipse(center, radius, radius)

        theta = math.radians(self.angle)
        tip = QPointF(center.x() + math.cos(theta) * (radius - 7), center.y() - math.sin(theta) * (radius - 7))
        tail = QPointF(center.x() - math.cos(theta) * 9, center.y() + math.sin(theta) * 9)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 4.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(tail, tip)
        painter.setPen(QPen(qcolor(self.color, 240), 2.0, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(tail, tip)
        painter.setBrush(qcolor(self.color, 255))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(tip, 4, 4)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._set_from_mouse(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._set_from_mouse(event.position())

    def _set_from_mouse(self, position: QPointF) -> None:
        dx = position.x() - self.width() / 2
        dy = self.height() / 2 - position.y()
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        self.set_angle(angle)
        self.angleChanged.emit(angle)


class VelocityControl(QWidget):
    changed = Signal()

    def __init__(self, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._updating = False
        self.dial = VelocityDial(color)
        self.angle_control = ParameterControl(ParameterSpec("角度", 0.0, 0.0, 360.0, 0.01), 38)
        self.speed_control = ParameterControl(ParameterSpec("大小", 0.8, 0.0, 4.0, 0.0001), 38)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.dial)
        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self.angle_control)
        controls.addWidget(self.speed_control)
        layout.addLayout(controls, 1)
        self.dial.angleChanged.connect(self._dial_changed)
        self.angle_control.changed.connect(self._controls_changed)
        self.speed_control.changed.connect(self._controls_changed)

    def angle(self) -> float:
        return self.angle_control.value() % 360.0

    def speed(self) -> float:
        return max(0.0, self.speed_control.value())

    def set_values(self, angle: float, speed: float, emit: bool = True) -> None:
        self._updating = True
        self.angle_control.set_value(angle, emit=False)
        self.speed_control.set_value(speed, emit=False)
        self.dial.set_angle(angle)
        self._updating = False
        if emit:
            self.changed.emit()

    def _dial_changed(self, angle: float) -> None:
        self._updating = True
        self.angle_control.set_value(angle, emit=False)
        self._updating = False
        self.changed.emit()

    def _controls_changed(self) -> None:
        if self._updating:
            return
        self.dial.set_angle(self.angle())
        self.changed.emit()

    def set_color(self, color: str) -> None:
        self.dial.set_color(color)


class PositionOverview(QWidget):
    pointMoved = Signal(int, float, float)
    pointDoubleClicked = Signal(int, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._states: List[BodyState] = []
        self._drag_index: Optional[int] = None
        self._hover_index = -1
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumHeight(320)
        self.setMaximumHeight(320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setToolTip("拖动恒星点改变初始位置，双击点输入精确坐标")

    def set_states(self, states: List[BodyState]) -> None:
        self._states = [BodyState(item.x, item.y, item.angle, item.speed, item.mass, item.color) for item in states]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        plot = self._plot_rect()
        painter.setPen(QPen(QColor(170, 196, 228, 34), 1.0))
        painter.setBrush(QColor(12, 19, 31, 92))
        painter.drawRoundedRect(plot, 18, 18)
        for index in range(9):
            ratio = index / 8.0
            x = plot.left() + ratio * plot.width()
            y = plot.top() + ratio * plot.height()
            is_center = index == 4
            line_color = QColor(184, 208, 236, 62 if is_center else 18)
            line_width = 1.7 if is_center else 0.8
            painter.setPen(QPen(line_color, line_width))
            painter.drawLine(QPointF(x, plot.top() + 2), QPointF(x, plot.bottom() - 2))
            painter.drawLine(QPointF(plot.left() + 2, y), QPointF(plot.right() - 2, y))

        for index, state in enumerate(self._states):
            point = self._to_widget(state.x, state.y)
            base_color = QColor(normalized_color(state.color) or BODY_COLORS[index % len(BODY_COLORS)])
            glow_radius = 12.0 + min(8.0, math.sqrt(max(state.mass, MASS_MIN)) * 1.2)
            outer_glow = QRadialGradient(point, glow_radius * 1.4)
            outer_glow.setColorAt(0.0, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
            outer_glow.setColorAt(0.42, QColor(base_color.red(), base_color.green(), base_color.blue(), 74))
            outer_glow.setColorAt(1.0, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(outer_glow)
            painter.drawEllipse(point, glow_radius * 1.4, glow_radius * 1.4)

            core_glow = QRadialGradient(point, glow_radius)
            core_glow.setColorAt(0.0, QColor(255, 251, 240, 245))
            core_glow.setColorAt(0.18, QColor(255, 250, 236, 210))
            core_glow.setColorAt(0.36, QColor(base_color.red(), base_color.green(), base_color.blue(), 148))
            core_glow.setColorAt(1.0, QColor(base_color.red(), base_color.green(), base_color.blue(), 0))
            painter.setBrush(core_glow)
            painter.drawEllipse(point, glow_radius, glow_radius)

            radius = 3.0 + min(2.2, math.sqrt(max(state.mass, MASS_MIN)) * 0.32)
            if index == self._hover_index or index == self._drag_index:
                painter.setPen(QPen(QColor(244, 249, 255, 110), 1.0))
                painter.setBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 72))
                painter.drawEllipse(point, radius + 5.2, radius + 5.2)
                painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 252, 240, 248))
            painter.drawEllipse(point, radius, radius)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        index = self._point_at(event.position())
        if index < 0:
            return
        self._drag_index = index
        self._emit_drag(event.position())
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_index is not None and event.buttons() & Qt.LeftButton:
            self._emit_drag(event.position())
            event.accept()
            return
        hover = self._point_at(event.position())
        if hover != self._hover_index:
            self._hover_index = hover
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_index is not None:
            self._drag_index = None
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return super().mouseDoubleClickEvent(event)
        index = self._point_at(event.position())
        if index >= 0:
            self.pointDoubleClicked.emit(index, event.position().x(), event.position().y())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        if self._hover_index != -1 and self._drag_index is None:
            self._hover_index = -1
            self.update()
        super().leaveEvent(event)

    def _plot_rect(self) -> QRectF:
        outer = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        size = max(40.0, min(outer.width(), outer.height()))
        return QRectF(outer.center().x() - size / 2, outer.center().y() - size / 2, size, size)

    def _to_widget(self, x: float, y: float) -> QPointF:
        plot = self._plot_rect()
        clamped_x = max(-POSITION_LIMIT, min(POSITION_LIMIT, x))
        clamped_y = max(-POSITION_LIMIT, min(POSITION_LIMIT, y))
        px = plot.left() + ((clamped_x + POSITION_LIMIT) / (2.0 * POSITION_LIMIT)) * plot.width()
        py = plot.bottom() - ((clamped_y + POSITION_LIMIT) / (2.0 * POSITION_LIMIT)) * plot.height()
        return QPointF(px, py)

    def _from_widget(self, point: QPointF) -> Tuple[float, float]:
        plot = self._plot_rect()
        px = max(plot.left(), min(plot.right(), point.x()))
        py = max(plot.top(), min(plot.bottom(), point.y()))
        x = ((px - plot.left()) / max(1.0, plot.width())) * (2.0 * POSITION_LIMIT) - POSITION_LIMIT
        y = ((plot.bottom() - py) / max(1.0, plot.height())) * (2.0 * POSITION_LIMIT) - POSITION_LIMIT
        return x, y

    def _point_at(self, point: QPointF) -> int:
        best_index = -1
        best_distance = 14.0
        for index in reversed(range(len(self._states))):
            mapped = self._to_widget(self._states[index].x, self._states[index].y)
            distance = math.hypot(mapped.x() - point.x(), mapped.y() - point.y())
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _emit_drag(self, point: QPointF) -> None:
        if self._drag_index is None:
            return
        x, y = self._from_widget(point)
        self.pointMoved.emit(self._drag_index, x, y)
        self._hover_index = self._drag_index
        self.update()


class BodyControl(GlassFrame):
    changed = Signal()
    deleteRequested = Signal(object)

    def __init__(self, index: int, state: BodyState, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(radius=20, padding=12, parent=parent)
        self.index = index
        self.random_count = 3
        self.color = normalized_color(state.color) or color
        self.custom_name = bool(state.name and state.name.strip())
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self.title_container = QWidget()
        title_layout = QHBoxLayout(self.title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        self.title = LockableParameterLabel(self._default_name())
        self.title.setStyleSheet("font-size: 12pt; font-weight: 700;")
        self.title.setToolTip("双击改名")
        self.title_edit = QLineEdit(self._default_name())
        self.title_edit.setVisible(False)
        self.title_edit.setFixedHeight(28)
        self.title_edit.setMaxLength(20)
        self.title_edit.setPlaceholderText(self._default_name())
        self.title_edit.installEventFilter(self)
        title_layout.addWidget(self.title)
        title_layout.addWidget(self.title_edit)
        self.color_button = QPushButton()
        self.color_button.setFixedSize(18, 18)
        self.color_button.setCursor(Qt.PointingHandCursor)
        self.color_button.setToolTip("点击选择星光色")
        self.color_popup = SpaceColorPopup(self)
        self.color_popup.colorSelected.connect(lambda value: self.set_color(value, emit=True))
        self.color_popup.visibilityChanged.connect(self._handle_overlay_visibility)
        self.mass_control = ParameterControl(ParameterSpec("质量", 1.0, MASS_MIN, MASS_MAX, 0.01, True), 30)
        self.mass_control.set_lock_style_mode("inline")
        self.mass_control.layout().setSpacing(5)
        self.mass_control.slider.setFixedWidth(62)
        self.mass_control.slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(255,255,255,0.08);
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(255,255,255,0.24);
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: rgba(247,251,255,0.96);
                width: 10px;
                height: 10px;
                margin: -3px 0;
                border-radius: 5px;
            }
            """
        )
        self.mass_control.entry.setFixedWidth(52)
        self.random_button = QPushButton("随机")
        self.delete_button = QPushButton("删除")
        self.random_button.setFixedWidth(56)
        self.delete_button.setFixedWidth(56)
        header.addWidget(self.title_container, 1)
        header.addWidget(self.color_button)
        header.addWidget(self.mass_control)
        header.addStretch(1)
        header.addWidget(self.random_button)
        header.addWidget(self.delete_button)
        self.layout.addLayout(header)
        self._sync_color_button()
        self._sync_title()

        self.x_control = ParameterControl(ParameterSpec("x", 0.0, -POSITION_LIMIT, POSITION_LIMIT, 0.01), 16)
        self.y_control = ParameterControl(ParameterSpec("y", 0.0, -POSITION_LIMIT, POSITION_LIMIT, 0.01), 16)
        self.x_control.hide()
        self.y_control.hide()
        self.velocity_control = VelocityControl(color)
        self.layout.addWidget(QLabel("初速度方向 / 大小"))
        self.layout.addWidget(self.velocity_control)

        for control in (self.mass_control, self.velocity_control):
            control.changed.connect(self.changed.emit)
        self.color_button.clicked.connect(self._choose_color)
        self.random_button.clicked.connect(self.randomize_without_restart)
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self))
        self.title.doubleClicked.connect(self._begin_title_edit)
        self.title_edit.editingFinished.connect(self._commit_title_edit)
        self.set_state(state, emit=False)

    def set_index(self, index: int) -> None:
        self.index = index
        if not self.custom_name:
            self._sync_title()

    def set_random_count(self, count: int) -> None:
        self.random_count = max(2, min(10, count))

    def state(self) -> BodyState:
        return BodyState(
            x=self.x_control.value(),
            y=self.y_control.value(),
            angle=self.velocity_control.angle(),
            speed=self.velocity_control.speed(),
            mass=self.mass_control.value(),
            color=self.color,
            name=self._display_name(),
        )

    def set_state(self, state: BodyState, emit: bool = True) -> None:
        color = normalized_color(state.color)
        if color:
            self.set_color(color, emit=False)
        self.custom_name = bool(state.name and state.name.strip() and state.name.strip() != self._default_name())
        self.title_edit.setText(state.name.strip() if state.name and state.name.strip() else self._default_name())
        self._sync_title()
        self.x_control.set_value(state.x, emit=False)
        self.y_control.set_value(state.y, emit=False)
        self.mass_control.set_value(state.mass, emit=False)
        self.velocity_control.set_values(state.angle, state.speed, emit=False)
        if emit:
            self.changed.emit()

    def set_color(self, color: str, emit: bool = True) -> None:
        parsed = normalized_color(color)
        if parsed is None:
            return
        self.color = parsed
        self.velocity_control.set_color(parsed)
        self._sync_color_button()
        if emit:
            self.changed.emit()

    def _sync_color_button(self) -> None:
        self.color_button.setStyleSheet(
            f"""
            QPushButton {{
                background: {self.color};
                border: 1px solid rgba(232, 242, 255, 0.28);
                border-radius: 5px;
            }}
            QPushButton:hover {{
                border: 1px solid rgba(238, 247, 255, 0.46);
            }}
            """
        )

    def _choose_color(self) -> None:
        self._pause_cursor_for_overlay()
        self.color_popup.show_for(self.color_button, self.color)

    def eventFilter(self, watched: QObject, event) -> bool:  # type: ignore[name-defined]
        if watched == self.title_edit and event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._cancel_title_edit()
            return True
        return super().eventFilter(watched, event)

    def _default_name(self) -> str:
        return f"恒星 {self.index + 1}"

    def _display_name(self) -> str:
        text = self.title_edit.text().strip()
        return text if text else self._default_name()

    def _sync_title(self) -> None:
        text = self.title_edit.text().strip() if self.custom_name else self._default_name()
        if not text:
            text = self._default_name()
        self.title.setText(text)
        self.title_edit.setText(text)
        self.title_edit.setPlaceholderText(self._default_name())

    def _begin_title_edit(self) -> None:
        self._pause_cursor_for_overlay()
        self.title.hide()
        self.title_edit.show()
        self.title_edit.setFocus()
        self.title_edit.selectAll()

    def _commit_title_edit(self) -> None:
        if not self.title_edit.isVisible():
            return
        text = self.title_edit.text().strip()
        self.custom_name = bool(text and text != self._default_name())
        if not self.custom_name:
            text = self._default_name()
        self.title_edit.setText(text)
        self.title.setText(text)
        self.title_edit.hide()
        self.title.show()
        self.changed.emit()
        self._resume_cursor_after_overlay()

    def _cancel_title_edit(self) -> None:
        self._sync_title()
        self.title_edit.hide()
        self.title.show()
        self._resume_cursor_after_overlay()

    def _pause_cursor_for_overlay(self) -> None:
        window = self.window()
        if hasattr(window, "_pause_cursor_hide_for_popup"):
            window._pause_cursor_hide_for_popup()

    def _resume_cursor_after_overlay(self) -> None:
        window = self.window()
        if hasattr(window, "_resume_cursor_hide_after_popup"):
            window._resume_cursor_hide_after_popup()

    def _handle_overlay_visibility(self, visible: bool) -> None:
        if visible:
            self._pause_cursor_for_overlay()
        else:
            self._resume_cursor_after_overlay()

    def _state_with_locked_parameters(self, random_state: BodyState) -> BodyState:
        current = self.state()
        return BodyState(
            x=current.x if self.x_control.is_locked() else random_state.x,
            y=current.y if self.y_control.is_locked() else random_state.y,
            angle=current.angle if self.velocity_control.angle_control.is_locked() else random_state.angle,
            speed=current.speed if self.velocity_control.speed_control.is_locked() else random_state.speed,
            mass=current.mass if self.mass_control.is_locked() else random_state.mass,
            color=self.color,
            name=current.name,
        )

    def randomize_from_state(self, random_state: BodyState, emit: bool = True) -> None:
        self.set_state(self._state_with_locked_parameters(random_state), emit=emit)

    def randomize_without_restart(self) -> None:
        self.randomize_from_state(random_body_state(self.index, self.random_count), emit=True)


class TrajectoryCanvas(QOpenGLWidget):
    doubleClicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.positions = np.zeros((0, 3), dtype=float)
        self.tails: List[deque[np.ndarray]] = []
        self.alive: Set[int] = set()
        self.body_colors: List[str] = []
        self.fade_progress = 0.0
        self.collision_flash = 0.0
        self.star_texture: Optional[int] = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_state(
        self,
        positions: np.ndarray,
        tails: List[deque[np.ndarray]],
        alive: Set[int],
        body_colors: List[str],
        fade_progress: float,
        collision_flash: float,
    ) -> None:
        self.positions = positions.copy()
        self.tails = tails
        self.alive = set(alive)
        self.body_colors = list(body_colors)
        self.fade_progress = max(0.0, min(1.0, fade_progress))
        self.collision_flash = max(0.0, min(1.0, collision_flash))
        self.update()

    def initializeGL(self) -> None:
        glClearColor(0.012, 0.018, 0.04, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        self.star_texture = self._create_star_texture()

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, width), max(1, height))

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        self._draw_background_gl()
        sampled_tails = self._sampled_tails()
        if sampled_tails:
            stacked = np.vstack(list(sampled_tails.values()))
            min_xy = stacked.min(axis=0)
            max_xy = stacked.max(axis=0)
            center = (min_xy + max_xy) / 2.0
            rect = QRectF(self.rect())
            span_x = max(float(max_xy[0] - min_xy[0]), 0.5)
            span_y = max(float(max_xy[1] - min_xy[1]), 0.5)
            padding = 1.24
            scale = min(rect.width() / (span_x * padding), rect.height() / (span_y * padding))

            def to_ndc(xy: np.ndarray) -> Tuple[float, float]:
                x = (float(xy[0]) - center[0]) * scale / max(1.0, rect.width() / 2.0)
                y = (float(xy[1]) - center[1]) * scale / max(1.0, rect.height() / 2.0)
                return x, y

            for index in sorted(self.alive):
                xy = sampled_tails.get(index)
                if xy is not None:
                    self._draw_tail_gl(xy, self._body_color(index), center, scale, rect)
            for index in sorted(self.alive):
                if index < len(self.positions):
                    self._draw_star_gl(to_ndc(self.positions[index, :2]), self._body_color(index))

        if self.collision_flash > 0.0:
            self._draw_overlay_gl((1.0, 0.96, 0.82), 0.62 * self.collision_flash)

        if self.fade_progress > 0.0:
            self._draw_overlay_gl((0.0, 0.0, 0.0), 0.90 * self.fade_progress)

    def _draw_background_gl(self) -> None:
        glBegin(GL_QUADS)
        glColor4f(0.012, 0.018, 0.04, 1.0)
        glVertex2f(-1.0, 1.0)
        glColor4f(0.020, 0.027, 0.055, 1.0)
        glVertex2f(1.0, 1.0)
        glColor4f(0.000, 0.008, 0.022, 1.0)
        glVertex2f(1.0, -1.0)
        glColor4f(0.005, 0.012, 0.030, 1.0)
        glVertex2f(-1.0, -1.0)
        glEnd()

    def _sampled_tails(self) -> Dict[int, np.ndarray]:
        points: Dict[int, np.ndarray] = {}
        max_segments = self._tail_draw_segments()
        for index in self.alive:
            if index < len(self.tails) and len(self.tails[index]) > 0:
                xy = np.array(self.tails[index], dtype=float)[:, :2]
                if index < len(self.positions):
                    current = self.positions[index, :2]
                    if len(xy) == 0 or float(np.sum((xy[-1] - current) ** 2)) > 1e-14:
                        xy = np.vstack((xy, current))
                if len(xy) > max_segments:
                    sample = np.linspace(0, len(xy) - 1, max_segments, dtype=int)
                    xy = xy[sample]
                points[index] = xy
        return points

    def _tail_draw_segments(self) -> int:
        body_count = max(1, len(self.alive))
        return max(1000, min(TAIL_DRAW_SEGMENTS, int(TAIL_DRAW_SEGMENTS * 3 / body_count)))

    def _draw_tail_gl(self, xy: np.ndarray, color: str, center: np.ndarray, scale: float, rect: QRectF) -> None:
        if len(xy) < 2:
            return
        total = len(xy) - 1
        width_half = max(1.0, rect.width() / 2.0)
        height_half = max(1.0, rect.height() / 2.0)
        ndc = np.empty((len(xy), 2), dtype=np.float32)
        ndc[:, 0] = (xy[:, 0] - center[0]) * scale / width_half
        ndc[:, 1] = (xy[:, 1] - center[1]) * scale / height_half
        red, green, blue = self._rgb(color)
        age_factor = min(1.0, total / 24.0)
        progress = np.linspace(0.0, 1.0, len(xy), dtype=np.float32)
        fade = progress * progress * (3.0 - 2.0 * progress)
        fade = np.power(fade, 1.25, dtype=np.float32) * age_factor
        fade[0] = 0.0
        vertices = np.ascontiguousarray(ndc)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, vertices)
        for width, alpha, rgb in (
            (9.3, 0.070, (red, green, blue)),
            (4.8, 0.145, (red, green, blue)),
            (1.73, 0.620, (min(1.0, red * 0.56 + 0.44), min(1.0, green * 0.56 + 0.44), min(1.0, blue * 0.56 + 0.44))),
        ):
            colors = np.empty((len(xy), 4), dtype=np.float32)
            colors[:, 0] = rgb[0]
            colors[:, 1] = rgb[1]
            colors[:, 2] = rgb[2]
            colors[:, 3] = fade * alpha
            glColorPointer(4, GL_FLOAT, 0, np.ascontiguousarray(colors))
            glLineWidth(width)
            glDrawArrays(GL_LINE_STRIP, 0, len(vertices))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_star_gl(self, position: Tuple[float, float], color: str) -> None:
        if self.star_texture is None:
            self.star_texture = self._create_star_texture()
        red, green, blue = self._rgb(color)
        warm = (
            min(1.0, red * 0.26 + 0.74),
            min(1.0, green * 0.24 + 0.65),
            min(1.0, blue * 0.18 + 0.38),
        )
        for size, alpha, rgb in (
            (78.0, 0.26, (red, green, blue)),
            (30.0, 0.58, warm),
            (10.5, 1.0, (1.0, 0.98, 0.80)),
            (4.2, 0.95, (1.0, 1.0, 0.96)),
        ):
            self._draw_star_quad(position, size, rgb, alpha)

    def _create_star_texture(self, size: int = 128) -> int:
        axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        radius = np.sqrt(xx * xx + yy * yy)
        halo = np.exp(-((radius / 0.48) ** 2)) * 0.72
        core = np.exp(-((radius / 0.105) ** 2)) * 1.10
        edge = np.clip((1.0 - radius) / 0.16, 0.0, 1.0)
        edge = edge * edge * (3.0 - 2.0 * edge)
        alpha = np.clip((halo + core) * edge, 0.0, 1.0)
        data = np.zeros((size, size, 4), dtype=np.uint8)
        data[:, :, :3] = 255
        data[:, :, 3] = (alpha * 255.0).astype(np.uint8)
        texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, np.ascontiguousarray(data))
        glBindTexture(GL_TEXTURE_2D, 0)
        return texture

    def _draw_star_quad(self, position: Tuple[float, float], pixel_size: float, rgb: Tuple[float, float, float], alpha: float) -> None:
        if self.star_texture is None:
            return
        x, y = position
        half_x = pixel_size / max(1.0, float(self.width()))
        half_y = pixel_size / max(1.0, float(self.height()))
        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindTexture(GL_TEXTURE_2D, self.star_texture)
        glColor4f(rgb[0], rgb[1], rgb[2], max(0.0, min(1.0, alpha)))
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x - half_x, y - half_y)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(x + half_x, y - half_y)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(x + half_x, y + half_y)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(x - half_x, y + half_y)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

    def _draw_overlay_gl(self, rgb: Tuple[float, float, float], alpha: float) -> None:
        glBegin(GL_QUADS)
        glColor4f(rgb[0], rgb[1], rgb[2], max(0.0, min(1.0, alpha)))
        glVertex2f(-1.0, -1.0)
        glVertex2f(1.0, -1.0)
        glVertex2f(1.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

    def _rgb(self, color: str) -> Tuple[float, float, float]:
        q = QColor(color)
        return q.redF(), q.greenF(), q.blueF()

    def _body_color(self, index: int) -> str:
        if 0 <= index < len(self.body_colors):
            color = normalized_color(self.body_colors[index])
            if color:
                return color
        return BODY_COLORS[index % len(BODY_COLORS)]

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ScreenTitleBar(QWidget):
    def __init__(self, window: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.window = window
        self.drag_position: Optional[QPointF] = None
        self.setObjectName("titleBar")
        self.setFixedHeight(42)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 10, 0)
        layout.setSpacing(8)

        layout.addStretch(1)

        self.min_button = self._make_button("—", "titleButton")
        self.max_button = self._make_button("□", "titleButton")
        self.close_button = self._make_button("×", "closeButton")
        layout.addWidget(self.min_button)
        layout.addWidget(self.max_button)
        layout.addWidget(self.close_button)

        self.min_button.clicked.connect(self.window.showMinimized)
        self.max_button.clicked.connect(self._toggle_maximized)
        self.close_button.clicked.connect(self.window.close)

    def _make_button(self, text: str, object_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setFixedSize(30, 26)
        button.setFocusPolicy(Qt.NoFocus)
        return button

    def _toggle_maximized(self) -> None:
        if self.window.isMaximized():
            self.window.showNormal()
            self.max_button.setText("□")
        else:
            self.window.showMaximized()
            self.max_button.setText("❐")

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            toggle_fullscreen = getattr(self.window, "_toggle_fullscreen_mode", None)
            if callable(toggle_fullscreen):
                toggle_fullscreen()
            else:
                self._toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition() - self.window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_position is None or event.buttons() != Qt.LeftButton or self.window.isMaximized():
            return
        self.window.move((event.globalPosition() - self.drag_position).toPoint())
        event.accept()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self.drag_position = None


class RecordingIndicatorOverlay(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFixedSize(96, 24)
        self._dot_on = True
        self._started_at = 0.0
        self._elapsed_text = "00:00"
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(520)
        self.blink_timer.timeout.connect(self._toggle_dot)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(200)
        self.refresh_timer.timeout.connect(self._refresh_elapsed)
        self.hide()

    def start(self, started_at: float) -> None:
        self._started_at = started_at
        self._dot_on = True
        self._refresh_elapsed()
        self.blink_timer.start()
        self.refresh_timer.start()
        self.show()
        self.raise_()
        self.update()

    def stop(self) -> None:
        self.blink_timer.stop()
        self.refresh_timer.stop()
        self.hide()

    def _toggle_dot(self) -> None:
        self._dot_on = not self._dot_on
        self.update()

    def _refresh_elapsed(self) -> None:
        elapsed = max(0, int(time.monotonic() - self._started_at))
        minutes = elapsed // 60
        seconds = elapsed % 60
        self._elapsed_text = f"{minutes:02d}:{seconds:02d}"
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(10, 14, 24, 188))
        painter.drawRoundedRect(self.rect(), 12, 12)

        dot_center = QPointF(14.0, self.height() / 2)
        if self._dot_on:
            glow = QRadialGradient(dot_center, 9.0)
            glow.setColorAt(0.0, QColor(255, 92, 92, 200))
            glow.setColorAt(0.55, QColor(255, 58, 58, 110))
            glow.setColorAt(1.0, QColor(255, 58, 58, 0))
            painter.setBrush(glow)
            painter.drawEllipse(dot_center, 9.0, 9.0)
            painter.setBrush(QColor(255, 70, 70, 250))
            painter.drawEllipse(dot_center, 4.0, 4.0)

        painter.setPen(QColor(245, 248, 255, 230))
        text_rect = QRectF(28.0, 0.0, self.width() - 34.0, float(self.height()))
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._elapsed_text)


class PresetComboButton(QPushButton):
    visibilityChanged = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("presetCombo")
        self.setCursor(Qt.PointingHandCursor)
        self._items: List[Tuple[str, object]] = []
        self._current_index = -1
        self._max_visible_items = 20

        self.popup = QFrame(self, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.popup.setObjectName("presetPopupFrame")
        self.popup.setAttribute(Qt.WA_TranslucentBackground)
        self.popup.installEventFilter(self)
        self.popup.setStyleSheet(
            f"""
            QFrame#presetPopupFrame {{
                background: transparent;
                border: 0;
            }}
            QListWidget#presetPopupList {{
                background: rgba(10, 17, 31, 0.98);
                border: 1px solid rgba(94,234,212,0.24);
                border-radius: 12px;
                padding: 7px;
                outline: 0;
                color: {TEXT};
            }}
            QListWidget#presetPopupList::item {{
                min-height: 29px;
                padding: 6px 10px;
                border-radius: 8px;
                color: rgba(238,244,255,0.90);
            }}
            QListWidget#presetPopupList::item:hover {{
                background: rgba(255,255,255,0.08);
                color: white;
            }}
            QListWidget#presetPopupList::item:selected {{
                background: rgba(94,234,212,0.18);
                color: #f7fffd;
            }}
            """
        )
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget(self.popup)
        self.list_widget.setObjectName("presetPopupList")
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setSpacing(3)
        self.list_widget.setMouseTracking(True)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollMode(QListView.ScrollPerPixel)
        popup_layout.addWidget(self.list_widget)

        self.clicked.connect(self.show_popup)
        self.list_widget.itemClicked.connect(self._choose_item)

    def clear(self) -> None:
        self._items.clear()
        self._current_index = -1
        self.list_widget.clear()
        self._sync_text()

    def addItem(self, text: str, data: object = None) -> None:
        self._items.append((text, data))
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, data)
        self.list_widget.addItem(item)
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def count(self) -> int:
        return len(self._items)

    def currentData(self) -> object:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def findData(self, data: object) -> int:
        for index, (_text, item_data) in enumerate(self._items):
            if item_data == data:
                return index
        return -1

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items):
            return
        self._current_index = index
        self.list_widget.setCurrentRow(index)
        self._sync_text()

    def setMaxVisibleItems(self, count: int) -> None:
        self._max_visible_items = max(1, count)

    def maxVisibleItems(self) -> int:
        return self._max_visible_items

    def view(self) -> QListWidget:
        return self.list_widget

    def show_popup(self) -> None:
        if not self._items:
            return
        visible_count = min(self._max_visible_items, len(self._items))
        row_height = 34
        width = max(self.width(), 260)
        height = visible_count * row_height + 14
        self.list_widget.setFixedSize(width, height)
        self.popup.setFixedSize(width, height)
        self.list_widget.setCurrentRow(self._current_index)
        position = self.mapToGlobal(QPointF(0, self.height() + 6).toPoint())
        self.popup.move(position)
        self.popup.show()
        self.list_widget.setFocus()

    def _choose_item(self, item: QListWidgetItem) -> None:
        row = self.list_widget.row(item)
        self.setCurrentIndex(row)
        self.popup.hide()

    def eventFilter(self, watched: QObject, event) -> bool:
        if watched == self.popup:
            if event.type() == QEvent.Show:
                self.visibilityChanged.emit(True)
            elif event.type() == QEvent.Hide:
                self.visibilityChanged.emit(False)
        return super().eventFilter(watched, event)

    def _sync_text(self) -> None:
        if 0 <= self._current_index < len(self._items):
            self.setText(self._items[self._current_index][0])
        else:
            self.setText("")


class ScreensaverWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.body_controls: List[BodyControl] = []
        self.saved_presets: List[Dict[str, object]] = []
        self.alive: Set[int] = set()
        self.positions = np.zeros((0, 3), dtype=float)
        self.velocities = np.zeros((0, 3), dtype=float)
        self.masses = np.zeros(0, dtype=float)
        self.tails: List[deque[np.ndarray]] = []
        self.time = 0.0
        self.fade_frame = 0
        self.fade_frames = 0
        self.collision_flash_frame = 0
        self.collision_flash_frames = 0
        self.ending = False
        self.ending_reason = ""
        self.stable_binary_started_at: Optional[float] = None
        self.next_interaction_check_time = 0.0
        self.collision_detection_enabled = DEFAULT_COLLISION_DETECTION_ENABLED
        self.normal_geometry = None
        self.panel_expanded = True
        self.dirty = False
        self.cursor_hidden = False
        self.screenshot_in_progress = False
        self.screenshot_directory_override: Optional[Path] = None
        self.recording_directory_override: Optional[Path] = None
        self.recording_active = False
        self.recording_writer = None
        self.recording_output_path: Optional[Path] = None
        self.recording_cursor_was_hidden = False
        self.recording_hidden_widgets: List[QWidget] = []
        self.recording_frame_size: Optional[Tuple[int, int]] = None
        self.recording_frames_written = 0
        self.recording_started_at = 0.0
        self.recording_capture_busy = False
        self.status_flash_text: Optional[str] = None
        self.status_flash_deadline = 0.0
        self.status_label: Optional[QLabel] = None
        self.record_indicator = RecordingIndicatorOverlay()
        self.position_editor_index = -1

        self._apply_theme()
        self._build_ui()
        self._load_saved_presets()
        self._load_app_settings()
        self._refresh_preset_combo()
        self._apply_builtin("builtin:figure_eight", restart=True)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)
        self.recording_timer = QTimer(self)
        self.recording_timer.setInterval(40)
        self.recording_timer.timeout.connect(self._capture_recording_frame)
        self.cursor_hide_timer = QTimer(self)
        self.cursor_hide_timer.setSingleShot(True)
        self.cursor_hide_timer.timeout.connect(self._hide_cursor)
        self._schedule_cursor_hide()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget {{
                background: transparent;
                color: {TEXT};
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 10pt;
            }}
            QLabel#title {{
                font-size: 17pt;
                font-weight: 700;
            }}
            QLabel#muted {{
                color: {MUTED};
            }}
            QLabel#section {{
                color: {ACCENT};
                font-weight: 700;
            }}
            QToolTip {{
                color: rgba(236, 241, 248, 0.94);
                background-color: rgba(18, 22, 29, 0.96);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 6px 8px;
            }}
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.11),
                    stop:1 rgba(255,255,255,0.05));
                border: 1px solid rgba(190,210,238,0.06);
                border-radius: 12px;
                padding: 8px 10px;
                color: rgba(245, 250, 255, 0.96);
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.16),
                    stop:1 rgba(255,255,255,0.08));
                border-color: rgba(202,222,246,0.10);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.08),
                    stop:1 rgba(255,255,255,0.04));
                border-color: rgba(94,234,212,0.10);
            }}
            QPushButton[dirty="true"] {{
                border-color: rgba(94,234,212,0.28);
                color: rgba(247,255,253,0.98);
            }}
            QPushButton:disabled {{
                color: rgba(238,244,255,0.35);
                background: rgba(255,255,255,0.04);
                border-color: rgba(255,255,255,0.08);
            }}
            QWidget#panelTrigger {{
                background: transparent;
                border: 0;
            }}
            QLineEdit {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.08),
                    stop:1 rgba(255,255,255,0.04));
                border: 1px solid rgba(188,210,238,0.06);
                border-radius: 8px;
                padding: 4px 7px;
            }}
            QLineEdit:hover {{
                border: 1px solid rgba(188,210,238,0.08);
            }}
            QLineEdit:focus {{
                border: 1px solid rgba(94,234,212,0.12);
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.10),
                    stop:1 rgba(255,255,255,0.05));
            }}
            QPushButton#presetCombo {{
                text-align: left;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.10),
                    stop:1 rgba(255,255,255,0.05));
                border: 1px solid rgba(190,210,238,0.06);
                border-radius: 10px;
                padding: 7px 34px 7px 11px;
                color: {TEXT};
            }}
            QPushButton#presetCombo:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.15),
                    stop:1 rgba(255,255,255,0.08));
                border-color: rgba(202,222,246,0.10);
            }}
            QFrame#presetPopupFrame {{
                background: transparent;
                border: 0;
            }}
            QListWidget#presetPopupList {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(31, 42, 59, 0.95),
                    stop:1 rgba(13, 19, 31, 0.97));
                border: 1px solid rgba(255,255,255,0.22);
                border-radius: 12px;
                padding: 7px;
                outline: 0;
                color: {TEXT};
            }}
            QListWidget#presetPopupList::item {{
                min-height: 29px;
                padding: 6px 10px;
                border-radius: 8px;
                color: rgba(238,244,255,0.90);
            }}
            QListWidget#presetPopupList::item:hover {{
                background: rgba(255,255,255,0.08);
                color: white;
            }}
            QListWidget#presetPopupList::item:selected {{
                background: rgba(255,255,255,0.12);
                color: #f7fffd;
            }}
            QListWidget#presetPopupList QScrollBar:vertical {{
                background: rgba(255,255,255,0.04);
                width: 7px;
                margin: 7px 2px 7px 0;
                border-radius: 3px;
            }}
            QListWidget#presetPopupList QScrollBar::handle:vertical {{
                background: rgba(94,234,212,0.34);
                border-radius: 3px;
                min-height: 34px;
            }}
            QSlider::groove:horizontal {{
                height: 5px;
                background: rgba(255,255,255,0.10);
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: rgba(94,234,212,0.65);
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #f7fbff;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QScrollArea {{
                border: 0;
                background: transparent;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 8px;
                margin: 2px 0 2px 0;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255,255,255,0.22);
                border-radius: 4px;
                min-height: 34px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
                background: transparent;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QWidget#titleBar {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(33, 44, 62, 0.90),
                    stop:1 rgba(14, 20, 31, 0.92));
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 16px;
            }}
            QLabel#windowTitle {{
                color: rgba(238,244,255,0.90);
                font-size: 10pt;
                font-weight: 600;
            }}
            QLabel#orbitDot {{
                background: qradialgradient(cx:0.35, cy:0.30, radius:0.8,
                    fx:0.35, fy:0.30,
                    stop:0 rgba(255,255,245,230),
                    stop:0.38 rgba(94,234,212,210),
                    stop:1 rgba(94,234,212,30));
                border-radius: 6px;
            }}
            QPushButton#titleButton, QPushButton#closeButton {{
                background: transparent;
                border: 0;
                border-radius: 8px;
                padding: 0;
                color: rgba(238,244,255,0.72);
                font-size: 11pt;
            }}
            QPushButton#titleButton:hover {{
                background: rgba(255,255,255,0.10);
                color: rgba(238,244,255,0.96);
            }}
            QPushButton#closeButton:hover {{
                background: rgba(255,86,112,0.82);
                color: white;
            }}
            """
        )

    def _build_ui(self) -> None:
        self.canvas = TrajectoryCanvas(self)
        self.canvas.setMouseTracking(True)
        self.canvas.doubleClicked.connect(self._handle_canvas_double_click)
        self.title_bar = ScreenTitleBar(self, self)
        self.title_bar.setMouseTracking(True)
        self.panel = GlassFrame(radius=26, padding=14, parent=self)
        self.panel.setFixedWidth(470)
        self.panel.installEventFilter(self)
        self.panel.setMouseTracking(True)
        self.panel_trigger = QWidget(self)
        self.panel_trigger.setObjectName("panelTrigger")
        self.panel_trigger.installEventFilter(self)
        self.panel_trigger.setMouseTracking(True)
        self.panel_animation = QPropertyAnimation(self.panel, b"geometry", self)
        self.panel_animation.setDuration(260)
        self.panel_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.panel_hide_timer = QTimer(self)
        self.panel_hide_timer.setSingleShot(True)
        self.panel_hide_timer.timeout.connect(self._hide_panel_if_cursor_out)

        top_buttons = QHBoxLayout()
        top_buttons.setSpacing(6)
        self.restart_button = QPushButton("应用 / 重启")
        self.random_button = QPushButton("随机刷新")
        self.add_body_button = QPushButton("添加恒星")
        self.screenshot_button = QPushButton("截图")
        self.record_button = QPushButton("录屏")
        top_buttons.addWidget(self.restart_button)
        top_buttons.addWidget(self.random_button)
        top_buttons.addWidget(self.add_body_button)
        top_buttons.addWidget(self.screenshot_button)
        top_buttons.addWidget(self.record_button)
        self.panel.layout.addLayout(top_buttons)
        self.restart_button.clicked.connect(lambda: self._restart_from_controls(clear_dirty=True))
        self.random_button.clicked.connect(self._randomize_all_and_restart)
        self.add_body_button.clicked.connect(self._add_random_body)
        self.screenshot_button.clicked.connect(self._save_screenshot)
        self.record_button.clicked.connect(self._toggle_recording)
        self.screenshot_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.screenshot_button.customContextMenuRequested.connect(self._choose_screenshot_directory)
        self.record_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.record_button.customContextMenuRequested.connect(self._choose_recording_directory)
        self.random_button.setToolTip("随机生成当前恒星数量的参数，并立即重启模拟")
        self.add_body_button.setToolTip("新增一颗恒星，最多 10 颗")

        preset_label = QLabel("预设")
        preset_label.setObjectName("section")
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self.preset_combo = PresetComboButton()
        self.preset_combo.setMaxVisibleItems(20)
        self.preset_combo.setToolTip("选择要加载、保存、修改或删除的预设")
        self.preset_combo.visibilityChanged.connect(self._handle_popup_visibility)
        self.load_preset_button = QPushButton("加载")
        self.save_preset_button = QPushButton("保存当前")
        self.update_preset_button = QPushButton("修改")
        self.delete_preset_button = QPushButton("删除")
        self.load_preset_button.setFixedWidth(54)
        self.save_preset_button.setFixedWidth(82)
        self.update_preset_button.setFixedWidth(58)
        self.delete_preset_button.setFixedWidth(58)
        self.load_preset_button.setToolTip("加载当前选中的预设")
        self.save_preset_button.setToolTip("把当前参数保存为新预设")
        self.update_preset_button.setToolTip("用当前参数覆盖选中的自定义预设")
        self.delete_preset_button.setToolTip("删除当前选中的自定义预设")
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.load_preset_button)
        preset_row.addWidget(self.save_preset_button)
        preset_row.addWidget(self.update_preset_button)
        preset_row.addWidget(self.delete_preset_button)
        self.panel.layout.addLayout(preset_row)
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        self.save_preset_button.clicked.connect(self._save_current_preset)
        self.update_preset_button.clicked.connect(self._update_selected_preset)
        self.delete_preset_button.clicked.connect(self._delete_selected_preset)

        self.position_overview = PositionOverview()
        self.position_overview.pointMoved.connect(self._set_body_position_from_overview)
        self.position_overview.pointDoubleClicked.connect(self._edit_body_position_from_overview)
        self.panel.layout.addWidget(self.position_overview)
        self.position_editor_popup = PositionEditorPopup(self)
        self.position_editor_popup.positionSelected.connect(self._apply_position_popup_value)
        self.position_editor_popup.visibilityChanged.connect(self._handle_popup_visibility)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.body_layout = QVBoxLayout(self.scroll_content)
        self.body_layout.setContentsMargins(0, 0, 8, 0)
        self.body_layout.setSpacing(12)
        self.scroll.setWidget(self.scroll_content)
        self.panel.layout.addWidget(self.scroll, 1)

        sim = GlassFrame(radius=20, padding=12)
        sim.layout.addWidget(QLabel("模拟控制"))
        self.dt_control = ParameterControl(ParameterSpec("速度", DEFAULT_DT, 0.0005, 0.03, 0.0005, True), 42)
        self.steps_control = ParameterControl(ParameterSpec("每帧", DEFAULT_STEPS, 1.0, 80.0, 1.0, True), 42)
        self.tail_control = ParameterControl(ParameterSpec("尾迹", DEFAULT_TAIL, 5.0, 10000.0, 5.0, True), 42)
        self.collision_control = ParameterControl(ParameterSpec("碰撞", DEFAULT_COLLISION_RADIUS, 0.01, 0.8, 0.01, True), 42)
        self.collision_control.label.doubleClicked.disconnect(self.collision_control.toggle_locked)
        self.collision_control.label.doubleClicked.connect(self._toggle_collision_detection)
        self.escape_control = ParameterControl(ParameterSpec("交互", DEFAULT_ESCAPE_DISTANCE, 2.0, 40.0, 0.5, True), 42)
        self._bind_sim_default_reset(self.dt_control, "控制模拟推进速度")
        self._bind_sim_default_reset(self.tail_control, "控制尾迹保留长度")
        for control in (self.dt_control, self.tail_control):
            control.changed.connect(self._mark_dirty)
            sim.layout.addWidget(control)
        self._sync_collision_control_state()
        self.panel.layout.addWidget(sim)
        self._update_restart_button_state()

    def resizeEvent(self, _event) -> None:
        self.canvas.setGeometry(self.rect())
        normal_mode = not self.isFullScreen()
        title_margin = 10
        title_height = 42
        self.title_bar.setVisible(normal_mode)
        if normal_mode:
            self.title_bar.setGeometry(12, title_margin, max(260, self.width() - 24), title_height)
            self.title_bar.raise_()
        margin = 18
        top_offset = title_margin + title_height + 12 if normal_mode else 0
        panel_top = margin + top_offset
        panel_height = max(220, self.height() - panel_top - margin)
        shown, hidden, trigger = self._current_panel_rects()
        self.panel_trigger.setGeometry(trigger)
        self.panel.setGeometry(shown if self.panel_expanded else hidden)
        if self.panel_expanded:
            self.panel.raise_()
        self.panel_trigger.raise_()
        self._position_record_indicator()

    def moveEvent(self, _event) -> None:
        self._position_record_indicator()
        super().moveEvent(_event)

    def _current_panel_rects(self) -> Tuple[QRect, QRect, QRect]:
        normal_mode = not self.isFullScreen()
        title_margin = 10
        title_height = 42
        margin = 18
        top_offset = title_margin + title_height + 12 if normal_mode else 0
        panel_top = margin + top_offset
        panel_height = max(220, self.height() - panel_top - margin)
        return self._panel_rects(margin, panel_top, panel_height)

    def _panel_rects(self, margin: int, panel_top: int, panel_height: int) -> Tuple[QRect, QRect, QRect]:
        panel_width = self.panel.width()
        shown = QRect(margin, panel_top, panel_width, panel_height)
        hidden = QRect(-panel_width - 10, panel_top, panel_width, panel_height)
        trigger = QRect(0, panel_top, 18, panel_height)
        return shown, hidden, trigger

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._leave_fullscreen()
        elif event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_R:
            self._restart_from_controls(clear_dirty=True)
        elif event.key() == Qt.Key_F9:
            self._toggle_recording()
        elif event.key() == Qt.Key_F11:
            self._toggle_fullscreen_mode()
        elif event.key() in (Qt.Key_F12, Qt.Key_Print):
            self._save_screenshot()
        elif event.key() == Qt.Key_Space:
            self._toggle_panel()

    def eventFilter(self, watched, event) -> bool:
        if isinstance(watched, QWidget) and (watched is self or self.isAncestorOf(watched) or watched.window() is self):
            if event.type() in (QEvent.MouseMove, QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.Wheel):
                self._register_cursor_activity()
        if watched == self.panel:
            if event.type() == QEvent.Enter:
                self.panel_hide_timer.stop()
                self._set_panel_expanded(True)
            elif event.type() == QEvent.Leave:
                self._schedule_panel_hide()
        elif watched == self.panel_trigger and event.type() == QEvent.Enter:
            self._set_panel_expanded(True)
        return super().eventFilter(watched, event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._toggle_fullscreen_mode()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _handle_canvas_double_click(self) -> None:
        self._toggle_fullscreen_mode()

    def _toggle_fullscreen_mode(self) -> None:
        if self.isFullScreen():
            self._leave_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        if self.isFullScreen():
            return
        if self.isVisible() and not self.isMaximized() and not self.isMinimized():
            self.normal_geometry = self.geometry()
        self.title_bar.hide()
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.showFullScreen()
        self._register_cursor_activity()

    def _leave_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.showNormal()
        self.title_bar.show()
        if self.normal_geometry is not None and self.normal_geometry.width() >= 640 and self.normal_geometry.height() >= 420:
            self.setGeometry(self.normal_geometry)
        else:
            screen = self.screen() or QApplication.primaryScreen()
            available = screen.availableGeometry() if screen is not None else None
            if available is not None:
                width = min(1280, max(900, int(available.width() * 0.78)))
                height = min(820, max(620, int(available.height() * 0.78)))
                self.resize(width, height)
                self.move(available.center() - self.rect().center())
            else:
                self.resize(1180, 760)
        self.title_bar.max_button.setText("□")
        self.raise_()
        self.activateWindow()
        self.cursor_hide_timer.stop()
        self._show_cursor()

    def _toggle_panel(self) -> None:
        if self.recording_active:
            return
        self._set_panel_expanded(not self.panel_expanded)

    def _schedule_panel_hide(self) -> None:
        self.panel_hide_timer.start(620)

    def _hide_panel_if_cursor_out(self) -> None:
        if self._cursor_over_panel() or self._interactive_popup_open():
            self._schedule_panel_hide()
            return
        self._set_panel_expanded(False)

    def _cursor_over_panel(self) -> bool:
        point = self.mapFromGlobal(QCursor.pos())
        return self.panel.geometry().contains(point)

    def _preset_popup_open(self) -> bool:
        view = self.preset_combo.view()
        return bool(view is not None and view.isVisible())

    def _interactive_popup_open(self) -> bool:
        if self._preset_popup_open() or self.position_editor_popup.isVisible():
            return True
        for control in self.body_controls:
            if control.color_popup.isVisible() or control.title_edit.isVisible():
                return True
        return False

    def _pause_cursor_hide_for_popup(self) -> None:
        self.cursor_hide_timer.stop()
        self._show_cursor()

    def _resume_cursor_hide_after_popup(self) -> None:
        self._register_cursor_activity()

    def _handle_popup_visibility(self, visible: bool) -> None:
        if visible:
            self._pause_cursor_hide_for_popup()
        else:
            self.position_editor_index = -1
            self._resume_cursor_hide_after_popup()

    def _set_panel_expanded(self, expanded: bool) -> None:
        if self.recording_active and expanded:
            return
        if self.panel_expanded == expanded and self.panel_animation.state() != QPropertyAnimation.Running:
            return
        self.panel_expanded = expanded
        shown, hidden, trigger = self._current_panel_rects()
        self.panel_trigger.setGeometry(trigger)
        target = shown if expanded else hidden
        self.panel_animation.stop()
        self.panel_animation.setStartValue(self.panel.geometry())
        self.panel_animation.setEndValue(target)
        self.panel_animation.start()
        if expanded:
            self.panel.raise_()
        self.panel_trigger.raise_()

    def _position_record_indicator(self) -> None:
        if not self.recording_active:
            return
        anchor = self.mapToGlobal(QPoint(self.width() - self.record_indicator.width() - 18, 18))
        self.record_indicator.move(anchor)

    def _load_saved_presets(self) -> None:
        if not PRESET_FILE.exists():
            self.saved_presets = []
            return
        try:
            raw = json.loads(PRESET_FILE.read_text(encoding="utf-8"))
            self.saved_presets = raw if isinstance(raw, list) else []
        except (OSError, json.JSONDecodeError):
            self.saved_presets = []

    def _write_saved_presets(self) -> None:
        PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRESET_FILE.write_text(json.dumps(self.saved_presets, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_app_settings(self) -> None:
        self.screenshot_directory_override = None
        self.recording_directory_override = None
        if not SETTINGS_FILE.exists():
            self._update_screenshot_button_tooltip()
            self._update_record_button_tooltip()
            return
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._update_screenshot_button_tooltip()
            self._update_record_button_tooltip()
            return
        if isinstance(raw, dict):
            screenshot_directory = raw.get("screenshot_directory")
            recording_directory = raw.get("recording_directory")
            if isinstance(screenshot_directory, str) and screenshot_directory.strip():
                self.screenshot_directory_override = Path(screenshot_directory).expanduser()
            if isinstance(recording_directory, str) and recording_directory.strip():
                self.recording_directory_override = Path(recording_directory).expanduser()
        self._update_screenshot_button_tooltip()
        self._update_record_button_tooltip()

    def _write_app_settings(self) -> None:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "screenshot_directory": str(self.screenshot_directory_override) if self.screenshot_directory_override is not None else "",
            "recording_directory": str(self.recording_directory_override) if self.recording_directory_override is not None else "",
        }
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_screenshot_button_tooltip(self) -> None:
        directory = self._screenshot_directory()
        self.screenshot_button.setToolTip(f"左键截图，右键设置默认保存路径\n快捷键：F12\n当前路径：{directory}")

    def _update_record_button_tooltip(self) -> None:
        directory = self._recording_directory()
        action = "点击停止录屏（快捷键：F9）" if self.recording_active else "左键开始录屏，右键设置默认保存路径（快捷键：F9）"
        self.record_button.setToolTip(f"{action}\n当前路径：{directory}")

    def _update_restart_button_state(self) -> None:
        if self.dirty:
            self.restart_button.setProperty("dirty", True)
            self.restart_button.setToolTip("应用当前参数并重启模拟\n快捷键：R\n当前有未应用的修改")
        else:
            self.restart_button.setProperty("dirty", False)
            self.restart_button.setToolTip("应用当前参数并重启模拟\n快捷键：R")
        self.restart_button.style().unpolish(self.restart_button)
        self.restart_button.style().polish(self.restart_button)
        self.restart_button.update()

    def _sync_record_button(self) -> None:
        self.record_button.setText("停止录屏" if self.recording_active else "录屏")
        self._update_record_button_tooltip()

    def _refresh_preset_combo(self) -> None:
        current = self.preset_combo.currentData()
        self.preset_combo.clear()
        for key, preset in builtin_presets().items():
            self.preset_combo.addItem(str(preset["name"]), key)
        for index, preset in enumerate(self.saved_presets):
            self.preset_combo.addItem(str(preset.get("name", f"自定义 {index + 1}")), f"saved:{index}")
        if current is not None:
            found = self.preset_combo.findData(current)
            if found >= 0:
                self.preset_combo.setCurrentIndex(found)

    def _load_selected_preset(self) -> None:
        key = self.preset_combo.currentData()
        if not key:
            return
        if str(key).startswith("builtin:"):
            self._apply_builtin(str(key), restart=True)
            return
        preset = self._saved_preset_from_key(str(key))
        if preset is not None:
            self._apply_preset_dict(preset, restart=True)

    def _save_current_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "保存预设", "预设名称：")
        if not ok or not name.strip():
            return
        preset = self._snapshot_preset(name.strip())
        existing_index = next((i for i, item in enumerate(self.saved_presets) if item.get("name") == name.strip()), None)
        if existing_index is None:
            self.saved_presets.append(preset)
        else:
            self.saved_presets[existing_index] = preset
        self._write_saved_presets()
        self._refresh_preset_combo()

    def _update_selected_preset(self) -> None:
        key = str(self.preset_combo.currentData())
        if not key.startswith("saved:"):
            QMessageBox.information(self, "不能修改", "内置预设不可修改，请先保存为自定义预设。")
            return
        index = int(key.split(":", 1)[1])
        name = str(self.saved_presets[index].get("name", f"自定义 {index + 1}"))
        self.saved_presets[index] = self._snapshot_preset(name)
        self._write_saved_presets()
        self._refresh_preset_combo()

    def _delete_selected_preset(self) -> None:
        key = str(self.preset_combo.currentData())
        if not key.startswith("saved:"):
            QMessageBox.information(self, "不能删除", "内置预设不可删除。")
            return
        index = int(key.split(":", 1)[1])
        del self.saved_presets[index]
        self._write_saved_presets()
        self._refresh_preset_combo()

    def _saved_preset_from_key(self, key: str) -> Optional[Dict[str, object]]:
        if not key.startswith("saved:"):
            return None
        index = int(key.split(":", 1)[1])
        if 0 <= index < len(self.saved_presets):
            return self.saved_presets[index]
        return None

    def _snapshot_preset(self, name: str) -> Dict[str, object]:
        return {
            "name": name,
            "bodies": [body_to_dict(control.state()) for control in self.body_controls],
            "simulation": {
                "dt": self.dt_control.value(),
                "steps": 1,
                "tail": self.tail_control.value(),
                "collision": self.collision_control.value(),
                "escape": self.escape_control.value(),
            },
        }

    def _apply_builtin(self, key: str, restart: bool) -> None:
        preset = builtin_presets()[key]
        self._set_body_states(list(preset["bodies"]))
        if restart:
            self._restart_from_controls(clear_dirty=True)

    def _apply_preset_dict(self, preset: Dict[str, object], restart: bool) -> None:
        bodies = [body_from_dict(item) for item in preset.get("bodies", []) if isinstance(item, dict)]
        if not bodies:
            return
        self._set_body_states(bodies[:10])
        sim = preset.get("simulation", {})
        if isinstance(sim, dict):
            self.dt_control.set_value(float(sim.get("dt", self.dt_control.value())), emit=False)
            self.steps_control.set_value(1.0, emit=False)
            self.tail_control.set_value(float(sim.get("tail", self.tail_control.value())), emit=False)
            self.collision_control.set_value(float(sim.get("collision", self.collision_control.value())), emit=False)
            self.escape_control.set_value(float(sim.get("escape", self.escape_control.value())), emit=False)
        if restart:
            self._restart_from_controls(clear_dirty=True)

    def _set_body_states(self, states: List[BodyState]) -> None:
        states = states[:10]
        while len(states) < 2:
            states.append(random_body_state(len(states), 2))
        for control in self.body_controls:
            self.body_layout.removeWidget(control)
            control.setParent(None)
            control.deleteLater()
        self.body_controls.clear()
        for state in states:
            self._append_body_control(state)
        self._update_body_buttons()
        self._sync_position_overview()

    def _append_body_control(self, state: BodyState) -> None:
        index = len(self.body_controls)
        control = BodyControl(index, state, BODY_COLORS[index % len(BODY_COLORS)])
        control.changed.connect(self._handle_body_control_changed)
        control.deleteRequested.connect(self._delete_body_control)
        self.body_layout.addWidget(control)
        self.body_controls.append(control)
        self._update_body_buttons()
        self._sync_position_overview()

    def _add_random_body(self) -> None:
        if len(self.body_controls) >= 10:
            return
        self._append_body_control(random_body_state(len(self.body_controls), len(self.body_controls) + 1))
        self._mark_dirty()

    def _delete_body_control(self, control: BodyControl) -> None:
        if len(self.body_controls) <= 2:
            return
        self.body_controls.remove(control)
        self.body_layout.removeWidget(control)
        control.setParent(None)
        control.deleteLater()
        for index, body_control in enumerate(self.body_controls):
            body_control.set_index(index)
        self._update_body_buttons()
        self._sync_position_overview()
        self._mark_dirty()

    def _update_body_buttons(self) -> None:
        count = len(self.body_controls)
        for control in self.body_controls:
            control.set_random_count(count)
            control.delete_button.setEnabled(count > 2)
        self.add_body_button.setEnabled(count < 10)

    def _sync_position_overview(self) -> None:
        if hasattr(self, "position_overview"):
            self.position_overview.set_states([control.state() for control in self.body_controls])

    def _handle_body_control_changed(self) -> None:
        self._sync_position_overview()
        self._mark_dirty()

    def _set_body_position_from_overview(self, index: int, x: float, y: float) -> None:
        if not 0 <= index < len(self.body_controls):
            return
        control = self.body_controls[index]
        control.x_control.set_value(x, emit=False)
        control.y_control.set_value(y, emit=False)
        self._sync_position_overview()
        self._mark_dirty()

    def _edit_body_position_from_overview(self, index: int, local_x: float, local_y: float) -> None:
        if not 0 <= index < len(self.body_controls):
            return
        current = self.body_controls[index].state()
        self.position_editor_index = index
        global_point = self.position_overview.mapToGlobal(QPoint(int(local_x), int(local_y)))
        self._pause_cursor_hide_for_popup()
        self.position_editor_popup.show_for(global_point, current.x, current.y)

    def _apply_position_popup_value(self, x: float, y: float) -> None:
        if self.position_editor_index >= 0:
            self._set_body_position_from_overview(self.position_editor_index, x, y)
        self.position_editor_index = -1

    def _toggle_collision_detection(self) -> None:
        self.collision_detection_enabled = not self.collision_detection_enabled
        self._sync_collision_control_state()
        self._mark_dirty()

    def _sync_collision_control_state(self) -> None:
        enabled = self.collision_detection_enabled
        self.collision_control.slider.setEnabled(enabled)
        self.collision_control.entry.setEnabled(enabled)
        self.collision_control.set_locked(False)
        self.collision_control.set_inactive(not enabled)
        self.collision_control.label.setToolTip("碰撞判定已启用，双击关闭" if enabled else "碰撞判定默认关闭，双击参数名启用")
        self.collision_control.setToolTip("碰撞判定已启用" if enabled else "碰撞判定默认关闭，双击参数名启用")

    def _bind_sim_default_reset(self, control: ParameterControl, description: str) -> None:
        try:
            control.label.doubleClicked.disconnect(control.toggle_locked)
        except (TypeError, RuntimeError):
            pass
        control.label.doubleClicked.connect(lambda c=control: self._reset_sim_control(c))
        tooltip = f"{description}\n双击参数名恢复默认值"
        control.label.setToolTip(tooltip)
        control.setToolTip(tooltip)

    def _reset_sim_control(self, control: ParameterControl) -> None:
        control.set_value(control.spec.default, emit=False)
        self._mark_dirty()

    def _restore_simulation_defaults(self) -> None:
        self.dt_control.set_value(DEFAULT_DT, emit=False)
        self.steps_control.set_value(1.0, emit=False)
        self.tail_control.set_value(DEFAULT_TAIL, emit=False)
        self.collision_control.set_value(DEFAULT_COLLISION_RADIUS, emit=False)
        self.collision_detection_enabled = DEFAULT_COLLISION_DETECTION_ENABLED
        self._sync_collision_control_state()
        self.escape_control.set_value(DEFAULT_ESCAPE_DISTANCE, emit=False)
        self._mark_dirty()

    def _randomize_all_and_restart(self) -> None:
        count = max(2, min(10, len(self.body_controls) or 3))
        random_states = random_body_states(count)
        if len(self.body_controls) == count:
            for control, state in zip(self.body_controls, random_states):
                control.randomize_from_state(state, emit=False)
        else:
            self._set_body_states(random_states)
        self._restart_from_controls(clear_dirty=True)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_restart_button_state()
        self._update_status()

    def _restart_from_controls(self, clear_dirty: bool) -> None:
        states = [control.state() for control in self.body_controls]
        count = len(states)
        self.positions = np.zeros((count, 3), dtype=float)
        self.velocities = np.zeros((count, 3), dtype=float)
        self.masses = np.zeros(count, dtype=float)
        for index, state in enumerate(states):
            vx, vy = state.velocity()
            self.positions[index, 0] = state.x
            self.positions[index, 1] = state.y
            self.velocities[index, 0] = vx
            self.velocities[index, 1] = vy
            self.masses[index] = state.mass

        self.alive = set(range(count))
        self._lock_center_of_mass(shift_tails=False)
        self.tails = [deque(maxlen=self._tail_length()) for _ in range(count)]
        for index in range(count):
            self.tails[index].append(self.positions[index].copy())
        self.time = 0.0
        self.fade_frame = 0
        self.fade_frames = 0
        self.collision_flash_frame = 0
        self.collision_flash_frames = 0
        self.ending = False
        self.ending_reason = ""
        self.stable_binary_started_at = None
        self.next_interaction_check_time = self.time + INTERACTION_CHECK_INTERVAL
        if clear_dirty:
            self.dirty = False
        self._update_restart_button_state()
        self._update_status()
        self._update_canvas()

    def _dt(self) -> float:
        return max(1e-6, self.dt_control.value())

    def _steps_per_frame(self) -> int:
        return 1

    def _tail_length(self) -> int:
        return max(2, int(round(self.tail_control.value())))

    def _collision_radius(self) -> float:
        return max(1e-6, self.collision_control.value())

    def _interaction_distance(self) -> float:
        return max(1.0, self.escape_control.value())

    def _tick(self) -> None:
        if self.ending:
            self.fade_frame += 1
            self._advance_collision_flash()
            if self.fade_frame >= self.fade_frames:
                self._randomize_all_and_restart()
            else:
                self._update_canvas()
            return

        total_dt = self._dt() * self._steps_per_frame()
        substeps = self._adaptive_substeps(total_dt)
        step_dt = total_dt / substeps
        for _ in range(substeps):
            self._integrate_one_step(step_dt)
            if self.ending:
                break
        if not self.ending:
            self._check_end_conditions()
        self._update_canvas()
        self._advance_collision_flash()

    def _adaptive_substeps(self, total_dt: float) -> int:
        substeps = max(self._steps_per_frame(), int(math.ceil(total_dt / MAX_PHYSICS_DT)))
        minimum_distance = self._minimum_alive_distance()
        if minimum_distance < 0.85:
            substeps *= min(5, max(2, int(math.ceil(0.85 / max(minimum_distance, 0.16)))))
        max_speed = self._max_alive_speed()
        travel = max_speed * total_dt
        if travel > 0.16:
            substeps = max(substeps, int(math.ceil(travel / 0.16)) * self._steps_per_frame())
        return max(1, min(MAX_SUBSTEPS_PER_FRAME, substeps))

    def _minimum_alive_distance(self) -> float:
        if len(self.alive) < 2:
            return float("inf")
        minimum = float("inf")
        alive_list = sorted(self.alive)
        for offset, i in enumerate(alive_list):
            for j in alive_list[offset + 1 :]:
                displacement = self.positions[j, :2] - self.positions[i, :2]
                minimum = min(minimum, math.sqrt(max(0.0, float(np.dot(displacement, displacement)))))
        return minimum

    def _max_alive_speed(self) -> float:
        maximum = 0.0
        for index in self.alive:
            velocity = self.velocities[index, :2]
            maximum = max(maximum, math.sqrt(max(0.0, float(np.dot(velocity, velocity)))))
        return maximum

    def _integrate_one_step(self, dt: float) -> None:
        alive_list = sorted(self.alive)
        if not alive_list:
            return
        acceleration = self._accelerations(self.positions)
        new_positions = self.positions.copy()
        new_positions[alive_list] = self.positions[alive_list] + self.velocities[alive_list] * dt + 0.5 * acceleration[alive_list] * dt * dt
        new_acceleration = self._accelerations(new_positions)
        new_velocities = self.velocities.copy()
        new_velocities[alive_list] = self.velocities[alive_list] + 0.5 * (acceleration[alive_list] + new_acceleration[alive_list]) * dt
        self._limit_speeds(new_velocities, alive_list)
        self.velocities = new_velocities
        self.positions = new_positions
        self.time += dt
        collided = self._handle_collisions()
        if not collided:
            self._lock_center_of_mass(shift_tails=False)
        for index in self.alive:
            self.tails[index].append(self.positions[index].copy())

    def _accelerations(self, positions: np.ndarray) -> np.ndarray:
        accelerations = np.zeros_like(positions)
        softening_squared = SOFTENING * SOFTENING
        alive_list = sorted(self.alive)
        for offset, i in enumerate(alive_list):
            for j in alive_list[offset + 1 :]:
                displacement = positions[j] - positions[i]
                distance_squared = float(np.dot(displacement, displacement) + softening_squared)
                factor = G * displacement / (distance_squared ** 1.5)
                accelerations[i] += self.masses[j] * factor
                accelerations[j] -= self.masses[i] * factor
        self._limit_accelerations(accelerations, alive_list)
        return accelerations

    def _limit_accelerations(self, accelerations: np.ndarray, alive_list: List[int]) -> None:
        for index in alive_list:
            vector = accelerations[index, :2]
            magnitude = math.sqrt(max(0.0, float(np.dot(vector, vector))))
            if magnitude > MAX_ACCELERATION:
                accelerations[index, :2] *= MAX_ACCELERATION / magnitude

    def _limit_speeds(self, velocities: np.ndarray, alive_list: List[int]) -> None:
        for index in alive_list:
            vector = velocities[index, :2]
            speed = math.sqrt(max(0.0, float(np.dot(vector, vector))))
            if speed > MAX_SPEED:
                velocities[index, :2] *= MAX_SPEED / speed

    def _handle_collisions(self) -> bool:
        if not self.collision_detection_enabled:
            return False
        if len(self.alive) < 2:
            return False
        radius = self._collision_radius()
        radius_squared = radius * radius
        collided: Set[int] = set()
        alive_list = sorted(self.alive)
        for offset, i in enumerate(alive_list):
            if i in collided:
                continue
            for j in alive_list[offset + 1 :]:
                if j in collided:
                    continue
                displacement = self.positions[j] - self.positions[i]
                if float(np.dot(displacement, displacement)) <= radius_squared:
                    collided.add(i)
                    collided.add(j)
                    break
        if collided:
            self.alive.difference_update(collided)
            self.stable_binary_started_at = None
            self._lock_center_of_mass(shift_tails=True)
            self._trigger_collision_flash()
            if len(self.alive) <= 1:
                self._begin_ending("仅剩一个恒星")
            return True
        return False

    def _lock_center_of_mass(self, shift_tails: bool) -> None:
        if not self.alive:
            return
        indices = sorted(self.alive)
        weights = self.masses[indices]
        total_mass = float(weights.sum())
        if total_mass <= 0.0:
            return
        center = np.average(self.positions[indices], axis=0, weights=weights)
        velocity = np.average(self.velocities[indices], axis=0, weights=weights)
        self.positions[indices] = self.positions[indices] - center
        self.velocities[indices] = self.velocities[indices] - velocity
        if shift_tails:
            for index in indices:
                if index >= len(self.tails) or not self.tails[index]:
                    continue
                self.tails[index] = deque((point - center for point in self.tails[index]), maxlen=self.tails[index].maxlen)

    def _check_end_conditions(self) -> None:
        alive_count = len(self.alive)
        if alive_count <= 1:
            self._begin_ending("仅剩一个恒星")
            return

        if alive_count == 2 and self._is_stable_binary():
            if self.stable_binary_started_at is None:
                self.stable_binary_started_at = time.monotonic()
            if time.monotonic() - self.stable_binary_started_at >= STABLE_BINARY_SECONDS:
                self._begin_ending("稳定双星保持一分钟")
            else:
                self._update_status()
        else:
            if self.stable_binary_started_at is not None:
                self.stable_binary_started_at = None
                self._update_status()

        if self.time >= self.next_interaction_check_time:
            self.next_interaction_check_time = self.time + INTERACTION_CHECK_INTERVAL
            if self._all_future_encounters_impossible():
                self._begin_ending("恒星已不会再交汇")

    def _all_future_encounters_impossible(self) -> bool:
        if len(self.alive) < 2:
            return False
        alive_list = sorted(self.alive)
        for offset, i in enumerate(alive_list):
            for j in alive_list[offset + 1 :]:
                if self._pair_can_still_interact(i, j):
                    return False
        return True

    def _pair_can_still_interact(self, first: int, second: int) -> bool:
        interaction_distance = self._interaction_distance()
        interaction_distance_squared = interaction_distance * interaction_distance
        displacement = self.positions[second, :2] - self.positions[first, :2]
        distance_squared = float(np.dot(displacement, displacement))
        if distance_squared <= interaction_distance_squared:
            return True

        relative_velocity = self.velocities[second, :2] - self.velocities[first, :2]
        speed_squared = float(np.dot(relative_velocity, relative_velocity))
        if speed_squared < 1e-10:
            return True

        approach = float(np.dot(displacement, relative_velocity))
        closest_time = -approach / speed_squared
        if closest_time <= 0.0:
            return False

        closest_distance_squared = max(0.0, distance_squared - approach * approach / speed_squared)
        return closest_distance_squared <= interaction_distance_squared

    def _is_stable_binary(self) -> bool:
        if len(self.alive) != 2:
            return False
        first, second = sorted(self.alive)
        displacement = self.positions[second, :2] - self.positions[first, :2]
        distance_squared = float(np.dot(displacement, displacement))
        return distance_squared <= self._interaction_distance() ** 2 and self._pair_can_still_interact(first, second)

    def _begin_ending(self, reason: str) -> None:
        if self.ending:
            return
        self.ending = True
        self.ending_reason = reason
        self.fade_frame = 0
        self.fade_frames = max(1, int(round(FADE_SECONDS * 1000 / FRAME_MS)))
        self._update_status()

    def _trigger_collision_flash(self) -> None:
        self.collision_flash_frame = 0
        self.collision_flash_frames = max(1, int(round(COLLISION_FLASH_SECONDS * 1000 / FRAME_MS)))

    def _advance_collision_flash(self) -> None:
        if self.collision_flash_frames <= 0:
            return
        self.collision_flash_frame += 1
        if self.collision_flash_frame >= self.collision_flash_frames:
            self.collision_flash_frame = 0
            self.collision_flash_frames = 0

    def _fade_progress(self) -> float:
        if not self.ending or self.fade_frames <= 0:
            return 0.0
        return min(1.0, self.fade_frame / self.fade_frames)

    def _collision_flash_progress(self) -> float:
        if self.collision_flash_frames <= 0:
            return 0.0
        t = self.collision_flash_frame / self.collision_flash_frames
        return max(0.0, math.cos(t * math.pi / 2.0) ** 2.2)

    def _schedule_cursor_hide(self) -> None:
        if self.recording_active:
            return
        if self._interactive_popup_open():
            self.cursor_hide_timer.stop()
            self._show_cursor()
            return
        if self.isFullScreen():
            self.cursor_hide_timer.start(1000)
        else:
            self.cursor_hide_timer.stop()
            self._show_cursor()

    def _register_cursor_activity(self) -> None:
        if self.recording_active:
            return
        self._show_cursor()
        self._schedule_cursor_hide()

    def _hide_cursor(self) -> None:
        if not self.isFullScreen() or self.cursor_hidden:
            return
        QApplication.setOverrideCursor(Qt.BlankCursor)
        self.cursor_hidden = True

    def _show_cursor(self) -> None:
        if not self.cursor_hidden:
            return
        QApplication.restoreOverrideCursor()
        self.cursor_hidden = False

    def _screenshot_directory(self) -> Path:
        candidates: List[Path] = []
        if self.screenshot_directory_override is not None:
            candidates.append(self.screenshot_directory_override.expanduser())
        pictures = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        if pictures:
            candidates.append(Path(pictures))
        home_pictures = Path.home() / "Pictures"
        if home_pictures not in candidates:
            candidates.append(home_pictures)
        candidates.append(Path.cwd() / "screenshots")
        for base in candidates:
            directory = base / APP_DIR_NAME
            try:
                directory.mkdir(parents=True, exist_ok=True)
                return directory
            except OSError:
                continue
        return Path.cwd()

    def _recording_directory(self) -> Path:
        candidates: List[Path] = []
        if self.recording_directory_override is not None:
            candidates.append(self.recording_directory_override.expanduser())
        pictures = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        if pictures:
            candidates.append(Path(pictures))
        home_videos = Path.home() / "Videos"
        if home_videos not in candidates:
            candidates.append(home_videos)
        candidates.append(Path.cwd() / "recordings")
        for base in candidates:
            directory = base / APP_DIR_NAME
            try:
                directory.mkdir(parents=True, exist_ok=True)
                return directory
            except OSError:
                continue
        return Path.cwd()

    def _choose_screenshot_directory(self) -> None:
        start_dir = self.screenshot_directory_override or self._screenshot_directory()
        selected = QFileDialog.getExistingDirectory(self, "选择默认截图保存路径", str(start_dir))
        if not selected:
            return
        self.screenshot_directory_override = Path(selected).expanduser()
        self._write_app_settings()
        self._update_screenshot_button_tooltip()
        self._flash_status(f"默认截图路径已设置：{self.screenshot_directory_override.name}")

    def _choose_recording_directory(self) -> None:
        start_dir = self.recording_directory_override or self._recording_directory()
        selected = QFileDialog.getExistingDirectory(self, "选择默认录屏保存路径", str(start_dir))
        if not selected:
            return
        self.recording_directory_override = Path(selected).expanduser()
        self._write_app_settings()
        self._update_record_button_tooltip()
        self._flash_status(f"默认录屏路径已设置：{self.recording_directory_override.name}")

    def _hide_overlays_for_capture(self) -> List[QWidget]:
        hidden: List[QWidget] = []
        self.panel_animation.stop()
        for widget in (self.panel, self.title_bar, self.preset_combo.popup, self.position_editor_popup):
            if widget.isVisible():
                widget.hide()
                hidden.append(widget)
        for control in self.body_controls:
            if control.color_popup.isVisible():
                control.color_popup.hide()
                hidden.append(control.color_popup)
            if control.title_edit.isVisible():
                control._commit_title_edit()
        return hidden

    def _restore_after_capture(self, hidden_widgets: List[QWidget], timer_was_active: bool, cursor_was_hidden: bool) -> None:
        for widget in hidden_widgets:
            widget.show()
        if not self.isFullScreen():
            self.title_bar.show()
            self.title_bar.raise_()
        if self.panel_expanded:
            self.panel.show()
            self.panel.raise_()
        if not cursor_was_hidden:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        if timer_was_active and not self.timer.isActive():
            self.timer.start(FRAME_MS)
        self._update_screenshot_button_tooltip()
        self._update_record_button_tooltip()
        self._schedule_cursor_hide()
        self.screenshot_in_progress = False

    def _capture_canvas_pixmap(self):
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else QApplication.primaryScreen()
        if screen is None:
            global_rect = QRect(self.canvas.mapToGlobal(self.canvas.rect().topLeft()), self.canvas.size())
            screen = QApplication.screenAt(global_rect.center())
            if screen is None:
                return None
            return screen.grabWindow(0, global_rect.x(), global_rect.y(), global_rect.width(), global_rect.height())
        return screen.grabWindow(int(self.winId()), self.canvas.x(), self.canvas.y(), self.canvas.width(), self.canvas.height())

    def _capture_screen_screenshot(self, hidden_widgets: List[QWidget], timer_was_active: bool, cursor_was_hidden: bool) -> None:
        pixmap = self._capture_canvas_pixmap()

        saved = False
        path: Optional[Path] = None
        if pixmap is not None and not pixmap.isNull():
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            directory = self._screenshot_directory()
            path = directory / f"ThreeBody-{timestamp}.png"
            suffix = 1
            while path.exists():
                path = directory / f"ThreeBody-{timestamp}-{suffix}.png"
                suffix += 1
            saved = pixmap.save(str(path), "PNG")

        self._restore_after_capture(hidden_widgets, timer_was_active, cursor_was_hidden)
        if saved and path is not None:
            self._flash_status(f"截图已保存：{path.name}")
        else:
            self._flash_status("截图失败")

    def _save_screenshot(self) -> None:
        if self.recording_active:
            self._flash_status("录屏中，请先停止录屏再截图")
            return
        if self.screenshot_in_progress:
            return
        self.screenshot_in_progress = True
        timer_was_active = self.timer.isActive()
        if timer_was_active:
            self.timer.stop()
        cursor_was_hidden = self.cursor_hidden
        if not cursor_was_hidden:
            QApplication.setOverrideCursor(Qt.BlankCursor)
        hidden_widgets = self._hide_overlays_for_capture()
        self.repaint()
        self.canvas.repaint()
        QApplication.processEvents()
        QTimer.singleShot(35, lambda: self._capture_screen_screenshot(hidden_widgets, timer_was_active, cursor_was_hidden))

    def _pixmap_to_bgr_frame(self, pixmap) -> Optional[np.ndarray]:
        if pixmap is None or pixmap.isNull():
            return None
        image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        width = image.width()
        height = image.height()
        if width <= 1 or height <= 1:
            return None
        bytes_per_line = image.bytesPerLine()
        ptr = image.bits()
        array = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line // 4, 4))[:, :width, :].copy()
        frame = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR) if cv2 is not None else None
        if frame is None:
            return None
        even_height = frame.shape[0] - (frame.shape[0] % 2)
        even_width = frame.shape[1] - (frame.shape[1] % 2)
        if even_height < 2 or even_width < 2:
            return None
        return frame[:even_height, :even_width]

    def _video_output_path(self, directory: Path, extension: str) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = directory / f"ThreeBody-{timestamp}{extension}"
        suffix = 1
        while path.exists():
            path = directory / f"ThreeBody-{timestamp}-{suffix}{extension}"
            suffix += 1
        return path

    def _open_video_writer(self, frame: np.ndarray) -> Tuple[Optional[object], Optional[Path]]:
        if cv2 is None:
            return None, None
        directory = self._recording_directory()
        size = (int(frame.shape[1]), int(frame.shape[0]))
        for codec, extension in (("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi"), ("MJPG", ".avi")):
            path = self._video_output_path(directory, extension)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), 25.0, size)
            if writer.isOpened():
                return writer, path
            writer.release()
        return None, None

    def _toggle_recording(self) -> None:
        if self.recording_active:
            self._stop_recording(save=True)
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self.screenshot_in_progress:
            self._flash_status("截图处理中，请稍后再录屏")
            return
        if self.recording_active:
            return
        if cv2 is None:
            self._flash_status("录屏依赖缺失：未安装 OpenCV")
            return
        self.recording_active = True
        self.recording_writer = None
        self.recording_output_path = None
        self.recording_frame_size = None
        self.recording_frames_written = 0
        self.recording_started_at = time.monotonic()
        self.recording_capture_busy = False
        self.recording_hidden_widgets = self._hide_overlays_for_capture()
        self.cursor_hide_timer.stop()
        self._show_cursor()
        self.recording_cursor_was_hidden = False
        self.panel_trigger.setEnabled(False)
        self._sync_record_button()
        self._position_record_indicator()
        self.record_indicator.start(self.recording_started_at)
        self.repaint()
        self.canvas.repaint()
        QApplication.processEvents()
        QTimer.singleShot(35, self._initialize_recording)

    def _initialize_recording(self) -> None:
        if not self.recording_active:
            return
        pixmap = self._capture_canvas_pixmap()
        frame = self._pixmap_to_bgr_frame(pixmap)
        if frame is None:
            self._stop_recording(save=False, message="录屏启动失败")
            return
        writer, path = self._open_video_writer(frame)
        if writer is None or path is None:
            self._stop_recording(save=False, message="录屏启动失败：无法创建视频文件")
            return
        self.recording_writer = writer
        self.recording_output_path = path
        self.recording_frame_size = (frame.shape[1], frame.shape[0])
        self.recording_writer.write(frame)
        self.recording_frames_written = 1
        self.recording_timer.start()
        self._flash_status(f"开始录屏：{path.name}（按 F9 结束）")

    def _capture_recording_frame(self) -> None:
        if not self.recording_active or self.recording_writer is None or cv2 is None:
            return
        if self.recording_capture_busy:
            return
        self.recording_capture_busy = True
        try:
            pixmap = self._capture_canvas_pixmap()
            frame = self._pixmap_to_bgr_frame(pixmap)
            if frame is None:
                return
            if self.recording_frame_size is not None and (frame.shape[1], frame.shape[0]) != self.recording_frame_size:
                frame = cv2.resize(frame, self.recording_frame_size, interpolation=cv2.INTER_AREA)
            self.recording_writer.write(frame)
            self.recording_frames_written += 1
        finally:
            self.recording_capture_busy = False

    def _stop_recording(self, save: bool, message: Optional[str] = None) -> None:
        if not self.recording_active and self.recording_writer is None:
            return
        self.recording_timer.stop()
        writer = self.recording_writer
        path = self.recording_output_path
        frames_written = self.recording_frames_written
        self.recording_writer = None
        self.recording_output_path = None
        self.recording_frame_size = None
        self.recording_frames_written = 0
        self.recording_capture_busy = False
        if writer is not None:
            writer.release()
        self.recording_active = False
        self.panel_trigger.setEnabled(True)
        self._restore_after_capture(self.recording_hidden_widgets, timer_was_active=False, cursor_was_hidden=self.recording_cursor_was_hidden)
        self.recording_hidden_widgets = []
        self.record_indicator.stop()
        self._sync_record_button()
        if message:
            self._flash_status(message)
        elif save and path is not None and frames_written > 0:
            self._flash_status(f"录屏已保存：{path.name}")
        elif save:
            self._flash_status("录屏已取消")

    def _flash_status(self, text: str, seconds: float = 2.6) -> None:
        self.status_flash_text = text
        self.status_flash_deadline = time.monotonic() + max(0.2, seconds)
        if self.status_label is not None:
            self.status_label.setText(text)

    def _update_canvas(self) -> None:
        colors = [control.color for control in self.body_controls]
        self.canvas.set_state(self.positions, self.tails, self.alive, colors, self._fade_progress(), self._collision_flash_progress())

    def _update_status(self) -> None:
        if self.status_label is None:
            return
        if self.status_flash_text and time.monotonic() < self.status_flash_deadline:
            self.status_label.setText(self.status_flash_text)
            return
        self.status_flash_text = None
        if self.ending:
            text = self.ending_reason
        elif self.stable_binary_started_at is not None:
            elapsed = min(STABLE_BINARY_SECONDS, time.monotonic() - self.stable_binary_started_at)
            text = f"稳定双星 {elapsed:.0f}/{STABLE_BINARY_SECONDS:.0f}"
        else:
            text = f"运行中 {len(self.alive)}/{len(self.body_controls)}"
        self.status_label.setText(text)

    def closeEvent(self, event) -> None:
        if self.recording_active or self.recording_writer is not None:
            self._stop_recording(save=True)
        self.record_indicator.stop()
        self.cursor_hide_timer.stop()
        self._show_cursor()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ScreensaverWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
