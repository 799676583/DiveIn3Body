"""Full-screen three-body screensaver with editable presets."""

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
from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QRect, QRectF, Qt, QPropertyAnimation, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QCursor, QLinearGradient, QMouseEvent, QPainter, QPen, QRadialGradient
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
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


APP_VERSION = "2.0.1"
G = 1.0
SOFTENING = 0.09
FRAME_MS = 20
DEFAULT_DT = 0.008
DEFAULT_STEPS = 1
DEFAULT_TAIL = 10000
DEFAULT_COLLISION_RADIUS = 0.02
DEFAULT_ESCAPE_DISTANCE = 9.0
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
        return Path(app_data) / "ThreeBodyScreensaver" / "saved_presets.json"
    return Path.home() / ".three_body_screensaver" / "saved_presets.json"


PRESET_FILE = preset_file_path()

APP_BG = "#02040b"
TEXT = "#eef4ff"
MUTED = "#a4afc4"
ACCENT = "#5eead4"
GLASS = "rgba(15, 24, 40, 0.84)"
GLASS_BORDER = "rgba(255, 255, 255, 0.22)"

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
    return data


def body_from_dict(raw: Dict[str, object]) -> BodyState:
    return BodyState(
        x=float(raw.get("x", 0.0)),
        y=float(raw.get("y", 0.0)),
        angle=float(raw.get("angle", 0.0)) % 360.0,
        speed=max(0.0, float(raw.get("speed", 0.8))),
        mass=max(0.05, float(raw.get("mass", 1.0))),
        color=normalized_color(raw.get("color")),
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
        shadow.setBlurRadius(34)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 120))
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
        if self._locked:
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setObjectName("spaceColorPopup")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._buttons: Dict[str, QPushButton] = {}
        self.setStyleSheet(
            f"""
            QFrame#spaceColorPopup {{
                background: rgba(8, 14, 26, 0.98);
                border: 1px solid rgba(94, 234, 212, 0.25);
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


class BodyControl(GlassFrame):
    changed = Signal()
    deleteRequested = Signal(object)

    def __init__(self, index: int, state: BodyState, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(radius=20, padding=12, parent=parent)
        self.index = index
        self.random_count = 3
        self.color = normalized_color(state.color) or color
        header = QHBoxLayout()
        self.title = QLabel(f"恒星 {index + 1}")
        self.title.setStyleSheet("font-size: 12pt; font-weight: 700;")
        self.color_button = QPushButton()
        self.color_button.setFixedSize(18, 18)
        self.color_button.setCursor(Qt.PointingHandCursor)
        self.color_button.setToolTip("点击选择星光色")
        self.color_popup = SpaceColorPopup(self)
        self.color_popup.colorSelected.connect(lambda value: self.set_color(value, emit=True))
        self.random_button = QPushButton("随机")
        self.delete_button = QPushButton("删除")
        self.random_button.setFixedWidth(56)
        self.delete_button.setFixedWidth(56)
        header.addWidget(self.title)
        header.addWidget(self.color_button)
        header.addStretch(1)
        header.addWidget(self.random_button)
        header.addWidget(self.delete_button)
        self.layout.addLayout(header)
        self._sync_color_button()

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        self.x_control = ParameterControl(ParameterSpec("x", 0.0, -8.0, 8.0, 0.01), 22)
        self.y_control = ParameterControl(ParameterSpec("y", 0.0, -8.0, 8.0, 0.01), 22)
        self.mass_control = ParameterControl(ParameterSpec("质量", 1.0, 0.05, 10.0, 0.05, True), 38)
        self.velocity_control = VelocityControl(color)
        grid.addWidget(QLabel("初始位置"), 0, 0)
        grid.addWidget(QLabel("质量"), 0, 1)
        grid.addWidget(self.x_control, 1, 0)
        grid.addWidget(self.y_control, 2, 0)
        grid.addWidget(self.mass_control, 1, 1, 2, 1)
        self.layout.addLayout(grid)
        self.layout.addWidget(QLabel("初速度方向 / 大小"))
        self.layout.addWidget(self.velocity_control)

        for control in (self.x_control, self.y_control, self.mass_control, self.velocity_control):
            control.changed.connect(self.changed.emit)
        self.color_button.clicked.connect(self._choose_color)
        self.random_button.clicked.connect(self.randomize_without_restart)
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self))
        self.set_state(state, emit=False)

    def set_index(self, index: int) -> None:
        self.index = index
        self.title.setText(f"恒星 {index + 1}")

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
        )

    def set_state(self, state: BodyState, emit: bool = True) -> None:
        color = normalized_color(state.color)
        if color:
            self.set_color(color, emit=False)
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
                border: 1px solid rgba(255, 255, 255, 0.72);
                border-radius: 5px;
            }}
            QPushButton:hover {{
                border: 2px solid rgba(255, 255, 255, 0.96);
            }}
            """
        )

    def _choose_color(self) -> None:
        self.color_popup.show_for(self.color_button, self.color)

    def _state_with_locked_parameters(self, random_state: BodyState) -> BodyState:
        current = self.state()
        return BodyState(
            x=current.x if self.x_control.is_locked() else random_state.x,
            y=current.y if self.y_control.is_locked() else random_state.y,
            angle=current.angle if self.velocity_control.angle_control.is_locked() else random_state.angle,
            speed=current.speed if self.velocity_control.speed_control.is_locked() else random_state.speed,
            mass=current.mass if self.mass_control.is_locked() else random_state.mass,
            color=self.color,
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
            (6.2, 0.070, (red, green, blue)),
            (3.2, 0.145, (red, green, blue)),
            (1.15, 0.620, (min(1.0, red * 0.56 + 0.44), min(1.0, green * 0.56 + 0.44), min(1.0, blue * 0.56 + 0.44))),
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

        self.orbit_dot = QLabel()
        self.orbit_dot.setObjectName("orbitDot")
        self.orbit_dot.setFixedSize(12, 12)
        self.title = QLabel(f"三体屏保 v{APP_VERSION}")
        self.title.setObjectName("windowTitle")
        layout.addWidget(self.orbit_dot)
        layout.addWidget(self.title)
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
            enter_fullscreen = getattr(self.window, "_enter_fullscreen", None)
            if callable(enter_fullscreen):
                enter_fullscreen()
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


class PresetComboButton(QPushButton):
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

    def _sync_text(self) -> None:
        if 0 <= self._current_index < len(self._items):
            self.setText(self._items[self._current_index][0])
        else:
            self.setText("")


class ScreensaverWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"三体屏保 v{APP_VERSION}")
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
        self.normal_geometry = None
        self.panel_expanded = True
        self.dirty = False

        self._apply_theme()
        self._build_ui()
        self._load_saved_presets()
        self._refresh_preset_combo()
        self._apply_builtin("builtin:figure_eight", restart=True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)

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
            QPushButton {{
                background: rgba(38, 53, 82, 0.82);
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 12px;
                padding: 8px 10px;
            }}
            QPushButton:hover {{
                background: rgba(54, 75, 112, 0.92);
            }}
            QPushButton:disabled {{
                color: rgba(238,244,255,0.35);
                background: rgba(35,43,60,0.45);
            }}
            QWidget#panelTrigger {{
                background: transparent;
                border: 0;
            }}
            QLineEdit {{
                background: rgba(7, 13, 26, 0.68);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 8px;
                padding: 4px 7px;
            }}
            QPushButton#presetCombo {{
                text-align: left;
                background: rgba(7, 13, 26, 0.86);
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 10px;
                padding: 7px 34px 7px 11px;
                color: {TEXT};
            }}
            QPushButton#presetCombo:hover {{
                background: rgba(12, 22, 40, 0.94);
                border-color: rgba(94,234,212,0.34);
            }}
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
                background: rgba(12, 18, 30, 0.88);
                border: 1px solid rgba(255,255,255,0.12);
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
        self.canvas.doubleClicked.connect(self._enter_fullscreen)
        self.title_bar = ScreenTitleBar(self, self)
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

        title = QLabel(f"三体屏保 v{APP_VERSION}")
        title.setObjectName("title")
        note = QLabel("Esc 缩小窗口，Q 退出。双击画面回到全屏，鼠标靠左唤出设置。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        self.panel.layout.addWidget(title)
        self.panel.layout.addWidget(note)

        top_buttons = QHBoxLayout()
        self.restart_button = QPushButton("应用 / 重启")
        self.random_button = QPushButton("随机刷新")
        self.add_body_button = QPushButton("添加恒星")
        top_buttons.addWidget(self.restart_button)
        top_buttons.addWidget(self.random_button)
        top_buttons.addWidget(self.add_body_button)
        self.panel.layout.addLayout(top_buttons)
        self.restart_button.clicked.connect(lambda: self._restart_from_controls(clear_dirty=True))
        self.random_button.clicked.connect(self._randomize_all_and_restart)
        self.add_body_button.clicked.connect(self._add_random_body)

        preset_label = QLabel("预设")
        preset_label.setObjectName("section")
        self.panel.layout.addWidget(preset_label)
        preset_row = QHBoxLayout()
        self.preset_combo = PresetComboButton()
        self.preset_combo.setMaxVisibleItems(20)
        self.load_preset_button = QPushButton("加载")
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.load_preset_button)
        self.panel.layout.addLayout(preset_row)
        preset_ops = QHBoxLayout()
        self.save_preset_button = QPushButton("保存当前")
        self.update_preset_button = QPushButton("修改")
        self.delete_preset_button = QPushButton("删除")
        preset_ops.addWidget(self.save_preset_button)
        preset_ops.addWidget(self.update_preset_button)
        preset_ops.addWidget(self.delete_preset_button)
        self.panel.layout.addLayout(preset_ops)
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        self.save_preset_button.clicked.connect(self._save_current_preset)
        self.update_preset_button.clicked.connect(self._update_selected_preset)
        self.delete_preset_button.clicked.connect(self._delete_selected_preset)

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
        self.dt_control = ParameterControl(ParameterSpec("步长", DEFAULT_DT, 0.0005, 0.03, 0.0005, True), 42)
        self.steps_control = ParameterControl(ParameterSpec("每帧", DEFAULT_STEPS, 1.0, 80.0, 1.0, True), 42)
        self.tail_control = ParameterControl(ParameterSpec("尾迹", DEFAULT_TAIL, 5.0, 10000.0, 5.0, True), 42)
        self.collision_control = ParameterControl(ParameterSpec("碰撞", DEFAULT_COLLISION_RADIUS, 0.01, 0.8, 0.01, True), 42)
        self.escape_control = ParameterControl(ParameterSpec("交互", DEFAULT_ESCAPE_DISTANCE, 2.0, 40.0, 0.5, True), 42)
        self.default_sim_button = QPushButton("恢复模拟默认值")
        for control in (self.dt_control, self.steps_control, self.tail_control, self.collision_control, self.escape_control):
            control.changed.connect(self._mark_dirty)
            sim.layout.addWidget(control)
        sim.layout.addWidget(self.default_sim_button)
        self.default_sim_button.clicked.connect(self._restore_simulation_defaults)
        self.panel.layout.addWidget(sim)

        self.status_label = QLabel("运行中")
        self.status_label.setObjectName("muted")
        self.panel.layout.addWidget(self.status_label)

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
        elif event.key() == Qt.Key_F11:
            self._toggle_fullscreen_mode()
        elif event.key() == Qt.Key_Space:
            self._toggle_panel()

    def eventFilter(self, watched, event) -> bool:
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
            self._enter_fullscreen()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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

    def _toggle_panel(self) -> None:
        self._set_panel_expanded(not self.panel_expanded)

    def _schedule_panel_hide(self) -> None:
        self.panel_hide_timer.start(620)

    def _hide_panel_if_cursor_out(self) -> None:
        if self._cursor_over_panel() or self._preset_popup_open():
            self._schedule_panel_hide()
            return
        self._set_panel_expanded(False)

    def _cursor_over_panel(self) -> bool:
        point = self.mapFromGlobal(QCursor.pos())
        return self.panel.geometry().contains(point)

    def _preset_popup_open(self) -> bool:
        view = self.preset_combo.view()
        return bool(view is not None and view.isVisible())

    def _set_panel_expanded(self, expanded: bool) -> None:
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
                "steps": self.steps_control.value(),
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
            self.steps_control.set_value(float(sim.get("steps", self.steps_control.value())), emit=False)
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

    def _append_body_control(self, state: BodyState) -> None:
        index = len(self.body_controls)
        control = BodyControl(index, state, BODY_COLORS[index % len(BODY_COLORS)])
        control.changed.connect(self._mark_dirty)
        control.deleteRequested.connect(self._delete_body_control)
        self.body_layout.addWidget(control)
        self.body_controls.append(control)
        self._update_body_buttons()

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
        self._mark_dirty()

    def _update_body_buttons(self) -> None:
        count = len(self.body_controls)
        for control in self.body_controls:
            control.set_random_count(count)
            control.delete_button.setEnabled(count > 2)
        self.add_body_button.setEnabled(count < 10)

    def _restore_simulation_defaults(self) -> None:
        self.dt_control.set_value(DEFAULT_DT, emit=False)
        self.steps_control.set_value(DEFAULT_STEPS, emit=False)
        self.tail_control.set_value(DEFAULT_TAIL, emit=False)
        self.collision_control.set_value(DEFAULT_COLLISION_RADIUS, emit=False)
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
        self._update_status()
        self._update_canvas()

    def _dt(self) -> float:
        return max(1e-6, self.dt_control.value())

    def _steps_per_frame(self) -> int:
        return max(1, int(round(self.steps_control.value())))

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

    def _update_canvas(self) -> None:
        colors = [control.color for control in self.body_controls]
        self.canvas.set_state(self.positions, self.tails, self.alive, colors, self._fade_progress(), self._collision_flash_progress())

    def _update_status(self) -> None:
        if self.ending:
            text = f"{self.ending_reason}，淡出后自动随机刷新"
        elif self.stable_binary_started_at is not None:
            elapsed = min(STABLE_BINARY_SECONDS, time.monotonic() - self.stable_binary_started_at)
            text = f"稳定双星计时：{elapsed:.0f} / {STABLE_BINARY_SECONDS:.0f} 秒"
        else:
            text = f"运行中：{len(self.alive)} / {len(self.body_controls)} 颗恒星"
        if self.dirty:
            text += "；参数已修改，请手动重启"
        self.status_label.setText(text)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ScreensaverWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
