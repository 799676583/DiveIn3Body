"""PySide6 three-body simulator with a glass-style interface."""

from __future__ import annotations

import math
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QMouseEvent, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


G = 1.0
SOFTENING = 0.02
FRAME_MS = 30
MAX_ENERGY_POINTS = 2500
APP_VERSION = "1.0.0"

APP_BG = "#070b14"
GLASS_CARD = "rgba(30, 43, 67, 0.62)"
GLASS_BORDER = "rgba(255, 255, 255, 0.16)"
TEXT = "#eef4ff"
MUTED = "#9eabc2"
ACCENT = "#5eead4"
SPACE_BG = "#050914"
PLOT_BG = "#0a1020"

BODY_COLORS = ("#ff596d", "#48e0cf", "#ffd166")
BODY_NAMES = ("恒星 1", "恒星 2", "恒星 3")

DEFAULT_POSITIONS = np.array(
    [
        [-0.97000436, 0.24308753, 0.0],
        [0.97000436, -0.24308753, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)
DEFAULT_VELOCITIES = np.array(
    [
        [0.466203685, 0.43236573, 0.0],
        [0.466203685, 0.43236573, 0.0],
        [-0.93240737, -0.86473146, 0.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    resolution: float
    positive: bool = False


def qcolor(hex_color: str, alpha: int = 255) -> QColor:
    color = QColor(hex_color)
    color.setAlpha(alpha)
    return color


def velocity_angle_and_speed(body_index: int) -> Tuple[float, float]:
    vx = float(DEFAULT_VELOCITIES[body_index, 0])
    vy = float(DEFAULT_VELOCITIES[body_index, 1])
    return math.degrees(math.atan2(vy, vx)) % 360.0, math.hypot(vx, vy)


def build_parameter_specs() -> List[ParameterSpec]:
    specs: List[ParameterSpec] = []
    for body_index in range(3):
        for axis_index, axis in enumerate(("x", "y")):
            specs.append(
                ParameterSpec(
                    key=f"p{body_index}_{axis}",
                    label=f"{BODY_NAMES[body_index]} {axis}",
                    default=float(DEFAULT_POSITIONS[body_index, axis_index]),
                    minimum=-5.0,
                    maximum=5.0,
                    resolution=0.01,
                )
            )

        angle, speed = velocity_angle_and_speed(body_index)
        specs.append(ParameterSpec(f"a{body_index}", f"{BODY_NAMES[body_index]} 速度角度", angle, 0.0, 360.0, 0.01))
        specs.append(ParameterSpec(f"s{body_index}", f"{BODY_NAMES[body_index]} 速度大小", speed, 0.0, 4.0, 0.0001))
        specs.append(ParameterSpec(f"m{body_index}", f"{BODY_NAMES[body_index]} 质量", 1.0, 0.05, 10.0, 0.05, True))

    specs.extend(
        [
            ParameterSpec("dt", "时间步长", 0.004, 0.0005, 0.03, 0.0005, True),
            ParameterSpec("steps", "每帧步数", 8.0, 1.0, 80.0, 1.0, True),
            ParameterSpec("tail", "尾迹长度", 260.0, 20.0, 1200.0, 10.0, True),
            ParameterSpec("collision", "碰撞距离", 0.08, 0.01, 0.5, 0.01, True),
        ]
    )
    return specs


PARAMETER_SPECS = build_parameter_specs()


class GlassFrame(QFrame):
    def __init__(self, radius: int = 24, padding: int = 14, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("glassFrame")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(
            f"""
            QFrame#glassFrame {{
                background: {GLASS_CARD};
                border: 1px solid {GLASS_BORDER};
                border-radius: {radius}px;
            }}
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 118))
        self.setGraphicsEffect(shadow)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(padding, padding, padding, padding)
        self.layout.setSpacing(10)


class ParameterControl(QWidget):
    changed = Signal()

    def __init__(self, spec: ParameterSpec, label_text: str = "", label_width: int = 46, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.spec = spec
        self.minimum = spec.minimum
        self.maximum = spec.maximum
        self.resolution = spec.resolution
        self._updating = False

        self.label = QLabel(label_text or spec.label)
        self.label.setFixedWidth(label_width)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self._steps())
        self.entry = QLineEdit()
        self.entry.setFixedWidth(66)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.entry)

        self.slider.valueChanged.connect(self._slider_changed)
        self.entry.editingFinished.connect(self._entry_changed)
        self.set_value(spec.default, emit=False)

    def value(self) -> float:
        return self.minimum + self.slider.value() * self.resolution

    def set_value(self, value: float, emit: bool = True) -> None:
        if self.spec.positive and value <= 0:
            value = self.minimum
        value = min(max(value, self.minimum), self.maximum)
        step = round((value - self.minimum) / self.resolution)
        self._updating = True
        self.slider.setValue(int(step))
        self.entry.setText(self._format_value(self.value()))
        self._updating = False
        if emit:
            self.changed.emit()

    def _steps(self) -> int:
        return max(1, int(round((self.maximum - self.minimum) / self.resolution)))

    def _format_value(self, value: float) -> str:
        if self.resolution >= 1:
            return str(int(round(value)))
        if self.resolution >= 0.01:
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
        if self.spec.positive and value <= 0:
            QMessageBox.warning(self, "无效数值", f"{self.spec.label} 必须为正数。")
            self.entry.setText(self._format_value(self.value()))
            return
        self.set_value(value)


class VelocityDial(QWidget):
    angleChanged = Signal(float)

    def __init__(self, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.color = color
        self.angle = 0.0
        self.setFixedSize(110, 110)
        self.setCursor(Qt.PointingHandCursor)

    def set_angle(self, angle: float) -> None:
        self.angle = angle % 360.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = min(self.width(), self.height()) * 0.39

        glow = QRadialGradient(center, radius * 1.15)
        glow.setColorAt(0.0, QColor(255, 255, 255, 22))
        glow.setColorAt(0.72, QColor(95, 112, 140, 28))
        glow.setColorAt(1.0, QColor(255, 255, 255, 4))
        painter.setBrush(glow)
        painter.setPen(QPen(QColor(255, 255, 255, 48), 1.2))
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QPen(QColor(190, 205, 230, 96), 1.0))
        for degree in range(0, 360, 45):
            theta = math.radians(degree)
            inner = radius - (8 if degree % 90 == 0 else 4)
            painter.drawLine(
                QPointF(center.x() + math.cos(theta) * inner, center.y() - math.sin(theta) * inner),
                QPointF(center.x() + math.cos(theta) * radius, center.y() - math.sin(theta) * radius),
            )

        theta = math.radians(self.angle)
        tip = QPointF(center.x() + math.cos(theta) * (radius - 8), center.y() - math.sin(theta) * (radius - 8))
        tail = QPointF(center.x() - math.cos(theta) * 10, center.y() + math.sin(theta) * 10)
        painter.setPen(QPen(QColor(255, 255, 255, 214), 4.2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(tail, tip)
        painter.setPen(QPen(qcolor(self.color, 235), 2.2, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(tail, tip)
        painter.setBrush(qcolor(self.color, 255))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(tip, 4.2, 4.2)

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

    def __init__(self, angle_spec: ParameterSpec, speed_spec: ParameterSpec, color: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._updating = False
        self.dial = VelocityDial(color)
        self.angle_control = ParameterControl(angle_spec, "角度", 42)
        self.speed_control = ParameterControl(speed_spec, "大小", 42)

        controls = QVBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self.angle_control)
        controls.addWidget(self.speed_control)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.dial)
        layout.addLayout(controls, 1)

        self.angle_control.changed.connect(self._parameters_changed)
        self.speed_control.changed.connect(self._parameters_changed)
        self.dial.angleChanged.connect(self._dial_changed)
        self.dial.set_angle(angle_spec.default)

    def angle(self) -> float:
        return self.angle_control.value() % 360.0

    def speed(self) -> float:
        return max(0.0, self.speed_control.value())

    def refresh(self) -> None:
        self.dial.set_angle(self.angle())

    def _parameters_changed(self) -> None:
        if self._updating:
            return
        self.dial.set_angle(self.angle())
        self.changed.emit()

    def _dial_changed(self, angle: float) -> None:
        self._updating = True
        self.angle_control.set_value(angle, emit=False)
        self._updating = False
        self.changed.emit()


class TrajectoryWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.positions = np.zeros((3, 3), dtype=float)
        self.tails: List[deque[np.ndarray]] = []
        self.collided_bodies: Set[int] = set()
        self.collision_point: Optional[np.ndarray] = None
        self.flash_frame = 0
        self.flash_total_frames = 0
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_state(
        self,
        positions: np.ndarray,
        tails: List[deque[np.ndarray]],
        collided_bodies: Set[int],
        collision_point: Optional[np.ndarray],
        flash_frame: int,
        flash_total_frames: int,
    ) -> None:
        self.positions = positions.copy()
        self.tails = tails
        self.collided_bodies = set(collided_bodies)
        self.collision_point = None if collision_point is None else collision_point.copy()
        self.flash_frame = flash_frame
        self.flash_total_frames = flash_total_frames
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(APP_BG))

        side = max(1, min(self.width(), self.height()) - 10)
        viewport = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        self._draw_space(painter, viewport)

        points = self._visible_points()
        if not points:
            return
        stacked = np.vstack(points)
        min_xy = stacked.min(axis=0)
        max_xy = stacked.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        span = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 0.5)
        half_range = span * 0.62

        def to_screen(xy: np.ndarray) -> QPointF:
            x = viewport.center().x() + (float(xy[0]) - center[0]) / half_range * viewport.width() / 2
            y = viewport.center().y() - (float(xy[1]) - center[1]) / half_range * viewport.height() / 2
            return QPointF(x, y)

        for index, color in enumerate(BODY_COLORS):
            if index in self.collided_bodies or index >= len(self.tails):
                continue
            self._draw_tail(painter, np.array(self.tails[index])[:, :2], color, to_screen)

        for index, color in enumerate(BODY_COLORS):
            if index in self.collided_bodies:
                continue
            self._draw_star(painter, to_screen(self.positions[index, :2]), color, viewport.width())

        self._draw_flash(painter)

    def _draw_space(self, painter: QPainter, viewport: QRectF) -> None:
        gradient = QLinearGradient(viewport.topLeft(), viewport.bottomRight())
        gradient.setColorAt(0.0, QColor("#080d19"))
        gradient.setColorAt(1.0, QColor("#03050d"))
        path = QPainterPath()
        path.addRoundedRect(viewport, 24, 24)
        painter.fillPath(path, gradient)
        painter.setPen(QPen(QColor(255, 255, 255, 26), 1))
        painter.drawPath(path)

    def _visible_points(self) -> List[np.ndarray]:
        points: List[np.ndarray] = []
        for index, tail in enumerate(self.tails):
            if index in self.collided_bodies or len(tail) == 0:
                continue
            points.append(np.array(tail)[:, :2])
        if self.collision_point is not None:
            points.append(self.collision_point[:2].reshape(1, 2))
        return points

    def _draw_tail(self, painter: QPainter, xy: np.ndarray, color: str, to_screen: Callable[[np.ndarray], QPointF]) -> None:
        if len(xy) < 2:
            return
        total = len(xy) - 1
        for segment_index in range(total):
            progress = segment_index / max(1, total - 1)
            p1 = to_screen(xy[segment_index])
            p2 = to_screen(xy[segment_index + 1])
            painter.setPen(QPen(qcolor(color, int(8 + 42 * progress**1.5)), 0.6 + 2.2 * progress**1.65, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(p1, p2)
            painter.setPen(QPen(qcolor(color, int(10 + 155 * progress**1.35)), 0.08 + 1.05 * progress**1.75, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(p1, p2)

    def _draw_star(self, painter: QPainter, center: QPointF, color: str, viewport_width: float) -> None:
        radius = max(17.0, viewport_width * 0.035)
        gradient = QRadialGradient(center, radius)
        gradient.setColorAt(0.0, QColor(255, 253, 239, 255))
        gradient.setColorAt(0.14, QColor(255, 252, 229, 245))
        gradient.setColorAt(0.35, QColor(255, 206, 104, 155))
        gradient.setColorAt(0.68, qcolor(color, 105))
        gradient.setColorAt(1.0, qcolor(color, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(QColor(255, 253, 242, 255))
        painter.drawEllipse(center, 3.1, 3.1)

    def _draw_flash(self, painter: QPainter) -> None:
        if self.flash_total_frames <= 0 or self.flash_frame >= self.flash_total_frames:
            return
        progress = self.flash_frame / max(1, self.flash_total_frames - 1)
        painter.fillRect(self.rect(), QColor(255, 247, 214, int(158 * (1.0 - progress) ** 2)))


class EnergyPlotWidget(QWidget):
    def __init__(self, title: str, color: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.title = title
        self.color = color
        self.times: List[float] = []
        self.values: List[float] = []
        self.setMinimumSize(420, 320)

    def set_data(self, times: List[float], values: List[float]) -> None:
        self.times = list(times)
        self.values = list(values)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(APP_BG))
        rect = self.rect().adjusted(20, 20, -20, -20)
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 24, 24)
        painter.fillPath(path, QColor(PLOT_BG))
        painter.setPen(QPen(QColor(255, 255, 255, 34), 1))
        painter.drawPath(path)

        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        painter.drawText(rect.adjusted(18, 14, -18, -14), Qt.AlignTop | Qt.AlignLeft, self.title)
        chart = rect.adjusted(50, 58, -24, -40)
        painter.setPen(QPen(QColor(255, 255, 255, 22), 1))
        for i in range(5):
            y = chart.top() + chart.height() * i / 4
            painter.drawLine(chart.left(), y, chart.right(), y)

        if len(self.times) < 2:
            return
        x_min, x_max = self.times[0], self.times[-1]
        y_min, y_max = min(self.values), max(self.values)
        y_span = y_max - y_min
        if y_span <= 1e-12:
            y_span = max(abs(y_max), 1.0) * 0.1
        line = QPainterPath()
        for index, (time_value, energy_value) in enumerate(zip(self.times, self.values)):
            x = chart.left() + (time_value - x_min) / max(1e-9, x_max - x_min) * chart.width()
            y = chart.bottom() - (energy_value - y_min) / y_span * chart.height()
            if index == 0:
                line.moveTo(x, y)
            else:
                line.lineTo(x, y)
        painter.setPen(QPen(qcolor(self.color, 245), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(line)


class TitleBar(QWidget):
    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
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
        self.title = QLabel(f"三体运动模拟器 v{APP_VERSION}")
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
            self._toggle_maximized()

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


class ThreeBodyWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"三体运动模拟器 v{APP_VERSION}")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(1320, 820)
        self.setMinimumSize(1120, 740)

        self.controls: Dict[str, ParameterControl] = {}
        self.velocity_controls: Dict[int, VelocityControl] = {}
        self.paused = False
        self.ended = False
        self.collision_info: Optional[Tuple[int, int, float, float]] = None
        self.collided_bodies: Set[int] = set()
        self.survivor_index: Optional[int] = None
        self.collision_point: Optional[np.ndarray] = None
        self.flash_frame = 0
        self.flash_total_frames = 0
        self.dirty_initial_conditions = False
        self.time = 0.0
        self.positions = np.zeros((3, 3), dtype=float)
        self.velocities = np.zeros((3, 3), dtype=float)
        self.masses = np.ones(3, dtype=float)
        self.tails: List[deque[np.ndarray]] = []
        self.energy_times: List[float] = []
        self.kinetic_history: List[float] = []
        self.potential_history: List[float] = []

        self._apply_theme()
        self._build_ui()
        self._reset_simulation(clear_dirty=True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(FRAME_MS)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: transparent;
            }}
            QWidget {{
                color: {TEXT};
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 10pt;
            }}
            QWidget#windowShell {{
                background: {APP_BG};
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 24px;
            }}
            QLabel#title {{
                font-size: 18pt;
                font-weight: 700;
            }}
            QLabel#muted {{
                color: {MUTED};
            }}
            QLabel#section {{
                color: {ACCENT};
                font-size: 9pt;
                font-weight: 700;
            }}
            QLabel#status {{
                background: rgba(24, 36, 58, 0.72);
                color: {ACCENT};
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 10px 12px;
            }}
            QPushButton {{
                background: rgba(38, 53, 82, 0.86);
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 13px;
                padding: 9px 12px;
            }}
            QPushButton:hover {{
                background: rgba(52, 72, 108, 0.94);
            }}
            QPushButton:pressed {{
                background: rgba(27, 39, 62, 0.94);
            }}
            QPushButton#accent {{
                background: rgba(37, 168, 156, 0.90);
                color: #ecfffb;
            }}
            QPushButton:disabled {{
                color: rgba(238,244,255,0.42);
                background: rgba(35, 43, 60, 0.55);
            }}
            QWidget#titleBar {{
                background: rgba(12, 18, 30, 0.86);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 16px;
            }}
            QLabel#windowTitle {{
                color: rgba(238,244,255,0.88);
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
                background: rgba(255, 87, 102, 0.86);
                color: white;
            }}
            QLineEdit {{
                background: rgba(8, 14, 27, 0.72);
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 9px;
                padding: 5px 7px;
                selection-background-color: rgba(94,234,212,0.42);
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
                border: 1px solid rgba(255,255,255,0.65);
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
                border: 0;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(255, 255, 255, 0.20);
                border-radius: 4px;
                min-height: 34px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(255, 255, 255, 0.32);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
                border: 0;
                background: transparent;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QMenu {{
                background: rgba(20, 30, 48, 0.96);
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 12px;
                padding: 7px;
                color: {TEXT};
            }}
            QMenu::item {{
                padding: 8px 24px 8px 12px;
                border-radius: 8px;
            }}
            QMenu::item:selected {{
                background: rgba(94, 234, 212, 0.18);
                color: #f8fffd;
            }}
            QTabWidget::pane {{
                border: 0;
                background: transparent;
            }}
            QTabBar::tab {{
                background: rgba(24, 35, 56, 0.74);
                color: {MUTED};
                padding: 10px 18px;
                margin-right: 6px;
                border-radius: 12px;
            }}
            QTabBar::tab:selected {{
                background: rgba(43, 61, 92, 0.90);
                color: {TEXT};
            }}
            """
        )

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("windowShell")
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        self.setCentralWidget(central)

        self.title_bar = TitleBar(self)
        root.addWidget(self.title_bar)

        body = QHBoxLayout()
        body.setContentsMargins(6, 0, 6, 6)
        body.setSpacing(16)
        root.addLayout(body, 1)

        left_panel = GlassFrame(radius=28, padding=16)
        left_panel.setMinimumWidth(450)
        left_panel.setMaximumWidth(510)
        body.addWidget(left_panel, 0)

        title = QLabel("三体观测台")
        title.setObjectName("title")
        note = QLabel("设置二维初始条件，观察三颗恒星的轨迹与系统能量变化。")
        note.setObjectName("muted")
        note.setWordWrap(True)
        left_panel.layout.addWidget(title)
        left_panel.layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.pause_button = QPushButton("暂停")
        self.apply_button = QPushButton("应用 / 重启")
        self.apply_button.setObjectName("accent")
        self.preset_button = QPushButton("轨道预设 ▾")
        self.preset_button.setMenu(self._build_preset_menu())
        self.random_button = QPushButton("随机参数")
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.preset_button)
        buttons.addWidget(self.random_button)
        left_panel.layout.addLayout(buttons)

        self.status_label = QLabel("运行中")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        left_panel.layout.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls_holder = QWidget()
        controls_layout = QVBoxLayout(controls_holder)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(12)
        scroll.setWidget(controls_holder)
        left_panel.layout.addWidget(scroll, 1)

        self._build_controls(controls_layout)

        self.pause_button.clicked.connect(self._toggle_pause)
        self.apply_button.clicked.connect(lambda: self._reset_simulation(clear_dirty=True))
        self.random_button.clicked.connect(self._randomize_parameters)

        self.tabs = QTabWidget()
        body.addWidget(self.tabs, 1)
        self.trajectory_widget = TrajectoryWidget()
        self.kinetic_widget = EnergyPlotWidget("总动能", "#ff8a4c")
        self.potential_widget = EnergyPlotWidget("总势能", "#72ddf7")
        self.tabs.addTab(self.trajectory_widget, "运动轨迹")
        self.tabs.addTab(self.kinetic_widget, "总动能")
        self.tabs.addTab(self.potential_widget, "总势能")

    def _build_preset_menu(self) -> QMenu:
        menu = QMenu(self)
        presets = [
            ("经典 8 字轨道", "figure_eight"),
            ("拉格朗日三角", "lagrange_triangle"),
            ("欧拉共线旋转", "euler_collinear"),
            ("混沌飞掠", "chaotic_flyby"),
            ("紧密双星 + 闯入者", "binary_intruder"),
            ("弹弓甩射", "slingshot"),
            ("慢速三角追逐", "slow_triangle"),
            ("高速三角旋涡", "fast_triangle"),
            ("不等质量玫瑰", "unequal_rose"),
            ("蝴蝶摆动", "butterfly"),
            ("远距伴星扰动", "distant_companion"),
            ("反向剪切", "counter_shear"),
            ("重星扰动", "heavy_perturber"),
            ("双星漂移", "drifting_binary"),
            ("近碰掠过", "near_miss"),
        ]
        for label, key in presets:
            action = menu.addAction(label)
            action.triggered.connect(lambda _checked=False, preset_key=key: self._apply_preset(preset_key))
        return menu

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 2, 0)
        grip_row.addStretch(1)
        grip_row.addWidget(QSizeGrip(self), 0, Qt.AlignRight)
        root.addLayout(grip_row)

    def _build_controls(self, layout: QVBoxLayout) -> None:
        specs = {spec.key: spec for spec in PARAMETER_SPECS}
        for body_index in range(3):
            card = GlassFrame(radius=22, padding=13)
            layout.addWidget(card)
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(7)
            card.layout.addLayout(grid)

            header = QLabel(BODY_NAMES[body_index])
            header.setStyleSheet("font-size: 12pt; font-weight: 700;")
            position_label = QLabel("初始位置")
            position_label.setObjectName("section")
            mass_label = QLabel("质量")
            mass_label.setObjectName("section")
            velocity_label = QLabel("初速度方向 / 大小")
            velocity_label.setObjectName("section")
            grid.addWidget(header, 0, 0, 1, 2)
            grid.addWidget(position_label, 1, 0)
            grid.addWidget(mass_label, 1, 1)

            for row, axis in enumerate(("x", "y"), start=2):
                key = f"p{body_index}_{axis}"
                control = ParameterControl(specs[key], axis, 24)
                control.changed.connect(self._mark_dirty)
                self.controls[key] = control
                grid.addWidget(control, row, 0)

            mass = ParameterControl(specs[f"m{body_index}"], "质量", 42)
            mass.changed.connect(self._mark_dirty)
            self.controls[f"m{body_index}"] = mass
            grid.addWidget(mass, 2, 1, 2, 1)

            velocity = VelocityControl(specs[f"a{body_index}"], specs[f"s{body_index}"], BODY_COLORS[body_index])
            velocity.changed.connect(self._mark_dirty)
            self.velocity_controls[body_index] = velocity
            self.controls[f"a{body_index}"] = velocity.angle_control
            self.controls[f"s{body_index}"] = velocity.speed_control
            grid.addWidget(velocity_label, 4, 0, 1, 2)
            grid.addWidget(velocity, 5, 0, 1, 2)

        simulation = GlassFrame(radius=22, padding=13)
        layout.addWidget(simulation)
        simulation.layout.addWidget(QLabel("模拟控制"))
        labels = {"dt": "步长", "steps": "每帧", "tail": "尾迹", "collision": "碰撞"}
        for key in ("dt", "steps", "tail", "collision"):
            control = ParameterControl(PARAMETER_SPECS[[spec.key for spec in PARAMETER_SPECS].index(key)], labels[key], 42)
            control.changed.connect(self._simulation_setting_changed)
            self.controls[key] = control
            simulation.layout.addWidget(control)
        layout.addStretch(1)

    def _read_initial_conditions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.zeros((3, 3), dtype=float)
        velocities = np.zeros((3, 3), dtype=float)
        masses = np.zeros(3, dtype=float)
        for body_index in range(3):
            positions[body_index, 0] = self.controls[f"p{body_index}_x"].value()
            positions[body_index, 1] = self.controls[f"p{body_index}_y"].value()
            angle = math.radians(self.velocity_controls[body_index].angle())
            speed = self.velocity_controls[body_index].speed()
            velocities[body_index, 0] = speed * math.cos(angle)
            velocities[body_index, 1] = speed * math.sin(angle)
            masses[body_index] = self.controls[f"m{body_index}"].value()
            if masses[body_index] <= 0:
                raise ValueError(f"{BODY_NAMES[body_index]} 的质量必须为正数。")
        return positions, velocities, masses

    def _reset_simulation(self, clear_dirty: bool) -> None:
        try:
            self.positions, self.velocities, self.masses = self._read_initial_conditions()
        except ValueError as exc:
            QMessageBox.warning(self, "初始条件无效", str(exc))
            return

        self.ended = False
        self.paused = False
        self.collision_info = None
        self.collided_bodies.clear()
        self.survivor_index = None
        self.collision_point = None
        self.flash_frame = 0
        self.flash_total_frames = 0
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self.time = 0.0
        self.tails = [deque(maxlen=self._tail_length()) for _ in range(3)]
        for index in range(3):
            self.tails[index].append(self.positions[index].copy())

        self.energy_times.clear()
        self.kinetic_history.clear()
        self.potential_history.clear()
        self._record_energy()
        if clear_dirty:
            self.dirty_initial_conditions = False
        self._check_for_collision()
        self._update_status()
        self._update_views()

    def _restore_defaults(self) -> None:
        self._apply_preset("figure_eight")

    def _apply_preset(self, preset_key: str) -> None:
        positions, velocities, masses = self._preset_state(preset_key)
        self._set_controls_from_state(positions, velocities, masses)
        self._reset_simulation(clear_dirty=True)

    def _preset_state(self, preset_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if preset_key == "figure_eight":
            return DEFAULT_POSITIONS.copy(), DEFAULT_VELOCITIES.copy(), np.ones(3, dtype=float)

        if preset_key == "lagrange_triangle":
            radius = 1.0
            omega = math.sqrt(1.0 / math.sqrt(3.0))
            positions = np.zeros((3, 3), dtype=float)
            velocities = np.zeros((3, 3), dtype=float)
            for index, theta in enumerate((math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6)):
                positions[index, 0] = radius * math.cos(theta)
                positions[index, 1] = radius * math.sin(theta)
                velocities[index, 0] = -omega * radius * math.sin(theta)
                velocities[index, 1] = omega * radius * math.cos(theta)
            return positions, velocities, np.ones(3, dtype=float)

        if preset_key == "euler_collinear":
            omega = math.sqrt(1.25)
            positions = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
            velocities = np.array([[0.0, -omega, 0.0], [0.0, 0.0, 0.0], [0.0, omega, 0.0]], dtype=float)
            return positions, velocities, np.ones(3, dtype=float)

        if preset_key == "chaotic_flyby":
            positions = np.array([[-0.62, -0.08, 0.0], [0.62, 0.08, 0.0], [-1.72, 1.42, 0.0]], dtype=float)
            velocities = np.array([[0.06, -0.84, 0.0], [-0.05, 0.86, 0.0], [1.04, -0.52, 0.0]], dtype=float)
            masses = np.array([1.15, 0.95, 0.5], dtype=float)
            return positions, velocities, masses

        if preset_key == "binary_intruder":
            positions = np.array([[-0.34, 0.0, 0.0], [0.34, 0.0, 0.0], [-2.3, 0.75, 0.0]], dtype=float)
            velocities = np.array([[0.0, -1.08, 0.0], [0.0, 1.08, 0.0], [1.18, -0.28, 0.0]], dtype=float)
            masses = np.array([1.0, 1.0, 0.42], dtype=float)
            return positions, velocities, masses

        if preset_key == "slingshot":
            positions = np.array([[-0.48, -0.05, 0.0], [0.48, 0.05, 0.0], [-2.2, -1.05, 0.0]], dtype=float)
            velocities = np.array([[0.12, -0.92, 0.0], [-0.12, 0.92, 0.0], [1.33, 0.62, 0.0]], dtype=float)
            masses = np.array([1.2, 1.0, 0.32], dtype=float)
            return positions, velocities, masses

        if preset_key == "slow_triangle":
            positions = np.array([[0.0, 1.15, 0.0], [-0.996, -0.575, 0.0], [0.996, -0.575, 0.0]], dtype=float)
            velocities = np.array([[-0.58, 0.0, 0.0], [0.29, -0.50, 0.0], [0.29, 0.50, 0.0]], dtype=float)
            masses = np.ones(3, dtype=float)
            return positions, velocities, masses

        if preset_key == "fast_triangle":
            positions = np.array([[0.0, 0.92, 0.0], [-0.797, -0.46, 0.0], [0.797, -0.46, 0.0]], dtype=float)
            velocities = np.array([[-1.05, 0.0, 0.0], [0.525, -0.91, 0.0], [0.525, 0.91, 0.0]], dtype=float)
            masses = np.ones(3, dtype=float)
            return positions, velocities, masses

        if preset_key == "unequal_rose":
            positions = np.array([[-0.95, 0.25, 0.0], [0.82, -0.18, 0.0], [0.12, 0.76, 0.0]], dtype=float)
            velocities = np.array([[0.38, 0.60, 0.0], [0.28, -0.73, 0.0], [-0.91, -0.08, 0.0]], dtype=float)
            masses = np.array([1.55, 0.82, 1.12], dtype=float)
            return positions, velocities, masses

        if preset_key == "butterfly":
            positions = np.array([[-0.78, 0.0, 0.0], [0.78, 0.0, 0.0], [0.0, 0.42, 0.0]], dtype=float)
            velocities = np.array([[0.24, 0.68, 0.0], [0.24, 0.68, 0.0], [-0.48, -1.36, 0.0]], dtype=float)
            masses = np.ones(3, dtype=float)
            return positions, velocities, masses

        if preset_key == "distant_companion":
            positions = np.array([[-0.42, 0.0, 0.0], [0.42, 0.0, 0.0], [2.55, 0.18, 0.0]], dtype=float)
            velocities = np.array([[0.0, -1.02, 0.0], [0.0, 1.02, 0.0], [-0.26, 0.58, 0.0]], dtype=float)
            masses = np.array([1.0, 1.0, 0.72], dtype=float)
            return positions, velocities, masses

        if preset_key == "counter_shear":
            positions = np.array([[-1.2, -0.42, 0.0], [1.2, 0.42, 0.0], [0.0, 0.0, 0.0]], dtype=float)
            velocities = np.array([[0.85, 0.28, 0.0], [-0.85, -0.28, 0.0], [0.0, 0.0, 0.0]], dtype=float)
            masses = np.array([0.9, 0.9, 1.45], dtype=float)
            return positions, velocities, masses

        if preset_key == "heavy_perturber":
            positions = np.array([[-0.72, 0.0, 0.0], [0.72, 0.0, 0.0], [0.0, 1.75, 0.0]], dtype=float)
            velocities = np.array([[0.0, -0.82, 0.0], [0.0, 0.82, 0.0], [-0.66, 0.0, 0.0]], dtype=float)
            masses = np.array([0.72, 0.72, 2.25], dtype=float)
            return positions, velocities, masses

        if preset_key == "drifting_binary":
            positions = np.array([[-0.28, -0.08, 0.0], [0.28, 0.08, 0.0], [1.75, -1.15, 0.0]], dtype=float)
            velocities = np.array([[0.28, -1.18, 0.0], [0.28, 1.18, 0.0], [-0.58, 0.24, 0.0]], dtype=float)
            masses = np.array([0.92, 0.92, 0.6], dtype=float)
            return positions, velocities, masses

        if preset_key == "near_miss":
            positions = np.array([[-1.35, 0.03, 0.0], [1.35, -0.03, 0.0], [0.0, 1.32, 0.0]], dtype=float)
            velocities = np.array([[0.78, 0.08, 0.0], [-0.78, -0.08, 0.0], [0.06, -0.72, 0.0]], dtype=float)
            masses = np.array([1.0, 1.0, 0.86], dtype=float)
            return positions, velocities, masses

        return DEFAULT_POSITIONS.copy(), DEFAULT_VELOCITIES.copy(), np.ones(3, dtype=float)

    def _set_controls_from_state(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray) -> None:
        for body_index in range(3):
            self.controls[f"p{body_index}_x"].set_value(float(positions[body_index, 0]), emit=False)
            self.controls[f"p{body_index}_y"].set_value(float(positions[body_index, 1]), emit=False)
            vx = float(velocities[body_index, 0])
            vy = float(velocities[body_index, 1])
            angle = math.degrees(math.atan2(vy, vx)) % 360.0 if abs(vx) + abs(vy) > 1e-12 else 0.0
            speed = math.hypot(vx, vy)
            self.controls[f"a{body_index}"].set_value(angle, emit=False)
            self.controls[f"s{body_index}"].set_value(speed, emit=False)
            self.controls[f"m{body_index}"].set_value(float(masses[body_index]), emit=False)
            self.velocity_controls[body_index].refresh()

    def _randomize_parameters(self) -> None:
        center_x = random.uniform(-0.25, 0.25)
        center_y = random.uniform(-0.25, 0.25)
        base_radius = random.uniform(0.75, 1.55)
        phase = random.uniform(0.0, 2.0 * math.pi)
        spin = random.choice((-1.0, 1.0))

        for body_index in range(3):
            theta = phase + body_index * 2.0 * math.pi / 3.0 + random.uniform(-0.38, 0.38)
            radius = base_radius * random.uniform(0.72, 1.28)
            x = center_x + math.cos(theta) * radius
            y = center_y + math.sin(theta) * radius

            tangent_angle = (math.degrees(theta) + spin * 90.0 + random.uniform(-34.0, 34.0)) % 360.0
            speed = random.uniform(0.42, 1.45)
            mass = random.uniform(0.65, 1.75)

            self.controls[f"p{body_index}_x"].set_value(x, emit=False)
            self.controls[f"p{body_index}_y"].set_value(y, emit=False)
            self.controls[f"a{body_index}"].set_value(tangent_angle, emit=False)
            self.controls[f"s{body_index}"].set_value(speed, emit=False)
            self.controls[f"m{body_index}"].set_value(mass, emit=False)
            self.velocity_controls[body_index].refresh()

        self._reset_simulation(clear_dirty=True)

    def _toggle_pause(self) -> None:
        if self.ended:
            return
        self.paused = not self.paused
        self.pause_button.setText("继续" if self.paused else "暂停")
        self._update_status()

    def _mark_dirty(self) -> None:
        self.dirty_initial_conditions = True
        self._update_status()

    def _simulation_setting_changed(self) -> None:
        new_length = self._tail_length()
        if self.tails:
            self.tails = [deque(tail, maxlen=new_length) for tail in self.tails]
        self._update_status()

    def _update_status(self) -> None:
        status = "碰撞结束" if self.ended else ("已暂停" if self.paused else "运行中")
        if self.dirty_initial_conditions:
            status += " - 初始值已修改，请点击“应用 / 重启”。"
        self.status_label.setText(status)

    def _tail_length(self) -> int:
        return max(2, int(round(self.controls["tail"].value())))

    def _steps_per_frame(self) -> int:
        return max(1, int(round(self.controls["steps"].value())))

    def _time_step(self) -> float:
        return max(1e-6, self.controls["dt"].value())

    def _collision_radius(self) -> float:
        return max(1e-6, self.controls["collision"].value())

    def _tick(self) -> None:
        if not self.paused and not self.ended:
            for _ in range(self._steps_per_frame()):
                self._integrate_one_step(self._time_step())
                if self.ended:
                    break
        elif self.ended and self.survivor_index is not None:
            self._advance_survivor_after_collision()
            if self.flash_frame < self.flash_total_frames:
                self.flash_frame += 1
        self._update_views()

    def _advance_survivor_after_collision(self) -> None:
        survivor = self.survivor_index
        if survivor is None:
            return
        drift_dt = self._time_step() * self._steps_per_frame() * 0.65
        self.positions[survivor] = self.positions[survivor] + self.velocities[survivor] * drift_dt
        self.tails[survivor].append(self.positions[survivor].copy())

    def _integrate_one_step(self, dt: float) -> None:
        acceleration = self._accelerations(self.positions)
        new_positions = self.positions + self.velocities * dt + 0.5 * acceleration * dt * dt
        new_acceleration = self._accelerations(new_positions)
        self.velocities = self.velocities + 0.5 * (acceleration + new_acceleration) * dt
        self.positions = new_positions
        self.time += dt
        for index in range(3):
            self.tails[index].append(self.positions[index].copy())
        self._record_energy()
        self._check_for_collision()

    def _check_for_collision(self) -> bool:
        closest_pair = None
        closest_distance = float("inf")
        for i in range(3):
            for j in range(i + 1, 3):
                distance = float(np.linalg.norm(self.positions[j] - self.positions[i]))
                if distance < closest_distance:
                    closest_pair = (i, j)
                    closest_distance = distance
        if closest_pair is None or closest_distance > self._collision_radius():
            return False

        first, second = closest_pair
        self.ended = True
        self.paused = True
        self.collision_info = (first, second, closest_distance, self.time)
        self.collided_bodies = {first, second}
        survivors = [index for index in range(3) if index not in self.collided_bodies]
        self.survivor_index = survivors[0] if survivors else None
        self.collision_point = (self.positions[first] + self.positions[second]) / 2.0
        self.flash_frame = 0
        self.flash_total_frames = 3
        self.pause_button.setText("已结束")
        self.pause_button.setEnabled(False)
        self._update_status()
        return True

    def _accelerations(self, positions: np.ndarray) -> np.ndarray:
        accelerations = np.zeros_like(positions)
        softening_squared = SOFTENING * SOFTENING
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                displacement = positions[j] - positions[i]
                distance_squared = float(np.dot(displacement, displacement) + softening_squared)
                accelerations[i] += G * self.masses[j] * displacement / (distance_squared ** 1.5)
        return accelerations

    def _energies(self) -> Tuple[float, float]:
        kinetic = 0.5 * float(np.sum(self.masses[:, None] * self.velocities * self.velocities))
        potential = 0.0
        softening_squared = SOFTENING * SOFTENING
        for i in range(3):
            for j in range(i + 1, 3):
                displacement = self.positions[j] - self.positions[i]
                distance = math.sqrt(float(np.dot(displacement, displacement) + softening_squared))
                potential -= G * self.masses[i] * self.masses[j] / distance
        return kinetic, potential

    def _record_energy(self) -> None:
        kinetic, potential = self._energies()
        self.energy_times.append(self.time)
        self.kinetic_history.append(kinetic)
        self.potential_history.append(potential)
        if len(self.energy_times) > MAX_ENERGY_POINTS:
            self.energy_times = self.energy_times[-MAX_ENERGY_POINTS:]
            self.kinetic_history = self.kinetic_history[-MAX_ENERGY_POINTS:]
            self.potential_history = self.potential_history[-MAX_ENERGY_POINTS:]

    def _update_views(self) -> None:
        self.trajectory_widget.set_state(
            self.positions,
            self.tails,
            self.collided_bodies,
            self.collision_point,
            self.flash_frame,
            self.flash_total_frames,
        )
        self.kinetic_widget.set_data(self.energy_times, self.kinetic_history)
        self.potential_widget.set_data(self.energy_times, self.potential_history)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ThreeBodyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
