"""Interactive two-dimensional three-body simulator."""

from __future__ import annotations

import math
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Optional, Tuple

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


G = 1.0
SOFTENING = 0.02
FRAME_MS = 30
MAX_ENERGY_POINTS = 2500

APP_BG = "#070b14"
PANEL_BG = "#0d1422"
CARD_BG = "#172236"
CARD_BORDER = "#40506b"
GLASS_PANEL = "#111b2d"
GLASS_CARD = "#18243a"
GLASS_HIGHLIGHT = "#6c7f9d"
SHADOW = "#01030a"
SPACE_BG = "#050914"
PLOT_BG = "#0a1020"
TEXT = "#e8eefc"
MUTED = "#91a1ba"
ACCENT = "#5eead4"
ACCENT_DARK = "#1f8f84"
WARNING = "#f0b429"


rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    resolution: float
    positive: bool = False


BODY_COLORS = ("#e84d5b", "#24a39a", "#f0b429")
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
DEFAULT_MASSES = np.array([1.0, 1.0, 1.0], dtype=float)


def build_parameter_specs() -> list[ParameterSpec]:
    specs: list[ParameterSpec] = []
    axes = ("x", "y")

    for body_index in range(3):
        for axis_index, axis in enumerate(axes):
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

        vx = float(DEFAULT_VELOCITIES[body_index, 0])
        vy = float(DEFAULT_VELOCITIES[body_index, 1])
        default_speed = math.hypot(vx, vy)
        default_angle = math.degrees(math.atan2(vy, vx)) % 360.0
        specs.append(
            ParameterSpec(
                key=f"a{body_index}",
                label=f"{BODY_NAMES[body_index]} 速度角度",
                default=default_angle,
                minimum=0.0,
                maximum=360.0,
                resolution=1.0,
            )
        )
        specs.append(
            ParameterSpec(
                key=f"s{body_index}",
                label=f"{BODY_NAMES[body_index]} 速度大小",
                default=default_speed,
                minimum=0.0,
                maximum=4.0,
                resolution=0.01,
            )
        )

        specs.append(
            ParameterSpec(
                key=f"m{body_index}",
                label=f"{BODY_NAMES[body_index]} 质量",
                default=float(DEFAULT_MASSES[body_index]),
                minimum=0.05,
                maximum=10.0,
                resolution=0.05,
                positive=True,
            )
        )

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


def color_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16) / 255.0
    green = int(hex_color[2:4], 16) / 255.0
    blue = int(hex_color[4:6], 16) / 255.0
    return red, green, blue, alpha


def make_star_glow_image(hex_color: str, size: int = 128) -> np.ndarray:
    edge_color = np.array(color_to_rgba(hex_color, 1.0)[:3])
    warm_color = np.array([1.0, 0.78, 0.34])
    core_color = np.array([1.0, 0.995, 0.94])

    axis = np.linspace(-1.0, 1.0, size)
    x_grid, y_grid = np.meshgrid(axis, axis)
    radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)

    outer = np.exp(-((radius / 0.72) ** 2)) * 0.28
    middle = np.exp(-((radius / 0.36) ** 2)) * 0.42
    core = np.exp(-((radius / 0.12) ** 2)) * 1.15
    total = outer + middle + core + 1e-9

    rgb = (
        edge_color[None, None, :] * outer[:, :, None]
        + warm_color[None, None, :] * middle[:, :, None]
        + core_color[None, None, :] * core[:, :, None]
    ) / total[:, :, None]

    edge_fade = np.clip((1.0 - radius) / 0.22, 0.0, 1.0)
    alpha = np.clip(outer * 0.45 + middle * 0.55 + core * 0.95, 0.0, 0.98) * edge_fade

    image = np.zeros((size, size, 4), dtype=float)
    image[:, :, :3] = rgb
    image[:, :, 3] = alpha
    return image


class ParameterControl(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        spec: ParameterSpec,
        on_change: callable,
        label_text: str = "",
        label_width: int = 7,
    ) -> None:
        super().__init__(master, style="Control.TFrame")
        self.spec = spec
        self.on_change = on_change
        self.minimum = spec.minimum
        self.maximum = spec.maximum
        self._updating = False

        self.var = tk.DoubleVar(value=spec.default)
        self.label = ttk.Label(
            self,
            text=label_text or spec.label,
            width=label_width,
            anchor="w",
            style="Control.TLabel",
        )
        self.scale = ttk.Scale(
            self,
            from_=self.minimum,
            to=self.maximum,
            variable=self.var,
            command=self._scale_changed,
            style="Dark.Horizontal.TScale",
        )
        self.entry = ttk.Entry(self, width=8, style="Dark.TEntry")
        self.entry.insert(0, self._format_value(spec.default))

        self.label.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.scale.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.entry.grid(row=0, column=2, sticky="e")
        self.columnconfigure(1, weight=1)

        self.entry.bind("<Return>", self._entry_committed)
        self.entry.bind("<FocusOut>", self._entry_committed)

    def value(self) -> float:
        return float(self.var.get())

    def set_value(self, value: float, notify: bool = True) -> None:
        if self.spec.positive and value <= 0:
            value = self.spec.minimum
        self._expand_scale_if_needed(value)
        self._updating = True
        self.var.set(value)
        self._set_entry_text(value)
        self._updating = False
        if notify:
            self.on_change()

    def _format_value(self, value: float) -> str:
        if self.spec.resolution >= 1:
            return str(int(round(value)))
        if self.spec.resolution >= 0.01:
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return f"{value:.5f}".rstrip("0").rstrip(".")

    def _set_entry_text(self, value: float) -> None:
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self._format_value(value))

    def _expand_scale_if_needed(self, value: float) -> None:
        changed = False
        if value < self.minimum:
            self.minimum = value
            changed = True
        if value > self.maximum:
            self.maximum = value
            changed = True
        if changed:
            self.scale.configure(from_=self.minimum, to=self.maximum)

    def _scale_changed(self, raw_value: str) -> None:
        if self._updating:
            return
        value = float(raw_value)
        if self.spec.resolution >= 1:
            value = round(value)
        else:
            value = round(value / self.spec.resolution) * self.spec.resolution
        self._updating = True
        self.var.set(value)
        self._set_entry_text(value)
        self._updating = False
        self.on_change()

    def _entry_committed(self, _event: tk.Event) -> None:
        if self._updating:
            return
        raw_text = self.entry.get().strip()
        try:
            value = float(raw_text)
        except ValueError:
            self._set_entry_text(self.value())
            return

        if self.spec.positive and value <= 0:
            messagebox.showerror("无效数值", f"{self.spec.label} 必须为正数。")
            self._set_entry_text(self.value())
            return
        self.set_value(value)


class VelocityControl(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        angle_spec: ParameterSpec,
        speed_spec: ParameterSpec,
        color: str,
        on_change: callable,
    ) -> None:
        super().__init__(master, style="Control.TFrame")
        self.color = color
        self.on_change = on_change
        self._updating = False
        self.dial_size = 108
        self.center = self.dial_size / 2
        self.radius = 42

        self.columnconfigure(1, weight=1)
        self.dial = tk.Canvas(
            self,
            width=self.dial_size,
            height=self.dial_size,
            background=GLASS_CARD,
            highlightthickness=0,
        )
        self.dial.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 10))
        self.dial.bind("<Button-1>", self._dial_changed)
        self.dial.bind("<B1-Motion>", self._dial_changed)

        self.angle_control = ParameterControl(
            self,
            angle_spec,
            self._parameter_changed,
            label_text="角度",
            label_width=4,
        )
        self.speed_control = ParameterControl(
            self,
            speed_spec,
            self._parameter_changed,
            label_text="大小",
            label_width=4,
        )
        self.angle_control.grid(row=0, column=1, sticky="ew", pady=(4, 3))
        self.speed_control.grid(row=1, column=1, sticky="ew", pady=(3, 4))
        self._draw_dial()

    def angle(self) -> float:
        return float(self.angle_control.value()) % 360.0

    def speed(self) -> float:
        return max(0.0, float(self.speed_control.value()))

    def refresh(self) -> None:
        self.angle_control.set_value(self.angle(), notify=False)
        self.speed_control.set_value(self.speed(), notify=False)
        self._draw_dial()

    def _parameter_changed(self) -> None:
        if self._updating:
            return
        self._updating = True
        self.angle_control.set_value(self.angle(), notify=False)
        self.speed_control.set_value(self.speed(), notify=False)
        self._updating = False
        self._draw_dial()
        self.on_change()

    def _dial_changed(self, event: tk.Event) -> None:
        dx = event.x - self.center
        dy = self.center - event.y
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        self._updating = True
        self.angle_control.set_value(angle, notify=False)
        self._updating = False
        self._draw_dial()
        self.on_change()

    def _draw_dial(self) -> None:
        self.dial.delete("all")
        c = self.center
        r = self.radius
        self.dial.create_oval(c - r, c - r, c + r, c + r, outline="#31415f", width=2)
        self.dial.create_oval(c - 4, c - 4, c + 4, c + 4, fill="#dffcff", outline="")

        for degree in range(0, 360, 45):
            theta = math.radians(degree)
            inner = r - (7 if degree % 90 == 0 else 4)
            outer = r
            x1 = c + math.cos(theta) * inner
            y1 = c - math.sin(theta) * inner
            x2 = c + math.cos(theta) * outer
            y2 = c - math.sin(theta) * outer
            self.dial.create_line(x1, y1, x2, y2, fill="#53647e", width=1)

        theta = math.radians(self.angle())
        pointer_length = r - 9
        tip_x = c + math.cos(theta) * pointer_length
        tip_y = c - math.sin(theta) * pointer_length
        tail_x = c - math.cos(theta) * 10
        tail_y = c + math.sin(theta) * 10
        self.dial.create_line(tail_x, tail_y, tip_x, tip_y, fill="#fffdf2", width=4, capstyle=tk.ROUND)
        self.dial.create_line(tail_x, tail_y, tip_x, tip_y, fill=self.color, width=2, capstyle=tk.ROUND)
        self.dial.create_oval(tip_x - 4, tip_y - 4, tip_x + 4, tip_y + 4, fill=self.color, outline="")


class GlassFrame(tk.Canvas):
    def __init__(
        self,
        master: tk.Widget,
        background: str,
        glass_color: str,
        content_style: str,
        radius: int = 24,
        padding: int = 14,
        shadow: bool = True,
        auto_height: bool = True,
    ) -> None:
        super().__init__(master, background=background, highlightthickness=0, borderwidth=0)
        self.glass_color = glass_color
        self.radius = radius
        self.padding = padding
        self.shadow = shadow
        self.auto_height = auto_height
        self.shadow_offset = 10 if shadow else 0
        self.content = ttk.Frame(self, style=content_style)
        self.window_id = self.create_window((0, 0), window=self.content, anchor="nw")

        self.bind("<Configure>", self._redraw)
        self.content.bind("<Configure>", self._content_changed)

    def _content_changed(self, _event: tk.Event) -> None:
        if self.auto_height:
            desired_height = self.content.winfo_reqheight() + self.padding * 2 + self.shadow_offset + 8
            self.configure(height=desired_height)
        self.after_idle(self._redraw)

    def _redraw(self, _event: Optional[tk.Event] = None) -> None:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        shadow_pad = self.shadow_offset
        x0 = 4
        y0 = 4
        x1 = max(x0 + 2, width - shadow_pad - 2)
        y1 = max(y0 + 2, height - shadow_pad - 2)

        self.delete("glass")
        if self.shadow:
            self._rounded_rect(x0 + 8, y0 + 9, x1 + 8, y1 + 9, self.radius, SHADOW, "", "glass")
            self._rounded_rect(x0 + 5, y0 + 5, x1 + 5, y1 + 5, self.radius, "#030815", "", "glass")

        self._rounded_rect(x0, y0, x1, y1, self.radius, self.glass_color, CARD_BORDER, "glass", width=1)
        self._rounded_rect(
            x0 + 1,
            y0 + 1,
            x1 - 1,
            y0 + max(18, self.radius + 4),
            self.radius - 2,
            "#22304a",
            "",
            "glass",
        )
        self._rounded_rect(
            x0 + 2,
            y0 + 2,
            x1 - 2,
            y1 - 2,
            max(1, self.radius - 2),
            "",
            GLASS_HIGHLIGHT,
            "glass",
            width=1,
        )
        self.tag_lower("glass")

        content_x = x0 + self.padding
        content_y = y0 + self.padding
        content_width = max(1, x1 - x0 - self.padding * 2)
        if self.auto_height:
            content_height = max(1, self.content.winfo_reqheight())
        else:
            content_height = max(1, y1 - y0 - self.padding * 2)
        self.coords(self.window_id, content_x, content_y)
        self.itemconfigure(self.window_id, width=content_width, height=content_height)

    def _rounded_rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        radius: int,
        fill: str,
        outline: str,
        tag: str,
        width: int = 1,
    ) -> None:
        radius = min(radius, int((x1 - x0) / 2), int((y1 - y0) / 2))
        points = [
            x0 + radius,
            y0,
            x1 - radius,
            y0,
            x1,
            y0,
            x1,
            y0 + radius,
            x1,
            y1 - radius,
            x1,
            y1,
            x1 - radius,
            y1,
            x0 + radius,
            y1,
            x0,
            y1,
            x0,
            y1 - radius,
            x0,
            y0 + radius,
            x0,
            y0,
        ]
        self.create_polygon(
            points,
            smooth=True,
            splinesteps=20,
            fill=fill,
            outline=outline,
            width=width,
            tags=tag,
        )


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master, style="Panel.TFrame")
        self.canvas = tk.Canvas(self, highlightthickness=0, background=GLASS_PANEL)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Panel.TFrame")
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if self.winfo_containing(event.x_root, event.y_root) is None:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class ThreeBodySimulator(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("三体运动模拟器")
        self.geometry("1320x820")
        self.minsize(1120, 740)
        self.configure(background=APP_BG)

        self.controls: dict[str, ParameterControl] = {}
        self.velocity_controls: dict[int, VelocityControl] = {}
        self.paused = False
        self.ended = False
        self.collision_info: Optional[Tuple[int, int, float, float]] = None
        self.collided_bodies: set[int] = set()
        self.survivor_index: Optional[int] = None
        self.collision_point: Optional[np.ndarray] = None
        self.flash_frame = 0
        self.flash_total_frames = 0
        self.dirty_initial_conditions = False
        self.time = 0.0
        self.positions = np.zeros((3, 3), dtype=float)
        self.velocities = np.zeros((3, 3), dtype=float)
        self.masses = np.ones(3, dtype=float)
        self.tails: list[deque[np.ndarray]] = []
        self.energy_times: list[float] = []
        self.kinetic_history: list[float] = []
        self.potential_history: list[float] = []

        self._configure_styles()
        self._build_layout()
        self._reset_simulation_from_controls(clear_dirty=True)
        self.after(FRAME_MS, self._animation_tick)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=APP_BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=APP_BG)
        style.configure("Panel.TFrame", background=GLASS_PANEL)
        style.configure("Card.TFrame", background=GLASS_CARD, relief="flat", borderwidth=0)
        style.configure("Control.TFrame", background=GLASS_CARD)
        style.configure("TLabel", background=APP_BG, foreground=TEXT)
        style.configure("Header.TLabel", background=GLASS_PANEL, foreground=TEXT, font=("Segoe UI", 15, "bold"))
        style.configure("CardTitle.TLabel", background=GLASS_CARD, foreground=TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabel", background=GLASS_CARD, foreground=ACCENT, font=("Segoe UI", 9, "bold"))
        style.configure("Control.TLabel", background=GLASS_CARD, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Note.TLabel", background=GLASS_PANEL, foreground=MUTED)
        style.configure("Status.TLabel", background="#152035", foreground=ACCENT, padding=(12, 8))
        style.configure("Collision.Status.TLabel", background="#2a1420", foreground="#ffb4bf", padding=(10, 7))
        style.configure(
            "TButton",
            background="#22314c",
            foreground=TEXT,
            bordercolor="#4c5d7a",
            focusthickness=0,
            padding=(10, 8),
        )
        style.map("TButton", background=[("active", "#2b3d5c"), ("pressed", "#19263d")])
        style.configure(
            "Accent.TButton",
            background="#25a89c",
            foreground="#ecfffb",
            bordercolor=ACCENT,
            padding=(10, 8),
        )
        style.map("Accent.TButton", background=[("active", "#27a79b"), ("pressed", "#176f67")])
        style.configure(
            "Dark.TEntry",
            fieldbackground="#0b1322",
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor="#41516c",
            lightcolor="#53647e",
            darkcolor="#25334d",
            padding=(5, 3),
        )
        style.configure("Dark.Horizontal.TScale", background=GLASS_CARD, troughcolor="#0b1322")
        style.configure(
            "TNotebook",
            background=APP_BG,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background="#121c2e",
            foreground=MUTED,
            padding=(18, 9),
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1b2940"), ("active", "#22324d")],
            foreground=[("selected", TEXT), ("active", TEXT)],
        )
        style.configure("Vertical.TScrollbar", background="#18243a", troughcolor=PANEL_BG, arrowcolor=MUTED)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0, minsize=450)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left_shell = GlassFrame(
            self,
            background=APP_BG,
            glass_color=GLASS_PANEL,
            content_style="Panel.TFrame",
            radius=28,
            padding=16,
            shadow=True,
            auto_height=False,
        )
        left_shell.grid(row=0, column=0, sticky="nsew", padx=(16, 10), pady=16)
        left_panel = left_shell.content
        left_panel.columnconfigure(0, weight=1)

        title = ttk.Label(left_panel, text="三体观测台", style="Header.TLabel")
        title.grid(row=0, column=0, sticky="w", pady=(0, 6))

        note = ttk.Label(
            left_panel,
            text="设置二维初始条件，观察三颗恒星的轨迹与系统能量变化。",
            style="Note.TLabel",
            wraplength=450,
        )
        note.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        buttons = ttk.Frame(left_panel, style="Panel.TFrame")
        buttons.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        for col in range(3):
            buttons.columnconfigure(col, weight=1)

        self.pause_button = ttk.Button(buttons, text="暂停", command=self._toggle_pause)
        self.apply_button = ttk.Button(
            buttons,
            text="应用 / 重启",
            command=self._apply_and_restart,
            style="Accent.TButton",
        )
        self.default_button = ttk.Button(buttons, text="默认值", command=self._restore_defaults)
        self.pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.apply_button.grid(row=0, column=1, sticky="ew", padx=5)
        self.default_button.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        self.status_label = ttk.Label(left_panel, text="", style="Status.TLabel", wraplength=450)
        self.status_label.grid(row=3, column=0, sticky="ew", pady=(0, 12))

        scroller = ScrollableFrame(left_panel)
        scroller.grid(row=4, column=0, sticky="nsew")
        left_panel.rowconfigure(4, weight=1)
        self._build_controls(scroller.content)

        right_panel = ttk.Frame(self, padding=(0, 16, 16, 16), style="TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._draw_active_tab())

        self._build_trajectory_tab()
        self._build_energy_tabs()

    def _build_controls(self, master: tk.Widget) -> None:
        row = 0
        specs_by_key = {spec.key: spec for spec in PARAMETER_SPECS}

        for body_index in range(3):
            card_shell = GlassFrame(
                master,
                background=GLASS_PANEL,
                glass_color=GLASS_CARD,
                content_style="Card.TFrame",
                radius=22,
                padding=12,
                shadow=True,
                auto_height=True,
            )
            card_shell.grid(row=row, column=0, sticky="ew", pady=(0, 12), padx=(0, 8))
            card = card_shell.content
            card.columnconfigure(0, weight=1, uniform="body_columns")
            card.columnconfigure(1, weight=1, uniform="body_columns")

            header = ttk.Label(card, text=BODY_NAMES[body_index], style="CardTitle.TLabel")
            header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

            position_label = ttk.Label(card, text="初始位置", style="Section.TLabel")
            velocity_label = ttk.Label(card, text="初速度方向 / 大小", style="Section.TLabel")
            position_label.grid(row=1, column=0, sticky="w", pady=(0, 4))
            velocity_label.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 4))

            for axis_row, axis in enumerate(("x", "y"), start=2):
                position_key = f"p{body_index}_{axis}"
                position_control = ParameterControl(
                    card,
                    specs_by_key[position_key],
                    self._mark_dirty,
                    label_text=axis,
                    label_width=2,
                )
                position_control.grid(row=axis_row, column=0, sticky="ew", pady=2, padx=(0, 8))
                self.controls[position_key] = position_control

            mass_control = ParameterControl(
                card,
                specs_by_key[f"m{body_index}"],
                self._mark_dirty,
                label_text="质量",
                label_width=4,
            )
            mass_control.grid(row=2, column=1, rowspan=2, sticky="ew", pady=2, padx=(12, 0))
            self.controls[f"m{body_index}"] = mass_control

            velocity_control = VelocityControl(
                card,
                specs_by_key[f"a{body_index}"],
                specs_by_key[f"s{body_index}"],
                BODY_COLORS[body_index],
                self._mark_dirty,
            )
            velocity_control.grid(row=5, column=0, columnspan=2, sticky="ew", pady=2)
            self.velocity_controls[body_index] = velocity_control
            self.controls[f"a{body_index}"] = velocity_control.angle_control
            self.controls[f"s{body_index}"] = velocity_control.speed_control
            row += 1

        simulation_shell = GlassFrame(
            master,
            background=GLASS_PANEL,
            glass_color=GLASS_CARD,
            content_style="Card.TFrame",
            radius=22,
            padding=12,
            shadow=True,
            auto_height=True,
        )
        simulation_shell.grid(row=row, column=0, sticky="ew", pady=(0, 8), padx=(0, 8))
        simulation_card = simulation_shell.content
        simulation_card.columnconfigure(0, weight=1)

        simulation_header = ttk.Label(simulation_card, text="模拟控制", style="CardTitle.TLabel")
        simulation_header.grid(row=0, column=0, sticky="w", pady=(0, 8))
        row += 1

        simulation_labels = {
            "dt": "步长",
            "steps": "每帧",
            "tail": "尾迹",
            "collision": "碰撞",
        }
        for control_row, key in enumerate(("dt", "steps", "tail", "collision"), start=1):
            control = ParameterControl(
                simulation_card,
                specs_by_key[key],
                self._simulation_setting_changed,
                label_text=simulation_labels[key],
                label_width=4,
            )
            control.grid(row=control_row, column=0, sticky="ew", pady=3)
            self.controls[key] = control

        master.columnconfigure(0, weight=1)

    def _build_trajectory_tab(self) -> None:
        tab = ttk.Frame(self.notebook, style="TFrame")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.notebook.add(tab, text="运动轨迹")

        holder = ttk.Frame(tab, style="TFrame")
        holder.grid(row=0, column=0)

        self.trajectory_figure = Figure(figsize=(7.0, 7.0), dpi=100, facecolor=SPACE_BG)
        self.trajectory_axis = self.trajectory_figure.add_subplot(111)
        self.trajectory_axis.set_facecolor(SPACE_BG)
        self.trajectory_axis.set_aspect("equal", adjustable="box")
        self.trajectory_axis.set_axis_off()
        self.trajectory_figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.tail_core_collections = []
        self.tail_glow_collections = []
        self.star_glows = []
        self.star_points = []
        for index, color in enumerate(BODY_COLORS):
            tail_glow = LineCollection([], capstyle="round", joinstyle="round", zorder=2)
            tail_core = LineCollection([], capstyle="round", joinstyle="round", zorder=3)
            self.trajectory_axis.add_collection(tail_glow)
            self.trajectory_axis.add_collection(tail_core)
            glow_image = self.trajectory_axis.imshow(
                make_star_glow_image(color),
                extent=(-1.0, 1.0, -1.0, 1.0),
                interpolation="bilinear",
                origin="lower",
                visible=False,
                zorder=4,
            )
            (star_point,) = self.trajectory_axis.plot(
                [],
                [],
                marker="o",
                markersize=5.8,
                color="#fffdf2",
                markeredgecolor=color,
                markeredgewidth=0.85,
                linestyle="None",
                zorder=5,
            )
            self.tail_glow_collections.append(tail_glow)
            self.tail_core_collections.append(tail_core)
            self.star_glows.append(glow_image)
            self.star_points.append(star_point)

        self.flash_overlay = Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=self.trajectory_axis.transAxes,
            facecolor="#fff7d6",
            edgecolor="none",
            alpha=0.0,
            visible=False,
            zorder=20,
        )
        self.trajectory_axis.add_patch(self.flash_overlay)

        self.trajectory_canvas = FigureCanvasTkAgg(self.trajectory_figure, master=holder)
        widget = self.trajectory_canvas.get_tk_widget()
        widget.configure(width=700, height=700, background=SPACE_BG, highlightthickness=0)
        widget.pack()
        self.trajectory_tab = tab
        self.trajectory_canvas_widget = widget
        tab.bind("<Configure>", self._resize_trajectory_canvas)

    def _build_energy_tabs(self) -> None:
        self.kinetic_figure, self.kinetic_axis, self.kinetic_canvas = self._make_energy_tab(
            "总动能", "#ff8a4c"
        )
        self.potential_figure, self.potential_axis, self.potential_canvas = self._make_energy_tab(
            "总势能", "#72ddf7"
        )

    def _make_energy_tab(self, title: str, color: str) -> tuple[Figure, object, FigureCanvasTkAgg]:
        tab = ttk.Frame(self.notebook, style="TFrame")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        self.notebook.add(tab, text=title)

        figure = Figure(figsize=(7.8, 5.8), dpi=100, facecolor=APP_BG)
        axis = figure.add_subplot(111)
        axis.set_facecolor(PLOT_BG)
        axis.set_title(title)
        axis.set_xlabel("时间")
        axis.set_ylabel("能量")
        axis.title.set_color(TEXT)
        axis.xaxis.label.set_color(MUTED)
        axis.yaxis.label.set_color(MUTED)
        axis.tick_params(colors=MUTED)
        for spine in axis.spines.values():
            spine.set_color("#26344f")
        axis.grid(True, color="#1d2a42", linewidth=0.8)
        (line,) = axis.plot([], [], color=color, linewidth=1.8)
        axis._energy_line = line
        figure.subplots_adjust(left=0.12, right=0.96, top=0.9, bottom=0.13)

        canvas = FigureCanvasTkAgg(figure, master=tab)
        widget = canvas.get_tk_widget()
        widget.configure(background=APP_BG, highlightthickness=0)
        widget.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        return figure, axis, canvas

    def _resize_trajectory_canvas(self, event: tk.Event) -> None:
        side = max(240, min(event.width, event.height) - 16)
        current_width = self.trajectory_canvas_widget.winfo_width()
        current_height = self.trajectory_canvas_widget.winfo_height()
        if abs(current_width - side) < 2 and abs(current_height - side) < 2:
            return

        dpi = self.trajectory_figure.get_dpi()
        self.trajectory_figure.set_size_inches(side / dpi, side / dpi, forward=False)
        self.trajectory_canvas_widget.configure(width=side, height=side)
        self.trajectory_canvas.draw_idle()

    def _read_initial_conditions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.zeros((3, 3), dtype=float)
        velocities = np.zeros((3, 3), dtype=float)
        masses = np.zeros(3, dtype=float)
        axes = ("x", "y")

        for body_index in range(3):
            for axis_index, axis in enumerate(axes):
                positions[body_index, axis_index] = self.controls[f"p{body_index}_{axis}"].value()
            angle = math.radians(self.velocity_controls[body_index].angle())
            speed = self.velocity_controls[body_index].speed()
            velocities[body_index, 0] = speed * math.cos(angle)
            velocities[body_index, 1] = speed * math.sin(angle)
            masses[body_index] = self.controls[f"m{body_index}"].value()
            if masses[body_index] <= 0:
                raise ValueError(f"{BODY_NAMES[body_index]} 的质量必须为正数。")

        return positions, velocities, masses

    def _reset_simulation_from_controls(self, clear_dirty: bool) -> None:
        try:
            self.positions, self.velocities, self.masses = self._read_initial_conditions()
        except ValueError as exc:
            messagebox.showerror("初始条件无效", str(exc))
            return

        self.ended = False
        self.paused = False
        self.collision_info = None
        self.collided_bodies.clear()
        self.survivor_index = None
        self.collision_point = None
        self.flash_frame = 0
        self.flash_total_frames = 0
        self.pause_button.state(["!disabled"])
        self.pause_button.configure(text="暂停")
        self.time = 0.0
        tail_length = self._tail_length()
        self.tails = [deque(maxlen=tail_length) for _ in range(3)]
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
        self._update_plots()
        self._draw_all_tabs()

    def _restore_defaults(self) -> None:
        for spec in PARAMETER_SPECS:
            self.controls[spec.key].set_value(spec.default, notify=False)
        for velocity_control in self.velocity_controls.values():
            velocity_control.refresh()
        self._reset_simulation_from_controls(clear_dirty=True)

    def _apply_and_restart(self) -> None:
        self._reset_simulation_from_controls(clear_dirty=True)

    def _toggle_pause(self) -> None:
        if self.ended:
            self._update_status()
            return
        self.paused = not self.paused
        self.pause_button.configure(text="继续" if self.paused else "暂停")
        self._update_status()

    def _mark_dirty(self) -> None:
        self.dirty_initial_conditions = True
        self._update_status()

    def _simulation_setting_changed(self) -> None:
        self._sync_tail_deques()
        self._update_status()

    def _update_status(self) -> None:
        self.status_label.configure(style="Status.TLabel")
        if self.ended and self.collision_info is not None:
            status = "碰撞结束"
        else:
            status = "已暂停" if self.paused else "运行中"
        if self.dirty_initial_conditions:
            status += " - 初始值已修改，请点击“应用 / 重启”。"
        self.status_label.configure(text=status)

    def _tail_length(self) -> int:
        return max(2, int(round(self.controls["tail"].value())))

    def _sync_tail_deques(self) -> None:
        new_length = self._tail_length()
        if not self.tails:
            return
        if all(tail.maxlen == new_length for tail in self.tails):
            return
        self.tails = [deque(tail, maxlen=new_length) for tail in self.tails]

    def _steps_per_frame(self) -> int:
        return max(1, int(round(self.controls["steps"].value())))

    def _time_step(self) -> float:
        return max(1e-6, float(self.controls["dt"].value()))

    def _collision_radius(self) -> float:
        return max(1e-6, float(self.controls["collision"].value()))

    def _animation_tick(self) -> None:
        if not self.paused and not self.ended:
            for _ in range(self._steps_per_frame()):
                self._integrate_one_step(self._time_step())
                if self.ended:
                    break
            self._update_plots()
            self._draw_active_tab()
        elif self.ended and self.survivor_index is not None:
            self._advance_survivor_after_collision()
            if self.flash_frame < self.flash_total_frames:
                self.flash_frame += 1
            self._update_trajectory_plot()
            self._draw_active_tab()
        self.after(FRAME_MS, self._animation_tick)

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
        new_velocities = self.velocities + 0.5 * (acceleration + new_acceleration) * dt

        self.positions = new_positions
        self.velocities = new_velocities
        self.time += dt

        for index in range(3):
            self.tails[index].append(self.positions[index].copy())
        self._record_energy()
        self._check_for_collision()

    def _check_for_collision(self) -> bool:
        radius = self._collision_radius()
        closest_pair = None
        closest_distance = float("inf")

        for i in range(3):
            for j in range(i + 1, 3):
                distance = float(np.linalg.norm(self.positions[j] - self.positions[i]))
                if distance < closest_distance:
                    closest_pair = (i, j)
                    closest_distance = distance

        if closest_pair is None or closest_distance > radius:
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
        self.pause_button.configure(text="已结束")
        self.pause_button.state(["disabled"])
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

    def _energies(self) -> tuple[float, float]:
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

    def _update_plots(self) -> None:
        self._update_trajectory_plot()
        self._update_energy_plot(self.kinetic_axis, self.kinetic_history)
        self._update_energy_plot(self.potential_axis, self.potential_history)

    def _update_trajectory_plot(self) -> None:
        all_xy_points = []

        for index in range(3):
            if index in self.collided_bodies:
                self._hide_body_artists(index)
                continue

            tail = np.array(self.tails[index])
            xy = tail[:, :2]
            all_xy_points.append(xy)
            self._update_tail_artists(index, xy)
            self.star_glows[index].set_visible(True)
            self.star_points[index].set_data([self.positions[index, 0]], [self.positions[index, 1]])

        if self.collision_point is not None:
            all_xy_points.append(self.collision_point[:2].reshape(1, 2))

        all_xy = np.vstack(all_xy_points)
        min_xy = all_xy.min(axis=0)
        max_xy = all_xy.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        span = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 0.5)
        half_range = span * 0.62
        self.trajectory_axis.set_xlim(center[0] - half_range, center[0] + half_range)
        self.trajectory_axis.set_ylim(center[1] - half_range, center[1] + half_range)
        self._update_star_glow_extents(half_range)
        self._update_flash_artists()

    def _hide_body_artists(self, index: int) -> None:
        self.tail_core_collections[index].set_segments([])
        self.tail_glow_collections[index].set_segments([])
        self.star_glows[index].set_visible(False)
        self.star_points[index].set_data([], [])

    def _update_star_glow_extents(self, half_range: float) -> None:
        glow_radius = half_range * 0.055
        for index, glow_image in enumerate(self.star_glows):
            if index in self.collided_bodies:
                glow_image.set_visible(False)
                continue
            x_position = float(self.positions[index, 0])
            y_position = float(self.positions[index, 1])
            glow_image.set_extent(
                (
                    x_position - glow_radius,
                    x_position + glow_radius,
                    y_position - glow_radius,
                    y_position + glow_radius,
                )
            )
            glow_image.set_visible(True)

    def _update_flash_artists(self) -> None:
        if self.flash_total_frames <= 0 or self.flash_frame >= self.flash_total_frames:
            self.flash_overlay.set_visible(False)
            return

        progress = self.flash_frame / max(1, self.flash_total_frames - 1)
        alpha = 0.62 * (1.0 - progress) ** 2
        self.flash_overlay.set_alpha(alpha)
        self.flash_overlay.set_visible(True)

    def _update_tail_artists(self, index: int, xy: np.ndarray) -> None:
        if len(xy) < 2:
            self.tail_core_collections[index].set_segments([])
            self.tail_glow_collections[index].set_segments([])
            return

        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        progress = np.linspace(0.0, 1.0, len(segments))
        color = BODY_COLORS[index]

        core_widths = 0.08 + 1.15 * np.power(progress, 1.75)
        glow_widths = 0.6 + 2.6 * np.power(progress, 1.65)
        core_colors = [color_to_rgba(color, 0.03 + 0.64 * value**1.35) for value in progress]
        glow_colors = [color_to_rgba(color, 0.015 + 0.16 * value**1.55) for value in progress]

        self.tail_core_collections[index].set_segments(segments)
        self.tail_core_collections[index].set_linewidths(core_widths)
        self.tail_core_collections[index].set_color(core_colors)
        self.tail_glow_collections[index].set_segments(segments)
        self.tail_glow_collections[index].set_linewidths(glow_widths)
        self.tail_glow_collections[index].set_color(glow_colors)

    def _update_energy_plot(self, axis: object, values: list[float]) -> None:
        line = axis._energy_line
        line.set_data(self.energy_times, values)
        if len(self.energy_times) < 2:
            return

        x_min = self.energy_times[0]
        x_max = self.energy_times[-1]
        y_min = min(values)
        y_max = max(values)
        y_span = y_max - y_min
        if y_span <= 1e-12:
            y_span = max(abs(y_max), 1.0) * 0.1
        axis.set_xlim(x_min, max(x_max, x_min + 1e-6))
        axis.set_ylim(y_min - y_span * 0.12, y_max + y_span * 0.12)

    def _draw_active_tab(self) -> None:
        selected = self.notebook.index(self.notebook.select())
        if selected == 0:
            self.trajectory_canvas.draw_idle()
        elif selected == 1:
            self.kinetic_canvas.draw_idle()
        elif selected == 2:
            self.potential_canvas.draw_idle()

    def _draw_all_tabs(self) -> None:
        self.trajectory_canvas.draw_idle()
        self.kinetic_canvas.draw_idle()
        self.potential_canvas.draw_idle()


if __name__ == "__main__":
    app = ThreeBodySimulator()
    app.mainloop()
