"""
local_viewer.py — Native single-window 2DGS viewer using PyQt5.

All controls and the rendered frame in one window.
No browser · No JPEG · No WebSocket.

Usage:
    python local_viewer.py path/to/scene.ply [--no-culling] [--no-profiling]

Camera navigation (in the render area):
    Left-drag    — orbit  (rotate around look-at point)
    Right-drag   — pan    (translate look-at point)
    Scroll       — zoom in / out
    WASD         — translate look-at (forward / back / strafe)
    E / Q        — translate up / down along world-up axis
    Arrow keys   — FPS look  (yaw / pitch in place)
    R            — reset camera to defaults

Settings are persisted to ~/.config/2dgs_viewer/config.json on close.
"""

from __future__ import annotations

import os, sys, math, time, threading, argparse, collections, json, pathlib
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "2d_gaussian_splatting"))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QSlider, QComboBox, QCheckBox, QPushButton,
    QGroupBox, QScrollArea, QSizePolicy, QSplitter,
    QSpinBox, QDoubleSpinBox,
)
from PyQt5.QtGui  import QImage, QPixmap, QPainter, QPen, QPainterPath, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QLocale

from internal.viewer               import ViewerRenderer
from internal.viewer               import GaussianModelforViewer as GaussianModel
from internal.cameras.cameras      import Cameras
from internal.utils.graphics_utils import fov2focal
from sensor import _detect_sh_degree, _load_octree, _load_ply_into_model


# ── Constants ──────────────────────────────────────────────────────────────────

UP_AXIS_OPTIONS = ["+Z", "+Y", "+X", "-Z", "-Y", "-X"]
UP_AXIS_VECTORS: dict[str, np.ndarray] = {
    "+Z": np.array([ 0.,  0.,  1.], dtype=np.float64),
    "+Y": np.array([ 0.,  1.,  0.], dtype=np.float64),
    "+X": np.array([ 1.,  0.,  0.], dtype=np.float64),
    "-Z": np.array([ 0.,  0., -1.], dtype=np.float64),
    "-Y": np.array([ 0., -1.,  0.], dtype=np.float64),
    "-X": np.array([-1.,  0.,  0.], dtype=np.float64),
}

AR_RATIO_OPTIONS = ["16:9", "4:3", "16:10", "21:9", "3:2", "1:1", "9:16", "3:4", "Custom"]
AR_RATIO_VALUES: dict[str, float | None] = {
    "16:9":  16/9, "4:3":  4/3,  "16:10": 16/10, "21:9": 21/9,
    "3:2":   3/2,  "1:1":  1.0,  "9:16":  9/16,  "3:4":  3/4,
    "Custom": None,
}

RESOLUTION_PRESETS: dict[str, tuple[int, int] | None] = {
    "Custom":       None,
    "360p":         (640,  360),
    "480p":         (854,  480),
    "720p (HD)":    (1280, 720),
    "1080p (FHD)":  (1920, 1080),
    "1440p (QHD)":  (2560, 1440),
    "4K (UHD)":     (3840, 2160),
}

RENDER_TYPES = [
    "RGB", "Edge", "Alpha", "Normal", "View-Normal",
    "Depth", "Depth-Distort", "Depth-to-Normal", "Depth-to-Curvature",
]
RENDER_TYPE_MAP = {
    "RGB":               "render",
    "Edge":              "edge",
    "Alpha":             "rend_alpha",
    "Normal":            "rend_normal",
    "View-Normal":       "view_normal",
    "Depth":             "surf_depth",
    "Depth-Distort":     "rend_dist",
    "Depth-to-Normal":   "surf_normal",
    "Depth-to-Curvature":"curvature",
}

_DEFAULT_W, _DEFAULT_H = 1280, 720

# ── Persistent config ──────────────────────────────────────────────────────────

CONFIG_PATH = pathlib.Path.home() / ".config" / "2dgs_viewer" / "config.json"

CONFIG_DEFAULTS: dict = {
    "fov_deg":           60.0,
    "move_speed":        1.0,
    "orbit_speed":       1.0,
    "mouse_inv_x":       True,
    "mouse_inv_y":       True,
    "kb_inv_x":          True,
    "kb_inv_y":          True,
    "world_up":          "+Z",
    "render_w":          _DEFAULT_W,
    "render_h":          _DEFAULT_H,
    "ar_preset":         "16:9",
    "lock_ar":           True,
    "depth_ratio":       0.0,
    "active_sh_degree":  -1,      # -1 = use PLY file's max
    "opacity_thresh":    0.05,
    "sparsity":          1,
    "scaling_mod":       1.0,
    "point_size":        0.01,
    "render_type":       "RGB",
    "show_fps_overlay":  True,
    "show_splat_overlay": True,
}


def _load_config() -> dict:
    cfg = CONFIG_DEFAULTS.copy()
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open() as fh:
                stored = json.load(fh)
            cfg.update({k: v for k, v in stored.items() if k in CONFIG_DEFAULTS})
        except Exception:
            pass
    return cfg


def _save_config(cfg: dict):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w") as fh:
            json.dump(cfg, fh, indent=2)
    except Exception:
        pass


def _fmt_splats(n: int) -> str:
    """Format a splat count as e.g. '1.2M' or '500K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


# ── Orbit camera ───────────────────────────────────────────────────────────────

class OrbitCamera:
    """Spherical-coordinate orbit camera with configurable world-up axis."""

    def __init__(self, look_at=(0., 0., 0.), distance=5.0,
                 yaw=0.0, pitch=0.3, fov_deg=60.0,
                 world_up: np.ndarray | None = None):
        self.look_at  = np.array(look_at, dtype=np.float64)
        self.distance = float(distance)
        self.yaw      = float(yaw)
        self.pitch    = float(pitch)
        self.fov_deg  = float(fov_deg)
        self.world_up = (world_up.copy() if world_up is not None
                         else np.array([0., 0., 1.], dtype=np.float64))

    def _basis(self) -> tuple[np.ndarray, np.ndarray]:
        """right_ref, fwd_ref — two unit vectors perpendicular to world_up."""
        up  = self.world_up / np.linalg.norm(self.world_up)
        ref = np.array([0., 0., 1.]) if abs(up[2]) < 0.9 else np.array([1., 0., 0.])
        right = np.cross(up, ref);   right /= np.linalg.norm(right)
        fwd   = np.cross(right, up); fwd   /= np.linalg.norm(fwd)
        return right, fwd

    @property
    def position(self) -> np.ndarray:
        up = self.world_up / np.linalg.norm(self.world_up)
        right_ref, fwd_ref = self._basis()
        e_yaw = math.cos(self.yaw) * fwd_ref + math.sin(self.yaw) * right_ref
        return self.look_at + self.distance * (
            math.cos(self.pitch) * e_yaw + math.sin(self.pitch) * up)

    def orbit(self, dyaw: float, dpitch: float):
        """Orbit camera around look_at — position moves, look_at fixed."""
        self.yaw  += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch,
                                   -math.pi / 2 + 0.01, math.pi / 2 - 0.01))

    def fps_look(self, dyaw: float, dpitch: float):
        """FPS-style look — camera position stays fixed, look_at moves."""
        pos = self.position.copy()
        self.yaw  += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch,
                                   -math.pi / 2 + 0.01, math.pi / 2 - 0.01))
        # Recompute look_at so that position stays at pos
        up = self.world_up / np.linalg.norm(self.world_up)
        right_ref, fwd_ref = self._basis()
        e_yaw = math.cos(self.yaw) * fwd_ref + math.sin(self.yaw) * right_ref
        offset = self.distance * (math.cos(self.pitch) * e_yaw
                                  + math.sin(self.pitch) * up)
        self.look_at = pos - offset

    def pan(self, dx: float, dy: float):
        up  = self.world_up / np.linalg.norm(self.world_up)
        pos = self.position
        fwd = self.look_at - pos;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right, _ = self._basis()
        else:
            right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        self.look_at += dx * right + dy * cam_up

    def move(self, fwd_d: float, right_d: float):
        """Translate look_at horizontally relative to current view direction."""
        up   = self.world_up / np.linalg.norm(self.world_up)
        pos  = self.position
        look = self.look_at - pos;  look /= np.linalg.norm(look)
        horiz = look - np.dot(look, up) * up
        if np.linalg.norm(horiz) < 1e-6:
            _, horiz = self._basis()
        else:
            horiz /= np.linalg.norm(horiz)
        right_vec = np.cross(horiz, up)
        if np.linalg.norm(right_vec) < 1e-6:
            right_vec, _ = self._basis()
        else:
            right_vec /= np.linalg.norm(right_vec)
        self.look_at = self.look_at + fwd_d * horiz + right_d * right_vec

    def translate_up(self, amount: float):
        """Move both camera and look_at along world_up (Q/E vertical translation)."""
        up = self.world_up / np.linalg.norm(self.world_up)
        self.look_at += amount * up

    def zoom(self, delta: float):
        self.distance = max(0.05, self.distance * (1.0 - delta * 0.15))

    def build_RT(self, cam_tf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        up  = self.world_up / np.linalg.norm(self.world_up)
        pos = self.position
        fwd = self.look_at - pos;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            _, fwd_ref = self._basis()
            right = np.cross(fwd, fwd_ref)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        # c2w in OpenGL convention (Y-up, Z-backward)
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, 0] = torch.tensor(right,  dtype=torch.float64)
        c2w[:3, 1] = torch.tensor(cam_up, dtype=torch.float64)
        c2w[:3, 2] = torch.tensor(-fwd,   dtype=torch.float64)
        c2w[:3, 3] = torch.tensor(pos,    dtype=torch.float64)
        c2w = torch.matmul(cam_tf, c2w)
        c2w[:3, 1:3] *= -1          # flip Y, Z — matches client.py:get_RT
        w2c = torch.linalg.inv(c2w)
        return w2c[:3, :3].float(), w2c[:3, 3].float()


# ── Render display widget ──────────────────────────────────────────────────────

class RenderWidget(QLabel):
    camera_changed = pyqtSignal()
    key_down       = pyqtSignal(int)
    key_up         = pyqtSignal(int)

    def __init__(self, camera: OrbitCamera):
        super().__init__()
        self.cam               = camera
        self.mouse_inv_x       = True    # default: inverted (drag-scene feel)
        self.mouse_inv_y       = True
        self._last             = None
        self._btns             = set()
        self._fps_history      = collections.deque(maxlen=90)
        self._splat_history    = collections.deque(maxlen=90)
        self._show_fps_overlay   = True
        self._show_splat_overlay = True
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #1a1a1a;")
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, img_np: np.ndarray):
        H, W = img_np.shape[:2]
        qimg = QImage(img_np.tobytes(), W, H, W * 3, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def mousePressEvent(self, e):
        self._last = e.pos()
        self._btns.add(e.button())
        self.setFocus()

    def mouseReleaseEvent(self, e):
        self._btns.discard(e.button())

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        dx = e.x() - self._last.x()
        dy = e.y() - self._last.y()
        self._last = e.pos()
        if Qt.LeftButton in self._btns:
            sx = -1.0 if self.mouse_inv_x else 1.0
            sy = -1.0 if self.mouse_inv_y else 1.0
            self.cam.orbit(sx * dx * 0.005, sy * dy * 0.005)
            self.camera_changed.emit()
        elif Qt.RightButton in self._btns:
            scale = self.cam.distance * 0.001
            self.cam.pan(-dx * scale, dy * scale)
            self.camera_changed.emit()

    def wheelEvent(self, e):
        self.cam.zoom(e.angleDelta().y() / 120.0)
        self.camera_changed.emit()

    def keyPressEvent(self, e):
        if not e.isAutoRepeat():
            self.key_down.emit(e.key())

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat():
            self.key_up.emit(e.key())

    def focusOutEvent(self, e):
        self.key_up.emit(-1)   # sentinel: clear all held keys
        super().focusOutEvent(e)

    # ── FPS overlay ────────────────────────────────────────────────────────────

    def update_fps(self, fps: float):
        self._fps_history.append(fps)
        self.update()

    def update_splat_count(self, n: int):
        self._splat_history.append(n)
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._show_fps_overlay and self._fps_history:
            p = QPainter(self)
            self._draw_fps_overlay(p)
            p.end()
        if self._show_splat_overlay and self._splat_history:
            p = QPainter(self)
            h_fps = 90 + 4 if (self._show_fps_overlay and self._fps_history) else 0
            self._draw_splat_overlay(p, y_offset=h_fps)
            p.end()

    def _draw_fps_overlay(self, p: QPainter):
        W_OV, H_OV = 150, 90
        MARGIN      = 8
        PAD_L       = 30    # width of Y-axis label column
        PAD_T       = 18    # height of fps-text row at top
        PAD_R       = 4
        PAD_B       = 4

        ox = self.width() - W_OV - MARGIN
        oy = MARGIN
        if ox < 0:
            return

        p.setRenderHint(QPainter.Antialiasing)

        # Semi-transparent background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(ox, oy, W_OV, H_OV, 6, 6)

        vals = list(self._fps_history)
        current_fps = vals[-1] if vals else 0.0

        # Adaptive Y max rounded to nearest nice value
        peak = max(vals) if vals else 30.0
        y_max = float(next((n for n in (30, 60, 90, 120, 144, 240)
                            if n >= peak * 0.8), 240))

        # Graph region
        gx = float(ox + PAD_L)
        gy = float(oy + PAD_T)
        gw = float(W_OV - PAD_L - PAD_R)
        gh = float(H_OV - PAD_T - PAD_B)

        # Y-axis grid lines + labels
        font = QFont(); font.setPointSize(7)
        p.setFont(font)
        for y_val in (0.0, y_max / 2, y_max):
            py = gy + gh - (y_val / y_max) * gh
            p.setPen(QPen(QColor(255, 255, 255, 45), 0.8))
            p.drawLine(QPointF(gx, py), QPointF(gx + gw, py))
            p.setPen(QColor(200, 200, 200, 200))
            p.drawText(QRectF(ox, py - 6.0, float(PAD_L - 3), 12.0),
                       Qt.AlignRight | Qt.AlignVCenter, str(int(y_val)))

        # FPS line plot
        if len(vals) >= 2:
            path = QPainterPath()
            n = len(vals)
            for i, v in enumerate(vals):
                x = gx + (i / (n - 1)) * gw
                y = gy + gh - min(v / y_max, 1.0) * gh
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            col = (QColor(80, 220, 80)  if current_fps >= 25 else
                   QColor(255, 200, 50) if current_fps >= 12 else
                   QColor(255, 80,  80))
            p.setPen(QPen(col, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        # Current fps label at top
        font2 = QFont(); font2.setPointSize(8); font2.setBold(True)
        p.setFont(font2)
        p.setPen(QColor(255, 255, 255, 230))
        p.drawText(QRectF(float(ox), float(oy) + 1.0, float(W_OV), float(PAD_T) - 2.0),
                   Qt.AlignCenter, f"{current_fps:.1f} fps")

    def _draw_splat_overlay(self, p: QPainter, y_offset: int = 0):
        W_OV, H_OV = 150, 90
        MARGIN      = 8
        PAD_L       = 36    # wider — splat labels like "2.5M" need space
        PAD_T       = 18
        PAD_R       = 4
        PAD_B       = 4

        ox = self.width() - W_OV - MARGIN
        oy = MARGIN + y_offset
        if ox < 0:
            return

        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRoundedRect(ox, oy, W_OV, H_OV, 6, 6)

        vals = list(self._splat_history)
        current_n = vals[-1] if vals else 0

        peak = max(vals) if vals else 1
        # Round to a nice power of magnitude
        magnitude = 10 ** math.floor(math.log10(max(peak, 1)))
        y_max = float(math.ceil(peak / magnitude) * magnitude)
        if y_max == 0:
            y_max = 1.0

        gx = float(ox + PAD_L)
        gy = float(oy + PAD_T)
        gw = float(W_OV - PAD_L - PAD_R)
        gh = float(H_OV - PAD_T - PAD_B)

        font = QFont(); font.setPointSize(7)
        p.setFont(font)
        for y_val in (0.0, y_max / 2, y_max):
            py = gy + gh - (y_val / y_max) * gh
            p.setPen(QPen(QColor(255, 255, 255, 45), 0.8))
            p.drawLine(QPointF(gx, py), QPointF(gx + gw, py))
            p.setPen(QColor(200, 200, 200, 200))
            p.drawText(QRectF(float(ox), py - 6.0, float(PAD_L - 3), 12.0),
                       Qt.AlignRight | Qt.AlignVCenter, _fmt_splats(int(y_val)))

        if len(vals) >= 2:
            path = QPainterPath()
            n = len(vals)
            for i, v in enumerate(vals):
                x = gx + (i / (n - 1)) * gw
                y = gy + gh - min(v / y_max, 1.0) * gh
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(QColor(100, 180, 255), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        font2 = QFont(); font2.setPointSize(8); font2.setBold(True)
        p.setFont(font2)
        p.setPen(QColor(255, 255, 255, 230))
        p.drawText(QRectF(float(ox), float(oy) + 1.0, float(W_OV), float(PAD_T) - 2.0),
                   Qt.AlignCenter, f"{_fmt_splats(current_n)} splats")


# ── Slider + spinbox widget ────────────────────────────────────────────────────

def _slider_spin(mn: float, mx: float, val: float, on_change,
                 is_float: bool = False,
                 decimals: int = 2,
                 step: float | None = None) -> QWidget:
    """Horizontal slider paired with a spinbox, kept bidirectionally in sync."""
    row = QWidget()
    hl  = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0)
    updating = False

    if is_float:
        if step is None:
            step = (mx - mn) / 100.0
        n_steps = max(1, round((mx - mn) / step))
        init_i  = min(n_steps, max(0, round((val - mn) / step)))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_steps)
        slider.setValue(init_i)

        spin = QDoubleSpinBox()
        spin.setRange(mn, mx); spin.setDecimals(decimals)
        spin.setSingleStep(step); spin.setFixedWidth(72)
        spin.setValue(mn + init_i * step)

        def on_s(i):
            nonlocal updating
            if updating: return
            v = mn + i * step
            updating = True; spin.setValue(v); updating = False
            on_change(v)

        def on_b(v):
            nonlocal updating
            if updating: return
            i = min(n_steps, max(0, round((v - mn) / step)))
            updating = True; slider.setValue(i); updating = False
            on_change(v)

        slider.valueChanged.connect(on_s)
        spin.valueChanged.connect(on_b)
    else:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(mn), int(mx))
        slider.setValue(int(val))

        spin = QSpinBox()
        spin.setRange(int(mn), int(mx))
        spin.setFixedWidth(55); spin.setValue(int(val))

        def on_s(i):
            nonlocal updating
            if updating: return
            updating = True; spin.setValue(i); updating = False
            on_change(i)

        def on_b(v):
            nonlocal updating
            if updating: return
            updating = True; slider.setValue(v); updating = False
            on_change(v)

        slider.valueChanged.connect(on_s)
        spin.valueChanged.connect(on_b)

    hl.addWidget(slider)
    hl.addWidget(spin)
    return row


def _combo(items, current, on_change) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    c.setCurrentText(current)
    c.currentTextChanged.connect(on_change)
    return c


# ── Main window ────────────────────────────────────────────────────────────────

class LocalViewer(QMainWindow):

    def __init__(self, ply_path: str, args):
        super().__init__()
        self.device   = torch.device("cuda")
        self.ply_path = ply_path
        self.cam_tf   = torch.eye(4, dtype=torch.float64)

        # Config (loaded before model so defaults are available everywhere)
        self._cfg = _load_config()
        cfg = self._cfg

        # Load model
        sh = _detect_sh_degree(ply_path) if args.sh_degree < 0 else args.sh_degree
        self._max_sh = sh
        model = GaussianModel(sh_degree=sh)
        _load_ply_into_model(model, ply_path)
        octree = _load_octree(ply_path)
        self._total_vram = (torch.cuda.get_device_properties(self.device)
                            .total_memory / 1024**2)
        bg = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device=self.device)
        self.renderer = ViewerRenderer(
            model, bg,
            do_initialize=True,
            octree=octree,
            culling_enabled=not args.no_culling,
            profiling_enabled=not args.no_profiling,
        )

        # Camera
        self.camera = OrbitCamera(
            world_up=UP_AXIS_VECTORS[cfg["world_up"]].copy()
        )

        # Render resolution
        self._render_w     = cfg["render_w"]
        self._render_h     = cfg["render_h"]
        self._aspect_ratio = cfg["render_w"] / max(cfg["render_h"], 1)
        self._lock_ar      = cfg["lock_ar"]

        # Render options
        self.fov_deg        = cfg["fov_deg"]
        self.render_type    = cfg["render_type"] if cfg["render_type"] in RENDER_TYPES else "RGB"
        self.render_type1   = "RGB"
        self.render_type2   = "RGB"
        self.split_enabled  = False
        self.split_pos      = 0.5
        self.depth_ratio    = cfg["depth_ratio"]
        _sh_cfg             = cfg["active_sh_degree"]
        self.sh_degree      = sh if _sh_cfg < 0 else min(_sh_cfg, sh)
        self.opacity_thresh = cfg["opacity_thresh"]
        self.sparsity       = cfg["sparsity"]
        self.scaling_mod    = cfg["scaling_mod"]
        self.point_size     = cfg["point_size"]
        self.show_ptc       = False
        self.surfel_disk    = False
        self.crop_enabled   = False
        self.crop_x         = [-4.0, 4.0]
        self.crop_y         = [-4.0, 4.0]
        self.crop_z         = [-4.0, 4.0]

        # Navigation speeds (multipliers)
        self._move_speed  = cfg["move_speed"]
        self._orbit_speed = cfg["orbit_speed"]

        # KB rotation inversion (arrow keys only, not WASD)
        self._kb_inv_x  = cfg["kb_inv_x"]
        self._kb_inv_y  = cfg["kb_inv_y"]
        self._keys_held = set()

        # Overlay visibility
        self._show_fps_overlay   = cfg["show_fps_overlay"]
        self._show_splat_overlay = cfg["show_splat_overlay"]

        # Frame slot
        self._frame_slot  = None
        self._frame_lock  = threading.Lock()
        self._render_trig = threading.Event()
        self._running     = True

        self._build_ui()
        self._start_render_thread()

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_frame)
        self._poll_timer.start(16)

        self._kb_timer = QTimer()
        self._kb_timer.timeout.connect(self._kb_tick)
        self._kb_timer.start(16)

        self._render_trig.set()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("2DGS Viewer")
        self.resize(1640, 900)

        self.render_widget = RenderWidget(self.camera)
        self.render_widget.mouse_inv_x       = self._cfg["mouse_inv_x"]
        self.render_widget.mouse_inv_y       = self._cfg["mouse_inv_y"]
        self.render_widget._show_fps_overlay   = self._show_fps_overlay
        self.render_widget._show_splat_overlay = self._show_splat_overlay
        self.render_widget.camera_changed.connect(self._on_camera_change)
        self.render_widget.key_down.connect(self._on_key_down)
        self.render_widget.key_up.connect(self._on_key_up)

        controls = self._build_controls()
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(370)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.render_widget)
        splitter.setSizes([370, 1270])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        self.setCentralWidget(splitter)

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        g = QGroupBox(title)
        f = QFormLayout(); f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        g.setLayout(f)
        return g, f

    def _build_controls(self) -> QWidget:
        w   = QWidget()
        vbl = QVBoxLayout(w); vbl.setContentsMargins(4, 4, 4, 4)

        # ── Status ─────────────────────────────────────────────────────────────
        g, f = self._group("Status")
        self._fps_label   = QLabel("--")
        self._gpu_label   = QLabel("--")
        self._splat_label = QLabel("--")
        f.addRow("FPS:",    self._fps_label)
        f.addRow("Splats:", self._splat_label)
        f.addRow("VRAM:",   self._gpu_label)

        overlay_row = QWidget(); ol = QHBoxLayout(overlay_row); ol.setContentsMargins(0, 0, 0, 0)
        fps_cb   = QCheckBox("FPS graph");   fps_cb.setChecked(self._show_fps_overlay)
        splat_cb = QCheckBox("Splat graph"); splat_cb.setChecked(self._show_splat_overlay)
        fps_cb.toggled.connect(lambda v: setattr(self.render_widget, '_show_fps_overlay',   v))
        splat_cb.toggled.connect(lambda v: setattr(self.render_widget, '_show_splat_overlay', v))
        ol.addWidget(fps_cb); ol.addWidget(splat_cb)
        f.addRow("Overlays:", overlay_row)
        vbl.addWidget(g)

        # ── Camera ─────────────────────────────────────────────────────────────
        g, f = self._group("Camera")

        f.addRow("World Up:", _combo(UP_AXIS_OPTIONS, self._cfg["world_up"], self._on_world_up_changed))

        f.addRow("FOV:",
            _slider_spin(10, 120, self.fov_deg, self._set_fov))

        f.addRow("Move Speed:",
            _slider_spin(0.1, 5.0, self._move_speed,
                         lambda v: setattr(self, '_move_speed', v),
                         is_float=True, decimals=2, step=0.05))
        f.addRow("Orbit Speed:",
            _slider_spin(0.1, 5.0, self._orbit_speed,
                         lambda v: setattr(self, '_orbit_speed', v),
                         is_float=True, decimals=2, step=0.05))

        # Mouse orbit inversion (independent of keyboard)
        mouse_row = QWidget(); ml = QHBoxLayout(mouse_row); ml.setContentsMargins(0,0,0,0)
        minv_x = QCheckBox("Inv X"); minv_x.setChecked(self._cfg["mouse_inv_x"])
        minv_y = QCheckBox("Inv Y"); minv_y.setChecked(self._cfg["mouse_inv_y"])
        minv_x.toggled.connect(lambda v: setattr(self.render_widget, 'mouse_inv_x', v))
        minv_y.toggled.connect(lambda v: setattr(self.render_widget, 'mouse_inv_y', v))
        ml.addWidget(minv_x); ml.addWidget(minv_y)
        f.addRow("Mouse Inv:", mouse_row)

        # Keyboard rotation inversion (arrow keys only)
        kb_row = QWidget(); kl = QHBoxLayout(kb_row); kl.setContentsMargins(0,0,0,0)
        kinv_x = QCheckBox("Inv X"); kinv_x.setChecked(self._cfg["kb_inv_x"])
        kinv_y = QCheckBox("Inv Y"); kinv_y.setChecked(self._cfg["kb_inv_y"])
        kinv_x.toggled.connect(lambda v: setattr(self, '_kb_inv_x', v))
        kinv_y.toggled.connect(lambda v: setattr(self, '_kb_inv_y', v))
        kl.addWidget(kinv_x); kl.addWidget(kinv_y)
        f.addRow("KB Rot Inv:", kb_row)

        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self._reset_camera)
        f.addRow(btn_reset)
        hint = QLabel("LMB: orbit  |  RMB: pan  |  Scroll: zoom\n"
                       "WASD: translate  |  Arrows: FPS look  |  R: reset")
        hint.setWordWrap(True)
        f.addRow(hint)
        vbl.addWidget(g)

        # ── Resolution ─────────────────────────────────────────────────────────
        g, f = self._group("Resolution")

        # Spinboxes first (preset combo fires on connect and needs them to exist)
        self._res_width_spin = QSpinBox()
        self._res_width_spin.setRange(2, 7680); self._res_width_spin.setSingleStep(2)
        self._res_width_spin.setValue(self._render_w)

        self._res_height_spin = QSpinBox()
        self._res_height_spin.setRange(2, 4320); self._res_height_spin.setSingleStep(2)
        self._res_height_spin.setValue(self._render_h)

        # AR combo + custom spinbox + Lock AR
        self._ar_combo = QComboBox()
        self._ar_combo.addItems(AR_RATIO_OPTIONS)

        self._ar_custom_spin = QDoubleSpinBox()
        self._ar_custom_spin.setRange(0.1, 10.0); self._ar_custom_spin.setDecimals(3)
        self._ar_custom_spin.setSingleStep(0.001); self._ar_custom_spin.setFixedWidth(72)
        self._ar_custom_spin.setValue(self._aspect_ratio)
        self._ar_custom_spin.setEnabled(False)    # enabled only for Custom AR

        self._lock_ar_cb = QCheckBox("Lock AR")
        self._lock_ar_cb.setChecked(self._lock_ar)
        self._lock_ar_cb.toggled.connect(lambda v: setattr(self, '_lock_ar', v))

        # Preset combo — connect after spinboxes exist
        preset_combo = QComboBox()
        preset_combo.addItems(list(RESOLUTION_PRESETS.keys()))
        preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_combo.setCurrentText("720p (HD)")           # fires safely; we override below

        # Override spinbox values from config (block signals to avoid feedback)
        for spin, val in [(self._res_width_spin, self._render_w),
                          (self._res_height_spin, self._render_h)]:
            spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)

        # Connect spinbox / AR signals after preset initialised values
        self._res_width_spin.valueChanged.connect(self._on_width_changed)
        self._res_height_spin.valueChanged.connect(self._on_height_changed)
        self._ar_combo.currentTextChanged.connect(self._on_ar_combo_changed)
        self._ar_custom_spin.valueChanged.connect(self._on_ar_custom_changed)
        self._ar_combo.setCurrentText(self._cfg["ar_preset"])  # fires safely

        f.addRow("Preset:", preset_combo)
        f.addRow("Width:",  self._res_width_spin)
        f.addRow("Height:", self._res_height_spin)
        ar_row = QWidget(); al = QHBoxLayout(ar_row); al.setContentsMargins(0,0,0,0)
        al.addWidget(self._ar_combo, 2)
        al.addWidget(self._ar_custom_spin, 1)
        al.addWidget(self._lock_ar_cb, 1)
        f.addRow("Aspect:", ar_row)
        vbl.addWidget(g)

        # ── Render Options ──────────────────────────────────────────────────────
        g, f = self._group("Render Options")
        f.addRow("Type:", _combo(RENDER_TYPES, self.render_type,
            lambda v: self._set("render_type", v)))
        f.addRow("Depth Ratio:",
            _slider_spin(0.0, 1.0, self.depth_ratio,
                lambda v: self._set("depth_ratio", v),
                is_float=True, decimals=2, step=0.05))
        split_cb = QCheckBox()
        split_cb.setChecked(self.split_enabled)
        split_cb.toggled.connect(lambda v: self._set("split_enabled", v))
        f.addRow("Split View:", split_cb)
        f.addRow("Split Pos:",
            _slider_spin(0.0, 1.0, self.split_pos,
                lambda v: self._set("split_pos", v),
                is_float=True, decimals=2, step=0.01))
        f.addRow("Left:",  _combo(RENDER_TYPES, self.render_type1,
            lambda v: self._set("render_type1", v)))
        f.addRow("Right:", _combo(RENDER_TYPES, self.render_type2,
            lambda v: self._set("render_type2", v)))
        vbl.addWidget(g)

        # ── Gaussian Model ──────────────────────────────────────────────────────
        g, f = self._group("Gaussian Model")
        if self._max_sh > 0:
            f.addRow("SH Degree:",
                _slider_spin(0, self._max_sh, self.sh_degree,
                    lambda v: self._set("sh_degree", int(v))))
        f.addRow("Opacity Thr:",
            _slider_spin(0.0, 0.5, self.opacity_thresh,
                lambda v: self._set("opacity_thresh", v),
                is_float=True, decimals=3, step=0.005))
        f.addRow("Sparsity:",
            _slider_spin(1, 10, self.sparsity,
                lambda v: self._set("sparsity", int(v))))
        f.addRow("Scale:",
            _slider_spin(0.1, 2.0, self.scaling_mod,
                lambda v: self._set("scaling_mod", v),
                is_float=True, decimals=2, step=0.05))
        ptc_cb = QCheckBox()
        ptc_cb.setChecked(self.show_ptc)
        ptc_cb.toggled.connect(lambda v: self._set("show_ptc", v))
        f.addRow("Pointcloud:", ptc_cb)
        disk_cb = QCheckBox("disk mode")
        disk_cb.setChecked(self.surfel_disk)
        disk_cb.toggled.connect(lambda v: self._set("surfel_disk", v))
        f.addRow("", disk_cb)
        f.addRow("Point Size:",
            _slider_spin(0.001, 0.1, self.point_size,
                lambda v: self._set("point_size", v),
                is_float=True, decimals=3, step=0.001))
        vbl.addWidget(g)

        # ── Crop Box ────────────────────────────────────────────────────────────
        g, f = self._group("Crop Box")
        crop_cb = QCheckBox()
        crop_cb.setChecked(self.crop_enabled)
        crop_cb.toggled.connect(lambda v: self._set("crop_enabled", v))
        f.addRow("Enable:", crop_cb)
        for ax, attr in [("X", "crop_x"), ("Y", "crop_y"), ("Z", "crop_z")]:
            vals = getattr(self, attr)
            f.addRow(f"{ax} min:",
                _slider_spin(-16.0, 16.0, vals[0],
                    lambda v, a=attr: self._set_crop(a, 0, v),
                    is_float=True, decimals=1, step=0.1))
            f.addRow(f"{ax} max:",
                _slider_spin(-16.0, 16.0, vals[1],
                    lambda v, a=attr: self._set_crop(a, 1, v),
                    is_float=True, decimals=1, step=0.1))
        vbl.addWidget(g)

        vbl.addStretch()
        return w

    # ── Settings helpers ───────────────────────────────────────────────────────

    def _set(self, attr: str, val):
        setattr(self, attr, val)
        self._render_trig.set()

    def _set_fov(self, val: float):
        self.fov_deg = float(val)
        self._render_trig.set()

    def _set_crop(self, attr: str, idx: int, val: float):
        getattr(self, attr)[idx] = val
        self._render_trig.set()

    def _on_camera_change(self):
        self._render_trig.set()

    def _reset_camera(self):
        self.camera.look_at  = np.array([0., 0., 0.])
        self.camera.distance = 5.0
        self.camera.yaw      = 0.0
        self.camera.pitch    = 0.3
        self._render_trig.set()

    def _on_world_up_changed(self, text: str):
        self.camera.world_up = UP_AXIS_VECTORS[text].copy()
        self._render_trig.set()

    # ── Resolution callbacks ───────────────────────────────────────────────────

    def _on_width_changed(self, val: int):
        self._render_w = val
        if self._lock_ar:
            new_h = max(2, round(val / self._aspect_ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self._render_h = new_h
        else:
            self._aspect_ratio = val / max(self._render_h, 1)
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            self._ar_combo.blockSignals(True)
            self._ar_combo.setCurrentText("Custom")
            self._ar_combo.blockSignals(False)
            self._ar_custom_spin.setEnabled(True)
        self._render_trig.set()

    def _on_height_changed(self, val: int):
        self._render_h = val
        if self._lock_ar:
            new_w = max(2, round(val * self._aspect_ratio))
            self._res_width_spin.blockSignals(True)
            self._res_width_spin.setValue(new_w)
            self._res_width_spin.blockSignals(False)
            self._render_w = new_w
        else:
            self._aspect_ratio = max(self._render_w, 1) / val
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            self._ar_combo.blockSignals(True)
            self._ar_combo.setCurrentText("Custom")
            self._ar_combo.blockSignals(False)
            self._ar_custom_spin.setEnabled(True)
        self._render_trig.set()

    def _on_preset_changed(self, text: str):
        wh = RESOLUTION_PRESETS.get(text)
        if wh is None:
            return
        w, h = wh
        self._render_w = w; self._render_h = h
        for spin, val in [(self._res_width_spin, w), (self._res_height_spin, h)]:
            spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)
        self._render_trig.set()

    def _on_ar_combo_changed(self, text: str):
        ratio = AR_RATIO_VALUES.get(text)
        if ratio is None:                       # Custom selected
            self._ar_custom_spin.setEnabled(True)
            self._ar_custom_spin.blockSignals(True)
            self._ar_custom_spin.setValue(self._aspect_ratio)
            self._ar_custom_spin.blockSignals(False)
            return
        self._ar_custom_spin.setEnabled(False)
        self._ar_custom_spin.blockSignals(True)
        self._ar_custom_spin.setValue(ratio)
        self._ar_custom_spin.blockSignals(False)
        self._aspect_ratio = ratio
        if self._lock_ar:
            new_h = max(2, round(self._render_w / ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self._render_h = new_h
        self._render_trig.set()

    def _on_ar_custom_changed(self, val: float):
        self._aspect_ratio = max(0.1, val)
        if self._lock_ar:
            new_h = max(2, round(self._render_w / self._aspect_ratio))
            self._res_height_spin.blockSignals(True)
            self._res_height_spin.setValue(new_h)
            self._res_height_spin.blockSignals(False)
            self._render_h = new_h
        self._render_trig.set()

    # ── Keyboard navigation ────────────────────────────────────────────────────

    def _on_key_down(self, key: int):
        if key == -1:
            self._keys_held.clear(); return
        if key == Qt.Key_R:
            self._reset_camera(); return
        self._keys_held.add(key)

    def _on_key_up(self, key: int):
        if key == -1:
            self._keys_held.clear(); return
        self._keys_held.discard(key)

    def _kb_tick(self):
        k = self._keys_held
        if not k:
            return

        move_speed  = self.camera.distance * 0.02 * self._move_speed
        orbit_speed = 0.02 * self._orbit_speed

        fwd_d = right_d = up_d = dyaw = dpitch = 0.0

        # WASD: translate look_at horizontally — NOT affected by inversion
        if Qt.Key_W in k: fwd_d   += move_speed
        if Qt.Key_S in k: fwd_d   -= move_speed
        if Qt.Key_A in k: right_d -= move_speed
        if Qt.Key_D in k: right_d += move_speed

        # E/Q: translate along world_up (up / down)
        if Qt.Key_E in k: up_d += move_speed
        if Qt.Key_Q in k: up_d -= move_speed

        # Arrow keys: FPS look — inversion applies to rotation only
        if Qt.Key_Left  in k: dyaw   -= orbit_speed
        if Qt.Key_Right in k: dyaw   += orbit_speed
        if Qt.Key_Up    in k: dpitch += orbit_speed
        if Qt.Key_Down  in k: dpitch -= orbit_speed

        if self._kb_inv_x: dyaw   = -dyaw
        if self._kb_inv_y: dpitch = -dpitch

        changed = False
        if fwd_d or right_d:
            self.camera.move(fwd_d, right_d); changed = True
        if up_d:
            self.camera.translate_up(up_d); changed = True
        if dyaw or dpitch:
            self.camera.fps_look(dyaw, dpitch); changed = True
        if changed:
            self._render_trig.set()

    # ── Camera construction ────────────────────────────────────────────────────

    def _build_camera(self, W: int, H: int) -> Cameras:
        fov_rad = math.radians(self.fov_deg)
        fx = torch.tensor([fov2focal(fov_rad, H)], dtype=torch.float)
        R, T = self.camera.build_RT(self.cam_tf)
        return Cameras(
            R=R.unsqueeze(0), T=T.unsqueeze(0), fx=fx, fy=fx,
            cx=torch.tensor([W // 2], dtype=torch.int),
            cy=torch.tensor([H // 2], dtype=torch.int),
            width=torch.tensor([W],   dtype=torch.int),
            height=torch.tensor([H],  dtype=torch.int),
            appearance_id=torch.tensor([0], dtype=torch.int),
            normalized_appearance_id=torch.tensor([0.], dtype=torch.float),
            time=torch.tensor([0.], dtype=torch.float),
            distortion_params=None,
            camera_type=torch.tensor([0], dtype=torch.int),
        )[0].to_device(self.device)

    # ── Render thread ──────────────────────────────────────────────────────────

    def _start_render_thread(self):
        threading.Thread(target=self._render_loop, daemon=True).start()

    def _render_loop(self):
        while self._running:
            self._render_trig.wait(timeout=0.2)
            self._render_trig.clear()
            if not self._running:
                break

            W = max(self._render_w, 2)
            H = max(self._render_h, 2)
            valid_range = (self.crop_x, self.crop_y, self.crop_z) if self.crop_enabled else None

            t0 = time.perf_counter()
            try:
                cam = self._build_camera(W, H)
                with torch.no_grad():
                    image = self.renderer.get_outputs(
                        cam,
                        valid_range       = valid_range,
                        split             = self.split_enabled,
                        slider            = self.split_pos,
                        active_sh_degree  = self.sh_degree,
                        scaling_modifier  = self.scaling_mod,
                        sparsity          = self.sparsity,
                        opacity_threshold = self.opacity_thresh,
                        depth_ratio       = self.depth_ratio,
                        render_type       = RENDER_TYPE_MAP[self.render_type],
                        render_type1      = RENDER_TYPE_MAP[self.render_type1],
                        render_type2      = RENDER_TYPE_MAP[self.render_type2],
                        show_ptc          = self.show_ptc and not self.surfel_disk,
                        show_disk         = self.show_ptc and self.surfel_disk,
                        point_size        = self.point_size,
                    )
            except Exception:
                import traceback; traceback.print_exc()
                continue

            img_np      = (image.clamp(0., 1.)
                           .permute(1, 2, 0).mul(255).byte().cpu().numpy())
            dt          = time.perf_counter() - t0
            fps_val     = 1.0 / dt if dt > 0 else 0.0
            fps_str     = f"{fps_val:.1f} fps" if dt > 0 else "--"
            n_splats    = self.renderer.last_visible_count
            splat_str   = _fmt_splats(n_splats)
            used        = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
            gpu_str     = f"{used / 1024**2:.0f} / {self._total_vram:.0f} MB"

            with self._frame_lock:
                self._frame_slot = (img_np, fps_str, gpu_str, fps_val, n_splats, splat_str)

    # ── Frame polling ──────────────────────────────────────────────────────────

    def _poll_frame(self):
        with self._frame_lock:
            slot = self._frame_slot
            self._frame_slot = None
        if slot is None:
            return
        img_np, fps_str, gpu_str, fps_val, n_splats, splat_str = slot
        self.render_widget.set_frame(img_np)
        if fps_val > 0:
            self.render_widget.update_fps(fps_val)
        if n_splats > 0:
            self.render_widget.update_splat_count(n_splats)
        self._fps_label.setText(fps_str)
        self._splat_label.setText(splat_str)
        self._gpu_label.setText(gpu_str)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def closeEvent(self, e):
        self._running = False
        self._render_trig.set()
        self._cfg.update({
            "fov_deg":           self.fov_deg,
            "move_speed":        self._move_speed,
            "orbit_speed":       self._orbit_speed,
            "mouse_inv_x":       self.render_widget.mouse_inv_x,
            "mouse_inv_y":       self.render_widget.mouse_inv_y,
            "kb_inv_x":          self._kb_inv_x,
            "kb_inv_y":          self._kb_inv_y,
            "world_up":          next((k for k, v in UP_AXIS_VECTORS.items()
                                       if np.allclose(v, self.camera.world_up)), "+Z"),
            "render_w":          self._render_w,
            "render_h":          self._render_h,
            "ar_preset":         self._ar_combo.currentText(),
            "lock_ar":           self._lock_ar,
            "depth_ratio":       self.depth_ratio,
            "active_sh_degree":  self.sh_degree,
            "opacity_thresh":    self.opacity_thresh,
            "sparsity":          self.sparsity,
            "scaling_mod":       self.scaling_mod,
            "point_size":        self.point_size,
            "render_type":       self.render_type,
            "show_fps_overlay":  self.render_widget._show_fps_overlay,
            "show_splat_overlay": self.render_widget._show_splat_overlay,
        })
        _save_config(self._cfg)
        super().closeEvent(e)


# ── Entry points ───────────────────────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2DGS Native Viewer")
    p.add_argument("ply_path",       type=str)
    p.add_argument("--sh_degree",    "--sh-degree", type=int, default=-1)
    p.add_argument("--no-culling",   action="store_true", default=False)
    p.add_argument("--no-profiling", action="store_true", default=False)
    p.add_argument("--build-index",  action="store_true", default=False)
    p.add_argument("--leaf-max",     type=int, default=5000)
    p.add_argument("--fp16-load",    action="store_true", default=False)
    return p


def run_local_viewer(ply_path: str, args) -> None:
    """Launch the Qt GUI.  Called either from __main__ or from viewer.py."""
    app = QApplication.instance() or QApplication(sys.argv)
    QLocale.setDefault(QLocale(QLocale.C))   # force dot as decimal separator
    app.setStyle("Fusion")
    viewer = LocalViewer(ply_path, args)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    _args = _make_parser().parse_args()
    run_local_viewer(_args.ply_path, _args)
