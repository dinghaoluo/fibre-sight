'''
Created on 12 May 2026

zooming, panning, and ROI display helpers for the workbench

@author: Dinghao Luo
'''

#%% imports
import colorsys

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication


#%% colours
def generate_distinct_colours(n):
    colours = []
    if n <= 0:
        return colours

    hue = 0.0
    # walking around the hue circle keeps neighbouring ROI labels distinct
    golden_ratio = 0.61803398875
    for _ in range(n):
        hue = (hue + golden_ratio) % 1.0
        colours.append(colorsys.hsv_to_rgb(hue, 0.68, 0.92))
    return colours


def normalise_for_display(
        image,
        black_percentile=1.0,
        white_percentile=99.7,
        ):
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)

    try:
        black_percentile = float(black_percentile)
        white_percentile = float(white_percentile)
    except (TypeError, ValueError):
        black_percentile, white_percentile = 1.0, 99.7

    if not np.isfinite(black_percentile):
        black_percentile = 1.0
    if not np.isfinite(white_percentile):
        white_percentile = 99.7

    black_percentile = np.clip(black_percentile, 0.0, 100.0)
    white_percentile = np.clip(white_percentile, 0.0, 100.0)
    if black_percentile > white_percentile:
        black_percentile, white_percentile = (
            white_percentile,
            black_percentile,
            )

    low, high = np.percentile(
        image[finite],
        [black_percentile, white_percentile],
        )
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    out = np.zeros_like(image, dtype=np.float32)
    out[finite] = (image[finite] - low) / (high - low)
    np.clip(out, 0, 1, out=out)
    return out


def squeeze_image(image):
    image = np.asarray(image)
    if image.ndim > 2:
        image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f'expected a 2D image, got shape {image.shape}')
    return image


#%% canvas
class ZoomableCanvas(FigureCanvas):
    _MAX_MAGNIFICATION = 32.0
    _BUTTON_ZOOM_FACTOR = 1.2

    def __init__(self, figure, ax):
        super().__init__(figure)
        self.ax = ax
        self._drag_start_pos = None
        self._drag_start_display = None
        self._drag_start_xlim = None
        self._drag_start_ylim = None
        self._is_panning = False
        self._release_was_dragged = False
        self._cursor_before_pan = None

    @property
    def release_was_dragged(self):
        return self._release_was_dragged

    def consume_dragged_release(self):
        dragged = self._release_was_dragged
        self._release_was_dragged = False
        return dragged

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.ax.images:
            self._drag_start_pos = event.pos()
            self._drag_start_display = self.mouseEventCoords(event)
            self._drag_start_xlim = self.ax.get_xlim()
            self._drag_start_ylim = self.ax.get_ylim()
            self._is_panning = False
            self._release_was_dragged = False
            self._cursor_before_pan = self.cursor()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self._drag_start_pos is not None:
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if not self._is_panning:
                self._is_panning = distance >= QApplication.startDragDistance()
                if self._is_panning:
                    self._release_was_dragged = True
                    self.setCursor(Qt.ClosedHandCursor)

        if self._is_panning:
            self.pan_view(event)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start_pos is not None:
            self._release_was_dragged = self._is_panning
            if self._is_panning and self._cursor_before_pan is not None:
                self.setCursor(self._cursor_before_pan)
            self._drag_start_pos = None
            self._drag_start_display = None
            self._drag_start_xlim = None
            self._drag_start_ylim = None
            self._is_panning = False
            self._cursor_before_pan = None

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if not self.ax.images:
            return

        x_display, y_display = self.mouseEventCoords(event)
        if not self.ax.bbox.contains(x_display, y_display):
            return

        pixel_delta = event.pixelDelta().y()
        delta = pixel_delta if pixel_delta else event.angleDelta().y()
        if not delta:
            return

        x_data, y_data = self.ax.transData.inverted().transform(
            (x_display, y_display),
            )
        delta = np.clip(float(delta), -2400.0, 2400.0)
        scale = np.exp(-delta * np.log(self._BUTTON_ZOOM_FACTOR) / 120.0)
        self._zoom_at(scale, x_data, y_data)
        event.accept()

    def pan_view(self, event):
        if self._drag_start_display is None:
            return

        x_display, y_display = self.mouseEventCoords(event)
        dx = x_display - self._drag_start_display[0]
        dy = y_display - self._drag_start_display[1]
        if self.ax.bbox.width <= 0 or self.ax.bbox.height <= 0:
            return

        x_span = self._drag_start_xlim[1] - self._drag_start_xlim[0]
        y_span = self._drag_start_ylim[1] - self._drag_start_ylim[0]
        dx_data = dx * x_span / self.ax.bbox.width
        dy_data = dy * y_span / self.ax.bbox.height
        xlim = (
            self._drag_start_xlim[0] - dx_data,
            self._drag_start_xlim[1] - dx_data,
            )
        ylim = (
            self._drag_start_ylim[0] - dy_data,
            self._drag_start_ylim[1] - dy_data,
            )
        self._set_clamped_view(xlim, ylim)

    def _image_limits(self):
        if not self.ax.images:
            return None

        left, right, bottom, top = self.ax.images[0].get_extent()
        limits = np.asarray([left, right, bottom, top], dtype=float)
        if not np.all(np.isfinite(limits)):
            return None
        if left == right or bottom == top:
            return None
        return (float(left), float(right)), (float(bottom), float(top))

    @staticmethod
    def _clamp_interval(limits, bounds):
        start, end = (float(value) for value in limits)
        bound_start, bound_end = (float(value) for value in bounds)
        direction = 1.0 if end >= start else -1.0
        width = min(abs(end - start), abs(bound_end - bound_start))
        bound_low, bound_high = sorted((bound_start, bound_end))
        low = min(start, end)
        low = min(max(low, bound_low), bound_high - width)
        high = low + width
        if direction > 0:
            return low, high
        return high, low

    def _set_clamped_view(self, xlim, ylim):
        image_limits = self._image_limits()
        if image_limits is None:
            return False

        xlim = self._clamp_interval(xlim, image_limits[0])
        ylim = self._clamp_interval(ylim, image_limits[1])
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.draw_idle()
        return True

    def _zoom_at(self, scale, x_data, y_data):
        image_limits = self._image_limits()
        if image_limits is None or not np.isfinite(scale) or scale <= 0:
            return False

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        if x_span == 0 or y_span == 0:
            return False

        fit_x_span = image_limits[0][1] - image_limits[0][0]
        fit_y_span = image_limits[1][1] - image_limits[1][0]
        min_x_width = abs(fit_x_span) / self._MAX_MAGNIFICATION
        min_y_width = abs(fit_y_span) / self._MAX_MAGNIFICATION
        x_width = np.clip(
            abs(x_span) * scale,
            min_x_width,
            abs(fit_x_span),
            )
        y_width = np.clip(
            abs(y_span) * scale,
            min_y_width,
            abs(fit_y_span),
            )
        new_x_span = np.copysign(x_width, x_span)
        new_y_span = np.copysign(y_width, y_span)

        rel_x = (x_data - xlim[0]) / x_span
        rel_y = (y_data - ylim[0]) / y_span
        new_xlim = (
            x_data - new_x_span * rel_x,
            x_data + new_x_span * (1 - rel_x),
            )
        new_ylim = (
            y_data - new_y_span * rel_y,
            y_data + new_y_span * (1 - rel_y),
            )
        return self._set_clamped_view(new_xlim, new_ylim)

    def zoom_in(self, factor=_BUTTON_ZOOM_FACTOR):
        if not np.isfinite(factor) or factor <= 0:
            return False
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        return self._zoom_at(
            1.0 / factor,
            sum(xlim) / 2,
            sum(ylim) / 2,
            )

    def zoom_out(self, factor=_BUTTON_ZOOM_FACTOR):
        if not np.isfinite(factor) or factor <= 0:
            return False
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        return self._zoom_at(
            factor,
            sum(xlim) / 2,
            sum(ylim) / 2,
            )

    def centre_on(self, x_data, y_data):
        if not np.isfinite(x_data) or not np.isfinite(y_data):
            return False
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        new_xlim = (
            x_data - x_span / 2,
            x_data + x_span / 2,
            )
        new_ylim = (
            y_data - y_span / 2,
            y_data + y_span / 2,
            )
        return self._set_clamped_view(new_xlim, new_ylim)

    def fit_to_image(self):
        image_limits = self._image_limits()
        if image_limits is None:
            return False

        self.ax.set_xlim(image_limits[0])
        self.ax.set_ylim(image_limits[1])
        self.draw_idle()
        return True

    def reset_view(self):
        return self.fit_to_image()
