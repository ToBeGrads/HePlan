"""
RulerTool — Interactive distance measurement overlay for pyqtgraph PlotWidgets.

Features:
  - Click once to anchor the start point, then a live preview line follows the
    cursor showing the real-time distance in mm.
  - Click again to confirm the measurement.
  - After placement, drag either endpoint to edit the measurement in real time.
  - Measurements are stored per-slice and persist when navigating away and back.
  - Toggle visibility and clear measurements per-slice or globally.

Usage:
    ruler = RulerTool(plot_widget, spacing_xy=(1.0, 1.0))
    ruler.set_active(True)
    ruler.set_slice(42)
"""

import math
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QPointF


class _DraggableEndpoint(pg.ScatterPlotItem):
    """A single draggable circle that emits position changes."""

    def __init__(self, x, y, color, on_moved, size=10):
        super().__init__(
            [x], [y],
            size=size, brush=pg.mkBrush(color),
            pen=pg.mkPen('#11111b', width=1.5),
            symbol='o'
        )
        self.setAcceptHoverEvents(True)
        self._dragging = False
        self._on_moved = on_moved  # callback(x, y)
        self.pos_x = x
        self.pos_y = y

    def update_pos(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.setData([x], [y])

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return

        ev.accept()
        if ev.isStart():
            self._dragging = True
        elif ev.isFinish():
            self._dragging = False

        pos = ev.pos()
        self.pos_x = pos.x()
        self.pos_y = pos.y()
        self.setData([self.pos_x], [self.pos_y])
        self._on_moved(self.pos_x, self.pos_y)

    def hoverEvent(self, ev):
        if ev.isExit():
            self.setPen(pg.mkPen('#11111b', width=1.5))
            self.setSize(10)
        else:
            self.setPen(pg.mkPen('#ffffff', width=2))
            self.setSize(13)


class _MeasurementLine:
    """A complete measurement: two draggable endpoints + line + label."""

    def __init__(self, plot_widget, x1, y1, x2, y2, spacing_x, spacing_y, color):
        self.plot_widget = plot_widget
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.color = color

        # Line
        self.line = pg.PlotDataItem(
            [x1, x2], [y1, y2],
            pen=pg.mkPen(color=color, width=2, style=Qt.SolidLine)
        )
        plot_widget.addItem(self.line)

        # Label
        self.label = pg.TextItem(text="", color=color, anchor=(0.5, 1.2))
        self.label.setFont(pg.QtGui.QFont("Arial", 9, pg.QtGui.QFont.Bold))
        plot_widget.addItem(self.label)

        # Draggable endpoints
        self.p1 = _DraggableEndpoint(x1, y1, color, self._on_p1_moved)
        self.p2 = _DraggableEndpoint(x2, y2, color, self._on_p2_moved)
        plot_widget.addItem(self.p1)
        plot_widget.addItem(self.p2)

        self._refresh()

    def _on_p1_moved(self, x, y):
        self._refresh()

    def _on_p2_moved(self, x, y):
        self._refresh()

    def _refresh(self):
        """Recompute line, label position, and distance after any endpoint move."""
        x1, y1 = self.p1.pos_x, self.p1.pos_y
        x2, y2 = self.p2.pos_x, self.p2.pos_y

        self.line.setData([x1, x2], [y1, y2])

        dx_mm = (x2 - x1) * self.spacing_x
        dy_mm = (y2 - y1) * self.spacing_y
        dist = math.sqrt(dx_mm ** 2 + dy_mm ** 2)

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        self.label.setText(f"{dist:.1f} mm")
        self.label.setPos(mid_x, mid_y)

    def set_visible(self, visible):
        self.line.setVisible(visible)
        self.label.setVisible(visible)
        self.p1.setVisible(visible)
        self.p2.setVisible(visible)

    def remove(self):
        self.plot_widget.removeItem(self.line)
        self.plot_widget.removeItem(self.label)
        self.plot_widget.removeItem(self.p1)
        self.plot_widget.removeItem(self.p2)


class RulerTool:
    """
    Attaches to a pg.PlotWidget for interactive distance measurement.

    Parameters
    ----------
    plot_widget : pg.PlotWidget
        The pyqtgraph view to attach to.
    spacing_xy : tuple of (float, float)
        Pixel spacing in the two in-plane directions (mm per pixel).
    color : str
        Hex color for the measurement lines and labels.
    """

    def __init__(self, plot_widget, spacing_xy=(1.0, 1.0), color='#f9e2af'):
        self.plot_widget = plot_widget
        self.spacing_x = spacing_xy[0]
        self.spacing_y = spacing_xy[1]
        self.color = color

        self._active = False
        self._visible = True
        self._current_slice = 0

        # Per-slice storage: {slice_idx: [_MeasurementLine, ...]}
        self._measurements = {}

        # Live preview state
        self._placing = False       # True after first click, before second
        self._anchor_x = 0.0
        self._anchor_y = 0.0
        self._preview_line = None   # pg.PlotDataItem
        self._preview_label = None  # pg.TextItem
        self._preview_dot = None    # pg.ScatterPlotItem (anchor marker)

        # Mouse move proxy for live preview
        self._proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved
        )

        # Mouse click handler
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # ── Public API ──────────────────────────────────────────────

    def set_active(self, active):
        """Enable or disable measurement mode."""
        self._active = active
        if not active:
            self._cancel_placement()

    def is_active(self):
        return self._active

    def set_spacing(self, sx, sy):
        """Update the in-plane pixel spacing (mm per pixel)."""
        self.spacing_x = sx
        self.spacing_y = sy

    def set_slice(self, slice_idx):
        """Switch visible slice — hides old measurements, shows new ones."""
        for m in self._measurements.get(self._current_slice, []):
            m.set_visible(False)

        self._current_slice = slice_idx

        if self._visible:
            for m in self._measurements.get(self._current_slice, []):
                m.set_visible(True)

        # Cancel any in-progress placement when switching slices
        self._cancel_placement()

    def set_visible(self, visible):
        """Toggle visibility of all measurement lines globally."""
        self._visible = visible
        for m in self._measurements.get(self._current_slice, []):
            m.set_visible(visible)

    def clear_slice(self, slice_idx=None):
        """Remove all measurements for the given slice (default: current)."""
        if slice_idx is None:
            slice_idx = self._current_slice
        for m in self._measurements.get(slice_idx, []):
            m.remove()
        self._measurements[slice_idx] = []

    def clear_all(self):
        """Remove all measurements across all slices."""
        for sid in list(self._measurements.keys()):
            for m in self._measurements[sid]:
                m.remove()
        self._measurements.clear()
        self._cancel_placement()

    def update_spacing_for_view(self, patient_data, view_key):
        """
        Convenience: set spacing based on the PatientData and the view plane.

        Axial    → spacing[1] (row), spacing[2] (col)
        Coronal  → spacing[0] (slice), spacing[2] (col)
        Sagittal → spacing[0] (slice), spacing[1] (row)
        """
        sp = patient_data.spacing
        if view_key == 'axial':
            self.set_spacing(sp[2], sp[1])
        elif view_key == 'coronal':
            self.set_spacing(sp[2], sp[0])
        elif view_key == 'sagittal':
            self.set_spacing(sp[1], sp[0])

    # ── Private: Live Preview ───────────────────────────────────

    def _on_mouse_moved(self, args):
        """Update the preview line as the cursor moves after the first click."""
        if not self._placing:
            return

        pos = args[0]
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        cx, cy = mouse_point.x(), mouse_point.y()

        # Update preview line
        if self._preview_line is not None:
            self._preview_line.setData(
                [self._anchor_x, cx],
                [self._anchor_y, cy]
            )

        # Update preview label with live distance
        if self._preview_label is not None:
            dx_mm = (cx - self._anchor_x) * self.spacing_x
            dy_mm = (cy - self._anchor_y) * self.spacing_y
            dist = math.sqrt(dx_mm ** 2 + dy_mm ** 2)
            self._preview_label.setText(f"{dist:.1f} mm")
            self._preview_label.setPos(
                (self._anchor_x + cx) / 2,
                (self._anchor_y + cy) / 2
            )

    def _on_mouse_clicked(self, event):
        """Handle clicks: first click anchors, second click confirms."""
        if not self._active:
            return
        if event.button() != Qt.LeftButton:
            return

        pos = event.scenePos()
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()

        if not self._placing:
            # ── First click: anchor the start point ──
            self._placing = True
            self._anchor_x = x
            self._anchor_y = y

            # Create preview visuals
            self._preview_dot = pg.ScatterPlotItem(
                [x], [y], size=10,
                brush=pg.mkBrush(self.color),
                pen=pg.mkPen('#11111b', width=1.5),
                symbol='o'
            )
            self.plot_widget.addItem(self._preview_dot)

            self._preview_line = pg.PlotDataItem(
                [x, x], [y, y],
                pen=pg.mkPen(color=self.color, width=1.5, style=Qt.DashLine)
            )
            self.plot_widget.addItem(self._preview_line)

            self._preview_label = pg.TextItem(
                text="0.0 mm", color=self.color, anchor=(0.5, 1.2)
            )
            self._preview_label.setFont(pg.QtGui.QFont("Arial", 9, pg.QtGui.QFont.Bold))
            self._preview_label.setPos(x, y)
            self.plot_widget.addItem(self._preview_label)

        else:
            # ── Second click: finalize the measurement ──
            x1, y1 = self._anchor_x, self._anchor_y
            x2, y2 = x, y

            # Remove preview visuals
            self._remove_preview()
            self._placing = False

            # Create the persistent, editable measurement
            measurement = _MeasurementLine(
                self.plot_widget,
                x1, y1, x2, y2,
                self.spacing_x, self.spacing_y,
                self.color
            )

            if self._current_slice not in self._measurements:
                self._measurements[self._current_slice] = []
            self._measurements[self._current_slice].append(measurement)

    def _cancel_placement(self):
        """Cancel an in-progress measurement placement."""
        if self._placing:
            self._remove_preview()
            self._placing = False

    def _remove_preview(self):
        """Remove all temporary preview items from the view."""
        if self._preview_dot is not None:
            self.plot_widget.removeItem(self._preview_dot)
            self._preview_dot = None
        if self._preview_line is not None:
            self.plot_widget.removeItem(self._preview_line)
            self._preview_line = None
        if self._preview_label is not None:
            self.plot_widget.removeItem(self._preview_label)
            self._preview_label = None
