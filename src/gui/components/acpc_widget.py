"""
ACPCWidget — Multi-planar AC/PC landmark definition.

Layout  (2×2 views + sidebar)
──────────────────────────────
  ┌──────────┬──────────┐
  │  Axial   │ Coronal  │   T1 views (3-plane MPR)
  ├──────────┼──────────┤
  │ Sagittal │  T2 Axial│   T2 with transferred AC/PC
  └──────────┴──────────┘

Coordinate conventions
──────────────────────
  Axial   : np.flipud(vol[zi, :, :])  → display (col=vx, row=H-1-vy)
  Coronal : vol[:, yi, :]             → display (col=vx, row=vz)
  Sagittal: vol[:, :, xi]             → display (col=vy, row=vz)

Landmarks are stored as 3D voxel tuples (vz, vy, vx).
Each landmark has one draggable marker per T1 view.
Dragging in a view updates the two coordinates visible in that plane.
"""

import os
import json
import numpy as np
import pyqtgraph as pg

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QScrollArea, QFrame, QSizePolicy, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QCursor

# ── pyqtgraph global config ────────────────────────────────────────────────────
pg.setConfigOption('background', '#181825')
pg.setConfigOption('foreground', '#cdd6f4')
pg.setConfigOption('imageAxisOrder', 'row-major')

# ── Colours ────────────────────────────────────────────────────────────────────
_AC  = '#89dceb'   # cyan  – Anterior Commissure
_PC  = '#fab387'   # peach – Posterior Commissure
_LINE = '#cdd6f4'  # white-ish dashed AC–PC line

# ── Shared styles ──────────────────────────────────────────────────────────────
_GRP = (
    "QGroupBox{font-weight:bold;font-size:13px;color:#cdd6f4;"
    "border:1px solid #45475a;border-radius:6px;margin-top:12px;"
    "padding-top:18px;}"
    "QGroupBox::title{subcontrol-origin:margin;"
    "subcontrol-position:top left;padding:0 5px;}"
)
_BTN = (
    "QPushButton{background:#313244;color:#cdd6f4;padding:6px;"
    "border-radius:4px;font-size:12px;}"
    "QPushButton:disabled{background:#1e1e2e;color:#6c7086;}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level draggable dot
# ══════════════════════════════════════════════════════════════════════════════

class _DraggableDot(pg.ScatterPlotItem):
    """Draggable circle that calls on_moved(x, y) continuously."""

    def __init__(self, x: float, y: float, color: str, on_moved, size: int = 14):
        super().__init__(
            [x], [y], size=size,
            brush=pg.mkBrush(color),
            pen=pg.mkPen('#11111b', width=1.5),
            symbol='o',
        )
        self.pos_x = x
        self.pos_y = y
        self._on_moved = on_moved
        self.setAcceptHoverEvents(True)

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        p = ev.pos()
        self.pos_x, self.pos_y = p.x(), p.y()
        self.setData([self.pos_x], [self.pos_y])
        if self._on_moved:
            self._on_moved(self.pos_x, self.pos_y)

    def hoverEvent(self, ev):
        if ev.isExit():
            self.setSize(14)
            self.setPen(pg.mkPen('#11111b', width=1.5))
        else:
            self.setSize(18)
            self.setPen(pg.mkPen('#ffffff', width=2))


# ══════════════════════════════════════════════════════════════════════════════
#  Landmark marker (cross-hair + draggable dot + label)
# ══════════════════════════════════════════════════════════════════════════════

class _LandmarkMarker:
    """
    Draggable cross-hair marker for one landmark on a single view.

    Parameters
    ----------
    view       : pg.PlotWidget – the view to draw on
    x, y       : float         – initial display position
    color      : str           – hex colour
    label      : str           – 'AC' or 'PC'
    on_dragged : callable(x,y) – called every time the dot is dragged
    draggable  : bool          – if False the dot does not respond to drags
    """
    _ARM = 10   # cross-arm half-length in view units

    def __init__(
        self,
        view: pg.PlotWidget,
        x: float,
        y: float,
        color: str,
        label: str,
        on_dragged=None,
        draggable: bool = True,
    ):
        self.view = view
        pen = pg.mkPen(color, width=2)

        self._h   = pg.PlotDataItem(pen=pen)
        self._v   = pg.PlotDataItem(pen=pen)
        self._lbl = pg.TextItem(text=label, color=color, anchor=(-0.25, 1.35))
        self._lbl.setFont(QFont('Arial', 9, QFont.Bold))

        cb = (lambda px, py: (self._refresh_cross(px, py) or on_dragged(px, py))) \
             if draggable and on_dragged else \
             (lambda px, py: self._refresh_cross(px, py)) if draggable else None

        self._dot = _DraggableDot(x, y, color, cb if draggable else None)
        if not draggable:
            self._dot._on_moved = None

        for item in (self._h, self._v, self._dot, self._lbl):
            view.addItem(item)

        self._set(x, y)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _refresh_cross(self, x: float, y: float):
        a = self._ARM
        self._h.setData([x - a, x + a], [y, y])
        self._v.setData([x, x], [y - a, y + a])
        self._lbl.setPos(x, y)

    def _set(self, x: float, y: float):
        self._dot.pos_x = x
        self._dot.pos_y = y
        self._dot.setData([x], [y])
        self._refresh_cross(x, y)

    # ── public API ────────────────────────────────────────────────────────────

    def update_pos(self, x: float, y: float):
        """Reposition the marker without triggering the drag callback."""
        self._set(x, y)

    def set_visible(self, v: bool):
        for item in (self._h, self._v, self._dot, self._lbl):
            item.setVisible(v)

    def remove(self):
        for item in (self._h, self._v, self._dot, self._lbl):
            self.view.removeItem(item)


# ══════════════════════════════════════════════════════════════════════════════
#  Main widget
# ══════════════════════════════════════════════════════════════════════════════

class ACPCWidget(QWidget):

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(self, parent=None):
        super().__init__(parent)

        # Loaded volumes (shared from patient loader)
        self.volumes: dict = {}

        # Active T1
        self.t1_name = None
        self.t1_vol  = None

        # Active T2 (for the 4th panel)
        self.t2_name = None
        self.t2_vol  = None

        # Slice indices for the three T1 views and T2 axial
        self.slice_idx = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self.slice_t2  = 0

        # Landmarks: None  |  {'voxel': (vz,vy,vx),
        #                       'markers': {view_key: _LandmarkMarker}}
        self.ac: dict = None
        self.pc: dict = None

        # T2 markers (static, no drag)
        self._t2_ac_marker: _LandmarkMarker = None
        self._t2_pc_marker: _LandmarkMarker = None
        self._t2_line = None           # pg.PlotDataItem

        # Projected AC-PC lines in the three T1 views (always visible when both set)
        self._proj_lines: dict = {}    # view_key → pg.PlotDataItem

        # Placement mode
        self._placing: str = None      # 'ac' | 'pc' | None

        self._init_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 2×2 view grid ─────────────────────────────────────────────────────
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)

        self.views: dict = {}
        self.imgs:  dict = {}
        for key in ('axial', 'coronal', 'sagittal', 't2'):
            v = pg.PlotWidget()
            v.hideAxis('left')
            v.hideAxis('bottom')
            v.setMouseEnabled(x=True, y=True)
            v.setMenuEnabled(False)
            v.setAspectLocked(True, 1.0)
            v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            img = pg.ImageItem()
            v.addItem(img)
            self.views[key] = v
            self.imgs[key]  = img

        self._hdrs: dict = {}
        label_map = {
            'axial':    'Axial (T1)',
            'coronal':  'Coronal (T1)',
            'sagittal': 'Sagittal (T1)',
            't2':       'T2 — AC/PC Transfer Result',
        }
        for key, title in label_map.items():
            self._hdrs[key] = self._make_header(title, key)

        # Axial TL, Coronal TR, Sagittal BL, T2 BR
        grid.addWidget(self._hdrs['axial'],     0, 0)
        grid.addWidget(self.views['axial'],     1, 0)
        grid.addWidget(self._hdrs['coronal'],   0, 1)
        grid.addWidget(self.views['coronal'],   1, 1)
        grid.addWidget(self._hdrs['sagittal'],  2, 0)
        grid.addWidget(self.views['sagittal'],  3, 0)
        grid.addWidget(self._hdrs['t2'],        2, 1)
        grid.addWidget(self.views['t2'],        3, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(3, 1)

        root.addWidget(grid_w, 3)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color:#1e1e2e;border:none;")
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        sc = QWidget()
        sl = QVBoxLayout(sc)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        sl.addWidget(self._build_volumes_grp())
        sl.addWidget(self._build_placement_grp())
        sl.addWidget(self._build_points_grp())
        sl.addWidget(self._build_transfer_grp())
        sl.addStretch()

        scroll.setWidget(sc)
        sb_lay.addWidget(scroll)
        root.addWidget(sidebar, 1)

        # ── Wire T1 view clicks ───────────────────────────────────────────────
        for key in ('axial', 'coronal', 'sagittal'):
            self.views[key].scene().sigMouseClicked.connect(
                lambda ev, k=key: self._on_view_clicked(ev, k)
            )

        # ── Create projected AC-PC line items ─────────────────────────────────
        for key in ('axial', 'coronal', 'sagittal'):
            line = pg.PlotDataItem(
                pen=pg.mkPen(_LINE, width=1.5, style=Qt.DashLine)
            )
            line.setVisible(False)
            self.views[key].addItem(line)
            self._proj_lines[key] = line

    # ── Header factory ────────────────────────────────────────────────────────

    def _make_header(self, title: str, view_key: str) -> QWidget:
        hdr = QWidget()
        hdr.setStyleSheet("background-color:#1e1e2e;padding:4px;")
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(4, 2, 4, 2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            "color:#89b4fa;font-weight:bold;font-size:11px;background:transparent;"
        )
        lay.addWidget(lbl_title)
        lay.addStretch()

        btn_prev = QPushButton("◀")
        btn_prev.setFixedSize(20, 20)
        lbl_sl = QLabel("- / -")
        lbl_sl.setStyleSheet(
            "color:#a6adc8;font-size:10px;min-width:40px;"
            "background-color:#181825;padding:2px 6px;border-radius:3px;"
        )
        lbl_sl.setAlignment(Qt.AlignCenter)
        btn_next = QPushButton("▶")
        btn_next.setFixedSize(20, 20)

        btn_prev.clicked.connect(lambda _, k=view_key: self._step_slice(k, -1))
        btn_next.clicked.connect(lambda _, k=view_key: self._step_slice(k,  1))

        hdr.lbl_slice = lbl_sl
        lay.addWidget(btn_prev)
        lay.addWidget(lbl_sl)
        lay.addWidget(btn_next)
        return hdr

    # ── Sidebar group factories ───────────────────────────────────────────────

    def _build_volumes_grp(self) -> QGroupBox:
        grp = QGroupBox("📁 Loaded Volumes")
        grp.setStyleSheet(_GRP)
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setAlignment(Qt.AlignTop)

        self.tbl = QTableWidget()
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels(["Name", "Mod", "Shape"])
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setMaximumHeight(120)
        self.tbl.setStyleSheet(
            "QTableWidget{background:#1e1e2e;color:#cdd6f4;"
            "border:1px solid #45475a;border-radius:4px;}"
            "QHeaderView::section{background:#313244;color:#a6adc8;"
            "font-weight:bold;border:none;padding:4px;}"
        )
        self.tbl.itemSelectionChanged.connect(self._on_vol_selected)
        lay.addWidget(self.tbl)
        return grp

    def _build_placement_grp(self) -> QGroupBox:
        grp = QGroupBox("🖱 Landmark Placement")
        grp.setStyleSheet(_GRP)
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        hint = QLabel(
            "Click a button to enter placement mode — the cursor will change "
            "to a crosshair. Click any T1 view to drop the landmark. "
            "Drag markers to refine."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#a6adc8;font-size:10px;")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_ac = QPushButton("🔵  Place AC")
        self.btn_ac.setCheckable(True)
        self.btn_ac.setEnabled(False)
        self.btn_ac.setStyleSheet(_BTN)
        self.btn_ac.toggled.connect(lambda on: self._toggle_place('ac', on))
        btn_row.addWidget(self.btn_ac)

        self.btn_pc = QPushButton("🟠  Place PC")
        self.btn_pc.setCheckable(True)
        self.btn_pc.setEnabled(False)
        self.btn_pc.setStyleSheet(_BTN)
        self.btn_pc.toggled.connect(lambda on: self._toggle_place('pc', on))
        btn_row.addWidget(self.btn_pc)

        lay.addLayout(btn_row)

        btn_clr = QPushButton("🗑 Clear All")
        btn_clr.setStyleSheet(_BTN)
        btn_clr.clicked.connect(self._clear_all)
        lay.addWidget(btn_clr)
        return grp

    def _build_points_grp(self) -> QGroupBox:
        grp = QGroupBox("📍 Defined Points")
        grp.setStyleSheet(_GRP)
        g = QGridLayout(grp)
        g.setContentsMargins(10, 10, 10, 10)
        g.setSpacing(6)

        # AC
        lbl_a = QLabel("AC")
        lbl_a.setStyleSheet(f"color:{_AC};font-weight:bold;font-size:14px;")
        self.lbl_ac = QLabel("—")
        self.lbl_ac.setStyleSheet("color:#cdd6f4;font-size:10px;")
        self.lbl_ac.setWordWrap(True)
        btn_ca = QPushButton("✕")
        btn_ca.setFixedSize(22, 22)
        btn_ca.setStyleSheet(_BTN)
        btn_ca.clicked.connect(self._clear_ac)

        g.addWidget(lbl_a,      0, 0)
        g.addWidget(self.lbl_ac, 0, 1)
        g.addWidget(btn_ca,     0, 2)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#45475a;")
        g.addWidget(sep, 1, 0, 1, 3)

        # PC
        lbl_p = QLabel("PC")
        lbl_p.setStyleSheet(f"color:{_PC};font-weight:bold;font-size:14px;")
        self.lbl_pc = QLabel("—")
        self.lbl_pc.setStyleSheet("color:#cdd6f4;font-size:10px;")
        self.lbl_pc.setWordWrap(True)
        btn_cp = QPushButton("✕")
        btn_cp.setFixedSize(22, 22)
        btn_cp.setStyleSheet(_BTN)
        btn_cp.clicked.connect(self._clear_pc)

        g.addWidget(lbl_p,      2, 0)
        g.addWidget(self.lbl_pc, 2, 1)
        g.addWidget(btn_cp,     2, 2)

        g.setColumnStretch(1, 1)
        return grp

    def _build_transfer_grp(self) -> QGroupBox:
        grp = QGroupBox("🔄 Transfer to T2")
        grp.setStyleSheet(_GRP)
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        lbl = QLabel(
            "Select a T2 volume. The AC/PC points will be mapped into "
            "its voxel space using the DICOM affine matrices."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#a6adc8;font-size:10px;")
        lay.addWidget(lbl)

        self.cmb_t2 = QComboBox()
        self.cmb_t2.addItem("— select T2 volume —")
        self.cmb_t2.setStyleSheet(
            "background:#313244;color:#cdd6f4;padding:4px;border-radius:4px;"
        )
        self.cmb_t2.currentIndexChanged.connect(self._refresh_transfer_btn)
        lay.addWidget(self.cmb_t2)

        self.btn_transfer = QPushButton("▶ Transfer AC/PC → T2")
        self.btn_transfer.setEnabled(False)
        self.btn_transfer.setStyleSheet(
            "QPushButton{background:#a6e3a1;color:#11111b;font-weight:bold;"
            "padding:8px;border-radius:6px;}"
            "QPushButton:disabled{background:#45475a;color:#a6adc8;}"
        )
        self.btn_transfer.clicked.connect(self._run_transfer)
        lay.addWidget(self.btn_transfer)

        self.btn_save = QPushButton("💾 Save to Patient Folder")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "QPushButton{background:#89b4fa;color:#11111b;font-weight:bold;"
            "padding:8px;border-radius:6px;}"
            "QPushButton:disabled{background:#45475a;color:#a6adc8;}"
        )
        self.btn_save.clicked.connect(self._save)
        lay.addWidget(self.btn_save)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#a6adc8;font-size:10px;margin-top:4px;")
        lay.addWidget(self.lbl_status)
        return grp

    # ══════════════════════════════════════════════════════════════════════════
    #  Volume management
    # ══════════════════════════════════════════════════════════════════════════

    def add_volume(self, patient_data, vol_name: str):
        """Called by MainWindow whenever a new series is loaded."""
        if vol_name in self.volumes:
            return
        self.volumes[vol_name] = patient_data

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QTableWidgetItem(vol_name))
        self.tbl.setItem(r, 1, QTableWidgetItem(patient_data.modality))
        self.tbl.setItem(r, 2, QTableWidgetItem(str(patient_data.get_shape())))

        self.cmb_t2.addItem(vol_name)

    def _on_vol_selected(self):
        row = self.tbl.currentRow()
        if row < 0:
            return
        name = self.tbl.item(row, 0).text()
        self._activate_t1(name)

    def _activate_t1(self, name: str):
        if name not in self.volumes:
            return
        self.t1_name = name
        self.t1_vol  = self.volumes[name]

        z, y, x = self.t1_vol.get_shape()
        self.slice_idx = {'axial': z // 2, 'coronal': y // 2, 'sagittal': x // 2}

        self.btn_ac.setEnabled(True)
        self.btn_pc.setEnabled(True)
        
        if self.ac is None and self.pc is None:
            self._load_from_cache()

        # Re-map existing AC/PC points into the new volume's voxel space using physical world coordinates
        for key_name, entry in [('ac', self.ac), ('pc', self.pc)]:
            if entry is not None and 'world' in entry:
                world_xyz = entry['world']
                # world_to_voxel expects (x, y, z) and returns (z, y, x)
                new_voxel = self.t1_vol.world_to_voxel(world_xyz)
                new_voxel = tuple(int(round(v)) for v in new_voxel)
                self._set_landmark(key_name, new_voxel, update_world=False)

        self._render_all_t1()

    # ══════════════════════════════════════════════════════════════════════════
    #  Rendering
    # ══════════════════════════════════════════════════════════════════════════

    def _step_slice(self, key: str, delta: int):
        if key == 't2':
            if self.t2_vol is None:
                return
            z = self.t2_vol.get_shape()[0]
            self.slice_t2 = max(0, min(self.slice_t2 + delta, z - 1))
            self._render_t2()
            return

        if self.t1_vol is None:
            return
        z, y, x = self.t1_vol.get_shape()
        limits = {'axial': z, 'coronal': y, 'sagittal': x}
        self.slice_idx[key] = max(
            0, min(self.slice_idx[key] + delta, limits[key] - 1)
        )
        self._render_t1_view(key)
        self._refresh_marker_visibility()
        self._refresh_proj_lines()

    def _render_all_t1(self):
        for key in ('axial', 'coronal', 'sagittal'):
            self._render_t1_view(key)
        self._refresh_marker_visibility()
        self._refresh_proj_lines()

    def _render_t1_view(self, key: str):
        if self.t1_vol is None:
            return
        vol = self.t1_vol.volume
        z, y, x = self.t1_vol.get_shape()
        zi = self.slice_idx['axial']
        yi = self.slice_idx['coronal']
        xi = self.slice_idx['sagittal']

        if key == 'axial':
            slc = np.flipud(vol[zi, :, :])
            total, cur = z, zi
            s_col, s_row = self.t1_vol.spacing[2], self.t1_vol.spacing[1]
        elif key == 'coronal':
            slc = vol[:, yi, :]
            total, cur = y, yi
            s_col, s_row = self.t1_vol.spacing[2], self.t1_vol.spacing[0]
        else:
            slc = vol[:, :, xi]
            total, cur = x, xi
            s_col, s_row = self.t1_vol.spacing[1], self.t1_vol.spacing[0]

        self.imgs[key].setImage(slc, autoLevels=True)
        self.imgs[key].setTransform(pg.QtGui.QTransform().scale(s_col, s_row))
        self._hdrs[key].lbl_slice.setText(f"{cur + 1}/{total}")

    def _render_t2(self):
        if self.t2_vol is None:
            self.imgs['t2'].clear()
            self._hdrs['t2'].lbl_slice.setText("- / -")
            return
        z = self.t2_vol.get_shape()[0]
        self.slice_t2 = max(0, min(self.slice_t2, z - 1))
        slc = np.flipud(self.t2_vol.volume[self.slice_t2])
        self.imgs['t2'].setImage(slc, autoLevels=True)
        s_col, s_row = self.t2_vol.spacing[2], self.t2_vol.spacing[1]
        self.imgs['t2'].setTransform(pg.QtGui.QTransform().scale(s_col, s_row))
        self._hdrs['t2'].lbl_slice.setText(f"{self.slice_t2 + 1}/{z}")
        self._refresh_t2_markers()

    # ══════════════════════════════════════════════════════════════════════════
    #  Coordinate helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _disp_to_voxel(self, view_key: str, dx: float, dy: float) -> tuple:
        """Map display (dx, dy) in a T1 view → 3D voxel (vz, vy, vx)."""
        Z, Y, X = self.t1_vol.get_shape()
        sz, sy, sx = self.t1_vol.spacing
        zi = self.slice_idx['axial']
        yi = self.slice_idx['coronal']
        xi = self.slice_idx['sagittal']

        if view_key == 'axial':
            # display is (col=x, row=Y-1-y)
            # scale is (sx, sy)
            dx_unscaled, dy_unscaled = dx / sx, dy / sy
            vz, vy, vx = zi, int(round(Y - 1 - dy_unscaled)), int(round(dx_unscaled))
        elif view_key == 'coronal':
            # display is (col=x, row=z)
            # scale is (sx, sz)
            dx_unscaled, dy_unscaled = dx / sx, dy / sz
            vz, vy, vx = int(round(dy_unscaled)), yi, int(round(dx_unscaled))
        else:  # sagittal
            # display is (col=y, row=z)
            # scale is (sy, sz)
            dx_unscaled, dy_unscaled = dx / sy, dy / sz
            vz, vy, vx = int(round(dy_unscaled)), int(round(dx_unscaled)), xi

        return (
            max(0, min(vz, Z - 1)),
            max(0, min(vy, Y - 1)),
            max(0, min(vx, X - 1)),
        )

    def _voxel_to_disp(self, view_key: str, voxel: tuple) -> tuple:
        """Map 3D voxel (vz, vy, vx) → display (dx, dy) in a T1 view."""
        vz, vy, vx = voxel
        Z, Y, X = self.t1_vol.get_shape()
        sz, sy, sx = self.t1_vol.spacing
        
        if view_key == 'axial':
            return float(vx) * sx, float(Y - 1 - vy) * sy
        elif view_key == 'coronal':
            return float(vx) * sx, float(vz) * sz
        else:  # sagittal
            return float(vy) * sy, float(vz) * sz

    # ══════════════════════════════════════════════════════════════════════════
    #  Placement mode
    # ══════════════════════════════════════════════════════════════════════════

    def _toggle_place(self, name: str, on: bool):
        if on:
            self._placing = name
            # Deactivate the other button without re-triggering this slot
            other = self.btn_pc if name == 'ac' else self.btn_ac
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            # Restore default style on the other button
            other.setStyleSheet(_BTN)
            # Highlight the active button
            active_col = _AC if name == 'ac' else _PC
            btn = self.btn_ac if name == 'ac' else self.btn_pc
            btn.setStyleSheet(
                f"QPushButton{{background:{active_col};color:#11111b;"
                f"font-weight:bold;padding:8px;border-radius:5px;font-size:12px;}}"
            )
            # Switch cursor to cross-hair on all T1 views
            for k in ('axial', 'coronal', 'sagittal'):
                self.views[k].setCursor(Qt.CrossCursor)
        else:
            if self._placing == name:
                self._placing = None
            btn = self.btn_ac if name == 'ac' else self.btn_pc
            btn.setStyleSheet(_BTN)
            # Restore arrow cursor
            for k in ('axial', 'coronal', 'sagittal'):
                self.views[k].setCursor(Qt.ArrowCursor)

    def _on_view_clicked(self, ev, view_key: str):
        if self._placing is None:
            return
        if ev.button() != Qt.LeftButton:
            return
        if self.t1_vol is None:
            return

        pos = ev.scenePos()
        pt  = self.views[view_key].plotItem.vb.mapSceneToView(pos)
        voxel = self._disp_to_voxel(view_key, pt.x(), pt.y())

        self._set_landmark(self._placing, voxel)

        # Exit placement mode (toggle the active button off)
        btn = self.btn_ac if self._placing == 'ac' else self.btn_pc
        btn.setChecked(False)
        # _toggle_place will be triggered by the toggled signal above

    # ══════════════════════════════════════════════════════════════════════════
    #  Landmark management
    # ══════════════════════════════════════════════════════════════════════════

    def _set_landmark(self, name: str, voxel: tuple, update_world: bool = True):
        """Place or move a landmark. Creates/replaces markers in all T1 views."""
        color = _AC if name == 'ac' else _PC
        label = 'AC' if name == 'ac' else 'PC'

        # Remove old markers
        old = self.ac if name == 'ac' else self.pc
        if old is not None:
            for m in old['markers'].values():
                m.remove()

        # Create fresh markers in all three T1 views
        markers = {}
        for key in ('axial', 'coronal', 'sagittal'):
            dx, dy = self._voxel_to_disp(key, voxel)
            m = _LandmarkMarker(
                self.views[key], dx, dy, color, label,
                on_dragged=lambda x, y, k=key, n=name: self._on_drag(n, k, x, y),
                draggable=True,
            )
            markers[key] = m

        entry = {'voxel': voxel, 'markers': markers,
                 'voxel_t2': None, 'world_t2': None}
                 
        if not update_world and old is not None and 'world' in old:
            entry['world'] = old['world']
            entry['volume_name'] = old['volume_name']
        else:
            entry['world'] = self.t1_vol.voxel_to_world(voxel)
            entry['volume_name'] = self.t1_name

        if name == 'ac':
            self.ac = entry
            vz, vy, vx = voxel
            self.lbl_ac.setText(f"z={vz}  y={vy}  x={vx}")
        else:
            self.pc = entry
            vz, vy, vx = voxel
            self.lbl_pc.setText(f"z={vz}  y={vy}  x={vx}")

        # Navigate all T1 views to the landmark slice
        self.slice_idx['axial']    = voxel[0]
        self.slice_idx['coronal']  = voxel[1]
        self.slice_idx['sagittal'] = voxel[2]
        self._render_all_t1()

        self._refresh_marker_visibility()
        self._refresh_proj_lines()
        self._refresh_transfer_btn()

    def _on_drag(self, name: str, view_key: str, dx: float, dy: float):
        """Called when a marker dot is dragged — updates 3D voxel coords."""
        entry = self.ac if name == 'ac' else self.pc
        if entry is None:
            return

        new_voxel = self._disp_to_voxel(view_key, dx, dy)
        entry['voxel'] = new_voxel
        entry['world'] = self.t1_vol.voxel_to_world(new_voxel)
        entry['volume_name'] = self.t1_name
        vz, vy, vx = new_voxel

        # Update label
        lbl = self.lbl_ac if name == 'ac' else self.lbl_pc
        lbl.setText(f"z={vz}  y={vy}  x={vx}")

        # Sync the other two view markers to the new position
        for key in ('axial', 'coronal', 'sagittal'):
            if key == view_key:
                continue
            ndx, ndy = self._voxel_to_disp(key, new_voxel)
            entry['markers'][key].update_pos(ndx, ndy)

        self._refresh_marker_visibility()
        self._refresh_proj_lines()

    # ── Marker visibility ─────────────────────────────────────────────────────

    def _refresh_marker_visibility(self):
        zi = self.slice_idx['axial']
        yi = self.slice_idx['coronal']
        xi = self.slice_idx['sagittal']

        for entry in (self.ac, self.pc):
            if entry is None:
                continue
            vz, vy, vx = entry['voxel']
            vis_map = {
                'axial':    zi == vz,
                'coronal':  yi == vy,
                'sagittal': xi == vx,
            }
            for key, vis in vis_map.items():
                entry['markers'][key].set_visible(vis)

    # ── Projected AC-PC line ──────────────────────────────────────────────────

    def _refresh_proj_lines(self):
        if self.ac is None or self.pc is None:
            for line in self._proj_lines.values():
                line.setVisible(False)
            return

        for key in ('axial', 'coronal', 'sagittal'):
            ax, ay = self._voxel_to_disp(key, self.ac['voxel'])
            px, py = self._voxel_to_disp(key, self.pc['voxel'])
            self._proj_lines[key].setData([ax, px], [ay, py])
            self._proj_lines[key].setVisible(True)

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _clear_landmark(self, name: str):
        entry = self.ac if name == 'ac' else self.pc
        if entry is not None:
            for m in entry['markers'].values():
                m.remove()
        if name == 'ac':
            self.ac = None
            self.lbl_ac.setText("—")
        else:
            self.pc = None
            self.lbl_pc.setText("—")
        self._refresh_proj_lines()
        self._refresh_transfer_btn()
        self.btn_save.setEnabled(False)

    def _clear_ac(self):
        self._clear_landmark('ac')

    def _clear_pc(self):
        self._clear_landmark('pc')

    def _clear_all(self):
        self._clear_landmark('ac')
        self._clear_landmark('pc')
        self._clear_t2_markers()
        self.lbl_status.setText("")

    # ══════════════════════════════════════════════════════════════════════════
    #  Transfer
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_transfer_btn(self):
        ok = (
            self.ac is not None
            and self.pc is not None
            and self.cmb_t2.currentIndex() > 0
        )
        self.btn_transfer.setEnabled(ok)

    def _run_transfer(self):
        t2_name = self.cmb_t2.currentText()
        if t2_name not in self.volumes or self.cmb_t2.currentIndex() == 0:
            self.lbl_status.setText("⚠ Select a valid T2 volume.")
            return
        if self.ac is None or self.pc is None:
            self.lbl_status.setText("⚠ Define both AC and PC first.")
            return

        self.t2_name = t2_name
        self.t2_vol  = self.volumes[t2_name]

        aff_inv = np.linalg.inv(self.t2_vol.affine)

        def map_to_t2(entry: dict) -> tuple:
            world = np.array([*self.t1_vol.voxel_to_world(entry['voxel']), 1.0])
            ijk   = aff_inv @ world
            return tuple(np.round(ijk[:3]).astype(int))

        ac_t2 = map_to_t2(self.ac)
        pc_t2 = map_to_t2(self.pc)

        self.ac['voxel_t2'] = ac_t2
        self.pc['voxel_t2'] = pc_t2

        # Navigate T2 view to the AC slice
        Z = self.t2_vol.get_shape()[0]
        self.slice_t2 = max(0, min(ac_t2[0], Z - 1))

        self._render_t2()

        self.lbl_status.setText(
            f"✅ Transferred to: {t2_name}\n"
            f"AC → z={ac_t2[0]}  y={ac_t2[1]}  x={ac_t2[2]}\n"
            f"PC → z={pc_t2[0]}  y={pc_t2[1]}  x={pc_t2[2]}"
        )
        self.btn_save.setEnabled(True)

    # ── T2 overlay markers ────────────────────────────────────────────────────

    def _clear_t2_markers(self):
        for m in (self._t2_ac_marker, self._t2_pc_marker):
            if m is not None:
                m.remove()
        self._t2_ac_marker = None
        self._t2_pc_marker = None
        if self._t2_line is not None:
            self.views['t2'].removeItem(self._t2_line)
            self._t2_line = None

    def _refresh_t2_markers(self):
        self._clear_t2_markers()
        if self.t2_vol is None:
            return

        H = self.t2_vol.get_shape()[1]
        sz, sy, sx = self.t2_vol.spacing

        def make(entry, color, label):
            if entry is None or entry.get('voxel_t2') is None:
                return None
            vz, vy, vx = entry['voxel_t2']
            if vz != self.slice_t2:
                return None
            dx, dy = float(vx) * sx, float(H - 1 - vy) * sy
            return _LandmarkMarker(
                self.views['t2'], dx, dy, color, label,
                draggable=False,
            )

        self._t2_ac_marker = make(self.ac, _AC, 'AC')
        self._t2_pc_marker = make(self.pc, _PC, 'PC')

        # Draw AC-PC line on T2 if both are visible on this slice
        if self._t2_ac_marker and self._t2_pc_marker:
            ac_v = self.ac['voxel_t2']
            pc_v = self.pc['voxel_t2']
            self._t2_line = pg.PlotDataItem(
                [float(ac_v[2]) * sx, float(pc_v[2]) * sx],
                [float(H - 1 - ac_v[1]) * sy, float(H - 1 - pc_v[1]) * sy],
                pen=pg.mkPen(_LINE, width=1.5, style=Qt.DashLine),
            )
            self.views['t2'].addItem(self._t2_line)

    # ══════════════════════════════════════════════════════════════════════════
    #  Save
    # ══════════════════════════════════════════════════════════════════════════

    def _resolve_cache_dir(self) -> str:
        """Mirror of MRIViewerWidget._resolve_cache_dir."""
        d = self.t1_vol
        clean = os.path.normpath(d.file_path)
        pname = d.metadata.get("patient_name", "").strip()
        folder = None

        if pname and pname.lower() in clean.lower():
            idx    = clean.lower().find(pname.lower()) + len(pname)
            folder = clean[:idx]

        if not folder:
            curr  = clean if os.path.isdir(clean) else os.path.dirname(clean)
            terms = ['dicom', 'mri', 'ct', 't1', 't2', 'series', 'study',
                     'ax', 'cor', 'sag']
            while curr != os.path.dirname(curr):
                if any(t in os.path.basename(curr).lower() for t in terms):
                    curr = os.path.dirname(curr)
                else:
                    break
            folder = curr

        return os.path.join(folder, "dbs_cache")

    def _load_from_cache(self):
        cache_dir = self._resolve_cache_dir()
        acpc_base = os.path.join(cache_dir, "acpc")
        if not os.path.isdir(acpc_base): return
        
        import glob
        import json
        files = glob.glob(os.path.join(acpc_base, "*", "acpc_points.json"))
        files.sort(key=os.path.getmtime, reverse=True)
        
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                found = False
                if 'ac' in data and 'world_mm' in data['ac'] and data['ac']['world_mm']:
                    world_ac = data['ac']['world_mm']
                    self.ac = {'world': world_ac, 'volume_name': 'cache', 'markers': {}}
                    found = True
                if 'pc' in data and 'world_mm' in data['pc'] and data['pc']['world_mm']:
                    world_pc = data['pc']['world_mm']
                    self.pc = {'world': world_pc, 'volume_name': 'cache', 'markers': {}}
                    found = True
                    
                if found:
                    break
            except Exception:
                continue

    def _save(self):
        if self.ac is None or self.pc is None:
            return

        cache   = self._resolve_cache_dir()
        out_dir = os.path.join(cache, "acpc", self.t1_name)
        os.makedirs(out_dir, exist_ok=True)

        def fmt(v):
            if v is None: return None
            return [float(x) if isinstance(x, float) or (hasattr(x, 'dtype') and x.dtype.kind == 'f') 
                    else int(x) for x in v]

        data = {
            "source_volume": self.t1_name,
            "ac": {
                "voxel_t1":      fmt(self.ac['voxel']),
                "world_mm":      fmt(self.ac.get('world')),
                "target_volume": self.t2_name,
                "voxel_t2":      fmt(self.ac['voxel_t2']),
            },
            "pc": {
                "voxel_t1":      fmt(self.pc['voxel']),
                "world_mm":      fmt(self.pc.get('world')),
                "target_volume": self.t2_name,
                "voxel_t2":      fmt(self.pc['voxel_t2']),
            },
        }

        path = os.path.join(out_dir, "acpc_points.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.lbl_status.setText(f"✅ Saved to:\n{path}")
        print(f"[ACPC] Saved → {path}")
