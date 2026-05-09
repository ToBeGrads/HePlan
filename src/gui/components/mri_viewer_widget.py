import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QSlider, QLabel, 
                             QGroupBox, QTableWidget, QTableWidgetItem, QAbstractItemView, 
                             QHeaderView, QPushButton, QSizePolicy, QComboBox, QScrollArea, QFrame)
from PyQt5.QtCore import Qt
from src.core.patient_data import PatientData

pg.setConfigOption('background', '#181825')
pg.setConfigOption('foreground', '#cdd6f4')
pg.setConfigOption('imageAxisOrder', 'row-major')

class MRIViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volumes = {}
        self.current_volume = None
        self.current_vol_name = None
        self.slice_idx = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self._updating_crosshairs = False
        self.volume_stats = {}
        self.maximized_view = None
        
        self._init_ui()


    def _init_ui(self):
        # Main horizontal layout: Views (left) + Sidebar (right)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ============ LEFT: 2×2 Grid for Views ============
        views_container = QWidget()
        views_layout = QGridLayout(views_container)
        views_layout.setContentsMargins(4, 4, 4, 4)
        # color
        views_layout.setSpacing(4)

        # Create plot widgets
        self.view_axial = pg.PlotWidget()
        self.view_coronal = pg.PlotWidget()
        self.view_sagittal = pg.PlotWidget()

        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.hideAxis('left'); v.hideAxis('bottom')
            v.setMouseEnabled(x=True, y=True)
            v.setMenuEnabled(False)
            v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            v.setAspectLocked(lock=True, ratio=1.0)

        # Headers with controls
        def create_header(title, view_key):
            header = QWidget()
            header.setStyleSheet("background-color: #1e1e2e; padding: 4px;")
            layout = QHBoxLayout(header)
            layout.setContentsMargins(4, 2, 4, 2)
            layout.setSpacing(4)
            
            lbl = QLabel(title)
            lbl.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 11px; background-color: transparent;")
            layout.addWidget(lbl)
            layout.addStretch()
            
            # Slice navigation
            btn_prev = QPushButton("◀")
            btn_prev.setFixedSize(20, 20)
            btn_prev.clicked.connect(lambda: self._step_slice(view_key, -1))
            
            lbl_slice = QLabel("0/0")
            lbl_slice.setStyleSheet("color: #a6adc8; font-size: 10px; min-width: 40px; background-color: #181825; padding: 2px 6px; border-radius: 3px;")
            lbl_slice.setAlignment(Qt.AlignCenter)
            
            btn_next = QPushButton("▶")
            btn_next.setFixedSize(20, 20)
            btn_next.clicked.connect(lambda: self._step_slice(view_key, 1))
            
            # Zoom buttons
            btn_zi = QPushButton("+")
            btn_zi.setFixedSize(20, 20)
            btn_zi.clicked.connect(lambda: self._zoom_view(view_key, 1.2))
            
            btn_zo = QPushButton("−")
            btn_zo.setFixedSize(20, 20)
            btn_zo.clicked.connect(lambda: self._zoom_view(view_key, 0.8))
            
            # Maximize button
            btn_max = QPushButton("◰")
            btn_max.setFixedSize(20, 20)
            btn_max.setToolTip("Maximize view")
            btn_max.clicked.connect(lambda: self._toggle_maximize(view_key))
            
            layout.addWidget(btn_prev)
            layout.addWidget(lbl_slice)
            layout.addWidget(btn_next)
            layout.addWidget(btn_zi)
            layout.addWidget(btn_zo)
            layout.addWidget(btn_max)
            
            header.lbl_slice = lbl_slice
            header.view_key = view_key
            header.btn_max = btn_max
            return header

        self.header_axial = create_header("Axial", "axial")
        self.header_coronal = create_header("Coronal", "coronal")
        self.header_sagittal = create_header("Sagittal", "sagittal")

        # 2×2 Grid Layout (4th quadrant empty)
        views_layout.addWidget(self.header_axial, 0, 0)
        views_layout.addWidget(self.view_axial, 1, 0)
        views_layout.addWidget(self.header_coronal, 0, 1)
        views_layout.addWidget(self.view_coronal, 1, 1)
        views_layout.addWidget(self.header_sagittal, 2, 0)
        views_layout.addWidget(self.view_sagittal, 3, 0)

        views_layout.setRowStretch(1, 2)
        views_layout.setRowStretch(3, 2)
        views_layout.setColumnStretch(0, 2)
        views_layout.setColumnStretch(1, 2)

        main_layout.addWidget(views_container, 3)  # 75% width

        # ============ RIGHT: Sidebar ============
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #1e1e2e; border: none")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(12)
        sidebar_layout.setAlignment(Qt.AlignTop)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Volumes Group
        grp_volumes = QGroupBox("📁 Loaded Volumes")
        grp_volumes.setObjectName("grpVolumes")
        vol_layout = QVBoxLayout(grp_volumes)
        vol_layout.setContentsMargins(8, 8, 8, 8)
        
        self.tbl_volumes = QTableWidget()
        self.tbl_volumes.setObjectName('tblVolumes')
        self.tbl_volumes.setColumnCount(3)
        self.tbl_volumes.setHorizontalHeaderLabels(["Name", "Mod", "Shape"])
        self.tbl_volumes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_volumes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_volumes.setAlternatingRowColors(True)
        self.tbl_volumes.setShowGrid(True)
        self.tbl_volumes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_volumes.verticalHeader().setVisible(False)
        vol_layout.addWidget(self.tbl_volumes)
        scroll_layout.addWidget(grp_volumes)

        # Window/Level Group
        grp_wl = QGroupBox("Window / Level")
        grp_wl.setStyleSheet('max-height: 200px;')  # Limit height to prevent excessive stretching
        wl_layout = QVBoxLayout(grp_wl)
        wl_layout.setContentsMargins(0, 0, 0, 0)
        wl_layout.setSpacing(0)
        
        self.lbl_wl = QLabel("W: 0  |  L: 0")
        self.lbl_wl.setStyleSheet("color: black; font-size: 11px; background-color: white; padding: 0px; border-radius: 4px; max-height: 20px;")
        self.lbl_wl.setAlignment(Qt.AlignCenter)
        wl_layout.addWidget(self.lbl_wl)
        
        self.sld_window = QSlider(Qt.Horizontal)
        self.sld_window.setRange(1, 4000)
        
        self.sld_level = QSlider(Qt.Horizontal)
        self.sld_level.setRange(-2000, 4000)
        
        wl_layout.addWidget(QLabel("Window (Contrast):"))
        wl_layout.addWidget(self.sld_window)
        wl_layout.addWidget(QLabel("Level (Brightness):"))
        wl_layout.addWidget(self.sld_level)
        
        # Preset combo
        preset_layout = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["Auto", "T1", "T2", "SWAN", "CT"])
        self.cmb_preset.setCurrentText("Auto")
        
        self.btn_apply_preset = QPushButton("Apply")
        preset_layout.addWidget(self.cmb_preset)
        preset_layout.addWidget(self.btn_apply_preset)
        wl_layout.addLayout(preset_layout)
        
        self.btn_reset_wl = QPushButton("🔄 Reset to Auto")
        wl_layout.addWidget(self.btn_reset_wl)
        
        scroll_layout.addWidget(grp_wl)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)
        main_layout.addWidget(sidebar, 1)  # 25% width

        # Image items
        self.img_axial = pg.ImageItem()
        self.img_coronal = pg.ImageItem()
        self.img_sagittal = pg.ImageItem()
        self.view_axial.addItem(self.img_axial)
        self.view_coronal.addItem(self.img_coronal)
        self.view_sagittal.addItem(self.img_sagittal)

        # Crosshairs
        self.crosshairs = {
            'axial': {'h': pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#89b4fa', width=1.5)),
                      'v': pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#89b4fa', width=1.5))},
            'coronal': {'h': pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#a6e3a1', width=1.5)),
                        'v': pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#a6e3a1', width=1.5))},
            'sagittal': {'h': pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#f38ba8', width=1.5)),
                         'v': pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#f38ba8', width=1.5))}
        }
        for view, lines in self.crosshairs.items():
            getattr(self, f'view_{view}').addItem(lines['h'])
            getattr(self, f'view_{view}').addItem(lines['v'])

        # Connect signals
        self.tbl_volumes.itemSelectionChanged.connect(self._on_volume_selected)
        self.sld_window.valueChanged.connect(self._update_window_level)
        self.sld_level.valueChanged.connect(self._update_window_level)
        self.btn_reset_wl.clicked.connect(self._auto_window_level)
        self.btn_apply_preset.clicked.connect(self._apply_preset)

        # Crosshair sync
        for view in ('axial', 'coronal', 'sagittal'):
            self.crosshairs[view]['h'].sigPositionChanged.connect(
                lambda line, v=view: self._sync_crosshair(v, 'h'))
            self.crosshairs[view]['v'].sigPositionChanged.connect(
                lambda line, v=view: self._sync_crosshair(v, 'v'))

    def _toggle_maximize(self, view_key):
        """Toggle maximize/restore for a specific view."""
        views = ['axial', 'coronal', 'sagittal']
        
        if self.maximized_view == view_key:
            # Restore all views
            self.maximized_view = None
            for v in views:
                header = getattr(self, f'header_{v}')
                view_widget = getattr(self, f'view_{v}')
                header.setVisible(True)
                view_widget.setVisible(True)
            
            # Restore grid layout
            self._setup_grid_layout()
        else:
            # Maximize this view
            self.maximized_view = view_key
            for v in views:
                header = getattr(self, f'header_{v}')
                view_widget = getattr(self, f'view_{v}')
                if v == view_key:
                    header.setVisible(True)
                    view_widget.setVisible(True)
                else:
                    header.setVisible(False)
                    view_widget.setVisible(False)
            
            # Setup single view layout
            views_layout = self.layout().itemAt(0).widget().layout()
            views_layout.setColumnStretch(0, 1)
            views_layout.setColumnStretch(1, 0)
            views_layout.setRowStretch(1, 1)
            views_layout.setRowStretch(3, 0)

    def _setup_grid_layout(self):
        """Restore the 2×2 grid layout."""
        views_layout = self.layout().itemAt(0).widget().layout()
        views_layout.setColumnStretch(0, 2)
        views_layout.setColumnStretch(1, 2)
        views_layout.setRowStretch(1, 2)
        views_layout.setRowStretch(3, 2)

    def add_volume(self, patient_data: PatientData, vol_name: str):
        """Add volume and precompute stats."""
        if vol_name in self.volumes:
            vol_name = f"{vol_name}_{len(self.volumes)}"
        self.volumes[vol_name] = patient_data

        vol = patient_data.volume.astype(np.float32)
        flat = vol.ravel()
        self.volume_stats[vol_name] = {
            'p05': np.percentile(flat, 0.5),
            'p995': np.percentile(flat, 99.5),
            'min': vol.min(),
            'max': vol.max()
        }

        row = self.tbl_volumes.rowCount()
        self.tbl_volumes.insertRow(row)
        self.tbl_volumes.setItem(row, 0, QTableWidgetItem(vol_name))
        self.tbl_volumes.setItem(row, 1, QTableWidgetItem(patient_data.modality))
        self.tbl_volumes.setItem(row, 2, QTableWidgetItem(str(patient_data.get_shape())))

    def _on_volume_selected(self):
        row = self.tbl_volumes.currentRow()
        if row < 0: return
        vol_name = self.tbl_volumes.item(row, 0).text()
        self._activate_volume(vol_name)

    def _activate_volume(self, vol_name: str):
        if vol_name not in self.volumes: return
        self.current_vol_name = vol_name
        self.current_volume = self.volumes[vol_name]
        
        z, y, x = self.current_volume.get_shape()
        self.slice_idx = {
            'axial': min(z // 2, z - 1),
            'coronal': min(y // 2, y - 1),
            'sagittal': min(x // 2, x - 1)
        }
        
        stats = self.volume_stats[vol_name]
        self.sld_window.setMinimum(1)
        self.sld_window.setMaximum(int((stats['max'] - stats['min']) * 1.5))
        self.sld_level.setMinimum(int(stats['min']))
        self.sld_level.setMaximum(int(stats['max']))
        
        self._auto_window_level()
        self._render_mpr()
        self._update_slice_labels()

    def _update_slice_labels(self):
        if self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        for view_name, (total, current) in [
            ('axial', (z, self.slice_idx['axial'])),
            ('coronal', (y, self.slice_idx['coronal'])),
            ('sagittal', (x, self.slice_idx['sagittal']))
        ]:
            header = getattr(self, f'header_{view_name}')
            header.lbl_slice.setText(f"{current+1}/{total}")

    def _get_tissue_values(self, vol):
        threshold = np.percentile(vol, 2.0)
        mask = vol > threshold
        if np.sum(mask) < 1000:
            return vol.flatten()
        return vol[mask]

    def _auto_window_level(self):
        if self.current_volume is None: return
        tissue_vals = self._get_tissue_values(self.current_volume.volume)
        if len(tissue_vals) == 0: return
        
        vmin = np.percentile(tissue_vals, 0.5)
        vmax = np.percentile(tissue_vals, 99.5)
        window = max(1, vmax - vmin)
        level = (vmax + vmin) / 2
        
        self.sld_window.setValue(int(window))
        self.sld_level.setValue(int(level))
        self._update_window_level()

    def _apply_preset(self):
        preset = self.cmb_preset.currentText()
        if self.current_volume is None: return
        
        tissue_vals = self._get_tissue_values(self.current_volume.volume)
        if len(tissue_vals) == 0: return
        
        base_min = np.percentile(tissue_vals, 1.0)
        base_max = np.percentile(tissue_vals, 99.0)
        base_range = base_max - base_min
        
        presets = {
            "T1": (0.6, 0.4),
            "T2": (0.8, 0.5),
            "SWAN": (0.7, 0.45),
            "CT": (0.3, 0.6)
        }
        
        if preset in presets:
            w_ratio, l_ratio = presets[preset]
            window = max(1, base_range * w_ratio)
            level = base_min + (base_range * l_ratio)
            self.sld_window.setValue(int(window))
            self.sld_level.setValue(int(level))
            self._update_window_level()
        elif preset == "Auto":
            self._auto_window_level()

    def _render_mpr(self):
        if self.current_volume is None: return
        vol = self.current_volume.volume
        if vol.ndim != 3: return

        z, y, x = self.current_volume.get_shape()
        
        # Keep indices in bounds
        self.slice_idx['axial'] = max(0, min(self.slice_idx['axial'], z-1))
        self.slice_idx['coronal'] = max(0, min(self.slice_idx['coronal'], y-1))
        self.slice_idx['sagittal'] = max(0, min(self.slice_idx['sagittal'], x-1))
        
        zi, yi, xi = self.slice_idx['axial'], self.slice_idx['coronal'], self.slice_idx['sagittal']

        # ✅ Axial: flip vertically to show eyes/anterior at top
        ax_slice = np.flipud(vol[zi, :, :])
        
        # ✅ Coronal: normal orientation
        cor_slice = vol[:, yi, :]
        
        # ✅ Sagittal: normal orientation  
        sag_slice = vol[:, :, xi]

        self.img_axial.setImage(ax_slice, autoLevels=False)
        self.img_coronal.setImage(cor_slice, autoLevels=False)
        self.img_sagittal.setImage(sag_slice, autoLevels=False)

        self._update_crosshair_positions()
        self._update_window_level()

    def _update_crosshair_positions(self):
        if self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        self._updating_crosshairs = True
        
        # Axial view crosshairs
        self.crosshairs['axial']['h'].setPos(y - 1 - self.slice_idx['coronal'])  # Flip for flipped image
        self.crosshairs['axial']['v'].setPos(self.slice_idx['sagittal'])
        
        # Coronal view crosshairs
        self.crosshairs['coronal']['h'].setPos(self.slice_idx['axial'])
        self.crosshairs['coronal']['v'].setPos(self.slice_idx['sagittal'])
        
        # Sagittal view crosshairs
        self.crosshairs['sagittal']['h'].setPos(self.slice_idx['axial'])
        self.crosshairs['sagittal']['v'].setPos(self.slice_idx['coronal'])  # ✅ No flip!
        
        self._updating_crosshairs = False

    def _sync_crosshair(self, view, axis):
        if self._updating_crosshairs or self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        pos = int(self.crosshairs[view][axis].value())

        if view == 'axial':
            if axis == 'h':
                # Flip back because axial image is flipped
                self.slice_idx['coronal'] = max(0, min(y-1, y - 1 - pos))
            else:
                self.slice_idx['sagittal'] = max(0, min(x-1, pos))
        elif view == 'coronal':
            if axis == 'h':
                self.slice_idx['axial'] = max(0, min(z-1, pos))
            else:
                self.slice_idx['sagittal'] = max(0, min(x-1, pos))
        elif view == 'sagittal':
            if axis == 'h':
                self.slice_idx['axial'] = max(0, min(z-1, pos))
            else:
                # ✅ No flip needed
                self.slice_idx['coronal'] = max(0, min(y-1, pos))

        self._render_mpr()
        self._update_slice_labels()

    def _step_slice(self, view_key, step):
        if self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        
        if view_key == 'axial':
            self.slice_idx['axial'] = max(0, min(z-1, self.slice_idx['axial'] + step))
        elif view_key == 'coronal':
            self.slice_idx['coronal'] = max(0, min(y-1, self.slice_idx['coronal'] + step))
        elif view_key == 'sagittal':
            self.slice_idx['sagittal'] = max(0, min(x-1, self.slice_idx['sagittal'] + step))
        
        self._render_mpr()
        self._update_slice_labels()

    def _zoom_view(self, view_key, factor):
        view = getattr(self, f'view_{view_key}')
        view.getViewBox().scaleBy((factor, factor))

    def _update_window_level(self):
        w = self.sld_window.value()
        l = self.sld_level.value()
        self.lbl_wl.setText(f"W:{w} L:{l}")
        vmin, vmax = l - w/2, l + w/2
        for img in (self.img_axial, self.img_coronal, self.img_sagittal):
            img.setLevels((vmin, vmax))