import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QSlider, QLabel, 
                             QGroupBox, QTableWidget, QTableWidgetItem, QAbstractItemView, \
                             QHeaderView, QPushButton, QSizePolicy, QComboBox, QScrollArea, QFrame, QProgressBar)
import os
from PyQt5.QtCore import Qt, pyqtSignal
from src.core.patient_data import PatientData

pg.setConfigOption('background', '#181825')
pg.setConfigOption('foreground', '#cdd6f4')
pg.setConfigOption('imageAxisOrder', 'row-major')

class MRIViewerWidget(QWidget):
    masks_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volumes = {}
        self.current_volume = None
        self.current_vol_name = None
        self.slice_idx = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self._updating_crosshairs = False
        self.volume_stats = {}
        self.maximized_view = None
        self.masks = {}
        self.current_opacity = 150  # Default alpha opacity out of 255

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ============ LEFT: 2×2 Grid for Views ============
        views_container = QWidget()
        views_layout = QGridLayout(views_container)
        views_layout.setContentsMargins(4, 4, 4, 4)
        views_layout.setSpacing(4)

        self.view_axial = pg.PlotWidget()
        self.view_coronal = pg.PlotWidget()
        self.view_sagittal = pg.PlotWidget()

        for v in (self.view_axial, self.view_coronal, self.view_sagittal):
            v.hideAxis('left'); v.hideAxis('bottom')
            v.setMouseEnabled(x=True, y=True)
            v.setMenuEnabled(False)
            v.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            v.setAspectLocked(lock=True, ratio=1.0)

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
            
            btn_prev = QPushButton("◀")
            btn_prev.setFixedSize(20, 20)
            btn_prev.clicked.connect(lambda: self._step_slice(view_key, -1))
            
            lbl_slice = QLabel("0/0")
            lbl_slice.setStyleSheet("color: #a6adc8; font-size: 10px; min-width: 40px; background-color: #181825; padding: 2px 6px; border-radius: 3px;")
            lbl_slice.setAlignment(Qt.AlignCenter)
            
            btn_next = QPushButton("▶")
            btn_next.setFixedSize(20, 20)
            btn_next.clicked.connect(lambda: self._step_slice(view_key, 1))
            
            btn_zi = QPushButton("+")
            btn_zi.setFixedSize(20, 20)
            btn_zi.clicked.connect(lambda: self._zoom_view(view_key, 1.2))
            
            btn_zo = QPushButton("−")
            btn_zo.setFixedSize(20, 20)
            btn_zo.clicked.connect(lambda: self._zoom_view(view_key, 0.8))
            
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

        main_layout.addWidget(views_container, 3)

        # ============ RIGHT: Sidebar ============
        sidebar = QWidget()
        sidebar.setStyleSheet("background-color: #1e1e2e; border: none")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(12)
        sidebar_layout.setAlignment(Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Volumes Group
        grp_volumes = QGroupBox("📁 Loaded Volumes")
        grp_volumes.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        vol_layout = QVBoxLayout(grp_volumes)
        vol_layout.setContentsMargins(10, 10, 10, 10)
        vol_layout.setAlignment(Qt.AlignTop)
        
        self.tbl_volumes = QTableWidget()
        self.tbl_volumes.setColumnCount(4)
        self.tbl_volumes.setHorizontalHeaderLabels(["Name", "Mod", "Shape", "Action"])
        self.tbl_volumes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_volumes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_volumes.setAlternatingRowColors(True)
        self.tbl_volumes.setShowGrid(True)
        self.tbl_volumes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_volumes.verticalHeader().setVisible(False)
        self.tbl_volumes.setStyleSheet("QTableWidget { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; } QHeaderView::section { background-color: #313244; color: #a6adc8; font-weight: bold; border: none; padding: 4px; }")
        vol_layout.addWidget(self.tbl_volumes)
        
        # Standalone Segmentation Button
        self.btn_run_seg = QPushButton("▶ Run AI Segmentation")
        self.btn_run_seg.setStyleSheet("QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; font-size: 13px; padding: 8px; border-radius: 6px; margin-top: 5px; } QPushButton:disabled { background-color: #45475a; color: #a6adc8; }")
        self.btn_run_seg.setEnabled(False)
        self.btn_run_seg.clicked.connect(self._on_segmentation_clicked)
        vol_layout.addWidget(self.btn_run_seg)
        vol_layout.addStretch()
        
        scroll_layout.addWidget(grp_volumes)

        # Window/Level Group
        grp_wl = QGroupBox("🎚 Window / Level")
        grp_wl.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        grp_wl.setStyleSheet('max-height: 220px;')
        wl_layout = QVBoxLayout(grp_wl)
        wl_layout.setContentsMargins(10, 10, 10, 10)
        wl_layout.setSpacing(6)
        wl_layout.setAlignment(Qt.AlignTop)
        
        self.lbl_wl = QLabel("W: 0  |  L: 0")
        self.lbl_wl.setStyleSheet("color: black; font-size: 11px; background-color: white; padding: 2px; border-radius: 4px; max-height: 20px;")
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
        
        preset_layout = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["Auto", "T1", "T2", "SWAN", "CT"])
        self.btn_apply_preset = QPushButton("Apply")
        preset_layout.addWidget(self.cmb_preset)
        preset_layout.addWidget(self.btn_apply_preset)
        wl_layout.addLayout(preset_layout)
        
        self.btn_reset_wl = QPushButton("🔄 Reset to Auto")
        wl_layout.addWidget(self.btn_reset_wl)
        wl_layout.addStretch()
        scroll_layout.addWidget(grp_wl)

        # --- PROGRESS INDICATOR (Waiting Mechanism Overlay) ---
        # --- PROGRESS INDICATOR (Waiting Mechanism Overlay) ---
        self.grp_progress = QGroupBox("⏳ AI Pipeline Execution")
        self.grp_progress.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #f9e2af; border: 1px solid #f9e2af; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        self.grp_progress.setVisible(False)
        prog_layout = QVBoxLayout(self.grp_progress)
        prog_layout.setContentsMargins(10, 10, 10, 10)
        prog_layout.setAlignment(Qt.AlignTop)
        
        self.lbl_progress_status = QLabel("Initializing pipeline updates...")
        self.lbl_progress_status.setStyleSheet("color: #cdd6f4; font-weight: bold; font-size: 11px;")
        self.lbl_progress_status.setWordWrap(True)
        prog_layout.addWidget(self.lbl_progress_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #45475a; border-radius: 3px; background-color: #1e1e2e; height: 10px; } QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }")
        prog_layout.addWidget(self.progress_bar)
        
        scroll_layout.addWidget(self.grp_progress)

        # --- NEW: COMPACT SEGMENTATION CONTROL TOOLBOX ---
        self.grp_seg_toolbox = QGroupBox("🧠 Segmentation Overlay")
        self.grp_seg_toolbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        self.grp_seg_toolbox.setVisible(False) # Hidden until mask finishes processing
        self.grp_seg_toolbox.setMaximumHeight(80) # Restrict height to save space
        
        box_layout = QVBoxLayout(self.grp_seg_toolbox)
        box_layout.setContentsMargins(8, 10, 8, 8)
        box_layout.setSpacing(0)
        box_layout.setAlignment(Qt.AlignTop)
        
        # Horizontal layout to pack controls side-by-side
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        
        # 1. Toggle Button (Compact Square Icon)
        self.btn_toggle_mask = QPushButton("👁")
        self.btn_toggle_mask.setToolTip("Toggle Mask Visibility")
        self.btn_toggle_mask.setFixedSize(26, 26) 
        self.btn_toggle_mask.setCheckable(True)
        self.btn_toggle_mask.setChecked(True)
        self.btn_toggle_mask.setStyleSheet("""
            QPushButton { background-color: #313244; border-radius: 4px; font-size: 14px; }
            QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; }
        """)
        self.btn_toggle_mask.clicked.connect(self._toggle_mask_visibility)
        controls_layout.addWidget(self.btn_toggle_mask)
        
        # 2. Opacity Label
        lbl_opacity = QLabel("Opacity:")
        lbl_opacity.setStyleSheet("color: #a6adc8; font-size: 11px;")
        controls_layout.addWidget(lbl_opacity)
        
        # 3. Opacity Slider
        self.sld_opacity = QSlider(Qt.Horizontal)
        self.sld_opacity.setRange(0, 255)
        self.sld_opacity.setValue(self.current_opacity)
        self.sld_opacity.valueChanged.connect(self._update_mask_opacity)
        controls_layout.addWidget(self.sld_opacity)
        
        box_layout.addLayout(controls_layout)
        box_layout.addStretch()
        scroll_layout.addWidget(self.grp_seg_toolbox)
        scroll_layout.addStretch()

        # ═╝ CRITICAL: RESTORE THESE LAYOUT ASSEMBLY LINES ╚═
        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)
        main_layout.addWidget(sidebar, 1)

        # Base Image layers
        self.img_axial = pg.ImageItem()
        self.img_coronal = pg.ImageItem()
        self.img_sagittal = pg.ImageItem()
        self.view_axial.addItem(self.img_axial)
        self.view_coronal.addItem(self.img_coronal)
        self.view_sagittal.addItem(self.img_sagittal)

        # Segmentation mask layers
        self.mask_axial = pg.ImageItem()
        self.mask_coronal = pg.ImageItem()
        self.mask_sagittal = pg.ImageItem()
        self.view_axial.addItem(self.mask_axial)
        self.view_coronal.addItem(self.mask_coronal)
        self.view_sagittal.addItem(self.mask_sagittal)

        
        self._apply_lut_table()

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

        self.tbl_volumes.itemSelectionChanged.connect(self._on_volume_selected)
        self.sld_window.valueChanged.connect(self._update_window_level)
        self.sld_level.valueChanged.connect(self._update_window_level)
        self.btn_reset_wl.clicked.connect(self._auto_window_level)
        self.btn_apply_preset.clicked.connect(self._apply_preset)

        for view in ('axial', 'coronal', 'sagittal'):
            self.crosshairs[view]['h'].sigPositionChanged.connect(lambda line, v=view: self._sync_crosshair(v, 'h'))
            self.crosshairs[view]['v'].sigPositionChanged.connect(lambda line, v=view: self._sync_crosshair(v, 'v'))

    def _apply_lut_table(self):
        """Builds lookup table for masks using current opacity level settings."""
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[0] = [0, 0, 0, 0]                                     # Background completely transparent
        lut[1] = [255, 0, 0, int(self.current_opacity)]           # STN target class 1 (Red)
        lut[2] = [0, 255, 0, int(self.current_opacity)]           # Target class 2 (Green)
        
        for mask_img in (self.mask_axial, self.mask_coronal, self.mask_sagittal):
            mask_img.setLookupTable(lut)

    def _update_mask_opacity(self, value):
        self.current_opacity = value
        self._apply_lut_table()
        self._render_mpr()

    def _toggle_mask_visibility(self):
        visible = self.btn_toggle_mask.isChecked()
        for mask_layer in (self.mask_axial, self.mask_coronal, self.mask_sagittal):
            mask_layer.setVisible(visible)
        self._render_mpr()

    def _run_segmentation(self, vol_name: str):
        patient_data = self.volumes[vol_name]
        
        # Determine cache path (Patient Folder level)
        import os
        import numpy as np
        
        clean_path = os.path.normpath(patient_data.file_path)
        pname = patient_data.metadata.get("patient_name", "").strip()
        patient_folder = None
        
        if pname and pname.lower() in clean_path.lower():
            # Safely cut path exactly after the patient's name folder (case-insensitive)
            idx = clean_path.lower().find(pname.lower()) + len(pname)
            patient_folder = clean_path[:idx]
            
        if not patient_folder:
            # Fallback heuristic: step up past generic DICOM/Series folders
            curr = clean_path if os.path.isdir(clean_path) else os.path.dirname(clean_path)
            generic_terms = ['dicom', 'mri', 'ct', 't1', 't2', 'series', 'study', 'ax', 'cor', 'sag']
            
            # Step up as long as the folder name looks like a generic sub-folder
            while curr != os.path.dirname(curr):
                if any(term in os.path.basename(curr).lower() for term in generic_terms):
                    curr = os.path.dirname(curr)
                else:
                    break
            patient_folder = curr
            
        cache_dir = os.path.join(patient_folder, "dbs_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "segmentation_mask.npy")
        
        def on_finished(mask_3d):
            # Save to cache if we ran the worker
            if not os.path.exists(cache_path):
                np.save(cache_path, mask_3d)
                
            self.grp_progress.setVisible(False)
            self.masks[vol_name] = mask_3d
            self.masks_updated.emit(self.masks)
            
            # Fallback for older direct reference (safe to keep or remove, but signal is better)
            if hasattr(self.parent(), 'registration_widget'):
                self.parent().registration_widget.load_masks(self.masks)
            
            # Show the control toolbox panel once completed successfully
            self.grp_seg_toolbox.setVisible(True)
            self.btn_toggle_mask.setChecked(True)
            self._toggle_mask_visibility()
            
            if self.current_vol_name == vol_name:
                self._render_mpr()

        def on_error(err_msg):
            self.lbl_progress_status.setText(f"Error encountered: {err_msg}")
        
        # Check cache
        if os.path.exists(cache_path):
            self.lbl_progress_status.setText("Found cached segmentation! Loading instantly...")
            self.grp_progress.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            
            # Use QTimer to briefly show progress before finishing
            import PyQt5.QtCore as QtCore
            def finish_cache_load():
                mask_3d = np.load(cache_path)
                on_finished(mask_3d)
                
            QtCore.QTimer.singleShot(500, finish_cache_load)
            return

        self.grp_progress.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate for worker
        self.lbl_progress_status.setText("Initializing compute worker...")
        
        from src.modules.ai.segmentation_worker import SegmentationWorker
        self.worker = SegmentationWorker(patient_data, vol_name, cache_dir=cache_dir)
        
        # Connect real-time progress update messages
        self.worker.progress.connect(self._handle_pipeline_progress)
        self.worker.error.connect(on_error)
        self.worker.finished.connect(on_finished)
        self.worker.start()

    def _handle_pipeline_progress(self, msg):
        print(f"AI Pipeline: {msg}")
        self.lbl_progress_status.setText(msg)

    def _toggle_maximize(self, view_key):
        views = ['axial', 'coronal', 'sagittal']
        if self.maximized_view == view_key:
            self.maximized_view = None
            for v in views:
                getattr(self, f'header_{v}').setVisible(True)
                getattr(self, f'view_{v}').setVisible(True)
            self._setup_grid_layout()
        else:
            self.maximized_view = view_key
            for v in views:
                if v == view_key:
                    getattr(self, f'header_{v}').setVisible(True)
                    getattr(self, f'view_{v}').setVisible(True)
                else:
                    getattr(self, f'header_{v}').setVisible(False)
                    getattr(self, f'view_{v}').setVisible(False)
            views_layout = self.layout().itemAt(0).widget().layout()
            views_layout.setColumnStretch(0, 1)
            views_layout.setColumnStretch(1, 0)
            views_layout.setRowStretch(1, 1)
            views_layout.setRowStretch(3, 0)

    def _setup_grid_layout(self):
        views_layout = self.layout().itemAt(0).widget().layout()
        views_layout.setColumnStretch(0, 2)
        views_layout.setColumnStretch(1, 2)
        views_layout.setRowStretch(1, 2)
        views_layout.setRowStretch(3, 2)

    def add_volume(self, patient_data: PatientData, vol_name: str):
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
        self.tbl_volumes.setItem(row, 3, QTableWidgetItem("Ready"))

    def _on_volume_selected(self):
        row = self.tbl_volumes.currentRow()
        if row < 0: return
        vol_name = self.tbl_volumes.item(row, 0).text()
        self._activate_volume(vol_name)
        
        patient_data = self.volumes[vol_name]
        is_t2 = "t2" in vol_name.lower() or "t2" in patient_data.modality.lower()
        self.btn_run_seg.setEnabled(is_t2)

    def _on_segmentation_clicked(self):
        if self.current_vol_name:
            self._run_segmentation(self.current_vol_name)

    def _activate_volume(self, vol_name: str):
        if vol_name not in self.volumes: return
        self.current_vol_name = vol_name
        self.current_volume = self.volumes[vol_name]
        
        # Check toolbox visibility based on mask availability
        self.grp_seg_toolbox.setVisible(vol_name in self.masks)
        
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
            "T1": (0.6, 0.4), "T2": (0.8, 0.5), "SWAN": (0.7, 0.45), "CT": (0.3, 0.6)
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
        
        self.slice_idx['axial'] = max(0, min(self.slice_idx['axial'], z-1))
        self.slice_idx['coronal'] = max(0, min(self.slice_idx['coronal'], y-1))
        self.slice_idx['sagittal'] = max(0, min(self.slice_idx['sagittal'], x-1))
        
        zi, yi, xi = self.slice_idx['axial'], self.slice_idx['coronal'], self.slice_idx['sagittal']

        ax_slice = np.flipud(vol[zi, :, :])
        cor_slice = vol[:, yi, :]
        sag_slice = vol[:, :, xi]

        # === FIX: Calculate window/level limits BEFORE setting the images ===
        w = self.sld_window.value()
        l = self.sld_level.value()
        vmin, vmax = l - w/2, l + w/2

        # Pass levels directly into setImage to prevent them from ever becoming None
        self.img_axial.setImage(ax_slice, autoLevels=False, levels=(vmin, vmax))
        self.img_coronal.setImage(cor_slice, autoLevels=False, levels=(vmin, vmax))
        self.img_sagittal.setImage(sag_slice, autoLevels=False, levels=(vmin, vmax))

        if self.current_vol_name in self.masks and self.btn_toggle_mask.isChecked():
            mask_vol = self.masks[self.current_vol_name]
            
            ax_mask = np.flipud(mask_vol[zi, :, :])
            cor_mask = mask_vol[:, yi, :]
            sag_mask = mask_vol[:, :, xi]
            
            self.mask_axial.setImage(ax_mask, autoLevels=False)
            self.mask_coronal.setImage(cor_mask, autoLevels=False)
            self.mask_sagittal.setImage(sag_mask, autoLevels=False)
        else:
            self.mask_axial.clear()
            self.mask_coronal.clear()
            self.mask_sagittal.clear()

        self._update_crosshair_positions()
        self._update_window_level()

    def _update_crosshair_positions(self):
        if self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        self._updating_crosshairs = True
        
        self.crosshairs['axial']['h'].setPos(y - 1 - self.slice_idx['coronal'])
        self.crosshairs['axial']['v'].setPos(self.slice_idx['sagittal'])
        
        self.crosshairs['coronal']['h'].setPos(self.slice_idx['axial'])
        self.crosshairs['coronal']['v'].setPos(self.slice_idx['sagittal'])
        
        self.crosshairs['sagittal']['h'].setPos(self.slice_idx['axial'])
        self.crosshairs['sagittal']['v'].setPos(self.slice_idx['coronal'])
        
        self._updating_crosshairs = False

    def _sync_crosshair(self, view, axis):
        if self._updating_crosshairs or self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        pos = int(self.crosshairs[view][axis].value())

        if view == 'axial':
            if axis == 'h':
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