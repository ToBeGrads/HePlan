import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QSlider, QLabel, 
                             QGroupBox, QTableWidget, QTableWidgetItem, QAbstractItemView, \
                             QHeaderView, QPushButton, QSizePolicy, QComboBox, QScrollArea, QFrame, QProgressBar, QDoubleSpinBox)
import os
from PyQt5.QtCore import Qt, pyqtSignal
from src.core.patient_data import PatientData
from src.gui.components.ruler_tool import RulerTool

pg.setConfigOption('background', '#181825')
pg.setConfigOption('foreground', '#cdd6f4')
pg.setConfigOption('imageAxisOrder', 'row-major')

class CollapsibleSection(QWidget):
    def __init__(self, title, expanded=False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.btn_toggle = QPushButton(f"{'▼' if expanded else '▶'}  {title}")
        self.btn_toggle.setStyleSheet("QPushButton { text-align: left; padding: 10px; background-color: #1e1e2e; color: #cdd6f4; font-weight: bold; border-radius: 4px; border: 1px solid #45475a; margin-bottom: 2px; } QPushButton:hover { background-color: #313244; }")
        self.btn_toggle.clicked.connect(self._toggle)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(5, 5, 5, 10)
        self.content_widget.setVisible(expanded)

        self.layout.addWidget(self.btn_toggle)
        self.layout.addWidget(self.content_widget)

    def _toggle(self):
        is_visible = not self.content_widget.isVisible()
        self.content_widget.setVisible(is_visible)
        title = self.btn_toggle.text().split("  ", 1)[1]
        self.btn_toggle.setText(f"{'▼' if is_visible else '▶'}  {title}")

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
        
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

class MRIViewerWidget(QWidget):
    masks_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volumes = {}
        self.current_volume = None
        self.current_vol_name = None
        self.slice_idx = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self.saved_slice_indices = {}  # Cache slice indices per volume
        self._updating_crosshairs = False
        self.volume_stats = {}
        self.maximized_view = None
        self.masks = {}
        self.current_opacity = 150  # Default alpha opacity out of 255
        self.bejjani_items = []     # Overlay items for Bejjani targeting
        self.bejjani_labels = []    # Text labels for target coordinates
        self.bejjani_slice = None   # Axial slice index where Bejjani targets were computed
        self.bejjani_cache = {}     # Cache for Bejjani targeting per volume

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
        scroll.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: #1e1e2e;
                width: 8px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #45475a;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #585b70;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(12)

        # Volumes Section
        sec_volumes = CollapsibleSection("Loaded Volumes", expanded=True)
        vol_layout = sec_volumes.content_layout
        vol_layout.setAlignment(Qt.AlignTop)
        
        self.tbl_volumes = QTableWidget()
        self.tbl_volumes.setColumnCount(3)
        self.tbl_volumes.setHorizontalHeaderLabels(["Name", "Mod", "Shape"])
        self.tbl_volumes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_volumes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_volumes.setAlternatingRowColors(True)
        self.tbl_volumes.setShowGrid(True)
        self.tbl_volumes.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_volumes.verticalHeader().setVisible(False)
        self.tbl_volumes.setMinimumHeight(150)
        self.tbl_volumes.setStyleSheet("QTableWidget { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; } QHeaderView::section { background-color: #313244; color: #a6adc8; font-weight: bold; border: none; padding: 4px; }")
        sec_volumes.addWidget(self.tbl_volumes)
        scroll_layout.addWidget(sec_volumes)
        
        # AI Segmentation Section
        sec_seg = CollapsibleSection("AI Segmentation", expanded=False)
        self.btn_run_seg = QPushButton("Run AI Segmentation")
        self.btn_run_seg.setStyleSheet("QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; font-size: 13px; padding: 8px; border-radius: 4px; } QPushButton:disabled { background-color: #45475a; color: #a6adc8; }")
        self.btn_run_seg.setEnabled(False)
        self.btn_run_seg.clicked.connect(self._on_segmentation_clicked)
        sec_seg.addWidget(self.btn_run_seg)
        
        # Segmentation Overlay Tools
        self.grp_seg_toolbox = QWidget()
        self.grp_seg_toolbox.setVisible(False)
        box_layout = QVBoxLayout(self.grp_seg_toolbox)
        box_layout.setContentsMargins(0, 10, 0, 0)
        box_layout.setSpacing(0)
        
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        
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
        
        lbl_opacity = QLabel("Opacity:")
        lbl_opacity.setStyleSheet("color: #a6adc8; font-size: 11px;")
        controls_layout.addWidget(lbl_opacity)
        
        self.sld_opacity = QSlider(Qt.Horizontal)
        self.sld_opacity.setRange(0, 255)
        self.sld_opacity.setValue(self.current_opacity)
        self.sld_opacity.valueChanged.connect(self._update_mask_opacity)
        controls_layout.addWidget(self.sld_opacity)
        
        box_layout.addLayout(controls_layout)
        sec_seg.addWidget(self.grp_seg_toolbox)

        # --- PROGRESS INDICATOR ---
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
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #45475a; border-radius: 3px; background-color: #1e1e2e; height: 10px; } QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }")
        prog_layout.addWidget(self.progress_bar)
        
        sec_seg.addWidget(self.grp_progress)

        scroll_layout.addWidget(sec_seg)

        # Electrode Positioning Section
        sec_ep = CollapsibleSection("Electrode Positioning", expanded=False)
        
        # AC/PC Reference
        acpc_row = QHBoxLayout()
        self.btn_toggle_acpc = QPushButton("Show AC/PC Line")
        self.btn_toggle_acpc.setCheckable(True)
        self.btn_toggle_acpc.setStyleSheet("""
            QPushButton { background-color: #313244; color: #cdd6f4; padding: 6px; border-radius: 4px; font-size: 12px; }
            QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; font-weight: bold; }
        """)
        self.btn_toggle_acpc.clicked.connect(self._update_acpc_overlay)
        acpc_row.addWidget(self.btn_toggle_acpc)
        
        self.lbl_acpc_dist = QLabel("Distance: -")
        self.lbl_acpc_dist.setStyleSheet("color: #a6adc8; font-size: 11px;")
        acpc_row.addWidget(self.lbl_acpc_dist)
        sec_ep.addLayout(acpc_row)

        # DBS Planning
        self.wgt_bejjani_container = QWidget()
        bejjani_container_layout = QVBoxLayout(self.wgt_bejjani_container)
        bejjani_container_layout.setContentsMargins(0, 10, 0, 0)
        bejjani_container_layout.setSpacing(6)
        self.wgt_bejjani_container.setEnabled(False)

        bejjani_row = QHBoxLayout()
        bejjani_row.setSpacing(6)

        self.btn_bejjani = QPushButton("Bejjani Target")
        self.btn_bejjani.setToolTip("Auto-detect Bejjani line, RN borders, and STN target at specified lateral offset")
        self.btn_bejjani.setStyleSheet("QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; font-size: 12px; padding: 8px; border-radius: 4px; } QPushButton:disabled { background-color: #45475a; color: #a6adc8; }")
        self.btn_bejjani.clicked.connect(self._run_bejjani_targeting)
        
        self.spn_bejjani_offset = QDoubleSpinBox()
        self.spn_bejjani_offset.setRange(1.0, 10.0)
        self.spn_bejjani_offset.setSingleStep(0.1)
        self.spn_bejjani_offset.setValue(3.5)
        self.spn_bejjani_offset.setSuffix(" mm")
        self.spn_bejjani_offset.setStyleSheet("QDoubleSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 6px; font-size: 12px; } QDoubleSpinBox:disabled { background-color: #1e1e2e; color: #6c7086; }")
        
        bejjani_row.addWidget(self.btn_bejjani, 3)
        bejjani_row.addWidget(self.spn_bejjani_offset, 1)
        bejjani_container_layout.addLayout(bejjani_row)

        self.lbl_bejjani_info = QLabel("")
        self.lbl_bejjani_info.setStyleSheet("color: #a6adc8; font-size: 10px;")
        self.lbl_bejjani_info.setWordWrap(True)
        bejjani_container_layout.addWidget(self.lbl_bejjani_info)

        # Target Spinboxes (Hidden until Bejjani is run)
        self.wgt_targets = QWidget()
        tgt_layout = QGridLayout(self.wgt_targets)
        tgt_layout.setContentsMargins(0, 5, 0, 5)
        
        tgt_layout.addWidget(QLabel("<b>L-STN (rel MC)</b>"), 0, 0, 1, 2)
        tgt_layout.addWidget(QLabel("<b>R-STN (rel MC)</b>"), 0, 2, 1, 2)

        self.spn_l_lat = QDoubleSpinBox(); self.spn_l_lat.setRange(-50, 50); self.spn_l_lat.setSingleStep(0.5); self.spn_l_lat.setPrefix("X: ")
        self.spn_l_ant = QDoubleSpinBox(); self.spn_l_ant.setRange(-50, 50); self.spn_l_ant.setSingleStep(0.5); self.spn_l_ant.setPrefix("Y: ")
        self.spn_l_sup = QDoubleSpinBox(); self.spn_l_sup.setRange(-50, 50); self.spn_l_sup.setSingleStep(0.5); self.spn_l_sup.setPrefix("Z: ")

        self.spn_r_lat = QDoubleSpinBox(); self.spn_r_lat.setRange(-50, 50); self.spn_r_lat.setSingleStep(0.5); self.spn_r_lat.setPrefix("X: ")
        self.spn_r_ant = QDoubleSpinBox(); self.spn_r_ant.setRange(-50, 50); self.spn_r_ant.setSingleStep(0.5); self.spn_r_ant.setPrefix("Y: ")
        self.spn_r_sup = QDoubleSpinBox(); self.spn_r_sup.setRange(-50, 50); self.spn_r_sup.setSingleStep(0.5); self.spn_r_sup.setPrefix("Z: ")

        for sp in (self.spn_l_lat, self.spn_l_ant, self.spn_l_sup, self.spn_r_lat, self.spn_r_ant, self.spn_r_sup):
            sp.setStyleSheet("QDoubleSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 2px; font-size: 11px; }")
            sp.valueChanged.connect(self._on_target_spinboxes_changed)

        tgt_layout.addWidget(self.spn_l_lat, 1, 0, 1, 2)
        tgt_layout.addWidget(self.spn_l_ant, 2, 0, 1, 2)
        tgt_layout.addWidget(self.spn_l_sup, 3, 0, 1, 2)

        tgt_layout.addWidget(self.spn_r_lat, 1, 2, 1, 2)
        tgt_layout.addWidget(self.spn_r_ant, 2, 2, 1, 2)
        tgt_layout.addWidget(self.spn_r_sup, 3, 2, 1, 2)

        self.wgt_targets.setVisible(False)
        bejjani_container_layout.addWidget(self.wgt_targets)

        self.btn_bejjani_clear = QPushButton("Clear Bejjani")
        self.btn_bejjani_clear.setStyleSheet("QPushButton { background-color: #313244; color: #cdd6f4; padding: 5px; border-radius: 4px; font-size: 11px; } QPushButton:disabled { background-color: #1e1e2e; color: #6c7086; }")
        self.btn_bejjani_clear.clicked.connect(self._clear_bejjani)
        bejjani_container_layout.addWidget(self.btn_bejjani_clear)

        sec_ep.addWidget(self.wgt_bejjani_container)
        scroll_layout.addWidget(sec_ep)

        # Tools Section
        sec_tools = CollapsibleSection("Tools", expanded=False)
        
        # Window / Level
        self.lbl_wl = QLabel("W: 0  |  L: 0")
        self.lbl_wl.setStyleSheet("color: black; font-size: 11px; background-color: white; padding: 2px; border-radius: 4px; max-height: 20px;")
        self.lbl_wl.setAlignment(Qt.AlignCenter)
        sec_tools.addWidget(self.lbl_wl)
        
        self.sld_window = QSlider(Qt.Horizontal)
        self.sld_window.setRange(1, 4000)
        self.sld_level = QSlider(Qt.Horizontal)
        self.sld_level.setRange(-2000, 4000)
        
        sec_tools.addWidget(QLabel("Window (Contrast):"))
        sec_tools.addWidget(self.sld_window)
        sec_tools.addWidget(QLabel("Level (Brightness):"))
        sec_tools.addWidget(self.sld_level)
        
        preset_layout = QHBoxLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["Auto", "T1", "T2", "SWAN", "CT"])
        self.btn_apply_preset = QPushButton("Apply")
        preset_layout.addWidget(self.cmb_preset)
        preset_layout.addWidget(self.btn_apply_preset)
        sec_tools.addLayout(preset_layout)
        
        self.btn_reset_wl = QPushButton("🔄 Reset to Auto")
        sec_tools.addWidget(self.btn_reset_wl)

        # Measurement Tools
        lbl_measure = QLabel("Measurement Tools:")
        lbl_measure.setStyleSheet("color: #a6adc8; font-size: 11px; margin-top: 10px;")
        sec_tools.addWidget(lbl_measure)
        
        ruler_btn_layout = QHBoxLayout()
        ruler_btn_layout.setSpacing(8)

        self.btn_ruler_toggle = QPushButton("Ruler")
        self.btn_ruler_toggle.setCheckable(True)
        self.btn_ruler_toggle.setChecked(False)
        self.btn_ruler_toggle.setToolTip("Toggle ruler mode: click two points to measure distance")
        self.btn_ruler_toggle.setStyleSheet("""
            QPushButton { background-color: #313244; color: #cdd6f4; padding: 6px 10px; border-radius: 4px; font-size: 12px; }
            QPushButton:checked { background-color: #f9e2af; color: #11111b; font-weight: bold; }
        """)
        self.btn_ruler_toggle.clicked.connect(self._toggle_ruler_mode)
        ruler_btn_layout.addWidget(self.btn_ruler_toggle)

        self.btn_ruler_show = QPushButton("👁")
        self.btn_ruler_show.setCheckable(True)
        self.btn_ruler_show.setChecked(True)
        self.btn_ruler_show.setFixedSize(30, 30)
        self.btn_ruler_show.setToolTip("Show/hide measurement lines")
        self.btn_ruler_show.setStyleSheet("""
            QPushButton { background-color: #313244; border-radius: 4px; font-size: 14px; }
            QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; }
        """)
        self.btn_ruler_show.clicked.connect(self._toggle_ruler_visibility)
        ruler_btn_layout.addWidget(self.btn_ruler_show)

        self.btn_ruler_clear = QPushButton("🗑")
        self.btn_ruler_clear.setFixedSize(30, 30)
        self.btn_ruler_clear.setToolTip("Clear measurements on current slices")
        self.btn_ruler_clear.setStyleSheet("QPushButton { background-color: #313244; color: #cdd6f4; border-radius: 4px; font-size: 14px; }")
        self.btn_ruler_clear.clicked.connect(self._clear_ruler_current)
        ruler_btn_layout.addWidget(self.btn_ruler_clear)

        sec_tools.addLayout(ruler_btn_layout)
        scroll_layout.addWidget(sec_tools)

        # Progress UI moved to AI Segmentation section

        scroll_layout.addStretch()

        # Prevent input widgets from hijacking the scroll wheel
        def ignore_wheel(event):
            event.ignore()
            
        self.sld_opacity.wheelEvent = ignore_wheel
        self.sld_window.wheelEvent = ignore_wheel
        self.sld_level.wheelEvent = ignore_wheel
        self.cmb_preset.wheelEvent = ignore_wheel
        self.spn_bejjani_offset.wheelEvent = ignore_wheel
        for sp in (self.spn_l_lat, self.spn_l_ant, self.spn_l_sup, self.spn_r_lat, self.spn_r_ant, self.spn_r_sup):
            sp.wheelEvent = ignore_wheel

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

        # Ruler tool instances (one per view)
        self.rulers = {
            'axial': RulerTool(self.view_axial, spacing_xy=(1.0, 1.0), color='#f9e2af'),
            'coronal': RulerTool(self.view_coronal, spacing_xy=(1.0, 1.0), color='#a6e3a1'),
            'sagittal': RulerTool(self.view_sagittal, spacing_xy=(1.0, 1.0), color='#f38ba8'),
        }

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

        # AC/PC Overlay Items
        from src.gui.components.acpc_widget import _LandmarkMarker
        self.acpc_items = []
        for view in (self.view_axial, self.view_coronal, self.view_sagittal):
            ac_marker = _LandmarkMarker(view, 0, 0, '#89dceb', 'AC', draggable=False)
            pc_marker = _LandmarkMarker(view, 0, 0, '#fab387', 'PC', draggable=False)
            line = pg.PlotDataItem(pen=pg.mkPen('#cdd6f4', width=2, style=Qt.DashLine))
            
            ac_marker.set_visible(False)
            pc_marker.set_visible(False)
            line.setVisible(False)
            
            view.addItem(line)
            
            self.acpc_items.append({
                'view_key': view.parent().objectName() if view.parent() else '', # Will not be used directly
                'ac': ac_marker,
                'pc': pc_marker,
                'line': line
            })
            
        self.ac_world = None
        self.pc_world = None
        self.mc_world = None

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

    def _resolve_cache_dir(self, patient_data):
        import os
        clean_path = os.path.normpath(patient_data.file_path)
        pname = patient_data.metadata.get("patient_name", "").strip()
        patient_folder = None
        
        if pname and pname.lower() in clean_path.lower():
            idx = clean_path.lower().find(pname.lower()) + len(pname)
            patient_folder = clean_path[:idx]
            
        if not patient_folder:
            curr = clean_path if os.path.isdir(clean_path) else os.path.dirname(clean_path)
            generic_terms = ['dicom', 'mri', 'ct', 't1', 't2', 'series', 'study', 'ax', 'cor', 'sag']
            while curr != os.path.dirname(curr):
                if any(term in os.path.basename(curr).lower() for term in generic_terms):
                    curr = os.path.dirname(curr)
                else:
                    break
            patient_folder = curr
            
        return os.path.join(patient_folder, "dbs_cache")

    def _run_segmentation(self, vol_name: str):
        patient_data = self.volumes[vol_name]
        
        import os
        import numpy as np
        
        cache_dir = self._resolve_cache_dir(patient_data)
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
            self.wgt_bejjani_container.setEnabled(True)
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
        views_layout = self.layout().itemAt(0).widget().layout()
        views = ['axial', 'coronal', 'sagittal']
        
        for v in views:
            views_layout.removeWidget(getattr(self, f'header_{v}'))
            views_layout.removeWidget(getattr(self, f'view_{v}'))
            getattr(self, f'header_{v}').setVisible(False)
            getattr(self, f'view_{v}').setVisible(False)
            
        if self.maximized_view == view_key:
            self.maximized_view = None
            views_layout.addWidget(self.header_axial, 0, 0)
            views_layout.addWidget(self.view_axial, 1, 0)
            views_layout.addWidget(self.header_coronal, 0, 1)
            views_layout.addWidget(self.view_coronal, 1, 1)
            views_layout.addWidget(self.header_sagittal, 2, 0)
            views_layout.addWidget(self.view_sagittal, 3, 0)
            for v in views:
                getattr(self, f'header_{v}').setVisible(True)
                getattr(self, f'view_{v}').setVisible(True)
            self._setup_grid_layout()
        else:
            self.maximized_view = view_key
            header = getattr(self, f'header_{view_key}')
            view = getattr(self, f'view_{view_key}')
            views_layout.addWidget(header, 0, 0, 1, 2)
            views_layout.addWidget(view, 1, 0, 1, 2)
            header.setVisible(True)
            view.setVisible(True)
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

    def _on_volume_selected(self):
        row = self.tbl_volumes.currentRow()
        if row < 0: return
        vol_name = self.tbl_volumes.item(row, 0).text()
        self._activate_volume(vol_name)
        
        patient_data = self.volumes[vol_name]
        is_t2 = "t2" in vol_name.lower() or "t2" in patient_data.modality.lower()
        self.btn_run_seg.setEnabled(is_t2)
        
        self._load_acpc_data()
        self._update_acpc_overlay()

    def _load_acpc_data(self):
        self.ac_world = None
        self.pc_world = None
        self.mc_world = None
        
        if not self.current_volume:
            return
            
        cache_dir = self._resolve_cache_dir(self.current_volume)
        acpc_base = os.path.join(cache_dir, "acpc")
        if not os.path.isdir(acpc_base): return
        
        import glob
        import json
        files = glob.glob(os.path.join(acpc_base, "*", "acpc_points.json"))
        # Sort by modification time to get the newest
        files.sort(key=os.path.getmtime, reverse=True)
        
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                if 'ac' in data and 'world_mm' in data['ac'] and data['ac']['world_mm']:
                    self.ac_world = np.array(data['ac']['world_mm'])
                if 'pc' in data and 'world_mm' in data['pc'] and data['pc']['world_mm']:
                    self.pc_world = np.array(data['pc']['world_mm'])
                    
                if self.ac_world is not None and self.pc_world is not None:
                    self.mc_world = (self.ac_world + self.pc_world) / 2.0
                    break
            except Exception:
                continue

    def _update_acpc_overlay(self):
        show = self.btn_toggle_acpc.isChecked()
        if not show or self.ac_world is None or self.pc_world is None or self.current_volume is None:
            for item in self.acpc_items:
                item['ac'].set_visible(False)
                item['pc'].set_visible(False)
                item['line'].setVisible(False)
            self.lbl_acpc_dist.setText("Distance: -" if not show else "Distance: Not Found")
            return
            
        dist = np.linalg.norm(self.ac_world - self.pc_world)
        self.lbl_acpc_dist.setText(f"Distance: {dist:.2f} mm")
        
        ac_vox = self.current_volume.world_to_voxel(self.ac_world)
        pc_vox = self.current_volume.world_to_voxel(self.pc_world)
        
        sz, sy, sx = self.current_volume.spacing
        Z, Y, X = self.current_volume.get_shape()
        
        def to_disp(key, voxel):
            vz, vy, vx = voxel
            if key == 'axial':
                return vx * sx, (Y - 1 - vy) * sy
            elif key == 'coronal':
                return vx * sx, vz * sz
            else: # sagittal
                return vy * sy, vz * sz
                
        keys = ['axial', 'coronal', 'sagittal']
        
        zi, yi, xi = self.slice_idx.get('axial', 0), self.slice_idx.get('coronal', 0), self.slice_idx.get('sagittal', 0)
        ac_z, ac_y, ac_x = map(lambda v: int(round(v)), ac_vox)
        pc_z, pc_y, pc_x = map(lambda v: int(round(v)), pc_vox)
        
        # User requested exact slice matching for Axial and Coronal, but always visible in Sagittal
        ac_vis = {'axial': ac_z == zi, 'coronal': ac_y == yi, 'sagittal': True}
        pc_vis = {'axial': pc_z == zi, 'coronal': pc_y == yi, 'sagittal': True}
        
        for i, item in enumerate(self.acpc_items):
            vk = keys[i]
            ax, ay = to_disp(vk, ac_vox)
            px, py = to_disp(vk, pc_vox)
            
            item['ac'].update_pos(ax, ay)
            item['pc'].update_pos(px, py)
            item['line'].setData([ax, px], [ay, py])
            
            item['ac'].set_visible(ac_vis[vk])
            item['pc'].set_visible(pc_vis[vk])
            item['line'].setVisible(True)

    def _on_segmentation_clicked(self):
        if self.current_vol_name:
            self._run_segmentation(self.current_vol_name)

    def _activate_volume(self, vol_name: str):
        if vol_name not in self.volumes: return
        
        # Save slice indices for the current volume before switching
        if self.current_vol_name and hasattr(self, 'slice_idx'):
            self.saved_slice_indices[self.current_vol_name] = self.slice_idx.copy()

        # Save Bejjani state before switching
        if self.current_vol_name:
            for item in self.bejjani_items:
                if hasattr(item, 'setVisible'):
                    item.setVisible(False)
                elif hasattr(item, 'set_visible'):
                    item.set_visible(False)
            self.bejjani_cache[self.current_vol_name] = {
                'items': self.bejjani_items,
                'labels': self.bejjani_labels,
                'slice': self.bejjani_slice,
                'info': self.lbl_bejjani_info.text()
            }

        self.current_vol_name = vol_name
        self.current_volume = self.volumes[vol_name]
        
        # Restore Bejjani state if available
        if vol_name in self.bejjani_cache:
            cache = self.bejjani_cache[vol_name]
            self.bejjani_items = cache['items']
            self.bejjani_labels = cache['labels']
            self.bejjani_slice = cache['slice']
            self.lbl_bejjani_info.setText(cache['info'])
        else:
            self.bejjani_items = []
            self.bejjani_labels = []
            self.bejjani_slice = None
            self.lbl_bejjani_info.setText("")
        
        # Check toolbox visibility based on mask availability
        has_mask = vol_name in self.masks
        self.grp_seg_toolbox.setVisible(has_mask)
        self.wgt_bejjani_container.setEnabled(has_mask)
        
        z, y, x = self.current_volume.get_shape()
        
        # Restore saved slice indices if available, otherwise default to middle
        if vol_name in self.saved_slice_indices:
            self.slice_idx = self.saved_slice_indices[vol_name].copy()
        else:
            self.slice_idx = {
                'axial': min(z // 2, z - 1),
                'coronal': min(y // 2, y - 1),
                'sagittal': min(x // 2, x - 1)
            }
        
        # Update ruler spacing for the new volume
        for view_key, ruler in self.rulers.items():
            ruler.clear_all()
            # View coordinates are already in physical units (mm) due to ImageItem scaling
            ruler.set_spacing(1.0, 1.0)
        
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

        w = self.sld_window.value()
        l = self.sld_level.value()
        vmin, vmax = l - w/2, l + w/2

        self.img_axial.setImage(ax_slice, autoLevels=False, levels=(vmin, vmax))
        self.img_coronal.setImage(cor_slice, autoLevels=False, levels=(vmin, vmax))
        self.img_sagittal.setImage(sag_slice, autoLevels=False, levels=(vmin, vmax))
        
        # Apply aspect ratio scaling based on physical spacing
        sz, sy, sx = self.current_volume.spacing
        self.img_axial.setTransform(pg.QtGui.QTransform().scale(sx, sy))
        self.img_coronal.setTransform(pg.QtGui.QTransform().scale(sx, sz))
        self.img_sagittal.setTransform(pg.QtGui.QTransform().scale(sy, sz))

        if self.current_vol_name in self.masks and self.btn_toggle_mask.isChecked():
            mask_vol = self.masks[self.current_vol_name]
            
            ax_mask = np.flipud(mask_vol[zi, :, :])
            cor_mask = mask_vol[:, yi, :]
            sag_mask = mask_vol[:, :, xi]
            
            self.mask_axial.setImage(ax_mask, autoLevels=False)
            self.mask_coronal.setImage(cor_mask, autoLevels=False)
            self.mask_sagittal.setImage(sag_mask, autoLevels=False)
            
            self.mask_axial.setTransform(pg.QtGui.QTransform().scale(sx, sy))
            self.mask_coronal.setTransform(pg.QtGui.QTransform().scale(sx, sz))
            self.mask_sagittal.setTransform(pg.QtGui.QTransform().scale(sy, sz))
        else:
            self.mask_axial.clear()
            self.mask_coronal.clear()
            self.mask_sagittal.clear()

        # Update ruler slice visibility
        self.rulers['axial'].set_slice(zi)
        self.rulers['coronal'].set_slice(yi)
        self.rulers['sagittal'].set_slice(xi)

        # Update Bejjani overlays visibility (only visible on the specific target slice)
        bejjani_visible = (self.bejjani_slice is not None and zi == self.bejjani_slice)
        for item in self.bejjani_items:
            if hasattr(item, 'set_visible'):
                item.set_visible(bejjani_visible)
            else:
                item.setVisible(bejjani_visible)

        self._update_crosshair_positions()
        self._update_window_level()
        self._update_acpc_overlay()

    def _update_crosshair_positions(self):
        if self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        sz, sy, sx = self.current_volume.spacing
        self._updating_crosshairs = True
        
        self.crosshairs['axial']['h'].setPos((y - 1 - self.slice_idx['coronal']) * sy)
        self.crosshairs['axial']['v'].setPos(self.slice_idx['sagittal'] * sx)
        
        self.crosshairs['coronal']['h'].setPos(self.slice_idx['axial'] * sz)
        self.crosshairs['coronal']['v'].setPos(self.slice_idx['sagittal'] * sx)
        
        self.crosshairs['sagittal']['h'].setPos(self.slice_idx['axial'] * sz)
        self.crosshairs['sagittal']['v'].setPos(self.slice_idx['coronal'] * sy)
        
        self._updating_crosshairs = False

    def _sync_crosshair(self, view, axis):
        if self._updating_crosshairs or self.current_volume is None: return
        z, y, x = self.current_volume.get_shape()
        sz, sy, sx = self.current_volume.spacing
        val = self.crosshairs[view][axis].value()

        if view == 'axial':
            if axis == 'h':
                pos = int(round(val / sy))
                self.slice_idx['coronal'] = max(0, min(y-1, y - 1 - pos))
            else:
                pos = int(round(val / sx))
                self.slice_idx['sagittal'] = max(0, min(x-1, pos))
        elif view == 'coronal':
            if axis == 'h':
                pos = int(round(val / sz))
                self.slice_idx['axial'] = max(0, min(z-1, pos))
            else:
                pos = int(round(val / sx))
                self.slice_idx['sagittal'] = max(0, min(x-1, pos))
        elif view == 'sagittal':
            if axis == 'h':
                pos = int(round(val / sz))
                self.slice_idx['axial'] = max(0, min(z-1, pos))
            else:
                pos = int(round(val / sy))
                self.slice_idx['coronal'] = max(0, min(y-1, pos))
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

    # ── Ruler Tool Actions ──────────────────────────────────────

    def _toggle_ruler_mode(self):
        """Activate or deactivate ruler measurement mode on all views."""
        active = self.btn_ruler_toggle.isChecked()
        for ruler in self.rulers.values():
            ruler.set_active(active)

    def _toggle_ruler_visibility(self):
        """Show or hide all measurement lines on all views."""
        visible = self.btn_ruler_show.isChecked()
        for ruler in self.rulers.values():
            ruler.set_visible(visible)

    def _clear_ruler_current(self):
        """Clear measurements on the currently displayed slices."""
        for ruler in self.rulers.values():
            ruler.clear_slice()

    # ── Bejjani DBS Targeting ──────────────────────────────────

    def _clear_bejjani(self):
        """Remove all Bejjani overlay items from the axial view."""
        for item in self.bejjani_items:
            self.view_axial.removeItem(item)
        self.bejjani_items.clear()
        self.bejjani_labels.clear()
        self.bejjani_slice = None
        self.lbl_bejjani_info.setText("")
        if self.current_vol_name in self.bejjani_cache:
            del self.bejjani_cache[self.current_vol_name]

    def _run_bejjani_targeting(self):
        """
        Automatic Bejjani targeting method:
        1. Find the axial slice where the Red Nucleus (RN, class=2) is largest
        2. Navigate to that slice
        3. Draw the Bejjani line (upper border of RN)
        4. Draw outer lateral border lines for left and right RN
        5. Mark STN target points at 3.5mm lateral from each outer RN border
        """
        if self.current_vol_name is None or self.current_vol_name not in self.masks:
            self.lbl_bejjani_info.setText("⚠ Run segmentation first!")
            return

        mask_vol = self.masks[self.current_vol_name]
        spacing = self.current_volume.spacing  # (sz, sy, sx)

        # Clear previous overlays
        self._clear_bejjani()

        # ── Step 1: Find the slice with the largest STN + RN area ──
        stn_class = 1  # STN
        rn_class = 2  # RN
        combined_areas = []
        for zi in range(mask_vol.shape[0]):
            stn_area = np.sum(mask_vol[zi, :, :] == stn_class)
            rn_area = np.sum(mask_vol[zi, :, :] == rn_class)
            combined_areas.append(stn_area + rn_area)

        best_slice = int(np.argmax(combined_areas))
        if combined_areas[best_slice] == 0:
            self.lbl_bejjani_info.setText("⚠ No STN or RN detected in any slice!")
            return

        # Navigate to the best slice
        self.slice_idx['axial'] = best_slice
        self._render_mpr()
        self._update_slice_labels()

        # ── Step 2: Analyze RN on the best slice (in display coords) ──
        ax_mask = np.flipud(mask_vol[best_slice, :, :])
        H, W = ax_mask.shape
        rn_mask = (ax_mask == rn_class)

        # pyqtgraph row-major: data[row, col] → display (x=col, y=row)
        rn_coords = np.argwhere(rn_mask)  # each row = [row, col]
        if len(rn_coords) == 0:
            self.lbl_bejjani_info.setText("⚠ RN detected but missing on the combined max slice!")
            return

        rn_rows = rn_coords[:, 0]
        rn_cols = rn_coords[:, 1]

        # Upper border of RN = maximum row index (highest y in display)
        # We add 1.0 so the line is drawn at the TOP edge of the topmost pixel
        upper_border_y = float(np.max(rn_rows)) + 1.0

        # ── Step 3: Separate left and right RN and find dense borders ──
        midline_x = W / 2.0
        left_mask = rn_cols < midline_x
        right_mask = rn_cols >= midline_x

        if not np.any(left_mask) or not np.any(right_mask):
            self.lbl_bejjani_info.setText("⚠ Could not separate left/right RN!")
            return

        # Find leftmost column of left RN that has >= 3 voxels (ignore 1-2 voxel outliers)
        left_cols = rn_cols[left_mask]
        unique_left_cols, counts_left = np.unique(left_cols, return_counts=True)
        valid_left_cols = unique_left_cols[counts_left >= 3]
        if len(valid_left_cols) == 0:
            valid_left_cols = unique_left_cols # Fallback to any if none have 3
        left_outer_x = float(np.min(valid_left_cols))

        # Find rightmost column of right RN that has >= 3 voxels
        right_cols = rn_cols[right_mask]
        unique_right_cols, counts_right = np.unique(right_cols, return_counts=True)
        valid_right_cols = unique_right_cols[counts_right >= 3]
        if len(valid_right_cols) == 0:
            valid_right_cols = unique_right_cols
        # We add 1.0 so the line is drawn at the RIGHT edge of the rightmost pixel
        right_outer_x = float(np.max(valid_right_cols)) + 1.0

        # Center Y of each RN cluster (for vertical line positioning)
        # Add 0.5 to draw in the middle of the pixels
        left_center_y = float(np.mean(rn_rows[left_mask])) + 0.5
        right_center_y = float(np.mean(rn_rows[right_mask])) + 0.5

        # ── Step 4: Compute offset in physical units ──
        offset_mm = self.spn_bejjani_offset.value()
        pixel_spacing_x = spacing[2]
        pixel_spacing_y = spacing[1]

        # Convert everything to view coordinates (mm) since ImageItem is scaled
        left_outer_x_mm = left_outer_x * pixel_spacing_x
        right_outer_x_mm = right_outer_x * pixel_spacing_x
        upper_border_y_mm = upper_border_y * pixel_spacing_y
        left_center_y_mm = left_center_y * pixel_spacing_y
        right_center_y_mm = right_center_y * pixel_spacing_y

        left_target_x_mm = left_outer_x_mm - offset_mm    # More lateral = more left
        right_target_x_mm = right_outer_x_mm + offset_mm   # More lateral = more right

        # ── Step 5: Draw overlays ──

        # 5a. Bejjani Line
        margin_mm = 15.0
        bejjani_line = pg.PlotDataItem(
            [left_outer_x_mm - margin_mm, right_outer_x_mm + margin_mm],
            [upper_border_y_mm, upper_border_y_mm],
            pen=pg.mkPen('#f9e2af', width=2, style=Qt.DashLine)
        )
        self.view_axial.addItem(bejjani_line)
        self.bejjani_items.append(bejjani_line)

        bejjani_lbl = pg.TextItem(
            text="Bejjani Line (upper RN border)",
            color='#f9e2af', anchor=(0.5, 1.3)
        )
        bejjani_lbl.setFont(pg.QtGui.QFont("Arial", 8, pg.QtGui.QFont.Bold))
        bejjani_lbl.setPos((left_outer_x_mm + right_outer_x_mm) / 2, upper_border_y_mm)
        self.view_axial.addItem(bejjani_lbl)
        self.bejjani_items.append(bejjani_lbl)

        # 5b. Left RN outer border
        vert_extent_mm = 12.0
        left_border_line = pg.PlotDataItem(
            [left_outer_x_mm, left_outer_x_mm],
            [left_center_y_mm - vert_extent_mm, left_center_y_mm + vert_extent_mm],
            pen=pg.mkPen('#89b4fa', width=2, style=Qt.DashLine)
        )
        self.view_axial.addItem(left_border_line)
        self.bejjani_items.append(left_border_line)

        # 5c. Right RN outer border
        right_border_line = pg.PlotDataItem(
            [right_outer_x_mm, right_outer_x_mm],
            [right_center_y_mm - vert_extent_mm, right_center_y_mm + vert_extent_mm],
            pen=pg.mkPen('#89b4fa', width=2, style=Qt.DashLine)
        )
        self.view_axial.addItem(right_border_line)
        self.bejjani_items.append(right_border_line)

        from src.gui.components.acpc_widget import _LandmarkMarker
        self.l_stn_marker = _LandmarkMarker(
            self.view_axial,
            left_target_x_mm, upper_border_y_mm,
            '#f38ba8', 'L-STN\nDrag Me',
            draggable=True,
            on_dragged=self._on_target_dragged_L
        )
        self.bejjani_items.append(self.l_stn_marker)

        # 5e. Right STN target
        self.r_stn_marker = _LandmarkMarker(
            self.view_axial,
            right_target_x_mm, upper_border_y_mm,
            '#f38ba8', 'R-STN\nDrag Me',
            draggable=True,
            on_dragged=self._on_target_dragged_R
        )
        self.bejjani_items.append(self.r_stn_marker)

        # 5f. Horizontal ruler lines
        left_ruler = pg.PlotDataItem(
            [left_target_x_mm, left_outer_x_mm],
            [upper_border_y_mm - 1.5, upper_border_y_mm - 1.5],
            pen=pg.mkPen('#cba6f7', width=1.5, style=Qt.DashDotLine)
        )
        self.view_axial.addItem(left_ruler)
        self.bejjani_items.append(left_ruler)

        left_ruler_lbl = pg.TextItem(text=f"{offset_mm} mm", color='#cba6f7', anchor=(0.5, 1.3))
        left_ruler_lbl.setFont(pg.QtGui.QFont("Arial", 7))
        left_ruler_lbl.setPos((left_target_x_mm + left_outer_x_mm) / 2, upper_border_y_mm - 1.5)
        self.view_axial.addItem(left_ruler_lbl)
        self.bejjani_items.append(left_ruler_lbl)

        right_ruler = pg.PlotDataItem(
            [right_outer_x_mm, right_target_x_mm],
            [upper_border_y_mm - 1.5, upper_border_y_mm - 1.5],
            pen=pg.mkPen('#cba6f7', width=1.5, style=Qt.DashDotLine)
        )
        self.view_axial.addItem(right_ruler)
        self.bejjani_items.append(right_ruler)

        right_ruler_lbl = pg.TextItem(text=f"{offset_mm} mm", color='#cba6f7', anchor=(0.5, 1.3))
        right_ruler_lbl.setFont(pg.QtGui.QFont("Arial", 7))
        right_ruler_lbl.setPos((right_outer_x_mm + right_target_x_mm) / 2, upper_border_y_mm - 1.5)
        self.view_axial.addItem(right_ruler_lbl)
        self.bejjani_items.append(right_ruler_lbl)

        # ── Step 6: Compute world coordinates for the targets ──
        # Convert display (x=col, y=row) back to voxel coordinates
        # We use the unscaled pixel target variables for this
        left_target_x = left_outer_x - (offset_mm / pixel_spacing_x)
        right_target_x = right_outer_x + (offset_mm / pixel_spacing_x)
        # We subtract the 1.0 we added for display to get back the actual row index
        orig_row_upper = H - 1 - int(upper_border_y - 1.0)
        left_voxel = (best_slice, orig_row_upper, int(left_target_x))
        # We subtract the 1.0 we added to right_target_x to get the voxel index inside the target
        right_voxel = (best_slice, orig_row_upper, int(right_target_x - 1.0))

        left_world = self.current_volume.voxel_to_world(left_voxel)
        right_world = self.current_volume.voxel_to_world(right_voxel)

        if hasattr(self, 'mc_world') and self.mc_world is not None:
            def fmt_mc(target_world):
                rel = target_world - self.mc_world
                x, y, z = rel
                
                x_str = f"{abs(x):.2f} {'Right' if x > 0 else 'Left'} Lateral"
                y_str = f"{abs(y):.2f} {'Anterior' if y > 0 else 'Posterior'}"
                z_str = f"{abs(z):.2f} {'Superior' if z > 0 else 'Inferior'}"
                
                return f"({x:.2f}, {y:.2f}, {z:.2f}) mm\n   => ({x_str}, {y_str}, {z_str})"
                
            info_text = (
                f"Slice: {best_slice + 1}\n"
                f"─────────────────────\n"
                f"L-STN Target (Rel to MC):\n   {fmt_mc(left_world)}\n\n"
                f"R-STN Target (Rel to MC):\n   {fmt_mc(right_world)}"
            )
            
            # Populate SpinBoxes without triggering valueChanged
            self.spn_l_lat.blockSignals(True)
            self.spn_l_ant.blockSignals(True)
            self.spn_l_sup.blockSignals(True)
            self.spn_r_lat.blockSignals(True)
            self.spn_r_ant.blockSignals(True)
            self.spn_r_sup.blockSignals(True)
            
            l_rel = left_world - self.mc_world
            self.spn_l_lat.setValue(l_rel[0])
            self.spn_l_ant.setValue(l_rel[1])
            self.spn_l_sup.setValue(l_rel[2])
            
            r_rel = right_world - self.mc_world
            self.spn_r_lat.setValue(r_rel[0])
            self.spn_r_ant.setValue(r_rel[1])
            self.spn_r_sup.setValue(r_rel[2])
            
            self.spn_l_lat.blockSignals(False)
            self.spn_l_ant.blockSignals(False)
            self.spn_l_sup.blockSignals(False)
            self.spn_r_lat.blockSignals(False)
            self.spn_r_ant.blockSignals(False)
            self.spn_r_sup.blockSignals(False)
            
            self.lbl_bejjani_info.setVisible(False)
            self.wgt_targets.setVisible(True)
        else:
            info_text = (
                f"Slice: {best_slice + 1} (max STN+RN area)\n"
                f"─────────────────────\n"
                f"L-STN world: ({left_world[0]:.2f}, {left_world[1]:.2f}, {left_world[2]:.2f}) mm\n"
                f"R-STN world: ({right_world[0]:.2f}, {right_world[1]:.2f}, {right_world[2]:.2f}) mm"
            )
            self.lbl_bejjani_info.setVisible(True)
            self.wgt_targets.setVisible(False)
            
        self.lbl_bejjani_info.setText(info_text)
        
        # Set the target slice so it only displays when scrolling to it
        self.bejjani_slice = best_slice
        
        # ── Step 7: Save to Patient Directory ──
        import os
        import json
        cache_dir = self._resolve_cache_dir(self.current_volume)
        pipeline_dir = os.path.join(cache_dir, "pipeline", self.current_vol_name)
        os.makedirs(pipeline_dir, exist_ok=True)
        
        target_data = {
            "volume_name": self.current_vol_name,
            "modality": self.current_volume.modality,
            "slice_index": best_slice,
            "left_stn": {
                "voxel": [int(v) for v in left_voxel],
                "world_mm": [round(float(v), 3) for v in left_world]
            },
            "right_stn": {
                "voxel": [int(v) for v in right_voxel],
                "world_mm": [round(float(v), 3) for v in right_world]
            }
        }
        
        save_path = os.path.join(pipeline_dir, "bejjani_targets.json")
        try:
            with open(save_path, "w") as f:
                json.dump(target_data, f, indent=4)
        except Exception as e:
            print(f"Error saving Bejjani targets: {e}")

    def _on_target_spinboxes_changed(self):
        if not hasattr(self, 'mc_world') or self.mc_world is None or self.current_volume is None:
            return
            
        if not hasattr(self, 'l_stn_marker') or not hasattr(self, 'r_stn_marker'):
            return

        sender = self.sender()
        sz, sy, sx = self.current_volume.spacing
        Z, Y, X = self.current_volume.get_shape()

        if sender in (self.spn_l_lat, self.spn_l_ant, self.spn_l_sup):
            # 1. Read values
            l_rel = np.array([self.spn_l_lat.value(), self.spn_l_ant.value(), self.spn_l_sup.value()])
            # 2. Compute absolute world
            l_world = self.mc_world + l_rel
            # 3. Compute voxel
            l_vox = self.current_volume.world_to_voxel(l_world)
            # 4. Compute display coords on axial
            lax, lay = l_vox[2] * sx, (Y - 1 - l_vox[1]) * sy
            self.l_stn_marker.update_pos(lax, lay)
            
        elif sender in (self.spn_r_lat, self.spn_r_ant, self.spn_r_sup):
            r_rel = np.array([self.spn_r_lat.value(), self.spn_r_ant.value(), self.spn_r_sup.value()])
            r_world = self.mc_world + r_rel
            r_vox = self.current_volume.world_to_voxel(r_world)
            rax, ray = r_vox[2] * sx, (Y - 1 - r_vox[1]) * sy
            self.r_stn_marker.update_pos(rax, ray)

    def _on_target_dragged_L(self, px, py):
        self._update_spinboxes_from_drag(px, py, is_left=True)

    def _on_target_dragged_R(self, px, py):
        self._update_spinboxes_from_drag(px, py, is_left=False)

    def _update_spinboxes_from_drag(self, px, py, is_left):
        if not hasattr(self, 'mc_world') or self.mc_world is None or self.current_volume is None: return
        
        sz, sy, sx = self.current_volume.spacing
        Z, Y, X = self.current_volume.get_shape()
        zi = self.bejjani_slice
        
        # Display to Voxel
        dx_u, dy_u = px / sx, py / sy
        vx, vy = dx_u, Y - 1 - dy_u
        
        world = self.current_volume.voxel_to_world((zi, vy, vx))
        rel = world - self.mc_world
        
        if is_left:
            self.spn_l_lat.blockSignals(True)
            self.spn_l_ant.blockSignals(True)
            self.spn_l_sup.blockSignals(True)
            
            self.spn_l_lat.setValue(rel[0])
            self.spn_l_ant.setValue(rel[1])
            self.spn_l_sup.setValue(rel[2])
            
            self.spn_l_lat.blockSignals(False)
            self.spn_l_ant.blockSignals(False)
            self.spn_l_sup.blockSignals(False)
        else:
            self.spn_r_lat.blockSignals(True)
            self.spn_r_ant.blockSignals(True)
            self.spn_r_sup.blockSignals(True)
            
            self.spn_r_lat.setValue(rel[0])
            self.spn_r_ant.setValue(rel[1])
            self.spn_r_sup.setValue(rel[2])
            
            self.spn_r_lat.blockSignals(False)
            self.spn_r_ant.blockSignals(False)
            self.spn_r_sup.blockSignals(False)