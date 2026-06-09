import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QSlider, QLabel, 
                             QGroupBox, QPushButton, QSizePolicy, QScrollArea, QFrame,
                             QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QComboBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from src.gui.components.ruler_tool import RulerTool

# Maintain existing pyqtgraph settings
pg.setConfigOption('background', '#181825')
pg.setConfigOption('foreground', '#cdd6f4')
pg.setConfigOption('imageAxisOrder', 'row-major')


class RegistrationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Internal state
        self.volumes = {}  # Dictionary to store loaded PatientData objects
        self.masks = {}    # Dictionary to store AI segmentation masks passed from the app
        
        self.fixed_name = None
        self.moving_name = None
        
        self.fixed_data = None
        self.moving_data = None
        
        self.fixed_volume = None
        self.moving_volume = None
        self.transformed_volume = None
        self.transformed_mask = None  # Stores the spatially registered mask layer
        
        self.slice_fixed = 0
        self.slice_moving = 0
        
        self.overlay_opacity = 0.5
        self._show_overlay = True

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ============ LEFT: 2x2 Grid for Views ============
        views_container = QWidget()
        views_layout = QGridLayout(views_container)
        views_layout.setContentsMargins(4, 4, 4, 4)
        views_layout.setSpacing(4)

        # Create PlotWidgets and ImageItems
        self.view_fixed, self.img_fixed = self._create_plot_widget()
        self.view_moving, self.img_moving = self._create_plot_widget()
        
        # Result view requires stacked image items (fixed bg -> transformed overlay -> seg mask top)
        self.view_result = pg.PlotWidget()
        self.view_result.hideAxis('left')
        self.view_result.hideAxis('bottom')
        self.view_result.setMouseEnabled(x=True, y=True)
        self.view_result.setMenuEnabled(False)
        self.view_result.setAspectLocked(lock=True, ratio=1.0)
        
        self.img_result_fixed = pg.ImageItem()
        self.img_result_mask = pg.ImageItem()          # Segmentation Mask Layer
        self.img_result_transformed = pg.ImageItem()   # Transformed Anatomy Layer
        
        # Setup Mask Colormap (Match MRI Viewer)
        self._apply_mask_lut(150)

        
        # Setup Transformed Colormap (warm tint) with a transparent background
        lut = pg.colormap.get('magma').getLookupTable()
        # Ensure the LUT has an alpha channel (RGBA)
        if lut.shape[1] == 3:
            alpha_channel = np.ones((lut.shape[0], 1), dtype=lut.dtype) * 255
            lut = np.hstack([lut, alpha_channel])

        lut[0] = [0, 0, 0, 0]  # Force the 0-value (background) to be 100% transparent
        self.img_result_transformed.setLookupTable(lut)
        self.img_result_transformed.setOpacity(self.overlay_opacity)
        
        # --- CRITICAL RE-ORDERING: Add mask LAST so it is rendered on top of everything ---
        self.view_result.addItem(self.img_result_fixed)
        self.view_result.addItem(self.img_result_transformed)
        self.view_result.addItem(self.img_result_mask)

        # Ruler tool instances (one per view)
        self.rulers = {
            'fixed': RulerTool(self.view_fixed, spacing_xy=(1.0, 1.0), color='#f9e2af'),
            'moving': RulerTool(self.view_moving, spacing_xy=(1.0, 1.0), color='#a6e3a1'),
            'result': RulerTool(self.view_result, spacing_xy=(1.0, 1.0), color='#f38ba8'),
        }

        # Create headers (with view keys attached for slice stepping)
        self.header_fixed = self._create_header("Fixed Image (Target Space)", view_key='fixed')
        self.header_moving = self._create_header("Moving Image", view_key='moving')
        
        # Result view shares 'fixed' slice domain!
        self.header_result = self._create_header("Registration Result (3-Way Overlay)", with_overlay_controls=True, view_key='fixed_result')

        # Add to Grid: Row 0 & 1 -> Fixed & Moving
        views_layout.addWidget(self.header_fixed, 0, 0)
        views_layout.addWidget(self.view_fixed, 1, 0)
        
        views_layout.addWidget(self.header_moving, 0, 1)
        views_layout.addWidget(self.view_moving, 1, 1)
        
        # Add to Grid: Row 2 & 3 -> Result spanning both columns
        views_layout.addWidget(self.header_result, 2, 0, 1, 2)
        views_layout.addWidget(self.view_result, 3, 0, 1, 2)

        views_layout.setRowStretch(1, 1)
        views_layout.setRowStretch(3, 2) 
        main_layout.addWidget(views_container, 3)

        # ============ RIGHT: Sidebar Menu ============
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
        
        grp_volumes = QGroupBox("📁 Loaded Volumes")
        grp_volumes.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        vol_layout = QVBoxLayout(grp_volumes)
        vol_layout.setContentsMargins(10, 10, 10, 10)
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
        self.tbl_volumes.setStyleSheet("QTableWidget { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; } QHeaderView::section { background-color: #313244; color: #a6adc8; font-weight: bold; border: none; padding: 4px; }")
        vol_layout.addWidget(self.tbl_volumes)
        
        btn_layout = QHBoxLayout()
        self.btn_set_fixed = QPushButton("Set as Fixed")
        self.btn_set_fixed.setStyleSheet("background-color: #45475a; color: #cdd6f4; padding: 4px;")
        self.btn_set_fixed.clicked.connect(self._assign_fixed) 
        
        self.btn_set_moving = QPushButton("Set as Moving")
        self.btn_set_moving.setStyleSheet("background-color: #45475a; color: #cdd6f4; padding: 4px;")
        self.btn_set_moving.clicked.connect(self._assign_moving)
        
        btn_layout.addWidget(self.btn_set_fixed)
        btn_layout.addWidget(self.btn_set_moving)
        vol_layout.addLayout(btn_layout)
        vol_layout.addStretch()
        scroll_layout.addWidget(grp_volumes)

        grp_controls = QGroupBox("⚙️ Registration Controls")
        grp_controls.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        ctrl_layout = QVBoxLayout(grp_controls)
        ctrl_layout.setContentsMargins(10, 10, 10, 10)
        ctrl_layout.setSpacing(8)
        ctrl_layout.setAlignment(Qt.AlignTop)

        # Display Mode Selection
        self.cmb_display_mode = QComboBox()
        self.cmb_display_mode.addItems(["Full Anatomy Overlay", "Electrode Extraction"])
        self.cmb_display_mode.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 6px; border-radius: 4px;")
        self.cmb_display_mode.currentIndexChanged.connect(lambda: self._update_views())
        ctrl_layout.addWidget(QLabel("Result Display Mode:"))
        ctrl_layout.addWidget(self.cmb_display_mode)
        
        self.btn_run_reg = QPushButton("▶ Run Co-Registration")
        self.btn_run_reg.setStyleSheet("QPushButton { background-color: #89b4fa; color: #11111b; font-weight: bold; font-size: 13px; padding: 8px; border-radius: 6px; margin-top: 10px; }")
        self.btn_run_reg.clicked.connect(self._run_coregistration)
        
        self.btn_reset = QPushButton("Reset Views")
        self.btn_reset.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 6px; border-radius: 4px;")
        self.btn_reset.clicked.connect(self._reset_views)
        
        # Progress UI
        self.grp_progress = QWidget()
        prog_layout = QVBoxLayout(self.grp_progress)
        prog_layout.setContentsMargins(0, 5, 0, 0)
        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet("color: #f9e2af; font-size: 11px;")
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 0)
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setStyleSheet("QProgressBar { border: 1px solid #45475a; border-radius: 3px; background-color: #1e1e2e; height: 10px; } QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }")
        prog_layout.addWidget(self.lbl_progress)
        prog_layout.addWidget(self.prog_bar)
        self.grp_progress.setVisible(False)
        
        ctrl_layout.addWidget(self.btn_run_reg)
        ctrl_layout.addWidget(self.grp_progress)
        ctrl_layout.addWidget(self.btn_reset)
        ctrl_layout.addStretch()
        
        scroll_layout.addWidget(grp_controls)
        
        # --- Segmentation Overlay Controls ---
        self.grp_seg_controls = QGroupBox("🧠 Segmentation Overlay")
        self.grp_seg_controls.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        seg_layout = QVBoxLayout(self.grp_seg_controls)
        seg_layout.setAlignment(Qt.AlignTop)
        
        mask_controls_layout = QHBoxLayout()
        self.btn_toggle_mask = QPushButton("👁 Toggle")
        self.btn_toggle_mask.setCheckable(True)
        self.btn_toggle_mask.setChecked(True)
        self.btn_toggle_mask.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 4px; border-radius: 2px;")
        self.btn_toggle_mask.clicked.connect(self._toggle_mask_visibility)
        
        self.slider_mask_opacity = QSlider(Qt.Horizontal)
        self.slider_mask_opacity.setRange(0, 255)
        self.slider_mask_opacity.setValue(150)
        self.slider_mask_opacity.valueChanged.connect(self._on_mask_opacity_changed)
        
        mask_controls_layout.addWidget(QLabel("Opacity:"))
        mask_controls_layout.addWidget(self.slider_mask_opacity)
        mask_controls_layout.addWidget(self.btn_toggle_mask)
        seg_layout.addLayout(mask_controls_layout)
        seg_layout.addStretch()
        scroll_layout.addWidget(self.grp_seg_controls)

        # --- MEASUREMENT TOOLS GROUP ---
        grp_ruler = QGroupBox("📏 Measurement Tools")
        grp_ruler.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }")
        ruler_layout = QVBoxLayout(grp_ruler)
        ruler_layout.setContentsMargins(10, 10, 10, 10)
        ruler_layout.setSpacing(6)
        ruler_layout.setAlignment(Qt.AlignTop)

        ruler_btn_layout = QHBoxLayout()
        ruler_btn_layout.setSpacing(8)

        self.btn_ruler_toggle = QPushButton("📏 Ruler")
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

        ruler_layout.addLayout(ruler_btn_layout)
        ruler_layout.addStretch()
        scroll_layout.addWidget(grp_ruler)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)
        
        main_layout.addWidget(sidebar, 1)

    def _create_plot_widget(self):
        """Helper to create standardized image view"""
        view = pg.PlotWidget()
        view.hideAxis('left')
        view.hideAxis('bottom')
        view.setMouseEnabled(x=True, y=True)
        view.setMenuEnabled(False)
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view.setAspectLocked(lock=True, ratio=1.0)
        
        img_item = pg.ImageItem()
        view.addItem(img_item)
        return view, img_item

    def _create_header(self, title, with_overlay_controls=False, view_key=''):
        """Header with slice traversal traversal connected based on View Key"""
        header = QWidget()
        header.setStyleSheet("background-color: #1e1e2e; padding: 4px;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)
        
        header.lbl_title = QLabel(title)
        header.lbl_title.setStyleSheet("color: #89b4fa; font-weight: bold; font-size: 11px; background-color: transparent;")
        layout.addWidget(header.lbl_title)
        
        layout.addStretch()
        
        if view_key:
            btn_prev = QPushButton("◀")
            btn_prev.setFixedSize(20, 20)
            btn_prev.clicked.connect(lambda _, k=view_key: self._step_slice(k, -1))
            
            header.lbl_slice = QLabel("- / -")
            header.lbl_slice.setStyleSheet("color: #a6adc8; font-size: 10px; min-width: 40px; background-color: #181825; padding: 2px 6px; border-radius: 3px;")
            header.lbl_slice.setAlignment(Qt.AlignCenter)
            
            btn_next = QPushButton("▶")
            btn_next.setFixedSize(20, 20)
            btn_next.clicked.connect(lambda _, k=view_key: self._step_slice(k, 1))

            layout.addWidget(btn_prev)
            layout.addWidget(header.lbl_slice)
            layout.addWidget(btn_next)

        if with_overlay_controls:
            lbl_opacity = QLabel("Opacity:")
            lbl_opacity.setStyleSheet("color: #a6adc8; font-size: 10px; background-color: transparent;")
            layout.addWidget(lbl_opacity)

            self.slider_opacity = QSlider(Qt.Horizontal)
            self.slider_opacity.setRange(0, 100)
            self.slider_opacity.setValue(50)
            self.slider_opacity.setFixedWidth(100)
            self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
            layout.addWidget(self.slider_opacity)

            self.btn_toggle = QPushButton("👁 Toggle")
            self.btn_toggle.setFixedSize(80, 22)
            self.btn_toggle.setStyleSheet("background-color: #45475a; color: #cdd6f4; font-size: 10px; border-radius: 2px;")
            self.btn_toggle.setCheckable(True)
            self.btn_toggle.setChecked(True)
            self.btn_toggle.clicked.connect(self._toggle_overlay)
            layout.addWidget(self.btn_toggle)

        return header

    # ============ Actions Logic ============
    
    def _step_slice(self, view_key, step):
        """Scroll through individual volumes z-axis smoothly"""
        if view_key in ['fixed', 'fixed_result'] and self.fixed_volume is not None:
            self.slice_fixed += step
        elif view_key == 'moving' and self.moving_volume is not None:
            self.slice_moving += step
        self._update_views()

    def _assign_fixed(self):
        selected = self.tbl_volumes.selectionModel().selectedRows()
        if not selected: return
        vol_name = self.tbl_volumes.item(selected[0].row(), 0).text()
        
        if vol_name in self.volumes:
            self.fixed_name = vol_name
            self.fixed_data = self.volumes[vol_name]
            self.fixed_volume = self.fixed_data.volume
            self.slice_fixed = self.fixed_volume.shape[0] // 2
            self.header_fixed.lbl_title.setText(f"Fixed: {vol_name}")
            
            # Update ruler spacing for fixed and result views
            self.rulers['fixed'].clear_all()
            self.rulers['fixed'].set_spacing(1.0, 1.0)
            self.rulers['result'].clear_all()
            self.rulers['result'].set_spacing(1.0, 1.0)
            
            self._update_views(auto_level_fixed=True)

    def _assign_moving(self):
        selected = self.tbl_volumes.selectionModel().selectedRows()
        if not selected: return
        vol_name = self.tbl_volumes.item(selected[0].row(), 0).text()
        
        if vol_name in self.volumes:
            self.moving_name = vol_name
            self.moving_data = self.volumes[vol_name]
            self.moving_volume = self.moving_data.volume
            self.slice_moving = self.moving_volume.shape[0] // 2
            self.header_moving.lbl_title.setText(f"Moving: {vol_name}")
            
            # Update ruler spacing for moving view
            self.rulers['moving'].clear_all()
            self.rulers['moving'].set_spacing(1.0, 1.0)
            
            # Clear previous registration results since a new volume was linked
            self.transformed_volume = None
            self.transformed_mask = None
            self._update_views(auto_level_moving=True)

    def _reset_views(self):
        """Fit standard scaling views on user resents"""
        self.view_fixed.autoRange()
        self.view_moving.autoRange()
        self.view_result.autoRange()

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

    def load_volumes(self, volumes_dict):
        """Sync global loading"""
        self.volumes = volumes_dict
        self.tbl_volumes.setRowCount(0)
        
        for name, patient_data in self.volumes.items():
            row = self.tbl_volumes.rowCount()
            self.tbl_volumes.insertRow(row)
            self.tbl_volumes.setItem(row, 0, QTableWidgetItem(name))
            self.tbl_volumes.setItem(row, 1, QTableWidgetItem(patient_data.modality))
            shape_str = str(patient_data.get_shape()) if hasattr(patient_data, 'get_shape') else "Unknown"
            self.tbl_volumes.setItem(row, 2, QTableWidgetItem(shape_str))

    def load_masks(self, masks_dict):
        """Inject segmentation masks generated from the MRI viewer tab to render on the overlay"""
        self.masks = masks_dict
        self._update_views()

    # ============ Overlay UI Logic ============

    def _apply_mask_lut(self, opacity):
        mask_lut = np.zeros((256, 4), dtype=np.uint8)
        mask_lut[0] = [0, 0, 0, 0]                               # Background transparent
        mask_lut[1] = [255, 0, 0, int(opacity)]                  # Target class 1 (Red)
        mask_lut[2] = [0, 255, 0, int(opacity)]                  # Target class 2 (Green)
        self.img_result_mask.setLookupTable(mask_lut)

    def _on_mask_opacity_changed(self, value):
        self._apply_mask_lut(value)

    def _toggle_mask_visibility(self):
        self.img_result_mask.setVisible(self.btn_toggle_mask.isChecked())

    def _on_opacity_changed(self, value):
        self.overlay_opacity = value / 100.0
        if self._show_overlay:
            self.img_result_transformed.setOpacity(self.overlay_opacity)

    def _toggle_overlay(self):
        self._show_overlay = self.btn_toggle.isChecked()
        if self._show_overlay:
            self.img_result_transformed.setOpacity(self.overlay_opacity)
            self.slider_opacity.setEnabled(True)
        else:
            self.img_result_transformed.setOpacity(0.0)
            self.slider_opacity.setEnabled(False)

    # ============ Rendering Display ============

    def _update_views(self, auto_level_fixed=False, auto_level_moving=False):
        """Core sync view that flips dimensions properly and updates labels."""
        if self.fixed_volume is not None:
            z_f = self.fixed_volume.shape[0]
            self.slice_fixed = max(0, min(self.slice_fixed, z_f - 1))
            
            # Orient flipped correctly like MPR viewer!
            fixed_slice_img = np.flipud(self.fixed_volume[self.slice_fixed])
            
            # === FIX: Safely preserve or automatically compute float levels ===
            if auto_level_fixed:
                self.img_fixed.setImage(fixed_slice_img, autoLevels=True)
                self.img_result_fixed.setImage(fixed_slice_img, autoLevels=True)
            else:
                levels_f = self.img_fixed.getLevels()
                if levels_f is None:
                    self.img_fixed.setImage(fixed_slice_img, autoLevels=True)
                    self.img_result_fixed.setImage(fixed_slice_img, autoLevels=True)
                else:
                    self.img_fixed.setImage(fixed_slice_img, autoLevels=False, levels=levels_f)
                    self.img_result_fixed.setImage(fixed_slice_img, autoLevels=False, levels=levels_f)
                    
            sz, sy, sx = self.fixed_data.spacing
            self.img_fixed.setTransform(pg.QtGui.QTransform().scale(sx, sy))
            self.img_result_fixed.setTransform(pg.QtGui.QTransform().scale(sx, sy))
            
            self.header_fixed.lbl_slice.setText(f"{self.slice_fixed + 1}/{z_f}")
            self.header_result.lbl_slice.setText(f"{self.slice_fixed + 1}/{z_f}")

            # Also update Segmentation Mask if selected!
            fixed_vol_name = self.header_fixed.lbl_title.text().replace("Fixed: ", "")
            if fixed_vol_name in self.masks:
                mask_slice = np.flipud(self.masks[fixed_vol_name][self.slice_fixed])
                # Explicitly set levels to (0, 255) to ensure mapping array values 1,2 directly map to LUT indices 1,2
                self.img_result_mask.setImage(mask_slice, autoLevels=False, levels=(0, 255))
                self.img_result_mask.setTransform(pg.QtGui.QTransform().scale(sx, sy))
            else:
                # === FIX: Use clear() instead of creating a default float64 array ===
                self.img_result_mask.clear()
            
        if self.moving_volume is not None:
            z_m = self.moving_volume.shape[0]
            self.slice_moving = max(0, min(self.slice_moving, z_m - 1))
            
            moving_slice_img = np.flipud(self.moving_volume[self.slice_moving])
            
            # === FIX: Safely preserve or automatically compute float levels ===
            if auto_level_moving:
                self.img_moving.setImage(moving_slice_img, autoLevels=True)
            else:
                levels_m = self.img_moving.getLevels()
                if levels_m is None:
                    self.img_moving.setImage(moving_slice_img, autoLevels=True)
                else:
                    self.img_moving.setImage(moving_slice_img, autoLevels=False, levels=levels_m)
                    
            sz, sy, sx = self.moving_data.spacing
            self.img_moving.setTransform(pg.QtGui.QTransform().scale(sx, sy))
                    
            self.header_moving.lbl_slice.setText(f"{self.slice_moving + 1}/{z_m}")
            
        if self.transformed_volume is not None:
            tf_slice = np.flipud(self.transformed_volume[self.slice_fixed]).copy()
            
            # Application of Targeted Extraction if enabled
            if "Electrode" in self.cmb_display_mode.currentText():
                try:
                    from skimage import morphology
                    if np.any(tf_slice):
                        # Use a less aggressive percentile (e.g., 98.0) or simply take the top 10% of the max intensity
                        # since electrodes (metal) are typically the absolute brightest objects in the scan.
                        max_val = np.max(tf_slice)
                        # We use 78% of the max value. This is highly robust because metal > bone in density
                        thresh = max(np.percentile(tf_slice, 98), max_val * 0.78)
                        
                        mask = tf_slice > thresh
                        
                        # Remove only very tiny single-pixel noise
                        mask = morphology.remove_small_objects(mask, min_size=2)
                        
                        # Make the electrode a solid, full circle
                        mask = morphology.closing(mask, morphology.disk(2))
                        mask = morphology.dilation(mask, morphology.disk(1))
                        
                        # Apply the mask
                        tf_slice[~mask] = 0
                except ImportError:
                    # Fallback to pure top-percentile visualization if skimage isn't installed
                    thresh = np.percentile(tf_slice, 98)
                    tf_slice[tf_slice < thresh] = 0

            self.img_result_transformed.setImage(tf_slice, autoLevels=True)
            sz, sy, sx = self.fixed_data.spacing
            self.img_result_transformed.setTransform(pg.QtGui.QTransform().scale(sx, sy))

        # Update ruler slice visibility
        if self.fixed_volume is not None:
            self.rulers['fixed'].set_slice(self.slice_fixed)
            self.rulers['result'].set_slice(self.slice_fixed)
        if self.moving_volume is not None:
            self.rulers['moving'].set_slice(self.slice_moving)

    # ============ The Scientific Coregistration Backend ============
    
    def _run_coregistration(self):
        """Runs strict spatial medical registration linking Space A --> Space B via non-blocking thread"""
        if self.fixed_data is None or self.moving_data is None:
            self.btn_run_reg.setText("Error: Set Both Images First!")
            return

        # Determine Cache Path (Patient Folder level)
        import os
        import numpy as np
        
        clean_path = os.path.normpath(self.moving_data.file_path)
        pname = self.moving_data.metadata.get("patient_name", "").strip()
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
        cache_vol_path = os.path.join(cache_dir, f"reg_volume_to_{self.fixed_name}.npy")
        cache_mask_path = os.path.join(cache_dir, f"reg_mask_to_{self.fixed_name}.npy")

        # Check Cache
        if os.path.exists(cache_vol_path):
            self.btn_run_reg.setText("Loading Cached Registration...")
            self.grp_progress.setVisible(True)
            self.lbl_progress.setText("Loaded from cache!")
            self.prog_bar.setRange(0, 100)
            self.prog_bar.setValue(100)
            
            import PyQt5.QtCore as QtCore
            def finish_cache_load():
                transformed_vol = np.load(cache_vol_path)
                transformed_mask = np.load(cache_mask_path) if os.path.exists(cache_mask_path) else None
                
                self.transformed_volume = transformed_vol
                self.transformed_mask = transformed_mask
                self.btn_run_reg.setText("▶ Run Co-Registration")
                self.btn_toggle.setChecked(True)
                self.grp_progress.setVisible(False)
                self._toggle_overlay()
                self._update_views()
                
            QtCore.QTimer.singleShot(500, finish_cache_load)
            return

        self.btn_run_reg.setText("⏳ Calculating (Please wait)...")
        self.btn_run_reg.setEnabled(False)
        self.grp_progress.setVisible(True)
        self.lbl_progress.setText("Running spatial registration engine...")
        self.prog_bar.setRange(0, 0) # Indeterminate

        # Extract moving mask if available to push down to the spatial engine pipeline
        moving_mask = self.masks.get(self.moving_name) if self.moving_name else None
        self.worker = RegistrationWorker(self.fixed_data, self.moving_data, moving_mask=moving_mask)
        
        def on_finished(transformed, transformed_mask):
            # Save to Cache
            np.save(cache_vol_path, transformed)
            if transformed_mask is not None:
                np.save(cache_mask_path, transformed_mask)
                
            self.transformed_volume = transformed
            self.transformed_mask = transformed_mask
            self.btn_run_reg.setText("▶ Run Co-Registration")
            self.btn_run_reg.setEnabled(True)
            self.btn_toggle.setChecked(True)
            self.grp_progress.setVisible(False)
            self._toggle_overlay()
            self._update_views()
            
        def on_error(msg):
            self.btn_run_reg.setText(f"Error! {msg[:10]}...")
            self.btn_run_reg.setEnabled(True)
            self.grp_progress.setVisible(False)
            print("Registration Error:", msg)

        self.worker.finished.connect(on_finished)
        self.worker.error.connect(on_error)
        self.worker.start()




# =====================================================================
# THREAD WORKER - PLACED OUTSIDE THE WIDGET CLASS
# =====================================================================

class RegistrationWorker(QThread):
    # Emits (transformed_anatomy_array, transformed_mask_array_or_None)
    finished = pyqtSignal(np.ndarray, object)
    error = pyqtSignal(str)

    def __init__(self, fixed_data, moving_data, moving_mask=None):
        super().__init__()
        self.fixed_data = fixed_data
        self.moving_data = moving_data
        self.moving_mask = moving_mask

    def run(self):
        try:
            import ants
            
            # --- 1. Construct ANTs Images from Arrays ---
            # Assume numpy natively is (z,y,x) we transpose to (x,y,z) for ants
            fixed_arr = self.fixed_data.volume.astype(np.float32).T
            moving_arr = self.moving_data.volume.astype(np.float32).T
            
            fixed_spacing = self.fixed_data.spacing[::-1] if hasattr(self.fixed_data, 'spacing') else (1,1,1)
            fixed_origin = self.fixed_data.origin[::-1] if hasattr(self.fixed_data, 'origin') else (0,0,0)
            
            moving_spacing = self.moving_data.spacing[::-1] if hasattr(self.moving_data, 'spacing') else (1,1,1)
            moving_origin = self.moving_data.origin[::-1] if hasattr(self.moving_data, 'origin') else (0,0,0)
            
            # Build Fixed Matrix
            fixed_img = ants.from_numpy(
                fixed_arr, 
                spacing=list(fixed_spacing), 
                origin=list(fixed_origin)
            )
            
            # Build Moving Matrix
            moving_img = ants.from_numpy(
                moving_arr, 
                spacing=list(moving_spacing), 
                origin=list(moving_origin)
            )
            
            # Force identical geometry before starting
            moving_img = ants.copy_image_info(fixed_img, moving_img)
            
            # --- 2. Execute ANTs Rigid/Affine Registration ---
            tx = ants.registration(
                fixed=fixed_img,
                moving=moving_img,
                type_of_transform="Affine"
            )
            
            # --- 3. Apply Transformations to Anatomy ---
            mri_reg = ants.apply_transforms(
                fixed=fixed_img,
                moving=moving_img,
                transformlist=tx['fwdtransforms'],
                interpolator="linear"
            )
            transformed_arr = mri_reg.numpy().T
            
            # --- 4. Apply Transformations to Segmentation Mask (If Present) ---
            transformed_mask_arr = None
            if self.moving_mask is not None:
                mask_arr = self.moving_mask.astype(np.float32).T
                moving_mask_img = ants.from_numpy(
                    mask_arr,
                    spacing=list(moving_spacing),
                    origin=list(moving_origin)
                )
                moving_mask_img = ants.copy_image_info(fixed_img, moving_mask_img)
                
                # CRITICAL: Use 'nearestNeighbor' to ensure voxel labels stay exactly 0, 1, or 2
                mask_reg = ants.apply_transforms(
                    fixed=fixed_img,
                    moving=moving_mask_img,
                    transformlist=tx['fwdtransforms'],
                    interpolator="nearestNeighbor"
                )
                transformed_mask_arr = mask_reg.numpy().T
            
            self.finished.emit(transformed_arr, transformed_mask_arr)
            
        except Exception as e:
            self.error.emit(str(e))