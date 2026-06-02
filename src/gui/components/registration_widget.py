import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QSlider, QLabel, 
                             QGroupBox, QPushButton, QSizePolicy, QScrollArea, QFrame,
                             QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

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
        mask_lut = np.zeros((256, 4), dtype=np.uint8)
        mask_lut[0] = [0, 0, 0, 0]             # Background transparent
        mask_lut[1] = [255, 0, 0, 150]         # Target class 1 (Red)
        mask_lut[2] = [0, 255, 0, 150]         # Target class 2 (Green)
        self.img_result_mask.setLookupTable(mask_lut)
        
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
        grp_volumes.setStyleSheet("color: #cdd6f4;")
        vol_layout = QVBoxLayout(grp_volumes)
        vol_layout.setContentsMargins(8, 8, 8, 8)
        
        self.tbl_volumes = QTableWidget()
        self.tbl_volumes.setColumnCount(3)
        self.tbl_volumes.setHorizontalHeaderLabels(["Name", "Mod", "Shape"])
        self.tbl_volumes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_volumes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_volumes.setAlternatingRowColors(True)
        self.tbl_volumes.setShowGrid(True)
        self.tbl_volumes.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_volumes.verticalHeader().setVisible(False)
        self.tbl_volumes.setStyleSheet("background-color: #181825; color: #cdd6f4;")
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
        scroll_layout.addWidget(grp_volumes)

        grp_controls = QGroupBox("⚙️ Registration Controls")
        grp_controls.setStyleSheet("color: #cdd6f4;")
        ctrl_layout = QVBoxLayout(grp_controls)

        # Display Mode Selection
        self.cmb_display_mode = QComboBox()
        self.cmb_display_mode.addItems(["Full Anatomy Overlay", "Targeted (Otsu Threshold)"])
        self.cmb_display_mode.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 4px;")
        self.cmb_display_mode.currentIndexChanged.connect(lambda: self._update_views())
        ctrl_layout.addWidget(QLabel("Result Display Mode:"))
        ctrl_layout.addWidget(self.cmb_display_mode)
        
        self.btn_run_reg = QPushButton("▶ Run Co-Registration")
        self.btn_run_reg.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold; padding: 6px; margin-top: 10px;")
        self.btn_run_reg.clicked.connect(self._run_coregistration)
        
        self.btn_reset = QPushButton("Reset Views")
        self.btn_reset.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 6px;")
        self.btn_reset.clicked.connect(self._reset_views)
        
        ctrl_layout.addWidget(self.btn_run_reg)
        ctrl_layout.addWidget(self.btn_reset)
        
        scroll_layout.addWidget(grp_controls)
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
            # Clear previous registration results since a new volume was linked
            self.transformed_volume = None
            self.transformed_mask = None
            self._update_views(auto_level_moving=True)

    def _reset_views(self):
        """Fit standard scaling views on user resents"""
        self.view_fixed.autoRange()
        self.view_moving.autoRange()
        self.view_result.autoRange()

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
            
            self.header_fixed.lbl_slice.setText(f"{self.slice_fixed + 1}/{z_f}")
            self.header_result.lbl_slice.setText(f"{self.slice_fixed + 1}/{z_f}")

            # Also update Segmentation Mask if selected!
            fixed_vol_name = self.header_fixed.lbl_title.text().replace("Fixed: ", "")
            if fixed_vol_name in self.masks:
                mask_slice = np.flipud(self.masks[fixed_vol_name][self.slice_fixed])
                self.img_result_mask.setImage(mask_slice, autoLevels=False)
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
                    
            self.header_moving.lbl_slice.setText(f"{self.slice_moving + 1}/{z_m}")
            
        if self.transformed_volume is not None:
            tf_slice = np.flipud(self.transformed_volume[self.slice_fixed]).copy()
            
            # Application of Targeted Otsu Threshold if enabled
            if "Otsu" in self.cmb_display_mode.currentText():
                try:
                    from skimage.filters import threshold_otsu
                    if np.any(tf_slice):
                        thresh = threshold_otsu(tf_slice)
                        tf_slice[tf_slice < thresh] = 0
                except ImportError:
                    # Fallback to pure top-percentile visualization if skimage isn't installed
                    thresh = np.percentile(tf_slice, 95)
                    tf_slice[tf_slice < thresh] = 0

            self.img_result_transformed.setImage(tf_slice, autoLevels=True)
    # ============ The Scientific Coregistration Backend ============
    
    def _run_coregistration(self):
        """Runs strict spatial medical registration linking Space A --> Space B via non-blocking thread"""
        if self.fixed_data is None or self.moving_data is None:
            self.btn_run_reg.setText("Error: Set Both Images First!")
            return

        self.btn_run_reg.setText("⏳ Calculating (Please wait)...")
        self.btn_run_reg.setEnabled(False)

        # Extract moving mask if available to push down to the spatial engine pipeline
        moving_mask = self.masks.get(self.moving_name) if self.moving_name else None
        self.worker = RegistrationWorker(self.fixed_data, self.moving_data, moving_mask=moving_mask)
        
        def on_finished(transformed, transformed_mask):
            self.transformed_volume = transformed
            self.transformed_mask = transformed_mask
            self.btn_run_reg.setText("▶ Run Co-Registration")
            self.btn_run_reg.setEnabled(True)
            self.btn_toggle.setChecked(True)
            self._toggle_overlay()
            self._update_views()
            
        def on_error(msg):
            self.btn_run_reg.setText(f"Error! {msg[:10]}...")
            self.btn_run_reg.setEnabled(True)
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