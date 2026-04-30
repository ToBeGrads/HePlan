import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QSpinBox, QGroupBox, QTableWidget, 
                             QTableWidgetItem, QAbstractItemView, QHeaderView,
                             QPushButton, QSplitter, QToolBar, QAction, QComboBox,
                             QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont
from src.core.patient_data import PatientData

class MPRCanvas(QWidget):
    """Advanced Canvas supporting Axial, Sagittal, Coronal, and Oblique views."""
    
    slice_changed = pyqtSignal(int, int) # axis, index
    
    def __init__(self, parent=None, axis=0):
        super().__init__(parent)
        self.volume = None
        self.slice_index = 0
        self.axis = axis  # 0=Axial, 1=Sagittal, 2=Coronal, 3=Oblique
        self.window_center = 0
        self.window_width = 1000
        
        # Oblique rotation angles (degrees)
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        
        self.is_maximized = False
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)

    def set_volume(self, volume: np.ndarray):
        self.volume = volume
        if volume is not None:
            min_val = np.min(volume)
            max_val = np.max(volume)
            self.window_center = (min_val + max_val) / 2
            self.window_width = max_val - min_val
            if self.window_width == 0: self.window_width = 1
            
            # Reset slice to middle
            if self.axis < 3:
                self.slice_index = self.volume.shape[self.axis] // 2
            else:
                self.slice_index = self.volume.shape[0] // 2 # Default for oblique
                
            self.update_slice()

    def set_axis(self, axis: int):
        self.axis = axis
        self.update_slice()

    def set_slice(self, index: int):
        if self.volume is None: return
        if self.axis < 3:
            max_idx = self.volume.shape[self.axis] - 1
        else:
            max_idx = self.volume.shape[0] - 1 # Simplified for oblique
            
        self.slice_index = max(0, min(index, max_idx))
        self.update_slice()
        self.slice_changed.emit(self.axis, self.slice_index)

    def set_oblique_rotation(self, x, y, z):
        self.rot_x = x
        self.rot_y = y
        self.rot_z = z
        if self.axis == 3:
            self.update_slice()

    def update_slice(self):
        self.update()

    def get_2d_slice(self):
        """Extracts the 2D array based on current axis and rotation."""
        if self.volume is None:
            return None

        if self.axis == 0: # Axial
            return self.volume[self.slice_index, :, :]
        elif self.axis == 1: # Sagittal
            return self.volume[:, self.slice_index, :]
        elif self.axis == 2: # Coronal
            return self.volume[:, :, self.slice_index]
        elif self.axis == 3: # Oblique (Simplified Reslicing)
            # Note: True arbitrary reslicing requires scipy.ndimage.map_coordinates
            # For performance in PyQt, we simulate by rotating the extracted axial slice
            # In a production medical app, you would use VTK or ITK for this.
            # Here we just return the axial slice at current index for stability,
            # but apply a visual rotation in paintEvent.
            return self.volume[self.slice_index, :, :]
            
        return None

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QImage, qRgb, QTransform
        painter = QPainter(self)
        
        if self.volume is None:
            painter.fillRect(self.rect(), Qt.black)
            painter.drawText(self.rect(), Qt.AlignCenter, "No Volume Loaded")
            return

        slice_2d = self.get_2d_slice()
        if slice_2d is None: return

        # Apply Window/Level
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        normalized = np.clip((slice_2d - min_val) / (max_val - min_val), 0, 1)
        
        # Convert to 8-bit grayscale
        img_data = (normalized * 255).astype(np.uint8)
        
        height, width = img_data.shape
        bytes_per_line = width
        qimg = QImage(img_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Scale to fit widget
        scaled_img = qimg.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Center the image
        x_offset = (self.width() - scaled_img.width()) // 2
        y_offset = (self.height() - scaled_img.height()) // 2
        
        # If Oblique, apply rotation transform
        if self.axis == 3:
            painter.save()
            painter.translate(self.width()/2, self.height()/2)
            painter.rotate(self.rot_z) # Simple Z-rotation for demo
            painter.translate(-self.width()/2, -self.height()/2)
            painter.drawImage(x_offset, y_offset, scaled_img)
            painter.restore()
        else:
            painter.drawImage(x_offset, y_offset, scaled_img)

    def wheelEvent(self, event):
        if self.volume is None: return
        delta = event.angleDelta().y()
        step = 1 if delta > 0 else -1
        new_idx = self.slice_index + step
        self.set_slice(new_idx)

    def mouseDoubleClickEvent(self, event):
        """Toggle Maximize/Restore"""
        self.is_maximized = not self.is_maximized
        # Emit signal to parent to handle layout change
        self.parent().toggle_view_maximization(self)


class MRIViewerWidget(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volumes = {} 
        self.current_volume_name = None
        self.views = {} # Store references to canvases
        self.current_volume = None # Currently active volume
        self._init_ui()


    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # --- Toolbar ---
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        
        self.act_grid = QAction("📐 3-View Grid", self)
        self.act_grid.triggered.connect(lambda: self.set_layout_mode('grid'))
        toolbar.addAction(self.act_grid)
        
        self.act_single = QAction("🖥️ Single View", self)
        self.act_single.triggered.connect(lambda: self.set_layout_mode('single'))
        toolbar.addAction(self.act_single)

        toolbar.addSeparator()
        
        lbl_mode = QLabel("Mode:")
        toolbar.addWidget(lbl_mode)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Axial", "Sagittal", "Coronal", "Oblique"])
        self.cmb_mode.currentIndexChanged.connect(self._change_global_mode)
        toolbar.addWidget(self.cmb_mode)

        main_layout.addWidget(toolbar)

        # --- Main Content Area ---
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setMaximumWidth(280)
        
        # Volume List
        grp_vols = QGroupBox("Loaded Volumes")
        lay_vols = QVBoxLayout(grp_vols)
        self.tbl_volumes = QTableWidget()
        self.tbl_volumes.setColumnCount(2)
        self.tbl_volumes.setHorizontalHeaderLabels(["Name", "Modality"])
        self.tbl_volumes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_volumes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_volumes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_volumes.verticalHeader().setVisible(False)
        self.tbl_volumes.itemSelectionChanged.connect(self._on_volume_selected)
        lay_vols.addWidget(self.tbl_volumes)
        sidebar_layout.addWidget(grp_vols)

        # Controls
        grp_ctrl = QGroupBox("Controls")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        
        # Slice Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.lbl_slice_info = QLabel("Slice: 0 / 0")
        self.lbl_slice_info.setAlignment(Qt.AlignCenter)
        
        self.btn_prev.clicked.connect(lambda: self.adjust_slice(-1))
        self.btn_next.clicked.connect(lambda: self.adjust_slice(1))
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_slice_info)
        nav_layout.addWidget(self.btn_next)
        lay_ctrl.addLayout(nav_layout)

        # Window/Level
        self.slider_center = QSlider(Qt.Horizontal)
        self.slider_center.setRange(-1000, 3000)
        self.slider_center.setValue(0)
        self.slider_center.valueChanged.connect(self._update_wl)
        lay_ctrl.addWidget(QLabel("Level (Center)"))
        lay_ctrl.addWidget(self.slider_center)

        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 4000)
        self.slider_width.setValue(1000)
        self.slider_width.valueChanged.connect(self._update_wl)
        lay_ctrl.addWidget(QLabel("Width (Contrast)"))
        lay_ctrl.addWidget(self.slider_width)

        # Oblique Controls (Only visible in Oblique mode)
        self.grp_oblique = QGroupBox("Oblique Rotation")
        lay_obl = QVBoxLayout(self.grp_oblique)
        self.slider_rot_z = QSlider(Qt.Horizontal)
        self.slider_rot_z.setRange(-180, 180)
        self.slider_rot_z.setValue(0)
        self.slider_rot_z.valueChanged.connect(self._update_oblique)
        lay_obl.addWidget(QLabel("Rotation Z"))
        lay_obl.addWidget(self.slider_rot_z)
        self.grp_oblique.setVisible(False)
        lay_ctrl.addWidget(self.grp_oblique)

        sidebar_layout.addWidget(grp_ctrl)
        sidebar_layout.addStretch()
        
        content_splitter.addWidget(sidebar)

        # Right: Viewport Container
        self.viewport_container = QWidget()
        self.viewport_layout = QVBoxLayout(self.viewport_container)
        self.viewport_layout.setContentsMargins(0,0,0,0)
        self.viewport_layout.setSpacing(2)
        
        # Initialize Views
        self.view_axial = MPRCanvas(axis=0)
        self.view_sagittal = MPRCanvas(axis=1)
        self.view_coronal = MPRCanvas(axis=2)
        self.view_oblique = MPRCanvas(axis=3)
        
        self.views = {
            'axial': self.view_axial,
            'sagittal': self.view_sagittal,
            'coronal': self.view_coronal,
            'oblique': self.view_oblique
        }
        
        # Connect slice changes to sync others (optional, kept simple here)
        for v in self.views.values():
            v.slice_changed.connect(self._on_slice_changed_internal)

        # Default Layout: Grid
        self.grid_widget = QWidget()
        self.grid_layout = QVBoxLayout(self.grid_widget)
        
        top_row = QHBoxLayout()
        top_row.addWidget(self.view_axial)
        top_row.addWidget(self.view_sagittal)
        
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.view_coronal)
        bottom_row.addWidget(self.view_oblique) # Show oblique in grid too
        
        self.grid_layout.addLayout(top_row)
        self.grid_layout.addLayout(bottom_row)
        
        self.single_widget = QWidget()
        self.single_layout = QVBoxLayout(self.single_widget)
        # Will be populated dynamically
        
        self.viewport_layout.addWidget(self.grid_widget)
        
        content_splitter.addWidget(self.viewport_container)
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 4)
        
        main_layout.addWidget(content_splitter)
        
        self.set_layout_mode('grid')

        def add_volume(self, patient_data: PatientData, name: str = None):
            if not name:
                name = f"{patient_data.modality}_{len(self.volumes)+1}"
            
            self.volumes[name] = patient_data
            
            row = self.tbl_volumes.rowCount()
            self.tbl_volumes.insertRow(row)
            self.tbl_volumes.setItem(row, 0, QTableWidgetItem(name))
            self.tbl_volumes.setItem(row, 1, QTableWidgetItem(patient_data.modality))
            
            if len(self.volumes) == 1:
                self.tbl_volumes.selectRow(0)
                self._load_volume_to_views(name)

    def _on_volume_selected(self):
        row = self.tbl_volumes.currentRow()
        if row < 0: return
        name = self.tbl_volumes.item(row, 0).text()
        self._load_volume_to_views(name)

    def _load_volume_to_views(self, name: str):
        if name not in self.volumes: return
        self.current_volume_name = name
        pd = self.volumes[name]
        
        for view in self.views.values():
            view.set_volume(pd.volume)
            
        # Update UI controls
        self.slider_center.setValue(int(self.view_axial.window_center))
        self.slider_width.setValue(int(self.view_axial.window_width))
        self._update_slice_label()

    def _change_global_mode(self, index):
        modes = ['axial', 'sagittal', 'coronal', 'oblique']
        selected_mode = modes[index]
        
        if selected_mode == 'oblique':
            self.grp_oblique.setVisible(True)
        else:
            self.grp_oblique.setVisible(False)
            
        # If in single view mode, switch the displayed view
        if self.layout_mode == 'single':
            self._show_single_view(selected_mode)

    def set_layout_mode(self, mode: str):
        self.layout_mode = mode
        # Clear viewport
        for i in reversed(range(self.viewport_layout.count())): 
            self.viewport_layout.itemAt(i).widget().setParent(None)
            
        if mode == 'grid':
            self.viewport_layout.addWidget(self.grid_widget)
            self.grid_widget.setVisible(True)
            self.single_widget.setVisible(False)
        else:
            self.viewport_layout.addWidget(self.single_widget)
            self.grid_widget.setVisible(False)
            self.single_widget.setVisible(True)
            # Default to Axial in single mode
            self._show_single_view('axial')
            self.cmb_mode.setCurrentIndex(0)

    def _show_single_view(self, view_key: str):
        # Clear single layout
        for i in reversed(range(self.single_layout.count())): 
            self.single_layout.itemAt(i).widget().setParent(None)
            
        view = self.views[view_key]
        self.single_layout.addWidget(view)
        view.is_maximized = True

    def toggle_view_maximization(self, canvas: MPRCanvas):
        if self.layout_mode == 'grid':
            # Switch to single view of the clicked canvas
            key = [k for k, v in self.views.items() if v == canvas][0]
            self.set_layout_mode('single')
            idx = ['axial', 'sagittal', 'coronal', 'oblique'].index(key)
            self.cmb_mode.setCurrentIndex(idx)
        else:
            # Return to grid
            self.set_layout_mode('grid')

    def adjust_slice(self, step: int):
        if not self.current_volume_name: return
        # Adjust all views synchronously for simplicity, or just active one
        # Here we adjust the currently focused view logic, but let's adjust all standard ones
        for key in ['axial', 'sagittal', 'coronal']:
            self.views[key].set_slice(self.views[key].slice_index + step)
        self._update_slice_label()

    def _on_slice_changed_internal(self, axis, index):
        self._update_slice_label()

    def _update_slice_label(self):
        if not self.current_volume_name: return
        pd = self.volumes[self.current_volume_name]
        # Show info for Axial as primary reference
        total = pd.volume.shape[0]
        current = self.view_axial.slice_index
        self.lbl_slice_info.setText(f"Axial Slice: {current} / {total}")

    def _update_wl(self):
        center = self.slider_center.value()
        width = self.slider_width.value()
        for view in self.views.values():
            view.window_center = center
            view.window_width = width
            view.update_slice()

    def _update_oblique(self):
        val = self.slider_rot_z.value()
        self.view_oblique.set_oblique_rotation(0, 0, val)
    
    

    def add_volume(self, patient_data: PatientData, vol_name: str):
        """Register a loaded volume and update the volumes table."""
        if vol_name in self.volumes:
            vol_name = f"{vol_name}_{len(self.volumes)}"

        self.volumes[vol_name] = patient_data

        row = self.tbl_volumes.rowCount()
        self.tbl_volumes.insertRow(row)
        self.tbl_volumes.setItem(row, 0, QTableWidgetItem(vol_name))
        self.tbl_volumes.setItem(row, 1, QTableWidgetItem(patient_data.modality))
        self.tbl_volumes.setItem(row, 2, QTableWidgetItem(str(patient_data.get_shape())))

        # Select new row & set as active immediately (bypasses signal race)
        self.tbl_volumes.selectRow(row)
        self.current_volume = patient_data
        self._render_mpr()

    def _on_volume_selected(self):
        """Called by table selection signal. Reads selected row directly."""
        row = self.tbl_volumes.currentRow()
        if row < 0:
            return

        vol_name = self.tbl_volumes.item(row, 0).text()
        if vol_name in self.volumes:
            self.current_volume = self.volumes[vol_name]
            self._render_mpr()

    def _render_mpr(self):
        """Placeholder for MPR slice rendering. Module 2 will replace this."""
        if self.current_volume is None:
            return
        # 🔜 Future: extract axial/sagittal/coronal slices, apply window/level, update views
        print(f"🖼️ Ready to render MPR for: {self.current_volume.metadata.get('SeriesDescription', 'Unknown')}")