import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from src.modules.patient_loader.loader import PatientLoader
from src.gui.components.dicom_browser_widget import DicomBrowserWidget
from src.core.patient_data import PatientData

class LoadWorker(QThread):
    finished = pyqtSignal(PatientData)
    error = pyqtSignal(str)

    def __init__(self, path: str, metadata: dict, is_nifti: bool = False):
        super().__init__()
        self.path = path
        self.metadata = metadata
        self.is_nifti = is_nifti

    def run(self):
        try:
            loader = PatientLoader()
            if self.is_nifti:
                data = loader.load_nifti(self.path)
            else:
                data = loader.load_dicom_series(self.path, self.metadata)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))

class PatientLoaderWidget(QWidget):
    data_loaded = pyqtSignal(PatientData)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.browser = DicomBrowserWidget()
        self.load_worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 50, 10, 0)
        layout.setSpacing(10)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setContentsMargins(50, 0, 50, 0)
        self.btn_scan = QPushButton("Load DICOM")
        self.btn_scan.setObjectName('dicomBtn')
        self.btn_nifti = QPushButton("Load NIfTI")
        self.btn_nifti.setObjectName('niftiBtn')
        self.btn_scan.setToolTip("Recursively scan a folder for DICOM series")
        self.btn_nifti.setToolTip("Load a preprocessed .nii or .nii.gz volume")
        btn_layout.addWidget(self.btn_scan)
        btn_layout.addWidget(self.btn_nifti)
        layout.addLayout(btn_layout)

        layout.addWidget(self.browser)

        self.lbl_status = QLabel("Ready. Select a patient to begin.")
        self.lbl_status.setStyleSheet("color: #a6adc8; font-style: italic;")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.lbl_status, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress)

        self.btn_scan.clicked.connect(self._scan_dicom)
        self.btn_nifti.clicked.connect(self._load_nifti)
        self.browser.series_selected.connect(self._start_loading)

    def _scan_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Root Folder")
        if folder:
            self.status_message.emit("Scanning DICOM directory...")
            self.browser.scan_directory(folder)

    def _load_nifti(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if file:
            self._start_nifti_loading(file)

    def _start_loading(self, folder: str, metadata: dict):
        desc = metadata.get('description', 'Unknown')
        self.lbl_status.setText(f"Loading series: {desc}...")
        self.status_message.emit(f"Loading: {desc}")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.load_worker = LoadWorker(folder, metadata, is_nifti=False)
        self.load_worker.finished.connect(self._on_success)
        self.load_worker.error.connect(self._on_error)
        self.load_worker.start()

    def _start_nifti_loading(self, path: str):
        self.lbl_status.setText(f"Loading NIfTI: {os.path.basename(path)}...")
        self.status_message.emit(f"Loading NIfTI: {os.path.basename(path)}")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.load_worker = LoadWorker(path, {}, is_nifti=True)
        self.load_worker.finished.connect(self._on_success)
        self.load_worker.error.connect(self._on_error)
        self.load_worker.start()

    def _on_success(self, patient_data: PatientData):
        self.progress.setVisible(False)
        pid = patient_data.metadata.get('PatientID', 'Unknown')
        self.lbl_status.setText(f"✅ Loaded: {pid} | {patient_data.modality} | Shape: {patient_data.get_shape()}")
        self.status_message.emit(f"Ready. Patient {pid} loaded.")
        self.data_loaded.emit(patient_data)

    def _on_error(self, msg: str):
        self.progress.setVisible(False)
        self.lbl_status.setText("❌ Load failed.")
        self.status_message.emit("Error during loading.")
        QMessageBox.critical(self, "Error", msg)