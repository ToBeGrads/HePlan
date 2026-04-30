import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QStatusBar, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from src.gui.components.patient_loader_widget import PatientLoaderWidget
from src.gui.components.mri_viewer_widget import MRIViewerWidget

# ⚠️ MUST be called BEFORE QApplication
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DBS Planning & Validation Workstation")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # System-aware font scaling
        base_font = QFont("Segoe UI", 10)
        base_font.setHintingPreference(QFont.PreferNoHinting)
        QApplication.setFont(base_font)

        # Load theme
        qss_path = os.path.join(os.path.dirname(__file__), "..", "assets", "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Tabs for Modules
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs)

        # Tab 1: Patient Loader
        self.loader_tab = PatientLoaderWidget()
        self.tabs.addTab(self.loader_tab, "1. Patient Loader")

        # Tab 2: MRI Viewer
        self.viewer_tab = MRIViewerWidget()
        self.tabs.addTab(self.viewer_tab, "2. MRI Viewer")

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a patient to begin.")

        # Connect Loader -> Viewer
        self.loader_tab.data_loaded.connect(self.on_patient_loaded)
        self.loader_tab.status_message.connect(self.status_bar.showMessage)

    def on_patient_loaded(self, patient_data):
        pid = patient_data.metadata.get("PatientID", "Unknown")
        modality = patient_data.modality
        desc = patient_data.metadata.get("SeriesDescription", "Scan")
        
        # Create a unique name for the volume
        vol_name = f"{pid}_{modality}_{desc}"
        
        # Add to viewer
        self.viewer_tab.add_volume(patient_data, vol_name)
        
        # Switch to viewer tab automatically
        self.tabs.setCurrentIndex(1)
        
        self.status_bar.showMessage(f"✅ Loaded: {vol_name} | Shape: {patient_data.get_shape()}")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()