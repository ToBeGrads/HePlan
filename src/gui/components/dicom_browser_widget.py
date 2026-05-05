import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QProgressBar,
                             QMessageBox, QHeaderView, QAbstractItemView, QSplitter, QGroupBox)
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from src.modules.patient_loader.dicom_scanner import scan_dicom_directory

class ScanWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            data = scan_dicom_directory(self.path)
            self.finished.emit(data)
        except Exception as e:
            self.error.emit(str(e))

class DicomBrowserWidget(QWidget):
    series_selected = pyqtSignal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.patients_data = {}
        self.current_patient_id = None
        self.current_study_id = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 40, 0, 0)
        
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizes([300, 300, 400])  # Initial proportional widths
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 2)     # Series table gets more space

        # Helper to apply responsive size policy
        def set_expanding(widget):
            from PyQt5.QtWidgets import QSizePolicy
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Patients Table
        grp_patients = QGroupBox("Patients")
        lay_p = QVBoxLayout(grp_patients)
        self.tbl_patients = QTableWidget()
        self.tbl_patients.setColumnCount(2)
        self.tbl_patients.setHorizontalHeaderLabels(["Patient ID", "Name"])
        self.tbl_patients.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_patients.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_patients.setAlternatingRowColors(True)
        self.tbl_patients.setShowGrid(False)
        self.tbl_patients.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_patients.verticalHeader().setVisible(False)
        set_expanding(self.tbl_patients)
        lay_p.addWidget(self.tbl_patients)

        # Studies Table
        grp_studies = QGroupBox("Studies")
        lay_s = QVBoxLayout(grp_studies)
        self.tbl_studies = QTableWidget()
        self.tbl_studies.setColumnCount(2)
        self.tbl_studies.setHorizontalHeaderLabels(["Date", "Study UID"])
        self.tbl_studies.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_studies.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_studies.setAlternatingRowColors(True)
        self.tbl_studies.setShowGrid(False)
        self.tbl_studies.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_studies.verticalHeader().setVisible(False)
        set_expanding(self.tbl_studies)
        lay_s.addWidget(self.tbl_studies)

        # Series Table
        grp_series = QGroupBox("Series")
        lay_ser = QVBoxLayout(grp_series)
        self.tbl_series = QTableWidget()
        self.tbl_series.setColumnCount(4)
        self.tbl_series.setHorizontalHeaderLabels(["Modality", "Description", "Files", "Folder"])
        self.tbl_series.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_series.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_series.setAlternatingRowColors(True)
        self.tbl_series.setShowGrid(False)
        self.tbl_series.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_series.verticalHeader().setVisible(False)
        set_expanding(self.tbl_series)
        lay_ser.addWidget(self.tbl_series)

        splitter.addWidget(grp_patients)
        splitter.addWidget(grp_studies)
        splitter.addWidget(grp_series)
        layout.addWidget(splitter)

        self.btn_load = QPushButton("Load Selected Series")
        self.btn_load.setObjectName('loadBtn')
        self.btn_load.setEnabled(False)
        self.btn_load.setToolTip("Load the highlighted series into the viewer")
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout.addWidget(self.btn_load, alignment=Qt.AlignCenter)
        layout.addWidget(self.progress)

        self.tbl_patients.itemSelectionChanged.connect(self._on_patient_selected)
        self.tbl_studies.itemSelectionChanged.connect(self._on_study_selected)
        self.tbl_series.itemSelectionChanged.connect(self._on_series_selected)
        self.btn_load.clicked.connect(self._emit_selection)
    def scan_directory(self, root_path: str):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.worker = ScanWorker(root_path)
        self.worker.finished.connect(self._on_scan_finished)
        self.worker.error.connect(self._on_scan_error)
        self.worker.start()

    def _on_scan_finished(self, data: dict):
        self.patients_data = data
        self.progress.setVisible(False)
        self._populate_patients()

    def _on_scan_error(self, msg: str):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Scan Error", msg)

    def _populate_patients(self):
        self.tbl_patients.setRowCount(0)
        for i, pid in enumerate(sorted(self.patients_data.keys())):
            self.tbl_patients.insertRow(i)
            self.tbl_patients.setItem(i, 0, QTableWidgetItem(pid))
            self.tbl_patients.setItem(i, 1, QTableWidgetItem(self.patients_data[pid]["name"]))
        self.tbl_studies.setRowCount(0)
        self.tbl_series.setRowCount(0)
        self.btn_load.setEnabled(False)

    def _on_patient_selected(self):
        row = self.tbl_patients.currentRow()
        if row < 0: return
        self.current_patient_id = self.tbl_patients.item(row, 0).text()
        self._populate_studies()

    def _populate_studies(self):
        self.tbl_studies.setRowCount(0)
        self.tbl_series.setRowCount(0)
        self.btn_load.setEnabled(False)
        if not self.current_patient_id: return

        studies = self.patients_data[self.current_patient_id]["studies"]
        for i, sid in enumerate(sorted(studies.keys())):
            self.tbl_studies.insertRow(i)
            self.tbl_studies.setItem(i, 0, QTableWidgetItem(studies[sid]["date"]))
            self.tbl_studies.setItem(i, 1, QTableWidgetItem(sid))

    def _on_study_selected(self):
        row = self.tbl_studies.currentRow()
        if row < 0: return
        self.current_study_id = self.tbl_studies.item(row, 1).text()
        self._populate_series()

    def _populate_series(self):
        self.tbl_series.setRowCount(0)
        self.btn_load.setEnabled(False)
        if not self.current_patient_id or not self.current_study_id: return

        series_dict = self.patients_data[self.current_patient_id]["studies"][self.current_study_id]["series"]
        for i, ser_uid in enumerate(sorted(series_dict.keys())):
            s = series_dict[ser_uid]
            self.tbl_series.insertRow(i)
            self.tbl_series.setItem(i, 0, QTableWidgetItem(s["modality"]))
            self.tbl_series.setItem(i, 1, QTableWidgetItem(s["description"]))
            self.tbl_series.setItem(i, 2, QTableWidgetItem(str(s["count"])))
            self.tbl_series.setItem(i, 3, QTableWidgetItem(os.path.basename(s["folder"])))

    def _on_series_selected(self):
        self.btn_load.setEnabled(self.tbl_series.currentRow() >= 0)

    def _emit_selection(self):
        row = self.tbl_series.currentRow()
        if row < 0: return

        series_dict = self.patients_data[self.current_patient_id]["studies"][self.current_study_id]["series"]
        ser_uid = sorted(series_dict.keys())[row]
        selected = series_dict[ser_uid]

        metadata = {
            "patient_id": self.current_patient_id,
            "patient_name": self.patients_data[self.current_patient_id]["name"],
            "study_date": self.patients_data[self.current_patient_id]["studies"][self.current_study_id]["date"],
            "modality": selected["modality"],
            "description": selected["description"],
            "series_uid": ser_uid
        }
        self.series_selected.emit(selected["folder"], metadata)