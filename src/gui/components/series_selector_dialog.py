from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QMessageBox,
                             QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt

class SeriesSelectorDialog(QDialog):
    def __init__(self, series_list, parent=None):
        super().__init__(parent)
        self.series_list = series_list
        self.selected_series = None
        self.setWindowTitle("Select DICOM Series to Load")
        self.resize(850, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Found {len(self.series_list)} series. Select one to load:"))

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Patient ID", "Patient Name", "Date", "Modality", "Description"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)

        for i, s in enumerate(self.series_list):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(s["patient_id"]))
            self.table.setItem(i, 1, QTableWidgetItem(s["patient_name"]))
            self.table.setItem(i, 2, QTableWidgetItem(s["study_date"]))
            self.table.setItem(i, 3, QTableWidgetItem(s["modality"]))
            self.table.setItem(i, 4, QTableWidgetItem(s["description"]))

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Selected")
        self.btn_cancel = QPushButton("Cancel")
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        self.btn_load.clicked.connect(self._on_load)
        self.btn_cancel.clicked.connect(self.reject)
        self.table.doubleClicked.connect(self._on_load)

    def _on_load(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No Selection", "Please select a series to load.")
            return
        self.selected_series = self.series_list[row]
        self.accept()