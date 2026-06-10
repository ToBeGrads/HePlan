from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt

class CollapsibleSection(QWidget):
    def __init__(self, title, expanded=False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.btn_toggle = QPushButton(f"{'▼' if expanded else '▶'}  {title}")
        self.btn_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding: 10px; background-color: #1e1e2e; "
            "color: #cdd6f4; font-weight: bold; border-radius: 4px; border: 1px solid #45475a; "
            "margin-bottom: 2px; } "
            "QPushButton:hover { background-color: #313244; }"
        )
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
