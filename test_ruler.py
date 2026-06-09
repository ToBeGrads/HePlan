import sys
from PyQt5.QtWidgets import QApplication
from src.main import MainWindow
import time
from threading import Timer

def close_app():
    QApplication.quit()

app = QApplication(sys.argv)
win = MainWindow()
win.show()
Timer(3.0, close_app).start()
sys.exit(app.exec_())
