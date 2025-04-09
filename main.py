import sys
from PyQt6.QtWidgets import QApplication
from annotation_ui import ImageAnnotationApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnnotationApp()
    window.show()
    sys.exit(app.exec())