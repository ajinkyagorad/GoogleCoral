import sys
import random
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

class DummyOverlayWidget(QtWidgets.QWidget):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.confidences = [0.0] * len(labels)

        # Set up the widget to be frameless and transparent
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setGeometry(500, 500, 600, 150)  # Adjusted position and dimensions for better visibility

        # Add a background color to help visualize if the widget is appearing
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(50, 50, 50, 180))  # Dark translucent background
        self.setPalette(p)

        # Timer for updating dummy data
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_dummy_data)
        self.timer.start(30)  # Update every second

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw each bar for each class
        bar_width = 20
        spacing = 10
        for i, confidence in enumerate(self.confidences):
            opacity = confidence  # Opacity based on confidence
            painter.setOpacity(opacity)

            # Set a color for each class
            color = QtGui.QColor(100, 200, 255, 255)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)

            # Draw the rectangle for the confidence bar
            painter.drawRect(i * (bar_width + spacing), 0, bar_width, 100 * confidence)

    def update_bars(self, confidences):
        self.confidences = confidences
        self.update()  # Trigger a repaint

    def update_dummy_data(self):
        # Generate random confidences between 0 and 1
        dummy_confidences = np.array([random.uniform(0, 1) for _ in self.labels])
        self.update_bars(dummy_confidences)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Labels for demonstration purposes
    labels = ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]

    # Initialize the dummy widget and show it
    widget = DummyOverlayWidget(labels)
    widget.show()
    widget.raise_()  # Bring the widget to the front
    widget.activateWindow()  # Activate the window

    sys.exit(app.exec_())
