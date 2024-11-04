import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import ctypes
import platform

# Screen settings for transparency and resolution (Lenovo Yoga Slim 7i Carbon)
SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1600
UPDATE_INTERVAL_MS = 50  # 50 ms for a responsive real-time effect

class SciFiPlotter(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Set the window to be frameless, stay on top, and without interaction
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # Ignore mouse events
        self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Make the window fully transparent to mouse input on Windows
        if platform.system() == "Windows":
            self.make_window_click_through()

        # Dummy data generation for various plot types
        self.data_pie = np.random.rand(5)
        self.data_line = np.cumsum(np.random.randn(100)) + 10
        self.data_bar = np.random.rand(10) * 100
        self.data_hist = np.random.normal(50, 15, 1000)
        self.data_contour_x, self.data_contour_y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        self.data_contour_z = np.sin(np.sqrt(self.data_contour_x**2 + self.data_contour_y**2))

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(UPDATE_INTERVAL_MS)

    def make_window_click_through(self):
        # Get the window handle (HWND) for the widget
        hwnd = int(self.winId())  # Get the native window ID
        # Get the Windows API SetWindowLong function
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20

        # Set the window as layered and transparent to input
        extended_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, extended_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw each plot type using custom functions
        self.draw_pie_chart(painter, self.data_pie, x=50, y=50, diameter=200)
        self.draw_line_series(painter, self.data_line, x=300, y=50, width=500, height=200)
        self.draw_bar_graph(painter, self.data_bar, x=50, y=300, width=300, height=200)
        self.draw_histogram(painter, self.data_hist, x=400, y=300, width=300, height=200)
        self.draw_contour_plot(painter, self.data_contour_x, self.data_contour_y, self.data_contour_z, x=750, y=50, width=500, height=300)

    def draw_pie_chart(self, painter, data, x, y, diameter):
        total = np.sum(data)
        angle_start = 0
        for value in data:
            angle_span = int(360 * (value / total) * 16)  # Convert to 1/16 degree units
            color = QtGui.QColor(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255), 200)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawPie(x, y, diameter, diameter, angle_start, angle_span)
            angle_start += angle_span

    def draw_line_series(self, painter, data, x, y, width, height):
        max_value = np.max(data)
        min_value = np.min(data)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255, 200), 2))
        scale_x = width / len(data)
        scale_y = height / (max_value - min_value)

        for i in range(len(data) - 1):
            x1 = x + i * scale_x
            y1 = y + height - (data[i] - min_value) * scale_y
            x2 = x + (i + 1) * scale_x
            y2 = y + height - (data[i + 1] - min_value) * scale_y
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

    def draw_bar_graph(self, painter, data, x, y, width, height):
        max_value = np.max(data)
        num_bars = len(data)
        bar_width = width / num_bars
        painter.setPen(QtCore.Qt.NoPen)

        for i, value in enumerate(data):
            color = QtGui.QColor(np.random.randint(100, 255), np.random.randint(50, 200), np.random.randint(50, 200), 180)
            painter.setBrush(color)
            rect_height = int((value / max_value) * height)
            painter.drawRect(x + i * bar_width, y + height - rect_height, bar_width - 2, rect_height)

    def draw_histogram(self, painter, data, x, y, width, height):
        counts, bins = np.histogram(data, bins=20)
        max_count = np.max(counts)
        bin_width = width / len(counts)
        painter.setPen(QtCore.Qt.NoPen)

        for i, count in enumerate(counts):
            color = QtGui.QColor(200, np.random.randint(100, 200), np.random.randint(100, 200), 160)
            painter.setBrush(color)
            rect_height = int((count / max_count) * height)
            painter.drawRect(x + i * bin_width, y + height - rect_height, bin_width - 2, rect_height)

    def draw_contour_plot(self, painter, x_data, y_data, z_data, x, y, width, height):
        levels = np.linspace(np.min(z_data), np.max(z_data), 10)
        for i in range(len(levels) - 1):
            mask = (z_data >= levels[i]) & (z_data < levels[i + 1])
            color = QtGui.QColor(np.random.randint(100, 255), np.random.randint(50, 255), np.random.randint(50, 255), 100)
            painter.setBrush(color)
            for row in range(mask.shape[0]):
                for col in range(mask.shape[1]):
                    if mask[row, col]:
                        px = x + col * (width / mask.shape[1])
                        py = y + row * (height / mask.shape[0])
                        painter.drawRect(px, py, width / mask.shape[1], height / mask.shape[0])

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    sci_fi_plotter = SciFiPlotter()
    sci_fi_plotter.show()
    sys.exit(app.exec_())
