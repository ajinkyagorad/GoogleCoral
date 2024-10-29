import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import ctypes
import platform
import random
from PIL import Image
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter
import time
import argparse
import cv2
import signal
import atexit

# Screen settings for transparency and resolution (for AR glasses, where black = transparent)
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
UPDATE_INTERVAL_MS = 100  # Faster refresh for dynamic updates

class ARDashboardViewer(QtWidgets.QWidget):
    def __init__(self, model_path):
        super().__init__()
        # Set the window to be frameless, stay on top, and without interaction
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool
        )
        self.setStyleSheet('background-color: black')
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)  # Ignore mouse events
        self.move_to_secondary_screen()

        # Make the window fully transparent to mouse input on Windows
        if platform.system() == "Windows":
            self.make_window_click_through()

        # Initialize the interpreter with the Edge TPU
        try:
            self.interpreter = make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.width, self.height = common.input_size(self.interpreter)
        except Exception as e:
            print(f"Error initializing Edge TPU: {str(e)}")
            sys.exit(1)

        # Data for various plot types and overlays
        self.data_pie = np.random.rand(5)
        self.data_line = np.cumsum(np.random.randn(100)) + 10
        self.data_bar = np.random.rand(10) * 100
        self.data_hist = np.random.normal(50, 15, 1000)
        self.data_contour_x, self.data_contour_y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        self.data_contour_z = np.sin(np.sqrt(self.data_contour_x**2 + self.data_contour_y**2))
        self.ship_positions = np.random.rand(10, 2) * [SCREEN_WIDTH, SCREEN_HEIGHT]  # Random ship positions
        self.sensor_status = {"Microphone Array": "Working", "Thermal Camera": "Faulty", "Visible Cameras": "Working", "Radar": "Working", "Sonar": "Working", "GPS": "Working"}
        self.weather_conditions = ["Cloudy", "Raining", "Sunny", "Snowing"]
        self.current_weather = random.choice(self.weather_conditions)
        self.crew_count = random.randint(50, 150)
        self.vessel_weight = random.uniform(50000, 80000)  # Weight in tonnes
        self.engine_status = "Running"
        self.fuel_level = random.uniform(30, 100)  # Fuel level percentage
        self.battery_level = random.uniform(40, 100)  # Battery level percentage

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(UPDATE_INTERVAL_MS)

        # Initialize webcam for real-time segmentation
        self.init_camera()

        # Set up signal handling for graceful exit
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        atexit.register(self.cleanup)

    def init_camera(self):
        """Try different backends to initialize the camera"""
        # List of backends to try
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Default")
        ]

        for backend, name in backends:
            print(f"Trying {name} backend...")
            self.cap = cv2.VideoCapture(0 + backend)

            if self.cap.isOpened():
                print(f"Successfully opened camera with {name} backend")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

                # Test reading a frame
                ret, _ = self.cap.read()
                if ret:
                    return
                else:
                    self.cap.release()

            print(f"Failed to initialize with {name} backend")

        print("Error: Could not initialize webcam with any backend.")
        self.cleanup()
        sys.exit(1)

    def move_to_secondary_screen(self):
        screens = QtWidgets.QApplication.screens()
        if len(screens) > 1:
            secondary_screen = screens[1]
            self.setGeometry(secondary_screen.geometry())
        else:
            self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

    def make_window_click_through(self):
        hwnd = int(self.winId())  # Get the native window ID
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20
        extended_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, extended_style | WS_EX_LAYERED | WS_EX_TRANSPARENT)

    def cleanup(self, *args):
        print("Cleaning up resources...")
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'interpreter') and self.interpreter is not None:
            try:
                self.interpreter._delegate = None
                self.interpreter = None
            except Exception as e:
                print(f"Error cleaning up Edge TPU: {str(e)}")
        QtWidgets.QApplication.quit()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Update positions to simulate movement
        self.update_positions()

        # Draw all visualizations on the periphery with a margin of 1/6th of the screen width
        margin_x = SCREEN_WIDTH // 6
        margin_y = SCREEN_HEIGHT // 6

        # Draw segmentation camera feed
        self.draw_camera_feed(painter, x=margin_x, y=SCREEN_HEIGHT - margin_y - 150, width=250, height=150)
        self.draw_semantic_segmentation(painter, x=SCREEN_WIDTH - margin_x - 250, y=SCREEN_HEIGHT - margin_y - 150, width=250, height=150)

        # Additional Sci-Fi Visualizations for AR glasses
        self.draw_weather_conditions(painter, x=margin_x, y=margin_y - 30)
        self.draw_engine_status(painter, x=SCREEN_WIDTH - margin_x - 250, y=margin_y)
        self.draw_line_series(painter, self.data_line, x=SCREEN_WIDTH - margin_x - 350, y=margin_y + 100, width=350, height=120)
        self.draw_radar_chart(painter, self.ship_positions, x=margin_x + 40, y=margin_y + 250, diameter=150)
        self.draw_text(painter, f"Crew Count: {self.crew_count}", x=margin_x + 40, y=margin_y + 460, size=10, color=QtGui.QColor(255, 255, 255))
        self.draw_text(painter, f"Vessel Weight: {self.vessel_weight:.2f} tonnes", x=margin_x + 40, y=margin_y + 480, size=10, color=QtGui.QColor(255, 255, 255))
        self.draw_emotional_topology_map(painter, x=SCREEN_WIDTH - margin_x - 200, y=SCREEN_HEIGHT - margin_y - 500, width=180, height=120)

    def update_positions(self):
        # Simulate movement of ships and vessels
        self.ship_positions += (np.random.rand(10, 2) - 0.5) * 10
        self.ship_positions = np.clip(self.ship_positions, [0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.crew_count = random.randint(50, 150)
        self.vessel_weight = random.uniform(50000, 80000)
        self.current_weather = random.choice(self.weather_conditions)

    def draw_camera_feed(self, painter, x, y, width, height):
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Could not read frame from webcam.")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_qimage = QtGui.QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QtGui.QImage.Format_RGB888)
        scaled_qimage = frame_qimage.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        painter.drawImage(QtCore.QRect(x, y, width, height), scaled_qimage)

    def draw_semantic_segmentation(self, painter, x, y, width, height):
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Could not read frame from webcam.")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        resized_img = pil_image.resize((self.width, self.height), Image.LANCZOS)
        common.set_input(self.interpreter, resized_img)
        self.interpreter.invoke()
        result = segment.get_output(self.interpreter)
        if len(result.shape) == 3:
            result = np.argmax(result, axis=-1)
        mask_img = Image.fromarray(self.label_to_color_image(result).astype(np.uint8))
        mask_img = mask_img.convert("RGBA")
        mask_qimage = QtGui.QImage(mask_img.tobytes(), mask_img.width, mask_img.height, QtGui.QImage.Format_RGBA8888)
        scaled_qimage = mask_qimage.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        painter.drawImage(QtCore.QRect(x, y, width, height), scaled_qimage)

    def label_to_color_image(self, label):
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')
        colormap = self.create_pascal_label_colormap()
        if np.max(label) >= len(colormap):
            raise ValueError('Label value too large.')
        return colormap[label]

    def create_pascal_label_colormap(self):
        colormap = np.zeros((256, 3), dtype=int)
        indices = np.arange(256, dtype=int)
        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((indices >> channel) & 1) << shift
            indices >>= 3
        return colormap

    def draw_weather_conditions(self, painter, x, y):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.setFont(QtGui.QFont('Arial', 14, QtGui.QFont.Bold))
        painter.drawText(x, y, f"Current Weather: {self.current_weather}")

    def draw_engine_status(self, painter, x, y):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.setFont(QtGui.QFont('Arial', 14, QtGui.QFont.Bold))
        painter.drawText(x, y, f"Engine Status: {self.engine_status}")
        painter.drawText(x, y + 20, f"Fuel Level: {self.fuel_level:.2f}%")
        painter.drawText(x, y + 40, f"Battery Level: {self.battery_level:.2f}%")

    def draw_radar_chart(self, painter, ship_positions, x, y, diameter):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 255, 0, 150))
        painter.drawEllipse(x, y, diameter, diameter)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 255)))
        for pos in ship_positions:
            px, py = pos
            if (x < px < x + diameter) and (y < py < y + diameter):
                painter.drawEllipse(px - 5, py - 5, 10, 10)  # Draw nearby ships

    def draw_emotional_topology_map(self, painter, x, y, width, height):
        painter.setPen(QtCore.Qt.NoPen)
        # Creating gradient emotional landscapes
        for i in range(0, width, 10):
            for j in range(0, height, 10):
                color = QtGui.QColor(
                    random.randint(100, 255), random.randint(50, 150), random.randint(50, 200), 150
                )
                painter.setBrush(color)
                painter.drawRect(x + i, y + j, 10, 10)

    def draw_line_series(self, painter, data, x, y, width, height):
        max_value = np.max(data)
        min_value = np.min(data)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255, 255), 2))
        scale_x = width / len(data)
        scale_y = height / (max_value - min_value)
        for i in range(len(data) - 1):
            x1 = x + i * scale_x
            y1 = y + height - (data[i] - min_value) * scale_y
            x2 = x + (i + 1) * scale_x
            y2 = y + height - (data[i + 1] - min_value) * scale_y
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

    def draw_text(self, painter, text, x, y, size=10, color=QtGui.QColor(255, 255, 255)):
        painter.setPen(QtGui.QPen(color))
        painter.setFont(QtGui.QFont('Arial', size))
        painter.drawText(x, y, text)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path of the segmentation model.')
    args = parser.parse_args()
    viewer = ARDashboardViewer(args.model)
    viewer.show()
    sys.exit(app.exec_())
