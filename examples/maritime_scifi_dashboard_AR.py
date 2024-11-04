import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
import ctypes
import platform
import random

# Screen settings for transparency and resolution (Lenovo Yoga Slim 7i Carbon)
SCREEN_WIDTH, SCREEN_HEIGHT = 2560, 1600
UPDATE_INTERVAL_MS = 100  # Faster refresh for dynamic updates

class SciFiPlotter(QtWidgets.QWidget):
    def __init__(self):
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

        # Data generation for various plot types
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
        self.semantic_image = QtGui.QImage(150, 100, QtGui.QImage.Format_ARGB32)
        self.semantic_image.fill(QtGui.QColor(100, 150, 200, 100))  # Placeholder image

        # Timer for real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(UPDATE_INTERVAL_MS)

    def move_to_secondary_screen(self):
        screens = QtWidgets.QApplication.screens()
        if len(screens) > 1:
            secondary_screen = screens[1]
            self.setGeometry(secondary_screen.geometry())
        else:
            self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

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

        # Update positions to simulate movement
        self.update_positions()

        # Draw each plot type using custom functions in the periphery
        # Top Section
        self.draw_radar_chart(painter, self.ship_positions, x=50, y=50, diameter=200)
        self.draw_text(painter, f"Latitude: 34.05° N, Longitude: 118.25° W", x=50, y=20, size=10, color=QtGui.QColor(255, 255, 255))
        self.draw_text(painter, f"Crew Count: {self.crew_count}", x=50, y=300, size=10, color=QtGui.QColor(255, 255, 255))
        self.draw_text(painter, f"Vessel Weight: {self.vessel_weight:.2f} tonnes", x=50, y=320, size=10, color=QtGui.QColor(255, 255, 255))

        # Right Section
        self.draw_weather_conditions(painter, x=SCREEN_WIDTH - 400, y=20)
        self.draw_line_series(painter, self.data_line, x=SCREEN_WIDTH - 500, y=150, width=400, height=150)  # Past hour wind speed, temperature, humidity

        # Bottom Section
        self.draw_engine_status(painter, x=400, y=SCREEN_HEIGHT - 150)
        self.draw_image(painter, self.semantic_image, x=SCREEN_WIDTH - 200, y=SCREEN_HEIGHT - 150, width=150, height=100)

        # Additional Sci-Fi Visualizations
        self.draw_quantum_state_tree(painter, x=200, y=500, width=150, height=200)
        self.draw_emotional_topology_map(painter, x=600, y=200, width=200, height=150)
        self.draw_dark_matter_flow_field(painter, x=1100, y=50, width=200, height=200)
        self.draw_memory_constellation_map(painter, x=100, y=SCREEN_HEIGHT - 300, width=150, height=150)
        self.draw_time_decay_visualizer(painter, x=SCREEN_WIDTH - 600, y=600, width=200, height=200)
        self.draw_dimensional_bleeding_monitor(painter, x=SCREEN_WIDTH - 300, y=400, width=200, height=200)
        self.draw_biorhythm_forests(painter, x=800, y=SCREEN_HEIGHT - 400, width=200, height=300)
        self.draw_temporal_echo_mapping(painter, x=300, y=100, width=200, height=200)
        self.draw_probability_storm_radar(painter, x=SCREEN_WIDTH - 900, y=SCREEN_HEIGHT - 500, width=200, height=200)

    def update_positions(self):
        # Simulate movement of ships and vessels
        self.ship_positions += (np.random.rand(10, 2) - 0.5) * 10
        self.ship_positions = np.clip(self.ship_positions, [0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        self.crew_count = random.randint(50, 150)
        self.vessel_weight = random.uniform(50000, 80000)
        self.current_weather = random.choice(self.weather_conditions)

    def draw_quantum_state_tree(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255, 200), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 255, 255, 50)))
        # Drawing a simplified tree-like structure
        for i in range(5):
            start_x = x + width // 2
            start_y = y + (i * (height // 5))
            for j in range(-i, i + 1, 2):
                end_x = start_x + j * (width // (2 * (i + 1)))
                end_y = start_y + (height // 5)
                painter.drawLine(start_x, start_y, end_x, end_y)
                painter.drawEllipse(end_x - 5, end_y - 5, 10, 10)

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

    def draw_dark_matter_flow_field(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 150), 1))
        # Simulating flow field lines
        for i in range(20):
            start_x = x + random.randint(0, width)
            start_y = y + random.randint(0, height)
            end_x = start_x + random.randint(-30, 30)
            end_y = start_y + random.randint(-30, 30)
            painter.drawLine(start_x, start_y, end_x, end_y)
            painter.drawEllipse(end_x - 2, end_y - 2, 4, 4)

    def draw_memory_constellation_map(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1))
        # Drawing star-like memory points
        for i in range(20):
            px = x + random.randint(0, width)
            py = y + random.randint(0, height)
            painter.drawEllipse(px - 1, py - 1, 2, 2)
        # Connecting some memories
        for _ in range(5):
            start_x = x + random.randint(0, width)
            start_y = y + random.randint(0, height)
            end_x = x + random.randint(0, width)
            end_y = y + random.randint(0, height)
            painter.drawLine(start_x, start_y, end_x, end_y)

    def draw_time_decay_visualizer(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255, 150), 1))
        # Drawing fracturing crystalline structures to represent decoherence
        for i in range(10):
            start_x = x + random.randint(0, width)
            start_y = y + random.randint(0, height)
            end_x = start_x + random.randint(-20, 20)
            end_y = start_y + random.randint(-20, 20)
            painter.drawLine(start_x, start_y, end_x, end_y)
            painter.drawEllipse(end_x - 3, end_y - 3, 6, 6)

    def draw_dimensional_bleeding_monitor(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 255, 100), 1))
        # Drawing membrane-like surfaces to represent dimensional boundaries
        for i in range(5):
            radius = width // 2 - (i * 10)
            painter.drawEllipse(x + width // 2 - radius, y + height // 2 - radius, radius * 2, radius * 2)

    def draw_biorhythm_forests(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 150), 1))
        # Drawing tree-like structures to represent biological data
        for i in range(3):
            start_x = x + (i * (width // 3)) + 20
            start_y = y + height
            end_y = y + random.randint(height // 2, height)
            painter.drawLine(start_x, start_y, start_x, end_y)
            painter.drawEllipse(start_x - 5, end_y - 5, 10, 10)

    def draw_temporal_echo_mapping(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1))
        # Drawing ripples to represent time travel events
        for i in range(5):
            radius = (i + 1) * 15
            painter.drawEllipse(x + width // 2 - radius, y + height // 2 - radius, radius * 2, radius * 2)

    def draw_probability_storm_radar(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 150), 2, QtCore.Qt.DashLine))
        # Drawing storm-like swirls to represent probability fields
        for i in range(3):
            start_angle = random.randint(0, 360) * 16
            span_angle = random.randint(60, 120) * 16
            painter.drawArc(x, y, width, height, start_angle, span_angle)

    def draw_radar_chart(self, painter, ship_positions, x, y, diameter):
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 255, 0, 150))
        painter.drawEllipse(x, y, diameter, diameter)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 255)))
        for pos in ship_positions:
            px, py = pos
            if (x < px < x + diameter) and (y < py < y + diameter):
                painter.drawEllipse(px - 5, py - 5, 10, 10)  # Draw nearby ships

    def draw_trajectory(self, painter, x, y, width, height):
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 255), 2, QtCore.Qt.SolidLine))
        painter.drawLine(x, y, x + width, y + height // 2)  # Simulated past trajectory
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, 255), 2, QtCore.Qt.DashLine))
        painter.drawLine(x + width, y + height // 2, x + width + 50, y + height)  # Future trajectory (dashed line)

    def draw_sensor_status(self, painter, x, y):
        painter.setPen(QtCore.Qt.NoPen)
        for i, (sensor, status) in enumerate(self.sensor_status.items()):
            color = QtGui.QColor(0, 255, 0, 255) if status == "Working" else QtGui.QColor(255, 0, 0, 255)
            painter.setBrush(color)
            painter.drawRect(x, y + i * 20, 150, 15)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            painter.drawText(x + 160, y + i * 20 + 12, f"{sensor}: {status}")
            painter.setPen(QtCore.Qt.NoPen)

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

    def draw_image(self, painter, image, x, y, width, height):
        painter.drawImage(QtCore.QRect(x, y, width, height), image)

    def draw_text(self, painter, text, x, y, size=10, color=QtGui.QColor(255, 255, 255)):
        painter.setPen(QtGui.QPen(color))
        painter.setFont(QtGui.QFont('Arial', size))
        painter.drawText(x, y, text)

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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    sci_fi_plotter = SciFiPlotter()
    sci_fi_plotter.show()
    sys.exit(app.exec_())