import argparse
import numpy as np
import sounddevice as sd
import csv
import time
import threading
from PyQt5 import QtWidgets, QtGui, QtCore
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import sys
import signal

class RealTimeClassifier(QtCore.QObject):
    update_signal = QtCore.pyqtSignal(np.ndarray)  # Signal to send confidences to the widget

    def __init__(self, model, labels_path):
        super().__init__()
        print("Initializing RealTimeClassifier...")  # Debug statement
        # Load the labels from the CSV file
        self.labels = []
        try:
            with open(labels_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels.append(row['display_name'])
            print("Labels loaded successfully.")  # Debug statement
        except Exception as e:
            print(f"Error loading labels: {e}")  # Debug statement

        # Initialize the interpreter
        try:
            self.interpreter = make_interpreter(model)
            self.interpreter.allocate_tensors()
            print("Interpreter initialized successfully.")  # Debug statement
        except Exception as e:
            print(f"Error initializing interpreter: {e}")  # Debug statement
        finally:
            #Reset if possible
            interpreter = None
        # Set input tensor
        input_details = self.interpreter.get_input_details()[0]
        self.input_shape = input_details['shape']
        self.num_samples = self.input_shape[0]  # Expected to be 15600 samples without batch dimension

        # Buffer for storing samples for classification
        self.audio_buffer = np.zeros(self.num_samples, dtype=np.float32)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")  # Debug statement

        # Append new data and maintain buffer size (sliding window)
        indata = indata[:, 0]  # Use the first channel if stereo
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:frames]

        # Check if buffer has enough samples
        if len(self.audio_buffer) == self.num_samples:
            try:
                # Use audio buffer as it is, without expanding dimensions
                input_data = self.audio_buffer.astype(np.float32)  # Shape should be (num_samples,)

                # Set tensor with the current buffer
                self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)

                # Run inference
                start_time = time.time()
                self.interpreter.invoke()
                end_time = time.time()
                inference_time = end_time - start_time

                # Get output
                output_data = np.squeeze(self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index']))  # Extract the output

                # Emit signal to update the widget
                self.update_signal.emit(output_data)

                # Sort output data in decreasing order of confidence
                sorted_indices = np.argsort(output_data)[::-1]
                top_5_indices = sorted_indices[:5]
                top_5_labels = [self.labels[idx] for idx in top_5_indices]
                top_5_confidences = [output_data[idx] for idx in top_5_indices]

                # Print the current classification
                print("\033[2K\033[F" * 7, end='')  # Clear lines and move cursor up to overwrite previous output

                print(f"Time segment: {time_info.inputBufferAdcTime:.1f} seconds".ljust(80))
                for label, confidence in zip(top_5_labels, top_5_confidences):
                    print(f"Class: {label.ljust(25)}, Confidence: {confidence:.3f}".ljust(80))
                print(f"Inference time: {inference_time:.3f} seconds".ljust(80))
            except Exception as e:
                print(f"Error during inference: {e}")  # Debug statement

    def classify_from_mic(self, device_index=1):
        print("Starting classification from microphone...")  # Debug statement
        self.audio_thread = threading.Thread(target=self._start_streaming, args=(device_index,), daemon=True)
        self.audio_thread.start()

    def _start_streaming(self, device_index):
        print("Starting audio stream...")  # Debug statement
        try:
            with sd.InputStream(channels=1, callback=self.audio_callback, samplerate=16000, blocksize=int(0.1 * 16000), device=device_index):
                print("Audio stream started successfully.")  # Debug statement
                while True:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio stream: {e}")  # Debug statement

    def cleanup(self):
        print("Cleaning up resources...")  # Debug statement
        if self.interpreter:
            del self.interpreter

class OverlayWidget(QtWidgets.QWidget):
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

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Draw each bar for each class
        bar_width = 20
        spacing = 10
        for i, confidence in enumerate(self.confidences):
            opacity = max(0.1, confidence)  # Ensure bars are visible even at low confidences
            painter.setOpacity(opacity)

            # Set a color for each class
            color = QtGui.QColor(100, 200, 255, 255)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)

            # Convert the floating confidence to integer for rectangle height
            rect_height = int(100 * confidence) + 10  # Ensure a minimum height for visibility
            painter.drawRect(i * (bar_width + spacing), 0, bar_width, rect_height)

    def update_bars(self, confidences):
        print("Updating bars with new confidences.")  # Debug statement
        self.confidences = confidences
        self.update()  # Trigger a repaint


def signal_handler(sig, frame):
    print("Signal received, cleaning up and exiting...")
    if classifier:
        classifier.cleanup()
    sys.exit(0)

if __name__ == '__main__':
    print("Starting application...")  # Debug statement
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the YAMNet model file (.tflite)')
    parser.add_argument('--audio', required=True, help='Path to the audio file or "0" for microphone input')
    parser.add_argument('--labels', required=True, help='Path to the labels file')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    # Initialize the classifier and the widget
    try:
        print('Initializing classifier...')
        classifier = RealTimeClassifier(args.model, args.labels)
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        sys.exit(1)

    widget = OverlayWidget(classifier.labels)
    widget.show()
    widget.raise_()  # Bring the widget to the front
    widget.activateWindow()  # Activate the window

    # Connect the classifier update signal to the widget update method
    classifier.update_signal.connect(widget.update_bars)

    # Register signal handler for graceful termination
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start audio classification
    if args.audio == '0':
        classifier.classify_from_mic()

    print("Entering Qt event loop...")  # Debug statement
    sys.exit(app.exec_())
