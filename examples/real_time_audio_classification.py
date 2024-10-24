import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
import csv
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

class RealTimeClassifier:
    def __init__(self, model, labels_path):
        # Load the labels from the CSV file
        self.labels = []
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.labels.append(row['display_name'])

        # Initialize the interpreter
        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()

        # Set input tensor
        input_details = self.interpreter.get_input_details()[0]
        self.input_shape = input_details['shape']
        self.num_samples = self.input_shape[0]  # Expected to be 15600 samples without batch dimension

        # Buffer for storing samples for classification
        self.audio_buffer = np.zeros(self.num_samples, dtype=np.float32)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, flush=True)

        # Append new data and maintain buffer size (sliding window)
        indata = indata[:, 0]  # Use the first channel if stereo
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:frames]

        # Check if buffer has enough samples
        if len(self.audio_buffer) == self.num_samples:
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

            # Sort output data in decreasing order of confidence
            sorted_indices = np.argsort(output_data)[::-1]
            top_5_indices = sorted_indices[:5]
            top_5_labels = [self.labels[idx] for idx in top_5_indices]
            top_5_confidences = [output_data[idx] for idx in top_5_indices]

            # Clear previous output and print the current classification
            print("\033[2K\033[F" * 7, end='')  # Clear lines and move cursor up to overwrite previous output

            print(f"Time segment: {time_info.inputBufferAdcTime:.1f} seconds".ljust(80))
            for label, confidence in zip(top_5_labels, top_5_confidences):
                print(f"Class: {label.ljust(25)}, Confidence: {confidence:.3f}".ljust(80))
            print(f"Inference time: {inference_time:.3f} seconds".ljust(80))

    def classify_from_mic(self):
        # Start streaming from the microphone
        with sd.InputStream(channels=1, callback=self.audio_callback, samplerate=16000, blocksize=int(0.1 * 16000)):
            print("Starting real-time audio classification... Press Ctrl+C to stop.")
            while True:
                time.sleep(0.1)

    def classify_from_file(self, audio_path):
        # Load audio file
        audio, sample_rate = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono if stereo

        # Initialize buffer for sliding window
        audio_buffer = np.zeros(self.num_samples, dtype=np.float32)

        # Iterate over the segments in the audio
        step_size = int(0.1 * sample_rate)  # 0.1-second step
        num_segments = len(audio) // step_size

        start_time = time.time()  # Keep track of the real time since processing started

        for i in range(num_segments):
            # Update audio buffer with new data
            segment = audio[i * step_size: i * step_size + step_size]

            # Append new data to the buffer and maintain the fixed length
            audio_buffer = np.roll(audio_buffer, -step_size)
            audio_buffer[-step_size:] = segment[:step_size]

            if len(audio_buffer) == self.num_samples:
                # Use audio buffer as it is, without expanding dimensions
                input_data = audio_buffer.astype(np.float32)  # Shape should be (num_samples,)

                # Set tensor with the current buffer
                self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)

                # Run inference
                self.interpreter.invoke()

                # Get output
                output_data = np.copy(np.squeeze(common.output_tensor(self.interpreter, 0)))

                # Sort output data in decreasing order of confidence
                sorted_indices = np.argsort(output_data)[::-1]
                top_5_indices = sorted_indices[:5]
                top_5_labels = [self.labels[idx] for idx in top_5_indices]
                top_5_confidences = [output_data[idx] for idx in top_5_indices]

                # Calculate the mid-time of the segment for display
                current_time = (i * step_size + self.num_samples / 2) / sample_rate

                # Synchronize with the real-time playback of audio
                expected_time = start_time + current_time
                elapsed_time = time.time()
                if elapsed_time < expected_time:
                    time.sleep(expected_time - elapsed_time)

                # Clear previous output and print the current classification
                print("\033[2K\033[F" * 7, end='')  # Clear lines and move cursor up to overwrite previous output

                print(f"Time segment: {current_time:.1f} seconds".ljust(80))
                for label, confidence in zip(top_5_labels, top_5_confidences):
                    print(f"Class: {label.ljust(25)}, Confidence: {confidence:.3f}".ljust(80))
                print(f"Inference time: {(time.time() - start_time):.3f} seconds".ljust(80))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the YAMNet model file (.tflite)')
    parser.add_argument('--audio', required=True, help='Path to the audio file or "0" for microphone input')
    parser.add_argument('--labels', required=True, help='Path to the labels file')
    args = parser.parse_args()

    classifier = RealTimeClassifier(args.model, args.labels)

    if args.audio == '0':
        classifier.classify_from_mic()
    else:
        classifier.classify_from_file(args.audio)
