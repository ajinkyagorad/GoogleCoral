import argparse
import numpy as np
import soundfile as sf
import csv
import time
import matplotlib.pyplot as plt
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common


def classify_audio(args):
    # Load the labels from the CSV file
    labels = []
    with open(args.labels, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row['display_name'])

    # Load audio file
    audio, sample_rate = sf.read(args.audio)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono if stereo

    # Initialize the interpreter
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    # Set input tensor
    input_details = interpreter.get_input_details()[0]
    input_shape = input_details['shape']
    if len(input_shape) == 2:
        num_samples = input_shape[1]  # Number of samples expected by the model
    elif len(input_shape) == 1:
        num_samples = input_shape[0]  # Handle models with 1D input shape
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}")

    segment_length = int(0.1 * sample_rate)  # 0.1-second segments
    num_segments = len(audio) // segment_length

    # Create data structures to store results over time
    time_steps = []
    class_confidences = {label: [] for label in labels}

    # Iterate over the segments in the audio
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = audio[start_idx:end_idx]

        # Truncate or pad the segment to match the expected input shape
        segment = segment[:num_samples]
        if len(segment) < num_samples:
            segment = np.pad(segment, (0, num_samples - len(segment)))

        # Reshape audio to match the expected input shape
        if len(input_shape) == 2:
            segment = np.expand_dims(segment, axis=0)  # Add batch dimension for 2D input
        segment = segment.astype(np.float32)

        interpreter.set_tensor(input_details['index'], segment)

        # Run inference
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        inference_time = end_time - start_time

        # Get output
        output_data = np.copy(np.squeeze(common.output_tensor(interpreter, 0)))  # Copy immediately to avoid memory references

        # Store the results
        time_steps.append(i * 0.1)  # Time in seconds
        for label_idx, confidence in enumerate(output_data):
            if label_idx < len(labels):
                class_confidences[labels[label_idx]].append(confidence)

        # Print the result
        top_class = np.argmax(output_data)
        confidence = output_data[top_class]
        print(f"Class: {labels[top_class]}, Confidence: {confidence:.3f}")
        print(f"Inference time: {inference_time:.3f} seconds")

    # Plot the results as a spectrogram-like representation
    plt.figure(figsize=(12, 8))
    for label, confidences in class_confidences.items():
        plt.plot(time_steps, confidences, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Confidence')
    plt.title('Audio Classification Confidence Over Time')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the YAMNet model file (.tflite)')
    parser.add_argument('--audio', required=True, help='Path to the audio file to be classified')
    parser.add_argument('--labels', required=True, help='Path to the labels file')
    args = parser.parse_args()

    classify_audio(args)
