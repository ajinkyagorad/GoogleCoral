import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import csv
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to the YAMNet model file (.tflite)')
    parser.add_argument('--audio', required=True, help='Path to the audio file to be classified')
    parser.add_argument('--labels', required=True, help='Path to the labels file')
    args = parser.parse_args()

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

    audio = audio[:num_samples]  # Truncate or pad to match input shape
    if len(audio) < num_samples:
        audio = np.pad(audio, (0, num_samples - len(audio)))

    # Reshape audio to match the expected input shape
    if len(input_shape) == 2:
        audio = np.expand_dims(audio, axis=0)  # Add batch dimension for 2D input
    audio = audio.astype(np.float32)
    
    interpreter.set_tensor(input_details['index'], audio)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    inference_time = end_time - start_time

    # Get output
    output_data = common.output_tensor(interpreter, 0)
    output_data = np.squeeze(output_data)  # Remove unnecessary dimensions
    
    # Sort output data in decreasing order of confidence
    sorted_indices = np.argsort(output_data)[::-1]
    sorted_confidences = output_data[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices if i < len(labels)]

    # Plot histogram of top predictions
    top_n = min(10, len(sorted_labels))
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_labels[:top_n][::-1], sorted_confidences[:top_n][::-1], color='skyblue')
    plt.xlabel('Confidence')
    plt.title('Top Audio Classification Predictions')
    plt.tight_layout()
    plt.show()

    # Print the result
    top_class = sorted_indices[0]
    confidence = sorted_confidences[0]
    print(f"Class: {labels[top_class]}, Confidence: {confidence:.3f}")
    print(f"Inference time: {inference_time:.3f} seconds")

if __name__ == '__main__':
    main()
