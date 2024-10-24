# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Performs continuous semantic segmentation with the laptop's webcam.

Simply run the script and it will display the segmentation result in real-time.

Usage:

    python3 real_time_segmentation.py --model path/to/your/model.tflite
Dependencies:
    - numpy
    - pillow (PIL)
    - opencv-python
    - pycoral

Ensure the Coral USB Accelerator is correctly set up and the model is compatible with the Edge TPU.

Press 'q' to exit the real-time display.
"""

import argparse
import numpy as np
from PIL import Image
import cv2  # OpenCV for capturing and displaying frames
from pycoral.adapters import common, segment
from pycoral.utils.edgetpu import make_interpreter
import time


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        A 2D array with floating type representing the colored segmentation map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('Label value too large.')

    return colormap[label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='Path of the segmentation model.')
    parser.add_argument('--keep_aspect_ratio',
                        action='store_true',
                        default=False,
                        help=(
                            'Keep the image aspect ratio when down-sampling the image by adding '
                            'black pixel padding (zeros) on bottom or right. '
                            'By default, the image is resized and reshaped without cropping. This '
                            'option should be the same as what is applied on input images during '
                            'model training. Otherwise, accuracy may be affected, and the '
                            'bounding box of detection results may be stretched.'))
    args = parser.parse_args()

    # Initialize the interpreter with the Edge TPU
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    width, height = common.input_size(interpreter)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Use the default camera (usually the front-facing one)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print('Starting real-time segmentation...')

    while True:
        # Capture frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert frame to PIL format
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image if necessary
        if args.keep_aspect_ratio:
            resized_img, _ = common.set_resized_input(
                interpreter, pil_image.size, lambda size: pil_image.resize(size, Image.LANCZOS))
        else:
            resized_img = pil_image.resize((width, height), Image.LANCZOS)
            common.set_input(interpreter, resized_img)

        # Run inference and measure time
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        print(f'Inference time: {inference_time:.3f} seconds')

        # Get the segmentation result
        result = segment.get_output(interpreter)
        if len(result.shape) == 3:
            result = np.argmax(result, axis=-1)

        # Remove padding if keeping aspect ratio
        new_width, new_height = resized_img.size
        result = result[:new_height, :new_width]
        mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))

        # Convert mask image to OpenCV format
        mask_img_cv = cv2.cvtColor(np.array(mask_img), cv2.COLOR_RGB2BGR)

        # Resize the mask to match the frame size for concatenation
        mask_img_cv_resized = cv2.resize(mask_img_cv, (frame.shape[1], frame.shape[0]))

        # Display the frame
        combined_image = np.hstack((frame, mask_img_cv_resized))
        cv2.imshow('Real-Time Semantic Segmentation', combined_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
