# GoogleCoral Examples with EdgeTPU

## 🌱 Overview

Welcome to the GoogleCoral examples repository! This repository contains scripts that utilize the Coral Edge TPU for real-time audio classification and semantic segmentation.
The repository structure is as follows:

```
.
├── examples
│   ├── classifyspectrum_audio.py
│   ├── classify_audio.py
│   ├── real_time_audio_classification.py
│   └── semantic_seg.py
└── test_data
    ├── deeplab_mobilenet_edgetpu_slim_cityscapes_quant_edgetpu.tflite
    ├── miaow_16k.wav
    ├── speech_whistling2.wav
    ├── yamnet_class_map.csv
    └── yamnet_edgetpu.tflite
```

## 🧜‍♂️ Setup and Environment Creation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ajinkyagorad/GoogleCoral.git
   cd GoogleCoral
   ```

2. **Create and activate a conda environment**:

   ```bash
   conda create --name pycoral_env python=3.8
   conda activate pycoral_env
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Coral USB Accelerator Setup**
   Refer to the official Coral documentation for setting up your Coral USB Accelerator: [Coral Documentation](https://coral.ai/docs/accelerator/get-started/)

## 🔗 Scripts in This Repository

### 🔢 Semantic Segmentation: `semantic_seg.py`

This script uses the DeeplabV3 MobileNet model for performing semantic segmentation on an input image.

**Command to Run**:

```bash
python examples/semantic_seg.py --model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite
```

**Output**:
This will output a segmented version of the image highlighting distinct features such as roads, cars, etc.

### 🎧 Real-Time Audio Classification: `real_time_audio_classification.py`

This script uses YAMNet for real-time audio classification via Coral TPU. It can take live input from a microphone or classify from an audio file.

**Command to Run with Live Microphone**:

```bash
python examples/real_time_audio_classification.py --model test_data/yamnet_edgetpu.tflite --audio 0 --labels test_data/yamnet_class_map.csv
```

- **Input ****`--audio`**: The `--audio` parameter takes either the path to an audio file or `0` for live microphone input.
- `--audio 0`: Uses the first microphone channel as input for real-time classification.

**Output**:

- Displays the top 5 most probable sound classes detected in real-time, such as "Silence", "Speech", or "Dog Bark" with corresponding confidence levels.

**Command to Run with an Audio File**:

```bash
python examples/real_time_audio_classification.py --model test_data/yamnet_edgetpu.tflite --audio test_data/miaow_16k.wav --labels test_data/yamnet_class_map.csv
```

**Example Terminal Output**:

```
Time segment: 0.0 seconds
Class: Silence                  , Confidence: 0.668
Class: Inside, small room       , Confidence: 0.020
Class: Speech                   , Confidence: 0.016
Class: Whispering               , Confidence: 0.016
Class: Clicking                 , Confidence: 0.008
Inference time: 0.000 seconds
```

## 🚧 Troubleshooting & Common Errors

Below are some common issues you might face and how to solve them:

### 🚀 Dimension Mismatch Error

**Error**:

```
ValueError: Cannot set tensor: Dimension mismatch. Got 2 but expected 1 for input 0.
```

**Solution**:

- This error occurs due to incorrect tensor shape being fed into the model. Ensure that the input tensor has the correct shape by removing any unnecessary batch dimension.

### 📈 Python-CFFI Callback Error

**Error**:

```
Exception ignored from cffi callback <function _StreamBase...>
```

**Solution**:

- Ensure the correct sample rate is used (e.g., `16000 Hz`) and that input tensors match the expected dimensions for the YAMNet model.
- Verify that `sounddevice` is correctly installed and the callback method matches the data requirement.

## 🛠️ Dependencies

To install dependencies, use the `requirements.txt` file, which includes:

- `numpy`
- `soundfile`
- `sounddevice`
- `tflite_runtime`
- `matplotlib`
- `pycoral`

You can install them using:

```bash
pip install -r requirements.txt
```

## 🚀 Contributions

Feel free to open pull requests or issues if you'd like to add more examples or improve existing scripts. Contributions are always welcome!

## 👨‍💻 Author

Ajinkya Gorad - [GitHub](https://github.com/ajinkyagorad)

## 🛡 Known Issues

- **Real-Time Audio Latency**: There may be slight delays in real-time audio classification due to inference time. Optimizing the script for faster inference or using more powerful hardware can reduce latency.
- **Console Output Overwriting**: Current script logic replaces only the classification output but keeps a running history above. This behavior ensures visibility of older classification outputs.

Happy coding! 🚀🌟

