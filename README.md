# raspi-road-vision
Real-time Brazilian traffic sign detection running on a Raspberry Pi 4 using lightweight deep learning models.

## Project Structure

```text
.

├── src/
│   ├── capture_video.py               # Script to capture video from the webcam
│   ├── config.py                      # Configuration file for models, paths, and parameters
│   ├── convert_to_onnx.py             # Script to convert .pt models to .onnx format
│   ├── inference_test.py              # Script to measure inference time and FPS on the test set
│   ├── inference_video.py             # Script to perform inference on pre-recorded videos
│   ├── logger.py                      # Module for logging configuration
│   ├── realtime_inference.py          # Script to perform real-time inference using the webcam
│   ├── train.py                       # Script to fine-tune the models defined in config.py
    └── notebooks/
        ├── 01-exploratory_analysis.ipynb  # Notebook for exploratory data analysis
        └── 02-results_analysis.ipynb      # Notebook for analyzing trained model results