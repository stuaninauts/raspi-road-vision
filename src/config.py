import os

DATASET_PATH = os.path.abspath("placas-transito-10/data.yaml")

EXPERIMENT_MODELS = [
    "yolov5nu.pt",
    "yolov8n.pt",
    "yolo11n.pt"
]

RESULTS_DIR = os.path.abspath("experiments_results")

EPOCHS = 50
IMG_SIZE = 640
BATCH = 32