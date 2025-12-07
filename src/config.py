import os

DATASET_PATH = os.path.abspath("placas-transito-10/data.yaml")

EXPERIMENT_MODELS = [
    "yolov5n.pt",
    "yolov8n.pt",
    "yolov11n.pt"
]

RESULTS_DIR = os.path.abspath("experiments_results")

EPOCHS = 1
IMG_SIZE = 640
BATCH = 32