import os
import time
import torch
import pandas as pd
from ultralytics import YOLO

from logger import get_logger
from config import (
    DATASET_PATH, EXPERIMENT_MODELS, RESULTS_DIR,
    EPOCHS, IMG_SIZE, BATCH
)

os.makedirs(RESULTS_DIR, exist_ok=True)

logger = get_logger(os.path.join(RESULTS_DIR, "training.log"))


def benchmark_model(model_path):
    model = YOLO(model_path)
    img_dir = os.path.join(os.path.dirname(DATASET_PATH), "valid/images")
    img = os.path.join(img_dir, os.listdir(img_dir)[0])

    for _ in range(3):
        model.predict(source=img, imgsz=IMG_SIZE, verbose=False)

    t0 = time.time()
    iterations = 20

    for _ in range(iterations):
        model.predict(source=img, imgsz=IMG_SIZE, verbose=False)

    t1 = time.time()
    avg_time = (t1 - t0) / iterations
    fps = 1 / avg_time
    return avg_time, fps


def run_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")

    results_table = []

    for model_name in EXPERIMENT_MODELS:
        logger.info(f"Starting training for model: {model_name}")

        model = YOLO(model_name)
        model.to(device)

        run_name = model_name.replace(".pt", "_placas")

        train_results = model.train(
            data=DATASET_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            name=run_name,
            project=RESULTS_DIR,
            verbose=True
        )

        best_weights = os.path.join(train_results.save_dir, "weights/best.pt")

        logger.info(f"Validation results for {model_name}")
        val_metrics = model.val()
        map50_95 = val_metrics.box.map
        map50 = val_metrics.box.map50

        logger.info(f"mAP50-95: {map50_95:.4f}")
        logger.info(f"mAP50: {map50:.4f}")

        logger.info("Exporting model to ONNX...")
        model.export(format="onnx")

        latency, fps = benchmark_model(best_weights)
        logger.info(f"Latency: {latency:.4f}s | FPS: {fps:.2f}")

        results_table.append({
            "Model": model_name,
            "mAP50-95": map50_95,
            "mAP50": map50,
            "Latency (s)": latency,
            "FPS": fps,
            "Weights": best_weights
        })

    df = pd.DataFrame(results_table)
    df.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"), index=False)

    logger.info("All experiments completed.\n")
    logger.info(df)


if __name__ == "__main__":
    run_experiments()
