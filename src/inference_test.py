import os
import time
import glob
import yaml
import torch
import pandas as pd
import argparse
from ultralytics import YOLO

from logger import get_logger
from config import (
    DATASET_PATH, EXPERIMENT_MODELS, RESULTS_DIR, IMG_SIZE
)

os.makedirs(RESULTS_DIR, exist_ok=True)

log_file = os.path.join(RESULTS_DIR, "test_evaluation.log")
logger = get_logger(log_file)


def get_test_data_path(yaml_path):
    """Lê o arquivo data.yaml e retorna o caminho para o conjunto de teste."""
    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        base_dir = os.path.dirname(yaml_path)
        test_path = os.path.join(base_dir, data_config['test'])
        
        if not os.path.isdir(test_path):
            logger.error(f"Diretório de teste não encontrado em: {test_path}")
            return None
        return test_path
    except Exception as e:
        logger.error(f"Erro ao ler o arquivo {yaml_path}: {e}")
        return None


def benchmark_model(model, img_path, iterations=50):
    """Mede a latência e o FPS de um modelo."""
    for _ in range(5):
        model.predict(source=img_path, imgsz=IMG_SIZE, verbose=False)

    t0 = time.time()
    for _ in range(iterations):
        model.predict(source=img_path, imgsz=IMG_SIZE, verbose=False)
    t1 = time.time()

    avg_time = (t1 - t0) / iterations
    fps = 1 / avg_time
    return avg_time, fps


def run_evaluation(model_format: str):
    """Executa a avaliação nos modelos com o formato especificado (pt ou onnx)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Iniciando avaliação para modelos *.{model_format} no dispositivo: {device}")

    fine_tuned_models = glob.glob(
        os.path.join(RESULTS_DIR, '**', 'weights', f'best.{model_format}'), 
        recursive=True
    )
    
    baseline_models = [m.replace('.pt', f'.{model_format}') for m in EXPERIMENT_MODELS]
    
    all_models_to_test = sorted(list(set(baseline_models + fine_tuned_models)))

    if not all_models_to_test:
        logger.error(f"Nenhum modelo *.{model_format} encontrado para testar.")
        return

    test_images_path = get_test_data_path(DATASET_PATH)
    if not test_images_path:
        return
    
    try:
        sample_image = os.path.join(test_images_path, os.listdir(test_images_path)[0])
    except (FileNotFoundError, IndexError):
        logger.error(f"Nenhuma imagem encontrada em {test_images_path} para o benchmark.")
        return

    results_list = []

    for model_path in all_models_to_test:
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Modelo não encontrado, pulando: {model_path}")
                continue

            logger.info(f"--- Avaliando modelo: {model_path} ---")
            model = YOLO(model_path)
            
            if model_format == 'pt':
                model.to(device)

            val_metrics = model.val(data=DATASET_PATH, split='test', imgsz=IMG_SIZE, verbose=False)
            
            map50_95 = val_metrics.box.map
            map50 = val_metrics.box.map50
            precision = val_metrics.box.p[0] 
            recall = val_metrics.box.r[0]
            f1_score = val_metrics.box.f1[0]

            logger.info(f"mAP50-95: {map50_95:.4f}, mAP50: {map50:.4f}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

            latency, fps = benchmark_model(model, sample_image)
            logger.info(f"Latência: {latency:.4f}s | FPS: {fps:.2f}")

            results_list.append({
                "Model": os.path.basename(model_path),
                "Type": "Fine-Tuned" if "experiments_results" in model_path else "Baseline",
                "mAP50-95": map50_95,
                "mAP50": map50,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "Latency(s)": latency,
                "FPS": fps,
                "Path": model_path
            })
        except Exception as e:
            logger.error(f"Falha ao avaliar o modelo {model_path}: {e}")

    if not results_list:
        logger.warning("Nenhum resultado foi gerado.")
        return

    results_df = pd.DataFrame(results_list)
    output_csv_path = os.path.join(RESULTS_DIR, f"test_results_{model_format}.csv")
    results_df.to_csv(output_csv_path, index=False)

    logger.info(f"\n--- Resultados Finais da Avaliação ({model_format.upper()}) ---")
    logger.info(f"\n{results_df.to_string()}")
    logger.info(f"\nResultados salvos em: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia modelos YOLOv5/8 com formato .pt ou .onnx.")
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['pt', 'onnx'], 
        default='pt', 
        help="Formato do modelo a ser avaliado: 'pt' ou 'onnx'."
    )
    args = parser.parse_args()
    
    run_evaluation(model_format=args.format)