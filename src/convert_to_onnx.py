import os
import glob
from ultralytics import YOLO

RESULTS_DIR = "experiments_results"

def convert_models_to_onnx():
    """
    Encontra todos os modelos 'best.pt' no diretório de resultados e os converte para o formato ONNX.
    """
    search_pattern = os.path.join(RESULTS_DIR, '**', 'weights', 'best.pt')
    
    model_paths = glob.glob(search_pattern, recursive=True)

    if not model_paths:
        print(f"Nenhum modelo 'best.pt' encontrado no diretório '{RESULTS_DIR}'.")
        return

    print(f"Encontrados {len(model_paths)} modelos para conversão para ONNX.")

    for model_path in model_paths:
        try:
            print(f"\n--- Convertendo modelo: {model_path} ---")
            
            model = YOLO(model_path)
            
            model.export(format="onnx")
            
        except Exception as e:
            print(f"ERRO: Falha ao converter o modelo {model_path}.")
            print(f"Detalhes do erro: {e}")

    print("\nProcesso de conversão concluído.")


if __name__ == "__main__":
    convert_models_to_onnx()