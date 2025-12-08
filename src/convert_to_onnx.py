import os
import glob
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
from config import IMG_SIZE

RESULTS_DIR = "experiments_results"

def convert_and_quantize():
    # Procura por best.pt
    search_pattern = os.path.join(RESULTS_DIR, '**', 'weights', 'best.pt')
    model_paths = glob.glob(search_pattern, recursive=True)

    if not model_paths:
        print("Nenhum modelo best.pt encontrado.")
        return

    for model_path in model_paths:
        print(f"\n--- Convertendo modelo: {model_path} ---")
        
        model = YOLO(model_path)

        # Exportar ONNX
        onnx_path = model_path.replace(".pt", ".onnx")
        model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            simplify=True,
            opset=12
        )

        print(f"ONNX gerado em: {onnx_path}")

        # Gerar nome do arquivo quantizado
        quantized_path = model_path.replace("best.pt", "best_int8.onnx")

        print(f"Quantizando INT8 → {quantized_path}")

        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8
        )

        print(f"✔ Quantização concluída: {quantized_path}")

    print("\nProcesso finalizado!")


if __name__ == "__main__":
    convert_and_quantize()
