import cv2
import time
import psutil
import csv
import os
import argparse
from ultralytics import YOLO
from datetime import datetime

def run_realtime_inference(model_path, source, output_dir):
    print(f"Carregando modelo: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Erro ao abrir a câmera (índice {source})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    if fps_cam == 0 or fps_cam is None:
        fps_cam = 30.0

    print(f"📷 Câmera iniciada: {width}x{height} @ {fps_cam:.2f} FPS")

    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    video_filename = os.path.join(output_dir, f"live_infer_{timestamp_str}.mp4")
    csv_filename = os.path.join(output_dir, f"live_data_{timestamp_str}.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps_cam, (width, height))

    print("Inferência em tempo real iniciada... Pressionar 'q' para parar.")
    
    prev_time = time.time()
    frame_count = 0

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", "timestamp", "class", "confidence", 
            "x1", "y1", "x2", "y2", 
            "inference_ms", "fps", "cpu_percent", "ram_percent"
        ])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Falha ao capturar frame.")
                    break

                frame_count += 1
                curr_timestamp = time.time()

                # --- INFERÊNCIA ---
                start_inf = time.time()
                results = model(frame, imgsz=640, verbose=False, stream=False) 
                inf_time = (time.time() - start_inf) * 1000  # ms

                # --- ANOTAÇÃO ---
                annotated_frame = results[0].plot()

                # --- MÉTRICAS ---
                curr_time = time.time()
                fps_real = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time

                cpu_percent = psutil.cpu_percent(interval=None)
                ram_percent = psutil.virtual_memory().percent

                # --- TEXTO NO VÍDEO ---
                cv2.putText(annotated_frame, f"FPS: {fps_real:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Inf: {inf_time:.1f}ms", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"CPU: {cpu_percent}%", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- CSV ---
                dets = results[0].boxes
                if len(dets) == 0:
                    writer.writerow([
                        frame_count, curr_timestamp, -1, 0, 
                        0, 0, 0, 0, 
                        inf_time, fps_real, cpu_percent, ram_percent
                    ])
                else:
                    for box in dets:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        writer.writerow([
                            frame_count, curr_timestamp, cls, conf,
                            x1, y1, x2, y2,
                            inf_time, fps_real, cpu_percent, ram_percent
                        ])

                # --- SAÍDA ---
                cv2.imshow("Real-time Inference", annotated_frame)
                out.write(annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Parado pelo usuário.")
                    break

        except KeyboardInterrupt:
            print("Interrompido via teclado.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n✔️ Vídeo salvo em: {video_filename}")
    print(f"📄 Dados salvos em: {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferência YOLO em tempo real via Webcam")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Caminho para o modelo (.pt ou .onnx)"
    )
    parser.add_argument(
        "--source", 
        type=int, 
        default=0, 
        help="Índice da câmera (padrão: 0, tente 2 se usar USB externa)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="realtime_results", 
        help="Pasta para salvar os vídeos e CSVs"
    )

    args = parser.parse_args()

    run_realtime_inference(args.model, args.source, args.output)