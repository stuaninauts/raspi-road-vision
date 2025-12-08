import cv2
import time
import psutil
import csv
from ultralytics import YOLO

def run_inference(video_path, model_path, output_path, csv_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    print(f"🎥 Vídeo: {width}x{height} @ {fps_in:.2f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    prev_time = time.time()
    frame_count = 0

    print("Inferência iniciada... Pressionar 'q' para parar")

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "timestamp",
            "class",
            "confidence",
            "x1", "y1", "x2", "y2",
            "inference_ms",
            "fps",
            "cpu_percent",
            "ram_percent"
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo.")
                break

            frame_count += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # segundos

            start_inf = time.time()
            results = model(frame, imgsz=640, verbose=False)
            inf_time = (time.time() - start_inf) * 1000  # ms

            annotated = results[0].plot()

            curr_time = time.time()
            fps_real = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cpu_percent = psutil.cpu_percent(interval=None)
            ram_percent = psutil.virtual_memory().percent

            dets = results[0].boxes
            num_det = len(dets)

            cv2.putText(annotated, f"FPS: {fps_real:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"Inferencia: {inf_time:.1f} ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"CPU: {cpu_percent:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"RAM: {ram_percent:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(annotated, f"Deteccoes: {num_det}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            for box in dets:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                writer.writerow([
                    frame_count,
                    timestamp,
                    cls,
                    conf,
                    x1, y1, x2, y2,
                    inf_time,
                    fps_real,
                    cpu_percent,
                    ram_percent
                ])

            cv2.imshow("Inferencia YOLO", annotated)
            out.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Parado pelo usuário")
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✔️ Vídeo anotado salvo como: {output_path}")
    print(f"📄 CSV salvo como: {csv_path}")


if __name__ == "__main__":
    numero = 8
    run_inference(
        video_path=f"capturas/captura_{numero}.mp4",
        model_path="/home/stuani/Desktop/2025-2/dli/raspi-road-vision/src/experiments_results/yolo11n_placas7/weights/best.pt",
        output_path=f"11output_inferencia_{numero}.mp4",
        csv_path=f"11deteccoes_{numero}.csv"
    )
