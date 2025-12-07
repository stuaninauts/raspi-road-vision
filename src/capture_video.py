import cv2
import os

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível acessar a câmera")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)

    if fps_cam == 0 or fps_cam is None:
        fps_cam = 30  

    print(f"Resolução detectada: {width}x{height}")
    print(f"FPS detectado: {fps_cam}")

    base_filename = "captura"
    extension = ".mp4"
    output_filename = f"{base_filename}{extension}"
    counter = 1

    while os.path.exists(output_filename):
        output_filename = f"{base_filename}_{counter}{extension}"
        counter += 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_filename,
        fourcc,
        fps_cam,
        (width, height)
    )

    print(f"Gravando em '{output_filename}'... pressionar 'q' para parar")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame.")
            break

        cv2.imshow("Webcam", frame)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Gravação encerrada. Vídeo salvo como '{output_filename}'")


if __name__ == "__main__":
    main()
