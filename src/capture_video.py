import cv2
import os

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Erro: não foi possível acessar a câmera")
        return

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

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
    out = cv2.VideoWriter(output_filename, fourcc, fps_cam, (width, height))

    print(f"Gravando em '{output_filename}'... pressionar 'q' para parar")

    cv2.namedWindow("Controles")

    cv2.createTrackbar("Exposicao", "Controles", 5, 20, nothing)
    cv2.createTrackbar("Brilho", "Controles", 128, 255, nothing)
    cv2.createTrackbar("Contraste", "Controles", 50, 255, nothing)
    cv2.createTrackbar("Saturacao", "Controles", 50, 255, nothing)
    cv2.createTrackbar("Ganho", "Controles", 0, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar frame.")
            break

        slider_exp = cv2.getTrackbarPos("Exposicao", "Controles")
        slider_bri = cv2.getTrackbarPos("Brilho", "Controles")
        slider_con = cv2.getTrackbarPos("Contraste", "Controles")
        slider_sat = cv2.getTrackbarPos("Saturacao", "Controles")
        slider_gain = cv2.getTrackbarPos("Ganho", "Controles")

        exposure_value = -(slider_exp)

        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, slider_bri)
        cap.set(cv2.CAP_PROP_CONTRAST, slider_con)
        cap.set(cv2.CAP_PROP_SATURATION, slider_sat)
        cap.set(cv2.CAP_PROP_GAIN, slider_gain)

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
