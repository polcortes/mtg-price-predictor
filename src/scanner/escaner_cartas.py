import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class CardScanner:
    CARD_WIDTH = 450
    CARD_HEIGHT = 630
    MIN_AREA = 5000
    CANNY_LOW = 50
    CANNY_HIGH = 150

    def __init__(self, camera_id: int = 0, output_dir: str = "cartas_recortadas"):
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.contador = 0
        self.cap = None
        self.ultima_carta_detectada = None

    def _preprocess_frame(self, frame):
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(blur, self.CANNY_LOW, self.CANNY_HIGH)
        return bordes

    def _detect_card_contours(self, frame, bordes):
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            area = cv2.contourArea(c)
            if area < self.MIN_AREA:
                continue

            hull = cv2.convexHull(c)
            perimetro = cv2.arcLength(hull, True)
            aproximacion = cv2.approxPolyDP(hull, 0.1 * perimetro, True)

            if len(aproximacion) == 4:
                cv2.drawContours(frame, [aproximacion], 0, (0, 255, 0), 3)

                puntos = aproximacion.reshape(4, 2)
                rect = self._order_corners(puntos)

                destino = np.array([[0, 0], [self.CARD_WIDTH, 0],
                                   [self.CARD_WIDTH, self.CARD_HEIGHT],
                                   [0, self.CARD_HEIGHT]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, destino)

                self.ultima_carta_detectada = cv2.warpPerspective(frame, M, (self.CARD_WIDTH, self.CARD_HEIGHT))
                cv2.imshow('CARTA ESCANEADA', self.ultima_carta_detectada)

    def _order_corners(self, puntos):
        rect = np.zeros((4, 2), dtype="float32")
        s = puntos.sum(axis=1)
        rect[0] = puntos[np.argmin(s)]
        rect[2] = puntos[np.argmax(s)]
        diff = np.diff(puntos, axis=1)
        rect[1] = puntos[np.argmin(diff)]
        rect[3] = puntos[np.argmax(diff)]
        return rect

    def save_detected_card(self) -> Optional[str]:
        if self.ultima_carta_detectada is not None:
            nombre_archivo = self.output_dir / f"carta_{self.contador}.jpg"
            cv2.imwrite(str(nombre_archivo), self.ultima_carta_detectada)
            print(f"--> ¡GUARDADA!: {nombre_archivo}")
            self.contador += 1
            return str(nombre_archivo)
        else:
            print("No veo ninguna carta para guardar...")
            return None

    def start_scanning(self):
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara (ID: {self.camera_id})")

        print("Escáner iniciado. Controles:")
        print("  's' - Guardar carta detectada")
        print("  'q' - Salir del escáner")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            bordes = self._preprocess_frame(frame)
            self._detect_card_contours(frame, bordes)

            cv2.imshow('IA Detector de Cartas', frame)

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord('s'):
                self.save_detected_card()
            elif tecla == ord('q'):
                break

        self.stop_scanning()

    def stop_scanning(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_last_card(self) -> Optional[np.ndarray]:
        return self.ultima_carta_detectada


def scan_card() -> Optional[str]:
    """
    Función principal para activar el escáner de cartas.

    Returns:
        Ruta del archivo guardado si la carta fue capturada, None en caso contrario.
    """
    scanner = CardScanner()
    scanner.start_scanning()
    return None


if __name__ == "__main__":
    scan_card()