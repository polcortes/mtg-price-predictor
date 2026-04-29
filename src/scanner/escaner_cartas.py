import cv2
import numpy as np
import os

# 1. Configuración inicial
if not os.path.exists('cartas_recortadas'):
    os.makedirs('cartas_recortadas')

cap = cv2.VideoCapture(0)
contador = 0
ultima_carta_detectada = None # Variable para guardar el último recorte

while True:
    ret, frame = cap.read()
    if not ret: break

    # Pre-procesamiento
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 5000: continue  #valor normal 10_000

        hull = cv2.convexHull(c)
        perimetro = cv2.arcLength(hull, True)
        aproximacion = cv2.approxPolyDP(hull, 0.1 * perimetro, True)

        if len(aproximacion) == 4:
            cv2.drawContours(frame, [aproximacion], 0, (0, 255, 0), 3)

            # Lógica de perspectiva
            puntos = aproximacion.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = puntos.sum(axis=1)
            rect[0] = puntos[np.argmin(s)]
            rect[2] = puntos[np.argmax(s)]
            diff = np.diff(puntos, axis=1)
            rect[1] = puntos[np.argmin(diff)]
            rect[3] = puntos[np.argmax(diff)]

            destino = np.array([[0, 0], [450, 0], [450, 630], [0, 630]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, destino)
            
            # Guardar el recorte en una variable temporal
            ultima_carta_detectada = cv2.warpPerspective(frame, M, (450, 630))
            cv2.imshow('CARTA ESCANEADA', ultima_carta_detectada)

    # 2. Punto de escucha
    tecla = cv2.waitKey(1) & 0xFF
    
    # Mostrar pantalla principal
    cv2.imshow('IA Detector de Cartas', frame)

    # Si pulsas 's' y hay una carta en el visor, la guarda
    if tecla == ord('s'):
        if ultima_carta_detectada is not None:
            nombre_archivo = f"cartas_recortadas/carta_{contador}.jpg"
            cv2.imwrite(nombre_archivo, ultima_carta_detectada)
            print(f"--> ¡GUARDADA!: {nombre_archivo}")
            contador += 1
        else:
            print("No veo ninguna carta para guardar...")

    # Salir con 'q'
    if tecla == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()