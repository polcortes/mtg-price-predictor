# Card Scanner Usage Guide

## Overview

El módulo scanner ha sido refactorizado para ser reutilizable en toda la aplicación. Ahora puedes integrar el escaneo de cartas de forma sencilla.

## Importación

```python
from src.scanner import CardScanner, scan_card
```

## Opciones de Uso

### 1. Función Simple (Recomendado para UI)

Para activar el escáner directamente desde un botón:

```python
from src.scanner import scan_card

# Ejecutar el escáner
scan_card()
```

### 2. Clase CardScanner (Control Avanzado)

Para mayor control sobre el proceso:

```python
from src.scanner import CardScanner

scanner = CardScanner(
    camera_id=0,  # ID de la cámara (por defecto: 0)
    output_dir="cartas_recortadas"  # Directorio de salida
)

# Iniciar escaneo interactivo
scanner.start_scanning()

# Obtener la última carta detectada
card_image = scanner.get_last_card()
```

### 3. Con Threading (Para No Bloquear UI)

```python
import threading
from src.scanner import CardScanner

def run_scanner():
    scanner = CardScanner()
    scanner.start_scanning()
    # Procesamiento posterior...

# Ejecutar en thread separado
thread = threading.Thread(target=run_scanner, daemon=True)
thread.start()
```

## Controles del Escáner

- **'s'** - Guardar la carta detectada
- **'q'** - Salir del escáner

## Configuración Personalizable

En la clase `CardScanner` puedes ajustar:

- `CARD_WIDTH = 450` - Ancho de la carta recortada
- `CARD_HEIGHT = 630` - Alto de la carta recortada
- `MIN_AREA = 5000` - Área mínima para detectar una carta
- `CANNY_LOW = 50` - Threshold bajo para detección de bordes
- `CANNY_HIGH = 150` - Threshold alto para detección de bordes

## Ejemplo Completo en MainWindow

```python
def start_card_scanner(self):
    def run_scanner():
        scanner = CardScanner()
        scanner.start_scanning()
        if scanner.get_last_card() is not None:
            self.show_card_from_scanner(scanner.get_last_card())

    thread = threading.Thread(target=run_scanner, daemon=True)
    thread.start()
```

## Métodos Disponibles

### `CardScanner`

- **`start_scanning()`** - Inicia el loop de escaneo interactivo
- **`stop_scanning()`** - Detiene la cámara y cierra ventanas OpenCV
- **`get_last_card()`** - Retorna la última carta detectada como numpy array
- **`save_detected_card()`** - Guarda la carta detectada y retorna la ruta del archivo

## Notas

- La cámara se abre al llamar `start_scanning()`
- Las cartas se guardan en el directorio especificado en formato JPG
- El array de la carta está en formato BGR (OpenCV)
- Si necesitas convertir a RGB: `cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)`
