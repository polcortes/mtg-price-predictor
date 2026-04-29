# -*- coding: utf-8 -*-

import os
import re
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Set

import cv2

# =====================
# CONFIGURAR TESSERACT ANTES DE IMPORTAR PYTESSERACT
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSERACT_EXE = os.path.join(BASE_DIR, "tesseract", "Tesseract-OCR", "tesseract.exe")
TESSDATA_DIR = os.path.join(BASE_DIR, "tesseract", "Tesseract-OCR", "tessdata")

if os.path.exists(TESSDATA_DIR):
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

if PYTESSERACT_AVAILABLE and os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

logger = logging.getLogger(__name__)


# =====================
# NORMALIZACIÓN
# =====================

def normalize_text(s: str) -> str:
    if not s:
        return ""

    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_name(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Mantener letras, espacios, apóstrofe y guion
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9'\- ]+", " ", text)

    # Quitar números sueltos
    text = re.sub(r"\d+", " ", text)

    # Colapsar espacios
    text = re.sub(r"\s+", " ", text).strip()

    # Quitar palabras basura frecuentes del OCR
    junk_words = {
        "ee", "ele", "le", "de", "i", "l", "a", "e", "el", "la", "lo"
    }
    words = text.split()
    while words and words[-1].lower() in junk_words:
        words.pop()

    return " ".join(words).strip()


# =====================
# RECORTES
# =====================

def _save_temp(img, filename: str) -> str:
    cv2.imwrite(filename, img)
    return filename


def crop_name_zones(image_path: str) -> List[str]:
    """
    Genera varios recortes de la franja del nombre para aumentar robustez.
    """
    paths = []

    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"No se pudo leer imagen: {image_path}")
        return [image_path]

    h, w = img.shape[:2]

    crop_specs = [
        # y1, y2, x1, x2
        (0.02, 0.12, 0.04, 0.96),
        (0.025, 0.13, 0.035, 0.93),
        (0.03, 0.14, 0.04, 0.90),   # recorte más corto a la derecha para evitar coste
        (0.035, 0.145, 0.05, 0.88), # aún más conservador
    ]

    for idx, (fy1, fy2, fx1, fx2) in enumerate(crop_specs):
        y1 = int(h * fy1)
        y2 = int(h * fy2)
        x1 = int(w * fx1)
        x2 = int(w * fx2)

        crop = img[y1:y2, x1:x2]
        temp_path = f"temp_name_zone_{idx}.png"
        paths.append(_save_temp(crop, temp_path))

    return paths


# =====================
# PREPROCESADO
# =====================

def preprocess_variants(image_path: str) -> List[str]:
    """
    Genera varias versiones preprocesadas de un recorte.
    """
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = []
    idx = 0

    # Variante 1: gris con contraste
    v1 = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    path1 = f"temp_pre_{idx}.png"
    variants.append(_save_temp(v1, path1))
    idx += 1

    # Variante 2: threshold binario
    _, v2 = cv2.threshold(v1, 150, 255, cv2.THRESH_BINARY)
    path2 = f"temp_pre_{idx}.png"
    variants.append(_save_temp(v2, path2))
    idx += 1

    # Variante 3: Otsu
    blur = cv2.GaussianBlur(v1, (3, 3), 0)
    _, v3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    path3 = f"temp_pre_{idx}.png"
    variants.append(_save_temp(v3, path3))
    idx += 1

    # Variante 4: ampliada
    scale = 2
    up = cv2.resize(v3, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    path4 = f"temp_pre_{idx}.png"
    variants.append(_save_temp(up, path4))
    idx += 1

    return variants


# =====================
# OCR
# =====================

def ocr_image(image_path: str) -> str:
    if not PYTESSERACT_AVAILABLE:
        logger.warning(f"pytesseract no disponible: {image_path}")
        return ""

    try:
        # Una sola línea
        config = "--oem 3 --psm 7"

        text = pytesseract.image_to_string(
            image_path,
            lang="eng+spa",
            config=config
        )

        return text.strip()

    except Exception as e:
        logger.warning(f"Error en OCR de {image_path}: {e}")
        return ""


def extract_card_name_candidates(image_path: str) -> List[str]:
    """
    Ejecuta OCR sobre múltiples recortes y preprocesados.
    Devuelve candidatos limpios, sin duplicados.
    """
    temp_files: List[str] = []
    candidates: List[str] = []
    seen: Set[str] = set()

    try:
        crops = crop_name_zones(image_path)
        temp_files.extend([p for p in crops if p != image_path])

        for crop_path in crops:
            preprocessed = preprocess_variants(crop_path)
            temp_files.extend([p for p in preprocessed if p != crop_path and p != image_path])

            for pre_path in preprocessed:
                raw = ocr_image(pre_path)
                cleaned = clean_name(raw)
                normalized = normalize_text(cleaned)

                if normalized and normalized not in seen and len(normalized) >= 3:
                    seen.add(normalized)
                    candidates.append(normalized)

        return candidates

    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass


# =====================
# UTILIDADES
# =====================

def is_ocr_available() -> bool:
    return PYTESSERACT_AVAILABLE and os.path.exists(TESSERACT_EXE)


def get_ocr_status() -> Dict[str, str]:
    return {
        "pytesseract_available": PYTESSERACT_AVAILABLE,
        "tesseract_exe_found": os.path.exists(TESSERACT_EXE),
        "tessdata_dir_found": os.path.exists(TESSDATA_DIR),
        "tesseract_path": TESSERACT_EXE,
        "tessdata_path": TESSDATA_DIR,
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  OCR CARTAS MAGIC - FUNCIONES")
    print("=" * 70 + "\n")

    status = get_ocr_status()
    for key, value in status.items():
        print(f"{key}: {value}")