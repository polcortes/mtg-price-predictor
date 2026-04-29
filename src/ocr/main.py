import json
from pathlib import Path
from difflib import get_close_matches

from func_tesser import extract_card_name_candidates, normalize_text

# =========================
# CONFIG
# =========================
MTGJSON_PATH = "mtgjson.json"
IMGS_FOLDER = "imgs"
JSONS_FOLDER = "jsons"

Path(IMGS_FOLDER).mkdir(exist_ok=True)
Path(JSONS_FOLDER).mkdir(exist_ok=True)


# =========================
# CARGA DE NOMBRES DE CARTAS
# =========================
def load_card_names(mtgjson_path: str):
    path = Path(mtgjson_path)
    if not path.exists():
        print(f"[AVISO] No se encontró {mtgjson_path}. Se usará solo OCR.")
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        names = set()

        # Soporta varios formatos posibles de mtgjson
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict):
                for _, set_data in data["data"].items():
                    if isinstance(set_data, dict) and "cards" in set_data:
                        for card in set_data["cards"]:
                            name = card.get("name")
                            if name:
                                names.add(normalize_text(name))
            elif "cards" in data and isinstance(data["cards"], list):
                for card in data["cards"]:
                    name = card.get("name")
                    if name:
                        names.add(normalize_text(name))

        names = sorted(n for n in names if n)
        print(f"[INFO] Cartas cargadas desde MTGJSON: {len(names)}")
        return names

    except Exception as e:
        print(f"[AVISO] Error cargando {mtgjson_path}: {e}")
        return []


CARD_NAMES = load_card_names(MTGJSON_PATH)




# =========================
# MATCH CONTRA NOMBRES REALES
# =========================
def match_card_name(candidates, card_names):
    if not candidates:
        return None

    # 1. Coincidencia exacta
    for candidate in candidates:
        if candidate in card_names:
            return candidate

    # 2. Coincidencia por contención
    for candidate in candidates:
        for real_name in card_names:
            if candidate in real_name or real_name in candidate:
                return real_name

    # 3. Similaridad difusa
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        matches = get_close_matches(candidate, card_names, n=1, cutoff=0.70)
        if matches:
            match = matches[0]

            # Heurística simple: más palabras en común = mejor
            c_words = set(candidate.split())
            m_words = set(match.split())
            overlap = len(c_words & m_words) / max(1, len(m_words))

            score = overlap
            if score > best_score:
                best_score = score
                best_match = match

    return best_match


# =========================
# PROCESAR UNA IMAGEN
# =========================
def process_image(image_path: str):
    try:
        candidates = extract_card_name_candidates(image_path)
        print(f"  [CANDIDATOS OCR] {candidates}")

        if not candidates:
            print(f"  [ERROR] No se pudo extraer nombre de {Path(image_path).name}")
            return None

        matched = match_card_name(candidates, CARD_NAMES) if CARD_NAMES else None

        final_name = matched or candidates[0]

        print(f"  [OK] Nombre final: {final_name}")

        return {
            "ocr_name": final_name,
            "ocr_candidates": candidates
        }

    except Exception as e:
        print(f"  [ERROR] Error procesando {Path(image_path).name}: {str(e)}")
        return None


# =========================
# MAIN
# =========================
def main():
    print("\n" + "=" * 70)
    print("  EXTRACTOR OCR - NOMBRE DE CARTAS MAGIC")
    print("=" * 70)

    imgs_path = Path(IMGS_FOLDER)
    if not imgs_path.exists():
        print(f"[ERROR] Carpeta '{IMGS_FOLDER}' no encontrada")
        return

    image_files = (
        list(imgs_path.glob("*.jpg")) +
        list(imgs_path.glob("*.JPG")) +
        list(imgs_path.glob("*.jpeg")) +
        list(imgs_path.glob("*.JPEG")) +
        list(imgs_path.glob("*.png")) +
        list(imgs_path.glob("*.PNG")) +
        list(imgs_path.glob("*.webp")) +
        list(imgs_path.glob("*.WEBP"))
    )

    if not image_files:
        print(f"[AVISO] No hay imágenes en '{IMGS_FOLDER}'")
        return

    print(f"\n[PROCESANDO] {len(image_files)} imagen(es)...\n")

    results = {}

    for img_path in sorted(image_files):
        print(f"[{img_path.name}]")
        result = process_image(str(img_path))

        if result:
            results[img_path.stem] = result

    output_file = Path(JSONS_FOLDER) / "resultados.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n[OK] Resultados guardados en: {output_file}")
    print(f"[INFO] Total procesadas correctamente: {len(results)}")


if __name__ == "__main__":
    main()