import os
import json
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from thefuzz import process, fuzz

# Load environment variables
load_dotenv()

class MTGCard(BaseModel):
    card_name: str = Field(description="The exact name of the Magic the Gathering card")
    set_name: str = Field(description="The name or the 3-letter code of the set the card belongs to.")

# ==========================================
# 1. CARGA EN MEMORIA (Ejecutado al iniciar)
# ==========================================
def load_mtg_database() -> dict:
    """
    Carga el JSON una vez y crea un diccionario optimizado:
    { "Nombre de Carta": [ {"uuid": "...", "set_name": "...", "set_code": "..."} ] }
    """
    file_path = Path(__file__).parent.parent / "datasets" / "AllPrintings.json"
    
    if not file_path.exists():
        print("Warning: AllPrintings.json not found.")
        return {}
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    card_index = {}
    for set_code, set_data in data.get('data', {}).items():
        set_name = set_data.get('name', '')
        
        for card in set_data.get('cards', []):
            name = card.get('name')
            if not name:
                continue
                
            if name not in card_index:
                card_index[name] = []
                
            card_index[name].append({
                "uuid": card.get('uuid'),
                "set_name": set_name,
                "set_code": set_code # Vital por si Gemini lee el código de 3 letras
            })
            
    return card_index

# Variable global para no recargar el JSON en cada llamada
MTG_DB = load_mtg_database()

# ==========================================
# 2. ANÁLISIS DE GEMINI
# ==========================================
def analyze_card_image(image_path: str) -> dict:
    """Uses Gemini to read the card image and return structured JSON."""
    client = genai.Client()
    img = Image.open(image_path)
    
    prompt = """
    You are an expert in Magic the Gathering. 
    Look at this image of an MTG card. Extract the card name. 
    Then, identify the set it belongs to based on the set symbol or the 3-letter set code in the bottom copyright text.
    Do not guess; read the text and look at the symbols carefully.

    These are the set names we have in our database: {set_names}
    """

    unique_sets = set(printing["set_name"] for printings in MTG_DB.values() for printing in printings)
    set_names = ", ".join(sorted(unique_sets))

    prompt = prompt.format(set_names=set_names)
    print(prompt)

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MTGCard,
            temperature=0.0,
        ),
    )
    
    return json.loads(response.text)

# ==========================================
# 3. PIPELINE PRINCIPAL CORREGIDO
# ==========================================
def process_mtg_card(image_path: str, match_threshold: int = 40) -> dict:
    """Handles OCR via Gemini and contextual validation via TheFuzz."""
    if not MTG_DB:
        return {"error": "Database missing. Cannot perform validation."}

    # 1. OCR Raw de Gemini
    try:
        ocr_result = analyze_card_image(image_path)
        raw_name = ocr_result.get("card_name", "")
        raw_set = ocr_result.get("set_name", "")
    except Exception as e:
        return {"error": f"OCR Pipeline Failed: {e}"}

    # 2. Validar primero la CARTA globalmente
    all_card_names = list(MTG_DB.keys())
    best_name_match = process.extractOne(raw_name, all_card_names, scorer=fuzz.token_sort_ratio)
    
    if not best_name_match or best_name_match[1] < match_threshold:
        return {
            "success": False,
            "error": "Card name could not be identified with confidence.",
            "ocr_raw": ocr_result
        }
        
    matched_card_name = best_name_match[0]
    name_score = best_name_match[1]
    
    # Obtener todas las impresiones (sets) donde existe esta carta específica
    printings = MTG_DB[matched_card_name]

    # 3. Validar el SET contextualmente (solo entre las opciones válidas de esta carta)
    best_uuid = None
    best_set_score = -1
    matched_set_name = None
    
    for printing in printings:
        # A) Revisar si Gemini devolvió el código exacto de 3 letras (ej. "10E")
        if raw_set.upper() == printing["set_code"].upper():
            best_uuid = printing["uuid"]
            matched_set_name = printing["set_name"]
            best_set_score = 100
            break
            
        # B) Si no, hacer Fuzzy Match contra el nombre completo del set
        score = fuzz.token_sort_ratio(raw_set, printing["set_name"])
        if score > best_set_score:
            best_set_score = score
            best_uuid = printing["uuid"]
            matched_set_name = printing["set_name"]

    is_set_valid = best_set_score >= match_threshold

    # 4. Construir respuesta
    return {
        "success": is_set_valid,
        "ocr_raw": ocr_result,
        # Si encuentra el set con confianza, devuelve el UUID. 
        # Si no, puedes optar por devolver la 1ra impresión por defecto (printings[0]["uuid"]) o None.
        "card_uuid": best_uuid if is_set_valid else None, 
        "validated_data": {
            "card_name": matched_card_name,
            "set_name": matched_set_name if is_set_valid else None,
        },
        "confidence_scores": {
            "card_name_score": name_score,
            "set_name_score": best_set_score
        }
    }