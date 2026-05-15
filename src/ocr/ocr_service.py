from pathlib import Path
from difflib import get_close_matches
from typing import Dict, Optional, List
import json

from .func_tesser import extract_card_name_candidates, normalize_text

class OCRService:
    """Servicio OCR para extraer y validar nombres de cartas MTG."""

    def __init__(self, mtgjson_path: str = "mtgjson.json"):
        self.mtgjson_path = mtgjson_path
        self.card_names = self._load_card_names()

    def _load_card_names(self) -> List[str]:
        """Carga nombres de cartas desde mtgjson."""
        path = Path(self.mtgjson_path)
        if not path.exists():
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            names = set()

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

            return sorted(n for n in names if n)

        except Exception:
            return []

    def _match_card_name(self, candidates: List[str]) -> Optional[str]:
        """Encuentra el mejor match entre candidatos y nombres reales."""
        if not candidates or not self.card_names:
            return candidates[0] if candidates else None

        # 1. Coincidencia exacta
        for candidate in candidates:
            if candidate in self.card_names:
                return candidate

        # 2. Coincidencia por contención
        for candidate in candidates:
            for real_name in self.card_names:
                if candidate in real_name or real_name in candidate:
                    return real_name

        # 3. Similaridad difusa
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            matches = get_close_matches(candidate, self.card_names, n=1, cutoff=0.70)
            if matches:
                match = matches[0]
                c_words = set(candidate.split())
                m_words = set(match.split())
                overlap = len(c_words & m_words) / max(1, len(m_words))

                if overlap > best_score:
                    best_score = overlap
                    best_match = match

        return best_match or (candidates[0] if candidates else None)

    def extract_card_name(self, image_path: str) -> Optional[Dict]:
        """
        Extrae el nombre de una carta desde una imagen.

        Args:
            image_path: Ruta a la imagen de la carta

        Returns:
            Dict con 'ocr_name' y 'ocr_candidates', o None si falla
        """
        try:
            candidates = extract_card_name_candidates(image_path)

            if not candidates:
                return None

            matched = self._match_card_name(candidates)

            return {
                "ocr_name": matched,
                "ocr_candidates": candidates
            }

        except Exception:
            return None

    def is_available(self) -> bool:
        """Verifica si el OCR está disponible."""
        return len(self.card_names) > 0
