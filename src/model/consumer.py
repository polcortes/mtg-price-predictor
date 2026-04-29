import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from typing import Dict, Union
from pathlib import Path

# Type alias para claridad
DatePricesMap = Dict[str, Union[int, float]]

model_path = Path(__file__).parent

class AIConsumer:
    def __init__(self):
        self.model = load_model(model_path / "model.keras")
        self.scaler = joblib.load(model_path / "scaler.pkl")
        self.config = joblib.load(model_path / "config.pkl")
        self.days_back = self.config["DAYS_BACK"]
        
    def predict(self, prices_map: DatePricesMap) -> Union[float, None]:
        """
        Predice la probabilidad de subida basada en el historial de precios.
        
        Args:
            prices_map: Diccionario {fecha: precio} ej: {"2024-01-01": 10.5, ...}
            
        Returns:
            float: Probabilidad entre 0 y 1, o None si no hay suficientes datos
        """
        if not prices_map or len(prices_map) < self.days_back:
            print(f"Error: Se necesitan al menos {self.days_back} días de datos")
            return None
        
        # Convertir a Series de pandas
        prices_series = pd.Series(prices_map)
        prices_series.index = pd.to_datetime(prices_series.index)
        prices_series = prices_series.sort_index()
        
        # Rellenar días faltantes (igual que en entrenamiento)
        prices_series = prices_series.resample('D').ffill()
        
        if len(prices_series) < self.days_back:
            return None
            
        # Tomar los últimos DAYS_BACK días
        recent_prices = prices_series.values.reshape(-1, 1)
        
        # IMPORTANTE: Usar transform, NO fit_transform (usar los parámetros del entrenamiento)
        normalized = self.scaler.transform(recent_prices)
        
        # Tomar la última ventana
        last_window = normalized[-self.days_back:]
        
        # Reshape para LSTM: (samples, time_steps, features) -> (1, 60, 1)
        X = last_window.reshape(1, self.days_back, 1)
        
        # Predecir
        prediction = self.model.predict(X, verbose=0)
        return float(prediction[0, 0])