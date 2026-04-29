import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from pymongo import MongoClient
from random import randint
from pathlib import Path

DAYS_BACK = 60 
caracteristicas = 1

model_path = Path(__file__).parent

def compile_model():
    model = Sequential([
        # Capa LSTM para analizar la secuencia temporal
        LSTM(64, return_sequences=True, input_shape=(DAYS_BACK, caracteristicas)),
        Dropout(0.2), # Apaga neuronas al azar para evitar que el model se memorice los datos (Overfitting)
        
        # Segunda capa LSTM para refinar los patrones
        LSTM(32),
        Dropout(0.2),
        
        # Capa de razonamiento final
        Dense(16, activation='relu'),
        
        # Capa de salida: 1 sola neurona con función Sigmoide.
        # Te devolverá un porcentaje entre 0 y 1 (ej. 0.85 significa 85% de probabilidad de que SUBA).
        Dense(1, activation='sigmoid') 
    ])

    # 2. Compilar el model
    # 'binary_crossentropy' es la función matemática ideal para respuestas de Sí/No (Sube/Baja)
    model.compile(optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy'])

    # Resumen de lo que acabamos de crear
    model.summary()

    # 3. Entrenar (Suponiendo que ya procesaste tus datos en X_train, y_train)
    # print("Entrenando experto en Magic...")
    # historial_entrenamiento = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    DB_NAME = "mtg_data"
    COLLECTION_NAME = "card_price"

    client = MongoClient("localhost", 27017)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    x_list = []
    y_list = []

    for card in collection.find():
        prices_map = card.get('datePriceMap', {})

        if not prices_map:
            continue

        prices_series = pd.Series(prices_map)
        prices_series.index = pd.to_datetime(prices_series.index)
        prices_series = prices_series.sort_index()

        prices_series = prices_series.resample('D').ffill()

        if len(prices_series) <= DAYS_BACK:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_arr = prices_series.values.reshape(-1, 1)
        normalized_prices = scaler.fit_transform(prices_arr)

        for i in range(DAYS_BACK, len(normalized_prices)):
            window = normalized_prices[i - DAYS_BACK: i]
            x_list.append(window)

            yesterday_price = prices_arr[i - 1][0]
            today_price = prices_arr[i][0]

            is_increasing = True if today_price > yesterday_price else 0
            y_list.append(is_increasing)

    x_train = np.array(x_list)
    y_train = np.array(y_list)

    model.fit(
        x_train, 
        y_train, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2
    )

    model.save(model_path / "model.keras")
    joblib.dump(scaler, model_path / "scaler.pkl")
    joblib.dump({
        "DAYS_BACK": DAYS_BACK,
        "caracteristicas": caracteristicas
    }, model_path / "config.pkl")

if __name__ == "__main__":
    compile_model()