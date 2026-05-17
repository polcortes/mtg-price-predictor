import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
from pathlib import Path
from contextlib import contextmanager

DAYS_BACK = 60
FEATURES = 1  # Renamed for clarity
BATCH_SIZE = 64  # Increased for better GPU utilization
EPOCHS = 15  # More epochs with early stopping

MODEL_DIR = Path(__file__).parent

global_scaler = None


@contextmanager
def get_mongo_collection(db_name="mtg_data", collection_name="card_price", host="localhost", port=27017):
    """Context manager to ensure MongoDB connection is properly closed."""
    client = MongoClient(host, port)
    try:
        yield client[db_name][collection_name]
    finally:
        client.close()


def build_model():
    """Builds and returns the compiled LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(DAYS_BACK, FEATURES)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def process_cards(collection, batch_size=1000):
    """
    Generator that yields training windows directly from MongoDB.
    Fixes the critical bug where each card had its own scaler but only the last was saved.
    Uses percentage returns instead of absolute prices, making the model card-agnostic.
    """
    global global_scaler
    x_buffer, y_buffer = [], []
    global_scaler = MinMaxScaler(feature_range=(0, 1))
    all_returns = []  # Collect returns to fit global scaler once
    
    # First pass: collect all returns to fit a single global scaler
    print("First pass: analyzing price distributions...")
    for card in collection.find({}, {'datePriceMap': 1}):
        price_map = card.get('datePriceMap')
        if not price_map or len(price_map) <= DAYS_BACK + 1:
            continue
            
        prices = pd.Series(price_map, dtype=float)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().resample('D').ffill().dropna()
        
        if len(prices) <= DAYS_BACK:
            continue
            
        # Use log returns instead of absolute prices (stationary, card-agnostic)
        returns = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)
        all_returns.append(returns)
    
    if not all_returns:
        raise ValueError("No valid card data found in MongoDB")
    
    # Fit global scaler on all returns
    all_returns_array = np.vstack(all_returns)
    global_scaler.fit(all_returns_array)
    del all_returns, all_returns_array  # Free memory
    
    print("Second pass: generating sequences...")
    for card in collection.find({}, {'datePriceMap': 1}):
        price_map = card.get('datePriceMap')
        if not price_map:
            continue
            
        prices = pd.Series(price_map, dtype=float)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index().resample('D').ffill().dropna()
        
        if len(prices) <= DAYS_BACK:
            continue
            
        # Calculate log returns (more stationary than raw prices)
        returns = np.log(prices / prices.shift(1)).dropna()
        returns_arr = returns.values.reshape(-1, 1)
        normalized = global_scaler.transform(returns_arr)
        
        # Generate sliding windows
        for i in range(DAYS_BACK, len(normalized)):
            x_buffer.append(normalized[i - DAYS_BACK:i])
            # Target: did price go up tomorrow?
            y_buffer.append(1 if returns.iloc[i] > 0 else 0)
            
            if len(x_buffer) >= batch_size:
                yield np.array(x_buffer), np.array(y_buffer)
                x_buffer, y_buffer = [], []
    
    # Yield remaining buffer
    if x_buffer:
        yield np.array(x_buffer), np.array(y_buffer)


def compile_model():
    global global_scaler
    model = build_model()
    model.summary()
    
    # Prepare callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Collect all data for proper train/validation split
    # (Generator approach is memory-efficient but prevents shuffling across cards)
    print("Loading data from MongoDB...")
    with get_mongo_collection() as collection:
        x_list, y_list = [], []
        for x_batch, y_batch in process_cards(collection, batch_size=5000):
            x_list.append(x_batch)
            y_list.append(y_batch)
    
    x_train = np.concatenate(x_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    
    # Shuffle to prevent card-order bias
    indices = np.random.permutation(len(x_train))
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    print(f"Total samples: {len(x_train)}")
    print(f"Class balance: {np.mean(y_train):.2%} positive")
    
    # Train with validation split
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
        shuffle=True,
        verbose=1
    )
    
    # Save artifacts
    model.save(MODEL_DIR / "model.keras")
    joblib.dump(global_scaler, MODEL_DIR / "scaler.pkl")  # BUG FIX: save the correct scaler
    joblib.dump({
        "DAYS_BACK": DAYS_BACK,
        "FEATURES": FEATURES
    }, MODEL_DIR / "config.pkl")
    
    print(f"Training complete. Best val_loss: {min(history.history['val_loss']):.4f}")
    return history


def save_scaler():
    """
    Fits a global MinMaxScaler on all card log returns and saves it.
    Does NOT touch the model. Safe to run while your model file is untouched.
    """
    print("Collecting returns from MongoDB to fit scaler...")
    all_returns = []

    with get_mongo_collection() as collection:
        for card in collection.find({}, {'datePriceMap': 1}):
            price_map = card.get('datePriceMap')
            if not price_map or len(price_map) <= DAYS_BACK + 1:
                continue

            prices = pd.Series(price_map, dtype=float)
            prices.index = pd.to_datetime(prices.index)
            prices = prices.sort_index().resample('D').ffill().dropna()

            if len(prices) <= DAYS_BACK:
                continue

            returns = np.log(prices / prices.shift(1)).dropna().values.reshape(-1, 1)
            all_returns.append(returns)

    if not all_returns:
        raise ValueError("No valid card data found in MongoDB")

    scaler = MinMaxScaler(feature_range=(0, 1))
    all_returns_array = np.vstack(all_returns)
    scaler.fit(all_returns_array)

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"\nScaler saved: {scaler_path}")
    print(f"  Samples used for fit: {len(all_returns_array):,}")
    print(f"  Return range: [{all_returns_array.min():.6f}, {all_returns_array.max():.6f}]")
    print(f"  Scale: {scaler.scale_[0]:.6f}, Min: {scaler.min_[0]:.6f}")
    joblib.dump({
        "DAYS_BACK": DAYS_BACK,
        "FEATURES": FEATURES
    }, MODEL_DIR / "config.pkl")
    return scaler


def validate_scaler_with_model():
    """
    Quick sanity check: loads your existing model + the newly saved scaler,
    generates a fake 60-day window, and runs a prediction.
    If the output is a normal probability (0-1), your files are compatible.
    """
    if not (MODEL_DIR / "model.keras").exists():
        print("No model.keras found — validation skipped.")
        return

    model = tf.keras.models.load_model(MODEL_DIR / "model.keras")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    # Create a synthetic window of realistic log returns (-0.05 to +0.05)
    fake_returns = np.random.uniform(-0.05, 0.05, (1, DAYS_BACK, FEATURES))
    normalized = scaler.transform(fake_returns.reshape(-1, 1)).reshape(1, DAYS_BACK, FEATURES)

    pred = model.predict(normalized, verbose=0)
    print(f"\nSanity check prediction: {pred[0][0]:.4f} (should be between 0.0 and 1.0)")

    if not (0.0 <= pred[0][0] <= 1.0):
        print("WARNING: Prediction out of range. Your model may have been trained with a different scaler.")
    else:
        print("Model and scaler appear compatible.")


if __name__ == "__main__":
    save_scaler()
    validate_scaler_with_model()
