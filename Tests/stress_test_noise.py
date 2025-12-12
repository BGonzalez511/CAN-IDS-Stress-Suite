import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# --- FIX: IMPORT THE CORRECT CLASS NAME ---
try:
    # If running as "python -m src.stress_test_noise"
    from src.dataset_loader import CANDatasetLoader 
except ImportError:
    # If running script directly
    from dataset_loader import CANDatasetLoader

def preprocess_for_stress_test(df, seq_len=10):
    """
    Minimal preprocessing to match LSTM input shape.
    Matches logic in your main_experiment.py pipeline.
    """
    # 1. Feature Selection (Matches your CANDatasetLoader output)
    feature_cols = ['can_id', 'dlc'] + [f'data_{i}' for i in range(8)]
    
    # 2. Scaling
    scaler = MinMaxScaler()
    features = scaler.fit_transform(df[feature_cols].values)
    
    # 3. Sequence Creation
    X, y = [], []
    # Ensure labels are numeric. If 'Attack'/'Benign', map them.
    if df['label'].dtype == object:
        labels = df['label'].apply(lambda x: 1 if str(x).lower() in ['attack', '1'] else 0).values
    else:
        labels = df['label'].values
    
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(labels[i + seq_len])
        
    return np.array(X), np.array(y)

def run_stress_test(model_path, X_test, y_test):
    print(f"\n{'='*20} STRESS TEST REPORT {'='*20}")
    print(f"Model: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Status: Model loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        return

    # TEST 1: LATENCY
    print(f"\n[TEST 1] Latency Check")
    start = time.time()
    _ = model.predict(X_test, verbose=0)
    latency = ((time.time() - start) * 1000) / len(X_test)
    print(f"   Avg Latency: {latency:.4f} ms")

    # TEST 2: NOISE ROBUSTNESS
    print(f"\n[TEST 2] Noise Robustness")
    for factor in [0.01, 0.05, 0.1]:
        noise = np.random.normal(0, factor, X_test.shape)
        X_noisy = X_test + noise
        acc = model.evaluate(X_noisy, y_test, verbose=0)[1]
        print(f"   Noise {factor}: Accuracy = {acc:.4f}")

if __name__ == "__main__":
    # CONFIGURATION
    MODEL_PATH = 'lstm_model.h5' 
    
    print("--- Setting up Stress Test ---")
    
    # 1. Trigger CANDatasetLoader
    # It automatically generates data into 'loader.raw_data' if file is missing
    print("1. Triggering CANDatasetLoader...")
    
    # Pass the string directly
    loader = CANDatasetLoader("data/non_existent_file.csv")

    # FIX: Access the .raw_data attribute directly
    df = loader.raw_data
    
    # 2. Process
    print("2. Processing data for LSTM...")
    if df is None or df.empty:
        print("ERROR: Dataframe is empty. The generator might have failed.")
    else:
        X, y = preprocess_for_stress_test(df, seq_len=10)
        
        # 3. Run
        if len(X) == 0:
            print("ERROR: No sequences created. Data might be too short.")
        else:
            split = int(len(X) * 0.8)
            run_stress_test(MODEL_PATH, X[split:], y[split:])