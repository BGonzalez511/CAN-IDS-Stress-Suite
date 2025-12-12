import numpy as np
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

# Import your modules
try:
    from deep_learning_models import HybridCNNLSTM
    from dataset_loader import CANDatasetLoader
except ImportError:
    # Fallback if running from a different directory depth
    from src.deep_learning_models import HybridCNNLSTM
    from src.dataset_loader import CANDatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions for Noise ---

def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds thermal noise (AWGN)."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def simulate_emi_bursts(signal: np.ndarray, probability: float = 0.01) -> np.ndarray:
    """Simulates sharp EMI spikes."""
    noisy_signal = signal.copy()
    burst_mask = np.random.random(signal.shape) < probability
    # Spikes +/- 2.0V
    spikes = np.random.uniform(-2.0, 2.0, size=signal.shape) 
    noisy_signal[burst_mask] += spikes[burst_mask]
    return noisy_signal

# --- Main Stress Test ---

def run_stress_test():
    logger.info("--- Starting Deep Learning Stress Test ---")
    
    # 1. GENERATE SYNTHETIC DATA
    logger.info("Step 1: Generating synthetic Voltage & Attack data...")
    loader = CANDatasetLoader("data/raw")
    # Generate enough samples for training and testing
    voltage_df = loader._create_sample_voltage_data(n_samples=4000)
    
    # Preprocess (Normalizing and formatting)
    X_raw, y = loader.preprocess_voltage_data(voltage_df)
    
    # Reshape for CNN: (N, 100) -> (N, 100, 1)
    X = X_raw[..., np.newaxis]
    
    # Split: 80% Train, 20% Test
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    logger.info(f"Data Ready. Train: {len(X_train)}, Test: {len(X_test)}")

    # 2. TRAIN THE MODEL (The Missing Step)
    logger.info("Step 2: Training Model (Quick Run)...")
    
    # Initialize the Hybrid Architecture
    model_wrapper = HybridCNNLSTM(cnn_filters=[32, 64], lstm_units=[32])
    model = model_wrapper.build_model(input_shape=(100, 1))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for 5 epochs - enough to learn the signal shapes
    model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=32, 
        validation_split=0.1, 
        verbose=1
    )
    logger.info("Model Trained. Baseline established.")

    # 3. DEFINE STRESS SCENARIOS
    scenarios = {
        "Baseline (No Noise)": X_test, 
        "Medium Thermal Noise (25dB)": add_gaussian_noise(X_test, snr_db=25),
        "Heavy Thermal Noise (10dB)": add_gaussian_noise(X_test, snr_db=10),
        "EMI Burst (1% Prob)": simulate_emi_bursts(X_test, probability=0.01),
        "Extreme Jitter (5dB)": add_gaussian_noise(X_test, snr_db=5),
    }

    # 4. RUN TESTS
    print(f"\n{'='*20} MODEL ROBUSTNESS RESULTS {'='*20}")
    test_results = {}
    
    for name, X_noisy in scenarios.items():
        # Evaluate
        loss, accuracy = model.evaluate(X_noisy, y_test, verbose=0)
        
        test_results[name] = {
            "accuracy": accuracy
        }
        
        print(f"Scenario: {name:<30} | Accuracy: {accuracy:.4f}")

    # 5. SAVE RESULTS
    with open("stress_test_dl_results.json", 'w') as f:
        json.dump(test_results, f, indent=4)
    logger.info(f"Results saved to stress_test_dl_results.json")

if __name__ == "__main__":
    run_stress_test()