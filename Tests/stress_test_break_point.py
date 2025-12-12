import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Import your modules
try:
    from deep_learning_models import HybridCNNLSTM
    from dataset_loader import CANDatasetLoader
except ImportError:
    from src.deep_learning_models import HybridCNNLSTM
    from src.dataset_loader import CANDatasetLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def add_variable_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds specific level of thermal noise."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def run_breakpoint_test():
    logger.info("--- Starting Break Point Analysis ---")
    
    # 1. SETUP DATA
    logger.info("Generating Data...")
    loader = CANDatasetLoader("data/raw")
    
    # Generate raw data (approx 10% attacks)
    full_df = loader._create_sample_voltage_data(n_samples=6000)
    
    # --- FIX: BALANCE THE DATA ---
    # We downsample the normal class to match the attack class count
    # This forces the model to learn the attack features quickly.
    attacks = full_df[full_df['label'] == 1]
    n_attacks = len(attacks)
    normals = full_df[full_df['label'] == 0].sample(n=n_attacks, random_state=42)
    
    balanced_df = pd.concat([attacks, normals]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Data Balanced: {len(balanced_df)} samples ({n_attacks} attacks, {n_attacks} normal)")
    
    X_raw, y = loader.preprocess_voltage_data(balanced_df)
    X = X_raw[..., np.newaxis]
    
    # Split
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # 2. TRAIN MODEL
    logger.info("Training Hybrid Model...")
    model_wrapper = HybridCNNLSTM(cnn_filters=[32, 64], lstm_units=[32])
    model = model_wrapper.build_model(input_shape=(100, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall'])
    
    # Increased epochs slightly to ensure convergence
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    logger.info("Model Trained.")

    # 3. RUN CONTINUOUS STRESS LOOP
    # We test SNR from 30dB (Very Clean) down to 0dB (Noise = Signal)
    snr_levels = [30, 25, 20, 15, 12, 10, 8, 5, 2, 0]
    results = []

    print(f"\n{'SNR (dB)':<10} | {'Recall (Attack Detection)':<25} | {'Status'}")
    print("-" * 50)

    for snr in snr_levels:
        # Create noisy version of test set
        X_test_noisy = np.array([add_variable_noise(x, snr) for x in X_test])
        
        # Evaluate
        loss, accuracy, recall = model.evaluate(X_test_noisy, y_test, verbose=0)
        
        # We use 0.85 as the threshold for "Secure" in a balanced dataset
        status = "✅ Secure" if recall > 0.85 else "⚠️ Degraded" if recall > 0.6 else "❌ FAILED"
        print(f"{snr:<10} | {recall:.4f}                    | {status}")
        
        results.append({
            "SNR_dB": snr,
            "Recall": recall,
            "Accuracy": accuracy
        })

    # 4. PLOT THE CURVE
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot Recall Curve
    sns.lineplot(data=df, x="SNR_dB", y="Recall", marker="o", linewidth=3, color="#d62728", label="Attack Detection Rate")
    
    # Add 'Danger Zone' shading
    plt.axhspan(0, 0.6, color='red', alpha=0.1, label="Unsafe Zone")
    plt.axhline(0.85, color='green', linestyle='--', label="Target Reliability")
    
    plt.title("IDS Break Point Analysis: Attack Detection vs. Noise", fontsize=14, weight='bold')
    plt.xlabel("Signal-to-Noise Ratio (dB) [Lower is Noisier]", fontsize=12)
    plt.ylabel("Recall Score", fontsize=12)
    plt.gca().invert_xaxis() # High SNR (Clean) on left, Low SNR (Noisy) on right
    plt.legend()
    
    filename = "presentation_breakpoint_curve.png"
    plt.savefig(filename, dpi=300)
    logger.info(f"\nCurve saved to {filename}")
    # plt.show() # Commented out to prevent blocking on remote servers

if __name__ == "__main__":
    run_breakpoint_test()