import time
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# Integration with your new loader
try:
    from dataset_loader import CANDatasetLoader
except ImportError:
    from src.dataset_loader import CANDatasetLoader

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [STRESS_TEST] - %(message)s')
logger = logging.getLogger(__name__)

class PhysicalLayerStressTest:
    """
    Adapts the logic of the previous stress test to work with 
    Voltage Waveforms instead of Logical IDs.
    """

    def __init__(self, datapath="data/raw"):
        self.loader = CANDatasetLoader(datapath)
        self.raw_signals = None
        self.labels = None
        
    def generate_baseline_data(self, n_samples=200):
        """
        Uses the dataset_loader's physics engine to create ground truth.
        """
        logger.info("Generatng baseline physical layer data...")
        # CITATION: Uses the synthetic generation from the uploaded file
        df = self.loader._create_sample_voltage_data(n_samples=n_samples)
        
        # Extract raw voltage arrays (handle string storage if necessary)
        if isinstance(df['voltage_samples'].iloc[0], str):
            self.raw_signals = np.array(df['voltage_samples'].apply(eval).tolist())
        else:
            self.raw_signals = np.array(df['voltage_samples'].tolist())
            
        self.labels = df['label'].values
        logger.info(f"Baseline generated: {self.raw_signals.shape} samples")

    def _inject_physics_noise(self, signal: np.ndarray, scenario: str) -> np.ndarray:
        """
        Applies specific automotive electrical faults.
        """
        noisy = signal.copy()
        
        if scenario == "THERMAL_NOISE":
            # Standard Gaussian noise (AWGN)
            noise = np.random.normal(0, 0.15, signal.shape)
            return noisy + noise
            
        elif scenario == "VOLTAGE_SAG":
            # Battery voltage drop (common during engine crank)
            return noisy * 0.85 
            
        elif scenario == "EMI_BURST":
            # High frequency bursts from ignition coils
            burst_mask = np.random.random(signal.shape) < 0.05
            noisy[burst_mask] += np.random.choice([-1.5, 1.5], size=np.sum(burst_mask))
            return noisy
            
        elif scenario == "GROUND_SHIFT":
            # Ground Potential Difference (GPD) between ECUs
            return noisy + 0.4
            
        return noisy

    def run_latency_check(self):
        """
        Measures preprocessing latency using the new loader's pipeline.
        Matches [TEST 1] from your old code.
        """
        print(f"\n{'='*10} [TEST 1] SYSTEM LATENCY {'='*10}")
        
        # Mock dataframe to test pipeline speed
        df_chunk = pd.DataFrame({
            'voltage_samples': list(self.raw_signals[:50]), 
            'label': self.labels[:50]
        })
        
        start = time.time()
        # CITATION: invoking the preprocessing method from loader
        _, _ = self.loader.preprocess_voltage_data(df_chunk)
        duration = time.time() - start
        
        latency_ms = (duration * 1000) / 50
        print(f"Pipeline Latency: {latency_ms:.4f} ms per message")
        
        if latency_ms < 1.0:
            print("Status: PASSED (Real-time capable)")
        else:
            print("Status: WARNING (Too slow for 1ms CAN cycle)")

    def run_robustness_check(self):
        """
        Measures signal degradation under stress.
        Matches [TEST 2] from your old code, but uses Signal Correlation instead of Accuracy.
        """
        print(f"\n{'='*10} [TEST 2] SIGNAL ROBUSTNESS {'='*10}")
        
        scenarios = ["THERMAL_NOISE", "VOLTAGE_SAG", "EMI_BURST", "GROUND_SHIFT"]
        
        results = []
        
        for scenario in scenarios:
            # Apply Noise
            noisy_signals = np.array([self._inject_physics_noise(s, scenario) for s in self.raw_signals])
            
            # Measure Degradation (Pearson Correlation Coefficient)
            # A correlation of 1.0 means the fingerprint is perfectly preserved.
            # A correlation < 0.8 means the hardware fingerprint is lost.
            correlations = [np.corrcoef(self.raw_signals[i], noisy_signals[i])[0,1] 
                           for i in range(len(self.raw_signals))]
            avg_corr = np.mean(correlations)
            
            status = "ROBUST" if avg_corr > 0.9 else "DEGRADED" if avg_corr > 0.75 else "FAILURE"
            
            print(f"Scenario: {scenario:<15} | Fidelity: {avg_corr:.4f} | Status: {status}")
            results.append((scenario, noisy_signals[0]))

        return results

    def visualize(self, results):
        """Helper to see the damage done to the signals"""
        plt.figure(figsize=(12, 8))
        
        # Plot Clean
        plt.subplot(len(results)+1, 1, 1)
        plt.plot(self.raw_signals[0], color='green', label='Clean ECU Fingerprint')
        plt.legend()
        plt.title("Reference Signal (Generated by CANDatasetLoader)")
        
        # Plot Scenarios
        for i, (name, signal) in enumerate(results):
            plt.subplot(len(results)+1, 1, i+2)
            plt.plot(signal, color='red', alpha=0.7, label=f'Stressed: {name}')
            plt.plot(self.raw_signals[0], color='green', alpha=0.2, linestyle='--') # Reference
            plt.legend(loc='upper right')
            
        plt.tight_layout()
        plt.savefig('stress_test_physics.png')
        print("\nVisualization saved to 'stress_test_physics.png'")

if __name__ == "__main__":
    # 1. Initialize
    test = PhysicalLayerStressTest()
    
    # 2. Generate Data (using the new loader's physics engine)
    test.generate_baseline_data(n_samples=200)
    
    # 3. Run Latency Test
    test.run_latency_check()
    
    # 4. Run Robustness Test
    scenario_data = test.run_robustness_check()
    
    # 5. Visualize
    test.visualize(scenario_data)