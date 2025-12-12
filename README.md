# CAN Bus IDS: Physics-Based Stress Testing Suite

**Deep Learning & Voltage Fingerprinting for Advanced Intrusion Detection**

This repository contains the **Stress Testing & Robustness Evaluation Framework** for a Hybrid CNN-LSTM Intrusion Detection System (IDS). It is designed to validate the system's resilience against physical layer electrical faults (EMI, Thermal Noise, Ground Shifts) and mathematically determine the operational limits of Voltage Fingerprinting.

## Repository Contents

## Project Structure
```text
CAN-IDS-Stress-Suite/
├── src/
│   ├── dataset_loader.py       # Physics simulation engine
│   └── deep_learning_models.py # Hybrid CNN-LSTM architecture
├── tests/
│   ├── stress_test_break_point.py  # Generates the S-Curve
│   ├── stress_test_physics.py      # Generates signal plots
│   └── stress_test_dl_performance.py # Validates accuracy
├── results/                    # Output folder for plots
├── requirements.txt            # Dependencies
└── README.md

### 1. Core Physics Engine (`src/`)
* **`dataset_loader.py`**: A custom simulation engine that generates synthetic CAN voltage waveforms. It simulates RLC circuit characteristics (Ringing, Rise Time, Overshoot) to mimic 5 distinct ECU hardware profiles, creating a realistic physical layer dataset.
* **`deep_learning_models.py`**: Implementation of the **HybridCNNLSTM** architecture. This model fuses a 1D-CNN (for voltage feature extraction) with an LSTM (for temporal sequence analysis).

### 2. Stress Test Modules (`tests/`)

| Script | Purpose | Key Metric |
| :--- | :--- | :--- |
| **`stress_test_physics.py`** | **Visual Verification**: Generates plots showing how "Green" reference signals are distorted by EMI Bursts, Voltage Sags, and Thermal Noise. | Signal Fidelity |
| **`stress_test_break_point.py`** | **Operational Limit Analysis**: Systematically degrades Signal-to-Noise Ratio (SNR) from 30dB to 0dB to find the exact point where detection fails. | Recall vs. SNR |
| **`stress_test_dl_performance.py`** | **Robustness Validation**: Measures classification accuracy stability under fixed noise conditions (e.g., Ignition Noise). | Accuracy Stability |

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/BGonzalez511/CAN-IDS-Stress-Suite.git](https://github.com/BGonzalez511/CAN-IDS-Stress-Suite.git)
   cd CAN-IDS-Stress-Suite
