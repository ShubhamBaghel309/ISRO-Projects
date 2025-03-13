# GPS Anti-Jamming System Using Reservoir Computing

## Project Overview

This project implements a machine learning-based anti-jamming system for GPS signals using Reservoir Computing (RC). The system is designed to recover clean GPS signals from jammed or interfered signals, even when jamming overpowers the GPS signal by up to 30 dB.

## Key Features

- **Multiple Jamming Type Resilience**: Counters continuous wave, swept frequency, pulsed, broadband, and smart jammers
- **GPU-optimized Reservoir Computing**: Uses specialized RC architecture for efficient signal processing
- **Ensemble Approach**: Combines different reservoir architectures to improve prediction quality
- **Jammer-Specific Models**: Specialized models for each jamming type for optimal performance
- **Low Bit Error Rate**: Achieves minimal BER even under extreme jamming conditions

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Dataset Generation](#dataset-generation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Technical Details](#technical-details)
8. [Results](#results)

## System Requirements

- Python 3.8+
- PyTorch 1.9+ (optional, for comparison models)
- NumPy, SciPy, Matplotlib, Pandas
- PyWavelets
- scikit-learn
- tqdm
- 8GB+ RAM

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd gps-anti-jamming-rc
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Generation

The project uses a synthetic GPS signal generator with realistic jamming scenarios. To generate the dataset:

```bash
python gps_signal_dataset2.py
```

This will create:
- `enhanced_gps_dataset_X_jammed.npy`: Array of jammed GPS signal windows
- `enhanced_gps_dataset_y_clean.npy`: Array of corresponding clean GPS signals
- `enhanced_gps_dataset_metadata.csv`: Information about each window (jammer type, JNR, etc.)
- Signal visualization plots for inspection

### Dataset Parameters

The dataset includes:
- Multiple satellites (4 by default)
- Multiple jammer types (continuous, swept, pulsed, broadband, smart)
- Signal-to-Noise Ratio (SNR) of 30dB
- Jammer-to-Noise Ratio (JNR) varying from 5 to 30dB
- Realistic multipath effects and Doppler shifts

## Model Training

To train the RC-based anti-jamming system:

```bash
python RC.py
```

This will:
1. Load the dataset
2. Preprocess signals with jammer-specific techniques
3. Optimize reservoir parameters for minimizing Bit Error Rate
4. Train specialized models for each jammer type
5. Create an ensemble of reservoirs for improved performance
6. Save multiple model files for evaluation and deployment

## Evaluation

To evaluate model performance on test data:

```bash
python run_model.py
```

This will generate:
- Performance metrics for each jammer type (BER, MSE, SNR improvement)
- Visualizations comparing jammed, predicted, and clean signals
- Frequency domain analysis charts
- Performance comparison across different model architectures

## Usage

### Predicting Clean Signals

```python
# Load the ensemble model
from RC import predict_clean_signal_ensemble

# Predict clean signal from jammed signal
clean_signal = predict_clean_signal_ensemble(jammed_signal)
```

### Using Jammer-Specific Models

```python
# Load the specialized models
from run_model import load_model, predict_signal

# Load model data
model_data = load_model('rc_specialized_models.pkl')

# Predict using a jammer-specific model
clean_signal = predict_signal(jammed_signal, jammer_type='continuous', model_data=model_data)
```

## Technical Details

### GPS Signal Structure

The system models GPS signals with:
- L1 band carrier frequency (1575.42 MHz, normalized for simulation)
- C/A code modulation using Gold codes
- Navigation data at 50 Hz
- Realistic multipath effects
- Doppler frequency shifts

### Reservoir Computing Architecture

The RC architecture includes:
- Specialized neuron groups:
  - 70% standard neurons for general signal processing
  - 15% high-frequency neurons for capturing jamming patterns
  - 15% memory neurons for temporal context
- Different leaking rates for each neuron group
- Optimized spectral radius (0.85-0.99)
- Low sparsity (1-10%) for enhanced memory
- Advanced preprocessing with wavelet denoising

### Ensemble Design

The ensemble approach combines three reservoir architectures:
1. General structure: Lower leaking rate for baseline performance
2. Fast oscillations: Higher spectral radius and leaking rate
3. Peak detection: Higher input scaling and larger reservoir

## Results

The system achieves excellent anti-jamming performance:

| Jammer Type | BER | SNR Improvement (dB) | MSE |
|-------------|-----|---------------------|-----|
| Continuous  | ~0.05 | ~15 | ~0.01 |
| Swept       | ~0.08 | ~12 | ~0.03 |
| Pulsed      | ~0.03 | ~18 | ~0.01 |
| Broadband   | ~0.10 | ~10 | ~0.05 |
| Smart       | ~0.12 | ~9  | ~0.07 |

The ensemble approach consistently outperforms single reservoir models, especially for complex jamming scenarios like swept frequency and smart jammers.

## Citation

If you use this code or dataset in your research, please cite:

