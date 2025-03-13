import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Time parameters
fs = 100  # Sampling frequency
duration = 30  # Duration in seconds
t = np.linspace(0, duration, int(fs * duration))  # Time vector

# Carrier signal parameters
carrier_freq = 1.0  # Normalized frequency for visualization
carrier = np.sin(2 * np.pi * carrier_freq * t)

# Generate C/A code (binary pseudorandom sequence)
def generate_ca_code(length, chip_duration):
    """Generate a simplified C/A code"""
    num_chips = int(length / chip_duration)
    # Generate random binary sequence for demonstration
    ca_seq = np.ones(num_chips)
    for i in range(1, num_chips):
        if np.random.rand() > 0.5:
            ca_seq[i] = -1
    
    # Expand each chip to chip_duration samples
    ca_code = np.repeat(ca_seq, chip_duration)
    # Ensure the length matches our time vector
    return ca_code[:length]

# Generate navigation data (much slower than C/A code)
def generate_nav_data(length, bit_duration):
    """Generate navigation data bits"""
    num_bits = int(length / bit_duration)
    # Generate random binary sequence
    nav_seq = np.ones(num_bits)
    for i in range(num_bits):
        if np.random.rand() > 0.5:
            nav_seq[i] = -1
    
    # Expand each bit to bit_duration samples
    nav_data = np.repeat(nav_seq, bit_duration)
    # Ensure the length matches our time vector
    return nav_data[:length]

# Generate signals
samples = len(t)
ca_code = generate_ca_code(samples, int(fs * 3))  # C/A code with chips of 3 seconds
nav_data = generate_nav_data(samples, int(fs * 10))  # Nav data with bits of 10 seconds

# Modulate the signals
ca_modulated = carrier * ca_code
gps_signal = ca_modulated * nav_data  # Complete modulated GPS signal

# Generate jamming signal and noise
jamming_power = 0.5
noise_power = 0.3
jamming = jamming_power * np.sin(2 * np.pi * 1.1 * carrier_freq * t + 0.3)  # Jamming with slight frequency offset
noise = noise_power * np.random.randn(samples)  # Gaussian noise

# Combine to create jammed signal
jammed_signal = gps_signal + jamming + noise

# Simplified demodulation (in reality, more complex algorithms would be used)
demodulated_signal = jammed_signal * np.sin(2 * np.pi * carrier_freq * t)

# Apply simple low-pass filtering (moving average) to simulate integration
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

integrated_signal = moving_average(demodulated_signal, 10)

# Create dataset
dataset = pd.DataFrame({
    'time': t,
    'carrier': carrier,
    'ca_code': ca_code,
    'navigation_data': nav_data,
    'modulated_signal': gps_signal,
    'jamming': jamming,
    'noise': noise,
    'jammed_signal': jammed_signal,
    'demodulated_signal': demodulated_signal,
    'integrated_signal': integrated_signal
})

# Export to CSV
dataset.to_csv('gps_signal_dataset.csv', index=False)

# Plot signals to verify similarity with the diagram
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(t, carrier)
plt.title('Carrier Signal')
plt.ylim(-1.2, 1.2)

plt.subplot(5, 1, 2)
plt.plot(t, ca_code)
plt.title('C/A Code')
plt.ylim(-1.2, 1.2)

plt.subplot(5, 1, 3)
plt.plot(t, nav_data)
plt.title('Navigation Data')
plt.ylim(-1.2, 1.2)

plt.subplot(5, 1, 4)
plt.plot(t, gps_signal)
plt.title('Modulated GPS Signal')
plt.ylim(-1.2, 1.2)

plt.subplot(5, 1, 5)
plt.plot(t, jammed_signal)
plt.title('Jammed GPS Signal')
plt.ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig('gps_signals.png')

# Create ML training dataset for anti-jamming
def create_ml_dataset(jammed_signals, original_ca_code, window_size=100):
    """Create a dataset for training an ML model to recover C/A code from jammed signals"""
    X = []  # Input features (jammed signal segments)
    y = []  # Output labels (original C/A code segments)
    
    for i in range(0, len(jammed_signals) - window_size, window_size // 2):
        X.append(jammed_signals[i:i+window_size])
        y.append(original_ca_code[i:i+window_size])
    
    return np.array(X), np.array(y)

# Create a simplified ML training dataset
X_train, y_train = create_ml_dataset(jammed_signal, ca_code, window_size=50)

# Export ML training data
np.save('jammed_signal_segments.npy', X_train)
np.save('original_ca_code_segments.npy', y_train)

print(f"Dataset created with {len(dataset)} samples")
print(f"ML training dataset created with {len(X_train)} segments")
print(f"Each segment contains {X_train.shape[1]} samples")

# Generate statistics about the dataset
stats = pd.DataFrame({
    'Signal': ['Carrier', 'C/A Code', 'Navigation Data', 'Modulated Signal', 
               'Jamming', 'Noise', 'Jammed Signal', 'Demodulated Signal'],
    'Mean': [carrier.mean(), ca_code.mean(), nav_data.mean(), gps_signal.mean(),
             jamming.mean(), noise.mean(), jammed_signal.mean(), demodulated_signal.mean()],
    'Std': [carrier.std(), ca_code.std(), nav_data.std(), gps_signal.std(),
            jamming.std(), noise.std(), jammed_signal.std(), demodulated_signal.std()],
    'Min': [carrier.min(), ca_code.min(), nav_data.min(), gps_signal.min(),
            jamming.min(), noise.min(), jammed_signal.min(), demodulated_signal.min()],
    'Max': [carrier.max(), ca_code.max(), nav_data.max(), gps_signal.max(),
            jamming.max(), noise.max(), jammed_signal.max(), demodulated_signal.max()]
})

stats.to_csv('signal_statistics.csv', index=False)
