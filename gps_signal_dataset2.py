import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

class GPSSignalGenerator:
    def __init__(self, 
                duration=60,           # Longer dataset (60 seconds)
                fs=1000,               # Higher sampling rate
                satellites=4,          # Multiple satellites
                carrier_freq=1575.42,  # GPS L1 frequency (normalized to 1.57542 for simulation)
                prn_chips=1023,        # Standard C/A code length
                ):
        
        self.duration = duration
        self.fs = fs
        self.satellites = satellites
        self.carrier_freq = carrier_freq / 1000  # Normalize frequency for simulation
        self.samples = int(fs * duration)
        self.t = np.linspace(0, duration, self.samples)
        self.prn_chips = prn_chips
        
        # Store all signal components
        self.signals = {}
        
        # Define standard PRN polynomial taps for satellites
        self.prn_taps = {
            1: [2, 6], 2: [3, 7], 3: [4, 8], 4: [5, 9],
            5: [1, 9], 6: [2, 10], 7: [1, 8], 8: [2, 9]
        }
        
        print(f"Initializing GPS dataset generator: {self.samples} samples, {satellites} satellites")

    def generate_ca_code(self, satellite_id):
        """Generate actual C/A code using Gold codes for a specific satellite"""
        # Create G1 register (10-bit)
        g1 = np.ones(10)
        # Create G2 register (10-bit)
        g2 = np.ones(10)
        
        # Get the specific tap combination for this satellite
        if satellite_id <= len(self.prn_taps):
            taps = self.prn_taps[satellite_id]
        else:
            taps = self.prn_taps[1]  # Default to SV1 taps
        
        # Generate the PRN C/A code (1023 chips)
        ca_code = np.zeros(self.prn_chips)
        for i in range(self.prn_chips):
            # Save output
            ca_code[i] = (g1[9] + g2[9]) % 2
            
            # Shift G1
            g1_new = (g1[2] + g1[9]) % 2
            g1 = np.roll(g1, 1)
            g1[0] = g1_new
            
            # Shift G2
            g2_new = (g2[1] + g2[2] + g2[5] + g2[7] + g2[8] + g2[9]) % 2
            g2 = np.roll(g2, 1)
            g2[0] = g2_new
        
        # Convert from 0/1 to -1/+1
        ca_code = 1 - 2*ca_code
        
        # Calculate samples per chip correctly - ensure it's at least 1
        chip_rate = 1.023  # MHz
        samples_per_chip = max(1, int(self.fs / (chip_rate * 1000)))
        
        # Debug information
        print(f"Chip rate: {chip_rate} MHz, Sampling rate: {self.fs} Hz")
        print(f"Calculated samples per chip: {samples_per_chip}")
        
        # Safety check
        if samples_per_chip == 0:
            samples_per_chip = 1
            print("WARNING: Samples per chip was 0, setting to 1")
        
        # Resample to our sampling rate (each chip gets repeated)
        ca_expanded = np.repeat(ca_code, samples_per_chip)
        
        # Safety check on expanded code length
        if len(ca_expanded) == 0:
            print("WARNING: CA expanded code has zero length, using original code")
            ca_expanded = ca_code
        
        # Ensure the expanded code spans our entire time vector by repeating as needed
        repeats = int(np.ceil(self.samples / len(ca_expanded)))
        ca_full = np.tile(ca_expanded, repeats)
        
        return ca_full[:self.samples]

    def generate_navigation_data(self):
        """Generate navigation data bits (50 Hz for GPS)"""
        # 50 Hz data rate means each bit lasts 20ms
        bit_duration_samples = int(self.fs * 0.02)
        num_bits = int(np.ceil(self.samples / bit_duration_samples))
        
        # Generate random data bits
        bits = np.ones(num_bits)
        for i in range(num_bits):
            if np.random.rand() > 0.5:
                bits[i] = -1
                
        # Expand each bit to the right duration
        nav_data = np.repeat(bits, bit_duration_samples)
        
        return nav_data[:self.samples]
    
    def generate_doppler_shift(self, max_freq_shift=10):
        """Generate Doppler shift effect (varies over time)"""
        # Simulate satellite motion with slow-changing frequency shift
        doppler_freq = max_freq_shift * np.sin(2 * np.pi * 0.05 * self.t)
        doppler_phase = np.cumsum(doppler_freq) / self.fs
        doppler_effect = np.cos(2 * np.pi * doppler_phase)
        
        return doppler_effect
    
    def generate_multipath(self, signal, delays=[0.5, 1.0, 1.5], attenuations=[0.4, 0.2, 0.1]):
        """Add multipath effects to the signal"""
        multipath_signal = np.copy(signal)
        
        for delay, attenuation in zip(delays, attenuations):
            # Convert delay from milliseconds to samples
            delay_samples = int(delay * self.fs / 1000)  # Convert ms to samples
            
            # Safety check: ensure delay is at least 1 sample and less than signal length
            delay_samples = max(1, min(delay_samples, len(signal) - 1))
            
            # Create delayed version of the signal
            delayed_signal = np.zeros_like(signal)
            
            # Only copy if we have a valid delay
            if delay_samples < len(signal):
                # Get the part of the signal to copy
                signal_part = signal[:-delay_samples]
                # Place it at the delayed position
                delayed_signal[delay_samples:] = signal_part
            
            # Add to multipath signal
            multipath_signal += attenuation * delayed_signal
            
        return multipath_signal

    def generate_jammer(self, jammer_type, jammer_power):
        """Generate different types of jammers"""
        jammer = np.zeros(self.samples)
        
        if jammer_type == "continuous":
            # Continuous wave jammer (single tone)
            jammer_freq = 1.1 * self.carrier_freq  # Slightly offset from carrier
            jammer = jammer_power * np.sin(2 * np.pi * jammer_freq * self.t + 0.3)
            
        elif jammer_type == "swept":
            # Swept frequency jammer
            freq_min = 0.9 * self.carrier_freq
            freq_max = 1.1 * self.carrier_freq
            sweep_rate = 0.2  # Hz, full sweep per 5 seconds
            
            freq_t = freq_min + (freq_max - freq_min) * 0.5 * (1 + np.sin(2 * np.pi * sweep_rate * self.t))
            phase_t = 2 * np.pi * np.cumsum(freq_t) / self.fs
            jammer = jammer_power * np.sin(phase_t)
            
        elif jammer_type == "pulsed":
            # Pulsed jammer
            pulse_freq = 2  # Hz (2 pulses per second)
            duty_cycle = 0.3  # 30% on time
            
            pulse_envelope = (np.sin(2 * np.pi * pulse_freq * self.t) > 1 - 2*duty_cycle).astype(float)
            jammer = jammer_power * pulse_envelope * np.sin(2 * np.pi * self.carrier_freq * 1.05 * self.t)
            
        elif jammer_type == "broadband":
            # Broadband noise jammer
            jammer = jammer_power * np.random.randn(self.samples)
            
        elif jammer_type == "smart":
            # Smart jammer that adapts to signal properties
            # Simplified version that follows the signal envelope
            envelope = np.sin(2 * np.pi * 0.1 * self.t) ** 2
            jammer = jammer_power * envelope * np.sin(2 * np.pi * self.carrier_freq * self.t)
            
        return jammer

    def generate_satellite_signal(self, satellite_id, doppler=True, multipath=True):
        """Generate a complete GPS signal for one satellite including realistic effects"""
        # Generate carrier signal
        carrier = np.sin(2 * np.pi * self.carrier_freq * self.t)
        
        # Generate C/A code and navigation data
        ca_code = self.generate_ca_code(satellite_id)
        nav_data = self.generate_navigation_data()
        
        # Basic GPS signal
        gps_signal = carrier * ca_code * nav_data
        
        # Add Doppler effect if requested
        if doppler:
            doppler_effect = self.generate_doppler_shift()
            gps_signal *= doppler_effect
            
        # Add multipath effects if requested
        if multipath:
            gps_signal = self.generate_multipath(gps_signal)
            
        return gps_signal, ca_code, nav_data, carrier

    def generate_dataset(self, jammer_types=["continuous", "swept", "pulsed", "broadband"], 
                        snr_db=30, jnr_db_range=[10, 40]):
        """Generate complete dataset with multiple satellites and jammers"""
        # Convert SNR and JNR to linear scale
        snr = 10**(snr_db/10)
        signal_power = 1.0
        noise_power = signal_power / snr
        
        # Generate individual satellite signals
        satellite_signals = []
        ca_codes = []
        nav_data_all = []
        
        for sat_id in range(1, self.satellites + 1):
            print(f"Generating signal for satellite {sat_id}...")
            sat_signal, ca, nav, carrier = self.generate_satellite_signal(sat_id)
            satellite_signals.append(sat_signal)
            ca_codes.append(ca)
            nav_data_all.append(nav)
            
            # Store components
            self.signals[f'carrier_{sat_id}'] = carrier
            self.signals[f'ca_code_{sat_id}'] = ca
            self.signals[f'nav_data_{sat_id}'] = nav
            self.signals[f'gps_signal_{sat_id}'] = sat_signal
        
        # Combine all satellite signals
        combined_signal = np.zeros(self.samples)
        for signal in satellite_signals:
            combined_signal += signal / self.satellites  # Normalize total power
        
        self.signals['clean_combined'] = combined_signal
        
        # Generate datasets for each jammer type
        datasets = []
        
        for jammer_type in jammer_types:
            # Use random JNR within range for this jammer
            jnr_db = np.random.uniform(jnr_db_range[0], jnr_db_range[1])
            jammer_power = signal_power * 10**(jnr_db/10)
            
            print(f"Generating {jammer_type} jammer with JNR={jnr_db:.2f}dB...")
            
            # Generate jammer
            jammer = self.generate_jammer(jammer_type, jammer_power)
            self.signals[f'jammer_{jammer_type}'] = jammer
            
            # Add noise
            noise = noise_power * np.random.randn(self.samples)
            
            # Create jammed signal
            jammed_signal = combined_signal + jammer + noise
            
            # Store in signals dictionary
            self.signals[f'jammed_{jammer_type}'] = jammed_signal
            
            # Demodulate signal
            demodulated = self.demodulate_signal(jammed_signal, noise_power)
            self.signals[f'demodulated_{jammer_type}'] = demodulated
            
            # Create time-indexed dataframe for this jammer scenario
            df = pd.DataFrame({
                'time': self.t,
                'clean_signal': combined_signal,
                'jammer': jammer,
                'noise': noise,
                'jammed_signal': jammed_signal,
                'demodulated': demodulated
            })
            
            # Add individual satellite signals to dataframe
            for i, signal in enumerate(satellite_signals):
                df[f'satellite_{i+1}'] = signal
                
            # Add metadata
            df.attrs['jammer_type'] = jammer_type
            df.attrs['jnr_db'] = jnr_db
            df.attrs['snr_db'] = snr_db
            
            datasets.append(df)
            
        return datasets
    
    def demodulate_signal(self, jammed_signal, noise_power):
        """More realistic signal demodulation using correlators"""
        # For simplicity, we'll use correlation with the first satellite's C/A code
        if 'ca_code_1' not in self.signals:
            return np.zeros(self.samples)
            
        ca_code = self.signals['ca_code_1']
        
        # Create a sliding window correlator
        corr_window = 20  # ms
        samples_per_window = int(corr_window * self.fs / 1000)
        
        # Initialize output
        demodulated = np.zeros(self.samples)
        
        # Process in chunks to simulate a real receiver
        for i in range(0, self.samples - samples_per_window, samples_per_window // 2):
            # Extract signal chunk
            signal_chunk = jammed_signal[i:i+samples_per_window]
            code_chunk = ca_code[i:i+samples_per_window]
            
            if len(signal_chunk) < samples_per_window:
                continue
                
            # Correlate with carrier replicas at different frequencies
            best_corr = 0
            best_freq = self.carrier_freq
            
            for freq_offset in np.linspace(-50, 50, 11):  # Search +/- 50 Hz
                test_freq = self.carrier_freq + freq_offset/1000
                carrier_replica = np.sin(2 * np.pi * test_freq * self.t[i:i+samples_per_window])
                
                # Mix and correlate
                mixed = signal_chunk * carrier_replica
                correlated = np.sum(mixed * code_chunk)
                
                if abs(correlated) > abs(best_corr):
                    best_corr = correlated
                    best_freq = test_freq
            
            # Fill output with correlation results
            demodulated[i:i+samples_per_window] = best_corr / samples_per_window
            
        return demodulated
    
    def create_ml_dataset(self, datasets, window_size=200, stride=100):
        """Create machine learning dataset with time windows for sequence models"""
        X = []  # Input features (jammed signal segments)
        y = []  # Output labels (clean signal segments)
        metadata = []  # Store metadata about each segment
        
        for dataset_idx, df in enumerate(datasets):
            jammer_type = df.attrs['jammer_type']
            jnr_db = df.attrs['jnr_db']
            
            jammed = df['jammed_signal'].values
            clean = df['clean_signal'].values
            
            # Create sliding windows
            for i in range(0, len(jammed) - window_size, stride):
                X.append(jammed[i:i+window_size])
                y.append(clean[i:i+window_size])
                metadata.append({
                    'start_time': self.t[i],
                    'end_time': self.t[i+window_size],
                    'jammer_type': jammer_type,
                    'jnr_db': jnr_db
                })
                
        return np.array(X), np.array(y), metadata
    
    def save_datasets(self, datasets, base_filename='enhanced_gps_dataset', 
                     save_csv=True, save_ml_ready=True):
        """Save all datasets to files
        
        Parameters:
        -----------
        datasets: list of DataFrames
            List of generated signal datasets
        base_filename: str
            Base name for saved files
        save_csv: bool
            Whether to save individual CSVs for each jammer scenario
        save_ml_ready: bool
            Whether to save ML-ready numpy arrays
        """
        # Save each jammer scenario to a separate CSV file (if requested)
        if save_csv:
            for i, df in enumerate(datasets):
                jammer_type = df.attrs['jammer_type']
                jnr = df.attrs['jnr_db']
                filename = f"{base_filename}_{jammer_type}_jnr{jnr:.0f}db.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")
        
        # Save combined ML dataset - ready for model training
        if save_ml_ready:
            print("Creating ML-ready dataset...")
            X, y, metadata = self.create_ml_dataset(datasets)
            
            # Save as numpy arrays
            np.save(f"{base_filename}_X_jammed.npy", X)
            np.save(f"{base_filename}_y_clean.npy", y)
            
            # Save metadata
            pd.DataFrame(metadata).to_csv(f"{base_filename}_metadata.csv", index=False)
            print(f"Saved ML-ready dataset: {X.shape[0]} windows, each with {X.shape[1]} samples")
            print(f"  - X data: {base_filename}_X_jammed.npy")
            print(f"  - y data: {base_filename}_y_clean.npy") 
            print(f"  - metadata: {base_filename}_metadata.csv")
        
        # Save signal statistics
        stats = []
        for key, signal in self.signals.items():
            if len(signal) > 0:
                stats.append({
                    'signal': key,
                    'mean': np.mean(signal),
                    'std': np.std(signal),
                    'min': np.min(signal),
                    'max': np.max(signal),
                    'power': np.mean(signal**2)
                })
                
        pd.DataFrame(stats).to_csv(f"{base_filename}_statistics.csv", index=False)
        print(f"Saved statistics for {len(stats)} signals")
        
    def plot_signals(self, datasets, base_filename='enhanced_gps_dataset'):
        """Plot key signals for visualization"""
        # Plot satellite signals
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        for i in range(1, self.satellites + 1):
            plt.plot(self.t[:1000], self.signals[f'gps_signal_{i}'][:1000], 
                     label=f'Satellite {i}', alpha=0.7)
        plt.title('Individual Satellite Signals (first 1000 samples)')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(self.t[:1000], self.signals['clean_combined'][:1000], label='Combined GPS Signal')
        plt.title('Clean Combined GPS Signal')
        plt.legend()
        
        # Plot different jammer types
        plt.subplot(3, 1, 3)
        jammer_types = [key.split('_')[1] for key in self.signals.keys() if key.startswith('jammer_')]
        
        for jammer_type in jammer_types:
            plt.plot(self.t[:5000], self.signals[f'jammer_{jammer_type}'][:5000], 
                     label=f'{jammer_type} Jammer', alpha=0.7)
        plt.title('Different Jammer Types (first 5000 samples)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{base_filename}_signals.png", dpi=300)
        
        # Plot spectrograms to show frequency characteristics
        plt.figure(figsize=(15, 10))
        
        # Clean signal spectrogram
        plt.subplot(3, 2, 1)
        f, t, Sxx = signal.spectrogram(self.signals['clean_combined'], fs=self.fs)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title('Clean GPS Signal Spectrogram')
        plt.ylabel('Frequency [Hz]')
        
        # Plot jammed signals spectrograms
        for i, jammer_type in enumerate(jammer_types[:5]):  # Limit to 5 jammers
            plt.subplot(3, 2, i+2)
            f, t, Sxx = signal.spectrogram(self.signals[f'jammed_{jammer_type}'], fs=self.fs)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title(f'{jammer_type.capitalize()} Jammer Spectrogram')
            plt.ylabel('Frequency [Hz]')
            if i >= 3:
                plt.xlabel('Time [sec]')
        
        plt.tight_layout()
        plt.savefig(f"{base_filename}_spectrograms.png", dpi=300)
        
        print("Signal plots saved")

def load_gps_dataset(base_filename='enhanced_gps_dataset'):
    """
    Helper function to load the GPS dataset for ML model training
    
    Parameters:
    -----------
    base_filename: str
        Base filename used when saving the dataset
        
    Returns:
    --------
    X: numpy array
        Input features (jammed signal windows)
    y: numpy array
        Target outputs (clean signal windows)
    metadata: pandas DataFrame
        Metadata about each window
    """
    X = np.load(f"{base_filename}_X_jammed.npy")
    y = np.load(f"{base_filename}_y_clean.npy")
    metadata = pd.read_csv(f"{base_filename}_metadata.csv")
    
    print(f"Loaded dataset with {X.shape[0]} windows")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, metadata

def main():
    # Create GPS signal generator
    generator = GPSSignalGenerator(
        duration=60,       # 60 seconds
        fs=1000,           # 1000 Hz sampling
        satellites=4,      # 4 satellites
        carrier_freq=1575.42,  # GPS L1 frequency
    )
    
    # Generate different datasets
    datasets = generator.generate_dataset(
        jammer_types=["continuous", "swept", "pulsed", "broadband", "smart"],
        snr_db=30,
        jnr_db_range=[5, 30]
    )
    
    # Save datasets - only ML-ready files by default, no CSVs
    generator.save_datasets(datasets, save_csv=True, save_ml_ready=True)
    
    # Plot signals
    generator.plot_signals(datasets)
    
    print("Dataset generation complete!")
    print("\nTo load this dataset for ML training, use:")
    print("X, y, metadata = load_gps_dataset()")

if __name__ == "__main__":
    main()