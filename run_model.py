import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from RC import create_gps_reservoir, preprocess_signal, calculate_ber
from gps_signal_dataset2 import load_gps_dataset

def load_model(model_file='rc_specialized_models.pkl'):
    """Load the trained RC model"""
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict_signal(jammed_signal, jammer_type=None, model_data=None):
    """Predict clean signal from jammed signal using specialized models"""
    if model_data is None:
        model_data = load_model()
    
    # Preprocess input
    processed_signal = preprocess_signal(jammed_signal, jammer_type=jammer_type)
    
    # Create reservoir states
    states = create_gps_reservoir(processed_signal, **model_data['reservoir_params'])
    
    # Use specialized model if available, otherwise use general model
    if jammer_type in model_data['jammer_models']:
        pred = model_data['jammer_models'][jammer_type].predict(states)
    else:
        pred = model_data['general_model'].predict(states)
    
    return pred

def main():
    # Load dataset
    print("Loading GPS signal dataset...")
    X_all, y_all, metadata = load_gps_dataset()
    
    # Load trained model
    print("Loading trained RC model...")
    model_data = load_model()
    
    # Select some test samples from each jammer type
    jammer_types = metadata['jammer_type'].unique()
    n_samples_per_jammer = 3
    
    # Split into training (70%), validation (15%), and test (15%)
    n_train = int(0.7 * len(X_all))
    n_val = int(0.15 * len(X_all))
    test_start = n_train + n_val
    
    test_metadata = metadata.iloc[test_start:]
    X_test = X_all[test_start:]
    y_test = y_all[test_start:]
    
    # Create figure for visualization
    fig, axes = plt.subplots(len(jammer_types), 3, figsize=(18, 4*len(jammer_types)))
    
    # Test for each jammer type
    for j, jammer in enumerate(jammer_types):
        # Get indices for this jammer type
        jammer_indices = test_metadata['jammer_type'] == jammer
        jammer_indices = np.where(jammer_indices)[0]
        
        if len(jammer_indices) == 0:
            continue
        
        # Select one sample
        idx = jammer_indices[0]
        
        # Get signals
        jammed_signal = X_test[idx]
        true_signal = y_test[idx]
        
        # Predict clean signal
        predicted_signal = predict_signal(jammed_signal, jammer_type=jammer, model_data=model_data)
        
        # Calculate BER
        ber = calculate_ber(true_signal, predicted_signal)
        
        # Plot results
        axrow = axes[j]
        
        # Plot jammed signal
        axrow[0].plot(jammed_signal)
        axrow[0].set_title(f'Jammed Signal: {jammer}')
        
        # Plot true vs predicted
        axrow[1].plot(true_signal, 'g-', label='True')
        axrow[1].plot(predicted_signal, 'r--', label='Predicted')
        axrow[1].legend()
        axrow[1].set_title(f'BER: {ber:.6f}')
        
        # Plot error
        axrow[2].plot(true_signal - predicted_signal)
        axrow[2].set_title('Error')
    
    plt.tight_layout()
    plt.savefig('rc_jammer_performance.png', dpi=300)
    plt.show()
    
    print("Performance analysis complete!")

if __name__ == "__main__":
    main()