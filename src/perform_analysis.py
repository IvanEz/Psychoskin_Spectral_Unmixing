import os
import numpy as np
import argparse
import configparser
from functions import *

def perform_analysis(preprocessed_spectra, known_substances, n_unknown, prefix, strategy="MinMax"):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    print("PreProcessed_matrix_dimensions:", preprocessed_spectra[0].spectral_data.shape)
    
    # Prepare wavelength axis from samples and align knowns
    wavelengths = preprocessed_spectra[0].spectral_axis
    # Clip negatives for non-negativity and stack sample matrix
    preprocessed_matrix = np.vstack([np.clip(spectrum.spectral_data, 0, None) for spectrum in preprocessed_spectra])

    # Align known substances to the same wavelength axis and clip negatives
    known_resampled = []
    n_interp = 0
    for substance in known_substances:
        if not np.array_equal(substance.spectral_axis, wavelengths):
            # Resample to the sample wavelength axis
            y = np.interp(wavelengths, substance.spectral_axis, substance.spectral_data)
            n_interp += 1
        else:
            y = substance.spectral_data
        known_resampled.append(np.clip(y, 0, None))
    known_substances_matrix = np.vstack(known_resampled) if known_resampled else np.empty((0, preprocessed_matrix.shape[1]))
    if n_interp > 0:
        print(f"Aligned {n_interp}/{len(known_substances)} known spectra to sample wavelengths via interpolation")
    
    print(f"Number of preprocessed spectra: {len(preprocessed_spectra)}")
    print(f"Number of known substances: {len(known_substances)}")
    
    # Normalize known rows for interpretability (sum to 1)
    if known_substances_matrix.size > 0:
        row_sums = known_substances_matrix.sum(axis=1, keepdims=True) + 1e-12
        H_known = known_substances_matrix / row_sums
    else:
        H_known = known_substances_matrix

    # Constrained NMF with fixed known components
    W, H = constrained_nmf_fixed_known(preprocessed_matrix, H_known, n_unknown)
    
    weights_filename = f"{output_dir}/{prefix}_{strategy}_weights_u{n_unknown}.txt"
    np.savetxt(weights_filename, W)
    print(f"Saved weights to {weights_filename}")
    # Save components as well
    components_filename = f"{output_dir}/{prefix}_{strategy}_H_components_u{n_unknown}.txt"
    np.savetxt(components_filename, H)
    print(f"Saved components to {components_filename}")

    print("\nWeights (contributions of components to each sample):")
    n_known = H_known.shape[0]
    print(W[:, :n_known])
    # Also show fractional contributions (row-normalized) for quick inspection
    W_frac = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    print("\nRow-normalized weights (fractions, first known columns):")
    print(W_frac[:, :n_known])
    # Report relative reconstruction error
    rel_err = np.linalg.norm(preprocessed_matrix - W @ H, ord='fro') / (np.linalg.norm(preprocessed_matrix, ord='fro') + 1e-12)
    print(f"Relative reconstruction error: {rel_err:.6e}")
    
    plot_filename = f"{output_dir}/{prefix}_{strategy}_spectra_u{n_unknown}.png"
    plot_and_save_spectra(
        data_known=known_substances_matrix,
        data_inferred=H,
        output_filename=plot_filename,
        known=True,
        wave_lengths=wavelengths
    )
    print(f"Saved spectral plot to {plot_filename}")
    
    return W, H

def main():
    parser = argparse.ArgumentParser(description="Spectral Analysis Tool")
    
    config = configparser.ConfigParser()
    config_path = 'config/config.ini'
    
    if os.path.exists(config_path):
        config.read(config_path)
        print(f"Loaded configuration from {config_path}")
        
        # Change 'DEFAULT' to 'Parameters' to match your config file
        data_dir = config.get('Parameters', 'data_dir', fallback='data')
        data_dir_known = config.get('Parameters', 'data_dir_known', fallback='data')
        prefix = config.get('Parameters', 'prefix', fallback='sample')
        prefix_known = config.get('Parameters', 'prefix_known', fallback='sample')
        strategy = config.get('Parameters', 'strategy', fallback='MinMax')
        n_unknown = config.getint('Parameters', 'n_unknown', fallback=2)
        
        # Remove quotes if they exist in the config values
        data_dir = data_dir.strip('"\'')
        data_dir_known = data_dir_known.strip('"\'')
        prefix = prefix.strip('"\'')
        strategy = strategy.strip('"\'')
        prefix_known = prefix_known.strip('"\'')
        
        print(f"Using parameters from config: data_dir={data_dir}, prefix={prefix}, "
              f"strategy={strategy}, n_unknown={n_unknown}")
    else:
        print(f"Config file not found at {config_path}, using default values")
        data_dir = 'data'
        data_dir_known = 'data'
        prefix = 'sample'
        prefix_known = 'sample'
        strategy = 'MinMax'
        n_unknown = 2
    
    spectral_objects = load_spectral_data(data_dir, prefix)
    preprocessed_spectra = pre_process_spectral_data(spectral_objects, strategy)
    # Load and preprocess known substances consistently
    known_substances_raw = load_known_substances(data_dir_known, prefix_known)
    known_substances = pre_process_spectral_data(known_substances_raw, strategy)
    
    perform_analysis(
        preprocessed_spectra, 
        known_substances, 
        n_unknown, 
        prefix, 
        strategy
    )

if __name__ == "__main__":
    main()
