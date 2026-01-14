#!/usr/bin/env python3
# filepath: /mnt/c/Spectral Analysis/spectral-analysis/src/umap_visualize.py

import os
import sys
import numpy as np
import configparser
import matplotlib.pyplot as plt
from umap import UMAP
from functions import load_spectral_data, pre_process_spectral_data

def generate_umap_comparison(base_prefix, strategy="MinMax", output_dir="results"):
    """
    Generate UMAP visualizations comparing stress and control samples.
    
    Parameters:
        base_prefix (str): Base prefix for the spectral data files (without _STR/_CTR).
        strategy (str): Preprocessing strategy to use (MinMax or AUC).
        output_dir (str): Directory to save output files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up paths for stress and control data
    base_dir = "/mnt/c/Spectral Analysis/spectral-analysis"
    stress_dir = f"{base_dir}/data/STR"
    control_dir = f"{base_dir}/data/CTR"
    
    stress_prefix = f"{base_prefix}_STR"
    control_prefix = f"{base_prefix}_CTR"
    
    print(f"Loading and preprocessing spectral data for {base_prefix} with {strategy} strategy...")
    
    # Load stress data
    stress_objects = load_spectral_data(stress_dir, stress_prefix)
    if not stress_objects:
        print(f"Warning: No stress data found in {stress_dir} with prefix {stress_prefix}")
    else:
        print(f"Found {len(stress_objects)} stress spectra")
    
    # Load control data
    control_objects = load_spectral_data(control_dir, control_prefix)
    if not control_objects:
        print(f"Warning: No control data found in {control_dir} with prefix {control_prefix}")
    else:
        print(f"Found {len(control_objects)} control spectra")
    
    # Check if we have at least some data
    if not stress_objects and not control_objects:
        print(f"Error: No data found for prefix {base_prefix}")
        return
    
    # Preprocess data
    stress_processed = pre_process_spectral_data(stress_objects, strategy) if stress_objects else []
    control_processed = pre_process_spectral_data(control_objects, strategy) if control_objects else []
    
    # Create data matrix and labels
    data_matrix = []
    labels = []
    
    if stress_processed:
        stress_matrix = np.vstack([spectrum.spectral_data for spectrum in stress_processed])
        data_matrix.append(stress_matrix)
        labels.extend(['Stress'] * len(stress_processed))
    
    if control_processed:
        control_matrix = np.vstack([spectrum.spectral_data for spectrum in control_processed])
        data_matrix.append(control_matrix)
        labels.extend(['Control'] * len(control_processed))
    
    # Combine data matrices
    combined_matrix = np.vstack(data_matrix)
    
    # Configure and run UMAP
    print("Generating UMAP embeddings...")
    umap_model = UMAP(
        n_neighbors=5,
        min_dist=0.3,
        metric='euclidean',
        random_state=42
    )
    umap_embeddings = umap_model.fit_transform(combined_matrix)
    
    # Plot UMAP with group colors
    plt.figure(figsize=(12, 10))
    
    # Define colors and create scatter plot for each group
    colors = {'Stress': 'red', 'Control': 'blue'}
    
    for group in set(labels):
        # Get indices for this group
        indices = [i for i, label in enumerate(labels) if label == group]
        
        # Plot this group
        plt.scatter(
            umap_embeddings[indices, 0], 
            umap_embeddings[indices, 1],
            s=70, 
            alpha=0.7,
            c=colors[group],
            label=group
        )
    
    # Add title and labels
    plt.title(f"UMAP Comparison of {base_prefix} Stress vs Control Spectra\nPreprocessing: {strategy}", fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save with meaningful filename
    output_filename = f"{output_dir}/{base_prefix}_{strategy}_stress_control_comparison_umap.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"UMAP comparison visualization saved to {output_filename}")
    return output_filename

def main():
    """Main function to process command line arguments or use config file."""
    # Set up default paths
    base_dir = "/mnt/c/Spectral Analysis/spectral-analysis"
    config_path = f"{base_dir}/config/config.ini"
    output_dir = f"{base_dir}/results"
    
    # Default values
    base_prefix = "f378"  # Without _STR or _CTR suffix
    strategy = "MinMax"
    
    # Try to load configuration file
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Extract base prefix from the config prefix
        prefix = config.get('Parameters', 'prefix', fallback="sample").strip('"\'')
        base_prefix = prefix.split('_')[0] if '_' in prefix else prefix
        
        strategy = config.get('Parameters', 'strategy', fallback="MinMax").strip('"\'')
        
        print(f"Loaded configuration from {config_path}")
        print(f"Using: base_prefix={base_prefix}, strategy={strategy}")
    else:
        print(f"No config file found at {config_path}, using default values")
    
    # Process command line arguments (override config if provided)
    if len(sys.argv) > 1:
        base_prefix = sys.argv[1]
    if len(sys.argv) > 2:
        strategy = sys.argv[2]
    
    # Generate UMAP visualization
    generate_umap_comparison(base_prefix, strategy, output_dir)

if __name__ == "__main__":
    main()
    print("UMAP comparison visualization complete!")