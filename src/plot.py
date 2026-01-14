import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu
from collections import defaultdict

def plot_weight_comparisons():
    """Plot comparisons between control and stress samples for all prefixes and unknown values."""
    
    # Paths
    results_dir = "/mnt/c/Spectral Analysis/spectral-analysis/results"
    plots_dir = "/mnt/c/Spectral Analysis/spectral-analysis/plots"
    stats_dir = os.path.join(plots_dir, "stats")
    
    # Create output directories if they don't exist
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Substance names - expand this list as needed
    substances = ["hydrocortisone", "IL1-alpha", "IL1-beta", "IL-8", "Sodium-lactate", "TNF-alpha"]
    
    # Find all weight files
    weight_files = [f for f in os.listdir(results_dir) if f.endswith('_weights_u0.txt') or '_weights_u' in f]
    
    # Organize files by prefix, type (CTR/STR), and unknown value
    file_groups = defaultdict(dict)
    for filename in weight_files:
        # Parse filename
        match = re.match(r'([^_]+)_(CTR|STR)_([^_]+)_weights_u(\d+)\.txt', filename)
        if match:
            prefix, sample_type, strategy, n_unknown = match.groups()
            key = (prefix, strategy, n_unknown)
            file_groups[key][sample_type] = os.path.join(results_dir, filename)
    
    # Process each prefix/unknown pair that has both CTR and STR files
    for (prefix, strategy, n_unknown), files in file_groups.items():
        if 'CTR' in files and 'STR' in files:
            print(f"Processing {prefix} with {strategy} strategy and {n_unknown} unknown components")
            
            # Read data
            control_df = pd.read_csv(files['CTR'], delim_whitespace=True, header=None)
            stress_df = pd.read_csv(files['STR'], delim_whitespace=True, header=None)
            
            # Determine number of columns (substances + inferred)
            num_columns = min(control_df.shape[1], stress_df.shape[1])
            
            # Extend substance list if needed for inferred components
            full_substances = substances.copy()
            while len(full_substances) < num_columns:
                full_substances.append(f"Inferred-{len(full_substances) - len(substances) + 1}")
            
            # Collect statistics rows for CSV
            stats_rows = []
            
            # Plot each substance
            for i in range(num_columns):
                plt.figure(figsize=(8, 6))
                
                # Statistical test
                control_vals = control_df[i].dropna().values
                stress_vals = stress_df[i].dropna().values
                
                # Skip if either dataset is empty
                if len(control_vals) == 0 or len(stress_vals) == 0:
                    plt.close()
                    continue
                
                try:
                    stat, p_value = mannwhitneyu(stress_vals, control_vals, alternative='greater')
                    p_text = f"Mann-Whitney p = {p_value:.3e}"
                except ValueError:
                    p_text = "Statistical test failed"
                    stat, p_value = np.nan, np.nan
                
                # Boxplot
                plt.boxplot(
                    [control_vals, stress_vals],
                    labels=['Control', 'Stress']
                )
                
                substance_name = full_substances[i] if i < len(full_substances) else f"Component-{i+1}"
                plt.title(f'{prefix} - {substance_name} Concentration\n{p_text}')
                plt.ylabel('Concentration')
                plt.grid(True)
                
                # Save figure
                output_path = os.path.join(
                    plots_dir, 
                    f'{prefix}_{strategy}_u{n_unknown}_{substance_name.replace(" ", "_")}.png'
                )
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved {output_path}")

                # Append stats for CSV output
                stats_rows.append(
                    {
                        "prefix": prefix,
                        "strategy": strategy,
                        "n_unknown": int(n_unknown),
                        "component_index": i + 1,
                        "component_name": substance_name,
                        "n_control": int(len(control_vals)),
                        "n_stress": int(len(stress_vals)),
                        "mean_control": float(np.mean(control_vals)) if len(control_vals) else np.nan,
                        "mean_stress": float(np.mean(stress_vals)) if len(stress_vals) else np.nan,
                        "median_control": float(np.median(control_vals)) if len(control_vals) else np.nan,
                        "median_stress": float(np.median(stress_vals)) if len(stress_vals) else np.nan,
                        "u_stat": float(stat) if stat == stat else np.nan,
                        "p_value_one_sided_stress_gt_control": float(p_value) if p_value == p_value else np.nan,
                        "median_diff_stress_minus_control": (
                            float(np.median(stress_vals) - np.median(control_vals))
                            if len(control_vals) and len(stress_vals) else np.nan
                        ),
                    }
                )

            # Write per-run statistics CSV
            if stats_rows:
                csv_output_path = os.path.join(
                    stats_dir, f"{prefix}_{strategy}_u{n_unknown}_stats.csv"
                )
                df_stats = pd.DataFrame(stats_rows)
                df_stats.to_csv(csv_output_path, index=False)
                print(f"  Saved stats CSV to {csv_output_path}")

if __name__ == "__main__":
    plot_weight_comparisons()
    print("Plotting complete!")
