import os
import numpy as np
import ramanspy as rp
import ramanspy.preprocessing as preprocessing
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from sklearn.decomposition import NMF, non_negative_factorization

def load_spectral_data(data_dir,prefix):
    """
    Load spectral data from text files in the specified directory.
    Each file should contain two columns: wavelengths and measurements.

    Returns:
        tuple: (wavelengths, spectral_objects)
               wavelengths: Numpy array of wavelengths taken from the first loaded file.
               spectral_objects: List of rp.Spectrum objects.
    """
    file_endding=prefix+".txt"
    print(file_endding)
    spectral_objects = []
    print(data_dir)
    wavelengths = None
    for filename in os.listdir(data_dir):
        if filename.endswith(file_endding):
            file_path = os.path.join(data_dir, filename)
            data = np.loadtxt(file_path)
            wavelengths = data[:, 0]
            measurements = data[:, 1]
            raman_object = rp.Spectrum(measurements,wavelengths)
            spectral_objects.append(raman_object)
    return  spectral_objects


def pre_process_spectral_data(spectral_objects,strategy):
    preprocessing_pipeline_MinMax = preprocessing.Pipeline([ 
    preprocessing.despike.WhitakerHayes(),
    preprocessing.denoise.Gaussian(),
    preprocessing.baseline.ASLS(),  
    preprocessing.normalise.MinMax()
        ])
    preprocessing_pipeline_AUC = preprocessing.Pipeline([
   preprocessing.despike.WhitakerHayes(),  # Remove cosmic rays
   preprocessing.denoise.Gaussian(),
   preprocessing.baseline.ASLS(),
   preprocessing.normalise.AUC()
    ])
    
    if strategy == "MinMax":
        preprocessed_spectra = preprocessing_pipeline_MinMax.apply(spectral_objects)
    elif strategy == "AUC":
        preprocessed_spectra = preprocessing_pipeline_AUC.apply(spectral_objects)
        
    return preprocessed_spectra
def load_known_substances(data_dir,prefix):
    """
    Load spectral data from text files in the specified directory.
    Each file should contain two columns: wavelengths and measurements.

    Returns:
        tuple: (wavelengths, spectral_objects)
               wavelengths: Numpy array of wavelengths taken from the first loaded file.
               spectral_objects: List of rp.Spectrum objects.
    """
    
    print(prefix)
    spectral_objects = []
    wavelengths = None
    for filename in os.listdir(data_dir):
        print(filename)
        if filename.startswith(prefix) and filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            data = np.loadtxt(file_path)
            wavelengths = data[:, 0]
            measurements = data[:, 1]
            raman_object = rp.Spectrum(measurements,wavelengths)
            spectral_objects.append(raman_object)
    return  spectral_objects


def plot_and_save_spectra(data_known, data_inferred, output_filename='spectra_plot.png', known=True, wave_lengths=None, show=False):
    """
    Plot known spectra and inferred components.

    Args:
        data_known (np.ndarray): Array of known spectra, shape (n_known, n_features).
        data_inferred (np.ndarray): Component matrix H, shape (n_components, n_features).
        output_filename (str): Path to save the plot.
        known (bool): If True, plot known + unknown; else plot all rows from data_inferred only.
        wave_lengths (np.ndarray): X-axis values.
    """
    x_axis = wave_lengths

    fig = plt.figure(figsize=(10, 6))
    if known:
        n_known = data_known.shape[0]
        # Plot known components
        for i in range(n_known):
            plt.plot(x_axis, data_known[i, :], label=f"Known {i+1}")
        # Plot unknown inferred components (rows after n_known)
        if data_inferred is not None and data_inferred.shape[0] > n_known:
            for i in range(n_known, data_inferred.shape[0]):
                plt.plot(x_axis, data_inferred[i, :], label=f"Unknown {i - n_known + 1}")
    else:
        # Plot all rows of inferred matrix
        if data_inferred is not None:
            for i in range(0, data_inferred.shape[0]):
                plt.plot(x_axis, data_inferred[i, :], label=f"Inferred {i+1}")

    plt.xlabel("Raman shift (cm^-1)", fontsize=12)
    plt.ylabel("Intensity", fontsize=12)
    plt.title("Spectra Plot", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    if show:
        plt.show()
    plt.close(fig)
    
    
    
    
def perform_nmf_rand(n, input_matrix):
    """
    Performs Non-negative Matrix Factorization (NMF) on the input matrix.It initiate everything with random values

    Args:
        n: The number of components (rank) for NMF.
        input_matrix: The input data matrix (e.g., spectra_matrix).

    Returns:
        tuple: A tuple containing the weights and the basis spectra (components).
               - weights: NumPy array of shape (n_samples, n_components).
               - components: NumPy array of shape (n_components, n_features).
    """
    nmf_model = NMF(n_components=n, init='random', random_state=42, max_iter=500)
    weights = nmf_model.fit_transform(input_matrix)
    components = nmf_model.components_
    return weights, components


def perform_nmf_custom_start(input_matrix,known_substances_spectra_matrix):
    n_samples, n_features = input_matrix.shape 
    n_known_components = known_substances_spectra_matrix.shape[0]
    W=np.random.rand(n_samples, n_known_components)
    H=known_substances_spectra_matrix
    nmf_model = NMF(n_components=n_known_components, init='custom', max_iter=500)
    weights = nmf_model.fit_transform(input_matrix, W=W, H=H)
    components = nmf_model.components_
    return weights, components
    
    
def constrained_nmf_sklearn(X,            # (n_samples, n_features)
                            H_known,      # (n_known,   n_features)
                            n_unknown,
                            max_iter=500,
                            tol=1e-4,
                            random_state=42):
    """
    Factor X ≈ W @ H with first rows of H fixed to H_known.

    Args:
        X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
                           Must contain non-negative values.
        H_known (numpy.ndarray): Known components matrix of shape (n_known, n_features).
        raise ValueError(f"X and H_known must have the same number of features. Got X.shape={X.shape} and H_known.shape={H_known.shape}")
        n_unknown (int): Number of unknown components to estimate.
        max_iter (int, optional): Maximum number of iterations for the optimization process. Default is 500.
        tol (float, optional): Tolerance for the stopping condition based on relative reconstruction error. Default is 1e-4.
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing:
            - W (numpy.ndarray): Weight matrix of shape (n_samples, n_components), where each row corresponds to a sample and each column represents the contribution of a component to that sample.
            - H (numpy.ndarray): Component matrix of shape (n_components, n_features), where each row corresponds to a component and each column represents the contribution of that component to a feature.
    """
    n_known, n_features = H_known.shape
    n_samples, n_features_X = X.shape
    if n_features_X != n_features:
        raise ValueError("X and H_known must have the same number of features")
    if np.any(X < 0) or np.any(H_known < 0):
        raise ValueError("NMF requires non‑negative inputs")

    rng = np.random.default_rng(random_state)
    H = np.vstack([H_known,
                   rng.random((n_unknown, n_features))])
    W = rng.random((n_samples, n_known + n_unknown))

    for _ in range(max_iter):
        W, H, _ = non_negative_factorization(
            X, W=W, H=H,
            update_H=True,      # update both W and H
            solver='mu',        # multiplicative updates are easy to re‑enter
            beta_loss='frobenius',
            init='custom',
            random_state=random_state,
            tol=0, max_iter=1)  # one MU step per outer loop

        # Re‑impose the constraint
        H[:n_known] = H_known

        # Check relative reconstruction error
        if np.linalg.norm(X - W @ H, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12) < tol:
            break

    return W, H


def constrained_nmf_fixed_known(X,            # (n_samples, n_features)
                                H_known,      # (n_known,   n_features)
                                n_unknown,
                                max_iter=500,
                                tol=1e-4,
                                random_state=42):
    """
    Constrained NMF with fixed known components: X ≈ W @ H,
    where the first n_known rows of H are fixed to H_known and
    only W and H_unknown are optimized.

    Uses multiplicative updates: W is updated with sklearn's MU with H fixed;
    H_unknown is updated manually by MU; H_known never changes.
    """
    if X.ndim != 2 or H_known.ndim != 2:
        raise ValueError("X and H_known must be 2D arrays")
    n_samples, n_features_X = X.shape
    n_known, n_features_H = H_known.shape
    if n_features_X != n_features_H:
        raise ValueError("X and H_known must have the same number of features")
    if np.any(X < 0) or np.any(H_known < 0):
        raise ValueError("NMF requires non-negative inputs")
    if n_unknown < 0:
        raise ValueError("n_unknown must be >= 0")

    rng = np.random.default_rng(random_state)
    # Initialize H_unknown and W
    if n_unknown > 0:
        H_unknown = rng.random((n_unknown, n_features_X))
    else:
        H_unknown = np.zeros((0, n_features_X))
    H = np.vstack([H_known, H_unknown])
    W = rng.random((n_samples, n_known + n_unknown))

    eps = 1e-12
    for _ in range(max_iter):
        # Update W with H fixed (sklearn MU step)
        W, _, _ = non_negative_factorization(
            X, W=W, H=H,
            update_H=False,
            solver='mu',
            beta_loss='frobenius',
            init='custom',
            random_state=random_state,
            tol=0, max_iter=1)

        # Update only H_unknown (if any) with MU
        if n_unknown > 0:
            Wk = W[:, :n_known]
            Wu = W[:, n_known:]
            Hk = H_known
            Hu = H_unknown

            # Current reconstruction
            Vhat = W @ np.vstack([Hk, Hu])

            # MU update for unknown rows of H
            num = Wu.T @ X
            den = (Wu.T @ Vhat) + eps
            Hu = Hu * (num / den)
            H_unknown = Hu
            H = np.vstack([Hk, H_unknown])
        else:
            H = H_known

        # Stopping criterion on relative error
        rel_err = np.linalg.norm(X - W @ H, 'fro') / (np.linalg.norm(X, 'fro') + eps)
        if rel_err < tol:
            break

    return W, H
