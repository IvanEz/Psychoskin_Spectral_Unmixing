"""
Shared utilities for multimodal separation strategies.

These helpers load paired modality spectra, prepare fused feature matrices,
run UMAP embeddings, compute simple separation metrics, and manage a
consistent on-disk layout for plots, embeddings, and metrics.
"""

from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from umap import UMAP

import ramanspy as rp

from functions import pre_process_spectral_data


DEFAULT_BASE_DIR = "/mnt/c/Spectral Analysis/spectral-analysis"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_PREPROCESSING = "MinMax"

# Ten modality pairs (all unique pairs from five modalities: f378, f445, REFL1, REFL2, RAM).
MODALITY_PAIRS: Sequence[Tuple[str, str]] = (
    ("f378", "f445"),
    ("f378", "REFL1"),
    ("f378", "REFL2"),
    ("f378", "RAM"),
    ("f445", "REFL1"),
    ("f445", "REFL2"),
    ("f445", "RAM"),
    ("REFL1", "REFL2"),
    ("REFL1", "RAM"),
    ("REFL2", "RAM"),
)

COLOR_MAP = {"Stress": "#d62728", "Control": "#1f77b4"}


@dataclass
class PairDataset:
    """
    Container holding fused features and bookkeeping for a modality pair.
    """

    modalities: Tuple[str, str]
    stress_subjects: List[str]
    control_subjects: List[str]
    stress_matrix: np.ndarray
    control_matrix: np.ndarray
    fused_matrix: np.ndarray
    labels: List[str]
    tags: List[str]
    per_modality_data: Dict[str, Dict[str, np.ndarray]]


def parse_pair_strings(values: Sequence[str] | None) -> List[Tuple[str, str]]:
    """
    Convert CLI pair strings (e.g. "f378+f445") into tuples.
    """
    if not values:
        return [tuple(pair) for pair in MODALITY_PAIRS]

    resolved: List[Tuple[str, str]] = []
    for raw in values:
        # Allow separators such as '+', ',', or '/'
        for sep in ("+", ",", "/"):
            raw = raw.replace(sep, " ")
        tokens = [token.strip() for token in raw.split() if token.strip()]
        if len(tokens) != 2:
            raise ValueError(
                f"Could not parse modality pair '{raw}'. Expected exactly two entries."
            )
        resolved.append((tokens[0], tokens[1]))
    return resolved


def load_modal_spectra(
    data_dir: str, modality: str, sample_type: str
) -> List[Tuple[str, rp.Spectrum]]:
    """
    Read spectra for a modality/sample type, retaining the subject ID.
    """
    suffix = f"{modality}_{sample_type}.txt"
    spectra: List[Tuple[str, rp.Spectrum]] = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(suffix):
            continue

        subject_id = filename.split()[0]
        file_path = os.path.join(data_dir, filename)
        raw = np.loadtxt(file_path)
        wavelengths = raw[:, 0]
        intensities = raw[:, 1]
        spectra.append((subject_id, rp.Spectrum(intensities, wavelengths)))

    return spectra


def preprocess_modalities(
    entries: List[Tuple[str, rp.Spectrum]], strategy: str
) -> Dict[str, rp.Spectrum]:
    """
    Apply the configured preprocessing pipeline and map spectra back to IDs.
    """
    if not entries:
        return {}

    ordered_ids, spectra = zip(*entries)
    processed = pre_process_spectral_data(list(spectra), strategy)
    return {subject_id: spectrum for subject_id, spectrum in zip(ordered_ids, processed)}


def _common_subjects(modality_map: Dict[str, Dict[str, rp.Spectrum]]) -> List[str]:
    """
    Subjects present across all modalities (intersection).
    """
    shared: Iterable[str] | None = None
    for subjects in modality_map.values():
        ids = set(subjects.keys())
        shared = ids if shared is None else shared & ids
    return sorted(shared or [])


def _build_matrix(
    modality_map: Dict[str, Dict[str, rp.Spectrum]],
    subject_ids: Sequence[str],
) -> np.ndarray:
    """
    Concatenate spectra from each modality for every subject.
    """
    rows: List[np.ndarray] = []
    for subject_id in subject_ids:
        feature_parts: List[np.ndarray] = []
        for modality, spectra in modality_map.items():
            spectrum = spectra[subject_id]
            feature_parts.append(spectrum.spectral_data)
        rows.append(np.concatenate(feature_parts))
    return np.vstack(rows) if rows else np.empty((0, 0))


def prepare_pair_dataset(
    modalities: Tuple[str, str],
    *,
    base_dir: str = DEFAULT_BASE_DIR,
    strategy: str = DEFAULT_PREPROCESSING,
) -> PairDataset:
    """
    Load, preprocess, and fuse data for a modality pair.
    """
    stress_dir = os.path.join(base_dir, "data", "STR")
    control_dir = os.path.join(base_dir, "data", "CTR")

    stress_processed: Dict[str, Dict[str, rp.Spectrum]] = {}
    control_processed: Dict[str, Dict[str, rp.Spectrum]] = {}

    for modality in modalities:
        stress_entries = load_modal_spectra(stress_dir, modality, "STR")
        control_entries = load_modal_spectra(control_dir, modality, "CTR")
        stress_processed[modality] = preprocess_modalities(stress_entries, strategy)
        control_processed[modality] = preprocess_modalities(control_entries, strategy)

    stress_subjects = _common_subjects(stress_processed)
    control_subjects = _common_subjects(control_processed)

    if not stress_subjects and not control_subjects:
        raise RuntimeError(f"No overlapping subjects for modalities {modalities}.")

    stress_matrix = _build_matrix(stress_processed, stress_subjects)
    control_matrix = _build_matrix(control_processed, control_subjects)

    if stress_matrix.size == 0 and control_matrix.size > 0:
        stress_matrix = np.zeros((0, control_matrix.shape[1]))
    if control_matrix.size == 0 and stress_matrix.size > 0:
        control_matrix = np.zeros((0, stress_matrix.shape[1]))
    fused_matrix = np.vstack([stress_matrix, control_matrix])

    labels = ["Stress"] * len(stress_subjects) + ["Control"] * len(control_subjects)
    tags = [f"STR-{sid}" for sid in stress_subjects] + [
        f"CTR-{sid}" for sid in control_subjects
    ]

    per_modality_data: Dict[str, Dict[str, np.ndarray]] = {}
    for modality in modalities:
        per_modality_data[modality] = {}
        for subject in stress_subjects:
            per_modality_data[modality][f"STR-{subject}"] = (
                stress_processed[modality][subject].spectral_data
            )
        for subject in control_subjects:
            per_modality_data[modality][f"CTR-{subject}"] = (
                control_processed[modality][subject].spectral_data
            )

    return PairDataset(
        modalities=modalities,
        stress_subjects=stress_subjects,
        control_subjects=control_subjects,
        stress_matrix=stress_matrix,
        control_matrix=control_matrix,
        fused_matrix=fused_matrix,
        labels=labels,
        tags=tags,
        per_modality_data=per_modality_data,
    )


def ensure_directory(path: str) -> str:
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)
    return path


def make_strategy_paths(
    *,
    base_dir: str,
    strategy_name: str,
    modality_tag: str,
    file_stub: str,
) -> Tuple[str, str, str]:
    """
    Resolve run directory, plot path, and embedding path for a strategy output.
    """
    root = ensure_directory(
        os.path.join(base_dir, DEFAULT_RESULTS_DIR, "strategies", strategy_name, modality_tag)
    )
    plot_path = os.path.join(root, f"{strategy_name}_umap_{file_stub}.png")
    embedding_path = os.path.join(root, f"{strategy_name}_embeddings_{file_stub}.npz")
    return root, plot_path, embedding_path


def metrics_csv_path(base_dir: str, strategy_name: str) -> str:
    """
    Location of the metrics CSV for a given strategy.
    """
    directory = ensure_directory(
        os.path.join(base_dir, DEFAULT_RESULTS_DIR, "strategies", strategy_name)
    )
    return os.path.join(directory, "metrics.csv")


def run_umap(
    features: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    """
    Run UMAP with standard configuration.
    """
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def plot_embedding(
    embeddings: np.ndarray,
    labels: Sequence[str],
    tags: Sequence[str],
    *,
    title: str,
    annotate: bool,
    output_path: str,
) -> None:
    """
    Render and save a UMAP embedding plot with optional annotations.
    """
    plt.figure(figsize=(11, 9))
    classes = sorted(set(labels))
    for group in classes:
        idx = [i for i, label in enumerate(labels) if label == group]
        plt.scatter(
            embeddings[idx, 0],
            embeddings[idx, 1],
            label=group,
            s=90,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.6,
            c=COLOR_MAP.get(group, "#7f7f7f"),
        )
    if annotate:
        for (x, y), tag in zip(embeddings, tags):
            plt.text(x + 0.015, y + 0.015, tag, fontsize=9, alpha=0.8)
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_float_tag(value: float, precision: int = 2) -> str:
    """
    Convert a float to a filesystem-friendly tag (e.g. 0.25 -> '0p25').
    """
    return f"{value:.{precision}f}".replace(".", "p")


def compute_centroid_separation(
    embeddings: np.ndarray, labels: Sequence[str]
) -> Dict[str, float]:
    """
    Distance between Stress and Control centroids, normalised by dispersion.
    """
    label_array = np.asarray(labels)
    stress_mask = label_array == "Stress"
    control_mask = label_array == "Control"

    result = {
        "metric_name": "centroid_ratio",
        "metric_value": 0.0,
        "centroid_distance": 0.0,
        "stress_dispersion": 0.0,
        "control_dispersion": 0.0,
    }

    if not stress_mask.any() or not control_mask.any():
        return result

    stress_embeddings = embeddings[stress_mask]
    control_embeddings = embeddings[control_mask]

    stress_mean = stress_embeddings.mean(axis=0)
    control_mean = control_embeddings.mean(axis=0)

    centroid_distance = float(np.linalg.norm(stress_mean - control_mean))
    stress_dispersion = float(
        np.mean(np.linalg.norm(stress_embeddings - stress_mean, axis=1))
    )
    control_dispersion = float(
        np.mean(np.linalg.norm(control_embeddings - control_mean, axis=1))
    )

    separation = centroid_distance / (stress_dispersion + control_dispersion + 1e-9)

    result.update(
        {
            "metric_value": separation,
            "centroid_distance": centroid_distance,
            "stress_dispersion": stress_dispersion,
            "control_dispersion": control_dispersion,
        }
    )
    return result


def append_metrics_record(
    csv_path: str,
    record: Dict[str, object],
) -> None:
    """
    Append a metrics record to the CSV, creating headers if needed.
    """
    fieldnames = [
        "strategy",
        "modalities",
        "preprocessing",
        "n_neighbors",
        "min_dist",
        "metric_name",
        "metric_value",
        "centroid_distance",
        "stress_dispersion",
        "control_dispersion",
        "num_stress",
        "num_control",
        "plot_path",
        "embedding_path",
        "notes",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def standardise(features: np.ndarray) -> np.ndarray:
    """
    Standardise features (zero mean, unit variance) per column.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def fit_nmf(
    fused_matrix: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper around scikit-learn's NMF to keep parameters consistent.
    Returns (W, H, reconstruction).
    """
    raw = np.clip(fused_matrix, a_min=0.0, a_max=None)
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        max_iter=max_iter,
        random_state=random_state,
    )
    W = nmf.fit_transform(raw)
    H = nmf.components_
    reconstruction = W @ H
    return W, H, reconstruction
