#!/usr/bin/env python3
"""
Multimodal UMAP visualisation for spectral data.

This script fuses multiple molecular measurements per subject by concatenating
their preprocessed spectra (and, optionally, simple summary statistics) before
running UMAP. The goal is to surface clearer group separation when several
modalities are available for the same patients.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP

import ramanspy as rp
from functions import pre_process_spectral_data


DEFAULT_MODALITIES: Sequence[str] = ("f378", "f445", "REFL1", "REFL2", "RAM")
FUSION_CHOICES: Sequence[str] = ("concat", "concat_stats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multimodal UMAP plots by fusing several molecular spectra.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples
            --------
            Use all default modalities with MinMax preprocessing:
              python src/umap_visualize_multimodal.py

            Provide a custom list of modalities and include summary statistics:
              python src/umap_visualize_multimodal.py --modalities f378 RAM \\
                  --fusion-mode concat_stats
            """
        ),
    )
    parser.add_argument(
        "--base-dir",
        default="/mnt/c/Spectral Analysis/spectral-analysis",
        help="Project root containing the data and results folders.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help="Modalities (molecular prefixes) to fuse, e.g. f378 f445 RAM.",
    )
    parser.add_argument(
        "--strategy",
        default="MinMax",
        choices=("MinMax", "AUC"),
        help="Preprocessing strategy defined in functions.pre_process_spectral_data.",
    )
    parser.add_argument(
        "--fusion-mode",
        default="concat",
        choices=FUSION_CHOICES,
        help="Feature fusion mode: raw concatenation or concatenation plus simple stats.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=8,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.25,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        help="Distance metric for UMAP.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable per-feature standardisation before UMAP.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable per-point annotations (annotations are on by default).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory (relative to base-dir unless absolute) for saving outputs.",
    )
    parser.add_argument(
        "--save-embeddings",
        help="Optional path (relative to output-dir unless absolute) to store UMAP embeddings as .npz.",
    )
    parser.add_argument(
        "--metrics-csv",
        default=os.path.join("multimodal", "metrics.csv"),
        help="Path (relative to output-dir unless absolute) for logging separation metrics.",
    )
    return parser.parse_args()


def load_modal_spectra(
    data_dir: str, modality: str, sample_type: str
) -> List[Tuple[str, rp.Spectrum]]:
    """
    Load spectra for a given modality/sample type while keeping subject identifiers.

    Parameters
    ----------
    data_dir : str
        Directory containing the spectra (e.g. data/STR or data/CTR).
    modality : str
        Molecular prefix such as 'f378' or 'REFL1'.
    sample_type : str
        'STR' for stress samples or 'CTR' for control samples.

    Returns
    -------
    List[Tuple[str, rp.Spectrum]]
        Each element is (subject_id, Spectrum).
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

    if not spectra:
        print(f"[WARN] No spectra found for modality '{modality}' in {data_dir}.")

    return spectra


def preprocess_modalities(
    entries: List[Tuple[str, rp.Spectrum]], strategy: str
) -> Dict[str, rp.Spectrum]:
    """
    Apply the configured preprocessing pipeline and map spectra back to subject IDs.
    """
    if not entries:
        return {}

    ordered_ids, spectra = zip(*entries)
    processed = pre_process_spectral_data(list(spectra), strategy)

    return {subject_id: spectrum for subject_id, spectrum in zip(ordered_ids, processed)}


def common_subjects(modality_map: Dict[str, Dict[str, rp.Spectrum]]) -> List[str]:
    """
    Compute subject IDs present across every modality (intersection).
    """
    shared: Iterable[str] | None = None

    for modality, subjects in modality_map.items():
        if not subjects:
            print(f"[WARN] Modality '{modality}' is empty; no subjects retained.")
            return []
        subject_ids = set(subjects.keys())
        shared = subject_ids if shared is None else shared & subject_ids

    return sorted(shared or [])


def build_feature_matrix(
    modality_map: Dict[str, Dict[str, rp.Spectrum]],
    subject_ids: Sequence[str],
    fusion_mode: str,
) -> np.ndarray:
    """
    Construct a fused feature matrix for the supplied subjects.
    """
    rows: List[np.ndarray] = []

    for subject_id in subject_ids:
        feature_parts: List[np.ndarray] = []

        for modality, spectra in modality_map.items():
            spectrum = spectra[subject_id]
            data = spectrum.spectral_data
            feature_parts.append(data)

            if fusion_mode == "concat_stats":
                stats = np.array(
                    [
                        data.mean(),
                        data.std(ddof=0),
                        data.max(),
                        np.trapz(data, spectrum.spectral_axis),
                    ],
                    dtype=float,
                )
                feature_parts.append(stats)

        rows.append(np.concatenate(feature_parts))

    return np.vstack(rows)


def ensure_output_path(base_dir: str, output_dir: str) -> str:
    """
    Resolve (and create) the output directory, allowing relative paths.
    """
    if os.path.isabs(output_dir):
        resolved = output_dir
    else:
        resolved = os.path.join(base_dir, output_dir)

    os.makedirs(resolved, exist_ok=True)
    return resolved


def format_float_tag(value: float, precision: int = 2) -> str:
    """
    Turn a float into a filesystem-friendly tag, e.g. 0.25 -> '0p25'.
    """
    return f"{value:.{precision}f}".replace(".", "p")


def compute_separation_metric(embeddings: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    """
    Quantify separation between Stress and Control embeddings using a simple
    centroid distance relative to average within-class dispersion.
    """
    label_array = np.asarray(labels)
    stress_mask = label_array == "Stress"
    control_mask = label_array == "Control"

    result = {
        "name": "centroid_ratio",
        "score": 0.0,
        "distance": 0.0,
        "stress_dispersion": 0.0,
        "control_dispersion": 0.0,
    }

    if not stress_mask.any() or not control_mask.any():
        print("[WARN] Separation metric skipped because one of the cohorts is empty.")
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

    separation_score = centroid_distance / (stress_dispersion + control_dispersion + 1e-9)

    result.update(
        {
            "score": float(separation_score),
            "distance": centroid_distance,
            "stress_dispersion": stress_dispersion,
            "control_dispersion": control_dispersion,
        }
    )
    return result


def append_metrics_record(csv_path: str, record: Dict[str, object]) -> None:
    """
    Append a metrics record to the CSV file, creating headers if needed.
    """
    fieldnames = [
        "modalities",
        "strategy",
        "fusion_mode",
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
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    args = parse_args()

    base_dir = args.base_dir
    stress_dir = os.path.join(base_dir, "data", "STR")
    control_dir = os.path.join(base_dir, "data", "CTR")

    base_output_dir = ensure_output_path(base_dir, args.output_dir)

    modalities = args.modalities
    modality_tag = "-".join(modalities)
    fusion_tag = args.fusion_mode
    annotate_points = not args.no_annotate

    run_dir = os.path.join(base_output_dir, "multimodal", args.strategy, fusion_tag, modality_tag)
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = args.metrics_csv
    if not os.path.isabs(metrics_path):
        metrics_path = os.path.join(base_output_dir, metrics_path)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    print(f"[INFO] Fusing modalities: {', '.join(modalities)}")

    stress_modalities: Dict[str, Dict[str, rp.Spectrum]] = {}
    control_modalities: Dict[str, Dict[str, rp.Spectrum]] = {}

    for modality in modalities:
        stress_entries = load_modal_spectra(stress_dir, modality, "STR")
        control_entries = load_modal_spectra(control_dir, modality, "CTR")

        stress_modalities[modality] = preprocess_modalities(stress_entries, args.strategy)
        control_modalities[modality] = preprocess_modalities(control_entries, args.strategy)

    stress_subjects = common_subjects(stress_modalities)
    control_subjects = common_subjects(control_modalities)

    if not stress_subjects and not control_subjects:
        raise RuntimeError("No subjects available after preprocessing; cannot build UMAP.")

    print(f"[INFO] Retained {len(stress_subjects)} stress subjects, {len(control_subjects)} control subjects.")

    data_blocks: List[np.ndarray] = []
    labels: List[str] = []
    point_tags: List[str] = []

    if stress_subjects:
        stress_matrix = build_feature_matrix(stress_modalities, stress_subjects, args.fusion_mode)
        data_blocks.append(stress_matrix)
        labels.extend(["Stress"] * len(stress_subjects))
        point_tags.extend([f"STR-{sid}" for sid in stress_subjects])

    if control_subjects:
        control_matrix = build_feature_matrix(control_modalities, control_subjects, args.fusion_mode)
        data_blocks.append(control_matrix)
        labels.extend(["Control"] * len(control_subjects))
        point_tags.extend([f"CTR-{sid}" for sid in control_subjects])

    fused_matrix = np.vstack(data_blocks)

    if not args.no_scale:
        scaler = StandardScaler()
        fused_matrix = scaler.fit_transform(fused_matrix)
        print("[INFO] Applied StandardScaler to fused features.")

    umap_model = UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    embeddings = umap_model.fit_transform(fused_matrix)
    print("[INFO] Computed UMAP embeddings.")

    plt.figure(figsize=(12, 10))
    palette = {"Stress": "#d62728", "Control": "#1f77b4"}

    for group in sorted(set(labels)):
        indices = [idx for idx, label in enumerate(labels) if label == group]
        plt.scatter(
            embeddings[indices, 0],
            embeddings[indices, 1],
            c=palette.get(group, "#7f7f7f"),
            label=group,
            s=90,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.6,
        )

    if annotate_points:
        for (x, y), tag in zip(embeddings, point_tags):
            plt.text(x + 0.015, y + 0.015, tag, fontsize=9, alpha=0.8)

    metric_info = compute_separation_metric(embeddings, labels)
    separation_score = metric_info["score"]
    title = (
        f"Multimodal UMAP ({modality_tag})\n"
        f"Strategy={args.strategy}, Fusion={fusion_tag}, SepScore={separation_score:.3f}"
    )

    plt.title(title, fontsize=15)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend(frameon=True)

    min_dist_tag = format_float_tag(args.min_dist, precision=2)
    plot_filename = os.path.join(
        run_dir,
        f"umap_{modality_tag}_n{args.n_neighbors}_d{min_dist_tag}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved UMAP plot to {plot_filename}")

    if args.save_embeddings:
        if os.path.isabs(args.save_embeddings):
            embedding_path = args.save_embeddings
        else:
            embedding_path = os.path.join(run_dir, args.save_embeddings)
    else:
        embedding_path = os.path.join(
            run_dir, f"embeddings_{modality_tag}_n{args.n_neighbors}_d{min_dist_tag}.npz"
        )

    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        labels=np.array(labels),
        tags=np.array(point_tags),
        modalities=np.array(modalities),
    )
    print(f"[INFO] Saved embeddings to {embedding_path}")

    append_metrics_record(
        metrics_path,
        {
            "modalities": modality_tag,
            "strategy": args.strategy,
            "fusion_mode": fusion_tag,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric_name": metric_info["name"],
            "metric_value": metric_info["score"],
            "centroid_distance": metric_info["distance"],
            "stress_dispersion": metric_info["stress_dispersion"],
            "control_dispersion": metric_info["control_dispersion"],
            "num_stress": len(stress_subjects),
            "num_control": len(control_subjects),
            "plot_path": os.path.abspath(plot_filename),
            "embedding_path": os.path.abspath(embedding_path),
        },
    )
    print(f"[INFO] Logged separation metric to {metrics_path}")


if __name__ == "__main__":
    main()
    print("Multimodal UMAP visualisation complete!")
