"""
Weighted UMAP strategy that rescales fused spectra using NMF activations.

Each spectrum is reconstructed via Non-negative Matrix Factorisation (NMF).
The reconstruction is normalised per sample and used as a multiplicative
weighting mask, emphasising bands that strongly contribute to the dominant
components. The weighted spectra feed directly into UMAP to produce an
embedding with higher contrast between stress and control groups.

Usage (defaults to the six predefined modality pairs):
    python src/weighted_umap_strategy.py

Additional knobs (number of components, weighting strength, etc.) are exposed
through the CLI for rapid experimentation.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
from strategy_utils import (
    DEFAULT_BASE_DIR,
    DEFAULT_PREPROCESSING,
    append_metrics_record,
    compute_centroid_separation,
    format_float_tag,
    fit_nmf,
    make_strategy_paths,
    metrics_csv_path,
    parse_pair_strings,
    plot_embedding,
    prepare_pair_dataset,
    run_umap,
    standardise,
)


STRATEGY_NAME = "weighted_umap"


def compute_weighted_features(
    fused_matrix: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    random_state: int,
    weight_strength: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run NMF on the fused spectra and return weighted features plus raw factors.

    The weighting follows: weighted = raw * (1 + alpha * normalised_reconstruction),
    where the reconstruction is the NMF approximation W @ H scaled to [0, 1] per
    sample. This emphasises spectral regions with strong component contributions.
    """
    raw = np.clip(fused_matrix, a_min=0.0, a_max=None)
    W, H, reconstruction = fit_nmf(
        fused_matrix,
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Normalise each reconstruction row to [0, 1] to create a soft weighting mask.
    row_max = reconstruction.max(axis=1, keepdims=True) + 1e-9
    reconstruction_normalised = reconstruction / row_max

    weighted = raw * (1.0 + weight_strength * reconstruction_normalised)
    return weighted, W, H, reconstruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weighted UMAP using NMF-derived activation masks."
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Project root containing the data, results, and src folders.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Optional modality pairs (format: f378+f445). Defaults to six predefined pairs.",
    )
    parser.add_argument(
        "--preprocessing",
        default=DEFAULT_PREPROCESSING,
        choices=("MinMax", "AUC"),
        help="Preprocessing pipeline defined in functions.pre_process_spectral_data.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=6,
        help="Number of NMF components.",
    )
    parser.add_argument(
        "--weight-strength",
        type=float,
        default=1.0,
        help="Scalar alpha controlling the impact of the NMF weighting mask.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Maximum number of NMF iterations.",
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
        help="UMAP distance metric.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-standardise",
        action="store_true",
        help="Disable StandardScaler on weighted features prior to UMAP.",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable point annotations on the output plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modality_pairs = parse_pair_strings(args.pairs)
    metrics_path = metrics_csv_path(args.base_dir, STRATEGY_NAME)

    for pair in modality_pairs:
        dataset = prepare_pair_dataset(pair, base_dir=args.base_dir, strategy=args.preprocessing)
        modality_tag = "-".join(pair)

        weighted_features, W, H, reconstruction = compute_weighted_features(
            dataset.fused_matrix,
            n_components=args.n_components,
            max_iter=args.max_iter,
            random_state=args.random_state,
            weight_strength=args.weight_strength,
        )

        features = (
            weighted_features
            if args.no_standardise
            else standardise(weighted_features)
        )

        embeddings = run_umap(
            features,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )

        separation = compute_centroid_separation(embeddings, dataset.labels)
        sep_score = separation["metric_value"]

        min_dist_tag = format_float_tag(args.min_dist)
        file_stub = (
            f"{modality_tag}_nc{args.n_components}_ws{args.weight_strength:.1f}"
            f"_n{args.n_neighbors}_d{min_dist_tag}"
        )
        run_dir, plot_path, embedding_path = make_strategy_paths(
            base_dir=args.base_dir,
            strategy_name=STRATEGY_NAME,
            modality_tag=modality_tag,
            file_stub=file_stub,
        )

        title = (
            f"Weighted UMAP ({modality_tag})\n"
            f"Preproc={args.preprocessing}, SepScore={sep_score:.3f}"
        )
        plot_embedding(
            embeddings,
            dataset.labels,
            dataset.tags,
            title=title,
            annotate=not args.no_annotate,
            output_path=plot_path,
        )

        # Persist embeddings and intermediate matrices for quick inspection.
        np.savez_compressed(
            embedding_path,
            embeddings=embeddings,
            labels=np.asarray(dataset.labels),
            tags=np.asarray(dataset.tags),
            modalities=np.asarray(pair),
            weighted_features=weighted_features,
            nmf_weights=W,
            nmf_components=H,
            reconstruction=reconstruction,
            params={
                "n_components": args.n_components,
                "weight_strength": args.weight_strength,
                "min_dist": args.min_dist,
                "n_neighbors": args.n_neighbors,
                "metric": args.metric,
            },
        )

        append_metrics_record(
            metrics_path,
            {
                "strategy": STRATEGY_NAME,
                "modalities": modality_tag,
                "preprocessing": args.preprocessing,
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "metric_name": separation["metric_name"],
                "metric_value": sep_score,
                "centroid_distance": separation["centroid_distance"],
                "stress_dispersion": separation["stress_dispersion"],
                "control_dispersion": separation["control_dispersion"],
                "num_stress": len(dataset.stress_subjects),
                "num_control": len(dataset.control_subjects),
                "plot_path": os.path.abspath(plot_path),
                "embedding_path": os.path.abspath(embedding_path),
                "notes": f"alpha={args.weight_strength}, nc={args.n_components}",
            },
        )

        print(
            f"[{STRATEGY_NAME}] {modality_tag}: SepScore={sep_score:.3f} "
            f"saved to {plot_path}"
        )


if __name__ == "__main__":
    main()
