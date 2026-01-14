"""
Supervised metric learning strategy based on NMF weights.

The fused spectra are first factorised via NMF. The resulting component weights
serve as input to a simple supervised learner (Linear Discriminant Analysis)
that maximises separation between stress and control groups. The transformed
features (optionally concatenated with the original weights) are then embedded
with UMAP.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


STRATEGY_NAME = "supervised_metric_umap"


def learn_supervised_projection(
    W: np.ndarray,
    labels: Sequence[str],
    *,
    n_components: int,
) -> Tuple[np.ndarray, LinearDiscriminantAnalysis]:
    """
    Fit a Linear Discriminant model and return the projected features.
    """
    y = np.array([1 if label == "Stress" else 0 for label in labels])
    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        raise ValueError("Supervised projection requires at least two classes.")

    max_components = min(n_components, unique_labels.size - 1)
    lda = LinearDiscriminantAnalysis(n_components=max_components)
    projected = lda.fit_transform(W, y)
    return projected, lda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised UMAP based on LDA projections of NMF weights."
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
        "--max-iter",
        type=int,
        default=400,
        help="Maximum number of NMF iterations.",
    )
    parser.add_argument(
        "--lda-components",
        type=int,
        default=1,
        help="Number of discriminant components to retain (<= classes-1).",
    )
    parser.add_argument(
        "--append-weights",
        action="store_true",
        help="Concatenate scaled NMF weights alongside the supervised projection.",
    )
    parser.add_argument(
        "--append-scale",
        type=float,
        default=0.5,
        help="Scaling applied to the appended weight vectors (when enabled).",
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

        W, H, reconstruction = fit_nmf(
            dataset.fused_matrix,
            n_components=args.n_components,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        W_standardised = standardise(W)

        supervised_features, lda_model = learn_supervised_projection(
            W_standardised,
            dataset.labels,
            n_components=args.lda_components,
        )

        if args.append_weights:
            features = np.hstack(
                [supervised_features, args.append_scale * W_standardised]
            )
        else:
            features = supervised_features

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
            f"{modality_tag}_nc{args.n_components}_lda{args.lda_components}"
            f"_n{args.n_neighbors}_d{min_dist_tag}"
        )
        run_dir, plot_path, embedding_path = make_strategy_paths(
            base_dir=args.base_dir,
            strategy_name=STRATEGY_NAME,
            modality_tag=modality_tag,
            file_stub=file_stub,
        )

        title = (
            f"Supervised Metric UMAP ({modality_tag})\n"
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

        np.savez_compressed(
            embedding_path,
            embeddings=embeddings,
            labels=np.asarray(dataset.labels),
            tags=np.asarray(dataset.tags),
            modalities=np.asarray(pair),
            lda_features=supervised_features,
            nmf_weights=W,
            nmf_components=H,
            reconstruction=reconstruction,
            params={
                "n_components": args.n_components,
                "lda_components": args.lda_components,
                "append_weights": args.append_weights,
                "append_scale": args.append_scale,
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
                "notes": (
                    f"lda_components={args.lda_components}, append={args.append_weights}, "
                    f"nc={args.n_components}"
                ),
            },
        )

        print(
            f"[{STRATEGY_NAME}] {modality_tag}: SepScore={sep_score:.3f} "
            f"saved to {plot_path}"
        )


if __name__ == "__main__":
    main()
