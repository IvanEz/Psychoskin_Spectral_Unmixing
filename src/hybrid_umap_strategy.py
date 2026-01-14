"""
Hybrid embedding strategy that concatenates raw spectra with NMF weights.

The fused spectra for each modality pair are jointly factorised via NMF. The
resulting component activations (weights) are concatenated to the original
features (with independent scaling controls) before UMAP embeds the combined
feature space. This preserves fine-grained spectral structure while injecting
biologically meaningful concentration information.
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


STRATEGY_NAME = "hybrid_umap"


def build_hybrid_features(
    fused_matrix: np.ndarray,
    *,
    W: np.ndarray,
    standardise_raw: bool,
    standardise_weights: bool,
    weight_scale: float,
) -> np.ndarray:
    """
    Concatenate (optionally standardised) raw spectra with scaled NMF weights.
    """
    raw = np.clip(fused_matrix, a_min=0.0, a_max=None)
    raw_part = standardise(raw) if standardise_raw else raw
    weight_part = standardise(W) if standardise_weights else W
    return np.hstack([raw_part, weight_scale * weight_part])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid embedding of raw spectra plus NMF weights."
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
        help="Number of NMF components to extract.",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="Scaling factor applied to the NMF weights before concatenation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Maximum number of NMF iterations.",
    )
    parser.add_argument(
        "--no-standardise-raw",
        action="store_true",
        help="Skip standardisation of the raw fused spectra.",
    )
    parser.add_argument(
        "--no-standardise-weights",
        action="store_true",
        help="Skip standardisation of the NMF weight vectors.",
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

        features = build_hybrid_features(
            dataset.fused_matrix,
            W=W,
            standardise_raw=not args.no_standardise_raw,
            standardise_weights=not args.no_standardise_weights,
            weight_scale=args.weight_scale,
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
            f"{modality_tag}_nc{args.n_components}_ws{args.weight_scale:.1f}"
            f"_n{args.n_neighbors}_d{min_dist_tag}"
        )
        run_dir, plot_path, embedding_path = make_strategy_paths(
            base_dir=args.base_dir,
            strategy_name=STRATEGY_NAME,
            modality_tag=modality_tag,
            file_stub=file_stub,
        )

        title = (
            f"Hybrid UMAP ({modality_tag})\n"
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
            hybrid_features=features,
            nmf_weights=W,
            nmf_components=H,
            reconstruction=reconstruction,
            params={
                "n_components": args.n_components,
                "weight_scale": args.weight_scale,
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
                    f"ws={args.weight_scale}, nc={args.n_components}, "
                    f"std_raw={not args.no_standardise_raw}, std_w={not args.no_standardise_weights}"
                ),
            },
        )

        print(
            f"[{STRATEGY_NAME}] {modality_tag}: SepScore={sep_score:.3f} "
            f"saved to {plot_path}"
        )


if __name__ == "__main__":
    main()
