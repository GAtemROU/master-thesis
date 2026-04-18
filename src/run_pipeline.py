from __future__ import annotations

import argparse
from pathlib import Path

from iv_pipeline.config import load_config
from iv_pipeline.data import load_dataset
from iv_pipeline.evaluate import compute_metrics, evaluate
from iv_pipeline.pipeline import MajorityVotePipeline, VerificationPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the single-sample internal verification pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).parent / "data" / "sample_math.jsonl"),
        help="Path to dataset file (JSON or JSONL, depending on dataset name).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="jsonl",
        choices=["jsonl", "gsm8k", "math500"],
        help="Dataset loader to use.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to pipeline config JSON.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="proposed",
        choices=["proposed", "majority_vote"],
        help="Select pipeline: proposed (constraints+verify) or majority_vote baseline.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples for majority vote baseline.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.pipeline == "majority_vote":
        pipeline = MajorityVotePipeline(config, args.num_samples)
    else:
        pipeline = VerificationPipeline(config)
    examples = load_dataset(args.dataset_name, args.dataset)
    results = evaluate(pipeline, examples)
    metrics = compute_metrics(results, examples)

    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(
        "Verification confusion matrix (TP/FP/FN/TN): "
        f"{metrics.verification_tp}/"
        f"{metrics.verification_fp}/"
        f"{metrics.verification_fn}/"
        f"{metrics.verification_tn}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
