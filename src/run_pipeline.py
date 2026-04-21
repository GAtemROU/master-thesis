from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from datetime import datetime, timezone

from iv_pipeline.config import load_config
from iv_pipeline.data import load_dataset
from iv_pipeline.evaluate import compute_metrics, evaluate
from iv_pipeline.pipeline import (
    MajorityVotePipeline,
    VerificationPipeline,
    _extract_final_answer,
    _extract_verdict,
    _majority_vote,
)


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
    parser.add_argument(
        "--trace-run",
        action="store_true",
        help="Save full single-example trace to JSON.",
    )
    parser.add_argument(
        "--trace-index",
        type=int,
        default=0,
        help="Index of example to trace (default: 0).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.pipeline == "majority_vote":
        pipeline = MajorityVotePipeline(config, args.num_samples)
    else:
        pipeline = VerificationPipeline(config)
    examples = load_dataset(args.dataset_name, args.dataset)
    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    if args.trace_run:
        if not (0 <= args.trace_index < len(examples)):
            raise ValueError(
                f"trace_index {args.trace_index} out of range for "
                f"{len(examples)} examples."
            )
        example = examples[args.trace_index]
        if isinstance(pipeline, VerificationPipeline):
            solution_prompt = pipeline.prompts.task_prompt.format(
                question=example.question
            )
            solution_text = pipeline.sampler_model.generate(solution_prompt)
            constraint_prompt = pipeline.prompts.constraint_prompt.format(
                question=example.question
            )
            constraints_text = pipeline.constraint_model.generate(constraint_prompt)
            verify_prompt = pipeline.prompts.verify_prompt.format(
                question=example.question,
                solution=solution_text,
                constraints=constraints_text,
            )
            verification_text = pipeline.verifier_model.generate(verify_prompt)
            final_answer = _extract_final_answer(solution_text)
            verdict = _extract_verdict(verification_text)
            trace_outputs = {
                "solution_text": solution_text,
                "constraints_text": constraints_text,
                "verification_text": verification_text,
                "final_answer": final_answer,
                "verification_verdict": verdict,
            }
            trace_prompts = {
                "task_prompt": solution_prompt,
                "constraint_prompt": constraint_prompt,
                "verify_prompt": verify_prompt,
            }
        else:
            solution_prompt = pipeline.prompts.task_prompt.format(
                question=example.question
            )
            samples = []
            answers = []
            for _ in range(pipeline.num_samples):
                solution_text = pipeline.sampler_model.generate(solution_prompt)
                samples.append(solution_text)
                answers.append(_extract_final_answer(solution_text))
            final_answer = _majority_vote(answers)
            trace_outputs = {
                "samples": samples,
                "answers": answers,
                "final_answer": final_answer,
            }
            trace_prompts = {
                "task_prompt": solution_prompt,
            }
        trace_dir = Path(__file__).parent / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"trace_{run_id}_idx{args.trace_index}.json"
        trace_record = {
            "trace_id": f"{run_id}_idx{args.trace_index}",
            "trace_timestamp": run_timestamp.isoformat(),
            "inputs": {
                "args": {
                    "dataset": args.dataset,
                    "dataset_name": args.dataset_name,
                    "config": args.config,
                    "pipeline": args.pipeline,
                    "num_samples": args.num_samples,
                    "trace_index": args.trace_index,
                },
                "example": {
                    "example_id": example.example_id,
                    "question": example.question,
                    "answer": example.answer,
                },
                "config": asdict(config),
                "prompts": trace_prompts,
            },
            "outputs": trace_outputs,
        }
        trace_path.write_text(json.dumps(trace_record, indent=2, sort_keys=True))
        print(f"Trace saved to: {trace_path}")

    results = evaluate(pipeline, examples)
    metrics = compute_metrics(results, examples)
    output_dir = Path(__file__).parent / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{run_id}.json"
    run_record = {
        "run_id": run_id,
        "run_timestamp": run_timestamp.isoformat(),
        "inputs": {
            "args": {
                "dataset": args.dataset,
                "dataset_name": args.dataset_name,
                "config": args.config,
                "pipeline": args.pipeline,
                "num_samples": args.num_samples,
            },
            "config": asdict(config),
            "prompts": {
                "task_prompt": pipeline.prompts.task_prompt,
                "constraint_prompt": pipeline.prompts.constraint_prompt,
                "verify_prompt": pipeline.prompts.verify_prompt,
            },
        },
        "metrics": {
            "accuracy": metrics.accuracy,
            "verification_tp": metrics.verification_tp,
            "verification_fp": metrics.verification_fp,
            "verification_fn": metrics.verification_fn,
            "verification_tn": metrics.verification_tn,
        },
    }
    output_path.write_text(json.dumps(run_record, indent=2, sort_keys=True))

    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(
        "Verification confusion matrix (TP/FP/FN/TN): "
        f"{metrics.verification_tp}/"
        f"{metrics.verification_fp}/"
        f"{metrics.verification_fn}/"
        f"{metrics.verification_tn}"
    )
    print(f"Run saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
