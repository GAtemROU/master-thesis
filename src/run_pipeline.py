from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from datetime import datetime, timezone

from iv_pipeline.config import load_config
from iv_pipeline.data import load_dataset, _parse_hf_spec
from iv_pipeline.evaluate import compute_metrics, evaluate
from iv_pipeline.logger import RunLogger, set_verbose, verbose_print
from iv_pipeline.pipeline import (
    MajorityVotePipeline,
    RangeOnlyPipeline,
    VerificationPipeline,
    _extract_final_answer,
    _hard_interval_check,
    _majority_vote,
    _normalize_interval_constraint,
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
        choices=["proposed", "majority_vote", "range_only"],
        help=(
            "Select pipeline: proposed (constraints+verify), "
            "majority_vote baseline, or range_only (constraints only)."
        ),
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
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size for evaluation (logs progress per batch).",
    )
    parser.add_argument(
        "--log-traces",
        action="store_true",
        help="Include per-example outputs in the run log.",
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Only run the trace step and skip full evaluation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print and log additional run details.",
    )
    args = parser.parse_args()
    if args.verbose:
        set_verbose(True)
    def _format_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"
    def _parse_split_range(split_value: str) -> tuple[str, int, int | None]:
        if "[" not in split_value or not split_value.endswith("]"):
            return split_value, 0, None
        base, range_part = split_value.split("[", 1)
        range_part = range_part[:-1]
        if ":" in range_part:
            start_text, end_text = range_part.split(":", 1)
            start_value = int(start_text) if start_text else 0
            end_value = int(end_text) if end_text else None
            return base, start_value, end_value
        start_value = int(range_part) if range_part else 0
        return base, start_value, start_value + 1

    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(__file__).parent / "runs"
    run_logger = RunLogger(run_id, run_timestamp, output_dir)
    run_logger.write_start(
        {
            "args": {
                "dataset": args.dataset,
                "dataset_name": args.dataset_name,
                "config": args.config,
                "pipeline": args.pipeline,
                "num_samples": args.num_samples,
                "trace_run": args.trace_run,
                "trace_index": args.trace_index,
                "trace_only": args.trace_only,
                "max_examples": args.max_examples,
                "batch_size": args.batch_size,
                "log_traces": args.log_traces,
                "verbose": args.verbose,
            },
            "config": None,
            "prompts": None,
        }
    )
    print(f"Run log initialized: {run_logger.output_path}")

    try:
        config = load_config(args.config)
        run_logger.write_start(
            {
                "args": {
                    "dataset": args.dataset,
                    "dataset_name": args.dataset_name,
                    "config": args.config,
                    "pipeline": args.pipeline,
                    "num_samples": args.num_samples,
                    "trace_run": args.trace_run,
                    "trace_index": args.trace_index,
                    "trace_only": args.trace_only,
                    "max_examples": args.max_examples,
                    "batch_size": args.batch_size,
                    "log_traces": args.log_traces,
                    "verbose": args.verbose,
                },
                "config": asdict(config),
                "prompts": None,
            }
        )
        if args.verbose:
            verbose_print(
                "Run config loaded: "
                f"pipeline={args.pipeline} "
                f"dataset_name={args.dataset_name} "
                f"dataset={args.dataset}"
            )
            run_logger.write_event(
                "Verbose mode enabled",
                {
                    "pipeline": args.pipeline,
                    "dataset_name": args.dataset_name,
                    "dataset": args.dataset,
                    "models": {
                        "sampler": asdict(config.sampler_model),
                        "constraint": asdict(config.constraint_model),
                        "verifier": asdict(config.verifier_model),
                    },
                    "max_examples": args.max_examples,
                    "num_samples": args.num_samples,
                    "trace_run": args.trace_run,
                    "trace_only": args.trace_only,
                },
            )
        if args.pipeline == "majority_vote":
            pipeline = MajorityVotePipeline(config, args.num_samples)
        elif args.pipeline == "range_only":
            pipeline = RangeOnlyPipeline(config)
        else:
            pipeline = VerificationPipeline(config)
        compute_accuracy = args.pipeline != "range_only"
        examples = None
        if args.trace_run or not (
            args.batch_size
            and args.dataset_name == "gsm8k"
            and str(args.dataset).startswith("hf:")
        ):
            examples = load_dataset(args.dataset_name, args.dataset)
            if args.max_examples is not None:
                if args.max_examples < 1:
                    raise ValueError("--max-examples must be >= 1.")
                examples = list(examples[: args.max_examples])
        if args.batch_size is not None and args.batch_size < 1:
            raise ValueError("--batch-size must be >= 1.")
        if args.verbose:
            if examples is None:
                verbose_print(
                    f"Dataset load: name={args.dataset_name} path={args.dataset} batch_size={args.batch_size}"
                )
            else:
                verbose_print(
                    f"Dataset loaded: name={args.dataset_name} path={args.dataset} count={len(examples)}"
                )
            run_logger.write_event(
                "Dataset loaded",
                {
                    "num_examples": None if examples is None else len(examples),
                    "dataset_name": args.dataset_name,
                    "batch_size": args.batch_size,
                },
            )
    except BaseException as exc:
        run_logger.write_error(exc)
        raise
    if args.trace_run:
        print(f"Trace run enabled (index {args.trace_index}).")
        if examples is None:
            raise ValueError("--trace-run requires full dataset load.")
        if not (0 <= args.trace_index < len(examples)):
            raise ValueError(
                f"trace_index {args.trace_index} out of range for "
                f"{len(examples)} examples."
            )
        example = examples[args.trace_index]
        if isinstance(pipeline, VerificationPipeline):
            constraint_prompt = pipeline.prompts.constraint_prompt.format(
                question=example.question,
                problem=example.question,
            )
            raw_constraints_text = pipeline.constraint_model.generate(constraint_prompt)
            constraints_text = _normalize_interval_constraint(raw_constraints_text)
            solution_prompt = pipeline.prompts.task_prompt.format(
                question=example.question,
                problem=example.question,
            )
            baseline_solution_text = pipeline.sampler_model.generate(
                solution_prompt,
                stop_after_line_prefixes=["FINAL:"],
                stop_after_prefix_min_chars=10,
            )
            baseline_final_answer = _extract_final_answer(baseline_solution_text)
            constrained_solution_prompt = pipeline.prompts.constrained_task_prompt.format(
                question=example.question,
                problem=example.question,
                constraints=constraints_text,
            )
            solution_text = pipeline.sampler_model.generate(
                constrained_solution_prompt,
                stop_after_line_prefixes=["FINAL:"],
                stop_after_prefix_min_chars=10,
            )
            final_answer = _extract_final_answer(solution_text)
            verdict, interval_reason = _hard_interval_check(
                final_answer,
                constraints_text,
            )
            verification_text = f"VERDICT: {verdict}\nREASON: {interval_reason}"
            trace_outputs = {
                "raw_constraints_text": raw_constraints_text,
                "baseline_solution_text": baseline_solution_text,
                "baseline_final_answer": baseline_final_answer,
                "solution_text": solution_text,
                "constraints_text": constraints_text,
                "verification_text": verification_text,
                "final_answer": final_answer,
                "verification_verdict": verdict,
            }
            trace_prompts = {
                "task_prompt": solution_prompt,
                "constrained_task_prompt": constrained_solution_prompt,
                "constraint_prompt": constraint_prompt,
            }
        elif isinstance(pipeline, RangeOnlyPipeline):
            constraint_prompt = pipeline.constraint_prompt.format(
                question=example.question,
                problem=example.question,
            )
            raw_constraints_text = pipeline.constraint_model.generate(constraint_prompt)
            constraints_text = _normalize_interval_constraint(raw_constraints_text)
            trace_outputs = {
                "raw_constraints_text": raw_constraints_text,
                "constraints_text": constraints_text,
            }
            trace_prompts = {
                "constraint_prompt": constraint_prompt,
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
        if args.trace_only:
            run_logger.write_results(
                {
                    "baseline_accuracy": 0.0,
                    "constrained_accuracy": 0.0,
                    "interval_unknown_fraction": 0.0,
                    "interval_includes_true_answer_fraction": 0.0,
                    "interval_includes_llm_solution_fraction": 0.0,
                },
                details={"trace_only": True, "trace_path": str(trace_path)},
            )
            print(f"Run saved to: {run_logger.output_path}")
            return 0

    run_logger.write_start(
        {
            "args": {
                "dataset": args.dataset,
                "dataset_name": args.dataset_name,
                "config": args.config,
                "pipeline": args.pipeline,
                "num_samples": args.num_samples,
                "trace_run": args.trace_run,
                "trace_index": args.trace_index,
                "trace_only": args.trace_only,
                "max_examples": args.max_examples,
                "batch_size": args.batch_size,
                "log_traces": args.log_traces,
                "verbose": args.verbose,
            },
            "config": asdict(config),
            "prompts": (
                {
                    "constraint_prompt": pipeline.constraint_prompt,
                }
                if isinstance(pipeline, RangeOnlyPipeline)
                else {
                    "task_prompt": pipeline.prompts.task_prompt,
                    "constrained_task_prompt": pipeline.prompts.constrained_task_prompt,
                    "constraint_prompt": pipeline.prompts.constraint_prompt,
                }
            ),
            "dataset": {
                "num_examples": None if examples is None else len(examples),
            },
        }
    )

    try:
        results = []
        result_examples = []
        if (
            args.batch_size
            and examples is None
            and args.dataset_name == "gsm8k"
            and str(args.dataset).startswith("hf:")
        ):
            dataset_id, split = _parse_hf_spec(str(args.dataset))
            base_split, split_start, split_end = _parse_split_range(split)
            max_examples = args.max_examples
            offset = split_start
            collected = 0
            while True:
                if max_examples is not None and collected >= max_examples:
                    break
                batch_end = offset + args.batch_size
                if split_end is not None:
                    batch_end = min(batch_end, split_end)
                if max_examples is not None:
                    batch_end = min(batch_end, split_start + max_examples)
                if batch_end <= offset:
                    break
                batch_split = f"{base_split}[{offset}:{batch_end}]"
                batch_path = f"hf:{dataset_id}?split={batch_split}"
                batch_examples = load_dataset(args.dataset_name, batch_path)
                if not batch_examples:
                    break
                run_logger.write_event(
                    "Batch loaded",
                    {
                        "batch_split": batch_split,
                        "batch_start": offset,
                        "batch_end": offset + len(batch_examples) - 1,
                        "batch_size": len(batch_examples),
                    },
                )
                if args.verbose:
                    verbose_print(
                        f"Batch loaded: {batch_split} size={len(batch_examples)}"
                    )
                    verbose_print(f"Batch evaluate: start {batch_split}")
                batch_results = evaluate(pipeline, batch_examples)
                results.extend(batch_results)
                result_examples.extend(batch_examples)
                batch_metrics = compute_metrics(
                    results, result_examples, compute_accuracy=compute_accuracy
                )
                run_logger.write_event(
                    "Batch completed",
                    {
                        "batch_split": batch_split,
                        "batch_start": offset,
                        "batch_end": offset + len(batch_examples) - 1,
                        "batch_size": len(batch_examples),
                        "baseline_accuracy": batch_metrics.baseline_accuracy,
                        "constrained_accuracy": batch_metrics.constrained_accuracy,
                        "interval_unknown_fraction": batch_metrics.interval_unknown_fraction,
                        "interval_includes_true_answer_fraction": batch_metrics.interval_includes_true_answer_fraction,
                        "interval_includes_llm_solution_fraction": batch_metrics.interval_includes_llm_solution_fraction,
                        "interval_width_rel_stats": batch_metrics.interval_width_rel_stats,
                        "interval_margin_z_stats": batch_metrics.interval_margin_z_stats,
                        "interval_signed_outside_stats": batch_metrics.interval_signed_outside_stats,
                    },
                )
                if args.verbose:
                    verbose_print(
                        f"Batch completed: {batch_split} size={len(batch_examples)}"
                    )
                collected += len(batch_examples)
                offset = batch_end
        elif args.batch_size and examples is not None:
            total = len(examples)
            for start in range(0, total, args.batch_size):
                batch_examples = examples[start : start + args.batch_size]
                if args.verbose:
                    verbose_print(
                        f"Batch evaluate: start {start}-{start + len(batch_examples) - 1} of {total}"
                    )
                batch_results = evaluate(pipeline, batch_examples)
                results.extend(batch_results)
                result_examples = examples[: start + len(batch_examples)]
                batch_metrics = compute_metrics(
                    results, result_examples, compute_accuracy=compute_accuracy
                )
                run_logger.write_event(
                    "Batch completed",
                    {
                        "batch_start": start,
                        "batch_end": start + len(batch_examples) - 1,
                        "batch_size": len(batch_examples),
                        "total_examples": total,
                        "baseline_accuracy": batch_metrics.baseline_accuracy,
                        "constrained_accuracy": batch_metrics.constrained_accuracy,
                        "interval_unknown_fraction": batch_metrics.interval_unknown_fraction,
                        "interval_includes_true_answer_fraction": batch_metrics.interval_includes_true_answer_fraction,
                        "interval_includes_llm_solution_fraction": batch_metrics.interval_includes_llm_solution_fraction,
                        "interval_width_rel_stats": batch_metrics.interval_width_rel_stats,
                        "interval_margin_z_stats": batch_metrics.interval_margin_z_stats,
                        "interval_signed_outside_stats": batch_metrics.interval_signed_outside_stats,
                    },
                )
                if args.verbose:
                    verbose_print(
                        f"Batch completed: {start}-{start + len(batch_examples) - 1} of {total}"
                    )
            result_examples = examples
        else:
            results = evaluate(pipeline, examples)
            result_examples = examples
        metrics = compute_metrics(
            results, result_examples, compute_accuracy=compute_accuracy
        )
    except BaseException as exc:
        run_logger.write_error(exc)
        raise

    metric_payload = {
        "baseline_accuracy": metrics.baseline_accuracy,
        "constrained_accuracy": metrics.constrained_accuracy,
        "interval_unknown_fraction": metrics.interval_unknown_fraction,
        "interval_includes_true_answer_fraction": metrics.interval_includes_true_answer_fraction,
        "interval_includes_llm_solution_fraction": metrics.interval_includes_llm_solution_fraction,
        "interval_width_rel_stats": metrics.interval_width_rel_stats,
        "interval_margin_z_stats": metrics.interval_margin_z_stats,
        "interval_signed_outside_stats": metrics.interval_signed_outside_stats,
    }
    details = None
    if args.log_traces:
        details = {
            "results": [
                {
                    "question": result.question,
                    "initial_solution_text": result.initial_solution_text,
                    "initial_verification_text": result.initial_verification_text,
                    "initial_final_answer": result.initial_final_answer,
                    "initial_verification_verdict": result.initial_verification_verdict,
                    "solution_text": result.solution_text,
                    "raw_constraints_text": result.raw_constraints_text,
                    "baseline_solution_text": result.baseline_solution_text,
                    "baseline_final_answer": result.baseline_final_answer,
                    "constraints_text": result.constraints_text,
                    "verification_text": result.verification_text,
                    "final_answer": result.final_answer,
                    "verification_verdict": result.verification_verdict,
                }
                for result in results
            ]
        }
    run_logger.write_results(metric_payload, details=details)

    print(f"Baseline accuracy: {_format_metric(metrics.baseline_accuracy)}")
    print(f"Constrained accuracy: {_format_metric(metrics.constrained_accuracy)}")
    print(f"Intervals unknown fraction: {metrics.interval_unknown_fraction:.3f}")
    print(
        "Intervals including true answer fraction: "
        f"{metrics.interval_includes_true_answer_fraction:.3f}"
    )
    print(
        "Intervals including LLM solution fraction: "
        f"{metrics.interval_includes_llm_solution_fraction:.3f}"
    )
    print(f"Interval relative width stats: {metrics.interval_width_rel_stats}")
    print(f"Interval margin z stats (inside): {metrics.interval_margin_z_stats}")
    print(f"Interval signed outside stats: {metrics.interval_signed_outside_stats}")
    print(f"Run saved to: {run_logger.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
