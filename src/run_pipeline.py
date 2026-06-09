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
    IntervalPromptSolvePipeline,
    MajorityVotePipeline,
    RangeOnlyPipeline,
    SolveOnlyPipeline,
    VerificationPipeline,
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
        choices=["proposed", "majority_vote", "range_only", "solve_only", "interval_solve"],
        help=(
            "Select pipeline: proposed (constraints+verify), "
            "majority_vote baseline, range_only (constraints only), "
            "solve_only (task prompt only), or interval_solve (task+interval)."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples for majority vote baseline.",
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
        "--trace",
        action="store_true",
        help="Write per-example trace file alongside the run.",
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
                "trace": args.trace,
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
                "trace": args.trace,
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
                    "trace": args.trace,
                },
            )
        if args.pipeline == "majority_vote":
            pipeline = MajorityVotePipeline(config, args.num_samples)
        elif args.pipeline == "range_only":
            pipeline = RangeOnlyPipeline(config)
        elif args.pipeline == "solve_only":
            pipeline = SolveOnlyPipeline(config)
        elif args.pipeline == "interval_solve":
            pipeline = IntervalPromptSolvePipeline(config)
        else:
            pipeline = VerificationPipeline(config)
        compute_accuracy = args.pipeline != "range_only"
        include_true_answer = args.pipeline != "interval_solve"
        interval_answer_source = "model" if args.pipeline == "interval_solve" else "true"
        examples = None
        if args.trace or not (
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


    run_logger.write_start(
        {
            "args": {
                "dataset": args.dataset,
                "dataset_name": args.dataset_name,
                "config": args.config,
                "pipeline": args.pipeline,
                "num_samples": args.num_samples,
                "trace": args.trace,
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
                }
                if isinstance(pipeline, SolveOnlyPipeline)
                else {
                    "constrained_task_prompt": pipeline.prompts.constrained_task_prompt,
                }
                if isinstance(pipeline, IntervalPromptSolvePipeline)
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
                if isinstance(pipeline, IntervalPromptSolvePipeline):
                    batch_results = [
                        pipeline.run(example.question, example.interval)
                        for example in batch_examples
                    ]
                else:
                    batch_results = evaluate(pipeline, batch_examples)
                results.extend(batch_results)
                result_examples.extend(batch_examples)
                batch_metrics = compute_metrics(
                    results,
                    result_examples,
                    compute_accuracy=compute_accuracy,
                    include_true_answer=include_true_answer,
                    interval_answer_source=interval_answer_source,
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
                if isinstance(pipeline, IntervalPromptSolvePipeline):
                    batch_results = [
                        pipeline.run(example.question, example.interval)
                        for example in batch_examples
                    ]
                else:
                    batch_results = evaluate(pipeline, batch_examples)
                results.extend(batch_results)
                result_examples = examples[: start + len(batch_examples)]
                batch_metrics = compute_metrics(
                    results,
                    result_examples,
                    compute_accuracy=compute_accuracy,
                    include_true_answer=include_true_answer,
                    interval_answer_source=interval_answer_source,
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
                    },
                )
                if args.verbose:
                    verbose_print(
                        f"Batch completed: {start}-{start + len(batch_examples) - 1} of {total}"
                    )
            result_examples = examples
        else:
            if isinstance(pipeline, IntervalPromptSolvePipeline):
                results = [
                    pipeline.run(example.question, example.interval)
                    for example in examples
                ]
            else:
                results = evaluate(pipeline, examples)
            result_examples = examples
        metrics = compute_metrics(
            results,
            result_examples,
            compute_accuracy=compute_accuracy,
            include_true_answer=include_true_answer,
            interval_answer_source=interval_answer_source,
        )
    except BaseException as exc:
        run_logger.write_error(exc)
        raise
    trace_path = None
    if args.trace:
        if examples is None:
            raise ValueError("--trace requires full dataset load.")
        trace_entries = []
        for result, example in zip(results, result_examples):
            prompts = {}
            if result.constraint_prompt:
                prompts["constraint_prompt"] = result.constraint_prompt
            if result.baseline_solution_prompt:
                prompts["baseline_solution_prompt"] = result.baseline_solution_prompt
            if result.constrained_solution_prompt:
                prompts["constrained_solution_prompt"] = (
                    result.constrained_solution_prompt
                )
            raw_output = (
                result.raw_constraints_text
                or result.baseline_solution_text
                or result.solution_text
            )
            outputs = {"raw_output": raw_output} if raw_output else {}
            outputs["parsed_answer"] = (
                result.final_answer
                if result.final_answer and result.final_answer != "unknown"
                else None
            )
            trace_entries.append(
                {
                    "inputs": {
                        "example_id": example.example_id,
                        "question": example.question,
                        "answer": example.answer,
                        "interval": list(example.interval) if example.interval else None,
                    },
                    "prompts": prompts,
                    "outputs": outputs,
                }
            )
        trace_dir = Path(__file__).parent / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"trace_{run_id}.json"
        trace_record = {
            "trace_id": run_id,
            "trace_timestamp": run_timestamp.isoformat(),
            "inputs": {
                "args": {
                    "dataset": args.dataset,
                    "dataset_name": args.dataset_name,
                    "config": args.config,
                    "pipeline": args.pipeline,
                    "num_samples": args.num_samples,
                    "max_examples": args.max_examples,
                    "batch_size": args.batch_size,
                    "log_traces": args.log_traces,
                    "trace": args.trace,
                },
                "config": asdict(config),
            },
            "entries": trace_entries,
        }
        trace_path.write_text(json.dumps(trace_record, indent=2, sort_keys=True))
        print(f"Trace saved to: {trace_path}")

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
                    "raw_output": (
                        result.raw_constraints_text
                        or result.baseline_solution_text
                        or result.solution_text
                    ),
                    "parsed_answer": (
                        result.final_answer
                        if result.final_answer and result.final_answer != "unknown"
                        else None
                    ),
                }
                for result in results
            ]
        }
    if trace_path is not None:
        if details is None:
            details = {}
        details["trace_path"] = str(trace_path)
    run_logger.write_results(metric_payload, details=details)

    print(f"Baseline accuracy: {_format_metric(metrics.baseline_accuracy)}")
    print(f"Constrained accuracy: {_format_metric(metrics.constrained_accuracy)}")
    print(f"Intervals unknown fraction: {metrics.interval_unknown_fraction:.3f}")
    print(
        "Intervals including true answer fraction: "
        f"{_format_metric(metrics.interval_includes_true_answer_fraction)}"
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
