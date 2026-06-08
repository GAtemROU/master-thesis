from __future__ import annotations

from dataclasses import dataclass
import re
import statistics
from typing import Dict, Iterable, List

from .data import Example
from .logger import verbose_print
from .pipeline import PipelineResult, VerificationPipeline


@dataclass
class EvaluationMetrics:
    baseline_accuracy: float | None
    constrained_accuracy: float | None
    interval_unknown_fraction: float
    interval_includes_true_answer_fraction: float
    interval_includes_llm_solution_fraction: float
    interval_width_rel_stats: Dict[str, float | None]
    interval_margin_z_stats: Dict[str, float | None]
    interval_signed_outside_stats: Dict[str, float | None]


def evaluate(
    pipeline: VerificationPipeline, examples: Iterable[Example]
) -> List[PipelineResult]:
    results: List[PipelineResult] = []
    examples_list = list(examples)
    total = len(examples_list)
    for index, example in enumerate(examples_list, start=1):
        question_preview = example.question.replace("\n", " ")[:80]
        verbose_print(
            f"Example start: {index}/{total} id={example.example_id} question={question_preview}"
        )
        result = pipeline.run(example.question)
        results.append(result)
        verbose_print(
            f"Example done: {index}/{total} id={example.example_id} initial_answer={result.initial_final_answer} final_answer={result.final_answer}"
        )
    return results


def compute_metrics(
    results: Iterable[PipelineResult],
    examples: Iterable[Example],
    compute_accuracy: bool = True,
) -> EvaluationMetrics:
    results_list = list(results)
    examples_list = list(examples)
    baseline_correct = 0
    constrained_correct = 0
    unknown_intervals = 0
    intervals_include_true_answer = 0
    intervals_include_llm_solution = 0
    width_rel_values: List[float] = []
    margin_z_values: List[float] = []
    signed_outside_values: List[float] = []
    for result, example in zip(results_list, examples_list):
        if compute_accuracy:
            baseline_answer = (
                result.baseline_final_answer
                if result.baseline_final_answer
                else result.final_answer
            )
            baseline_is_correct = _normalize(baseline_answer) == _normalize(
                example.answer
            )
            constrained_is_correct = _normalize(result.final_answer) == _normalize(
                example.answer
            )
            if baseline_is_correct:
                baseline_correct += 1
            if constrained_is_correct:
                constrained_correct += 1
        interval_bounds = _extract_interval_bounds(result.constraints_text)
        if interval_bounds is None:
            unknown_intervals += 1
            continue
        lower, upper = interval_bounds
        width = upper - lower
        center = (lower + upper) / 2.0
        width_rel_values.append(width / max(1e-9, abs(center)))
        true_answer_value = _parse_number(example.answer)
        if true_answer_value is not None:
            inside = _is_within_interval(true_answer_value, interval_bounds)
            if inside:
                intervals_include_true_answer += 1
                margin_z_values.append(
                    (true_answer_value - center) / max(1e-9, width / 2.0)
                )
                signed_outside_values.append(0.0)
            else:
                if true_answer_value < lower:
                    signed_outside_values.append(true_answer_value - lower)
                else:
                    signed_outside_values.append(true_answer_value - upper)
        llm_answer_value = _parse_number(result.final_answer)
        if (
            llm_answer_value is not None
            and _is_within_interval(llm_answer_value, interval_bounds)
        ):
            intervals_include_llm_solution += 1
    total = max(1, len(examples_list))
    baseline_accuracy = baseline_correct / total if compute_accuracy else None
    constrained_accuracy = constrained_correct / total if compute_accuracy else None
    return EvaluationMetrics(
        baseline_accuracy=baseline_accuracy,
        constrained_accuracy=constrained_accuracy,
        interval_unknown_fraction=unknown_intervals / total,
        interval_includes_true_answer_fraction=intervals_include_true_answer / total,
        interval_includes_llm_solution_fraction=intervals_include_llm_solution / total,
        interval_width_rel_stats=_summary_stats(width_rel_values),
        interval_margin_z_stats=_summary_stats(margin_z_values),
        interval_signed_outside_stats=_summary_stats(signed_outside_values),
    )


def _normalize(value: str) -> str:
    return value.strip().lower()


def _extract_interval_bounds(text: str) -> tuple[float, float] | None:
    match = re.search(
        r"INTERVAL:\s*[\[\(]\s*([^,\]\)]+)\s*,\s*([^,\]\)]+)\s*[\]\)]",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    lower = _parse_number(match.group(1))
    upper = _parse_number(match.group(2))
    if lower is None or upper is None:
        return None
    return lower, upper


def _parse_number(text: str) -> float | None:
    token_match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text)
    if token_match is None:
        return None
    token = token_match.group(0)
    if "/" in token:
        numerator_text, denominator_text = token.split("/", 1)
        denominator = float(denominator_text)
        if denominator == 0:
            return None
        return float(numerator_text) / denominator
    return float(token)


def _is_within_interval(value: float, bounds: tuple[float, float]) -> bool:
    lower, upper = bounds
    return lower <= value <= upper


def _summary_stats(values: Iterable[float]) -> Dict[str, float | None]:
    values_list = list(values)
    if not values_list:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values_list),
        "mean": float(statistics.fmean(values_list)),
        "median": float(statistics.median(values_list)),
        "std": float(statistics.pstdev(values_list)),
        "min": float(min(values_list)),
        "max": float(max(values_list)),
    }
