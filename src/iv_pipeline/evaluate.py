from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .data import Example
from .logger import verbose_print
from .pipeline import PipelineResult, VerificationPipeline


@dataclass
class EvaluationMetrics:
    accuracy: float
    final_accuracy: float
    verification_tp: int
    verification_fp: int
    verification_fn: int
    verification_tn: int


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
            f"Example done: {index}/{total} id={example.example_id} initial_answer={result.initial_final_answer} final_answer={result.final_answer} correction_applied={result.correction_applied}"
        )
    return results


def compute_metrics(
    results: Iterable[PipelineResult], examples: Iterable[Example]
) -> EvaluationMetrics:
    results_list = list(results)
    examples_list = list(examples)
    correct = 0
    final_correct = 0
    tp = fp = fn = tn = 0
    for result, example in zip(results_list, examples_list):
        initial_answer = (
            result.initial_final_answer if result.initial_final_answer else result.final_answer
        )
        is_correct = _normalize(initial_answer) == _normalize(example.answer)
        final_is_correct = (
            _normalize(result.final_answer) == _normalize(example.answer)
        )
        if is_correct:
            correct += 1
        if final_is_correct:
            final_correct += 1
        verdict_value = (
            result.initial_verification_verdict
            if result.initial_verification_verdict
            else result.verification_verdict
        )
        verdict_pass = verdict_value == "PASS"
        if is_correct and verdict_pass:
            tp += 1
        elif (not is_correct) and verdict_pass:
            fp += 1
        elif is_correct and (not verdict_pass):
            fn += 1
        else:
            tn += 1
    accuracy = correct / max(1, len(examples_list))
    final_accuracy = final_correct / max(1, len(examples_list))
    return EvaluationMetrics(
        accuracy=accuracy,
        final_accuracy=final_accuracy,
        verification_tp=tp,
        verification_fp=fp,
        verification_fn=fn,
        verification_tn=tn,
    )


def _normalize(value: str) -> str:
    return value.strip().lower()
