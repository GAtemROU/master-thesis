from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .data import Example
from .pipeline import PipelineResult, VerificationPipeline


@dataclass
class EvaluationMetrics:
    accuracy: float
    verification_tp: int
    verification_fp: int
    verification_fn: int
    verification_tn: int


def evaluate(
    pipeline: VerificationPipeline, examples: Iterable[Example]
) -> List[PipelineResult]:
    return [pipeline.run(example.question) for example in examples]


def compute_metrics(
    results: Iterable[PipelineResult], examples: Iterable[Example]
) -> EvaluationMetrics:
    results_list = list(results)
    examples_list = list(examples)
    correct = 0
    tp = fp = fn = tn = 0
    for result, example in zip(results_list, examples_list):
        is_correct = _normalize(result.final_answer) == _normalize(example.answer)
        if is_correct:
            correct += 1
        verdict_pass = result.verification_verdict == "PASS"
        if is_correct and verdict_pass:
            tp += 1
        elif (not is_correct) and verdict_pass:
            fp += 1
        elif is_correct and (not verdict_pass):
            fn += 1
        else:
            tn += 1
    accuracy = correct / max(1, len(examples_list))
    return EvaluationMetrics(
        accuracy=accuracy,
        verification_tp=tp,
        verification_fp=fp,
        verification_fn=fn,
        verification_tn=tn,
    )


def _normalize(value: str) -> str:
    return value.strip().lower()
