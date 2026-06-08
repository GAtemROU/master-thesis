from __future__ import annotations

from dataclasses import dataclass
import time
import re
from typing import Dict, List, Optional, Tuple

from .config import PipelineConfig
from .logger import verbose_print
from .models import get_model
from .prompts import load_constraint_prompt, load_prompt_set


@dataclass
class PipelineResult:
    question: str
    solution_text: str
    constraints_text: str
    verification_text: str
    final_answer: str
    verification_verdict: str
    raw_constraints_text: str = ""
    baseline_solution_text: str = ""
    baseline_final_answer: str = ""
    initial_solution_text: str = ""
    initial_verification_text: str = ""
    initial_final_answer: str = ""
    initial_verification_verdict: str = ""


class VerificationPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        if config.max_samples != 1:
            raise ValueError("Single-sample pipeline only; set max_samples=1.")
        self.config = config
        self.prompts = load_prompt_set(config.prompts)
        self.sampler_model = get_model(
            config.sampler_model.name, config.sampler_model.params
        )
        self.constraint_model = get_model(
            config.constraint_model.name, config.constraint_model.params
        )
        verbose_print(
            "Initialized VerificationPipeline with models: "
            f"sampler={config.sampler_model.name} "
            f"constraint={config.constraint_model.name} "
            "interval_check=hard"
        )

    def run(self, question: str) -> PipelineResult:
        step_times = {}
        start_time = time.perf_counter()
        question_preview = question.replace("\n", " ")[:80]
        verbose_print(f"Pipeline start: question={question_preview}")
        prompt_time = time.perf_counter()

        constraint_prompt = self.prompts.constraint_prompt.format(
            question=question,
            problem=question,
        )
        verbose_print("Generate constraints: start")
        raw_constraints_text = self.constraint_model.generate(constraint_prompt)
        constraints_text = _normalize_interval_constraint(raw_constraints_text)
        constraints_time = time.perf_counter()
        verbose_print(
            f"Generate constraints: done elapsed={constraints_time - prompt_time:.3f}s"
        )

        solution_prompt = self.prompts.task_prompt.format(
            question=question,
            problem=question,
        )
        verbose_print("Generate baseline solution: start")
        baseline_solution_text = self.sampler_model.generate(
            solution_prompt,
            stop_after_line_prefixes=["FINAL:"],
            stop_after_prefix_min_chars=10,
        )
        baseline_solution_time = time.perf_counter()
        verbose_print(
            "Generate baseline solution: done "
            f"elapsed={baseline_solution_time - constraints_time:.3f}s"
        )

        constrained_solution_prompt = self.prompts.constrained_task_prompt.format(
            question=question,
            problem=question,
            constraints=constraints_text,
        )
        verbose_print("Generate constrained solution: start")
        solution_text = self.sampler_model.generate(
            constrained_solution_prompt,
            stop_after_line_prefixes=["FINAL:"],
            stop_after_prefix_min_chars=10,
        )
        solution_time = time.perf_counter()
        verbose_print(
            "Generate constrained solution: done "
            f"elapsed={solution_time - baseline_solution_time:.3f}s"
        )
        initial_final_answer = _extract_final_answer(solution_text)
        verbose_print("Interval hard check: start")
        interval_check_start = time.perf_counter()
        initial_verdict, interval_reason = _hard_interval_check(
            initial_final_answer,
            constraints_text,
        )
        interval_check_time = time.perf_counter()
        verification_text = (
            f"VERDICT: {initial_verdict}\nREASON: {interval_reason}"
        )
        verbose_print(
            "Interval hard check: done "
            f"elapsed={interval_check_time - interval_check_start:.3f}s"
        )
        step_times.update(
            {
                "prompt_format_s": prompt_time - start_time,
                "constraints_gen_s": constraints_time - prompt_time,
                "baseline_solution_gen_s": baseline_solution_time - constraints_time,
                "constrained_solution_gen_s": solution_time - baseline_solution_time,
                "interval_check_s": interval_check_time - interval_check_start,
                "total_s": interval_check_time - start_time,
            }
        )
        baseline_final_answer = _extract_final_answer(baseline_solution_text)
        final_answer = initial_final_answer
        final_verdict = initial_verdict
        final_verification_text = verification_text
        verbose_print(
            "Step timings: "
            + " ".join(f"{key}={value:.3f}s" for key, value in step_times.items())
        )
        verbose_print(
            f"Pipeline done: initial_verdict={initial_verdict} final_answer={final_answer}"
        )
        return PipelineResult(
            question=question,
            solution_text=solution_text,
            constraints_text=constraints_text,
            raw_constraints_text=raw_constraints_text,
            verification_text=final_verification_text,
            final_answer=final_answer,
            verification_verdict=final_verdict,
            baseline_solution_text=baseline_solution_text,
            baseline_final_answer=baseline_final_answer,
            initial_solution_text=solution_text,
            initial_verification_text=verification_text,
            initial_final_answer=initial_final_answer,
            initial_verification_verdict=initial_verdict,
        )


class MajorityVotePipeline:
    def __init__(self, config: PipelineConfig, num_samples: int) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        self.config = config
        self.num_samples = num_samples
        self.prompts = load_prompt_set(config.prompts)
        self.sampler_model = get_model(
            config.sampler_model.name, config.sampler_model.params
        )
        verbose_print(
            "Initialized MajorityVotePipeline with model: "
            f"sampler={config.sampler_model.name} num_samples={self.num_samples}"
        )

    def run(self, question: str) -> PipelineResult:
        solution_prompt = self.prompts.task_prompt.format(question=question)
        samples: List[str] = []
        answers: List[str] = []
        for _ in range(self.num_samples):
            solution_text = self.sampler_model.generate(solution_prompt)
            samples.append(solution_text)
            answers.append(_extract_final_answer(solution_text))

        final_answer = _majority_vote(answers)
        return PipelineResult(
            question=question,
            solution_text="\n\n".join(samples),
            constraints_text="",
            raw_constraints_text="",
            verification_text="",
            final_answer=final_answer,
            verification_verdict="UNKNOWN",
            baseline_solution_text="\n\n".join(samples),
            baseline_final_answer=final_answer,
            initial_solution_text="\n\n".join(samples),
            initial_verification_text="",
            initial_final_answer=final_answer,
            initial_verification_verdict="UNKNOWN",
        )


class RangeOnlyPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        if config.max_samples != 1:
            raise ValueError("Single-sample pipeline only; set max_samples=1.")
        self.config = config
        self.constraint_prompt = load_constraint_prompt(config.prompts)
        self.constraint_model = get_model(
            config.constraint_model.name, config.constraint_model.params
        )
        verbose_print(
            "Initialized RangeOnlyPipeline with model: "
            f"constraint={config.constraint_model.name}"
        )

    def run(self, question: str) -> PipelineResult:
        start_time = time.perf_counter()
        question_preview = question.replace("\n", " ")[:80]
        verbose_print(f"Range-only start: question={question_preview}")

        constraint_prompt = self.constraint_prompt.format(
            question=question,
            problem=question,
        )
        raw_constraints_text = self.constraint_model.generate(constraint_prompt)
        constraints_text = _normalize_interval_constraint(raw_constraints_text)
        end_time = time.perf_counter()
        verbose_print(f"Range-only done: elapsed={end_time - start_time:.3f}s")
        return PipelineResult(
            question=question,
            solution_text="",
            constraints_text=constraints_text,
            raw_constraints_text=raw_constraints_text,
            verification_text="",
            final_answer="unknown",
            verification_verdict="UNKNOWN",
            baseline_solution_text="",
            baseline_final_answer="unknown",
            initial_solution_text="",
            initial_verification_text="",
            initial_final_answer="unknown",
            initial_verification_verdict="UNKNOWN",
        )


def _extract_final_answer(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip().startswith("FINAL:"):
            candidate = line.split("FINAL:", 1)[1].strip()
            number = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", candidate)
            if number:
                return number.group(0)
            return candidate
    return "unknown"


def _extract_verdict(text: str) -> str:
    for line in text.splitlines():
        if "VERDICT:" in line:
            return line.split("VERDICT:", 1)[1].strip().upper()
    return "UNKNOWN"


def _normalize_interval_constraint(text: str) -> str:
    interval = _extract_interval(text)
    if interval is None:
        return "INTERVAL: unknown"
    return f"INTERVAL: {interval}"

def _extract_interval(text: str) -> Optional[str]:
    for line in text.splitlines():
        if "INTERVAL:" in line.upper():
            _, rest = line.split(":", 1)
            candidate = rest.strip()
            if candidate:
                bracketed = re.search(r"[\[\(][^\]\)]*[\]\)]", candidate)
                if bracketed:
                    return bracketed.group(0)
                return candidate
            break
    pattern = re.compile(r"[\[\(]\s*[^,\]\)]+\s*,\s*[^,\]\)]+\s*[\]\)]")
    match = pattern.search(text)
    if match:
        return match.group(0)
    return None

def _hard_interval_check(answer_text: str, interval_text: str) -> Tuple[str, str]:
    answer_value = _parse_number(answer_text)
    if answer_value is None:
        return "FAIL", "could not parse numeric final answer"
    bounds = _extract_interval_bounds(interval_text)
    if bounds is None:
        return "FAIL", "could not parse valid interval bounds"
    lower, upper = bounds
    if lower <= answer_value <= upper:
        return "PASS", "final answer is inside interval"
    return "FAIL", "final answer is outside interval"

def _extract_interval_bounds(text: str) -> Optional[Tuple[float, float]]:
    interval = _extract_interval(text)
    if interval is None:
        return None
    match = re.search(r"[\[\(]\s*([^,\]\)]+)\s*,\s*([^,\]\)]+)\s*[\]\)]", interval)
    if match is None:
        return None
    lower = _parse_number(match.group(1))
    upper = _parse_number(match.group(2))
    if lower is None or upper is None:
        return None
    return lower, upper

def _parse_number(text: str) -> Optional[float]:
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



def _majority_vote(answers: List[str]) -> str:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]
