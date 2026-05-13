from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List

from .config import PipelineConfig
from .logger import verbose_print
from .models import get_model
from .prompts import load_prompt_set


@dataclass
class PipelineResult:
    question: str
    solution_text: str
    constraints_text: str
    verification_text: str
    final_answer: str
    verification_verdict: str
    initial_solution_text: str = ""
    initial_verification_text: str = ""
    initial_final_answer: str = ""
    initial_verification_verdict: str = ""
    correction_applied: bool = False
    correction_reason: str = ""
    corrected_solution_text: str = ""
    corrected_verification_text: str = ""
    corrected_final_answer: str = ""
    corrected_verification_verdict: str = ""


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
        self.verifier_model = get_model(
            config.verifier_model.name, config.verifier_model.params
        )
        verbose_print(
            "Initialized VerificationPipeline with models: "
            f"sampler={config.sampler_model.name} "
            f"constraint={config.constraint_model.name} "
            f"verifier={config.verifier_model.name}"
        )

    def run(self, question: str) -> PipelineResult:
        step_times = {}
        start_time = time.perf_counter()
        question_preview = question.replace("\n", " ")[:80]
        verbose_print(f"Pipeline start: question={question_preview}")
        solution_prompt = self.prompts.task_prompt.format(question=question)
        prompt_time = time.perf_counter()
        verbose_print("Generate solution: start")
        solution_text = self.sampler_model.generate(solution_prompt)
        solution_time = time.perf_counter()
        verbose_print(
            f"Generate solution: done elapsed={solution_time - prompt_time:.3f}s"
        )

        constraint_prompt = self.prompts.constraint_prompt.format(question=question)
        verbose_print("Generate constraints: start")
        constraints_text = self.constraint_model.generate(constraint_prompt)
        constraints_time = time.perf_counter()
        verbose_print(
            f"Generate constraints: done elapsed={constraints_time - solution_time:.3f}s"
        )

        verify_prompt = self.prompts.verify_prompt.format(
            question=question,
            solution=solution_text,
            constraints=constraints_text,
        )
        verbose_print("Verify solution: start")
        verification_text = self.verifier_model.generate(verify_prompt)
        verify_time = time.perf_counter()
        verbose_print(
            f"Verify solution: done elapsed={verify_time - constraints_time:.3f}s"
        )
        step_times.update(
            {
                "prompt_format_s": prompt_time - start_time,
                "solution_gen_s": solution_time - prompt_time,
                "constraints_gen_s": constraints_time - solution_time,
                "verify_gen_s": verify_time - constraints_time,
                "total_s": verify_time - start_time,
            }
        )

        initial_final_answer = _extract_final_answer(solution_text)
        initial_verdict = _extract_verdict(verification_text)
        reason = _extract_reason(verification_text)
        correction_applied = False
        corrected_solution_text = ""
        corrected_final_answer = ""
        final_answer = initial_final_answer
        final_verdict = initial_verdict
        final_verification_text = verification_text
        if initial_verdict != "PASS":
            correction_applied = True
            correction_prompt = self.prompts.correction_prompt.format(
                question=question,
                solution=solution_text,
                reason=reason,
            )
            correction_prompt_time = time.perf_counter()
            verbose_print(
                f"Correction step: start reason={reason if reason else 'unknown'}"
            )
            corrected_solution_text = self.sampler_model.generate(correction_prompt)
            correction_solution_time = time.perf_counter()
            verbose_print(
                f"Correction step: done elapsed={correction_solution_time - correction_prompt_time:.3f}s"
            )
            corrected_final_answer = _extract_final_answer(corrected_solution_text)
            final_answer = corrected_final_answer
            step_times.update(
                {
                    "correction_prompt_format_s": correction_prompt_time - verify_time,
                    "correction_solution_gen_s": correction_solution_time
                    - correction_prompt_time,
                    "total_s": correction_solution_time - start_time,
                }
            )
        verbose_print(
            "Step timings: "
            + " ".join(f"{key}={value:.3f}s" for key, value in step_times.items())
        )
        verbose_print(
            f"Pipeline done: initial_verdict={initial_verdict} correction_applied={correction_applied} final_answer={final_answer}"
        )
        return PipelineResult(
            question=question,
            solution_text=corrected_solution_text if correction_applied else solution_text,
            constraints_text=constraints_text,
            verification_text=final_verification_text,
            final_answer=final_answer,
            verification_verdict=final_verdict,
            initial_solution_text=solution_text,
            initial_verification_text=verification_text,
            initial_final_answer=initial_final_answer,
            initial_verification_verdict=initial_verdict,
            correction_applied=correction_applied,
            correction_reason=reason,
            corrected_solution_text=corrected_solution_text,
            corrected_final_answer=corrected_final_answer,
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
            verification_text="",
            final_answer=final_answer,
            verification_verdict="UNKNOWN",
            initial_solution_text="\n\n".join(samples),
            initial_verification_text="",
            initial_final_answer=final_answer,
            initial_verification_verdict="UNKNOWN",
        )


def _extract_final_answer(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.strip().startswith("FINAL:"):
            return line.split("FINAL:", 1)[1].strip()
    return "unknown"


def _extract_verdict(text: str) -> str:
    for line in text.splitlines():
        if "VERDICT:" in line:
            return line.split("VERDICT:", 1)[1].strip().upper()
    return "UNKNOWN"


def _extract_reason(text: str) -> str:
    for line in text.splitlines():
        if "REASON:" in line:
            return line.split("REASON:", 1)[1].strip()
    return "unknown"


def _majority_vote(answers: List[str]) -> str:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]
