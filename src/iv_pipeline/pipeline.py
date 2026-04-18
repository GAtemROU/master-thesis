from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import PipelineConfig
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

    def run(self, question: str) -> PipelineResult:
        solution_prompt = self.prompts.task_prompt.format(question=question)
        solution_text = self.sampler_model.generate(solution_prompt)

        constraint_prompt = self.prompts.constraint_prompt.format(question=question)
        constraints_text = self.constraint_model.generate(constraint_prompt)

        verify_prompt = self.prompts.verify_prompt.format(
            question=question,
            solution=solution_text,
            constraints=constraints_text,
        )
        verification_text = self.verifier_model.generate(verify_prompt)

        final_answer = _extract_final_answer(solution_text)
        verdict = _extract_verdict(verification_text)
        return PipelineResult(
            question=question,
            solution_text=solution_text,
            constraints_text=constraints_text,
            verification_text=verification_text,
            final_answer=final_answer,
            verification_verdict=verdict,
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


def _majority_vote(answers: List[str]) -> str:
    counts: Dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]
