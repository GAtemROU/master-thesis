from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import PromptConfig

DEFAULT_PROMPTS_DIR = Path(__file__).parent / "prompt_templates"
DEFAULT_TASK_PROMPT_PATH = DEFAULT_PROMPTS_DIR / "task_prompt.txt"
DEFAULT_CONSTRAINT_PROMPT_PATH = DEFAULT_PROMPTS_DIR / "constraint_prompt.txt"
DEFAULT_VERIFY_PROMPT_PATH = DEFAULT_PROMPTS_DIR / "verify_prompt.txt"
DEFAULT_CONSTRAINED_TASK_PROMPT_PATH = (
    DEFAULT_PROMPTS_DIR / "task_prompt_with_constraints_extended.txt"
)


@dataclass(frozen=True)
class PromptSet:
    task_prompt: str
    constraint_prompt: str
    verify_prompt: str
    constrained_task_prompt: str


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_prompt_set(prompt_config: Optional[PromptConfig]) -> PromptSet:
    if prompt_config is None:
        prompt_config = PromptConfig()
    task_path = prompt_config.task_prompt_path or DEFAULT_TASK_PROMPT_PATH
    constraint_path = (
        prompt_config.constraint_prompt_path or DEFAULT_CONSTRAINT_PROMPT_PATH
    )
    verify_path = prompt_config.verify_prompt_path or DEFAULT_VERIFY_PROMPT_PATH
    constrained_task_path = (
        prompt_config.constrained_task_prompt_path
        or DEFAULT_CONSTRAINED_TASK_PROMPT_PATH
    )
    return PromptSet(
        task_prompt=_load_prompt(Path(task_path)),
        constraint_prompt=_load_prompt(Path(constraint_path)),
        verify_prompt=_load_prompt(Path(verify_path)),
        constrained_task_prompt=_load_prompt(Path(constrained_task_path)),
    )


def load_constraint_prompt(prompt_config: Optional[PromptConfig]) -> str:
    if prompt_config is None:
        prompt_config = PromptConfig()
    constraint_path = (
        prompt_config.constraint_prompt_path or DEFAULT_CONSTRAINT_PROMPT_PATH
    )
    return _load_prompt(Path(constraint_path))
