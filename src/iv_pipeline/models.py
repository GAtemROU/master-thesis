from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Dict


class BaseModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


class MockMathModel(BaseModel):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        if "List domain- or task-related constraints" in prompt:
            return "- answer should be a number\n- arithmetic should be consistent"
        if "VERDICT:" in prompt or "Verify the proposed solution" in prompt:
            return "VERDICT: PASS\nREASON: constraints appear satisfied"
        expression = _extract_expression(prompt)
        if expression is None:
            return "Solution: unable to parse expression\nFINAL: unknown"
        value = _safe_eval(expression)
        return f"Solution: computed {expression}\nFINAL: {value}"


class MockVerifierModel(BaseModel):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "VERDICT: PASS\nREASON: mock verifier does not reject solutions"


class EchoModel(BaseModel):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        return prompt


def get_model(name: str, params: Dict[str, Any] | None = None) -> BaseModel:
    params = params or {}
    if name == "mock_math":
        return MockMathModel()
    if name == "mock_constraints":
        return MockMathModel()
    if name == "mock_verifier":
        return MockVerifierModel()
    if name == "echo":
        return EchoModel()
    raise ValueError(f"Unknown model name: {name}")


def _extract_expression(prompt: str) -> str | None:
    match = re.search(r"(-?\d+)\s*([+\-*/])\s*(-?\d+)", prompt)
    if not match:
        return None
    return f"{match.group(1)} {match.group(2)} {match.group(3)}"


def _safe_eval(expression: str) -> str:
    parts = expression.split()
    if len(parts) != 3:
        return "unknown"
    left, op, right = parts
    try:
        a = int(left)
        b = int(right)
    except ValueError:
        return "unknown"
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    if op == "/":
        if b == 0:
            return "undefined"
        return str(a / b)
    return "unknown"
