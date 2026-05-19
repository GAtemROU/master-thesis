from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict

from .logger import verbose_print


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


class HuggingFaceCausalLMModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        torch_dtype: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_torch_dtype(torch, torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.generation_config.max_length = None
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def generate(self, prompt: str, **kwargs: Any) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        generate_kwargs = dict(self.generate_kwargs)
        generate_kwargs.update(kwargs)
        stop_after_line_prefixes = generate_kwargs.pop("stop_after_line_prefixes", None)
        stop_after_prefix_min_chars = int(
            generate_kwargs.pop("stop_after_prefix_min_chars", 1)
        )
        if stop_after_line_prefixes:
            class _StopAfterLinePrefix(StoppingCriteria):
                def __init__(
                    self,
                    tokenizer: Any,
                    start_len: int,
                    prefixes: list[str],
                    min_chars: int,
                ) -> None:
                    self.tokenizer = tokenizer
                    self.start_len = start_len
                    self.prefixes = tuple(prefixes)
                    self.min_chars = min_chars

                def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
                    generated_ids = input_ids[0][self.start_len:]
                    if generated_ids.numel() == 0:
                        return False
                    text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    for prefix in self.prefixes:
                        index = text.find(prefix)
                        if index != -1:
                            after_prefix = text[index + len(prefix) :].strip()
                            if len(after_prefix) >= self.min_chars:
                                return True
                    return False

            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    _StopAfterLinePrefix(
                        self.tokenizer,
                        input_len,
                        stop_after_line_prefixes,
                        stop_after_prefix_min_chars,
                    )
                ]
            )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        generated = output_ids[0][input_len:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if stop_after_line_prefixes:
            text = _strip_after_prefixed_line(text, stop_after_line_prefixes)
        return text


_MODEL_CACHE: Dict[str, BaseModel] = {}


def get_model(name: str, params: Dict[str, Any] | None = None) -> BaseModel:
    params = params or {}
    cache_key = json.dumps({"name": name, "params": params}, sort_keys=True)
    if cache_key in _MODEL_CACHE:
        verbose_print(f"Model cache hit: {name} params={params}")
        return _MODEL_CACHE[cache_key]
    verbose_print(f"Loading model: {name} params={params}")
    if name == "mock_math":
        model = MockMathModel()
    elif name == "mock_constraints":
        model = MockMathModel()
    elif name == "mock_verifier":
        model = MockVerifierModel()
    elif name == "echo":
        model = EchoModel()
    elif name == "hf_causal_lm":
        model = HuggingFaceCausalLMModel(**params)
    else:
        raise ValueError(f"Unknown model name: {name}")
    _MODEL_CACHE[cache_key] = model
    return model


def _strip_after_prefixed_line(text: str, prefixes: list[str]) -> str:
    best_end = None
    for prefix in prefixes:
        index = text.find(prefix)
        if index == -1:
            continue
        line_end = text.find("\n", index)
        end = len(text) if line_end == -1 else line_end
        if best_end is None or end < best_end:
            best_end = end
    if best_end is None:
        return text
    return text[:best_end].strip()


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


def _resolve_torch_dtype(torch: Any, torch_dtype: str | None) -> Any | None:
    if torch_dtype is None:
        return None
    normalized = torch_dtype.lower().strip()
    if normalized in {"auto", "none"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
    return mapping[normalized]
