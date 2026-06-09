
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List

from .logger import verbose_print


@dataclass
class Example:
    example_id: str
    question: str
    answer: str
    interval: tuple[float, float] | None = None


def _parse_interval_field(value: object) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            lower = float(value[0])
            upper = float(value[1])
        except (TypeError, ValueError):
            return None
        return (lower, upper)
    return None


def load_jsonl(path: str | Path) -> List[Example]:
    examples: List[Example] = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        interval = _parse_interval_field(data.get("interval"))
        examples.append(
            Example(
                example_id=str(data.get("id", index)),
                question=data["question"],
                answer=str(data["answer"]),
                interval=interval,
            )
        )
    verbose_print(f"Loaded {len(examples)} examples from jsonl: {path}")
    return examples


def load_gsm8k(path: str | Path) -> List[Example]:
    path_value = str(path)
    if path_value.startswith("hf:"):
        dataset_id, split = _parse_hf_spec(path_value)
        return _load_gsm8k_from_hf(dataset_id, split)
    examples: List[Example] = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        answer = _extract_gsm8k_answer(str(data.get("answer", "")))
        examples.append(
            Example(
                example_id=str(data.get("id", index)),
                question=data["question"],
                answer=answer,
                interval=_parse_interval_field(data.get("interval")),
            )
        )
    verbose_print(f"Loaded {len(examples)} examples from gsm8k: {path}")
    return examples


def load_math500(path: str | Path) -> List[Example]:
    raw = Path(path).read_text()
    if path.endswith(".json"):
        items = json.loads(raw)
    else:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]

    examples: List[Example] = []
    for index, data in enumerate(items):
        question = data.get("problem") or data.get("question")
        solution = data.get("solution") or ""
        answer = data.get("answer") or data.get("final_answer")
        interval = _parse_interval_field(data.get("interval"))
        if answer is None:
            answer = _extract_math_answer(str(solution))
        else:
            answer = str(answer)
        examples.append(
            Example(
                example_id=str(data.get("id", index)),
                question=question,
                answer=answer,
                interval=interval,
            )
        )
    verbose_print(f"Loaded {len(examples)} examples from math500: {path}")
    return examples


def load_dataset(name: str, path: str | Path) -> List[Example]:
    name = name.lower()
    verbose_print(f"Loading dataset '{name}' from {path}")
    if name == "jsonl":
        return load_jsonl(path)
    if name == "gsm8k":
        return load_gsm8k(path)
    if name == "math500":
        return load_math500(path)
    raise ValueError("Unknown dataset name: {}".format(name))


def _load_gsm8k_from_hf(dataset_id: str, split: str) -> List[Example]:
    from datasets import load_dataset
    verbose_print(
        f"Loading GSM8K from Hugging Face: dataset_id={dataset_id} split={split}"
    )
    dataset = load_dataset(dataset_id, "main", split=split)
    examples: List[Example] = []
    for index, row in enumerate(dataset):
        answer = _extract_gsm8k_answer(str(row.get("answer", "")))
        examples.append(
            Example(
                example_id=str(row.get("id", index)),
                question=row["question"],
                answer=answer,
            )
        )
    verbose_print(
        f"Loaded {len(examples)} examples from HF dataset {dataset_id} split {split}"
    )
    return examples


def _parse_hf_spec(spec: str) -> tuple[str, str]:
    default_split = "train[:1000]"
    remainder = spec.split(":", 1)[1] if ":" in spec else ""
    if not remainder:
        return "gsm8k", default_split
    dataset_part, split_part = remainder, ""
    if "?split=" in remainder:
        dataset_part, split_part = remainder.split("?split=", 1)
    elif ":" in remainder:
        dataset_part, split_part = remainder.split(":", 1)
    dataset_id = dataset_part or "gsm8k"
    split = split_part or default_split
    return dataset_id, split


def _extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        return text.split("####", 1)[1].strip()
    return text.strip()


def _extract_math_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text.strip()
