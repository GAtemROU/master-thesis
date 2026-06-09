import json
import math
import random
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# ==== TEMPLATE CONFIG (edit here) ====
SEED = 42
NUM_ROWS = 100
WAGE_RANGE = (10, 30)  # inclusive
HOURS_RANGE = (0, 8)  # inclusive
MINUTE_OPTIONS = [10, 20, 30, 40, 50]
NAMES = [
    "Weng",
    "Rina",
    "Carlos",
    "Ava",
    "Noah",
    "Mina",
    "Hiro",
    "Liam",
    "Sofia",
    "Zane",
]
JOBS = [
    "babysitting",
    "dog walking",
    "tutoring",
    "lawn mowing",
    "house cleaning",
    "car washing",
    "grocery delivery",
    "pet sitting",
    "yard work",
    "dishwashing",
]
OUTPUT_PATH = Path("src/datasets/synthetic_wage_time.jsonl")
# ================================


def _format_hours(hours: int) -> str:
    unit = "hour" if hours == 1 else "hours"
    return f"{hours} {unit}"


def _build_problem(name: str, wage: int, hours: int, minutes: int, job: str) -> str:
    hours_text = _format_hours(hours)
    return (
        f"{name} earns ${wage} an hour for {job}. "
        f"Yesterday, they did {hours_text} and {minutes} minutes of {job}. "
        f"How much did they earn?"
    )


def _compute_answer(wage: int, hours: int, minutes: int) -> int:
    total_minutes = (hours * 60) + minutes
    total_pay = wage * total_minutes
    return total_pay // 60


def _compute_interval(wage: int, hours: int, minutes: int) -> list[int]:
    total_hours = hours + (minutes / 60.0)
    lower = hours * wage
    upper = math.ceil(total_hours) * wage
    return [lower, upper]


def _valid_minutes_for_wage(wage: int) -> list[int]:
    return [m for m in MINUTE_OPTIONS if (wage * m) % 60 == 0]


def generate_rows() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for _ in range(NUM_ROWS):
        wage = rng.randint(WAGE_RANGE[0], WAGE_RANGE[1])
        valid_minutes = _valid_minutes_for_wage(wage)
        while not valid_minutes:
            wage = rng.randint(WAGE_RANGE[0], WAGE_RANGE[1])
            valid_minutes = _valid_minutes_for_wage(wage)
        hours = rng.randint(HOURS_RANGE[0], HOURS_RANGE[1])
        minutes = rng.choice(valid_minutes)
        name = rng.choice(NAMES)
        job = rng.choice(JOBS)
        problem = _build_problem(name, wage, hours, minutes, job)
        answer = _compute_answer(wage, hours, minutes)
        interval = _compute_interval(wage, hours, minutes)
        rows.append(
            {
                "problem": problem,
                "answer": answer,
                "interval": interval,
            }
        )
    return rows


def main() -> None:
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
