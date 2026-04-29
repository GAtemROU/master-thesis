from __future__ import annotations

import argparse
import json
from pathlib import Path


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show summary for latest run logs.")
    parser.add_argument("--limit", type=int, default=5, help="Number of runs to show.")
    args = parser.parse_args()

    runs_dir = Path(__file__).parent / "runs"
    if not runs_dir.exists():
        print("No runs directory found.")
        return 0

    run_files = sorted(runs_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        print("No run logs found.")
        return 0

    for path in run_files[: args.limit]:
        data = json.loads(path.read_text())
        inputs = data.get("inputs", {})
        args_data = inputs.get("args", {})
        metrics = data.get("metrics", {})
        details = data.get("details", {})
        line = (
            f"{data.get('run_id', '-')} | "
            f"{data.get('status', '-')} | "
            f"{_format_value(args_data.get('dataset_name'))} | "
            f"{_format_value(args_data.get('dataset'))} | "
            f"acc={_format_value(metrics.get('accuracy'))} | "
            f"trace_only={_format_value(details.get('trace_only'))} | "
            f"trace_path={_format_value(details.get('trace_path'))} | "
            f"path={path}"
        )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
