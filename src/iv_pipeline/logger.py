from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import traceback
from typing import Any, Dict


class RunLogger:
    def __init__(self, run_id: str, run_timestamp: datetime, output_dir: Path) -> None:
        self.run_id = run_id
        self.run_timestamp = run_timestamp
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"run_{run_id}.json"

    def write_start(self, inputs: Dict[str, Any]) -> None:
        record = {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp.isoformat(),
            "status": "running",
            "inputs": inputs,
        }
        self._write_record(record)

    def write_results(self, metrics: Dict[str, Any], details: Dict[str, Any] | None = None) -> None:
        record = self._read_record()
        record["status"] = "completed"
        record["metrics"] = metrics
        if details is not None:
            record["details"] = details
        record["completed_timestamp"] = datetime.utcnow().isoformat()
        self._write_record(record)

    def write_error(self, exc: BaseException) -> None:
        record = self._read_record()
        record["status"] = "failed"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        record["completed_timestamp"] = datetime.utcnow().isoformat()
        self._write_record(record)

    def _read_record(self) -> Dict[str, Any]:
        if self.output_path.exists():
            return json.loads(self.output_path.read_text())
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp.isoformat(),
        }

    def _write_record(self, record: Dict[str, Any]) -> None:
        self.output_path.write_text(json.dumps(record, indent=2, sort_keys=True))
