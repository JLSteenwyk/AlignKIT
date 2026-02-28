"""JSONL metrics logger for training and evaluation."""

import json
import time
from pathlib import Path
from typing import Any, Dict


class JSONLLogger:
    """Append-only JSONL logger for training metrics."""

    def __init__(self, log_dir: Path, prefix: str = "train"):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{prefix}.jsonl"
        self._entries = []

    def log(self, metrics: Dict[str, Any], step: int):
        """Log a metrics dict with step number and timestamp."""
        entry = {"step": step, "timestamp": time.time()}
        entry.update(metrics)
        self._entries.append(entry)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self):
        """Read all logged entries."""
        entries = []
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries
