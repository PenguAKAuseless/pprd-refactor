import json
from datetime import datetime
from pathlib import Path
from typing import Dict


class LitLogger:
    """Lightweight structured logger for paper-grade experiment artifacts."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_fp = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")
        self.metrics_fp = open(self.run_dir / "metrics.csv", "a", encoding="utf-8")

        if self.metrics_fp.tell() == 0:
            self.metrics_fp.write("timestamp,step,key,value\n")
            self.metrics_fp.flush()

    def log_text(self, message: str) -> None:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": "text",
            "message": message,
        }
        self.events_fp.write(json.dumps(payload) + "\n")
        self.events_fp.flush()

    def log_event(self, event_name: str, payload: Dict[str, object]) -> None:
        event = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "type": event_name,
            "payload": payload,
        }
        self.events_fp.write(json.dumps(event) + "\n")
        self.events_fp.flush()

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for key, value in metrics.items():
            self.metrics_fp.write(f"{ts},{step},{key},{value}\n")
        self.metrics_fp.flush()

    def close(self) -> None:
        self.events_fp.close()
        self.metrics_fp.close()
