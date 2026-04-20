import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_split_run_artifacts(summary_path: Path) -> Dict[str, Path | None]:
    run_dir = summary_path.parent
    tasks_path = run_dir / "results_tasks.json"
    step_path = run_dir / "results_step_eval.json"
    diagnostics_path = run_dir / "results_diagnostics.json"
    summary_out = run_dir / "results_summary.json"
    combined_path = run_dir / "results.json"

    if combined_path.exists():
        combined = load_json(combined_path)
        if not tasks_path.exists() and "tasks" in combined:
            with open(tasks_path, "w", encoding="utf-8") as f:
                json.dump(combined["tasks"], f, indent=2)
        if not step_path.exists() and "step_eval" in combined:
            with open(step_path, "w", encoding="utf-8") as f:
                json.dump(combined["step_eval"], f, indent=2)
        if not summary_out.exists() and "summary" in combined:
            summary_payload = dict(combined["summary"])
            if "run" in combined:
                summary_payload["run"] = combined["run"]
            with open(summary_out, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
        if not diagnostics_path.exists() and "diagnostics_over_stages" in combined:
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(combined["diagnostics_over_stages"], f, indent=2)

    return {
        "tasks_path": tasks_path if tasks_path.exists() else None,
        "step_eval_path": step_path if step_path.exists() else None,
        "diagnostics_path": diagnostics_path if diagnostics_path.exists() else None,
        "summary_path": summary_out if summary_out.exists() else summary_path,
        "combined_path": combined_path if combined_path.exists() else None,
    }
