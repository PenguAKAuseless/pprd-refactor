import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from artifact_utils import ensure_split_run_artifacts, load_json


def build_run_name(backbone: str, mode: str, args: argparse.Namespace) -> str:
    order_tag = ""
    if args.task_order:
        canonical = "-".join([p.strip() for p in str(args.task_order).split(",") if p.strip()])
        order_tag = f"_to{canonical}"
    return (
        f"lab_proto_{backbone}_{mode}_e{args.epochs}_le{args.linear_epochs}"
        f"_b{args.batch_size}_r{args.replay_size}_s{args.seed}{order_tag}"
    )


def find_existing_summary(log_dir: Path, run_name: str) -> Path | None:
    candidates = sorted(log_dir.glob(f"*/{run_name}/results_summary.json"))
    if not candidates:
        return None
    return candidates[-1]


def write_per_mode_records(out_dir: Path, records: List[Dict[str, object]]) -> None:
    per_mode_dir = out_dir / "by_mode"
    per_mode_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        mode = str(record["patch_prototype_mode"])
        backbone = str(record["backbone"])
        path = per_mode_dir / f"{backbone}_{mode}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


def run_training_for_mode(
    backbone: str,
    mode: str,
    args: argparse.Namespace,
    run_name: str,
    python_bin: str,
) -> Path:
    cmd = [
        python_bin,
        "train.py",
        "--backbone",
        backbone,
        "--patch-prototype-mode",
        mode,
        "--patch-proto-sharpness",
        str(args.patch_proto_sharpness),
        "--run-name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--linear-epochs",
        str(args.linear_epochs),
        "--batch-size",
        str(args.batch_size),
        "--replay-size",
        str(args.replay_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--step-eval-every",
        str(args.step_eval_every),
        "--max-step-eval-batches",
        str(args.max_step_eval_batches),
    ]

    if args.task_order:
        cmd.extend(["--task-order", str(args.task_order)])

    if args.enable_csv:
        cmd.append("--enable-csv")
    if args.enable_tb:
        cmd.append("--enable-tb")
    if args.use_wandb:
        cmd.append("--use-wandb")
    if args.max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])

    subprocess.run(cmd, check=True)

    summary_path = find_existing_summary(Path(args.log_dir), run_name)
    if summary_path is None:
        raise RuntimeError(f"Could not find summary after run for mode={mode}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare patch prototype definitions under the same continual setup"
    )
    parser.add_argument("--backbone", type=str, choices=["patch", "roi_patch"], default="patch")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["class_mean_ema", "class_confidence_ema", "class_position_ema"],
    )
    parser.add_argument("--patch-proto-sharpness", type=float, default=1.0)

    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--out-dir", type=str, default="./lab/results")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--linear-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--task-order",
        type=str,
        default=None,
        help="Comma-separated permutation of task indices, e.g. '1,2,3,4,0'.",
    )

    parser.add_argument("--step-eval-every", type=int, default=50)
    parser.add_argument("--max-step-eval-batches", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--enable-csv", action="store_true")
    parser.add_argument("--enable-tb", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    log_dir = (project_root / args.log_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable

    records: List[Dict[str, object]] = []
    for mode in args.modes:
        run_name = build_run_name(args.backbone, mode, args)

        summary_path = None if args.force else find_existing_summary(log_dir, run_name)
        reused = summary_path is not None

        if summary_path is None:
            summary_path = run_training_for_mode(
                backbone=args.backbone,
                mode=mode,
                args=args,
                run_name=run_name,
                python_bin=python_bin,
            )

        split_paths = ensure_split_run_artifacts(summary_path)
        summary = load_json(Path(split_paths["summary_path"]))

        record = {
            "backbone": args.backbone,
            "patch_prototype_mode": mode,
            "run_name": run_name,
            "task_order": args.task_order,
            "task_classes": summary.get("task_classes"),
            "summary_path": str(split_paths["summary_path"]),
            "tasks_path": str(split_paths["tasks_path"]) if split_paths["tasks_path"] is not None else None,
            "step_eval_path": str(split_paths["step_eval_path"]) if split_paths["step_eval_path"] is not None else None,
            "diagnostics_path": str(split_paths["diagnostics_path"]) if split_paths["diagnostics_path"] is not None else None,
            "combined_path": str(split_paths["combined_path"]) if split_paths["combined_path"] is not None else None,
            "reused_existing": reused,
            "final_seen_avg_accuracy": summary.get("final_seen_avg_accuracy"),
            "final_seen_avg_loss": summary.get("final_seen_avg_loss"),
            "final_mean_forgetting": summary.get("final_mean_forgetting"),
        }
        records.append(record)

        acc = float(record["final_seen_avg_accuracy"] or 0.0)
        forget = float(record["final_mean_forgetting"] or 0.0)
        print(
            f"[{mode}] reused={reused} | "
            f"acc={acc:.4f} | "
            f"forget={forget:.4f}"
        )

    best_by_acc = max(records, key=lambda r: float(r["final_seen_avg_accuracy"] or 0.0)) if records else None
    best_by_forgetting = (
        min(records, key=lambda r: float(r["final_mean_forgetting"] or 1e9)) if records else None
    )

    payload = {
        "backbone": args.backbone,
        "records": records,
        "best_by_accuracy": best_by_acc,
        "best_by_forgetting": best_by_forgetting,
    }

    out_path = out_dir / "patch_prototype_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_per_mode_records(out_dir, records)

    print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()
