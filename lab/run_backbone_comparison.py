import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from artifact_utils import ensure_split_run_artifacts, load_json


def build_run_name(backbone: str, args: argparse.Namespace) -> str:
    mode_tag = args.patch_prototype_mode.replace("_", "-")
    order_tag = ""
    if args.task_order:
        canonical = "-".join([p.strip() for p in str(args.task_order).split(",") if p.strip()])
        order_tag = f"_to{canonical}"
    return (
        f"lab_{backbone}_{mode_tag}_e{args.epochs}_le{args.linear_epochs}"
        f"_b{args.batch_size}_r{args.replay_size}_s{args.seed}{order_tag}"
    )


def find_existing_summary(log_dir: Path, run_name: str) -> Path | None:
    candidates = sorted(log_dir.glob(f"*/{run_name}/results_summary.json"))
    if not candidates:
        return None
    return candidates[-1]


def find_existing_run_dir(log_dir: Path, run_name: str) -> Path | None:
    candidates = sorted(log_dir.glob(f"*/{run_name}"))
    if not candidates:
        return None
    return candidates[-1]


def _extract_task_id_from_name(path: Path) -> Optional[int]:
    name = path.name
    match_model_task = re.search(r"model_task_(\d+)\.pth$", name)
    if match_model_task:
        return int(match_model_task.group(1))

    match_ckpt_task = re.search(r"task_(\d+)(?:[-_].*)?\.ckpt$", name)
    if match_ckpt_task:
        return int(match_ckpt_task.group(1))

    if name == "model_final.pth":
        return 4
    return None


def find_best_checkpoint(run_dir: Path) -> Tuple[Path | None, Optional[int]]:
    final_model = run_dir / "model_final.pth"
    if final_model.exists():
        return final_model, 4

    model_task_files = []
    for path in run_dir.glob("model_task_*.pth"):
        task_id = _extract_task_id_from_name(path)
        if task_id is not None:
            model_task_files.append((task_id, path))
    if model_task_files:
        task_id, path = max(model_task_files, key=lambda x: x[0])
        return path, task_id

    ckpt_task_files = []
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob("task_*.ckpt"):
            task_id = _extract_task_id_from_name(path)
            if task_id is not None:
                ckpt_task_files.append((task_id, path))
        if ckpt_task_files:
            task_id, path = max(ckpt_task_files, key=lambda x: x[0])
            return path, task_id

        last_ckpt = checkpoint_dir / "last.ckpt"
        if last_ckpt.exists():
            inferred_task_id = max((item[0] for item in ckpt_task_files), default=None)
            return last_ckpt, inferred_task_id

    return None, None


def step_eval_reached_task_4(step_eval_path: Path | None) -> bool:
    if step_eval_path is None or not step_eval_path.exists():
        return False
    try:
        payload = load_json(step_eval_path)
    except Exception:
        return False
    if not isinstance(payload, list) or not payload:
        return False
    seen_task_ids = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        value = row.get("seen_task_id", None)
        try:
            seen_task_ids.append(int(float(value)))
        except Exception:
            continue
    if not seen_task_ids:
        return False
    return max(seen_task_ids) >= 4


def write_split_comparison_records(out_dir: Path, records: List[Dict[str, object]]) -> None:
    per_backbone_dir = out_dir / "by_backbone"
    per_backbone_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        backbone = str(record["backbone"])
        path = per_backbone_dir / f"{backbone}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


def run_training_for_backbone(
    backbone: str,
    args: argparse.Namespace,
    run_name: str,
    python_bin: str,
    log_dir: Path,
) -> Path:
    cmd = [
        python_bin,
        "train.py",
        "--backbone",
        backbone,
        "--patch-prototype-mode",
        args.patch_prototype_mode,
        "--patch-proto-sharpness",
        str(args.patch_proto_sharpness),
        "--prototype-momentum",
        str(args.prototype_momentum),
        "--lambda-patch-ce",
        str(args.lambda_patch_ce),
        "--lambda-prd",
        str(args.lambda_prd),
        "--nce-temp",
        str(args.nce_temp),
        "--current-temp",
        str(args.current_temp),
        "--past-temp",
        str(args.past_temp),
        "--roi-min-scale",
        str(args.roi_min_scale),
        "--roi-max-scale",
        str(args.roi_max_scale),
        "--roi-prob",
        str(args.roi_prob),
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

    summary_path = find_existing_summary(log_dir, run_name)
    if summary_path is None:
        raise RuntimeError(f"Could not find summary after run for {backbone}")
    return summary_path


def run_eval_for_backbone_from_checkpoint(
    backbone: str,
    args: argparse.Namespace,
    run_name: str,
    python_bin: str,
    run_dir: Path,
    checkpoint_path: Path,
    eval_task_id: Optional[int],
    log_dir: Path,
) -> Path:
    cmd = [
        python_bin,
        "train.py",
        "--backbone",
        backbone,
        "--patch-prototype-mode",
        args.patch_prototype_mode,
        "--patch-proto-sharpness",
        str(args.patch_proto_sharpness),
        "--prototype-momentum",
        str(args.prototype_momentum),
        "--lambda-patch-ce",
        str(args.lambda_patch_ce),
        "--lambda-prd",
        str(args.lambda_prd),
        "--nce-temp",
        str(args.nce_temp),
        "--current-temp",
        str(args.current_temp),
        "--past-temp",
        str(args.past_temp),
        "--roi-min-scale",
        str(args.roi_min_scale),
        "--roi-max-scale",
        str(args.roi_max_scale),
        "--roi-prob",
        str(args.roi_prob),
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
        "--eval-only",
        "--eval-from",
        str(checkpoint_path),
        "--eval-run-dir",
        str(run_dir),
    ]

    if args.task_order:
        cmd.extend(["--task-order", str(args.task_order)])

    if eval_task_id is not None:
        cmd.extend(["--eval-task-id", str(eval_task_id)])
    if args.max_eval_batches is not None:
        cmd.extend(["--max-eval-batches", str(args.max_eval_batches)])

    subprocess.run(cmd, check=True)

    summary_path = run_dir / "results_summary.json"
    if not summary_path.exists():
        found = find_existing_summary(log_dir, run_name)
        if found is None:
            raise RuntimeError(f"Could not find eval summary after checkpoint eval for {backbone}")
        return found
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backbone comparison lab and skip existing runs")
    parser.add_argument("--backbones", nargs="+", default=["patch", "roi_patch"])
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--out-dir", type=str, default="./lab/results")

    parser.add_argument(
        "--patch-prototype-mode",
        type=str,
        choices=["class_mean_ema", "class_confidence_ema", "class_position_ema"],
        default="class_position_ema",
    )
    parser.add_argument("--patch-proto-sharpness", type=float, default=1.0)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)

    parser.add_argument("--nce-temp", type=float, default=0.07)
    parser.add_argument("--current-temp", type=float, default=1.0)
    parser.add_argument("--past-temp", type=float, default=2.0)
    parser.add_argument("--lambda-patch-ce", type=float, default=1.0)
    parser.add_argument("--lambda-prd", type=float, default=1.0)

    parser.add_argument("--roi-min-scale", type=float, default=0.55)
    parser.add_argument("--roi-max-scale", type=float, default=1.0)
    parser.add_argument("--roi-prob", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--linear-epochs", type=int, default=50)
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
    parser.add_argument("--require-complete-task4", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    log_dir = (project_root / args.log_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable

    records: List[Dict[str, object]] = []
    for backbone in args.backbones:
        run_name = build_run_name(backbone, args)

        summary_path = None
        reused = False
        checkpoint_recovered = False
        run_dir = None if args.force else find_existing_run_dir(log_dir, run_name)

        if run_dir is not None:
            summary_candidate = run_dir / "results_summary.json"
            if summary_candidate.exists():
                summary_path = summary_candidate
                reused = True
            else:
                checkpoint_path, eval_task_id = find_best_checkpoint(run_dir)
                if checkpoint_path is not None:
                    print(
                        f"[{backbone}] found checkpoint without summary, evaluating: {checkpoint_path.name}"
                    )
                    summary_path = run_eval_for_backbone_from_checkpoint(
                        backbone=backbone,
                        args=args,
                        run_name=run_name,
                        python_bin=python_bin,
                        run_dir=run_dir,
                        checkpoint_path=checkpoint_path,
                        eval_task_id=eval_task_id,
                        log_dir=log_dir,
                    )
                    reused = True
                    checkpoint_recovered = True

        if summary_path is None:
            summary_path = run_training_for_backbone(backbone, args, run_name, python_bin, log_dir)
            run_dir = summary_path.parent

        split_paths = ensure_split_run_artifacts(summary_path)

        if args.require_complete_task4 and not step_eval_reached_task_4(split_paths["step_eval_path"]):
            print(
                f"[{backbone}] existing run is incomplete (task 4 missing in step eval). "
                "Rerunning this backbone because --require-complete-task4 is set."
            )
            summary_path = run_training_for_backbone(backbone, args, run_name, python_bin, log_dir)
            split_paths = ensure_split_run_artifacts(summary_path)
            reused = False
            checkpoint_recovered = False

        summary = load_json(Path(split_paths["summary_path"]))
        complete_task4 = step_eval_reached_task_4(split_paths["step_eval_path"])

        record = {
            "backbone": backbone,
            "run_name": run_name,
            "task_order": args.task_order,
            "task_classes": summary.get("task_classes"),
            "summary_path": str(split_paths["summary_path"]),
            "tasks_path": str(split_paths["tasks_path"]) if split_paths["tasks_path"] is not None else None,
            "step_eval_path": str(split_paths["step_eval_path"]) if split_paths["step_eval_path"] is not None else None,
            "diagnostics_path": str(split_paths["diagnostics_path"]) if split_paths["diagnostics_path"] is not None else None,
            "combined_path": str(split_paths["combined_path"]) if split_paths["combined_path"] is not None else None,
            "reused_existing": reused,
            "checkpoint_recovered": checkpoint_recovered,
            "is_complete_task4": complete_task4,
            "eval_task_id": summary.get("eval_task_id"),
            "final_seen_avg_accuracy": summary.get("final_seen_avg_accuracy"),
            "final_seen_avg_loss": summary.get("final_seen_avg_loss"),
            "final_mean_forgetting": summary.get("final_mean_forgetting"),
        }
        records.append(record)
        acc = float(record["final_seen_avg_accuracy"] or 0.0)
        forget = float(record["final_mean_forgetting"] or 0.0)
        print(
            f"[{backbone}] reused={reused} | recovered_ckpt={checkpoint_recovered} | "
            f"complete_task4={complete_task4} | "
            f"acc={acc:.4f} | "
            f"forget={forget:.4f}"
        )

    out_path = out_dir / "backbone_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)
    write_split_comparison_records(out_dir, records)

    print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()
