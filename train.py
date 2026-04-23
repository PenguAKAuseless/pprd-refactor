import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import numpy as np
except Exception:
    np = None

try:
    import pytorch_lightning as pl
except Exception:
    pl = None

try:
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
except Exception:
    CSVLogger = None
    TensorBoardLogger = None
    WandbLogger = None
    LearningRateMonitor = None
    ModelCheckpoint = None

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from data.datasets import SplitCIFAR10Manager
from models.patch_backbone import PrototypePatchBackbone
from utils.eval_diagnostics import (
    init_confusion_matrix,
    summarize_confusion,
    update_confusion_matrix,
)
from utils.litlogger import LitLogger
from utils.losses import ISSupConLoss, prd_loss

LightningModuleBase = pl.LightningModule if pl is not None else nn.Module


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if pl is not None:
        pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_task_order(task_order_text: Optional[str], total_tasks: int) -> Optional[List[int]]:
    if task_order_text is None:
        return None

    text = str(task_order_text).strip()
    if not text:
        return None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != total_tasks:
        raise ValueError(
            f"--task-order must contain exactly {total_tasks} comma-separated integers. "
            f"Received: {task_order_text!r}"
        )

    try:
        order = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"--task-order must contain only integers. Received: {task_order_text!r}"
        ) from exc

    expected = set(range(total_tasks))
    if set(order) != expected:
        raise ValueError(
            f"--task-order must be a permutation of {sorted(expected)}. Received: {order}"
        )
    return order


def _build_run_name_and_id(args: argparse.Namespace) -> Tuple[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = (
            f"splitcifar10_{args.backbone}_e{args.epochs}_le{args.linear_epochs}"
            f"_b{args.batch_size}_r{args.replay_size}_s{args.seed}_{stamp}"
        )
    run_id = args.run_id if args.run_id else hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:12]
    return run_name, run_id


def _make_run_dir(args: argparse.Namespace, run_name: str) -> Path:
    base = Path(args.log_dir)
    base.mkdir(parents=True, exist_ok=True)
    date_dir = base / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    run_dir = date_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_run_dir_for_eval(args: argparse.Namespace, run_name: str) -> Path:
    if args.eval_run_dir:
        run_dir = Path(args.eval_run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return _make_run_dir(args, run_name)


def _save_run_artifacts(args: argparse.Namespace, run_dir: Path, model: nn.Module) -> None:
    command = " ".join(sys.argv)

    (run_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")
    run_script = "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n"
    (run_dir / "run.sh").write_text(run_script, encoding="utf-8")

    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    (run_dir / "model.txt").write_text(str(model) + "\n", encoding="utf-8")


def _extract_model_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")

    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        # PyTorch Lightning checkpoints store module weights under `state_dict`
        # and prefix model parameters as `model.` inside ContinualLightningModule.
        source_state = payload["state_dict"]
        state_dict: Dict[str, torch.Tensor] = {}
        for key, value in source_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not key.startswith("model."):
                continue
            clean_key = key[len("model.") :]
            state_dict[clean_key] = value
        if state_dict:
            return state_dict

    state_dict = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        state_dict[clean_key] = value
    if not state_dict:
        raise RuntimeError("No model parameter tensors found in checkpoint payload.")
    return state_dict


def _infer_eval_task_id_from_checkpoint_path(checkpoint_path: Path, total_tasks: int = 5) -> Optional[int]:
    name = checkpoint_path.name

    if name == "model_final.pth":
        return total_tasks - 1

    match_model_task = re.search(r"model_task_(\d+)\.pth$", name)
    if match_model_task:
        return int(match_model_task.group(1))

    match_ckpt_task = re.search(r"task_(\d+)(?:[-_].*)?\.ckpt$", name)
    if match_ckpt_task:
        return int(match_ckpt_task.group(1))

    if name.startswith("last") and checkpoint_path.suffix == ".ckpt":
        task_candidates = []
        for task_ckpt in checkpoint_path.parent.glob("task_*.ckpt"):
            match = re.search(r"task_(\d+)(?:[-_].*)?\.ckpt$", task_ckpt.name)
            if match:
                task_candidates.append(int(match.group(1)))
        if task_candidates:
            return max(task_candidates)

    return None


def _load_historical_best_by_task(run_dir: Path, upto_task: int) -> Dict[int, float]:
    tasks_path = run_dir / "results_tasks.json"
    if not tasks_path.exists():
        return {}

    try:
        with open(tasks_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    if not isinstance(payload, list):
        return {}

    best_by_task: Dict[int, float] = {}
    for stage in payload:
        if not isinstance(stage, dict):
            continue
        per_task = stage.get("seen_task_metrics", [])
        if not isinstance(per_task, list):
            continue
        for metric in per_task:
            if not isinstance(metric, dict):
                continue
            try:
                task_id = int(metric.get("task_id", -1))
                acc = float(metric.get("accuracy", 0.0))
            except Exception:
                continue
            if task_id < 0 or task_id > upto_task:
                continue
            best_by_task[task_id] = max(best_by_task.get(task_id, acc), acc)

    return best_by_task


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    if not args.eval_from:
        raise ValueError("--eval-only requires --eval-from <checkpoint_path>.")

    set_seed(args.seed)
    device, _, _ = _resolve_runtime(args)

    checkpoint_path = Path(args.eval_from).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_name, run_id = _build_run_name_and_id(args)
    run_dir = _resolve_run_dir_for_eval(args, run_name)

    total_tasks = 5
    task_order = _parse_task_order(args.task_order, total_tasks)

    manager = SplitCIFAR10Manager(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay_size=args.replay_size,
        tasks=total_tasks,
        classes_per_task=2,
        seed=args.seed,
        task_order=task_order,
    )
    model = _build_model(args, device)

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_model_state_dict(payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    inferred_task_id = _infer_eval_task_id_from_checkpoint_path(
        checkpoint_path, total_tasks=manager.tasks
    )
    if args.eval_task_id is not None:
        eval_task_id = int(args.eval_task_id)
    elif inferred_task_id is not None:
        eval_task_id = inferred_task_id
    else:
        eval_task_id = manager.tasks - 1
        print(
            "[Eval] Could not infer task id from checkpoint name. "
            f"Defaulting evaluation to task {manager.tasks - 1}. Use --eval-task-id to override."
        )
    eval_task_id = max(0, min(manager.tasks - 1, eval_task_id))

    seen_tasks = list(range(eval_task_id + 1))
    seen_train_loader = manager.get_seen_train_loader(eval_task_id, batch_size=args.batch_size)
    seen_test_loaders = {
        seen_task_id: manager.get_task_test_loader(seen_task_id, batch_size=args.batch_size)
        for seen_task_id in seen_tasks
    }

    eval_summary = linear_eval_seen_tasks(
        model=model,
        seen_train_loader=seen_train_loader,
        seen_test_loaders=seen_test_loaders,
        seen_tasks=seen_tasks,
        device=device,
        num_classes=10,
        epochs=args.linear_epochs,
        lr=args.linear_lr,
        max_batches=args.max_eval_batches,
        return_diagnostics=True,
    )

    best_hist = _load_historical_best_by_task(run_dir, eval_task_id)
    per_task_eval = []
    for metric in eval_summary["per_task"]:
        seen_task_id = int(metric["task_id"])
        acc = float(metric["accuracy"])
        loss = float(metric["loss"])
        previous_best = best_hist.get(seen_task_id, acc)
        forgetting = max(0.0, previous_best - acc)
        per_task_eval.append(
            {
                "task_id": seen_task_id,
                "loss": loss,
                "accuracy": acc,
                "forgetting": forgetting,
            }
        )

    raw_task_diagnostics = eval_summary.get("per_task_diagnostics", [])
    if not isinstance(raw_task_diagnostics, list):
        raw_task_diagnostics = []

    seen_avg_accuracy = float(eval_summary["seen_avg_accuracy"])
    seen_avg_loss = float(eval_summary["seen_avg_loss"])
    mean_forgetting = (
        float(sum(m["forgetting"] for m in per_task_eval) / len(per_task_eval))
        if per_task_eval
        else 0.0
    )
    stage_diagnostics = _build_stage_diagnostics(
        stage_task_id=eval_task_id,
        seen_tasks=seen_tasks,
        per_task_eval=per_task_eval,
        per_task_diagnostics=raw_task_diagnostics,
        seen_avg_accuracy=seen_avg_accuracy,
        num_classes=10,
    )

    existing_step_eval = []
    step_eval_path = run_dir / "results_step_eval.json"
    if step_eval_path.exists():
        try:
            with open(step_eval_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                existing_step_eval = payload
        except Exception:
            existing_step_eval = []

    existing_tasks = []
    tasks_path = run_dir / "results_tasks.json"
    if tasks_path.exists():
        try:
            with open(tasks_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                existing_tasks = [row for row in payload if isinstance(row, dict)]
        except Exception:
            existing_tasks = []

    stage_result = {
        "task_id": eval_task_id,
        "avg_train_loss": None,
        "seen_tasks": seen_tasks,
        "seen_avg_loss": seen_avg_loss,
        "seen_avg_accuracy": seen_avg_accuracy,
        "mean_forgetting": mean_forgetting,
        "problematic_tasks": stage_diagnostics["problematic_tasks"],
        "seen_task_metrics": per_task_eval,
        "eval_only": True,
        "checkpoint_path": str(checkpoint_path),
    }

    by_stage: Dict[int, Dict[str, object]] = {}
    for row in existing_tasks:
        try:
            stage_id = int(row.get("task_id", -1))
        except Exception:
            continue
        if stage_id < 0:
            continue
        by_stage[stage_id] = row
    by_stage[eval_task_id] = stage_result
    tasks_out = [by_stage[stage_id] for stage_id in sorted(by_stage.keys())]

    behavior_over_stages: List[Dict[str, object]] = []
    for row in tasks_out:
        try:
            stage_id = int(row.get("task_id", -1))
        except Exception:
            continue
        behavior_over_stages.append(
            {
                "stage_task_id": stage_id,
                "seen_tasks": row.get("seen_tasks", []),
                "per_task": row.get("seen_task_metrics", []),
            }
        )

    diagnostics_path = run_dir / "results_diagnostics.json"
    existing_diagnostics: List[Dict[str, object]] = []
    if diagnostics_path.exists():
        try:
            with open(diagnostics_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                existing_diagnostics = [row for row in payload if isinstance(row, dict)]
        except Exception:
            existing_diagnostics = []

    diagnostics_by_stage: Dict[int, Dict[str, object]] = {}
    for row in existing_diagnostics:
        try:
            stage_id = int(row.get("stage_task_id", -1))
        except Exception:
            continue
        if stage_id < 0:
            continue
        diagnostics_by_stage[stage_id] = row
    diagnostics_by_stage[eval_task_id] = stage_diagnostics
    diagnostics_over_stages = [
        diagnostics_by_stage[stage_id] for stage_id in sorted(diagnostics_by_stage.keys())
    ]

    summary_payload = {
        "run": {"name": run_name, "id": run_id},
        "task_classes": manager.task_classes,
        "final_seen_avg_accuracy": seen_avg_accuracy,
        "final_seen_avg_loss": seen_avg_loss,
        "final_mean_forgetting": mean_forgetting,
        "diagnostics_artifact": "results_diagnostics.json",
        "eval_only": True,
        "eval_task_id": eval_task_id,
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }

    with open(run_dir / "results_tasks.json", "w", encoding="utf-8") as f:
        json.dump(tasks_out, f, indent=2)
    with open(run_dir / "results_step_eval.json", "w", encoding="utf-8") as f:
        json.dump(existing_step_eval, f, indent=2)
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_over_stages, f, indent=2)
    with open(run_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    with open(run_dir / "results_eval_from_ckpt.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": {"name": run_name, "id": run_id},
                "task_classes": manager.task_classes,
                "tasks": tasks_out,
                "summary": {
                    "final_seen_avg_accuracy": seen_avg_accuracy,
                    "final_seen_avg_loss": seen_avg_loss,
                    "final_mean_forgetting": mean_forgetting,
                    "diagnostics_artifact": "results_diagnostics.json",
                },
                "behavior_over_stages": behavior_over_stages,
                "step_eval": existing_step_eval,
                "diagnostics_over_stages": diagnostics_over_stages,
                "eval": {
                    "checkpoint_path": str(checkpoint_path),
                    "eval_task_id": eval_task_id,
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                },
            },
            f,
            indent=2,
        )

    _save_run_artifacts(args, run_dir, model)
    print(f"[Eval] checkpoint: {checkpoint_path}")
    print(f"[Eval] run dir: {run_dir}")
    print(f"[Eval] seen tasks: {seen_tasks}")
    print(f"[Eval] seen_avg_accuracy={seen_avg_accuracy:.2f}%")
    print(f"[Eval] mean_forgetting={mean_forgetting:.2f}%")


def _resolve_runtime(args: argparse.Namespace) -> Tuple[torch.device, str, int]:
    requested = args.device.lower()

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if requested == "auto":
        if cuda_ok:
            return torch.device("cuda"), "gpu", 1
        if mps_ok:
            return torch.device("mps"), "mps", 1
        return torch.device("cpu"), "cpu", 1

    if requested == "cuda":
        if not cuda_ok:
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda"), "gpu", 1

    if requested == "mps":
        if not mps_ok:
            raise RuntimeError("--device mps requested, but MPS is not available.")
        return torch.device("mps"), "mps", 1

    if requested == "cpu":
        return torch.device("cpu"), "cpu", 1

    raise ValueError(f"Unsupported device option: {args.device}")


def _resolve_precision(precision: str):
    value = precision.strip().lower()
    if value in {"16", "32", "64"}:
        return int(value)
    return precision


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    common = dict(
        num_classes=10,
        img_size=32,
        proj_dim=args.proj_dim,
        codebook_size=args.codebook_size,
        codebook_momentum=args.prototype_momentum,
        patch_prototype_mode=args.patch_prototype_mode,
        patch_proto_sharpness=args.patch_proto_sharpness,
        backbone_mode=args.backbone,
        roi_min_scale=args.roi_min_scale,
        roi_max_scale=args.roi_max_scale,
        roi_prob=args.roi_prob,
    )
    return PrototypePatchBackbone(**common).to(device)


def train_linear_eval(
    model: PrototypePatchBackbone,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int = 3,
    lr: float = 0.1,
    max_batches: Optional[int] = None,
) -> float:
    """Linear probe on frozen global features from the current backbone."""
    model = model.to(device)
    model.eval()
    linear = nn.Linear(model.head[-1].out_features, num_classes).to(device)
    optimizer = optim.SGD(linear.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        linear.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                feat = model.extract_global_feature(images)

            logits = linear(feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    linear.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feat = model.extract_global_feature(images)
            pred = linear(feat).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    return 100.0 * correct / max(1, total)


def _eval_linear_on_loader(
    model: PrototypePatchBackbone,
    linear: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    if linear is not None:
        linear.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = init_confusion_matrix(num_classes)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if linear is None:
                out = model(images)
                bsz = images.size(0)
                num_patches = int(model.num_patches)
                logits = out["logits"].view(bsz, num_patches, -1).mean(dim=1)
            else:
                feat = model.extract_global_feature(images)
                logits = linear(feat)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_samples += int(batch_size)
            update_confusion_matrix(confusion, labels, preds)

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    avg_loss = total_loss / max(1, total_samples)
    accuracy = 100.0 * total_correct / max(1, total_samples)
    payload: Dict[str, object] = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "samples": float(total_samples),
    }
    payload["diagnostics"] = summarize_confusion(confusion, overall_accuracy=accuracy)
    return payload


def linear_eval_seen_tasks(
    model: PrototypePatchBackbone,
    seen_train_loader: DataLoader,
    seen_test_loaders: Dict[int, DataLoader],
    seen_tasks: List[int],
    device: torch.device,
    num_classes: int,
    epochs: int = 3,
    lr: float = 0.1,
    max_batches: Optional[int] = None,
    return_diagnostics: bool = False,
) -> Dict[str, object]:
    """Train one linear probe on seen train data, then report per-task test metrics."""
    model = model.to(device)
    model.eval()
    linear = nn.Linear(model.head[-1].out_features, num_classes).to(device)
    optimizer = optim.SGD(linear.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        linear.train()
        for batch_idx, (images, labels) in enumerate(seen_train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                feat = model.extract_global_feature(images)

            logits = linear(feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    per_task_metrics: List[Dict[str, float]] = []
    per_task_diagnostics: List[Dict[str, object]] = []
    weighted_correct = 0.0
    weighted_total = 0.0
    weighted_loss_sum = 0.0

    for seen_task_id in seen_tasks:
        task_metrics = _eval_linear_on_loader(
            model=model,
            linear=linear,
            loader=seen_test_loaders[seen_task_id],
            device=device,
            num_classes=num_classes,
            max_batches=max_batches,
        )
        samples = task_metrics["samples"]
        per_task_metrics.append(
            {
                "task_id": float(seen_task_id),
                "loss": task_metrics["loss"],
                "accuracy": task_metrics["accuracy"],
            }
        )
        weighted_loss_sum += task_metrics["loss"] * samples
        weighted_correct += (task_metrics["accuracy"] / 100.0) * samples
        weighted_total += samples

        if return_diagnostics:
            diagnostics_payload = task_metrics.get("diagnostics")
            if isinstance(diagnostics_payload, dict):
                per_task_diagnostics.append(
                    {
                        "task_id": float(seen_task_id),
                        **diagnostics_payload,
                    }
                )

    seen_avg_loss = weighted_loss_sum / max(1.0, weighted_total)
    seen_avg_accuracy = 100.0 * weighted_correct / max(1.0, weighted_total)

    output: Dict[str, object] = {
        "per_task": per_task_metrics,
        "seen_avg_loss": seen_avg_loss,
        "seen_avg_accuracy": seen_avg_accuracy,
    }
    if return_diagnostics:
        output["per_task_diagnostics"] = per_task_diagnostics
    return output


def _build_stage_diagnostics(
    stage_task_id: int,
    seen_tasks: List[int],
    per_task_eval: List[Dict[str, float]],
    per_task_diagnostics: List[Dict[str, object]],
    seen_avg_accuracy: float,
    num_classes: int,
) -> Dict[str, object]:
    diagnostics_by_task: Dict[int, Dict[str, object]] = {}
    for item in per_task_diagnostics:
        if not isinstance(item, dict):
            continue
        try:
            task_id = int(float(item.get("task_id", -1)))
        except Exception:
            continue
        if task_id < 0:
            continue
        diagnostics_by_task[task_id] = item

    chance_accuracy = 100.0 / max(1.0, float(num_classes))
    per_task_payload: List[Dict[str, object]] = []

    for metric in per_task_eval:
        task_id = int(metric["task_id"])
        diagnostics = diagnostics_by_task.get(task_id, {})
        accuracy = float(metric["accuracy"])

        per_task_payload.append(
            {
                "task_id": task_id,
                "loss": float(metric["loss"]),
                "accuracy": accuracy,
                "forgetting": float(metric["forgetting"]),
                "chance_accuracy": float(diagnostics.get("chance_accuracy", chance_accuracy)),
                "is_near_chance": bool(diagnostics.get("is_near_chance", accuracy <= chance_accuracy + 2.0)),
                "prediction_collapse_flag": bool(diagnostics.get("prediction_collapse_flag", False)),
                "dominant_predicted_class": diagnostics.get("dominant_predicted_class", None),
                "max_predicted_class_ratio": float(diagnostics.get("max_predicted_class_ratio", 0.0)),
                "worst_classes": diagnostics.get("worst_classes", []),
                "top_confusions": diagnostics.get("top_confusions", []),
                "per_class_accuracy": diagnostics.get("per_class_accuracy", []),
                "per_class_support": diagnostics.get("per_class_support", []),
                "confusion_matrix": diagnostics.get("confusion_matrix", []),
            }
        )

    problematic_tasks = [item["task_id"] for item in per_task_payload if item["is_near_chance"]]
    return {
        "stage_task_id": int(stage_task_id),
        "seen_tasks": list(seen_tasks),
        "seen_avg_accuracy": float(seen_avg_accuracy),
        "chance_accuracy": chance_accuracy,
        "problematic_tasks": problematic_tasks,
        "per_task": per_task_payload,
    }


def _format_behavior_line(current_task: int, per_task_eval: List[Dict[str, float]]) -> str:
    parts = []
    for metric in per_task_eval:
        parts.append(
            f"T{int(metric['task_id'])}: acc={metric['accuracy']:.2f}% "
            f"loss={metric['loss']:.4f} fg={metric['forgetting']:.2f}%"
        )
    joined = " | ".join(parts)
    return f"[Behavior] stage task {current_task} -> {joined}"


class ContinualLightningModule(LightningModuleBase):
    """Single-task training module used in a task-by-task continual loop."""

    def __init__(
        self,
        model: PrototypePatchBackbone,
        old_model: Optional[PrototypePatchBackbone],
        args: argparse.Namespace,
        seen_tasks: Optional[List[int]] = None,
        seen_test_loaders: Optional[Dict[int, DataLoader]] = None,
        lit_logger: Optional[LitLogger] = None,
        stage_task_id: int = 0,
        step_offset: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.old_model = old_model
        self.args = args
        self.seen_tasks = list(seen_tasks) if seen_tasks is not None else []
        self.seen_test_loaders = seen_test_loaders if seen_test_loaders is not None else {}
        self.lit_logger = lit_logger
        self.stage_task_id = stage_task_id
        self.step_offset = step_offset

        self.criterion_nce = ISSupConLoss(temperature=args.nce_temp)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.latest_epoch_loss = 0.0
        self.best_step_acc_by_task: Dict[int, float] = {}
        self.step_eval_records: List[Dict[str, float]] = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            images, labels, replay_flags = batch
            replay_mask = replay_flags.to(device=images.device, non_blocking=True).bool()
        else:
            images, labels = batch
            replay_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=images.device)

        out = self.model(images)
        num_patches = self.model.num_patches
        labels_patch = labels.repeat_interleave(num_patches)
        patch_indices = torch.arange(num_patches, device=images.device).repeat(images.size(0))

        importance_weight = torch.ones_like(labels_patch, dtype=torch.float, device=images.device)
        index = torch.arange(labels_patch.size(0), device=images.device)
        score_mask = torch.ones_like(labels_patch, dtype=torch.bool, device=images.device)

        loss_nce = self.criterion_nce(
            out["proj"],
            labels_patch,
            importance_weight=importance_weight,
            index=index,
            score_mask=score_mask,
        )

        logits_image = out["logits"].view(images.size(0), num_patches, -1).mean(dim=1)
        loss_ce = self.criterion_ce(logits_image, labels)

        loss_distill = torch.tensor(0.0, device=images.device)
        # Apply PPRD distillation on the full batch (not only replay samples).
        if self.old_model is not None:
            with torch.no_grad():
                old_out = self.old_model(images)

            # patch embeddings: [B, N, D]
            patch_embeds_cur = out["proj"].view(images.size(0), num_patches, -1)
            patch_embeds_old = old_out["proj"].view(images.size(0), num_patches, -1)

            # active prototypes from both models: [K, D]
            prototypes_cur = self.model.get_active_prototypes()
            prototypes_old = self.old_model.get_active_prototypes()

            # ensure prototypes are on the same device/dtype as embeddings
            prototypes_cur = prototypes_cur.to(images.device, dtype=patch_embeds_cur.dtype)
            prototypes_old = prototypes_old.to(images.device, dtype=patch_embeds_old.dtype)

            loss_distill = prd_loss(
                patch_embeds_cur=patch_embeds_cur,
                patch_embeds_old=patch_embeds_old,
                prototypes_cur=prototypes_cur,
                prototypes_old=prototypes_old,
                current_temp=self.args.current_temp,
                past_temp=self.args.past_temp,
            )

        self.model.update_codebook(
            out["proj"],
            labels_patch,
            patch_logits=out.get("logits", None),
            patch_indices=patch_indices,
        )

        total_loss = (
            self.args.lambda_nce * loss_nce
            + self.args.lambda_patch_ce * loss_ce
            + self.args.lambda_prd * loss_distill
        )

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_nce", loss_nce, on_step=True, on_epoch=True)
        self.log("train/loss_ce", loss_ce, on_step=True, on_epoch=True)
        self.log("train/loss_prd", loss_distill, on_step=True, on_epoch=True)
        self.log("train/loss_pprd", loss_distill, on_step=True, on_epoch=True)
        return total_loss

    def _eval_current_model_on_loader(
        self,
        loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Tuple[float, float, int]:
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                out = self.model(images)
                logits_image = out["logits"].view(images.size(0), self.model.num_patches, -1).mean(dim=1)
                loss = criterion(logits_image, labels)
                total_loss += float(loss.item()) * int(labels.size(0))
                total_correct += int((logits_image.argmax(dim=1) == labels).sum().item())
                total_samples += int(labels.size(0))
                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break

        avg_loss = total_loss / max(1, total_samples)
        acc = 100.0 * total_correct / max(1, total_samples)
        return avg_loss, acc, total_samples

    def _run_step_eval(self) -> None:
        if not self.seen_tasks or not self.seen_test_loaders:
            return

        was_training = self.model.training
        self.model.eval()

        per_task = []
        weighted_loss = 0.0
        weighted_acc = 0.0
        weighted_total = 0
        max_batches = self.args.max_step_eval_batches if self.args.max_step_eval_batches is not None else None

        for seen_task_id in self.seen_tasks:
            loader = self.seen_test_loaders[seen_task_id]
            avg_loss, acc, samples = self._eval_current_model_on_loader(loader, max_batches=max_batches)
            previous_best = self.best_step_acc_by_task.get(seen_task_id, acc)
            forgetting = max(0.0, previous_best - acc)
            self.best_step_acc_by_task[seen_task_id] = max(previous_best, acc)

            per_task.append(
                {
                    "stage_task_id": float(self.stage_task_id),
                    "global_step": float(self.step_offset + self.global_step),
                    "seen_task_id": float(seen_task_id),
                    "loss": avg_loss,
                    "accuracy": acc,
                    "forgetting": forgetting,
                }
            )

            weighted_loss += avg_loss * samples
            weighted_acc += (acc / 100.0) * samples
            weighted_total += samples

        seen_avg_loss = weighted_loss / max(1, weighted_total)
        seen_avg_acc = 100.0 * weighted_acc / max(1, weighted_total)
        mean_forgetting = sum(item["forgetting"] for item in per_task) / max(1, len(per_task))

        for item in per_task:
            self.step_eval_records.append(item)
            self.log(
                f"step_eval/task_{int(item['seen_task_id'])}/accuracy",
                item["accuracy"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"step_eval/task_{int(item['seen_task_id'])}/forgetting",
                item["forgetting"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        self.log("step_eval/seen_avg_accuracy", seen_avg_acc, on_step=True, on_epoch=False, logger=True)
        self.log("step_eval/seen_avg_loss", seen_avg_loss, on_step=True, on_epoch=False, logger=True)
        self.log("step_eval/mean_forgetting", mean_forgetting, on_step=True, on_epoch=False, logger=True)

        if self.lit_logger is not None:
            self.lit_logger.log_metrics(
                step=self.step_offset + self.global_step,
                metrics={
                    "step_eval/seen_avg_accuracy": seen_avg_acc,
                    "step_eval/seen_avg_loss": seen_avg_loss,
                    "step_eval/mean_forgetting": mean_forgetting,
                },
            )
            for item in per_task:
                seen_task_id = int(item["seen_task_id"])
                self.lit_logger.log_metrics(
                    step=self.step_offset + self.global_step,
                    metrics={
                        f"step_eval/task_{seen_task_id}/accuracy": item["accuracy"],
                        f"step_eval/task_{seen_task_id}/loss": item["loss"],
                        f"step_eval/task_{seen_task_id}/forgetting": item["forgetting"],
                    },
                )

        if was_training:
            self.model.train()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        step_eval_every = max(0, int(self.args.step_eval_every))
        if step_eval_every <= 0:
            return
        if self.global_step == 0:
            return
        if self.global_step % step_eval_every != 0:
            return
        self._run_step_eval()

    def on_train_epoch_end(self) -> None:
        metric = self.trainer.callback_metrics.get("train/loss_epoch")
        if metric is not None:
            self.latest_epoch_loss = float(metric.detach().cpu().item())

    def configure_optimizers(self):
        return optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=0.9,
            weight_decay=self.args.weight_decay,
        )


def _build_loggers(args: argparse.Namespace, run_dir: Path, run_name: str, run_id: str):
    loggers: List[object] = []

    if args.enable_csv:
        if CSVLogger is None:
            raise ImportError("CSVLogger unavailable. Install pytorch-lightning.")
        loggers.append(CSVLogger(save_dir=str(run_dir), name="csv_logs"))

    if args.enable_tb:
        if TensorBoardLogger is None:
            raise ImportError(
                "TensorBoardLogger unavailable. Install pytorch-lightning and tensorboard."
            )
        loggers.append(TensorBoardLogger(save_dir=str(run_dir), name="tb_logs"))

    if not args.use_wandb:
        return loggers

    if WandbLogger is None:
        raise ImportError(
            "pytorch_lightning WandbLogger is unavailable. Install wandb and pytorch-lightning first."
        )
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None
    loggers.append(
        WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or run_name,
        version=run_id,
        id=run_id,
        save_dir=str(run_dir),
        offline=args.wandb_offline,
        tags=tags,
        log_model=False,
    )
    )
    return loggers


def run_training(args: argparse.Namespace) -> None:
    if pl is None:
        raise ImportError(
            "pytorch_lightning is not installed. Install it with: pip install pytorch-lightning"
        )

    if load_dotenv is not None:
        load_dotenv()

    set_seed(args.seed)
    device, accelerator, devices = _resolve_runtime(args)

    total_tasks = 5
    task_order = _parse_task_order(args.task_order, total_tasks)

    manager = SplitCIFAR10Manager(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay_size=args.replay_size,
        tasks=total_tasks,
        classes_per_task=2,
        seed=args.seed,
        task_order=task_order,
    )

    model = _build_model(args, device)

    run_name, run_id = _build_run_name_and_id(args)
    run_dir = _make_run_dir(args, run_name)
    _save_run_artifacts(args, run_dir, model)
    lit_logger = LitLogger(run_dir=run_dir)
    all_loggers = _build_loggers(args, run_dir, run_name=run_name, run_id=run_id)
    lightning_precision = _resolve_precision(args.precision)
    log_fp = open(run_dir / "training.log", "w", encoding="utf-8")
    step_eval_results: List[Dict[str, float]] = []
    step_offset = 0

    def log(message: str) -> None:
        print(message)
        log_fp.write(message + "\n")
        log_fp.flush()
        lit_logger.log_text(message)

    log(f"Run directory: {run_dir}")
    log(f"Run name: {run_name}")
    log(f"Run id: {run_id}")
    log(f"Task classes: {manager.task_classes}")
    log(f"Device: {device}")
    log(f"Lightning accelerator={accelerator}, devices={devices}, precision={lightning_precision}")
    lit_logger.log_event(
        "runtime",
        {
            "device": str(device),
            "accelerator": accelerator,
            "devices": devices,
            "precision": args.precision,
            "run_name": run_name,
            "run_id": run_id,
            "task_classes": manager.task_classes,
        },
    )

    task_results = []
    seen_tasks: List[int] = []
    best_acc_by_task: Dict[int, float] = {}
    behavior_over_stages: List[Dict[str, object]] = []
    diagnostics_over_stages: List[Dict[str, object]] = []

    for task_id in range(manager.tasks):
        model = model.to(device)
        model.train()
        seen_tasks_for_stage = list(range(task_id + 1))
        seen_test_loaders_for_stage = {
            seen_task_id: manager.get_task_test_loader(seen_task_id, batch_size=args.batch_size)
            for seen_task_id in seen_tasks_for_stage
        }
        old_model = None
        if task_id > 0:
            old_model = copy.deepcopy(model).to(device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        train_loader = manager.get_task_train_loader(task_id)

        lit_module = ContinualLightningModule(
            model=model,
            old_model=old_model,
            args=args,
            seen_tasks=seen_tasks_for_stage,
            seen_test_loaders=seen_test_loaders_for_stage,
            lit_logger=lit_logger,
            stage_task_id=task_id,
            step_offset=step_offset,
        )

        callbacks = []
        if LearningRateMonitor is not None:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        if ModelCheckpoint is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=run_dir / "checkpoints",
                    filename=f"task_{task_id}" + "-{epoch:02d}",
                    monitor="train/loss_epoch",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    auto_insert_metric_name=False,
                )
            )

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=accelerator,
            devices=devices,
            logger=all_loggers if all_loggers else None,
            default_root_dir=str(run_dir),
            enable_checkpointing=True,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            deterministic=True,
            num_sanity_val_steps=0,
            callbacks=callbacks,
            precision=lightning_precision,
            limit_train_batches=args.max_train_batches if args.max_train_batches is not None else 1.0,
        )
        trainer.fit(lit_module, train_dataloaders=train_loader)
        step_offset += int(trainer.global_step)
        if lit_module.step_eval_records:
            step_eval_results.extend(lit_module.step_eval_records)
        # Lightning may move module weights back to CPU after fit; force target runtime device.
        model = lit_module.model.to(device)
        avg_loss = lit_module.latest_epoch_loss
        log(f"Task {task_id} | Epochs: {args.epochs} | Avg Loss: {avg_loss:.4f}")

        manager.update_replay_from_task(task_id)

        seen_tasks.append(task_id)
        seen_train_loader = manager.get_seen_train_loader(task_id, batch_size=args.batch_size)
        seen_test_loaders = {
            seen_task_id: manager.get_task_test_loader(seen_task_id, batch_size=args.batch_size)
            for seen_task_id in seen_tasks
        }
        eval_summary = linear_eval_seen_tasks(
            model=model,
            seen_train_loader=seen_train_loader,
            seen_test_loaders=seen_test_loaders,
            seen_tasks=seen_tasks,
            device=device,
            num_classes=10,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
            max_batches=args.max_eval_batches,
            return_diagnostics=True,
        )

        per_task_eval = []
        for metric in eval_summary["per_task"]:
            seen_task_id = int(metric["task_id"])
            acc = float(metric["accuracy"])
            loss = float(metric["loss"])
            previous_best = best_acc_by_task.get(seen_task_id, acc)
            forgetting = max(0.0, previous_best - acc)
            best_acc_by_task[seen_task_id] = max(previous_best, acc)
            per_task_eval.append(
                {
                    "task_id": seen_task_id,
                    "loss": loss,
                    "accuracy": acc,
                    "forgetting": forgetting,
                }
            )

        raw_task_diagnostics = eval_summary.get("per_task_diagnostics", [])
        if not isinstance(raw_task_diagnostics, list):
            raw_task_diagnostics = []

        seen_avg_accuracy = float(eval_summary["seen_avg_accuracy"])
        seen_avg_loss = float(eval_summary["seen_avg_loss"])
        mean_forgetting = (
            float(sum(m["forgetting"] for m in per_task_eval) / len(per_task_eval))
            if per_task_eval
            else 0.0
        )
        stage_diagnostics = _build_stage_diagnostics(
            stage_task_id=task_id,
            seen_tasks=seen_tasks,
            per_task_eval=per_task_eval,
            per_task_diagnostics=raw_task_diagnostics,
            seen_avg_accuracy=seen_avg_accuracy,
            num_classes=10,
        )
        diagnostics_over_stages.append(stage_diagnostics)

        behavior_over_stages.append(
            {
                "stage_task_id": task_id,
                "seen_tasks": list(seen_tasks),
                "per_task": [dict(metric) for metric in per_task_eval],
            }
        )

        for metric in per_task_eval:
            log(
                f"[Seen Task Eval] after task {task_id} | task {metric['task_id']} "
                f"| loss={metric['loss']:.4f} | acc={metric['accuracy']:.2f}% "
                f"| forgetting={metric['forgetting']:.2f}%"
            )

        log(
            f"[Seen Summary] after task {task_id} | avg_loss={seen_avg_loss:.4f} "
            f"| avg_acc={seen_avg_accuracy:.2f}% | mean_forgetting={mean_forgetting:.2f}%"
        )
        log(_format_behavior_line(task_id, per_task_eval))

        task_result = {
            "task_id": task_id,
            "avg_train_loss": avg_loss,
            "seen_tasks": list(seen_tasks),
            "seen_avg_loss": seen_avg_loss,
            "seen_avg_accuracy": seen_avg_accuracy,
            "mean_forgetting": mean_forgetting,
            "problematic_tasks": stage_diagnostics["problematic_tasks"],
            "seen_task_metrics": per_task_eval,
        }
        task_results.append(task_result)

        lit_logger.log_metrics(
            step=task_id,
            metrics={
                "task/avg_train_loss": avg_loss,
                "task/seen_avg_loss": seen_avg_loss,
                "task/seen_avg_accuracy": seen_avg_accuracy,
                "task/mean_forgetting": mean_forgetting,
            },
        )

        for metric in per_task_eval:
            lit_logger.log_metrics(
                step=task_id,
                metrics={
                    f"eval/task_{metric['task_id']}/loss": metric["loss"],
                    f"eval/task_{metric['task_id']}/accuracy": metric["accuracy"],
                    f"eval/task_{metric['task_id']}/forgetting": metric["forgetting"],
                },
            )

        for logger in all_loggers:
            if isinstance(logger, WandbLogger):
                payload = {
                    "task/avg_train_loss": avg_loss,
                    "task/seen_avg_loss": seen_avg_loss,
                    "task/seen_avg_accuracy": seen_avg_accuracy,
                    "task/mean_forgetting": mean_forgetting,
                    "task/id": task_id,
                }
                for metric in per_task_eval:
                    payload[f"eval/task_{metric['task_id']}/loss"] = metric["loss"]
                    payload[f"eval/task_{metric['task_id']}/accuracy"] = metric["accuracy"]
                    payload[f"eval/task_{metric['task_id']}/forgetting"] = metric["forgetting"]
                logger.log_metrics(payload)

        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run": {"name": run_name, "id": run_id},
                    "task_classes": manager.task_classes,
                    "tasks": task_results,
                    "summary": {
                        "final_seen_avg_accuracy": seen_avg_accuracy,
                        "final_seen_avg_loss": seen_avg_loss,
                        "final_mean_forgetting": mean_forgetting,
                        "diagnostics_artifact": "results_diagnostics.json",
                    },
                    "behavior_over_stages": behavior_over_stages,
                    "step_eval": step_eval_results,
                    "diagnostics_over_stages": diagnostics_over_stages,
                },
                f,
                indent=2,
            )

        with open(run_dir / "results_tasks.json", "w", encoding="utf-8") as f:
            json.dump(task_results, f, indent=2)
        with open(run_dir / "results_step_eval.json", "w", encoding="utf-8") as f:
            json.dump(step_eval_results, f, indent=2)
        with open(run_dir / "results_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics_over_stages, f, indent=2)
        with open(run_dir / "results_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run": {"name": run_name, "id": run_id},
                    "task_classes": manager.task_classes,
                    "final_seen_avg_accuracy": seen_avg_accuracy,
                    "final_seen_avg_loss": seen_avg_loss,
                    "final_mean_forgetting": mean_forgetting,
                    "diagnostics_artifact": "results_diagnostics.json",
                },
                f,
                indent=2,
            )

        torch.save(model.state_dict(), run_dir / f"model_task_{task_id}.pth")
        log(f"[Linear Eval] up to task {task_id}: {seen_avg_accuracy:.2f}%")

    torch.save(model.state_dict(), run_dir / "model_final.pth")
    log("Continual training completed.")

    for logger in all_loggers:
        if isinstance(logger, WandbLogger) and hasattr(logger, "experiment"):
            logger.experiment.finish()

    log_fp.close()
    lit_logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split CIFAR-10 continual learning with unified patch/roi_patch prototype backbone"
    )
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument(
        "--patch-prototype-mode",
        type=str,
        choices=["class_mean_ema", "class_confidence_ema", "class_position_ema"],
        default="class_mean_ema",
    )
    parser.add_argument("--patch-proto-sharpness", type=float, default=1.0)
    parser.add_argument("--backbone", type=str, choices=["patch", "roi_patch"], default="patch")
    parser.add_argument("--roi-min-scale", type=float, default=0.55)
    parser.add_argument("--roi-max-scale", type=float, default=1.0)
    parser.add_argument("--roi-prob", type=float, default=1.0)

    parser.add_argument("--replay-size", type=int, default=500)
    parser.add_argument("--nce-temp", type=float, default=0.07)
    parser.add_argument("--current-temp", type=float, default=1.0)
    parser.add_argument("--past-temp", type=float, default=2.0)

    parser.add_argument("--lambda-patch-ce", type=float, default=1.0)
    parser.add_argument("--lambda-nce", type=float, default=1.0)
    parser.add_argument("--lambda-prd", "--lambda-pprd", dest="lambda_prd", type=float, default=1.0)

    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--linear-lr", type=float, default=0.1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", type=str, default="32")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="pprd")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--enable-csv", action="store_true")
    parser.add_argument("--enable-tb", action="store_true")

    # Useful for automatic smoke testing in constrained environments.
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--step-eval-every", type=int, default=50)
    parser.add_argument("--max-step-eval-batches", type=int, default=5)

    parser.add_argument(
        "--task-order",
        type=str,
        default=None,
        help=(
            "Comma-separated permutation of task indices. "
            "Example: '1,2,3,4,0' starts with classes [2,3], then [4,5], ..."
        ),
    )

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--eval-task-id", type=int, default=None)
    parser.add_argument("--eval-run-dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        evaluate_checkpoint(args)
    else:
        run_training(args)
