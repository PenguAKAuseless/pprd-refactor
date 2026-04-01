import argparse
import copy
import json
import os
import random
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
from models.patch_backbone import SupConWrapper
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


def _make_run_dir(args: argparse.Namespace) -> Path:
    base = Path(args.log_dir)
    base.mkdir(parents=True, exist_ok=True)
    date_dir = base / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name if args.run_name else f"run_{stamp}"
    run_dir = date_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run_artifacts(args: argparse.Namespace, run_dir: Path, model: nn.Module) -> None:
    command = " ".join(sys.argv)

    (run_dir / "run_command.txt").write_text(command + "\n", encoding="utf-8")
    run_script = "#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n"
    (run_dir / "run.sh").write_text(run_script, encoding="utf-8")

    with open(run_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    (run_dir / "model.txt").write_text(str(model) + "\n", encoding="utf-8")


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


def train_linear_eval(
    model: SupConWrapper,
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


class ContinualLightningModule(LightningModuleBase):
    """Single-task training module used in a task-by-task continual loop."""

    def __init__(
        self,
        model: SupConWrapper,
        old_model: Optional[SupConWrapper],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.model = model
        self.old_model = old_model
        self.args = args

        self.criterion_nce = ISSupConLoss(temperature=args.nce_temp)
        self.latest_epoch_loss = 0.0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        images, labels = batch
        out = self.model(images)
        num_patches = self.model.num_patches
        labels_patch = labels.repeat_interleave(num_patches)

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

        loss_distill = torch.tensor(0.0, device=images.device)
        if self.old_model is not None:
            with torch.no_grad():
                old_out = self.old_model(images)
            loss_distill = prd_loss(
                out["logits"],
                old_out["logits"],
                current_temp=self.args.current_temp,
                past_temp=self.args.past_temp,
            )

        total_loss = loss_nce + loss_distill

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_nce", loss_nce, on_step=True, on_epoch=True)
        self.log("train/loss_distill", loss_distill, on_step=True, on_epoch=True)
        return total_loss

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


def _build_loggers(args: argparse.Namespace, run_dir: Path):
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
        name=args.wandb_name or args.run_name,
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

    manager = SplitCIFAR10Manager(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        replay_size=args.replay_size,
        tasks=5,
        classes_per_task=2,
        seed=args.seed,
    )

    model = SupConWrapper(
        num_classes=10,
        img_size=32,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        proj_dim=args.proj_dim,
        codebook_size=args.codebook_size,
        patch_temp=args.current_temp,
    ).to(device)

    run_dir = _make_run_dir(args)
    _save_run_artifacts(args, run_dir, model)
    lit_logger = LitLogger(run_dir=run_dir)
    all_loggers = _build_loggers(args, run_dir)
    lightning_precision = _resolve_precision(args.precision)
    log_fp = open(run_dir / "training.log", "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_fp.write(message + "\n")
        log_fp.flush()
        lit_logger.log_text(message)

    log(f"Run directory: {run_dir}")
    log(f"Device: {device}")
    log(f"Lightning accelerator={accelerator}, devices={devices}, precision={lightning_precision}")
    lit_logger.log_event(
        "runtime",
        {
            "device": str(device),
            "accelerator": accelerator,
            "devices": devices,
            "precision": args.precision,
        },
    )

    task_results = []

    for task_id in range(5):
        model = model.to(device)
        old_model = None
        if task_id > 0:
            old_model = copy.deepcopy(model).to(device)
            old_model.eval()
            for p in old_model.parameters():
                p.requires_grad = False

        train_loader = manager.get_task_train_loader(task_id)

        lit_module = ContinualLightningModule(model=model, old_model=old_model, args=args)

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
        # Lightning may move module weights back to CPU after fit; force target runtime device.
        model = lit_module.model.to(device)
        avg_loss = lit_module.latest_epoch_loss
        log(f"Task {task_id} | Epochs: {args.epochs} | Avg Loss: {avg_loss:.4f}")

        manager.update_replay_from_task(task_id)

        seen_train_loader = manager.get_seen_train_loader(task_id, batch_size=args.batch_size)
        seen_test_loader = manager.get_seen_test_loader(task_id, batch_size=args.batch_size)
        linear_acc = train_linear_eval(
            model,
            seen_train_loader,
            seen_test_loader,
            device,
            num_classes=10,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
            max_batches=args.max_eval_batches,
        )
        task_result = {
            "task_id": task_id,
            "avg_train_loss": avg_loss,
            "linear_eval_acc": linear_acc,
        }
        task_results.append(task_result)

        lit_logger.log_metrics(
            step=task_id,
            metrics={
                "task/avg_train_loss": avg_loss,
                "task/linear_eval_acc": linear_acc,
            },
        )

        for logger in all_loggers:
            if isinstance(logger, WandbLogger):
                logger.log_metrics(
                    {
                        "task/avg_train_loss": avg_loss,
                        "task/linear_eval_acc": linear_acc,
                        "task/id": task_id,
                    }
                )

        with open(run_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"tasks": task_results}, f, indent=2)

        torch.save(model.state_dict(), run_dir / f"model_task_{task_id}.pth")
        log(f"[Linear Eval] up to task {task_id}: {linear_acc:.2f}%")

    torch.save(model.state_dict(), run_dir / "model_final.pth")
    log("Continual training completed.")

    for logger in all_loggers:
        if isinstance(logger, WandbLogger) and hasattr(logger, "experiment"):
            logger.experiment.finish()

    log_fp.close()
    lit_logger.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split CIFAR-10 continual learning with patch backbone + PPRD")
    parser.add_argument("--data-root", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=64)

    parser.add_argument("--replay-size", type=int, default=500)
    parser.add_argument("--nce-temp", type=float, default=0.07)
    parser.add_argument("--current-temp", type=float, default=0.10)
    parser.add_argument("--past-temp", type=float, default=0.04)

    parser.add_argument("--lambda-patch-ce", type=float, default=1.0)
    parser.add_argument("--lambda-pprd", type=float, default=1.0)

    parser.add_argument("--linear-epochs", type=int, default=2)
    parser.add_argument("--linear-lr", type=float, default=0.1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
