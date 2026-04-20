from __future__ import annotations

from typing import Dict, List

import torch


def init_confusion_matrix(num_classes: int) -> torch.Tensor:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return torch.zeros((num_classes, num_classes), dtype=torch.long)


def update_confusion_matrix(
    confusion: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
) -> torch.Tensor:
    if confusion.dim() != 2 or confusion.size(0) != confusion.size(1):
        raise ValueError("confusion must be a square matrix")

    if labels.numel() == 0 or preds.numel() == 0:
        return confusion

    labels_cpu = labels.detach().to(dtype=torch.long).view(-1).cpu()
    preds_cpu = preds.detach().to(dtype=torch.long).view(-1).cpu()

    if labels_cpu.size(0) != preds_cpu.size(0):
        raise ValueError("labels and preds must have the same number of elements")

    num_classes = int(confusion.size(0))
    valid = (
        (labels_cpu >= 0)
        & (labels_cpu < num_classes)
        & (preds_cpu >= 0)
        & (preds_cpu < num_classes)
    )
    if not valid.any():
        return confusion

    packed = labels_cpu[valid] * num_classes + preds_cpu[valid]
    counts = torch.bincount(packed, minlength=num_classes * num_classes)
    confusion += counts.view(num_classes, num_classes)
    return confusion


def summarize_confusion(
    confusion: torch.Tensor,
    overall_accuracy: float,
    chance_tolerance: float = 2.0,
    top_k_confusions: int = 5,
    top_k_worst: int = 3,
) -> Dict[str, object]:
    if confusion.dim() != 2 or confusion.size(0) != confusion.size(1):
        raise ValueError("confusion must be a square matrix")

    num_classes = int(confusion.size(0))
    total = int(confusion.sum().item())
    row_totals = confusion.sum(dim=1)
    col_totals = confusion.sum(dim=0)
    diagonal = confusion.diag()

    per_class_accuracy: List[float | None] = []
    for class_id in range(num_classes):
        support = int(row_totals[class_id].item())
        if support <= 0:
            per_class_accuracy.append(None)
            continue
        cls_acc = 100.0 * float(diagonal[class_id].item()) / float(support)
        per_class_accuracy.append(cls_acc)

    per_class_support = [int(v.item()) for v in row_totals]

    worst_classes = []
    for class_id, cls_acc in enumerate(per_class_accuracy):
        if cls_acc is None:
            continue
        worst_classes.append(
            {
                "class_id": class_id,
                "accuracy": float(cls_acc),
                "support": int(row_totals[class_id].item()),
            }
        )
    worst_classes.sort(key=lambda item: (item["accuracy"], -item["support"], item["class_id"]))
    worst_classes = worst_classes[: max(1, int(top_k_worst))]

    top_confusions = []
    for true_class in range(num_classes):
        support = int(row_totals[true_class].item())
        for pred_class in range(num_classes):
            if true_class == pred_class:
                continue
            count = int(confusion[true_class, pred_class].item())
            if count <= 0:
                continue
            row_rate = 100.0 * float(count) / float(max(1, support))
            top_confusions.append(
                {
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "count": count,
                    "row_rate": row_rate,
                }
            )
    top_confusions.sort(key=lambda item: (-item["count"], -item["row_rate"], item["true_class"], item["pred_class"]))
    top_confusions = top_confusions[: max(1, int(top_k_confusions))]

    chance_accuracy = 100.0 / float(max(1, num_classes))
    near_chance_threshold = chance_accuracy + max(0.0, float(chance_tolerance))

    if total > 0:
        dominant_predicted_class = int(torch.argmax(col_totals).item())
        max_predicted_ratio = 100.0 * float(col_totals.max().item()) / float(total)
    else:
        dominant_predicted_class = None
        max_predicted_ratio = 0.0

    return {
        "num_classes": num_classes,
        "samples": total,
        "chance_accuracy": chance_accuracy,
        "is_near_chance": bool(float(overall_accuracy) <= near_chance_threshold),
        "dominant_predicted_class": dominant_predicted_class,
        "max_predicted_class_ratio": max_predicted_ratio,
        "prediction_collapse_flag": bool(max_predicted_ratio >= 70.0),
        "per_class_accuracy": per_class_accuracy,
        "per_class_support": per_class_support,
        "worst_classes": worst_classes,
        "top_confusions": top_confusions,
        "confusion_matrix": confusion.tolist(),
    }
