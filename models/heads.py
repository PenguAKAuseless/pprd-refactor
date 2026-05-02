from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


VALID_PATCH_PROTO_MODES = {
    "class_mean_ema",
    "class_confidence_ema",
    "class_position_ema",
}


class MLPProjectionHead(nn.Sequential):
    """Two-layer MLP projection head used before classification/prototypes."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )


def _build_etf_matrix(num_classes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if num_classes <= 1:
        raise ValueError("ETF requires num_classes >= 2")
    eye = torch.eye(num_classes, device=device, dtype=dtype)
    ones = torch.ones((num_classes, num_classes), device=device, dtype=dtype) / float(num_classes)
    etf = eye - ones
    return F.normalize(etf, dim=0)


def build_etf_weights(
    in_dim: int,
    num_classes: int,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if in_dim < num_classes:
        raise ValueError("ETF requires in_dim >= num_classes")
    device = device if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32

    etf = _build_etf_matrix(num_classes=num_classes, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(
        torch.randn(in_dim, num_classes, device=device, dtype=dtype, generator=generator)
    )
    weights = q @ etf
    weights = F.normalize(weights, dim=0)
    return weights.t()


class ETF_Classifier(nn.Module):
    """Fixed ETF classifier with optional learnable logit scale."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale: float = 10.0,
        learnable_scale: bool = False,
        normalize_input: bool = True,
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
        weight = build_etf_weights(in_dim, num_classes, generator=gen)
        self.register_buffer("weight", weight)
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(float(scale)))
        self.normalize_input = bool(normalize_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = F.normalize(x, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        return F.linear(x, weight) * self.logit_scale


class LinearClassifier(nn.Linear):
    """Trainable linear classifier for ablations or debugging."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__(in_dim, num_classes)


class FixedCodebook(nn.Module):
    """Fixed prototypes that never update after initialization."""

    def __init__(
        self,
        num_classes: int,
        proj_dim: int,
        codebook_size: int = 64,
        num_patches: int = 4,
        patch_prototype_mode: str = "class_mean_ema",
        init_mode: str = "random",
        init_seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        if patch_prototype_mode not in VALID_PATCH_PROTO_MODES:
            raise ValueError(
                f"Unsupported patch_prototype_mode: {patch_prototype_mode}. "
                f"Expected one of {sorted(VALID_PATCH_PROTO_MODES)}"
            )
        self.num_classes = int(num_classes)
        self.num_patches = int(num_patches)
        self.patch_prototype_mode = patch_prototype_mode
        self.codebook_size = max(int(codebook_size), self.num_classes)

        gen = None
        if init_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(init_seed))

        if init_mode == "etf" and proj_dim >= self.codebook_size:
            prototypes = build_etf_weights(proj_dim, self.codebook_size, generator=gen)
        else:
            prototypes = torch.randn(self.codebook_size, proj_dim, generator=gen)
            prototypes = F.normalize(prototypes, dim=-1)

        self.register_buffer("prototype_codebook", prototypes)
        self.register_buffer(
            "prototype_pos_codebook",
            prototypes.unsqueeze(1).expand(self.codebook_size, self.num_patches, proj_dim).clone(),
        )

    def update_codebook(
        self,
        patch_proj: torch.Tensor,
        patch_labels: torch.Tensor,
        patch_logits: Optional[torch.Tensor] = None,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> None:
        _ = patch_proj, patch_labels, patch_logits, patch_indices

    def get_active_prototypes(self) -> torch.Tensor:
        if self.patch_prototype_mode != "class_position_ema":
            return self.prototype_codebook[: self.num_classes]
        pos_proto = self.prototype_pos_codebook[: self.num_classes]
        return F.normalize(pos_proto.view(self.num_classes * self.num_patches, -1), dim=-1)

    def get_active_prototypes_for_classes(self, class_ids) -> torch.Tensor:
        proj_dim = self.prototype_codebook.size(-1)
        device = self.prototype_codebook.device
        sorted_ids = sorted({int(c) for c in class_ids if 0 <= int(c) < self.codebook_size})
        if not sorted_ids:
            return self.prototype_codebook.new_zeros((0, proj_dim))

        idx = torch.tensor(sorted_ids, dtype=torch.long, device=device)
        if self.patch_prototype_mode != "class_position_ema":
            return self.prototype_codebook.index_select(0, idx)

        pos_proto = self.prototype_pos_codebook.index_select(0, idx)
        return F.normalize(pos_proto.view(idx.size(0) * self.num_patches, -1), dim=-1)


class EMACodebook(nn.Module):
    """EMA-updated prototypes for patch-to-prototype distillation."""

    def __init__(
        self,
        num_classes: int,
        proj_dim: int,
        codebook_size: int = 64,
        num_patches: int = 4,
        codebook_momentum: float = 0.9,
        patch_prototype_mode: str = "class_mean_ema",
        patch_proto_sharpness: float = 1.0,
    ) -> None:
        super().__init__()
        if patch_prototype_mode not in VALID_PATCH_PROTO_MODES:
            raise ValueError(
                f"Unsupported patch_prototype_mode: {patch_prototype_mode}. "
                f"Expected one of {sorted(VALID_PATCH_PROTO_MODES)}"
            )
        self.num_classes = int(num_classes)
        self.num_patches = int(num_patches)
        self.codebook_size = max(int(codebook_size), self.num_classes)
        self.codebook_momentum = float(codebook_momentum)
        self.patch_prototype_mode = patch_prototype_mode
        self.patch_proto_sharpness = max(1e-4, float(patch_proto_sharpness))

        self.register_buffer("prototype_codebook", torch.zeros(self.codebook_size, proj_dim))
        self.register_buffer("prototype_counts", torch.zeros(self.codebook_size, dtype=torch.long))
        self.register_buffer(
            "prototype_pos_codebook",
            torch.zeros(self.codebook_size, self.num_patches, proj_dim),
        )
        self.register_buffer(
            "prototype_pos_counts",
            torch.zeros(self.codebook_size, self.num_patches, dtype=torch.long),
        )

    @torch.no_grad()
    def update_codebook(
        self,
        patch_proj: torch.Tensor,
        patch_labels: torch.Tensor,
        patch_logits: Optional[torch.Tensor] = None,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> None:
        if patch_proj.dim() != 2:
            raise ValueError("patch_proj must be [B*N, D]")
        if patch_labels.dim() != 1:
            raise ValueError("patch_labels must be [B*N]")
        if patch_proj.size(0) != patch_labels.size(0):
            raise ValueError("patch_proj and patch_labels must have the same first dimension")

        patch_proj = F.normalize(patch_proj.detach(), dim=-1)
        patch_labels = patch_labels.detach().long()

        if patch_indices is None:
            patch_indices = (
                torch.arange(patch_proj.size(0), device=patch_proj.device, dtype=torch.long)
                % self.num_patches
            )
        else:
            patch_indices = patch_indices.detach().long()
            if patch_indices.dim() != 1 or patch_indices.size(0) != patch_proj.size(0):
                raise ValueError("patch_indices must be [B*N]")
            patch_indices = patch_indices.clamp(min=0, max=self.num_patches - 1)

        patch_weights = torch.ones(patch_proj.size(0), device=patch_proj.device)
        if self.patch_prototype_mode == "class_confidence_ema" and patch_logits is not None:
            if patch_logits.dim() != 2 or patch_logits.size(0) != patch_proj.size(0):
                raise ValueError("patch_logits must be [B*N, C]")
            probs = F.softmax(patch_logits.detach() * self.patch_proto_sharpness, dim=-1)
            class_probs = probs.gather(1, patch_labels.view(-1, 1)).squeeze(1)
            patch_weights = class_probs.clamp_min(1e-6)

        present_classes = patch_labels.unique()
        if present_classes.numel() == 0:
            return

        for class_id in present_classes.tolist():
            class_id_int = int(class_id)
            if class_id_int < 0 or class_id_int >= self.codebook_size:
                continue
            class_mask = patch_labels == class_id_int
            if not class_mask.any():
                continue

            if self.patch_prototype_mode == "class_position_ema":
                for pos in range(self.num_patches):
                    pos_mask = class_mask & (patch_indices == pos)
                    if not pos_mask.any():
                        continue
                    pos_proto = patch_proj[pos_mask].mean(dim=0)
                    pos_proto = F.normalize(pos_proto, dim=-1)

                    if self.prototype_pos_counts[class_id_int, pos] == 0:
                        self.prototype_pos_codebook[class_id_int, pos] = pos_proto
                    else:
                        prev_pos = self.prototype_pos_codebook[class_id_int, pos]
                        updated_pos = self.codebook_momentum * prev_pos + (1.0 - self.codebook_momentum) * pos_proto
                        self.prototype_pos_codebook[class_id_int, pos] = F.normalize(updated_pos, dim=-1)
                    self.prototype_pos_counts[class_id_int, pos] += 1

                pos_codes = self.prototype_pos_codebook[class_id_int]
                pos_counts = self.prototype_pos_counts[class_id_int]
                available = pos_counts > 0
                if available.any():
                    class_proto = pos_codes[available].mean(dim=0)
                else:
                    class_proto = patch_proj[class_mask].mean(dim=0)
                class_proto = F.normalize(class_proto, dim=-1)
            else:
                class_patch = patch_proj[class_mask]
                class_weights = patch_weights[class_mask].view(-1, 1)
                class_proto = (class_patch * class_weights).sum(dim=0) / class_weights.sum().clamp_min(1e-6)
                class_proto = F.normalize(class_proto, dim=-1)

            if self.prototype_counts[class_id_int] == 0:
                self.prototype_codebook[class_id_int] = class_proto
            else:
                prev = self.prototype_codebook[class_id_int]
                updated = self.codebook_momentum * prev + (1.0 - self.codebook_momentum) * class_proto
                self.prototype_codebook[class_id_int] = F.normalize(updated, dim=-1)
            self.prototype_counts[class_id_int] += 1

    def get_active_prototypes(self) -> torch.Tensor:
        if self.patch_prototype_mode != "class_position_ema":
            return self.prototype_codebook[: self.num_classes]

        pos_proto = self.prototype_pos_codebook[: self.num_classes].clone()
        pos_counts = self.prototype_pos_counts[: self.num_classes]
        class_proto = self.prototype_codebook[: self.num_classes]
        missing = pos_counts == 0
        if missing.any():
            pos_proto[missing] = class_proto.unsqueeze(1).expand_as(pos_proto)[missing]

        return F.normalize(pos_proto.view(self.num_classes * self.num_patches, -1), dim=-1)

    def get_active_prototypes_for_classes(self, class_ids) -> torch.Tensor:
        proj_dim = self.prototype_codebook.size(-1)
        device = self.prototype_codebook.device
        sorted_ids = sorted({int(c) for c in class_ids if 0 <= int(c) < self.codebook_size})
        if not sorted_ids:
            return self.prototype_codebook.new_zeros((0, proj_dim))

        idx = torch.tensor(sorted_ids, dtype=torch.long, device=device)

        if self.patch_prototype_mode != "class_position_ema":
            return self.prototype_codebook.index_select(0, idx)

        pos_proto = self.prototype_pos_codebook.index_select(0, idx).clone()
        pos_counts = self.prototype_pos_counts.index_select(0, idx)
        class_proto = self.prototype_codebook.index_select(0, idx)
        missing = pos_counts == 0
        if missing.any():
            pos_proto[missing] = class_proto.unsqueeze(1).expand_as(pos_proto)[missing]

        return F.normalize(pos_proto.view(idx.size(0) * self.num_patches, -1), dim=-1)
