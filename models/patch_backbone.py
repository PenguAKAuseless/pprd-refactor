from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PrototypePatchBackbone(nn.Module):
    """Shared patch/roi_patch backbone with a single prototype update pipeline."""

    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        proj_dim: int = 128,
        mlp_hidden_dim: int = 512,
        codebook_size: int = 64,
        codebook_momentum: float = 0.9,
        patch_prototype_mode: str = "class_mean_ema",
        patch_proto_sharpness: float = 1.0,
        backbone_mode: str = "patch",
        roi_min_scale: float = 0.55,
        roi_max_scale: float = 1.0,
        roi_prob: float = 1.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_patches = 4
        self.num_classes = num_classes
        self.codebook_size = max(codebook_size, num_classes)
        self.codebook_momentum = codebook_momentum
        self.patch_prototype_mode = patch_prototype_mode
        self.patch_proto_sharpness = max(1e-4, float(patch_proto_sharpness))

        valid_backbone_modes = {"patch", "roi_patch"}
        if backbone_mode not in valid_backbone_modes:
            raise ValueError(
                f"Unsupported backbone_mode: {backbone_mode}. "
                f"Expected one of {sorted(valid_backbone_modes)}"
            )
        self.backbone_mode = backbone_mode

        valid_proto_modes = {
            "class_mean_ema",
            "class_confidence_ema",
            "class_position_ema",
        }
        if self.patch_prototype_mode not in valid_proto_modes:
            raise ValueError(
                f"Unsupported patch_prototype_mode: {self.patch_prototype_mode}. "
                f"Expected one of {sorted(valid_proto_modes)}"
            )

        self.roi_min_scale = float(max(0.1, min(roi_min_scale, 1.0)))
        self.roi_max_scale = float(max(self.roi_min_scale, min(roi_max_scale, 1.0)))
        self.roi_prob = float(max(0.0, min(roi_prob, 1.0)))

        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(512, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

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

    def _extract_and_upsample_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2x2 patches via tensor ops only, then upsample to original HxW."""
        bsz, channels, height, width = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("Input height and width must be even for 2x2 patch split.")

        patch_h = height // 2
        patch_w = width // 2

        # [B, C, 2, 2, H/2, W/2] -> [B*4, C, H/2, W/2]
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(bsz * self.num_patches, channels, patch_h, patch_w)

        return F.interpolate(patches, size=(height, width), mode="bilinear", align_corners=False)

    def _random_roi_crop_and_resize(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        out = torch.empty_like(x)

        for i in range(batch):
            if torch.rand(1, device=x.device).item() > self.roi_prob:
                out[i] = x[i]
                continue

            scale = torch.empty(1, device=x.device).uniform_(self.roi_min_scale, self.roi_max_scale).item()
            crop_h = max(1, int(height * scale))
            crop_w = max(1, int(width * scale))

            max_top = max(0, height - crop_h)
            max_left = max(0, width - crop_w)
            top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, (1,), device=x.device).item())
            left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, (1,), device=x.device).item())

            crop = x[i : i + 1, :, top : top + crop_h, left : left + crop_w]
            out[i] = F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False).squeeze(0)

        return out

    def _apply_patch_transform(self, patch_inputs: torch.Tensor) -> torch.Tensor:
        if self.backbone_mode != "roi_patch":
            return patch_inputs
        if not self.training or self.roi_prob <= 0.0:
            return patch_inputs
        return self._random_roi_crop_and_resize(patch_inputs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = x.size(0)
        patch_inputs = self._extract_and_upsample_patches(x)
        patch_inputs = self._apply_patch_transform(patch_inputs)

        feat_512 = self.backbone(patch_inputs)
        proj_128 = self.head(feat_512)
        logits = self.classifier(proj_128)

        proj_image = proj_128.view(bsz, self.num_patches, -1).mean(dim=1)

        return {
            "patch_inputs": patch_inputs,
            "feat_512": feat_512,
            "proj": proj_128,
            "logits": logits,
            "proj_image": proj_image,
            "num_patches": torch.tensor(self.num_patches, device=x.device),
        }

    @torch.no_grad()
    def extract_global_feature(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        if was_training:
            self.eval()
        out = self.forward(x)
        if was_training:
            self.train()
        return F.normalize(out["proj_image"], dim=-1)

    @torch.no_grad()
    def update_codebook(
        self,
        patch_proj: torch.Tensor,
        patch_labels: torch.Tensor,
        patch_logits: torch.Tensor | None = None,
        patch_indices: torch.Tensor | None = None,
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
                torch.arange(patch_proj.size(0), device=patch_proj.device, dtype=torch.long) % self.num_patches
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

        # Only classes present in this batch are updated. Unseen classes are untouched.
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
                # Ensure the prototype remains on the unit hypersphere after EMA.
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
        """Return prototype rows restricted to the given class ids.

        Class-level modes return [len(class_ids), D]. class_position_ema returns
        [len(class_ids) * num_patches, D] with rows grouped per class.
        """
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


class SupConWrapper(PrototypePatchBackbone):
    """Backward-compatible name for patch mode."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("backbone_mode", "patch")
        super().__init__(*args, **kwargs)


class ROIPatchWrapper(PrototypePatchBackbone):
    """Backward-compatible name for roi_patch mode."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["backbone_mode"] = "roi_patch"
        super().__init__(*args, **kwargs)
