from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalPatchExtractor(nn.Module):
    """Extracts 2x2 patches and upsamples each patch to full size."""

    def __init__(self, num_splits: int = 2) -> None:
        super().__init__()
        if num_splits < 1:
            raise ValueError("num_splits must be >= 1")
        self.num_splits = int(num_splits)
        self.num_patches = self.num_splits * self.num_splits

    def _extract_and_upsample_patches(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        if height % self.num_splits != 0 or width % self.num_splits != 0:
            raise ValueError("Input height/width must be divisible by num_splits.")

        patch_h = height // self.num_splits
        patch_w = width // self.num_splits

        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(bsz * self.num_patches, channels, patch_h, patch_w)

        return F.interpolate(patches, size=(height, width), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_inputs = self._extract_and_upsample_patches(x)
        patch_indices = torch.arange(
            self.num_patches, device=x.device, dtype=torch.long
        ).repeat(x.size(0))
        return patch_inputs, patch_indices


class ROIPatchExtractor(NormalPatchExtractor):
    """Applies random ROI crop-and-resize after patch extraction."""

    def __init__(
        self,
        num_splits: int = 2,
        roi_min_scale: float = 0.55,
        roi_max_scale: float = 1.0,
        roi_prob: float = 1.0,
    ) -> None:
        super().__init__(num_splits=num_splits)
        self.roi_min_scale = float(max(0.1, min(roi_min_scale, 1.0)))
        self.roi_max_scale = float(max(self.roi_min_scale, min(roi_max_scale, 1.0)))
        self.roi_prob = float(max(0.0, min(roi_prob, 1.0)))

    def _random_roi_crop_and_resize(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        out = torch.empty_like(x)

        for i in range(batch):
            if torch.rand(1, device=x.device).item() > self.roi_prob:
                out[i] = x[i]
                continue

            scale = (
                torch.empty(1, device=x.device)
                .uniform_(self.roi_min_scale, self.roi_max_scale)
                .item()
            )
            crop_h = max(1, int(height * scale))
            crop_w = max(1, int(width * scale))

            max_top = max(0, height - crop_h)
            max_left = max(0, width - crop_w)
            top = (
                0
                if max_top == 0
                else int(torch.randint(0, max_top + 1, (1,), device=x.device).item())
            )
            left = (
                0
                if max_left == 0
                else int(torch.randint(0, max_left + 1, (1,), device=x.device).item())
            )

            crop = x[i : i + 1, :, top : top + crop_h, left : left + crop_w]
            out[i] = (
                F.interpolate(crop, size=(height, width), mode="bilinear", align_corners=False)
                .squeeze(0)
            )

        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patch_inputs, patch_indices = super().forward(x)
        if self.training and self.roi_prob > 0.0:
            patch_inputs = self._random_roi_crop_and_resize(patch_inputs)
        return patch_inputs, patch_indices
