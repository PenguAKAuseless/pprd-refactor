from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SupConWrapper(nn.Module):
    """2x2 patch pipeline + ResNet18 + MLP projection + linear classifier."""

    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        proj_dim: int = 128,
        mlp_hidden_dim: int = 512,
        **_: object,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_patches = 4

        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(512, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, proj_dim),
        )
        self.classifier = nn.Linear(proj_dim, num_classes)

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = x.size(0)
        patch_inputs = self._extract_and_upsample_patches(x)  # [B*4, C, H, W]
        feat_512 = self.backbone(patch_inputs)  # [B*4, 512]
        proj_128 = self.head(feat_512)  # [B*4, 128]
        logits = self.classifier(proj_128)  # [B*4, num_classes]

        # Useful for image-level utilities (e.g., linear eval on original B items).
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
        self.eval()
        out = self.forward(x)
        return F.normalize(out["proj_image"], dim=-1)
