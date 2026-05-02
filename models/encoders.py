from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Encoder(nn.Module):
    """ResNet18 feature extractor that outputs a 512-d vector per image."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.out_dim = 512

    def forward(self, x):
        return self.backbone(x)


def build_encoder(name: str = "resnet18") -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return ResNet18Encoder()
    raise ValueError(f"Unsupported encoder: {name}")
