from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import ResNet18Encoder
from models.extractors import NormalPatchExtractor, ROIPatchExtractor
from models.heads import (
    EMACodebook,
    ETF_Classifier,
    FixedCodebook,
    LinearClassifier,
    MLPProjectionHead,
    VALID_PATCH_PROTO_MODES,
)


class PrototypePatchBackbone(nn.Module):
    """Composable patch/roi_patch backbone with pluggable encoder/extractor/head/codebook."""

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
        classifier_type: str = "etf",
        codebook_mode: str = "ema",
        etf_scale: float = 10.0,
        etf_learnable_scale: bool = False,
        etf_seed: Optional[int] = 0,
        fixed_codebook_init: str = "random",
        fixed_codebook_seed: Optional[int] = 0,
        encoder: Optional[nn.Module] = None,
        extractor: Optional[nn.Module] = None,
        projection_head: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
        codebook: Optional[nn.Module] = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.num_classes = int(num_classes)
        self.proj_dim = int(proj_dim)

        if patch_prototype_mode not in VALID_PATCH_PROTO_MODES:
            raise ValueError(
                f"Unsupported patch_prototype_mode: {patch_prototype_mode}. "
                f"Expected one of {sorted(VALID_PATCH_PROTO_MODES)}"
            )
        self.patch_prototype_mode = patch_prototype_mode

        valid_backbone_modes = {"patch", "roi_patch"}
        if backbone_mode not in valid_backbone_modes:
            raise ValueError(
                f"Unsupported backbone_mode: {backbone_mode}. Expected one of {sorted(valid_backbone_modes)}"
            )
        self.backbone_mode = backbone_mode

        if encoder is None:
            encoder = ResNet18Encoder()
        self.backbone = encoder
        encoder_out_dim = getattr(encoder, "out_dim", None)
        if encoder_out_dim is None:
            raise ValueError("Encoder must expose out_dim for projection head sizing.")

        if extractor is None:
            if backbone_mode == "patch":
                extractor = NormalPatchExtractor(num_splits=2)
            else:
                extractor = ROIPatchExtractor(
                    num_splits=2,
                    roi_min_scale=roi_min_scale,
                    roi_max_scale=roi_max_scale,
                    roi_prob=roi_prob,
                )
        self.extractor = extractor
        self.num_patches = int(getattr(self.extractor, "num_patches", 4))

        if projection_head is None:
            projection_head = MLPProjectionHead(
                in_dim=encoder_out_dim,
                hidden_dim=mlp_hidden_dim,
                out_dim=self.proj_dim,
            )
        self.head = projection_head

        if classifier is None:
            classifier_type = classifier_type.lower()
            if classifier_type == "etf":
                classifier = ETF_Classifier(
                    in_dim=self.proj_dim,
                    num_classes=self.num_classes,
                    scale=etf_scale,
                    learnable_scale=etf_learnable_scale,
                    seed=etf_seed,
                )
            elif classifier_type == "linear":
                classifier = LinearClassifier(in_dim=self.proj_dim, num_classes=self.num_classes)
            else:
                raise ValueError("classifier_type must be one of: etf, linear")
        self.classifier = classifier

        if codebook is None:
            codebook_mode = codebook_mode.lower()
            if codebook_mode == "ema":
                codebook = EMACodebook(
                    num_classes=self.num_classes,
                    proj_dim=self.proj_dim,
                    codebook_size=codebook_size,
                    num_patches=self.num_patches,
                    codebook_momentum=codebook_momentum,
                    patch_prototype_mode=patch_prototype_mode,
                    patch_proto_sharpness=patch_proto_sharpness,
                )
            elif codebook_mode == "fixed":
                codebook = FixedCodebook(
                    num_classes=self.num_classes,
                    proj_dim=self.proj_dim,
                    codebook_size=codebook_size,
                    num_patches=self.num_patches,
                    patch_prototype_mode=patch_prototype_mode,
                    init_mode=fixed_codebook_init,
                    init_seed=fixed_codebook_seed,
                )
            else:
                raise ValueError("codebook_mode must be one of: ema, fixed")
        self.codebook = codebook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz = x.size(0)
        patch_inputs, patch_indices = self.extractor(x)
        feat_512 = self.backbone(patch_inputs)
        proj_128 = self.head(feat_512)
        logits = self.classifier(proj_128)

        proj_image = proj_128.view(bsz, self.num_patches, -1).mean(dim=1)

        return {
            "patch_inputs": patch_inputs,
            "patch_indices": patch_indices,
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
        patch_logits: Optional[torch.Tensor] = None,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.codebook.update_codebook(
            patch_proj=patch_proj,
            patch_labels=patch_labels,
            patch_logits=patch_logits,
            patch_indices=patch_indices,
        )

    def get_active_prototypes(self) -> torch.Tensor:
        return self.codebook.get_active_prototypes()

    def get_active_prototypes_for_classes(self, class_ids) -> torch.Tensor:
        return self.codebook.get_active_prototypes_for_classes(class_ids)


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
