from __future__ import annotations

from typing import Optional

from models.encoders import ResNet18Encoder, build_encoder
from models.extractors import NormalPatchExtractor, ROIPatchExtractor
from models.heads import (
    EMACodebook,
    ETF_Classifier,
    FixedCodebook,
    LinearClassifier,
    MLPProjectionHead,
)
from models.patch_backbone import PrototypePatchBackbone


def build_patch_model(
    *,
    num_classes: int,
    img_size: int = 32,
    proj_dim: int = 128,
    mlp_hidden_dim: int = 512,
    encoder_name: str = "resnet18",
    patch_mode: str = "normal",
    codebook_mode: str = "ema",
    patch_prototype_mode: str = "class_mean_ema",
    codebook_size: int = 64,
    prototype_momentum: float = 0.9,
    patch_proto_sharpness: float = 1.0,
    roi_min_scale: float = 0.55,
    roi_max_scale: float = 1.0,
    roi_prob: float = 1.0,
    classifier_type: str = "etf",
    etf_scale: float = 10.0,
    etf_learnable_scale: bool = False,
    etf_seed: Optional[int] = 0,
    fixed_codebook_init: str = "random",
    fixed_codebook_seed: Optional[int] = 0,
) -> PrototypePatchBackbone:
    normalized_patch_mode = patch_mode.strip().lower()
    if normalized_patch_mode in {"patch", "normal"}:
        normalized_patch_mode = "normal"
    elif normalized_patch_mode in {"roi", "roi_patch"}:
        normalized_patch_mode = "roi"
    else:
        raise ValueError("patch_mode must be one of: patch, normal, roi, roi_patch")

    encoder = build_encoder(encoder_name)
    if not isinstance(encoder, ResNet18Encoder):
        encoder_out_dim = getattr(encoder, "out_dim", None)
    else:
        encoder_out_dim = encoder.out_dim
    if encoder_out_dim is None:
        raise ValueError("Encoder must expose out_dim for projection head sizing.")

    if normalized_patch_mode == "normal":
        extractor = NormalPatchExtractor(num_splits=2)
    else:
        extractor = ROIPatchExtractor(
            num_splits=2,
            roi_min_scale=roi_min_scale,
            roi_max_scale=roi_max_scale,
            roi_prob=roi_prob,
        )

    projection_head = MLPProjectionHead(
        in_dim=encoder_out_dim,
        hidden_dim=mlp_hidden_dim,
        out_dim=proj_dim,
    )

    normalized_classifier = classifier_type.strip().lower()
    if normalized_classifier == "etf":
        classifier = ETF_Classifier(
            in_dim=proj_dim,
            num_classes=num_classes,
            scale=etf_scale,
            learnable_scale=etf_learnable_scale,
            seed=etf_seed,
        )
    elif normalized_classifier == "linear":
        classifier = LinearClassifier(in_dim=proj_dim, num_classes=num_classes)
    else:
        raise ValueError("classifier_type must be one of: etf, linear")

    normalized_codebook = codebook_mode.strip().lower()
    if normalized_codebook == "ema":
        codebook = EMACodebook(
            num_classes=num_classes,
            proj_dim=proj_dim,
            codebook_size=codebook_size,
            num_patches=extractor.num_patches,
            codebook_momentum=prototype_momentum,
            patch_prototype_mode=patch_prototype_mode,
            patch_proto_sharpness=patch_proto_sharpness,
        )
    elif normalized_codebook == "fixed":
        codebook = FixedCodebook(
            num_classes=num_classes,
            proj_dim=proj_dim,
            codebook_size=codebook_size,
            num_patches=extractor.num_patches,
            patch_prototype_mode=patch_prototype_mode,
            init_mode=fixed_codebook_init,
            init_seed=fixed_codebook_seed,
        )
    else:
        raise ValueError("codebook_mode must be one of: ema, fixed")

    return PrototypePatchBackbone(
        num_classes=num_classes,
        img_size=img_size,
        proj_dim=proj_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        patch_prototype_mode=patch_prototype_mode,
        patch_proto_sharpness=patch_proto_sharpness,
        backbone_mode="patch" if normalized_patch_mode == "normal" else "roi_patch",
        roi_min_scale=roi_min_scale,
        roi_max_scale=roi_max_scale,
        roi_prob=roi_prob,
        encoder=encoder,
        extractor=extractor,
        projection_head=projection_head,
        classifier=classifier,
        codebook=codebook,
    )
