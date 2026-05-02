from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISSupConLoss(nn.Module):
    """Importance-scaled supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        importance_weight: Optional[torch.Tensor] = None,
        index: Optional[torch.Tensor] = None,
        score_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError("features must be [B, D]")

        device = features.device
        bsz = features.shape[0]

        features = F.normalize(features, dim=-1)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != bsz:
            raise ValueError("labels length mismatch with features")

        mask = torch.eq(labels, labels.t()).float().to(device)
        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(mask) - torch.eye(bsz, device=device)
        positive_mask = mask * logits_mask

        if importance_weight is not None:
            importance = importance_weight

        if importance is None:
            pair_importance = torch.ones_like(logits)
        else:
            importance = importance.view(-1, 1).to(device)
            pair_importance = (importance + importance.t()) * 0.5

        exp_logits = torch.exp(logits) * logits_mask
        # Importance scales denominator for non-self pairs.
        denom = (exp_logits * pair_importance).sum(dim=1, keepdim=True) + 1e-12

        log_prob = logits - torch.log(denom)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        mean_log_prob_pos = torch.zeros(bsz, device=device)
        mean_log_prob_pos[valid] = (
            (positive_mask[valid] * log_prob[valid]).sum(dim=1) / positive_count[valid]
        )

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # Keep `index` in the signature for compatibility with CL pipelines that
        # pass sample indices from replay buffers.
        _ = index

        if score_mask is not None:
            if score_mask.dim() != 1 or score_mask.size(0) != bsz:
                raise ValueError("score_mask must be [B]")
            valid = valid & score_mask.to(device=device, dtype=torch.bool)

        if valid.any():
            return loss[valid].mean()
        return loss.mean()

    @torch.no_grad()
    def score_calculate(
        self,
        global_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Returns confidence-gap score used for replay prioritization."""
        probs = F.softmax(global_logits, dim=-1)
        target_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        top2 = probs.topk(k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        # High score => uncertain but label-supported sample.
        score = (1.0 - margin) * target_prob
        return score.detach().cpu()


def pprd_loss(
    patch_embeds_cur: torch.Tensor,
    patch_embeds_old: torch.Tensor,
    prototypes_cur: torch.Tensor,
    prototypes_old: torch.Tensor,
    past_temp: float = 0.04,
    patch_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Patch-to-prototype relation distillation (PPRD).

    Applies past_temp symmetrically to student and teacher prototype-relation
    logits, then KL-matches the softmax distributions per patch.

    Args:
        patch_embeds_cur: [B, N, D]
        patch_embeds_old: [B, N, D]
        prototypes_cur: [K, D]
        prototypes_old: [K, D]
    """
    q_cur_logits = torch.einsum(
        "bnd,kd->bnk",
        F.normalize(patch_embeds_cur, dim=-1),
        F.normalize(prototypes_cur, dim=-1),
    )
    q_old_logits = torch.einsum(
        "bnd,kd->bnk",
        F.normalize(patch_embeds_old, dim=-1),
        F.normalize(prototypes_old, dim=-1),
    )

    log_q_cur = F.log_softmax(q_cur_logits / past_temp, dim=-1)
    q_old = F.softmax(q_old_logits / past_temp, dim=-1)

    # KL(q_old || q_cur): compare soft teacher distribution against student log-probs.
    per_patch_kl = F.kl_div(log_q_cur, q_old, reduction="none").sum(dim=-1)  # [B, N]

    if patch_weights is None:
        return per_patch_kl.mean()

    patch_weights = patch_weights.clamp_min(1e-6)
    patch_weights = patch_weights / patch_weights.sum(dim=1, keepdim=True)
    return (per_patch_kl * patch_weights).sum(dim=1).mean()


def ird_loss(
    embeds_cur: torch.Tensor,
    embeds_old: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Instance-Relation Distillation (CO2L-style).

    Distills the within-batch instance similarity structure from teacher to
    student. Each row of the [B, B] cosine-similarity matrix (with self-pairs
    masked out) is read as a probability distribution; cross-entropy aligns the
    student's row to the teacher's, regularizing encoder drift.

    Args:
        embeds_cur: [B, D] student instance embeddings.
        embeds_old: [B, D] teacher instance embeddings.
        temperature: softmax temperature applied symmetrically to both rows.
    """
    if embeds_cur.dim() != 2 or embeds_old.dim() != 2:
        raise ValueError("embeds must be [B, D]")
    if embeds_cur.size(0) != embeds_old.size(0):
        raise ValueError("embeds_cur and embeds_old must share the same batch size")
    if embeds_cur.size(0) < 2:
        return embeds_cur.new_zeros(())

    z_cur = F.normalize(embeds_cur, dim=-1)
    z_old = F.normalize(embeds_old, dim=-1)

    sim_cur = torch.matmul(z_cur, z_cur.t()) / temperature
    sim_old = torch.matmul(z_old, z_old.t()) / temperature

    bsz = z_cur.size(0)
    self_mask = torch.eye(bsz, device=z_cur.device, dtype=torch.bool)
    sim_cur = sim_cur.masked_fill(self_mask, float("-inf"))
    sim_old = sim_old.masked_fill(self_mask, float("-inf"))

    log_p_cur = F.log_softmax(sim_cur, dim=-1)
    p_old = F.softmax(sim_old, dim=-1)

    # log_softmax produces -inf on the masked diagonal; zero both factors there
    # so the cross-entropy avoids 0 * -inf -> NaN while preserving off-diag terms.
    p_old = p_old.masked_fill(self_mask, 0.0)
    log_p_cur = log_p_cur.masked_fill(self_mask, 0.0)

    return -(p_old * log_p_cur).sum(dim=-1).mean()


def prd_loss(
    patch_embeds_cur: torch.Tensor,
    patch_embeds_old: torch.Tensor,
    prototypes_cur: torch.Tensor,
    prototypes_old: torch.Tensor,
    current_temp: float = 1.0,
    past_temp: float = 2.0,
    patch_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Patch-to-prototype relation distillation.

    Args:
        patch_embeds_cur: [B, N, D]
        patch_embeds_old: [B, N, D]
        prototypes_cur: [K, D]
        prototypes_old: [K, D]
    """
    q_cur_logits = torch.einsum(
        "bnd,kd->bnk",
        F.normalize(patch_embeds_cur, dim=-1),
        F.normalize(prototypes_cur, dim=-1),
    )
    q_old_logits = torch.einsum(
        "bnd,kd->bnk",
        F.normalize(patch_embeds_old, dim=-1),
        F.normalize(prototypes_old, dim=-1),
    )

    log_q_cur = F.log_softmax(q_cur_logits / current_temp, dim=-1)
    q_old = F.softmax(q_old_logits / past_temp, dim=-1)

    per_patch_ce = -(q_old * log_q_cur).sum(dim=-1)  # [B, N]

    if patch_weights is None:
        return per_patch_ce.mean()

    patch_weights = patch_weights.clamp_min(1e-6)
    patch_weights = patch_weights / patch_weights.sum(dim=1, keepdim=True)
    return (per_patch_ce * patch_weights).sum(dim=1).mean()
