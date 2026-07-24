"""ConeColBERT losses (spec 6.2-6.4) and weak token alignment (spec 5.4)."""
import torch


def cone_margin_loss(v_pos, v_neg, margin=0.10, pos_weights=None):
    """L = sum w*E(pos) + sum max(0, margin - E(neg))."""
    pos = v_pos if pos_weights is None else v_pos * pos_weights
    loss = pos.mean() if pos.numel() else v_pos.new_zeros(())
    if v_neg.numel():
        loss = loss + (margin - v_neg).clamp_min(0).mean()
    return loss


def radial_order_loss(rho_parent, rho_child, margin=0.05):
    """Explicit ordered pairs only: parent must sit inward of child."""
    return (rho_parent + margin - rho_child).clamp_min(0).mean()


def radius_variance_loss(rho, valid, std_min=0.03, rho_safe=0.90):
    r = rho[valid.bool()]
    if r.numel() < 2:
        return rho.new_zeros(())
    anti_collapse = (std_min - r.std()).clamp_min(0)
    boundary = (r - rho_safe).clamp_min(0).pow(2).mean()
    return anti_collapse + boundary


def weak_alignment(pair_e, pos_idx, query_content_mask, doc_valid,
                   min_cosine=0.45, confidence_power=2.0):
    """From pairwise Euclidean scores pick j* = argmax for each content query
    token against its POSITIVE doc; keep pairs with cosine > threshold.

    pair_e [B,n,Lq,Ld]; pos_idx [B]; returns (b_idx, q_idx, j_star, conf)."""
    B = pair_e.shape[0]
    e_pos = pair_e[torch.arange(B), pos_idx]              # [B,Lq,Ld]
    e_pos = e_pos.masked_fill(~doc_valid[torch.arange(B), pos_idx].unsqueeze(1), -1e4)
    best, j_star = e_pos.max(dim=-1)                      # [B,Lq]
    keep = query_content_mask.bool() & (best > min_cosine)
    b_idx, q_idx = keep.nonzero(as_tuple=True)
    conf = best[b_idx, q_idx].clamp(0, 1).pow(confidence_power)
    return b_idx, q_idx, j_star[b_idx, q_idx], conf
