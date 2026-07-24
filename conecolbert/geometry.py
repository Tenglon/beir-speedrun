"""Canonical Poincare entailment cones (Ganea et al. 2018), per ConeColBERT spec 2.4.

All angle math runs in fp32 regardless of input dtype (force_fp32_geometry).
"""
import math

import torch

EPS = 1e-6


def min_valid_radius(cone_k):
    """Smallest radius at which the cone aperture is defined: 2K/(1+sqrt(1+4K^2))."""
    return 2.0 * cone_k / (1.0 + math.sqrt(1.0 + 4.0 * cone_k * cone_k))


def cone_aperture(x, cone_k, eps=EPS):
    """psi(x) = arcsin(K (1-|x|^2)/|x|) for x [..., H] inside the ball."""
    x = x.float()
    norm_x = x.norm(dim=-1).clamp_min(eps)
    asin_input = (cone_k * (1.0 - norm_x**2) / norm_x).clamp(-1 + eps, 1 - eps)
    return torch.asin(asin_input)


def exterior_angle(x, y, eps=EPS):
    """Xi(x,y): angle at x between the geodesic ray from the origin through x
    (extended outward) and the geodesic from x to y. x [...,Q,H], y [...,D,H]
    broadcast to [...,Q,D]."""
    x = x.float().unsqueeze(-2)   # [...,Q,1,H]
    y = y.float().unsqueeze(-3)   # [...,1,D,H]
    xy = (x * y).sum(-1)
    nx2 = (x * x).sum(-1)
    ny2 = (y * y).sum(-1)
    # clamp BEFORE sqrt: sqrt'(0) is inf and 0*inf = NaN poisons autograd even
    # through masked_fill-ed (zero-gradient) positions
    norm_x = nx2.clamp_min(eps * eps).sqrt()
    norm_xy = (nx2 + ny2 - 2 * xy).clamp_min(eps * eps).sqrt()
    sqrt_term = (1.0 + nx2 * ny2 - 2.0 * xy).clamp_min(eps).sqrt()
    acos_input = ((xy * (1.0 + nx2) - nx2 * (1.0 + ny2))
                  / (norm_x * norm_xy * sqrt_term)).clamp(-1 + eps, 1 - eps)
    return torch.acos(acos_input)


def cone_violation(query_points, document_points, query_mask=None, document_mask=None,
                   cone_k=0.10, eps=EPS, identical_tol=1e-5, padding_violation=1e4):
    """V(x,y) = max(0, Xi(x,y) - psi(x)), [B,Q,D]; finite, >= 0.

    Near-identical pairs get violation 0 (||x-y|| is singular there); padding
    pairs get a violation large enough that MaxSim never selects them."""
    q = query_points.float()
    d = document_points.float()
    # padding points are zero vectors where every angle term is singular; swap
    # in a safe interior dummy (violations there are overwritten below anyway)
    safe = torch.zeros_like(q[..., :1, :])
    safe[..., 0] = 0.5
    if query_mask is not None:
        q = torch.where(query_mask.bool().unsqueeze(-1), q, safe)
    if document_mask is not None:
        d = torch.where(document_mask.bool().unsqueeze(-1), d, safe)
    xi = exterior_angle(q, d, eps)
    psi = cone_aperture(q, cone_k, eps).unsqueeze(-1)
    v = (xi - psi).clamp_min(0.0)

    dist2 = ((q.unsqueeze(-2) - d.unsqueeze(-3)) ** 2).sum(-1)
    v = torch.where(dist2 < identical_tol**2, torch.zeros_like(v), v)

    if document_mask is not None:
        v = v.masked_fill(~document_mask.bool().unsqueeze(-2), padding_violation)
    if query_mask is not None:
        v = v.masked_fill(~query_mask.bool().unsqueeze(-1), 0.0)
    return v
