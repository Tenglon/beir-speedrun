import math

import pytest
import torch

from conecolbert.geometry import cone_aperture, cone_violation, min_valid_radius
from conecolbert.modeling import HyperbolicTokenHead

K = 0.10


def ray(direction, rho):
    d = torch.tensor(direction, dtype=torch.float32)
    return (rho * d / d.norm()).unsqueeze(0)


def test_point_inside_ball():
    head = HyperbolicTokenHead(hidden_dim=16)
    h = torch.randn(2, 5, 16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    pts, u, rho, valid = head(h, "query", mask)
    norms = pts.norm(dim=-1)
    assert (norms < 1).all()
    assert (rho >= head.rho_min - 1e-6).all() and (rho <= head.rho_max + 1e-6).all()
    assert torch.allclose(u.norm(dim=-1), torch.ones(2, 5), atol=1e-4)
    assert head.rho_min > min_valid_radius(K)


def test_aperture_monotonicity():
    rhos = torch.linspace(0.15, 0.95, 20)
    pts = rhos.unsqueeze(-1) * torch.tensor([[1.0, 0.0]])
    ap = cone_aperture(pts, K)
    assert (ap[1:] < ap[:-1]).all(), "aperture must shrink as radius grows"


def test_same_ray_descendant():
    x = ray([1.0, 0.0, 0.0], 0.3).unsqueeze(0)
    y = ray([1.0, 0.0, 0.0], 0.7).unsqueeze(0)
    v = cone_violation(x, y, cone_k=K)
    assert v.item() < 1e-3, "outward point on the same ray must satisfy the cone"


def test_sibling_violation():
    x = ray([1.0, 0.0, 0.0], 0.5).unsqueeze(0)
    desc = ray([1.0, 0.05, 0.0], 0.8).unsqueeze(0)
    sib = ray([0.0, 1.0, 0.0], 0.5).unsqueeze(0)
    v_desc = cone_violation(x, desc, cone_k=K).item()
    v_sib = cone_violation(x, sib, cone_k=K).item()
    assert v_sib > v_desc + 0.1, (v_sib, v_desc)


def test_identical_points():
    x = ray([0.3, 0.4, 0.0], 0.5).unsqueeze(0)
    v = cone_violation(x, x.clone(), cone_k=K)
    assert torch.isfinite(v).all() and v.item() < 1e-3


def test_padding_mask():
    q = torch.randn(1, 3, 8) * 0.2
    d = torch.randn(1, 4, 8) * 0.2
    dmask = torch.tensor([[True, True, False, False]])
    v = cone_violation(q, d, document_mask=dmask, cone_k=K)
    assert (v[0, :, 2:] >= 1e3).all(), "padded docs must carry huge violation"


def test_boundary_stability():
    x = ray([1.0, 1.0, 0.0], 0.9499).unsqueeze(0).requires_grad_(True)
    y = ray([1.0, 0.9, 0.0], 0.9499).unsqueeze(0)
    v = cone_violation(x, y, cone_k=K)
    v.sum().backward()
    assert torch.isfinite(v).all() and torch.isfinite(x.grad).all()


def test_fp16_bf16():
    for dtype in (torch.float16, torch.bfloat16):
        q = (torch.randn(1, 3, 8) * 0.2).to(dtype)
        d = (torch.randn(1, 4, 8) * 0.2).to(dtype)
        v = cone_violation(q, d, cone_k=K)
        assert v.dtype == torch.float32 and torch.isfinite(v).all()


def test_gradients_flow():
    head = HyperbolicTokenHead(hidden_dim=16)
    h = torch.randn(1, 4, 16, requires_grad=True)
    mask = torch.ones(1, 4, dtype=torch.bool)
    q_pts, _, q_rho, _ = head(h, "query", mask)
    d_pts, _, _, _ = head(h + 0.1, "document", mask)
    loss = cone_violation(q_pts, d_pts, cone_k=K).mean() + q_rho.std()
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
