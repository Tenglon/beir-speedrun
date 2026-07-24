"""Spec 9.4: a hand-built parent/child/sibling set must be overfittable by the
hyperbolic head alone (backbone frozen, hidden states cached)."""
import os

import pytest
import torch

pylate = pytest.importorskip("pylate")
from pylate import models  # noqa: E402

from conecolbert.geometry import cone_violation  # noqa: E402
from conecolbert.losses import cone_margin_loss, radial_order_loss, radius_variance_loss  # noqa: E402
from conecolbert.modeling import HyperbolicTokenHead  # noqa: E402

ROOT = os.environ.get("BEIR_ROOT", "/gpfs/scratch/ehpc821/uoa994647/beir_speedrun")
CKPT = os.environ.get("COLBERT_CHECKPOINT", ROOT + "/colbert/run2_8gpu/checkpoint-60000")

PARENTS = ["benefit", "city", "tax policy", "disease"]
CHILDREN = ["childcare benefit", "amsterdam", "dutch tax policy in 2024", "type 2 diabetes"]
SIBLINGS = ["housing benefit", "rotterdam", "us tax policy in 1990", "influenza"]


@pytest.fixture(scope="module")
def hiddens():
    if not os.path.isdir(CKPT):
        pytest.skip("checkpoint unavailable")
    colbert = models.ColBERT(model_name_or_path=CKPT, device="cpu")
    feats = colbert.tokenize(PARENTS + CHILDREN + SIBLINGS, is_query=False)
    with torch.no_grad():
        out = colbert[0](feats)
    h = out["token_embeddings"]
    mask = feats["attention_mask"].bool()
    pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    n = len(PARENTS)
    return pooled[:n], pooled[n:2 * n], pooled[2 * n:]


def test_tiny_overfit(hiddens):
    hp, hc, hs = hiddens
    head = HyperbolicTokenHead(hidden_dim=hp.shape[-1])
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    mask = torch.ones(hp.shape[0], 1, dtype=torch.bool)

    def points(h, role):
        pts, _, rho, _ = head(h.unsqueeze(1), role, mask)
        return pts, rho

    first = None
    for step in range(400):
        pp, rp = points(hp, "query")
        cp, rc = points(hc, "document")
        sp, _ = points(hs, "document")
        v_pos = cone_violation(pp, cp).squeeze(-1).squeeze(-1)
        v_sib = cone_violation(pp, sp).squeeze(-1).squeeze(-1)
        loss = (cone_margin_loss(v_pos, v_sib, margin=0.10)
                + 0.1 * radial_order_loss(rp, rc, margin=0.05)
                + 0.01 * radius_variance_loss(torch.cat([rp, rc]).squeeze(-1),
                                              torch.ones(2 * rp.shape[0], 1)))
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    pp, rp = points(hp, "query")
    cp, rc = points(hc, "document")
    sp, _ = points(hs, "document")
    v_pos = cone_violation(pp, cp).squeeze(-1).squeeze(-1).mean().item()
    v_sib = cone_violation(pp, sp).squeeze(-1).squeeze(-1).mean().item()
    assert loss.item() < first * 0.5, (first, loss.item())
    assert v_pos < 0.05, v_pos
    assert v_sib > v_pos + 0.05, (v_pos, v_sib)
    assert (rp.squeeze(-1) < rc.squeeze(-1)).all(), "parents must sit inward"
