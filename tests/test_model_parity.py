"""Spec 9.2: with hyperbolic_scale == 0 ConeColBERT must equal original ColBERT.

Runs on CPU against the real run2 checkpoint when BEIR_ROOT is set (BSC),
otherwise against a fresh bert-base ColBERT if available locally.
"""
import os

import pytest
import torch

pylate = pytest.importorskip("pylate")
from pylate import models  # noqa: E402

from conecolbert.modeling import ConeColBERT  # noqa: E402

ROOT = os.environ.get("BEIR_ROOT", "/gpfs/scratch/ehpc821/uoa994647/beir_speedrun")
CKPT = os.environ.get("COLBERT_CHECKPOINT", ROOT + "/colbert/run2_8gpu/checkpoint-60000")


@pytest.fixture(scope="module")
def setup():
    if not os.path.isdir(CKPT):
        pytest.skip("checkpoint not available: " + CKPT)
    colbert = models.ColBERT(model_name_or_path=CKPT, device="cpu")
    model = ConeColBERT(colbert).eval()
    queries = ["what treats lung cancer", "capital of france"]
    docs = ["chemotherapy is a common treatment for lung cancer",
            "paris is the capital of france",
            "the 2024 olympics were held in paris",
            "a recipe for sourdough bread"]
    qf = colbert.tokenize(queries, is_query=True)
    df = colbert.tokenize([d for d in docs for _ in range(1)] * 1, is_query=False)
    # 2 docs per query
    return model, colbert, qf, {k: v for k, v in df.items()}


def _baseline_scores(colbert, qf, df, n_docs):
    """Original pylate ColBERT MaxSim (normalized dot, doc-mask aware)."""
    q = torch.nn.functional.normalize(colbert(qf)["token_embeddings"], dim=-1)
    d = torch.nn.functional.normalize(colbert(df)["token_embeddings"], dim=-1)
    B, Lq, dim = q.shape
    d = d.view(B, n_docs, -1, dim)
    sim = torch.einsum("bqh,bnth->bnqt", q, d)
    dmask = df["attention_mask"].bool().view(B, n_docs, -1)
    sim = sim.masked_fill(~dmask.unsqueeze(2), torch.finfo(sim.dtype).min)
    return sim.max(dim=-1).values.sum(-1)


def test_scale_zero_parity(setup):
    model, colbert, qf, df = setup
    with torch.no_grad():
        out = model(qf, df, n_docs=2, train_aggregation="max")
        base = _baseline_scores(colbert, qf, df, n_docs=2)
    assert model.hyperbolic_scale().item() == 0.0
    assert torch.isfinite(out.scores).all()
    diff = (out.scores - base).abs().max().item()
    assert diff < 1e-5, "scale=0 fused scores must match original ColBERT, diff=%g" % diff
    assert (out.scores.argsort(-1) == base.argsort(-1)).all()
    assert (out.cone_corrections.abs() < 1e-6).all()


def test_new_keys_only(setup):
    model, colbert, _, _ = setup
    base_keys = {("colbert." + k) for k in colbert.state_dict()}
    new = [k for k in model.state_dict() if k not in base_keys]
    assert all(k.startswith(("hyper_head.", "gate.", "scale_raw")) for k in new)


def test_scale_zero_no_retrieval_grad_to_hyper(setup):
    model, _, qf, df = setup
    model.train()
    out = model(qf, df, n_docs=2, train_aggregation="logsumexp")
    out.scores.sum().backward()
    g = model.hyper_head.direction.weight.grad
    assert g is None or torch.allclose(g, torch.zeros_like(g), atol=1e-9)
