"""ConeColBERT: frozen Euclidean ColBERT + residual entailment-cone branch.

Invariant (spec 2.5): with hyperbolic_scale == 0 the fused scores equal the
original ColBERT scores exactly — verified by tests/test_parity.py.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .geometry import cone_violation, min_valid_radius


@dataclass
class ConeColBERTOutput:
    scores: torch.Tensor                # [B, num_docs] fused
    euclidean_scores: torch.Tensor      # [B, num_docs]
    cone_corrections: torch.Tensor      # [B, num_docs] (scores - euclidean)
    pairwise_euclidean: Optional[torch.Tensor] = None
    pairwise_violation: Optional[torch.Tensor] = None
    query_radii: Optional[torch.Tensor] = None
    document_radii: Optional[torch.Tensor] = None
    gates: Optional[torch.Tensor] = None


class HyperbolicTokenHead(nn.Module):
    """Direction (shared) + role-specific radius -> Poincare point rho * u."""

    def __init__(self, hidden_dim, hyper_dim=32, rho_min=0.12, rho_max=0.95,
                 cone_k=0.10, eps=1e-6):
        super().__init__()
        assert rho_min > min_valid_radius(cone_k), \
            "rho_min must exceed the minimal valid cone radius eps_K"
        self.direction = nn.Linear(hidden_dim, hyper_dim)
        self.radial = nn.ModuleDict({
            "query": nn.Linear(hidden_dim, 1),
            "document": nn.Linear(hidden_dim, 1),
        })
        self.rho_min, self.rho_max, self.eps = rho_min, rho_max, eps

    def forward(self, hidden_states, role, attention_mask):
        h = hidden_states.float()
        raw = self.direction(h)
        u = raw / (raw.norm(dim=-1, keepdim=True) + self.eps)
        rho = self.rho_min + (self.rho_max - self.rho_min) * torch.sigmoid(
            self.radial[role](h).squeeze(-1))
        points = rho.unsqueeze(-1) * u
        valid = attention_mask.bool()
        points = points * valid.unsqueeze(-1)  # padding points -> 0, masked in scoring
        return points, u, rho, valid


class ConeColBERT(nn.Module):
    """Wraps a pylate ColBERT (Transformer + Dense). The Euclidean path is the
    original model verbatim; the cone branch subtracts gated violations."""

    def __init__(self, colbert, hyper_dim=32, rho_min=0.12, rho_max=0.95,
                 cone_k=0.10, gate_bias_init=-2.0, hyperbolic_scale_init=0.0,
                 hyperbolic_scale_max=0.25):
        super().__init__()
        self.colbert = colbert  # pylate models.ColBERT (modules: Transformer, Dense)
        hidden_dim = colbert[0].auto_model.config.hidden_size
        self.hyper_head = HyperbolicTokenHead(hidden_dim, hyper_dim, rho_min, rho_max, cone_k)
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias_init)
        self.scale_raw = nn.Parameter(torch.tensor(float(hyperbolic_scale_init)))
        self.scale_max = hyperbolic_scale_max
        self.cone_k = cone_k
        # violation threshold: corrections fire only on V > v0 (OOD noise guard);
        # v0 raw <=0 keeps relu(v0)=0 => exact parity behavior preserved at init
        self.v0_raw = nn.Parameter(torch.tensor(-2.0))

    def v0(self):
        return torch.nn.functional.softplus(self.v0_raw) * 0.1

    def hyperbolic_scale(self):
        return self.scale_raw.clamp(0.0, self.scale_max)

    def _encode(self, features, role):
        """Run the pylate module pipeline, capturing backbone hidden states."""
        out = self.colbert[0](features)              # Transformer: 768-d token embs
        hidden = out["token_embeddings"]
        for module in list(self.colbert)[1:]:
            out = module(out)                        # Dense -> 128-d (pylate normalizes in scores)
        euclid = torch.nn.functional.normalize(out["token_embeddings"], p=2, dim=-1)
        mask = features["attention_mask"].bool()
        points, _, rho, valid = self.hyper_head(hidden, role, mask)
        gates = torch.sigmoid(self.gate(hidden.float()).squeeze(-1)) if role == "query" else None
        return euclid, hidden, points, rho, valid, gates

    def forward(self, query_features, document_features, n_docs,
                train_aggregation="max", lse_temperature=0.05, return_diagnostics=True):
        q_e, _, q_pts, q_rho, q_valid, gates = self._encode(query_features, "query")
        d_e, _, d_pts, d_rho, d_valid, _ = self._encode(document_features, "document")

        B, Lq, dim = q_e.shape
        d_e = d_e.view(B, n_docs, -1, dim)
        d_pts = d_pts.view(B, n_docs, d_pts.shape[1], -1)
        d_rho = d_rho.view(B, n_docs, -1)
        d_valid = d_valid.view(B, n_docs, -1)

        e = torch.einsum("bqh,bnth->bnqt", q_e, d_e)            # [B,n,Lq,Ld]
        v = cone_violation(
            q_pts.unsqueeze(1).expand(-1, n_docs, -1, -1).reshape(B * n_docs, Lq, -1),
            d_pts.reshape(B * n_docs, d_pts.shape[2], -1),
            query_mask=q_valid.unsqueeze(1).expand(-1, n_docs, -1).reshape(B * n_docs, Lq),
            document_mask=d_valid.reshape(B * n_docs, -1),
            cone_k=self.cone_k,
        ).view(B, n_docs, Lq, -1)

        scale = self.hyperbolic_scale()
        v_eff = (v - self.v0()).clamp_min(0.0)
        m = e - scale * gates.unsqueeze(1).unsqueeze(-1) * v_eff
        e_masked = e.masked_fill(~d_valid.unsqueeze(2), torch.finfo(e.dtype).min)
        m_masked = m.masked_fill(~d_valid.unsqueeze(2), torch.finfo(m.dtype).min)

        def aggregate(pair, mode):
            if mode == "logsumexp":
                best = lse_temperature * torch.logsumexp(pair / lse_temperature, dim=-1)
            else:
                best = pair.max(dim=-1).values
            # original ColBERT semantics: [MASK] query-expansion tokens count
            return best.sum(-1)

        scores = aggregate(m_masked, train_aggregation)
        with torch.no_grad() if not self.training else torch.enable_grad():
            e_scores = aggregate(e_masked, train_aggregation)
        return ConeColBERTOutput(
            scores=scores, euclidean_scores=e_scores, cone_corrections=scores - e_scores,
            pairwise_euclidean=e if return_diagnostics else None,
            pairwise_violation=v if return_diagnostics else None,
            query_radii=q_rho if return_diagnostics else None,
            document_radii=d_rho if return_diagnostics else None,
            gates=gates if return_diagnostics else None,
        )
