"""HALO-style Lorentz geometry for ColBERT (ref: github.com/Tenglon/HALO).

HALOLift is a sentence-transformers module appended after the Dense projection:
it lifts raw (un-normalized) token embeddings onto the hyperboloid
x0 = sqrt(1/c + ||x||^2), carrying learnable curvature (softplus-parameterized)
and a CLIP-style learnable logit_scale. Token similarity is the negative Lorentz
distance -acosh(-c<q,d>_L)/sqrt(c) (HALO mainline `negative_lorentz_distance`).
Geometry always runs in fp32.
"""
import json
import math
import os

import torch
from torch import nn


def inv_softplus(y):
    return math.log(math.expm1(y))


class HALOLift(nn.Module):
    def __init__(self, curv_init=0.1, logit_scale_init=2.6592):
        super().__init__()
        self.curv_init = curv_init
        self.logit_scale_init = logit_scale_init
        self.curv_raw = nn.Parameter(torch.tensor(inv_softplus(curv_init), dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))

    def curv(self):
        return torch.nn.functional.softplus(self.curv_raw) + 1e-6

    def forward(self, features):
        x = features["token_embeddings"].float()
        c = self.curv()
        x0 = torch.sqrt(torch.clamp(1.0 / c + (x * x).sum(-1), min=1e-6))
        features["token_embeddings"] = torch.cat([x0.unsqueeze(-1), x], dim=-1)
        features["halo_curv"] = c
        features["halo_logit_scale"] = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return features

    def get_config_dict(self):
        return {"curv_init": self.curv_init, "logit_scale_init": self.logit_scale_init}

    def save(self, output_path, *args, **kwargs):
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path):
        # exactly one parameter => sentence-transformers uses the old-style loader,
        # which resolves the module SUBFOLDER and passes it here (ST 5.x dispatch)
        with open(os.path.join(input_path, "config.json")) as f:
            cfg = json.load(f)
        module = HALOLift(**cfg)
        sd = os.path.join(input_path, "pytorch_model.bin")
        if os.path.exists(sd):
            module.load_state_dict(torch.load(sd, map_location="cpu", weights_only=True))
        return module


def find_lift(model):
    for module in model:
        if isinstance(module, HALOLift):
            return module
    return None


def load_colbert_with_lift(model_dir, **kwargs):
    """pylate's ColBERT loader silently drops non-Transformer/Dense modules;
    re-attach the HALOLift recorded in modules.json if it went missing."""
    from pylate import models

    model = models.ColBERT(model_name_or_path=model_dir, **kwargs)
    if find_lift(model) is None:
        modules_json = os.path.join(model_dir, "modules.json")
        if os.path.exists(modules_json):
            with open(modules_json) as f:
                for entry in json.load(f):
                    if entry["type"].endswith("HALOLift"):
                        model.append(HALOLift.load(os.path.join(model_dir, entry["path"])))
    return model


def lorentz_pairwise_sim(q, d, curv, eps=1e-6):
    """Negative Lorentz distance between lifted token sets.

    q [..., Lq, D+1], d [..., Ld, D+1] (fp32) -> sim [..., Lq, Ld]."""
    ip = torch.matmul(q[..., 1:], d[..., 1:].transpose(-1, -2))
    ip = ip - q[..., :1] * d[..., 0].unsqueeze(-2)
    return -torch.acosh(torch.clamp(-curv * ip, min=1.0 + eps)) / torch.sqrt(curv)


def lorentz_kd_scores(q, docs, curv, logit_scale, queries_mask=None, documents_mask=None):
    """MaxSim over negative Lorentz distance, scaled by logit_scale.

    q [Nq, Lq, D+1], docs [Nq, B, Ld, D+1] -> scores [Nq, B]."""
    sim = lorentz_pairwise_sim(q.float().unsqueeze(1), docs.float(), curv)  # [Nq,B,Lq,Ld]
    if documents_mask is not None:
        sim = sim.masked_fill(~documents_mask.unsqueeze(2).bool(), float("-inf"))
    best = torch.nan_to_num(sim.max(dim=-1).values, neginf=0.0)  # [Nq,B,Lq]
    if queries_mask is not None:
        best = best * queries_mask.unsqueeze(1).float()
    return logit_scale * best.sum(dim=-1)
