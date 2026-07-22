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
import torch.distributed.tensor  # noqa: F401  (peft probes DTensor when loading LoRA checkpoints)
from torch import nn


def inv_softplus(y):
    return math.log(math.expm1(y))


class HALOLift(nn.Module):
    def __init__(self, curv_init=0.1, logit_scale_init=2.6592, score_mode="lorentz",
                 kd_scale_init=1.0, sq_dist=False, squash_radius_init=0.0):
        super().__init__()
        self.curv_init = curv_init
        self.logit_scale_init = logit_scale_init
        self.kd_scale_init = kd_scale_init
        self.score_mode = score_mode  # "lorentz" | "hybrid" (HALO mainline: 0.5 cos + 0.5 -dist)
        self.sq_dist = sq_dist        # squared Lorentz distance (finite gradient at 0)
        self.squash_radius_init = squash_radius_init  # 0 disables the norm squash
        self.curv_raw = nn.Parameter(torch.tensor(inv_softplus(curv_init), dtype=torch.float32))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init), dtype=torch.float32))
        self.kd_scale = nn.Parameter(torch.tensor(float(kd_scale_init), dtype=torch.float32))
        if squash_radius_init > 0:
            self.radius_raw = nn.Parameter(
                torch.tensor(inv_softplus(squash_radius_init), dtype=torch.float32))

    def curv(self):
        return torch.nn.functional.softplus(self.curv_raw) + 1e-6

    def forward(self, features):
        x = features["token_embeddings"].float()
        if self.squash_radius_init > 0:
            # norm squash: direction kept, norms mapped monotonically into (0, R)
            r = torch.nn.functional.softplus(self.radius_raw) + 1e-6
            x = r * x / (1.0 + x.norm(dim=-1, keepdim=True))
        c = self.curv()
        x0 = torch.sqrt(torch.clamp(1.0 / c + (x * x).sum(-1), min=1e-6))
        features["token_embeddings"] = torch.cat([x0.unsqueeze(-1), x], dim=-1)
        features["halo_curv"] = c
        features["halo_logit_scale"] = self.logit_scale.clamp(max=math.log(100.0)).exp()
        features["halo_kd_scale"] = self.kd_scale.clamp(max=math.log(100.0)).exp()
        return features

    def get_config_dict(self):
        return {"curv_init": self.curv_init, "logit_scale_init": self.logit_scale_init,
                "score_mode": self.score_mode, "kd_scale_init": self.kd_scale_init,
                "sq_dist": self.sq_dist, "squash_radius_init": self.squash_radius_init}

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
            # strict=False: older checkpoints predate kd_scale
            module.load_state_dict(
                torch.load(sd, map_location="cpu", weights_only=True), strict=False)
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


def lorentz_pairwise_sim(q, d, curv, eps=1e-4, squared=False):
    """Negative (squared) Lorentz distance between lifted token sets.

    q [..., Lq, D+1], d [..., Ld, D+1] (fp32) -> sim [..., Lq, Ld].
    squared=True: -d^2 = -acosh(z)^2/c — d(acosh^2)/dz -> 2 as z -> 1+, so the
    gradient stays FINITE at coincident pairs (plain -d has a 1/sqrt(z^2-1)
    singularity there, which blew up iter-A once the gathered NCE pool made
    near-duplicate token pairs frequent)."""
    ip = torch.matmul(q[..., 1:], d[..., 1:].transpose(-1, -2))
    ip = ip - q[..., :1] * d[..., 0].unsqueeze(-2)
    a = torch.acosh(torch.clamp(-curv * ip, min=1.0 + eps))
    if squared:
        return -(a * a) / curv
    return -a / torch.sqrt(curv)


def _maxsim_reduce(sim, logit_scale, queries_mask, documents_mask):
    if documents_mask is not None:
        sim = sim.masked_fill(~documents_mask.unsqueeze(2).bool(), float("-inf"))
    best = torch.nan_to_num(sim.max(dim=-1).values, neginf=0.0)  # [Nq,B,Lq]
    if queries_mask is not None:
        best = best * queries_mask.unsqueeze(1).float()
    return logit_scale * best.sum(dim=-1)


def lorentz_kd_scores(q, docs, curv, logit_scale, queries_mask=None, documents_mask=None,
                      squared=False):
    """MaxSim over negative (squared) Lorentz distance, scaled by logit_scale.

    q [Nq, Lq, D+1], docs [Nq, B, Ld, D+1] -> scores [Nq, B]."""
    sim = lorentz_pairwise_sim(q.float().unsqueeze(1), docs.float(), curv, squared=squared)
    return _maxsim_reduce(sim, logit_scale, queries_mask, documents_mask)


def cosine_kd_scores(q, docs, logit_scale, queries_mask=None, documents_mask=None):
    """MaxSim over cosine similarity of the SPATIAL components of lifted embeddings."""
    qs = torch.nn.functional.normalize(q[..., 1:].float(), dim=-1)
    ds = torch.nn.functional.normalize(docs[..., 1:].float(), dim=-1)
    sim = torch.matmul(qs.unsqueeze(1), ds.transpose(-1, -2))  # [Nq,B,Lq,Ld]
    return _maxsim_reduce(sim, logit_scale, queries_mask, documents_mask)


def gather_all_docs(d, d_mask, pad_len):
    """All-gather doc token embeddings + masks across ranks for NCE negatives.

    Pads Ld to pad_len first (per-rank collators pad to different batch maxima).
    HALO-style gradient trick: the local rank's slot keeps its autograd graph,
    remote slots are detached. Returns ([world*Nd, pad_len, D], mask, rank)."""
    import torch.distributed as dist

    pad = pad_len - d.shape[1]
    if pad > 0:
        d = torch.nn.functional.pad(d, (0, 0, 0, pad))
        d_mask = torch.nn.functional.pad(d_mask, (0, pad))
    d_mask = d_mask.to(torch.uint8)  # NCCL all_gather chokes on bool
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return d, d_mask, 0
    world, rank = dist.get_world_size(), dist.get_rank()
    d_list = [torch.zeros_like(d) for _ in range(world)]
    m_list = [torch.zeros_like(d_mask) for _ in range(world)]
    dist.all_gather(d_list, d.contiguous())
    dist.all_gather(m_list, d_mask.contiguous())
    d_list[rank] = d
    return torch.cat(d_list, dim=0), torch.cat(m_list, dim=0), rank


def inbatch_nce_scores(q, docs_flat, curv, logit_scale, mode,
                       queries_mask=None, documents_mask=None, squared=False):
    """Every query vs every document in the batch, eval-consistent token scoring.

    q [Nq, Lq, D+1], docs_flat [Nd, Ld, D+1] -> scores [Nq, Nd].
    mode "hybrid": 0.5 * spatial cosine + 0.5 * negative Lorentz distance."""
    qf = q.float().unsqueeze(1)              # [Nq,1,Lq,D+1]
    df = docs_flat.float().unsqueeze(0)      # [1,Nd,Ld,D+1]
    sim = lorentz_pairwise_sim(qf, df, curv, squared=squared)
    if mode == "hybrid":
        qs = torch.nn.functional.normalize(qf[..., 1:], dim=-1)
        ds = torch.nn.functional.normalize(df[..., 1:], dim=-1)
        sim = 0.5 * torch.matmul(qs, ds.transpose(-1, -2)) + 0.5 * sim
    dmask = documents_mask.unsqueeze(0) if documents_mask is not None else None
    return _maxsim_reduce(sim, logit_scale, queries_mask, dmask)
