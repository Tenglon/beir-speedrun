"""Phase 2 (spec 7): frozen-backbone geometry warm-up for ConeColBERT.

Backbone + euclidean head frozen; trains direction/radial heads and gate.
First --geo-steps: cone/order/variance losses only (scale pinned at 0).
Then scale warms 0 -> scale_max linearly over --warmup-steps while a listwise
KD KL on fused scores feeds the gate (it has no gradient path before that).
"""
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from pylate import models, utils

from conecolbert.losses import cone_margin_loss, radius_variance_loss, weak_alignment
from conecolbert.geometry import cone_violation
from conecolbert.modeling import ConeColBERT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=8, help="1 positive + sampled negatives")
    ap.add_argument("--geo-steps", type=int, default=4000)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--total-steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale-max", type=float, default=0.10)
    ap.add_argument("--min-cosine", type=float, default=0.45)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    colbert = models.ColBERT(model_name_or_path=args.ckpt, device=device)
    model = ConeColBERT(colbert, hyperbolic_scale_max=args.scale_max).to(device)
    model.colbert.requires_grad_(False)
    model.colbert.eval()
    trainable = [p for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("colbert.")]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    train = load_dataset(path=args.data_dir, name="train", split="train")
    queries = load_dataset(path=args.data_dir, name="queries", split="train")
    documents = load_dataset(path=args.data_dir, name="documents", split="train")
    train.set_transform(utils.KDProcessing(queries=queries, documents=documents).transform)
    collator = utils.ColBERTCollator(model.colbert.tokenize)
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=lambda rows: collator(rows))

    special = set(model.colbert[0].tokenizer.all_special_ids)
    step = 0
    for batch in loader:
        if step >= args.total_steps:
            break
        labels = batch["label"].to(device)                      # [B, 32] teacher
        top = labels.argsort(dim=-1, descending=True)[:, :args.n_docs]
        qf = {k.replace("query_", ""): v.to(device) for k, v in batch.items()
              if k.startswith("query_")}
        df_all = {k.replace("documents_", ""): v.to(device) for k, v in batch.items()
                  if k.startswith("documents_")}
        B, n_all = labels.shape
        sel = (torch.arange(B, device=device).unsqueeze(1) * n_all + top).reshape(-1)
        df = {k: v[sel] for k, v in df_all.items()}
        t_scores = labels.gather(1, top)

        warm = max(0.0, min(1.0, (step - args.geo_steps) / max(args.warmup_steps, 1)))
        with torch.no_grad():
            model.scale_raw.fill_(warm * args.scale_max)

        out = model(qf, df, n_docs=args.n_docs, train_aggregation="logsumexp",
                    lse_temperature=0.10)
        content = qf["attention_mask"].bool()
        for sid in special:
            content &= qf["input_ids"] != sid
        d_valid = df["attention_mask"].bool().view(B, args.n_docs, -1)
        b_i, q_i, j_star, conf = weak_alignment(
            out.pairwise_euclidean.detach(), torch.zeros(B, dtype=torch.long, device=device),
            content, d_valid, min_cosine=args.min_cosine)
        v = out.pairwise_violation                              # [B,n,Lq,Ld]
        v_pos = v[b_i, 0, q_i, j_star]
        v_neg = v[b_i, 1:, q_i, :].min(dim=-1).values.reshape(-1)  # best token in each sibling
        geo_loss = (cone_margin_loss(v_pos, v_neg, pos_weights=conf)
                    + 0.01 * radius_variance_loss(out.query_radii, content)
                    + 0.01 * radius_variance_loss(
                        out.document_radii, d_valid.view(B, args.n_docs, -1)))
        loss = geo_loss
        if warm > 0:
            # spec 6.6: teacher-residual warm-up on centered scores — only the
            # correction is trained, so the gate cannot be pushed to saturation
            # the way full listwise KL pushes it (phase-2 full-run failure mode)
            r_t = t_scores - out.euclidean_scores.detach()
            r_t = r_t - r_t.mean(-1, keepdim=True)
            c_h = out.cone_corrections
            c_h = c_h - c_h.mean(-1, keepdim=True)
            loss = loss + torch.nn.functional.huber_loss(c_h, r_t)
            loss = loss + 0.05 * (out.gates[content].mean() - 0.3).pow(2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_every == 0:
            g = out.gates[content]
            rho_all = torch.cat([out.query_radii[content],
                                 out.document_radii.reshape(-1)])
            print("step=%d loss=%.4f vpos=%.3f vneg=%.3f pairs=%d scale=%.3f "
                  "gate_mean=%.3f rho_std=%.4f" % (
                      step, loss.item(), v_pos.mean().item() if v_pos.numel() else -1,
                      v_neg.mean().item() if v_neg.numel() else -1, v_pos.numel(),
                      model.hyperbolic_scale().item(), g.mean().item(),
                      rho_all.std().item()), flush=True)
            if rho_all.std().item() < 0.01 or g.mean().item() < 0.01 or g.mean().item() > 0.99:
                print("COLLAPSE ALARM at step %d" % step, flush=True)
        step += 1

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"hyper_head": model.hyper_head.state_dict(),
                "gate": model.gate.state_dict(),
                "scale_raw": model.scale_raw.detach().cpu(),
                "config": vars(args)}, os.path.join(args.out_dir, "cone_phase2.pt"))
    print("saved", os.path.join(args.out_dir, "cone_phase2.pt"))


if __name__ == "__main__":
    main()
