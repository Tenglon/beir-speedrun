"""Joint end-to-end ConeColBERT: euclid head + cone branch trained together
from bert-base with the proven KD recipe (stock pylate objective on FUSED
scores, min-max normalized). Parity init (scale=0): hyperbolic capacity is
used only where the KD objective finds it useful.

torchrun --nproc_per_node=4 train/train_colbert_joint.py ...
"""
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from datasets import load_dataset
from pylate import models, utils

from conecolbert.losses import radius_variance_loss
from conecolbert.modeling import ConeColBERT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--total-steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cone-lr", type=float, default=1e-4)
    ap.add_argument("--scale-max", type=float, default=0.25)
    ap.add_argument("--save-every", type=int, default=5000)
    args = ap.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    device = "cuda"

    colbert = models.ColBERT(model_name_or_path=args.model_dir,
                             embedding_size=128, document_length=300, device=device)
    model = ConeColBERT(colbert, hyperbolic_scale_max=args.scale_max).to(device)
    cone_names = [n for n, _ in model.named_parameters() if not n.startswith("colbert.")]
    params = [
        {"params": [p for n, p in model.named_parameters() if n.startswith("colbert.")],
         "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("colbert.")],
         "lr": args.cone_lr},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * max(0.0, 1 - s / args.total_steps))
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True) if world > 1 else model

    train = load_dataset(path=args.data_dir, name="train", split="train")
    queries = load_dataset(path=args.data_dir, name="queries", split="train")
    documents = load_dataset(path=args.data_dir, name="documents", split="train")
    train.set_transform(utils.KDProcessing(queries=queries, documents=documents).transform)
    collator = utils.ColBERTCollator(model.colbert.tokenize)
    sampler = torch.utils.data.DistributedSampler(train, world, rank, shuffle=True) \
        if world > 1 else None
    loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=4, collate_fn=lambda rows: collator(rows))

    step = 0
    for batch in loader:
        if step >= args.total_steps:
            break
        labels = batch["label"].to(device)
        qf = {k.replace("query_", ""): v.to(device) for k, v in batch.items()
              if k.startswith("query_")}
        df = {k.replace("documents_", ""): v.to(device) for k, v in batch.items()
              if k.startswith("documents_")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ddp(qf, df, n_docs=args.n_docs, train_aggregation="max",
                      return_diagnostics=True)
        s = out.scores.float()
        mn, mx = s.min(-1, keepdim=True).values, s.max(-1, keepdim=True).values
        s = (s - mn) / (mx - mn + 1e-8)
        loss = torch.nn.functional.kl_div(
            torch.log_softmax(s, -1), torch.log_softmax(labels.float(), -1),
            log_target=True, reduction="batchmean")
        loss = loss + 0.01 * radius_variance_loss(
            out.query_radii, qf["attention_mask"]) \
            + 0.01 * radius_variance_loss(out.document_radii,
                                          df["attention_mask"].view_as(out.document_radii))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if rank == 0 and step % 100 == 0:
            print("step=%d loss=%.4f scale=%.4f v0=%.4f gate=%.3f corr=%.4f" % (
                step, loss.item(), model.hyperbolic_scale().item(), model.v0().item(),
                out.gates.mean().item(), out.cone_corrections.abs().mean().item()),
                flush=True)
        if rank == 0 and (step + 1) % args.save_every == 0:
            d = os.path.join(args.out_dir, "step%d" % (step + 1))
            model.colbert.save_pretrained(os.path.join(d, "colbert"))
            torch.save({"hyper_head": model.hyper_head.state_dict(),
                        "gate": model.gate.state_dict(),
                        "scale_raw": model.scale_raw.detach().cpu(),
                        "v0_raw": model.v0_raw.detach().cpu()},
                       os.path.join(d, "cone.pt"))
            print("saved", d, flush=True)
        step += 1


if __name__ == "__main__":
    main()
