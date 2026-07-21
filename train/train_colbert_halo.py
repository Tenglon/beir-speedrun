"""Hyperbolic (HALO-style) ColBERT: LoRA backbone + Lorentz lift head, KD training.

Differences vs train_colbert.py:
- backbone trained via LoRA (r32/alpha64/dropout .05 on query,key,value,dense — the
  proven text-retrieval recipe); Dense projection + lift params train fully
- HALOLift module appended: hyperboloid lift, learnable curvature (init 0.1) and
  learnable temperature (CLIP-style logit_scale)
- token similarity: negative Lorentz distance; KD scores scaled by exp(logit_scale)
  with pylate's min-max score normalization DISABLED (it absorbs any positive
  scaling, which would freeze the learnable temperature)
"""
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed.tensor  # noqa: F401  (peft probes DTensor; torch 2.11 needs the explicit import)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from pylate import losses, models, utils
from pylate.losses.contrastive import extract_skiplist_mask
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from beirspeed.halo import (
    HALOLift,
    cosine_kd_scores,
    inbatch_nce_scores,
    lorentz_kd_scores,
)


class LorentzDistillation(losses.Distillation):
    """KD loss over HALO geometry. score_mode="lorentz": pure negative Lorentz
    distance. score_mode="hybrid" (HALO mainline): 0.5 * KL(cosine branch) +
    0.5 * KL(lorentz branch), mirroring HALO's hybrid loss mixing.

    KD branch: student scores z-scored per candidate list, then a learnable
    kd_scale — standardization removes the global-scale degree of freedom
    (runs 3/4 collapsed to the hyperboloid vertex through that channel).
    InfoNCE branch (w_nce > 0): in-batch contrastive over RAW eval-consistent
    scores x logit_scale — absolute distance scale receives real gradient
    pressure, so the geometry's magnitude structure is part of the objective."""

    def __init__(self, model, score_mode="lorentz", w_kd=0.5, w_nce=0.5):
        super().__init__(model=model, normalize_scores=False)
        self.score_mode = score_mode
        self.w_kd = w_kd
        self.w_nce = w_nce

    @staticmethod
    def _zscore(scores):
        return (scores - scores.mean(dim=-1, keepdim=True)) / (
            scores.std(dim=-1, keepdim=True) + 1e-6)

    def forward(self, sentence_features, labels):
        q_out = self.model(sentence_features[0])
        d_out = self.model(sentence_features[1])
        q = q_out["token_embeddings"]
        docs = d_out["token_embeddings"].view(q.size(0), -1, *d_out["token_embeddings"].shape[1:])

        inner = self.model if hasattr(self.model, "skiplist") else self.model.module
        masks = extract_skiplist_mask(sentence_features=sentence_features, skiplist=inner.skiplist)
        documents_mask = masks[1].view(q.size(0), -1, *masks[1].shape[1:])
        queries_mask = None if inner.do_query_expansion else masks[0]

        curv = q_out["halo_curv"]
        nce_scale, kd_scale = q_out["halo_logit_scale"], q_out["halo_kd_scale"]
        one = torch.ones((), device=q.device, dtype=torch.float32)
        log_teacher = torch.nn.functional.log_softmax(labels, dim=-1)

        lor = lorentz_kd_scores(q, docs, curv, one,
                                queries_mask=queries_mask, documents_mask=documents_mask)
        kd_loss = self.loss_function(
            torch.nn.functional.log_softmax(kd_scale * self._zscore(lor), dim=-1), log_teacher)
        if self.score_mode == "hybrid":
            cos = cosine_kd_scores(q, docs, one,
                                   queries_mask=queries_mask, documents_mask=documents_mask)
            cos_loss = self.loss_function(
                torch.nn.functional.log_softmax(kd_scale * self._zscore(cos), dim=-1), log_teacher)
            kd_loss = 0.5 * cos_loss + 0.5 * kd_loss
        if self.w_nce <= 0:
            return kd_loss

        n_docs_per_query = docs.size(1)
        nce = inbatch_nce_scores(
            q, d_out["token_embeddings"], curv, nce_scale, self.score_mode,
            queries_mask=queries_mask, documents_mask=masks[1])
        targets = labels.argmax(dim=-1) + torch.arange(
            q.size(0), device=q.device) * n_docs_per_query
        nce_loss = torch.nn.functional.cross_entropy(nce, targets)
        return self.w_kd * kd_loss + self.w_nce * nce_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--curv-init", type=float, default=0.1)
    ap.add_argument("--score-mode", choices=["lorentz", "hybrid"], default="hybrid")
    ap.add_argument("--w-kd", type=float, default=0.5)
    ap.add_argument("--w-nce", type=float, default=0.5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=0)
    args = ap.parse_args()

    model = models.ColBERT(
        model_name_or_path=args.model_dir, embedding_size=128, document_length=300)
    model[0].auto_model = get_peft_model(
        model[0].auto_model,
        LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                   target_modules=["query", "key", "value", "dense"], bias="none"))
    model.append(HALOLift(curv_init=args.curv_init, score_mode=args.score_mode))

    rank0 = int(os.environ.get("RANK", "0")) == 0
    if rank0:
        total = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cfg = model[0].auto_model.config
        print("attn=%s trainable=%.2fM/%.1fM (%.1f%%) curv_init=%g lr=%g" % (
            getattr(cfg, "_attn_implementation", "?"), train_p / 1e6, total / 1e6,
            100.0 * train_p / total, args.curv_init, args.lr), flush=True)

    train = load_dataset(path=args.data_dir, name="train", split="train")
    queries = load_dataset(path=args.data_dir, name="queries", split="train")
    documents = load_dataset(path=args.data_dir, name="documents", split="train")
    train.set_transform(utils.KDProcessing(queries=queries, documents=documents).transform)

    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=True,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        logging_steps=10,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train,
        loss=LorentzDistillation(model=model, score_mode=args.score_mode,
                                 w_kd=args.w_kd, w_nce=args.w_nce),
        data_collator=utils.ColBERTCollator(model.tokenize))

    import time
    t0 = time.time()
    trainer.train()
    if rank0:
        lift = model[-1]
        dt, steps = time.time() - t0, trainer.state.global_step
        print("steps=%d time=%.0fs %.2fs/step peak_mem_gb=%.1f curv=%.4f logit_scale=%.3f kd_scale=%.3f" % (
            steps, dt, dt / max(steps, 1), torch.cuda.max_memory_allocated() / 2**30,
            lift.curv().item(), lift.logit_scale.item(), lift.kd_scale.item()), flush=True)
        model[0].auto_model = model[0].auto_model.merge_and_unload()
        model.save_pretrained(os.path.join(args.out_dir, "final"))


if __name__ == "__main__":
    main()
