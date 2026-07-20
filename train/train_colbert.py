"""Knowledge-distillation training of a ColBERT on bert-base-uncased.

Data: lightonai/ms-marco-en-bge-gemma (query + ~32 scored docs per example,
teacher = bge-reranker-v2-gemma, negatives mined by ColBERT). Recipe follows
PyLate's GTE-ModernColBERT example, adapted to bert-base.

Run under torchrun for multi-GPU:
  torchrun --nproc_per_node=4 train/train_colbert.py --data-dir ... --out-dir ...
"""
import argparse
import os

# forked dataloader workers deadlock with the rust tokenizer threadpool
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from pylate import losses, models, utils
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="bert-base-uncased snapshot")
    ap.add_argument("--data-dir", required=True, help="local ms-marco-en-bge-gemma snapshot")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16, help="per device")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--embedding-size", type=int, default=128)
    ap.add_argument("--doc-length", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=-1, help="override for smoke runs")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--no-math-sdp", action="store_true",
                    help="forbid the math SDPA fallback (memory blowup); error instead")
    args = ap.parse_args()
    if args.no_math_sdp:
        torch.backends.cuda.enable_math_sdp(False)

    model = models.ColBERT(
        model_name_or_path=args.model_dir,
        embedding_size=args.embedding_size,
        document_length=args.doc_length,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        cfg = model[0].auto_model.config
        print("attn_implementation=%s dtype=bf16 batch/device=%d lr=%g" % (
            getattr(cfg, "_attn_implementation", "?"), args.batch_size, args.lr), flush=True)

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
        save_strategy="epoch",
        save_total_limit=None,
        logging_steps=10,
        dataloader_num_workers=args.num_workers,
        report_to=[],
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        loss=losses.Distillation(model=model),
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        batch = next(iter(trainer.get_train_dataloader()))
        for k, v in batch.items():
            if hasattr(v, "shape"):
                print("batch %s %s" % (k, tuple(v.shape)), flush=True)

    import time
    t0 = time.time()
    trainer.train()
    if int(os.environ.get("RANK", "0")) == 0:
        dt = time.time() - t0
        steps = trainer.state.global_step
        print("steps=%d time=%.0fs %.2fs/step peak_mem_gb=%.1f" % (
            steps, dt, dt / max(steps, 1), torch.cuda.max_memory_allocated() / 2**30))
        model.save_pretrained(os.path.join(args.out_dir, "final"))


if __name__ == "__main__":
    main()
