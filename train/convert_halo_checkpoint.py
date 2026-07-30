"""Convert an adapter-form HALO trainer checkpoint into a merged, load-anywhere model.

Trainer checkpoints save the PEFT-wrapped Transformer (adapter + resized-embedding
weights only). Plain loading then fails: the base bert has vocab 30522 while pylate's
ColBERT adds the [Q]/[D] marker tokens (30524). Rebuild the training-time module
stack in the same order (resize happens inside ColBERT init), restore all weights,
merge LoRA, and save a standalone model.
"""
import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: F401
import torch.distributed.tensor  # noqa: F401
import safetensors.torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from pylate import models

from beirspeed.halo import HALOLift, find_lift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="original bert-base snapshot")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    args = ap.parse_args()

    model = models.ColBERT(model_name_or_path=args.base, embedding_size=128, document_length=300)
    model[0].auto_model = get_peft_model(
        model[0].auto_model,
        LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                   target_modules=["query", "key", "value", "dense"], bias="none"))

    adapter_sd = safetensors.torch.load_file(os.path.join(args.checkpoint, "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model[0].auto_model, adapter_sd)
    unexpected = [k for k in getattr(result, "unexpected_keys", []) if "lora" in k.lower()]
    assert not unexpected, "unexpected adapter keys: %s" % unexpected[:5]

    dense_sd = safetensors.torch.load_file(os.path.join(args.checkpoint, "1_Dense", "model.safetensors"))
    model[1].load_state_dict(dense_sd)
    model.append(HALOLift.load(os.path.join(args.checkpoint, "2_HALOLift")))

    model[0].auto_model = model[0].auto_model.merge_and_unload()
    model.save_pretrained(args.out)
    lift = find_lift(model)
    print("converted %s -> %s | curv=%.4f logit_scale=%.3f" % (
        args.checkpoint, args.out, lift.curv().item(), lift.logit_scale.item()))


if __name__ == "__main__":
    main()
