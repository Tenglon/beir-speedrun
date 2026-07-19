#!/bin/bash
# Run on any machine WITH internet (e.g. your laptop) to stage everything an
# offline cluster needs: BEIR dataset zips, the encoder model, and a cp312
# manylinux wheelhouse for torch 2.11.0+cu126 + transformers + pytrec_eval.
# Then:  rsync -a $STAGE/datasets/ user@cluster:$BEIR_ROOT/staging/datasets/
#        rsync -a $STAGE/models/   user@cluster:$BEIR_ROOT/models/
#        rsync -a $STAGE/wheelhouse312/ user@cluster:$BEIR_ROOT/staging/wheelhouse312/
#
# Offline-packaging pitfalls this script works around:
# - NVIDIA wheels are NOT on PyPI anymore (placeholder 0.0.1.dev5 only); they live
#   on pypi.nvidia.com and download.pytorch.org, with tags like manylinux_2_18/_2_27.
# - torch 2.11 pulls most CUDA libs through the cuda-toolkit meta-package whose
#   extras carry `sys_platform == linux` markers; pip evaluates markers against the
#   RUNNING platform, so on macOS they silently vanish — hence the explicit list.
# - pip's `--implementation cp` silently rejects py3-none-manylinux wheels; do not use it.
# - Needs pip >= 22.2 for multiple --platform flags: python3 -m pip install -U pip
set -uo pipefail
STAGE="${STAGE:-$PWD/stage}"
mkdir -p "$STAGE/datasets" "$STAGE/models/bert-base-uncased" "$STAGE/wheelhouse312"
fail=0

BEIR_BASE="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS="scifact nfcorpus arguana fiqa scidocs quora trec-covid webis-touche2020 cqadupstack hotpotqa msmarco fever climate-fever dbpedia-entity nq"
for d in $DATASETS; do
  out="$STAGE/datasets/$d.zip"
  [ -f "$out.done" ] && continue
  echo "=== dataset $d ==="
  curl -sSL --retry 3 -C - -o "$out" "$BEIR_BASE/$d.zip" \
    && unzip -tq "$out" >/dev/null && touch "$out.done" || { echo "FAIL $d"; fail=1; }
done

echo "=== model bert-base-uncased ==="
for f in config.json vocab.txt tokenizer.json tokenizer_config.json model.safetensors; do
  out="$STAGE/models/bert-base-uncased/$f"
  [ -f "$out.done" ] && continue
  curl -sSL --retry 3 -C - -o "$out" \
    "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/$f" \
    && touch "$out.done" || { echo "FAIL $f"; fail=1; }
done

echo "=== wheelhouse (cp312, torch 2.11.0+cu126) ==="
W="$STAGE/wheelhouse312"
PIPARGS="--only-binary=:all: --python-version 312 --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_18_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64"
IDX="--index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.nvidia.com --extra-index-url https://pypi.org/simple"
python3 -m pip download -q --no-deps $PIPARGS $IDX -d "$W" "torch==2.11.0+cu126" || fail=1
# torch's Linux-only deps, resolved by hand from its METADATA (see pitfalls above);
# versions follow the cuda-toolkit[...]==12.6.3 extras plus torch's direct pins.
for req in \
  "nvidia-cublas-cu12==12.6.4.1.*" "nvidia-cuda-runtime-cu12==12.6.77.*" \
  "nvidia-cufft-cu12==11.3.0.4.*" "nvidia-cufile-cu12==1.11.1.6.*" \
  "nvidia-cuda-cupti-cu12==12.6.80.*" "nvidia-curand-cu12==10.3.7.77.*" \
  "nvidia-cusolver-cu12==11.7.1.2.*" "nvidia-cusparse-cu12==12.5.4.2.*" \
  "nvidia-nvjitlink-cu12==12.6.85.*" "nvidia-cuda-nvrtc-cu12==12.6.85.*" \
  "nvidia-nvtx-cu12==12.6.77.*" "nvidia-cudnn-cu12==9.10.2.21" \
  "nvidia-cusparselt-cu12==0.7.1" "nvidia-nvshmem-cu12==3.4.5" \
  "nvidia-nccl-cu12==2.28.9" "triton==3.6.0" "cuda-bindings<13,>=12.9.4" "cuda-toolkit==12.6.3"; do
  python3 -m pip download -q --no-deps $PIPARGS $IDX -d "$W" "$req" || { echo "FAIL $req"; fail=1; }
done
python3 -m pip download -q $PIPARGS $IDX -d "$W" \
  "transformers==4.46.3" pytrec-eval-terrier numpy setuptools \
  filelock "typing-extensions>=4.10" "sympy>=1.13.3" networkx jinja2 fsspec importlib-metadata \
  || { echo "FAIL stack"; fail=1; }

du -sh "$STAGE"/*
echo "overall_fail=$fail"
