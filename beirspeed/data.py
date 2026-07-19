"""Loaders for BEIR-format datasets: corpus.jsonl, queries.jsonl, qrels/*.tsv."""
import csv
import itertools
import json
import os


def corpus_path(data_root, dataset):
    return os.path.join(data_root, dataset, "corpus.jsonl")


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            n += block.count(b"\n")
    # count a trailing line without newline
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                n += 1
    return n


def iter_corpus(data_root, dataset, start, end):
    """Yield (doc_id, "title text") for corpus lines in [start, end)."""
    with open(corpus_path(data_root, dataset), encoding="utf-8") as f:
        for line in itertools.islice(f, start, end):
            d = json.loads(line)
            text = (d.get("title") or "").strip()
            body = (d.get("text") or "").strip()
            text = (text + " " + body).strip() if text else body
            yield str(d["_id"]), text


def load_queries(data_root, dataset, qids=None):
    out = {}
    with open(os.path.join(data_root, dataset, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qid = str(d["_id"])
            if qids is None or qid in qids:
                out[qid] = d["text"]
    return out


def load_qrels(data_root, dataset, split):
    qrels = {}
    with open(os.path.join(data_root, dataset, "qrels", split + ".tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header[0].lower().replace("_", "-") == "query-id", header
        for qid, did, rel in reader:
            qrels.setdefault(str(qid), {})[str(did)] = int(rel)
    return qrels
