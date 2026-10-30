"""Build the nearest-neighbour index behind the app's Retrieval tab and measure retrieval quality (P@k).

For every model the script embeds the real training images (the penultimate-layer features, L2-normalised), saves them
with small thumbnails, then uses every real test image as a query against the training images:

    P@k = share of the k nearest training images that have the same scene class as the query (averaged over queries)

    python scripts/build_retrieval_index.py                              # final ViT and the ResNet-18 baseline
    python scripts/build_retrieval_index.py --runs vit_real_synf --k 1 5 10

`python app.py` calls build(..., only_missing=True) at startup, so the index and thumbnails appear on their own once
data/ exists. Needs data/manifest.csv (scripts/prepare_data.py). Writes to assets/retrieval/:
    <run>.npz            embeddings, labels and thumbnail names of the training images (small, safe to commit)
    <run>_metrics.json   P@k on the real test set (and on your own photos, if data/own has any)
    thumbs/NNN.jpg       one thumbnail per training image, shared by all models (git-ignored: Places365 photos)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, ROOT, RUNS_DIR  # noqa: E402
from src.data import build_transforms, load_image, read_rows  # noqa: E402
from src.models import load_checkpoint  # noqa: E402

OUT_DIR = ROOT / "assets" / "retrieval"
THUMB_DIR = OUT_DIR / "thumbs"
THUMB_SIZE = 192
DEFAULT_RUNS = ["vit_real_synf", "baseline_resnet18"]


def checkpoint_path(run):
    for p in (RUNS_DIR / run / "best.pt", ROOT / "weights" / f"{run}.pt"):
        if p.exists():
            return p
    raise SystemExit(f"No checkpoint for {run!r} in runs/ or weights/")


@torch.no_grad()
def embed(model, meta, rows, device, batch_size=32):
    tf = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
    out = []
    for b in range(0, len(rows), batch_size):
        x = torch.stack([tf(load_image(ROOT / r["path"])) for r in rows[b:b + batch_size]]).to(device)
        f = model.forward_head(model.forward_features(x), pre_logits=True).float()
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()


def precision_at_k(query_emb, query_labels, index_emb, index_labels, ks):
    """Mean share of same-class images among the k most similar training images (cosine similarity)."""
    sims = query_emb @ index_emb.T
    order = np.argsort(-sims, axis=1)
    same = index_labels[order] == query_labels[:, None]
    return {f"P@{k}": float(same[:, :k].mean()) for k in ks}


def save_thumbnails(train_rows):
    """Writes the thumbnails that do not exist yet and returns the name of every one."""
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    names = []
    for i, r in enumerate(train_rows):
        name = f"{i:03d}.jpg"
        names.append(name)
        if not (THUMB_DIR / name).exists():
            img = load_image(ROOT / r["path"])
            img.thumbnail((THUMB_SIZE, THUMB_SIZE))
            img.save(THUMB_DIR / name, quality=82)
    return names


def build(runs=DEFAULT_RUNS, ks=(1, 5, 10), only_missing=False):
    """Index + P@k for each run. With only_missing, finished runs are kept and only absent files are created."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    todo = [r for r in runs if not (only_missing and (OUT_DIR / f"{r}.npz").exists()
                                    and (OUT_DIR / f"{r}_metrics.json").exists())]
    train = [r for r in read_rows("train") if r["source"] != "synthetic"]  # the index holds real photos only
    thumbs_missing = not all((THUMB_DIR / f"{i:03d}.jpg").exists() for i in range(len(train)))
    if not todo and not thumbs_missing:
        return
    print(f"Retrieval index: {len(train)} real training photos, building {todo or 'thumbnails only'} ...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thumbs = save_thumbnails(train)
    labels = np.array([r["label"] for r in train])
    queries = {s: read_rows(s) for s in ("test", "own_test")}
    queries = {s: rows for s, rows in queries.items() if rows}

    for run in todo:
        model, meta = load_checkpoint(checkpoint_path(run), device)
        index_emb = embed(model, meta, train, device)
        np.savez_compressed(OUT_DIR / f"{run}.npz", emb=index_emb.astype(np.float16), labels=labels,
                            thumbs=np.array(thumbs), paths=np.array([r["path"] for r in train]))

        metrics = {"run": run, "index_size": len(train), "embedding_dim": int(index_emb.shape[1]),
                   "random_baseline": float(sum(np.mean(labels == c) ** 2 for c in CLASSES))}
        for split, rows in queries.items():
            q_emb = embed(model, meta, rows, device)
            q_labels = np.array([r["label"] for r in rows])
            metrics[split] = {"n": len(rows), **precision_at_k(q_emb, q_labels, index_emb, labels, ks)}
        (OUT_DIR / f"{run}_metrics.json").write_text(json.dumps(metrics, indent=1))
        print(run, json.dumps({s: v for s, v in metrics.items() if s in queries}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    args = ap.parse_args()
    build(args.runs, args.k)


if __name__ == "__main__":
    main()
