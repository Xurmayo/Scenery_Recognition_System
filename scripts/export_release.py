"""Pack everything the demo app needs into the repository, so it runs right after `git clone`.

    python scripts/export_release.py

Writes (all small enough for GitHub):
    weights/<run>.pt           half-precision copies of the final and baseline models (~43 MB and ~22 MB)
    weights/<run>.json         their metrics (used by the app's "most confused with" logic)
    assets/examples/<class>.jpg  one real Places365 test image per class (correctly classified, resized)
    assets/synthetic/...       4 quality-checked MaskGen images per class + synthetic_meta.csv

Half precision halves the file size; the script re-evaluates the exported copy on the test set to show the
accuracy is unchanged.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, ROOT, RUNS_DIR, SYNTH_META  # noqa: E402
from src.data import SceneDataset, build_transforms, load_image, read_rows  # noqa: E402
from src.models import load_checkpoint  # noqa: E402

WEIGHTS_DIR = ROOT / "weights"
ASSETS_DIR = ROOT / "assets"


@torch.no_grad()
def accuracy(model, meta, rows, device):
    tf = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
    dl = DataLoader(SceneDataset(rows, tf), batch_size=32)
    correct = 0
    for x, y in dl:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(rows)


def export_weights(runs, device):
    WEIGHTS_DIR.mkdir(exist_ok=True)
    test_rows = read_rows("test")
    for run in runs:
        src = RUNS_DIR / run / "best.pt"
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        half = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()}
        dst = WEIGHTS_DIR / f"{run}.pt"
        torch.save({"state_dict": half, "meta": ckpt["meta"]}, dst)
        shutil.copy2(RUNS_DIR / run / "metrics.json", WEIGHTS_DIR / f"{run}.json")

        full, meta = load_checkpoint(src, device)
        small, _ = load_checkpoint(dst, device)
        a_full, a_small = accuracy(full, meta, test_rows, device), accuracy(small, meta, test_rows, device)
        print(f"{run:22s} {src.stat().st_size / 1e6:6.1f} MB -> {dst.stat().st_size / 1e6:5.1f} MB | "
              f"test acc full {a_full:.3f}, exported {a_small:.3f}")


@torch.no_grad()
def export_examples(judge_run, device):
    """One clearly classified real test image per class, so the demo examples look sensible."""
    out_dir = ASSETS_DIR / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, meta = load_checkpoint(RUNS_DIR / judge_run / "best.pt", device)
    tf = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
    rows = read_rows("test")
    for c in CLASSES:
        best, best_p = None, -1.0
        for r in [r for r in rows if r["label"] == c]:
            img = load_image(ROOT / r["path"])
            p = torch.softmax(model(tf(img).unsqueeze(0).to(device)).float(), 1)[0, CLASSES.index(c)].item()
            if p > best_p:
                best, best_p = img, p
        best.thumbnail((512, 512))
        best.save(out_dir / f"{c}.jpg", quality=90)
    print(f"examples -> {out_dir} ({len(CLASSES)} images)")


def export_synthetic(per_class):
    out_dir = ASSETS_DIR / "synthetic"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    with open(SYNTH_META) as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in CLASSES and (ROOT / r["path"]).exists()]
    quality = SYNTH_META.parent / "synthetic_quality.csv"
    if quality.exists():
        with open(quality) as f:
            rejected = {q["path"] for q in csv.DictReader(f) if q["quality"] == "reject"}
        rows = [r for r in rows if r["path"] not in rejected]
    kept = []
    for c in CLASSES:
        first = {}  # one image per distinct scene description
        for r in rows:
            if r["label"] == c:
                first.setdefault(r["prompt"].split(",")[0], r)
        distinct = list(first.values())
        step = max(1, len(distinct) // per_class)
        for r in distinct[::step][:per_class]:
            dst = out_dir / c / Path(r["path"]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / r["path"], dst)  # keeps the embedded "Synthetic: true" PNG metadata
            kept.append(dict(r, path=dst.relative_to(ROOT).as_posix()))
    with open(out_dir / "synthetic_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    print(f"synthetic samples -> {out_dir} ({len(kept)} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["vit_real_synf", "baseline_resnet18"])
    ap.add_argument("--synthetic-per-class", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    export_weights(args.runs, device)
    export_examples(args.runs[0], device)
    export_synthetic(args.synthetic_per_class)
    total = sum(p.stat().st_size for p in list(WEIGHTS_DIR.rglob("*")) + list(ASSETS_DIR.rglob("*")) if p.is_file())
    print(f"\nTotal added to the repository: {total / 1e6:.0f} MB. Commit weights/ and assets/.")


if __name__ == "__main__":
    main()
