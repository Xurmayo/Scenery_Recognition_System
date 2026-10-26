"""Build the real-image dataset from Places365 (+ your own photos) and write data/manifest.csv.

Quick mode (default, ~0.5 GB download):
    python scripts/prepare_data.py
    -> uses the Places365 *validation* set (100 images per class) split 60/15/25
       into train/val/test. Small on purpose: it mimics a low-data A1 dataset,
       which is exactly where synthetic "gap filling" with MaskGen should help.

Bigger mode (~24 GB download, much better accuracy):
    python scripts/prepare_data.py --train-source train-standard --train-per-class 500
    -> train images come from Places365-Standard train; val/test from the
       Places365 validation set (40/60 per class).

Your own photos (recommended for the defence, "in-the-wild" test):
    put them in data/own/<class>/*.jpg  (e.g. data/own/parking_lot/IMG_001.jpg)
    they are added with split "own_test" and never used for training.
"""
import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATA_DIR, MANIFEST, PLACES365_NAMES, ROOT  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_places(root: Path, split: str):
    """Return {class_name: [image paths]} for our classes from a Places365 split."""
    from torchvision.datasets import Places365

    root.mkdir(parents=True, exist_ok=True)
    try:
        ds = Places365(str(root), split=split, small=True, download=True)
    except RuntimeError as e:  # torchvision refuses to re-download over existing files
        print(f"[info] {e} -> using the files already on disk")
        ds = Places365(str(root), split=split, small=True, download=False)

    wanted = {PLACES365_NAMES[c]: c for c in CLASSES}
    idx_to_ours = {}
    for name, idx in ds.class_to_idx.items():
        key = name if name.startswith("/") else "/" + name
        if key in wanted:
            idx_to_ours[idx] = wanted[key]
    missing = set(CLASSES) - set(idx_to_ours.values())
    if missing:
        raise SystemExit(f"Classes not found in Places365: {missing}")

    out = {c: [] for c in CLASSES}
    for path, target in ds.imgs:
        if target in idx_to_ours:
            out[idx_to_ours[target]].append(Path(path))
    return out


def copy_split(items, split, rows):
    for cls, paths in items.items():
        dst_dir = DATA_DIR / "real" / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dst = dst_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
            rows.append({"path": dst.relative_to(ROOT).as_posix(), "label": cls, "split": split, "source": "places365"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", choices=["val", "train-standard"], default="val")
    ap.add_argument("--train-per-class", type=int, default=500, help="only used with --train-source train-standard")
    ap.add_argument("--places-root", default=str(DATA_DIR / "raw" / "places365"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.places_root)
    rows = []

    val_imgs = load_places(root, "val")
    if args.train_source == "val":
        train, val, test = {}, {}, {}
        for c, paths in val_imgs.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            n = len(paths)
            n_tr, n_va = int(0.60 * n), int(0.15 * n)
            train[c], val[c], test[c] = paths[:n_tr], paths[n_tr:n_tr + n_va], paths[n_tr + n_va:]
    else:
        tr_imgs = load_places(root, "train-standard")
        train = {}
        for c, paths in tr_imgs.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            train[c] = paths[: args.train_per_class]
        val, test = {}, {}
        for c, paths in val_imgs.items():
            paths = sorted(paths)
            rng.shuffle(paths)
            val[c], test[c] = paths[:40], paths[40:]

    copy_split(train, "train", rows)
    copy_split(val, "val", rows)
    copy_split(test, "test", rows)

    # Your own photos -> separate in-the-wild test split.
    own_dir = DATA_DIR / "own"
    for c in CLASSES:
        (own_dir / c).mkdir(parents=True, exist_ok=True)
        for p in sorted((own_dir / c).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"path": p.relative_to(ROOT).as_posix(), "label": c, "split": "own_test", "source": "own"})

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "split", "source"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter((r["split"], r["label"]) for r in rows)
    print(f"\nWrote {MANIFEST} ({len(rows)} images)")
    splits = ["train", "val", "test", "own_test"]
    print(f"{'class':<12}" + "".join(f"{s:>10}" for s in splits))
    for c in CLASSES:
        print(f"{c:<12}" + "".join(f"{counts[(s, c)]:>10}" for s in splits))
    if not any(r["split"] == "own_test" for r in rows):
        print("\n[tip] Add 10-20 of your own photos per class to data/own/<class>/ and re-run "
              "for an in-the-wild test set (good material for the defence).")


if __name__ == "__main__":
    main()
