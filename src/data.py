"""Datasets and transforms."""
import csv
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import CLASSES, MANIFEST, ROOT, SYNTH_META


def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # phone photos come rotated otherwise
    return img.convert("RGB")


def read_rows(split=None):
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def synthetic_rows(frac=1.0, seed=0, classes=None, drop_rejected=False):
    if frac <= 0 or not Path(SYNTH_META).exists():
        return []
    with open(SYNTH_META) as f:
        rows = [dict(r, split="train", source="synthetic") for r in csv.DictReader(f)]
    classes = set(classes or CLASSES)  # ignores images of classes no longer in config.CLASSES
    rows = [r for r in rows if (ROOT / r["path"]).exists() and r["label"] in classes]
    quality = Path(SYNTH_META).parent / "synthetic_quality.csv"
    if drop_rejected:
        if not quality.exists():
            raise SystemExit("--filter-synthetic needs data/synthetic/synthetic_quality.csv: run scripts/filter_synthetic.py first")
        with open(quality) as f:
            rejected = {q["path"] for q in csv.DictReader(f) if q["quality"] == "reject"}
        rows = [r for r in rows if r["path"] not in rejected]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: int(round(len(rows) * frac))]


def build_transforms(mean, std, size=224, train=True, augment=True):
    norm = T.Normalize(mean, std)
    if train and augment:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.2, 0.02),
            T.ToTensor(), norm,
        ])
    if train:  # ablation: no augmentation
        return T.Compose([T.Resize((size, size)), T.ToTensor(), norm])
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), norm])


class SceneDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(ROOT / r["path"])
        return self.transform(img), self.class_to_idx[r["label"]]

    def targets(self):
        return torch.tensor([self.class_to_idx[r["label"]] for r in self.rows])
