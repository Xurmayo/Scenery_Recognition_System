"""Train one model and evaluate it on the real test set (and your own photos, if any).

Examples (the full set of runs used in the report is in run_all.sh / run_all.bat):
    python train.py --arch resnet18  --run baseline_resnet18                      # "A1" baseline
    python train.py --arch vit_small --run vit_real                               # SOTA classifier, real data only
    python train.py --arch vit_small --run vit_real_syn --synthetic-frac 1.0      # + MaskGen synthetic scenes
    python train.py --arch vit_small --run vit_real_syn_noaug --synthetic-frac 1.0 --no-aug   # ablation

Outputs in runs/<run>/: best.pt, history.json, metrics.json, curves.png, confusion.png
Test sets contain REAL images only; synthetic images are only ever used for training.
"""
import argparse
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, INDOOR, RUNS_DIR  # noqa: E402
from src.data import SceneDataset, build_transforms, read_rows, synthetic_rows  # noqa: E402
from src.models import build_model, load_checkpoint, save_checkpoint  # noqa: E402


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, preds, gts = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        tot_loss += loss.item() * x.size(0)
        preds.append(logits.argmax(1).cpu())
        gts.append(y.cpu())
    preds, gts = torch.cat(preds).numpy(), torch.cat(gts).numpy()
    return tot_loss / len(gts), accuracy_score(gts, preds), f1_score(gts, preds, average="macro"), preds, gts


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        gts.append(y)
    return torch.cat(preds).numpy(), torch.cat(gts).numpy()


def summarize(preds, gts):
    labels = list(range(len(CLASSES)))
    io = np.array([c in INDOOR for c in CLASSES])
    rep = classification_report(gts, preds, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0)
    return {
        "n": int(len(gts)),
        "accuracy": float(accuracy_score(gts, preds)),
        "macro_f1": float(f1_score(gts, preds, average="macro", labels=labels, zero_division=0)),
        "indoor_outdoor_accuracy": float((io[preds] == io[gts]).mean()),
        "per_class_f1": {c: float(rep[c]["f1-score"]) for c in CLASSES},
        "confusion_matrix": confusion_matrix(gts, preds, labels=labels).tolist(),
    }


@torch.no_grad()
def measure_latency(model, device, size, n=50):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def plot_curves(hist, path, title):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, hist["train_loss"], label="train")
    ax[0].plot(ep, hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, hist["train_acc"], label="train acc")
    ax[1].plot(ep, hist["val_acc"], label="val acc")
    ax[1].plot(ep, hist["val_f1"], label="val macro-F1", linestyle="--")
    ax[1].set_title("Accuracy / F1"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1); ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vit_small", choices=["resnet18", "vit_small", "vit_base"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--synthetic-frac", type=float, default=0.0, help="share of generated images to add (0 = real only)")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--filter-synthetic", action="store_true", help="drop synthetic images rejected by scripts/filter_synthetic.py")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 5e-4 (resnet) / 1e-4 (vit)")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train images")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only: random init")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS_DIR / args.run
    out.mkdir(parents=True, exist_ok=True)

    model, meta = build_model(args.arch, pretrained=not args.no_pretrained)
    model.to(device)
    size = meta["img_size"]
    tf_train = build_transforms(meta["mean"], meta["std"], size, train=True, augment=not args.no_aug)
    tf_eval = build_transforms(meta["mean"], meta["std"], size, train=False)

    real_train = read_rows("train")
    syn = synthetic_rows(args.synthetic_frac, seed=args.seed, drop_rejected=args.filter_synthetic)
    train_rows = real_train + syn
    if args.limit:
        random.Random(args.seed).shuffle(train_rows)
        train_rows = train_rows[: args.limit]
    loaders = {}
    dl = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    loaders["train"] = DataLoader(SceneDataset(train_rows, tf_train), shuffle=True, drop_last=len(train_rows) > args.batch_size, **dl)
    for split in ["val", "test", "own_test"]:
        rows = read_rows(split)
        if rows:
            loaders[split] = DataLoader(SceneDataset(rows, tf_eval), shuffle=False, **dl)

    counts = Counter(r["source"] for r in train_rows)
    print(f"Device: {device} | train images: {len(train_rows)} {dict(counts)} | "
          + " | ".join(f"{k}: {len(v.dataset)}" for k, v in loaders.items() if k != "train"))

    lr = args.lr or (5e-4 if args.arch == "resnet18" else 1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_f1, t_start = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta, _, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer, scheduler, scaler)
        vl, va, vf, _, _ = run_epoch(model, loaders["val"], device, criterion)
        for k, v in zip(hist, [tl, ta, vl, va, vf]):
            hist[k].append(float(v))
        flag = ""
        if vf > best_f1:
            best_f1 = vf
            save_checkpoint(out / "best.pt", model, meta)
            flag = "  *best*"
        print(f"epoch {ep:2d}/{args.epochs}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f} F1 {vf:.3f}{flag}")
    train_minutes = (time.time() - t_start) / 60
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    plot_curves(hist, out / "curves.png", args.run)

    model, _ = load_checkpoint(out / "best.pt", device)
    metrics = {
        "run": args.run,
        "arch": args.arch,
        "timm_name": meta["timm_name"],
        "synthetic_frac": args.synthetic_frac,
        "augment": not args.no_aug,
        "synthetic_filtered": args.filter_synthetic,
        "train_images": dict(counts),
        "epochs": args.epochs,
        "lr": lr,
        "best_val_macro_f1": float(best_f1),
        "train_minutes": round(train_minutes, 2),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "latency_ms_batch1": round(measure_latency(model, device, size), 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU",
    }
    for split in ["test", "own_test"]:
        if split in loaders:
            p, g = predict(model, loaders[split], device)
            metrics[split] = summarize(p, g)
            plot_confusion(metrics[split]["confusion_matrix"], out / f"confusion_{split}.png", f"{args.run} ({split})")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    t = metrics["test"]
    print(f"\nTEST  acc {t['accuracy']:.3f}  macro-F1 {t['macro_f1']:.3f}  indoor/outdoor acc {t['indoor_outdoor_accuracy']:.3f}"
          f"  | latency {metrics['latency_ms_batch1']} ms  | saved to {out}")


if __name__ == "__main__":
    main()
