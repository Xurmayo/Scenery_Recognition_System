"""Build every table and figure the report needs from the finished runs.

    python make_figures.py
    python make_figures.py --baseline baseline_resnet18 --real vit_real --sota vit_real_syn

Writes to figures/:
    results.md / results.csv        metrics table (baseline vs SOTA vs ablation)
    confusion_shift.png             row-normalized confusion matrices + their difference
    val_f1_curves.png               validation macro-F1 per epoch for all runs
    per_class_f1.png                per-class F1 bars
    explain_gallery.png             Grad-CAM (baseline) vs attention rollout (SOTA) on test images
    synthetic_samples.png           labelled MaskGen samples per class
    dataset_counts.md               dataset size / balance / synthetic share
"""
import argparse
import csv
import json
import random
import re
import textwrap
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, FIG_DIR, ROOT, RUNS_DIR, SYNTH_META  # noqa: E402
from src.data import build_transforms, load_image, read_rows  # noqa: E402
from src.explain import display_crop, make_explainers, overlay  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.watermark import stamp_synthetic  # noqa: E402


def load_runs():
    runs = {}
    for p in sorted(RUNS_DIR.glob("*/metrics.json")):
        m = json.loads(p.read_text())
        m["history"] = json.loads((p.parent / "history.json").read_text())
        runs[p.parent.name] = m
    return runs


def results_table(runs):
    cols = ["run", "arch", "synthetic", "aug", "train imgs (real+syn)", "test acc", "test macro-F1",
            "indoor/outdoor acc", "own-photo acc", "own-photo F1", "latency ms", "params M"]
    lines = []
    for name, m in runs.items():
        ti = m["train_images"]
        own = m.get("own_test")
        lines.append([
            name, m["arch"], f"{m['synthetic_frac']:.0%}" + (" (filtered)" if m.get("synthetic_filtered") else ""), "yes" if m["augment"] else "no",
            f"{ti.get('places365', 0)}+{ti.get('synthetic', 0)}",
            f"{m['test']['accuracy']:.3f}", f"{m['test']['macro_f1']:.3f}", f"{m['test']['indoor_outdoor_accuracy']:.3f}",
            f"{own['accuracy']:.3f}" if own else "-", f"{own['macro_f1']:.3f}" if own else "-",
            f"{m['latency_ms_batch1']}", f"{m['params_millions']}",
        ])
    with open(FIG_DIR / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(lines)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in lines]
    (FIG_DIR / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


SEED_SUFFIX = re.compile(r"_s\d+$")


def seed_table(runs):
    """Mean ± std over seeds: runs named <config>_s1, <config>_s2 ... are grouped with <config>."""
    groups = {}
    for name, m in runs.items():
        groups.setdefault(SEED_SUFFIX.sub("", name), []).append(m)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not groups:
        return
    def ms(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
    cols = ["config", "seeds", "test acc", "test macro-F1", "own-photo acc", "own-photo F1"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for k, ms_ in groups.items():
        row = [k, str(len(ms_)), ms([m["test"]["accuracy"] for m in ms_]), ms([m["test"]["macro_f1"] for m in ms_])]
        own = [m["own_test"] for m in ms_ if "own_test" in m]
        row += [ms([o["accuracy"] for o in own]), ms([o["macro_f1"] for o in own])] if own else ["-", "-"]
        lines.append("| " + " | ".join(row) + " |")
    (FIG_DIR / "results_seeds.md").write_text("\n".join(lines) + "\n")
    print("\nMean ± std over seeds:\n" + "\n".join(lines))


def norm_cm(m, split="test"):
    cm = np.array(m[split]["confusion_matrix"], dtype=float)
    return cm / cm.sum(1, keepdims=True).clip(min=1)


def confusion_shift(runs, a, b, split="test"):
    A, B = norm_cm(runs[a], split), norm_cm(runs[b], split)
    split_name = "Places365 test set" if split == "test" else "own photos"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, M, title, cmap, vmin in [(axes[0], A, a, "Blues", 0), (axes[1], B, b, "Blues", 0),
                                     (axes[2], B - A, f"shift: {b} − {a}", "RdBu", -1)]:
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=1)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{M[i, j]:+.2f}" if vmin < 0 else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
        ax.set_title(title, fontsize=10); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Confusion matrix shift on {split_name} (rows normalized; diagonal up = better, off-diagonal down = better)")
    fig.tight_layout()
    suffix = "" if split == "test" else "_own"
    fig.savefig(FIG_DIR / f"confusion_shift_{a}_vs_{b}{suffix}.png", dpi=150)
    plt.close(fig)


def curves(runs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.get_cmap("tab10").colors
    for k, (name, m) in enumerate(runs.items()):
        h, c = m["history"], colors[k % len(colors)]  # one colour per run, shared by its train and val lines
        ax[0].plot(range(1, len(h["val_f1"]) + 1), h["val_f1"], color=c, label=name)
        ax[1].plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], color=c, label=name)
        ax[1].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], color=c, linestyle="--")
    ax[0].set_title("Validation macro-F1"); ax[0].set_xlabel("epoch"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    ax[1].set_title("Loss (solid = train, dashed = val)"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_f1_curves.png", dpi=150)
    plt.close(fig)


def per_class(runs):
    names = list(runs)
    x = np.arange(len(CLASSES))
    w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, n in enumerate(names):
        ax.bar(x + k * w - 0.4 + w / 2, [runs[n]["test"]["per_class_f1"][c] for c in CLASSES], w, label=n)
    ax.set_xticks(x, CLASSES); ax.set_ylim(0, 1); ax.set_ylabel("F1 (real test set)"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)


def explain_gallery(baseline, sota, per_class=1, seed=0, split="own_test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not read_rows(split):
        split = "test"
    rows_by_cls = {}
    for r in read_rows(split):
        rows_by_cls.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    picks = [r for c in CLASSES for r in rng.sample(rows_by_cls.get(c, []), min(per_class, len(rows_by_cls.get(c, []))))]
    models = []
    for run in [baseline, sota]:
        ck = RUNS_DIR / run / "best.pt"
        if ck.exists():
            model, meta = load_checkpoint(ck, device)
            for method, explainer in make_explainers(model, meta).items():
                models.append((run, model, meta, method, explainer))
    if not models or not picks:
        return
    fig, axes = plt.subplots(len(picks), 1 + len(models), figsize=(3 * (1 + len(models)), 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(picks):
        img = load_image(ROOT / r["path"])
        axes[i, 0].imshow(display_crop(img)); axes[i, 0].set_title(f"true: {r['label']}", fontsize=9)
        for j, (run, model, meta, method, explainer) in enumerate(models, start=1):
            x = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)(img).unsqueeze(0).to(device)
            logits, heat = explainer(x)
            p = torch.softmax(logits.float(), 1)[0]
            k = int(p.argmax())
            axes[i, j].imshow(overlay(display_crop(img, meta["img_size"]), heat))
            axes[i, j].set_title(f"{run}\n{method}: {CLASSES[k]} ({p[k]:.2f})", fontsize=8,
                                 color="green" if CLASSES[k] == r["label"] else "red")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "explain_gallery.png", dpi=130)
    plt.close(fig)


def synthetic_grid(per_class=3):
    if not SYNTH_META.exists():
        return
    with open(SYNTH_META) as f:
        rows = list(csv.DictReader(f))
    by = {}
    for c in CLASSES:  # one image per distinct scene description, so each row shows variety
        rc = [r for r in rows if r["label"] == c and (ROOT / r["path"]).exists()]
        first = {}
        for r in rc:
            first.setdefault(r["prompt"].split(",")[0].rstrip("."), r)
        distinct = list(first.values())
        step = max(1, len(distinct) // per_class)
        by[c] = (distinct[::step] + rc)[:per_class]
    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(3.2 * per_class + 0.8, 3.5 * len(CLASSES)))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(CLASSES):
        for j in range(per_class):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(c.replace("_", " "), fontsize=12, fontweight="bold", labelpad=8)
            if j < len(by[c]):
                r = by[c][j]
                ax.imshow(stamp_synthetic(load_image(ROOT / r["path"])))
                short = r["prompt"].removeprefix("A photo of ").rstrip(".")
                ax.set_title(textwrap.fill(short, 42), fontsize=7.5)
    fig.suptitle("Synthetic training scenes (TA-TiTok + MaskGen) — AI-generated, not real photos", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIG_DIR / "synthetic_samples.png", dpi=130)
    plt.close(fig)


def dataset_counts():
    rows = read_rows()
    syn = []
    if SYNTH_META.exists():
        with open(SYNTH_META) as f:
            syn = list(csv.DictReader(f))
    cnt = Counter((r["split"], r["label"]) for r in rows)
    scnt = Counter(r["label"] for r in syn)
    splits = ["train", "val", "test", "own_test"]
    lines = ["| class | " + " | ".join(splits) + " | synthetic (train) |", "|---" * (len(splits) + 2) + "|"]
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(str(cnt[(s, c)]) for s in splits) + f" | {scnt[c]} |")
    tot_real_train = sum(cnt[("train", c)] for c in CLASSES)
    tot_syn = sum(scnt[c] for c in CLASSES)
    lines.append(f"| **total** | " + " | ".join(str(sum(cnt[(s, c)] for c in CLASSES)) for s in splits) + f" | {tot_syn} |")
    share = tot_syn / max(1, tot_syn + tot_real_train)
    lines.append(f"\nSynthetic share of the training set (when 100% of generated images are used): **{share:.1%}**")
    (FIG_DIR / "dataset_counts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline_resnet18")
    ap.add_argument("--real", default="vit_real")
    ap.add_argument("--sota", default="vit_real_synf")
    ap.add_argument("--gallery-split", default="own_test", choices=["own_test", "test"],
                    help="images for explain_gallery.png (default: your own photos)")
    args = ap.parse_args()
    FIG_DIR.mkdir(exist_ok=True)

    runs = load_runs()
    if not runs:
        raise SystemExit("No finished runs in runs/. Train first.")
    dataset_counts()
    results_table(runs)
    seed_table(runs)
    main_runs = {k: v for k, v in runs.items() if not SEED_SUFFIX.search(k)}  # keep plots readable
    curves(main_runs)
    per_class(main_runs)
    for a, b in [(args.baseline, args.sota), (args.real, args.sota)]:
        if a in runs and b in runs:
            for split in ["test", "own_test"]:
                if split in runs[a] and split in runs[b]:
                    confusion_shift(runs, a, b, split)
    explain_gallery(args.baseline, args.sota, split=args.gallery_split)
    synthetic_grid()
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
