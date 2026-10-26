"""Quality check for synthetic images: does each one actually look like its intended class?

Every generated image is classified by a model trained on REAL images only (default: runs/vit_real), so the check
is not circular. An image is rejected when that model gives its intended class a low probability, i.e. MaskGen drew
something else (for example "a parking lot next to an office building" that came out as just an office building).

    python scripts/filter_synthetic.py                    # writes data/synthetic/synthetic_quality.csv + a report
    python scripts/filter_synthetic.py --min-prob 0.3     # stricter

Then train on the filtered set with:
    python train.py --arch vit_small --run vit_real_synf --synthetic-frac 1.0 --filter-synthetic
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, ROOT, RUNS_DIR, SYNTH_META  # noqa: E402
from src.data import build_transforms, load_image  # noqa: E402
from src.models import load_checkpoint  # noqa: E402

QUALITY_CSV = SYNTH_META.parent / "synthetic_quality.csv"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="vit_real", help="run whose best.pt scores the images (must be trained on real data only)")
    ap.add_argument("--min-prob", type=float, default=0.2, help="reject if P(intended class) is below this")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(RUNS_DIR / args.judge / "best.pt", device)
    if meta["classes"] != CLASSES:
        raise SystemExit(f"{args.judge} was trained on {meta['classes']}, not {CLASSES}. Retrain it first.")
    tf = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)

    with open(SYNTH_META) as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in CLASSES and (ROOT / r["path"]).exists()]
    out = []
    for b in range(0, len(rows), args.batch_size):
        batch = rows[b:b + args.batch_size]
        x = torch.stack([tf(load_image(ROOT / r["path"])) for r in batch]).to(device)
        probs = torch.softmax(model(x).float(), 1).cpu()
        for r, p in zip(batch, probs):
            k = CLASSES.index(r["label"])
            pred = CLASSES[int(p.argmax())]
            out.append({"path": r["path"], "label": r["label"], "pred": pred, "p_label": f"{p[k]:.3f}",
                        "quality": "reject" if p[k] < args.min_prob else "keep", "prompt": r["prompt"]})

    with open(QUALITY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"Judge: {args.judge} (real data only) · reject if P(intended class) < {args.min_prob}\n")
    print(f"{'class':<12}{'images':>8}{'rejected':>10}   most often drawn as")
    for c in CLASSES:
        rc = [o for o in out if o["label"] == c]
        rej = [o for o in rc if o["quality"] == "reject"]
        drawn = Counter(o["pred"] for o in rej).most_common(2)
        print(f"{c:<12}{len(rc):>8}{len(rej):>10}   " + ", ".join(f"{k} ({v})" for k, v in drawn))
    n_rej = sum(o["quality"] == "reject" for o in out)
    print(f"\nTotal: {n_rej} of {len(out)} rejected ({n_rej / len(out):.0%}). Saved to {QUALITY_CSV}")

    bad_prompts = Counter(o["prompt"].split(",")[0] for o in out if o["quality"] == "reject").most_common(5)
    print("\nPrompts that most often produced off-class images:")
    for pr, k in bad_prompts:
        print(f"  {k:3d} × {pr}")


if __name__ == "__main__":
    main()
