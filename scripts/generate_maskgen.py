"""SOTA step: generate synthetic scenes with TA-TiTok + MaskGen (ByteDance, 2025).

Gap filling: text prompts target conditions that are rare in the real data
(night, backlight, over-exposure, unusual viewpoints, weather) so the classifier
sees them during training.

Setup (once):
    git clone https://github.com/bytedance/1d-tokenizer external/1d-tokenizer
    pip install open_clip_torch omegaconf einops diffusers transformers huggingface_hub accelerate scipy

Run (needs an NVIDIA GPU; MaskGen-L VQ fits in ~6 GB VRAM):
    python scripts/generate_maskgen.py --per-class 100
    python scripts/generate_maskgen.py --weights-from runs/baseline_resnet18/metrics.json --total 600
        -> gives more images to the classes the baseline is worst at (targeted gap filling)

Outputs:
    data/synthetic/<class>/SYN_<class>_0000.png   clean 256x256 images used for training
    data/synthetic/synthetic_meta.csv             prompt, seed and model for every image
Every PNG also carries an embedded "Synthetic" text tag. The app and report figures add a
visible SYNTHETIC label; training copies stay clean so the label can't become a shortcut.
"""
import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, INDOOR, ROOT, SYNTH_DIR, SYNTH_META  # noqa: E402

SUBJECTS = {
    "bedroom": ["a bedroom with a double bed and nightstands", "a small messy student dorm bedroom",
                "a minimalist bedroom with white walls", "a child's bedroom with toys on the floor",
                "a hotel bedroom with two beds", "a cozy attic bedroom with a slanted ceiling"],
    "kitchen": ["a modern kitchen with white cabinets", "a small apartment kitchen with a gas stove",
                "a messy kitchen with dishes in the sink", "a kitchen with a wooden table and chairs",
                "a restaurant-style steel kitchen", "a rustic country kitchen"],
    "office": ["an open-plan office with desks and computers", "a small home office with a laptop",
               "a corporate office with glass walls", "a cluttered office desk with papers and monitors",
               "an empty office cubicle", "a manager's office with a large desk"],
    "park": ["a city park with trees and a walking path", "a park with a pond and benches",
             "a green lawn in a public park", "a park playground surrounded by trees",
             "an autumn park with fallen leaves", "a park with a fountain"],
    "parking_lot": ["a large outdoor parking lot with rows of parked cars", "a supermarket parking lot with shopping carts",
                    "an almost empty parking lot with painted white lines", "a parking lot next to an office building",
                    "a shopping mall parking lot with parked cars and light poles",
                    "a parking lot with numbered parking spaces and a few parked cars"],
    "street": ["a busy city street with cars and shops", "a narrow old street with buildings",
               "a residential street with parked cars", "a wide avenue with traffic lights",
               "a street with a crosswalk and pedestrians seen from far away", "a quiet street with apartment blocks"],
}
INDOOR_CONDITIONS = ["", "at night with warm artificial light", "in very dim lighting",
                     "with strong backlight from a window", "overexposed photo", "wide-angle photo",
                     "photographed from a low angle", "slightly blurry phone photo", "photographed from the doorway"]
OUTDOOR_CONDITIONS = ["", "at night with street lights", "on a rainy day", "in winter with snow",
                      "at sunset with strong backlight", "in fog", "overexposed photo", "wide-angle photo",
                      "slightly blurry phone photo"]


def prompts_for(cls, n, rng):
    conds = INDOOR_CONDITIONS if cls in INDOOR else OUTDOOR_CONDITIONS
    combos = list(itertools.product(SUBJECTS[cls], conds))
    rng.shuffle(combos)
    out = []
    for subj, cond in itertools.islice(itertools.cycle(combos), n):
        out.append(f"A photo of {subj}" + (f", {cond}." if cond else "."))
    return out


def counts_from_weights(path, total, min_per_class):
    """More images for classes with lower F1 in a previous run (targeted gap filling)."""
    per_class = json.loads(Path(path).read_text())["test"]["per_class_f1"]
    gaps = np.array([max(1e-3, 1.0 - per_class[c]) for c in CLASSES])
    raw = gaps / gaps.sum() * total
    return {c: max(min_per_class, int(round(r))) for c, r in zip(CLASSES, raw)}


def patch_open_clip_compat(maskgen_module):
    """open_clip >= 3 changed text_global_pool to return only the pooled vector.
    MaskGen was written for the old API that returned (pooled, tokens); restore that."""
    original = maskgen_module.text_global_pool

    def text_global_pool(x, text=None, pool_type="argmax", *args, **kwargs):
        out = original(x, text, pool_type, *args, **kwargs)
        if isinstance(out, tuple):  # old open_clip: already (pooled, tokens)
            return out
        tokens = {"first": x[:, 1:], "last": x[:, :-1]}.get(pool_type, x)
        return out, tokens

    maskgen_module.text_global_pool = text_global_pool


def load_models(variant, repo, device):
    sys.path.insert(0, str(repo))
    import open_clip
    import modeling.maskgen as maskgen_module
    from modeling.maskgen import MaskGen_KL, MaskGen_VQ
    from modeling.tatitok import TATiTok

    patch_open_clip_compat(maskgen_module)

    kind, size = variant.split("_")  # e.g. vq_l
    tok_id = "turkeyju/tokenizer_tatitok_bl128_vq" if kind == "vq" else "turkeyju/tokenizer_tatitok_bl32_vae"
    gen_cls = MaskGen_VQ if kind == "vq" else MaskGen_KL
    tokenizer = TATiTok.from_pretrained(tok_id).eval().requires_grad_(False).to(device)
    generator = gen_cls.from_pretrained(f"turkeyju/generator_maskgen_{kind}_{size}").eval().requires_grad_(False).to(device)

    # The OpenAI CLIP weights were trained with QuickGELU; open_clip >= 3 no longer applies it
    # automatically, which would silently change the text embeddings MaskGen was trained on.
    clip_encoder, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai", force_quick_gelu=True)
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
    clip_encoder.transformer.batch_first = False
    clip_encoder = clip_encoder.eval().requires_grad_(False).to(device)
    return kind, tokenizer, generator, clip_encoder, clip_tokenizer


@torch.no_grad()
def text_guidance(clip_encoder, clip_tokenizer, texts, device):
    # Same text conditioning for the de-tokenizer as in the official README_MaskGen example.
    t = clip_tokenizer(texts).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    x = clip_encoder.token_embedding(t).to(cast_dtype) + clip_encoder.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)
    x = clip_encoder.transformer(x, attn_mask=clip_encoder.attn_mask)
    x = x.permute(1, 0, 2)
    return clip_encoder.ln_final(x)


@torch.no_grad()
def generate_batch(kind, tokenizer, generator, clip_encoder, clip_tokenizer, texts, args, device):
    if kind == "vq":
        tokens = generator.generate(captions=texts, guidance_scale=args.guidance, randomize_temperature=2.0,
                                    sample_aesthetic_score=6.5, clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder)
    else:
        tokens = generator.sample_tokens(len(texts), clip_tokenizer, clip_encoder, num_iter=32,
                                         cfg=args.guidance, aes_scores=6.5, captions=texts)
    imgs = tokenizer.decode_tokens(tokens, text_guidance(clip_encoder, clip_tokenizer, texts, device))
    imgs = (torch.clamp(imgs, 0.0, 1.0) * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return [Image.fromarray(a) for a in imgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(ROOT / "external" / "1d-tokenizer"), help="path to the cloned bytedance/1d-tokenizer")
    ap.add_argument("--variant", default="vq_l", choices=["vq_l", "vq_xl", "kl_l", "kl_xl"])
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--weights-from", help="metrics.json of a run; allocate --total images by (1 - F1)")
    ap.add_argument("--total", type=int, default=600)
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=None, help="default 12 for VQ, 3 for KL (official values)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="keep existing synthetic images and add more")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("MaskGen generation needs a CUDA GPU.")
    repo = Path(args.repo)
    if not (repo / "modeling" / "maskgen.py").exists():
        raise SystemExit(f"Clone the official repo first:\n  git clone https://github.com/bytedance/1d-tokenizer {repo}")
    device = "cuda"
    if args.guidance is None:
        args.guidance = 12.0 if args.variant.startswith("vq") else 3.0

    if args.weights_from:
        counts = counts_from_weights(args.weights_from, args.total, args.min_per_class)
    else:
        counts = {c: args.per_class for c in args.classes}
    counts = {c: n for c, n in counts.items() if c in args.classes}
    print("Images to generate per class:", counts)

    kind, tokenizer, generator, clip_encoder, clip_tokenizer = load_models(args.variant, repo, device)

    rows = []
    if SYNTH_META.exists():  # keep rows of classes we are not regenerating (and that still exist)
        with open(SYNTH_META) as f:
            rows = [r for r in csv.DictReader(f) if r["label"] in CLASSES]
    model_name = f"TA-TiTok + MaskGen-{args.variant.upper().replace('_', '-')}"

    for cls, n in counts.items():
        rng = random.Random(f"{args.seed}-{cls}")
        out_dir = SYNTH_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        start = len(list(out_dir.glob("SYN_*.png"))) if args.append else 0
        if not args.append:
            for old in out_dir.glob("SYN_*.png"):
                old.unlink()
            rows = [r for r in rows if r["label"] != cls]
        prompts = prompts_for(cls, n, rng)
        for b in range(0, n, args.batch_size):
            batch = prompts[b:b + args.batch_size]
            seed = args.seed * 100000 + CLASSES.index(cls) * 10000 + start + b
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            imgs = generate_batch(kind, tokenizer, generator, clip_encoder, clip_tokenizer, batch, args, device)
            for k, (img, prompt) in enumerate(zip(imgs, batch)):
                idx = start + b + k
                path = out_dir / f"SYN_{cls}_{idx:04d}.png"
                meta = PngInfo()
                meta.add_text("Synthetic", "true")
                meta.add_text("Generator", model_name)
                meta.add_text("Prompt", prompt)
                meta.add_text("Disclosure", "AI-generated image, not a real photograph")
                img.save(path, pnginfo=meta)
                rows.append({"path": path.relative_to(ROOT).as_posix(), "label": cls, "prompt": prompt,
                             "model": model_name, "seed": seed, "guidance": args.guidance})
            print(f"  {cls}: {min(b + args.batch_size, n)}/{n}", end="\r")
        print(f"  {cls}: {n} images done        ")

    SYNTH_META.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTH_META, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "prompt", "model", "seed", "guidance"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {SYNTH_META}")


if __name__ == "__main__":
    main()
