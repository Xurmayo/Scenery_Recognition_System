"""Gradio demo: upload / webcam -> scene prediction -> visual explanation.

    python app.py                       # http://127.0.0.1:7860  (open on your phone via --share)
    python app.py --share               # temporary public link for the defence / phone camera
    python app.py --sota vit_real_syn --baseline baseline_resnet18
"""
import argparse
import csv
import json
import platform
import random
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, FIG_DIR, INDOOR, ROOT, RUNS_DIR, SYNTH_META  # noqa: E402
from src.data import build_transforms, load_image  # noqa: E402
from src.explain import display_crop, make_explainers, overlay  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.watermark import stamp_synthetic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAT = defaultdict(lambda: deque(maxlen=100))  # rolling inference latency per model


def load_models(names):
    models = {}
    for n in names:
        ck = RUNS_DIR / n / "best.pt"
        if not ck.exists():
            continue
        model, meta = load_checkpoint(ck, DEVICE)
        explainers = make_explainers(model, meta)
        tf = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
        metrics_p = RUNS_DIR / n / "metrics.json"
        metrics = json.loads(metrics_p.read_text()) if metrics_p.exists() else {}
        models[n] = dict(model=model, meta=meta, explainers=explainers, tf=tf, metrics=metrics)
    return models


def browser_from(request):
    ua = (request.headers.get("user-agent", "") if request else "") or ""
    for name, pat in [("Edge", r"Edg/([\d.]+)"), ("Opera", r"OPR/([\d.]+)"), ("Chrome", r"Chrome/([\d.]+)"),
                      ("Firefox", r"Firefox/([\d.]+)"), ("Safari", r"Version/([\d.]+).*Safari")]:
        m = re.search(pat, ua)
        if m:
            os_ = "Android" if "Android" in ua else "iOS" if ("iPhone" in ua or "iPad" in ua) else \
                  "Windows" if "Windows" in ua else "macOS" if "Mac OS" in ua else "Linux" if "Linux" in ua else "?"
            return f"{name} {m.group(1).split('.')[0]} on {os_}"
    return "unknown browser"


def device_name():
    if DEVICE == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return f"CPU: {platform.processor() or platform.machine()}"


@torch.no_grad()
def timed_forward(m, x):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = m["model"](x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return logits, (time.perf_counter() - t0) * 1000


def run_one(name, img, method=None):
    m = MODELS[name]
    if method not in m["explainers"]:
        method = next(iter(m["explainers"]))
    x = m["tf"](img).unsqueeze(0).to(DEVICE)
    logits, ms = timed_forward(m, x)
    LAT[name].append(ms)
    t0 = time.perf_counter()
    _, heat = m["explainers"][method](x)
    exp_ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits.float(), 1)[0].cpu().numpy()
    vis = overlay(display_crop(img, m["meta"]["img_size"]), heat)
    return probs, vis, ms, exp_ms, method


def classify(img, model_name, method, compare, request: gr.Request):
    if img is None:
        raise gr.Error("Upload a photo or take one with the camera first.")
    img = img.convert("RGB")
    probs, vis, ms, exp_ms, used = run_one(model_name, img, method)
    top = int(np.argmax(probs))
    label = {c: float(p) for c, p in zip(CLASSES, probs)}
    io = "indoor" if CLASSES[top] in INDOOR else "outdoor"

    other_vis, other_txt = None, ""
    others = [n for n in MODELS if n != model_name]
    if compare and others:
        o = others[0]
        p2, other_vis, ms2, _, used2 = run_one(o, img, method)
        k2 = int(np.argmax(p2))
        other_txt = f"\n\n**Comparison — `{o}`** ({used2}): **{CLASSES[k2]}** ({p2[k2]:.0%}), {ms2:.1f} ms"

    avg = np.mean(LAT[model_name])
    info = (f"### {CLASSES[top]}  ·  {io}\n"
            f"**Model:** `{model_name}` ({MODELS[model_name]['meta']['timm_name']})  ·  explanation: {used}\n\n"
            f"**Latency:** {ms:.1f} ms this image  ·  **avg {avg:.1f} ms** over {len(LAT[model_name])} runs  "
            f"(+{exp_ms:.0f} ms to compute the heatmap)\n\n"
            f"**Environment:** {device_name()} · PyTorch {torch.__version__} · {platform.system()}  ·  "
            f"client: {browser_from(request)}" + other_txt)
    return label, vis, other_vis, info, CLASSES[top]


# ---------- synthetic gallery ----------
def synth_rows():
    if not SYNTH_META.exists():
        return []
    with open(SYNTH_META) as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in CLASSES and (ROOT / r["path"]).exists()]
    q = SYNTH_META.parent / "synthetic_quality.csv"
    if q.exists():
        with open(q) as f:
            verdict = {r["path"]: r["quality"] for r in csv.DictReader(f)}
        for r in rows:
            r["quality"] = verdict.get(r["path"], "")
    return rows


def most_confused_with(cls):
    """Class the SOTA model most often mistakes `cls` for (from its test confusion matrix)."""
    for n in [ARGS.sota, ARGS.baseline]:
        m = MODELS.get(n, {}).get("metrics", {})
        if "test" in m:
            cm = np.array(m["test"]["confusion_matrix"])
            i = CLASSES.index(cls)
            row = cm[i].copy()
            row[i] = -1
            return CLASSES[int(row.argmax())] if row.max() > 0 else None
    return None


def synth_caption(r):
    return r["prompt"].removeprefix("A photo of ").rstrip(".")


def synth_gallery(cls, n):
    pool = [r for r in SYNTH if r["label"] == cls]
    good = [r for r in pool if r.get("quality") != "reject"]  # hide images the quality check flagged
    pool = good or pool
    random.shuffle(pool)
    return [(stamp_synthetic(load_image(ROOT / r["path"])), synth_caption(r)) for r in pool[:n]]


def complementary(cls, n=4):
    if not SYNTH:
        msg = "No synthetic images yet. Run `python scripts/generate_maskgen.py` first."
        return [], [], msg, "", ""
    if not cls:
        cls = random.choice(CLASSES)
    confused = most_confused_with(cls)
    nice = lambda c: c.replace("_", " ")
    note = ("All images below were generated offline with TA-TiTok + MaskGen and are **AI-generated, not real photos**. "
            "They are training data, not predictions.")
    head1 = f"### Synthetic **{nice(cls)}** scenes"
    head2 = (f"### Synthetic **{nice(confused)}** scenes, the class the model most often confuses with {nice(cls)}"
             if confused else "")
    return synth_gallery(cls, n), (synth_gallery(confused, n) if confused else []), note, head1, head2


def results_md():
    p = FIG_DIR / "results.md"
    return p.read_text() if p.exists() else "Run `python make_figures.py` after training to fill this tab."


ETHICS = """
### Ethics, consent and disclosure
- **Synthetic media is labelled.** Every generated image shown here carries a visible *SYNTHETIC* banner, and every
  PNG file stores a `Synthetic: true` tag with the generator name and prompt. Test sets use **real photos only**.
- **People.** Prompts avoid identifiable people; scenes are empty or show people only from far away. Own photos used
  for testing were taken in public/own spaces; photos with recognisable people need their consent or must be blurred.
- **Privacy.** Uploaded and webcam images are processed in memory for this request and are not stored.
- **Bias & limits.** Places365 is mostly Western / North-American imagery, so local kitchens, streets and parking lots
  can be misclassified. Synthetic images inherit the generator's own biases (CLIP / open web data). The model only knows
  6 classes and will force any photo into one of them.
- **Misuse.** Scene recognition can reveal where a photo was taken; don't use it to locate or profile people.

### Credits
TA-TiTok + MaskGen — Kim et al., 2025 ([paper](https://arxiv.org/abs/2501.07730), [code](https://github.com/bytedance/1d-tokenizer)),
summarised by Neurohive. ViT-S/16 (AugReg) via `timm`. Places365 — Zhou et al., 2017. Attention rollout — Abnar & Zuidema, 2020. Grad-CAM — Selvaraju et al., 2017.
"""


def build_ui():
    names = list(MODELS)
    default = ARGS.sota if ARGS.sota in MODELS else names[0]
    with gr.Blocks(title="Scene Classifier") as demo:
        gr.Markdown(f"# Scene classification · {', '.join(CLASSES)}\n"
                    f"Upload a photo or use the camera → prediction → heatmap of what the model looked at.")
        last_cls = gr.State(None)
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", sources=["upload", "webcam", "clipboard"], label="Scene photo")
                    model_dd = gr.Dropdown(names, value=default, label="Model")
                    method_rb = gr.Radio(["Grad-CAM", "Attention rollout"], value="Grad-CAM", label="Explanation method",
                                         info="Attention rollout is ViT-only; the CNN baseline always uses Grad-CAM")
                    compare = gr.Checkbox(value=len(names) > 1, label="Also show the other model's explanation")
                    btn = gr.Button("Classify", variant="primary")
                    ex_dir = ROOT / "data" / "real" / "test"
                    examples = [str(next((ex_dir / c).glob("*"))) for c in CLASSES if (ex_dir / c).exists() and any((ex_dir / c).iterdir())]
                    if examples:
                        gr.Examples(examples, inputs=inp, label="Test images")
                with gr.Column():
                    out_label = gr.Label(num_top_classes=3, label="Prediction")
                    out_vis = gr.Image(label="Explanation (selected model)")
                    out_vis2 = gr.Image(label="Explanation (other model)")
                    out_info = gr.Markdown()
            btn.click(classify, [inp, model_dd, method_rb, compare], [out_label, out_vis, out_vis2, out_info, last_cls])
        with gr.Tab("Complementary scenes (synthetic)"):
            gr.Markdown("Shows offline-generated MaskGen training scenes for a class and for the class the model confuses "
                        "it with most, so you can compare them. **Every image here is AI-generated.**")
            with gr.Row():
                cls_dd = gr.Dropdown([None] + CLASSES, value=None, label="Class (empty = last prediction)")
                gen_btn = gr.Button("Generate complementary scenes", variant="primary")
            gal_note = gr.Markdown()
            head1 = gr.Markdown()
            gal1 = gr.Gallery(columns=4, height="auto", show_label=False)
            head2 = gr.Markdown()
            gal2 = gr.Gallery(columns=4, height="auto", show_label=False)
            gen_btn.click(lambda c, last: complementary(c or last), [cls_dd, last_cls], [gal1, gal2, gal_note, head1, head2])
        with gr.Tab("Results"):
            gr.Markdown(results_md())
            shift_figs = sorted(p.name for p in FIG_DIR.glob(f"confusion_shift_*_vs_{ARGS.sota}*.png")) if FIG_DIR.exists() else []
            for f in shift_figs + ["val_f1_curves.png", "per_class_f1.png", "explain_gallery.png"]:
                if (FIG_DIR / f).exists():
                    gr.Image(str(FIG_DIR / f), label=f, show_label=True)
        with gr.Tab("About & ethics"):
            gr.Markdown(ETHICS)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sota", default="vit_real_synf")
    ap.add_argument("--baseline", default="baseline_resnet18")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ARGS = ap.parse_args()
    MODELS = load_models([ARGS.sota, ARGS.baseline] + [p.parent.name for p in sorted(RUNS_DIR.glob("*/best.pt"))])
    if not MODELS:
        raise SystemExit("No trained models in runs/. Run train.py first.")
    SYNTH = synth_rows()
    print(f"Loaded models: {list(MODELS)} on {DEVICE}; synthetic images: {len(SYNTH)}")
    build_ui().launch(server_name="0.0.0.0", server_port=ARGS.port, share=ARGS.share)
