"""Gradio demo: upload / webcam -> scene prediction -> visual explanation.

    python app.py                       # http://127.0.0.1:7860  (open on your phone via --share)
    python app.py --share               # temporary public link for the defence / phone camera
    python app.py --sota vit_real_syn --baseline baseline_resnet18
"""
import argparse
import csv
import json
import os
import platform
import re
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASSES, FIG_DIR, INDOOR, MANIFEST, ROOT, RUNS_DIR, SYNTH_META  # noqa: E402
from src.data import build_transforms, load_image  # noqa: E402
from src.explain import display_crop, make_explainers, overlay  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.ui_menu import MENU_HEAD  # noqa: E402
from src.watermark import stamp_synthetic  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = 1280  # uploaded photos are shrunk to this size before the models see them
LAT = defaultdict(lambda: deque(maxlen=100))  # rolling inference latency per model


def load_models(names):
    models = {}
    for n in names:
        if n in models:
            continue
        ck, metrics_p = RUNS_DIR / n / "best.pt", RUNS_DIR / n / "metrics.json"
        if not ck.exists():  # fresh clone: use the bundled half-precision copy
            ck, metrics_p = ROOT / "weights" / f"{n}.pt", ROOT / "weights" / f"{n}.json"
        if not ck.exists():
            continue
        model, meta = load_checkpoint(ck, DEVICE)
        explainers = make_explainers(model, meta)
        tf = build_transforms(meta["mean"], meta["std"], meta["img_size"], train=False)
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
    """Name the scene in a photo (six classes) and show a heatmap of what the model looked at."""
    if img is None:
        raise gr.Error("Upload a photo or take one with the camera first.")
    img = load_image(img, MAX_SIDE)  # `img` is the path of the uploaded file
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
    meta_csv = SYNTH_META if SYNTH_META.exists() else ROOT / "assets" / "synthetic" / "synthetic_meta.csv"
    if not meta_csv.exists():
        return []
    with open(meta_csv) as f:
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


def nice(cls):
    return cls.replace("_", " ")


def synth_gallery(cls):
    """Every pre-generated image of `cls`, with the visible SYNTHETIC banner."""
    pool = [r for r in SYNTH if r["label"] == cls]
    good = [r for r in pool if r.get("quality") != "reject"]  # hide images the quality check flagged
    return [(stamp_synthetic(load_image(ROOT / r["path"])), synth_caption(r)) for r in (good or pool)]


def all_synth_gallery():
    """All pre-generated images, grouped by class."""
    return [(img, f"{nice(c)}: {cap}") for c in CLASSES for img, cap in synth_gallery(c)]


def complementary(cls):
    """Show the pre-generated synthetic scenes of a class and of the class it is most often confused with."""
    if not SYNTH or not cls:
        return "", [], "", []
    confused = most_confused_with(cls)
    head1 = f"### Selected class: **{nice(cls)}**"
    if confused:
        head2 = (f"### Most often confused with: **{nice(confused)}**\n"
                 f"On the real test set the model most often mistakes *{nice(cls)}* for *{nice(confused)}*, "
                 f"so these are the scenes to compare against.")
    else:
        head2 = f"### Most often confused with: none\nThe model made no *{nice(cls)}* errors on the real test set."
    return head1, synth_gallery(cls), head2, (synth_gallery(confused) if confused else [])


# ---------- retrieval: nearest real training photos ----------
RETR_DIR = ROOT / "assets" / "retrieval"


def load_retrieval():
    """Per-model index built by scripts/build_retrieval_index.py (embeddings of the real training photos)."""
    index = {}
    for n in MODELS:
        p = RETR_DIR / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        mp = RETR_DIR / f"{n}_metrics.json"
        index[n] = dict(emb=z["emb"].astype(np.float32), labels=z["labels"], thumbs=z["thumbs"],
                        paths=z["paths"] if "paths" in z.files else z["thumbs"],
                        metrics=json.loads(mp.read_text()) if mp.exists() else {})
    return index


def neighbour_image(ix, i):
    """Thumbnail, else the original training photo, else a grey tile: the photos themselves are not in the repository."""
    for p in (RETR_DIR / "thumbs" / ix["thumbs"][i], ROOT / str(ix["paths"][i])):
        if p.is_file():
            return Image.open(p)
    tile = Image.new("RGB", (192, 192), (70, 70, 70))
    ImageDraw.Draw(tile).text((10, 90), f"{nice(ix['labels'][i])}\n(photo not included)", fill="white")
    return tile


@torch.no_grad()
def retrieve(img, model_name, k):
    """Find the real training photos that look most like the given photo."""
    if img is None or model_name not in RETR:
        return [], ""
    m, ix = MODELS[model_name], RETR[model_name]
    x = m["tf"](load_image(img, MAX_SIDE)).unsqueeze(0).to(DEVICE)
    t0 = time.perf_counter()
    fmap = m["model"].forward_features(x)
    query = F.normalize(m["model"].forward_head(fmap, pre_logits=True).float(), dim=1)[0].cpu().numpy()
    sims = ix["emb"] @ query  # unit vectors: dot product = cosine similarity
    top = np.argsort(-sims)[:int(k)]
    ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(m["model"].forward_head(fmap).float(), 1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    gallery = [(neighbour_image(ix, i), f"{nice(ix['labels'][i])} · similarity {sims[i]:.2f}") for i in top]
    vote, n_vote = Counter(ix["labels"][top]).most_common(1)[0]
    info = (f"**Nearest-neighbour vote:** {nice(vote)} ({n_vote} of {len(top)} neighbours)  ·  "
            f"**Classifier:** {nice(CLASSES[pred])} ({probs[pred]:.0%})  ·  "
            f"embedding + search: {ms:.1f} ms over {len(ix['labels'])} real training photos\n\n"
            f"The neighbours are real Places365 training photos, found by comparing the model's internal features "
            f"(cosine similarity). Mixed classes among them mean the photo sits between those classes.")
    return gallery, info


def retrieval_table_md():
    rows = [(n, ix["metrics"]) for n, ix in RETR.items() if ix["metrics"]]
    if not rows:
        return ""
    ks = [c for c in next(iter(rows))[1].get("test", {}) if c.startswith("P@")]
    lines = ["| model | queries | " + " | ".join(ks) + " |", "|---|---|" + "---|" * len(ks)]
    for n, met in rows:
        for split, label in (("test", "Places365 test"), ("own_test", "in-the-wild")):
            if split in met:
                lines.append(f"| `{n}` | {label} ({met[split]['n']}) | " + " | ".join(f"{met[split][c]:.3f}" for c in ks) + " |")
    chance = rows[0][1].get("random_baseline")
    return "\n".join(lines) + (f"\n\nRandom retrieval would score about {chance:.2f} (one class in six)." if chance else "")


def results_md():
    p = FIG_DIR / "results.md"
    return p.read_text() if p.exists() else "Run `python make_figures.py` after training to fill this tab."


ETHICS = """
### Ethics, consent and disclosure
- **Synthetic media is labelled.** Every generated image shown here carries a visible *SYNTHETIC* banner, and every
  PNG file stores a `Synthetic: true` tag with the generator name and prompt. Test sets use **real photos only**.
- **People.** Prompts avoid identifiable people; scenes are empty or show people only from far away. Own photos used
  for testing were taken in public/own spaces; photos with recognisable people need their consent or must be blurred.
- **Privacy.** Uploaded and webcam images are processed in memory for this request and are not stored by the app. When the app runs on a hosted server (for example Hugging Face Spaces), your photo is sent to that server and the host's own terms apply; do not upload photos of people who have not agreed to it. The Run history list is kept only in your own browser.
- **Bias & limits.** Places365 is mostly Western / North-American imagery, so local kitchens, streets and parking lots
  can be misclassified. Synthetic images inherit the generator's own biases (CLIP / open web data). The model only knows
  6 classes and will force any photo into one of them.
- **Misuse.** Scene recognition can reveal where a photo was taken; don't use it to locate or profile people.

### Credits
TA-TiTok + MaskGen — Kim et al., 2025 ([paper](https://arxiv.org/abs/2501.07730), [code](https://github.com/bytedance/1d-tokenizer)),
summarised by [Neurohive](https://neurohive.io/en/state-of-the-art/bytedance-unveil-ta-titok-tokenizer-achieving-sota-in-text-to-image-generation-using-only-public-data/). ViT-S/16 (AugReg) via `timm`. Places365 — Zhou et al., 2017. Attention rollout — Abnar & Zuidema, 2020. Grad-CAM — Selvaraju et al., 2017.
"""


def build_ui():
    names = list(MODELS)
    default = ARGS.sota if ARGS.sota in MODELS else names[0]
    with gr.Blocks(title="Scene Classifier") as demo:
        chips = "".join(f"<span>{nice(c)}</span>" for c in CLASSES)
        gr.HTML('<h1>Scene classification</h1>'
                '<p class="app-sub">Upload a photo or take one with the camera: the model names the scene and shows '
                'what it looked at.</p>'
                f'<p class="app-classes">{chips}</p>', elem_id="app-header")
        examples = [str(ROOT / "assets" / "examples" / f"{c}.jpg") for c in CLASSES
                    if (ROOT / "assets" / "examples" / f"{c}.jpg").exists()]
        if not examples:
            ex_dir = ROOT / "data" / "real" / "test"
            examples = [str(next((ex_dir / c).glob("*"))) for c in CLASSES
                        if (ex_dir / c).exists() and any((ex_dir / c).iterdir())]
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="filepath", sources=["upload", "webcam", "clipboard"], label="Scene photo")
                    model_dd = gr.Dropdown(names, value=default, label="Model")
                    method_rb = gr.Radio(["Grad-CAM", "Attention rollout"], value="Grad-CAM", label="Explanation method",
                                         info="Attention rollout is ViT-only; the CNN baseline always uses Grad-CAM")
                    compare = gr.Checkbox(value=len(names) > 1, label="Also show the other model's explanation")
                    btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    out_label = gr.Label(num_top_classes=3, label="Prediction")
                    out_vis = gr.Image(label="Explanation (selected model)")
                    out_vis2 = gr.Image(label="Explanation (other model)")
                    out_info = gr.Markdown()
            if examples:  # below the result, not between the controls and the result
                gr.Examples(examples, inputs=inp, label="Test images")
        with gr.Tab("Similar scenes"):
            if not RETR:
                gr.Markdown("No retrieval index found. Run `python scripts/prepare_data.py` and "
                            "`python scripts/build_retrieval_index.py` first.")
            else:
                gr.Markdown(
                    "**Retrieval** answers *\"which training photos look most like mine?\"* Each photo is turned into a "
                    "feature vector by the model, and the closest vectors among the real training photos are shown. "
                    "It is another way to explain a prediction: *the model says kitchen because these are the kitchens "
                    "it has seen*. Quality is measured with **P@k**: the share of the k nearest training photos that "
                    "have the same class as the query, averaged over all real test photos.\n\n"
                    "A photo you add on the **Classify** tab appears here automatically and the search runs at once.")
                with gr.Row():
                    with gr.Column(scale=1):
                        r_inp = gr.Image(type="filepath", sources=["upload", "webcam", "clipboard"], label="Scene photo")
                        r_model = gr.Dropdown(list(RETR), value=ARGS.sota if ARGS.sota in RETR else next(iter(RETR)),
                                              label="Model (its features define 'similar')")
                        r_k = gr.Slider(3, 10, value=5, step=1, label="Number of neighbours (k)")
                    with gr.Column(scale=2):
                        r_gal = gr.Gallery(columns=3, height="auto", label="Most similar real training photos")
                        r_info = gr.Markdown()
                if examples:
                    gr.Examples(examples, inputs=r_inp, label="Test images")
                gr.on([r_inp.change, r_model.change, r_k.release], retrieve, [r_inp, r_model, r_k], [r_gal, r_info])
                table = retrieval_table_md()
                if table:
                    n_index = len(next(iter(RETR.values()))["labels"])
                    gr.Markdown(f"### Retrieval quality (P@k)\nEach real test photo is a query against the {n_index} "
                                f"real training photos.\n\n" + table)
        with gr.Tab("Synthetic scenes"):
            gr.Markdown(
                f"The {len(SYNTH)} scenes on this tab were **generated in advance** with TA-TiTok + MaskGen-L on an "
                "NVIDIA GeForce RTX 5070 Laptop GPU and then quality-checked. **The app does not generate images**; it only "
                "displays this ready-made set, so no GPU is needed here. **Every image is AI-generated, not a real photo**, "
                "and carries a visible SYNTHETIC banner. They are training data, not predictions.\n\n"
                "Pick a class: its scenes and the scenes of the class the model most often confuses it with appear "
                "immediately. After a prediction on the Classify tab, the predicted class is selected here automatically.")
            cls_dd = gr.Dropdown(CLASSES, value=CLASSES[0], label="Class")
            head1 = gr.Markdown()
            gal1 = gr.Gallery(columns=4, height="auto", show_label=False)
            head2 = gr.Markdown()
            gal2 = gr.Gallery(columns=4, height="auto", show_label=False)
            cls_dd.change(complementary, cls_dd, [head1, gal1, head2, gal2], api_name="synthetic_scenes")
            demo.load(complementary, cls_dd, [head1, gal1, head2, gal2], api_visibility="private")
            with gr.Accordion(f"All {len(SYNTH)} pre-generated scenes", open=True):
                gal_all = gr.Gallery(columns=6, height="auto", show_label=False, elem_id="all-synth")
            demo.load(all_synth_gallery, None, gal_all, api_visibility="private")
        btn.click(classify, [inp, model_dd, method_rb, compare], [out_label, out_vis, out_vis2, out_info, cls_dd],
                  api_name="classify")
        if RETR:  # a photo added on the Classify tab is copied to the Retrieval tab, which then searches for it
            inp.change(lambda path: path, inp, r_inp, queue=False, show_progress="hidden", api_visibility="private")
        with gr.Tab("Results"):
            gr.Markdown(results_md())
            shift_figs = sorted(p.name for p in FIG_DIR.glob(f"confusion_shift_*_vs_{ARGS.sota}*.png")) if FIG_DIR.exists() else []
            for f in shift_figs + ["val_f1_curves.png", "per_class_f1.png", "explain_gallery.png"]:
                if (FIG_DIR / f).exists():
                    gr.Image(str(FIG_DIR / f), label=f, show_label=True)
        with gr.Tab("About"):
            gr.Markdown(ETHICS)
    return demo


def find_installed_ngrok():
    """The real ngrok program on PATH. The virtual environment is skipped: pyngrok puts its own `ngrok` launcher
    into .venv/Scripts, which only tries to download the program again."""
    venv_scripts = Path(sys.executable).resolve().parent
    folders = [f for f in os.environ.get("PATH", "").split(os.pathsep) if f and Path(f).resolve() != venv_scripts]
    return shutil.which("ngrok", path=os.pathsep.join(folders))


def start_ngrok(domain, port):
    """Public HTTPS tunnel with a fixed address (a free ngrok static domain). The token comes from NGROK_AUTHTOKEN."""
    try:
        from pyngrok import ngrok
        from pyngrok.conf import PyngrokConfig
        from pyngrok.exception import PyngrokNgrokInstallError
    except ImportError:
        raise SystemExit("The ngrok tunnel needs one more package: pip install pyngrok  (see README, 'Phone access').")
    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        raise SystemExit("Set NGROK_AUTHTOKEN (your ngrok authtoken) first; see README, 'Phone access'.")
    domain = domain.removeprefix("https://").removeprefix("http://").rstrip("/")
    # The ngrok program and its config live inside the project (git-ignored): some Windows setups do not let
    # Python create folders in the user profile (AppData\Local\ngrok), where pyngrok would otherwise put them.
    home = ROOT / ".ngrok"
    home.mkdir(exist_ok=True)
    own_copy = home / ("ngrok.exe" if os.name == "nt" else "ngrok")
    # Prefer a copy placed in .ngrok, then an installed ngrok (e.g. from the Microsoft Store); download only as a last resort.
    binary = str(own_copy) if own_copy.exists() else (find_installed_ngrok() or str(own_copy))
    config = PyngrokConfig(auth_token=token, ngrok_path=binary, config_path=str(home / "ngrok.yml"))
    # pyngrok 9 talks to ngrok's endpoint API, where a fixed address is given as `url` (not `domain`)
    try:
        tunnel = ngrok.connect(port, "http", url=f"https://{domain}", pyngrok_config=config)
    except PyngrokNgrokInstallError as e:
        raise SystemExit(f"{e}\n\nDownload the ngrok program by hand instead (https://ngrok.com/download, Windows), unzip "
                         f"it and put ngrok.exe into {home}, then start the demo again.")
    if domain not in (tunnel.public_url or ""):
        raise SystemExit(f"ngrok opened {tunnel.public_url} instead of https://{domain}: check NGROK_DOMAIN in .ngrok.env.")
    print(f"\nPublic link (stays the same on every start): {tunnel.public_url}\n", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sota", default="vit_real_synf")
    ap.add_argument("--baseline", default="baseline_resnet18")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--ngrok-domain", default=os.environ.get("NGROK_DOMAIN"),
                    help="fixed public address from your free ngrok account, e.g. name.ngrok-free.dev "
                         "(needs NGROK_AUTHTOKEN and pip install pyngrok)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--all-models", action="store_true", help="also load every other run in runs/")
    ARGS = ap.parse_args()
    names = [ARGS.sota, ARGS.baseline, "vit_real"]  # final, baseline, real-only ViT: a readable dropdown for the demo
    if ARGS.all_models:
        names += [p.parent.name for p in sorted(RUNS_DIR.glob("*/best.pt"))]
    MODELS = load_models(names)
    if not MODELS:
        raise SystemExit("No models found in runs/ or weights/. Train with train.py, or run scripts/export_release.py.")
    SYNTH = synth_rows()
    if MANIFEST.exists():  # data/ is present: create the retrieval index and thumbnails that are still missing
        from scripts.build_retrieval_index import build
        build([n for n in (ARGS.sota, ARGS.baseline) if n in MODELS], only_missing=True)
    RETR = load_retrieval()
    print(f"Loaded models: {list(MODELS)} on {DEVICE}; synthetic images: {len(SYNTH)}; retrieval indexes: {list(RETR)}")
    if ARGS.ngrok_domain:
        start_ngrok(ARGS.ngrok_domain, ARGS.port)
    print(f"Open on this computer: http://127.0.0.1:{ARGS.port}", flush=True)  # 0.0.0.0 below is not a browser address
    build_ui().launch(server_name="0.0.0.0", server_port=ARGS.port, share=ARGS.share, head=MENU_HEAD,
                         favicon_path=str(ROOT / "assets" / "icon.png"), pwa=True)  # pwa: installable on a phone
