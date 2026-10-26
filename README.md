# Scenery Recognition System

Scene classification with a Vision Transformer and **TA-TiTok + MaskGen** synthetic data, plus a Gradio web app that
shows *why* the model made each prediction.

Six scene classes: **bedroom, kitchen, office** (indoor) and **park, parking lot, street** (outdoor).

> Computer Vision — Assignment 2 (Advanced track: scene classification).
> SOTA component: TA-TiTok + MaskGen (ByteDance & POSTECH, 2025) for synthetic gap-filling.

![Explanation gallery](figures/explain_gallery.png)
*Grad-CAM of the CNN baseline vs Grad-CAM and attention rollout of the ViT on Places365 test images.*

---

## Highlights

- **ViT-S/16 beats the ResNet-18 baseline by about 10 points** on both the Places365 test set and the in-the-wild set.
- **Quality-filtered synthetic data helps on both test sets:** in-the-wild macro-F1 0.925 → 0.962 and Places365 accuracy
  0.907 → 0.918 (means of 3 seeds) over the real-only ViT.
- **Synthetic-data quality filter:** a real-only classifier rejects generated images that don't look like their class
  (40 of 601), exposing ambiguous prompts such as "corporate office with glass walls" (16 of 21 drawn as street).
- **Targeted synthetic data:** MaskGen generates 600 scenes in hard conditions (night, snow, rain, backlight, odd angles),
  and classes the baseline was worst at get more images.
- **Honest evaluation:** test sets contain only real photos, the key runs are repeated over 3 seeds, and results are
  reported as mean ± std.
- **Explainability:** Grad-CAM for both models and attention rollout for the ViT.
- **Web app:** upload or webcam → top-3 prediction → heatmap, with live latency, device and browser, plus a gallery of
  clearly labelled synthetic scenes.

## Pipeline

```
Places365 (6 classes) ──► ResNet-18 baseline ──► per-class F1 ──► MaskGen prompt budget
        │                                                             │
        │                     TA-TiTok + MaskGen-L (text → image) ◄────┘
        │                                  │  601 synthetic scenes (night, snow, rain, ...)
        │                                  ▼
        │                     quality filter (real-only ViT) ──► 561 kept, 40 rejected
        ▼                                  ▼
   real train set ─────────────► ViT-S/16 fine-tuning ──► evaluation on REAL images only
                                                            ├─ Places365 test (150)
                                                            └─ in-the-wild set (79)
```

## Results

### Dataset

| Class | Train | Val | Test | In-the-wild | Synthetic (train) |
|---|---|---|---|---|---|
| bedroom | 60 | 15 | 25 | 20 | 52 |
| kitchen | 60 | 15 | 25 | 11 | 90 |
| office | 60 | 15 | 25 | 10 | 133 |
| park | 60 | 15 | 25 | 16 | 82 |
| parking lot | 60 | 15 | 25 | 9 | 114 |
| street | 60 | 15 | 25 | 13 | 130 |
| **Total** | **360** | **90** | **150** | **79** | **601** |

The synthetic share of the training set is 62.5% when all generated images are used. The in-the-wild set mixes own
photos and web images (test-only, never trained on), with many in hard conditions.

### Main comparison (single run, seed 42)

| Run | Model | Synthetic | Aug | Test acc | Test macro-F1 | In-the-wild acc | In-the-wild F1 | Latency |
|---|---|---|---|---|---|---|---|---|
| `baseline_resnet18` | ResNet-18 | — | ✓ | 0.813 | 0.811 | 0.797 | 0.792 | 4.1 ms |
| `vit_real` | ViT-S/16 | — | ✓ | **0.913** | **0.913** | 0.924 | 0.905 | 3.9 ms |
| `vit_real_syn` | ViT-S/16 | 100% | ✓ | 0.893 | 0.893 | **0.975** | 0.969 | 4.0 ms |
| `vit_real_synf` (final) | ViT-S/16 | 100%, filtered | ✓ | 0.907 | 0.907 | **0.975** | **0.970** | 5.1 ms |
| `vit_real_syn50` | ViT-S/16 | 50% | ✓ | 0.887 | 0.887 | 0.949 | 0.936 | 4.1 ms |
| `vit_real_syn_noaug` | ViT-S/16 | 100% | ✗ | 0.867 | 0.865 | 0.949 | 0.936 | 4.1 ms |

Latency is batch-1 inference on an NVIDIA GeForce RTX 5070 Laptop GPU. The final model gets 77 of 79 in-the-wild
images right (real-only ViT: 73, baseline: 63).

### Synthetic data, unfiltered vs filtered (mean ± std over 3 seeds)

| Config | Test acc | Test macro-F1 | In-the-wild acc | In-the-wild F1 |
|---|---|---|---|---|
| ViT, real only | 0.907 ± 0.014 | 0.906 ± 0.015 | 0.937 ± 0.018 | 0.925 ± 0.020 |
| ViT + synthetic | 0.902 ± 0.013 | 0.901 ± 0.013 | 0.966 ± 0.012 | 0.961 ± 0.012 |
| ViT + filtered synthetic (final) | **0.918 ± 0.011** | **0.917 ± 0.011** | **0.966 ± 0.022** | **0.962 ± 0.023** |

### Synthetic-data quality filter

`scripts/filter_synthetic.py` classifies every generated image with the real-only ViT (it never saw synthetic data, so
the check is not circular) and rejects images whose intended class gets a probability below 0.2.
40 of 601 images (6.7%) were rejected, almost all from four ambiguous prompts:

| Prompt | Rejected | What MaskGen drew instead |
|---|---|---|
| "a corporate office with glass walls" | 16 / 21 | the outside of a glass building (→ street) |
| "a wide avenue with traffic lights" | 12 / 21 | traffic lights over open asphalt (→ parking lot) |
| "a residential street with parked cars" | 6 / 21 | rows of parked cars (→ parking lot) |
| "a supermarket parking lot with shopping carts" | 4 / 18 | mostly the store building (→ office / street) |

Bedroom, kitchen and park images were essentially all on-class.

### What the results show

- **The architecture is the biggest win.** The ViT, pretrained on ImageNet-21k, improves on the baseline by about
  10 points on both test sets, far beyond seed-to-seed noise.
- **Synthetic data helps where it was aimed: hard real-world photos.** On the in-the-wild set (night, snow, rain,
  unusual angles), synthetic data adds about +3 accuracy and +3.7 macro-F1 on average.
- **Unfiltered synthetic data slightly hurts the clean test set** (−0.5 points): off-class images, such as glass office
  buildings labelled "office", teach wrong cues. **Filtering them fixes this:** the filtered model is the best on the
  clean test set (+1.1 points over real-only) *and* keeps the full in-the-wild gain.
- **Prompt quality matters as much as quantity.** An earlier generation round with two bad parking-lot prompts gave only
  half the in-the-wild gain; replacing those prompts and then filtering produced the final result.
- **Statistical caution:** with 3 seeds per configuration none of the synthetic-data gains are significant at the 5%
  level (Welch's t-test: in-the-wild F1 p ≈ 0.16, test accuracy p ≈ 0.44). They are consistent trends across seeds,
  not proven effects; more seeds or a larger test set would be needed to confirm them.
- **Augmentation matters:** without it, test accuracy drops to 0.867, and `street → parking lot` errors rise from 3 to 5.
- **Hard pairs:** street ↔ parking lot (cars and asphalt) and office ↔ kitchen (counters, desks, appliances).
- **Seed noise is large:** the same configuration ranges from 0.887 to 0.920 test accuracy, so single-run comparisons
  (like 50% vs 100% synthetic) should not be over-interpreted.

### Explainability

The ResNet-18 baseline's Grad-CAM tends to lock onto **people** (a person on a bed, cyclists, a runner) and uses them as a
shortcut. The ViT's Grad-CAM spreads over **scene context**: counters and the range hood in a kitchen, vegetation and path
in a park. Attention rollout highlights a few background patches; this is the known ViT "artifact token" effect
(Darcet et al., 2023), so the app uses Grad-CAM by default.

### Synthetic scenes

![Synthetic samples](figures/synthetic_samples.png)
*All images above are AI-generated with TA-TiTok + MaskGen and are labelled as such.*

## Quick start

Requirements: Python 3.11+, an NVIDIA GPU (MaskGen generation needs CUDA; training and the app also run on CPU, slowly).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # match your CUDA driver
pip install -r requirements.txt
```

**Run the demo** (after training, or with your own weights in `runs/`):

```bash
python app.py            # http://127.0.0.1:7860
python app.py --share    # temporary public link, e.g. to use a phone camera
```

**Reproduce everything:** `run_all.bat` (Windows) or `bash run_all.sh` (Linux/macOS), or step by step:

```bash
python scripts/prepare_data.py                                          # Places365 subset (~0.5 GB)
python train.py --arch resnet18 --run baseline_resnet18                 # baseline

git clone https://github.com/bytedance/1d-tokenizer external/1d-tokenizer
pip install -r requirements-maskgen.txt
python scripts/generate_maskgen.py --weights-from runs/baseline_resnet18/metrics.json --total 600

python train.py --arch vit_small --run vit_real
python train.py --arch vit_small --run vit_real_syn --synthetic-frac 1.0
python train.py --arch vit_small --run vit_real_syn50 --synthetic-frac 0.5
python train.py --arch vit_small --run vit_real_syn_noaug --synthetic-frac 1.0 --no-aug
python train.py --arch vit_small --run vit_real_s1 --seed 1             # extra seeds
python train.py --arch vit_small --run vit_real_s2 --seed 2
python train.py --arch vit_small --run vit_real_syn_s1 --synthetic-frac 1.0 --seed 1
python train.py --arch vit_small --run vit_real_syn_s2 --synthetic-frac 1.0 --seed 2
python scripts/filter_synthetic.py                                      # quality check of synthetic images
python train.py --arch vit_small --run vit_real_synf --synthetic-frac 1.0 --filter-synthetic   # final model
python train.py --arch vit_small --run vit_real_synf_s1 --synthetic-frac 1.0 --filter-synthetic --seed 1
python train.py --arch vit_small --run vit_real_synf_s2 --synthetic-frac 1.0 --filter-synthetic --seed 2
python make_figures.py                                                  # tables + figures -> figures/
```

**Own test photos:** put them in `data/own/<class>/` before running `prepare_data.py`. They become the in-the-wild test set.

Model weights (`runs/*/best.pt`), datasets and the MaskGen repo are not stored in this repository; the commands above
recreate them. Training each model takes 2–4 minutes on a laptop GPU.

## Repository layout

```
app.py                       Gradio web app
train.py                     training + evaluation (accuracy, macro-F1, confusion matrix, latency)
make_figures.py              all tables and figures in figures/
config.py                    classes, paths, model names
scripts/prepare_data.py      Places365 subset + own photos -> data/manifest.csv
scripts/generate_maskgen.py  TA-TiTok + MaskGen synthetic scene generation
scripts/filter_synthetic.py  quality check: rejects synthetic images that don't look like their class
src/data.py                  datasets and augmentation
src/models.py                model creation, checkpoints
src/explain.py               Grad-CAM (CNN and ViT), attention rollout
src/watermark.py             visible SYNTHETIC label
figures/                     result tables and figures
runs/<run>/                  metrics.json, training curves, confusion matrices per run
```

## Ethics & disclosure

- **Synthetic media is labelled.** Every generated PNG stores `Synthetic: true`, the generator name and the prompt as
  metadata, and shows a visible **SYNTHETIC** banner wherever it is displayed (app, figures). Training copies stay clean
  so the banner can't become a shortcut feature.
- **Real-only evaluation.** Synthetic images are used for training only; every reported test number is on real images.
- **People and privacy.** Prompts avoid identifiable people (the generator still occasionally adds a synthetic person,
  who is not a real individual). Uploaded and webcam images are processed in memory and not
  stored. Scene recognition can reveal where a photo was taken, so it shouldn't be used to locate or profile people.
- **Bias.** Places365 is skewed toward Western scenes, so local streets and parking lots may be misclassified;
  MaskGen inherits the biases of its open web training data. The model knows only 6 classes and forces every photo into
  one of them (for example, indoor parking garages are out of scope).
- **Licences.** Places365 and the MaskGen weights are released for research and education.

## References

- D. Kim, J. He, Q. Yu, C. Yang, X. Shen, S. Kwak, L.-C. Chen. *Democratizing Text-to-Image Masked Generative Models with
  Compact Text-Aware One-Dimensional Tokens.* arXiv:2501.07730, 2025. Code: github.com/bytedance/1d-tokenizer.
  Summary: Neurohive, "ByteDance Unveil TA-TiTok Tokenizer Achieving SOTA in Text-to-Image Generation Using Only Public Data".
- B. Zhou et al. *Places: A 10 Million Image Database for Scene Recognition.* TPAMI, 2017.
- A. Steiner et al. *How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers.* 2021.
- R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
- S. Abnar, W. Zuidema. *Quantifying Attention Flow in Transformers.* ACL, 2020.
- T. Darcet et al. *Vision Transformers Need Registers.* 2023.
