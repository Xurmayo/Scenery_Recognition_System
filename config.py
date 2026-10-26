"""Shared settings for the scene-classification project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "figures"

# Six scene classes: three indoor, three outdoor.
CLASSES = ["bedroom", "kitchen", "office", "park", "parking_lot", "street"]
INDOOR = {"bedroom", "kitchen", "office"}

# Category names as they appear in Places365 (torchvision keeps the "/x/name" form).
PLACES365_NAMES = {
    "bedroom": "/b/bedroom",
    "kitchen": "/k/kitchen",
    "office": "/o/office",
    "park": "/p/park",
    "parking_lot": "/p/parking_lot",
    "street": "/s/street",
}

# timm model names. The baseline plays the role of the "A1 baseline".
ARCHS = {
    "resnet18": "resnet18.a1_in1k",                               # CNN baseline (Grad-CAM)
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",    # SOTA classifier (attention rollout)
    "vit_base": "vit_base_patch16_224.augreg2_in21k_ft_in1k",     # bigger option if your GPU allows
}

MANIFEST = DATA_DIR / "manifest.csv"           # every real / own / synthetic image with its split
SYNTH_DIR = DATA_DIR / "synthetic"             # MaskGen outputs (clean copies used for training)
SYNTH_META = SYNTH_DIR / "synthetic_meta.csv"  # prompt, seed, model for every synthetic image
