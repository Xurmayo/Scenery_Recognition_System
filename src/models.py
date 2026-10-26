"""Model creation, checkpoint save/load."""
import timm
import torch

from config import ARCHS, CLASSES


def build_model(arch, num_classes=len(CLASSES), pretrained=True):
    model = timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=num_classes)
    cfg = timm.data.resolve_data_config({}, model=model)
    meta = {
        "arch": arch,
        "timm_name": ARCHS[arch],
        "classes": list(CLASSES),
        "mean": list(cfg["mean"]),
        "std": list(cfg["std"]),
        "img_size": int(cfg["input_size"][-1]),
    }
    return model, meta


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = timm.create_model(meta["timm_name"], pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta
