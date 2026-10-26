"""Explainability: attention rollout (ViT) and Grad-CAM (CNN), plus heatmap overlays.

Attention rollout: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
Grad-CAM: Selvaraju et al., ICCV 2017.
"""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms as T


def display_crop(img: Image.Image, size=224):
    """The exact crop the model sees at eval time (so the heatmap lines up)."""
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size)])(img)


def overlay(img: Image.Image, heat: np.ndarray, alpha=0.45):
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_img = Image.fromarray(np.uint8(heat * 255)).resize(img.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat_img) / 255.0)[..., :3]
    base = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * colored
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))


class AttentionRollout:
    """Collects the softmax attention of every block of a timm ViT and rolls it out."""

    def __init__(self, model, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attns = []
        self.active = False
        self.handles = []
        for blk in model.blocks:
            blk.attn.fused_attn = False  # fused SDPA hides the attention matrix
            self.handles.append(blk.attn.attn_drop.register_forward_hook(self._hook))

    def _hook(self, module, inp, out):
        if self.active:  # only record during our own forward pass
            self.attns.append(out.detach())

    @torch.no_grad()
    def __call__(self, x):
        self.attns = []
        self.active = True
        try:
            logits = self.model(x)
        finally:
            self.active = False
        n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        result = None
        for a in self.attns:
            a = a[0].mean(0)  # average heads -> (N, N)
            if self.discard_ratio > 0:  # drop the weakest links to reduce noise
                flat = a.flatten()
                k = int(flat.numel() * self.discard_ratio)
                if k > 0:
                    idx = flat.topk(k, largest=False).indices
                    idx = idx[idx != 0]  # never drop the CLS->CLS entry
                    flat[idx] = 0
            a = a + torch.eye(a.size(-1), device=a.device)
            a = a / a.sum(-1, keepdim=True)
            result = a if result is None else a @ result
        cls_to_patches = result[0, n_prefix:]
        g = int(cls_to_patches.numel() ** 0.5)
        return logits, cls_to_patches.reshape(g, g).cpu().numpy()


class GradCAM:
    """Grad-CAM for a CNN feature map (B, C, H, W) or, with `n_prefix` set, for ViT tokens
    (B, N, C): the patch tokens are reshaped back into a grid (as in pytorch-grad-cam)."""

    def __init__(self, model, layer, n_prefix=None):
        self.model = model
        self.n_prefix = n_prefix
        self.acts = self.grads = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def _to_grid(self, t):
        t = t[:, self.n_prefix:, :]
        g = int(t.shape[1] ** 0.5)
        return t.reshape(t.shape[0], g, g, t.shape[2]).permute(0, 3, 1, 2)

    def __call__(self, x, target=None):
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            t = logits.argmax(1).item() if target is None else target
            logits[0, t].backward()
        acts, grads = self.acts, self.grads
        if self.n_prefix is not None:
            acts, grads = self._to_grid(acts), self._to_grid(grads)
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * acts).sum(1))[0]
        return logits.detach(), cam.detach().float().cpu().numpy()


def make_explainers(model, meta):
    """All explanation methods available for this model, the default one first."""
    if meta["arch"].startswith("vit"):
        n_prefix = getattr(model, "num_prefix_tokens", 1)
        return {
            "Grad-CAM": GradCAM(model, model.blocks[-1].norm1, n_prefix=n_prefix),
            "Attention rollout": AttentionRollout(model),
        }
    return {"Grad-CAM": GradCAM(model, model.layer4)}


def make_explainer(model, meta):
    name, fn = next(iter(make_explainers(model, meta).items()))
    return name, fn
