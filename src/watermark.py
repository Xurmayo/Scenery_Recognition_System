"""Visible disclosure label for synthetic images (used in the app and report figures)."""
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for name in ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stamp_synthetic(img: Image.Image, text="SYNTHETIC · AI-generated (MaskGen)"):
    img = img.convert("RGB").copy()
    w, h = img.size
    band = max(16, h // 12)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, h - band, w, h], fill=(200, 30, 30, 210))
    font = _font(int(band * 0.6))
    tw = draw.textlength(text, font=font)
    if tw > w - 8:  # shorten on small images
        text = "SYNTHETIC"
        tw = draw.textlength(text, font=font)
    draw.text(((w - tw) / 2, h - band + band * 0.18), text, fill="white", font=font)
    return img
