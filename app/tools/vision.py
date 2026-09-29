from pathlib import Path
from PIL import Image, ImageStat
import numpy as np

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def analyze_image(path: str | None) -> dict:
    if not path:
        return {
            "available": False,
            "message": "No image supplied.",
            "quality_score": 0.0,
            "flags": ["missing_image"]
        }

    p = Path(path)
    if not p.exists():
        return {
            "available": False,
            "message": f"Image not found: {p}",
            "quality_score": 0.0,
            "flags": ["image_not_found"]
        }
    if p.suffix.lower() not in ALLOWED_EXT:
        return {
            "available": False,
            "message": "Unsupported image format for this research prototype.",
            "quality_score": 0.0,
            "flags": ["unsupported_format"]
        }

    with Image.open(p) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        stat = ImageStat.Stat(rgb)
        mean = float(gray.mean())
        std = float(gray.std())
        dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        quality = float(np.clip(0.45 + 0.7 * min(std, 0.35) + 0.7 * min(dynamic_range, 0.8), 0, 1))

        flags = []
        if rgb.width < 128 or rgb.height < 128:
            flags.append("low_resolution")
        if std < 0.03:
            flags.append("low_contrast")
        if mean < 0.05 or mean > 0.95:
            flags.append("extreme_brightness")

        return {
            "available": True,
            "width": rgb.width,
            "height": rgb.height,
            "mean_intensity": round(mean, 5),
            "intensity_std": round(std, 5),
            "dynamic_range": round(dynamic_range, 5),
            "quality_score": round(quality, 4),
            "flags": flags,
            "research_note": "These are generic image-quality features, not disease findings."
        }
