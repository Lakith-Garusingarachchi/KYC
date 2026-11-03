"""Cheap upload-quality checks run before any LLM call."""

from __future__ import annotations

import io
from typing import Dict

import cv2
import numpy as np
from PIL import Image, ImageOps

try:  # iPhone HEIC/HEIF photos; converted to JPEG on upload
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

MIN_SHORT_SIDE = 500
MIN_SHARPNESS = 60.0
MIN_BRIGHTNESS, MAX_BRIGHTNESS = 50.0, 225.0


def assess(data: bytes) -> Dict:
    """Resolution, sharpness (variance of Laplacian) and exposure of an image."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"readable": False, "issues": ["File is not a readable image"]}
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Measure sharpness at a fixed width so big photos aren't favoured.
    scale = 1000 / w
    norm = cv2.resize(gray, (1000, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    brightness = float(gray.mean())
    glare = float((gray > 250).mean())

    issues = []
    if min(h, w) < MIN_SHORT_SIDE:
        issues.append(f"Low resolution ({w}x{h}); use a photo at least {MIN_SHORT_SIDE}px on the short side")
    if sharpness < MIN_SHARPNESS:
        issues.append("Image looks blurry")
    if brightness < MIN_BRIGHTNESS:
        issues.append("Image is too dark")
    elif brightness > MAX_BRIGHTNESS:
        issues.append("Image is overexposed")
    if glare > 0.08:
        issues.append("Strong glare on the document")
    return {
        "readable": True,
        "width": w,
        "height": h,
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "glare_ratio": round(glare, 3),
        "issues": issues,
    }


def normalize_image(data: bytes) -> bytes:
    """Apply the EXIF rotation and drop metadata (GPS etc.), re-saving as JPEG.

    The LLM and the face models then see exactly the same, upright pixels.
    """
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()
