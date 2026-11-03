"""Cheap upload-quality checks run before any LLM call."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

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
