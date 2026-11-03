import io

import cv2
import numpy as np
import pytest
from PIL import Image

from app.services import pdf_pages


def _pdf(img: np.ndarray, dpi: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PDF", resolution=dpi)
    return buf.getvalue()


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_card_scanned_on_a4_is_cropped_to_the_card():
    page = np.full((3508, 2480, 3), 255, np.uint8)  # A4 at 300 DPI
    page[400:1038, 300:1311] = 120                 # an 85.6 x 54 mm card
    h, w = _decode(pdf_pages.first_page_jpeg(_pdf(page, 300))).shape[:2]
    # The page renders at ~3000 px tall; only the card (plus a small margin) is kept.
    assert 800 < w < 1100
    assert 500 < h < 750


def test_full_page_photo_is_kept_whole():
    photo = np.random.default_rng(2).integers(60, 200, (640, 1000, 3), dtype=np.uint8)
    h, w = _decode(pdf_pages.first_page_jpeg(_pdf(photo, 150))).shape[:2]
    # 1000 px at 150 DPI renders at 300 DPI: 2000 px wide, nothing trimmed.
    assert abs(w - 2000) <= 1
    assert abs(h - 1280) <= 1


def test_unreadable_pdf_is_rejected():
    with pytest.raises(ValueError):
        pdf_pages.first_page_jpeg(b"%PDF-1.4\n% not a real pdf\n")
