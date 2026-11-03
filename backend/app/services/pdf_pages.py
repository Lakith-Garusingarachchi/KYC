"""Render the first page of an uploaded PDF as an image.

The LLM reads the original PDF; this image is for the OpenCV checks (finding
the photo on a PDF ID for the face match), which cannot open PDFs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pypdfium2 as pdfium

RENDER_DPI = 300
MAX_SIDE = 3000  # px; caps memory for oversized pages
MARGIN = 0.02    # padding kept around the content, as a share of the longer side


def first_page_jpeg(data: bytes) -> bytes:
    """First page as JPEG, cropped to its content so a card scanned on A4 fills the image.

    Raises ValueError when the PDF cannot be opened (corrupt, password-protected, no pages).
    """
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as err:
        raise ValueError("PDF could not be opened") from err
    try:
        if len(pdf) == 0:
            raise ValueError("PDF has no pages")
        page = pdf[0]
        width, height = page.get_size()  # points (1/72 inch)
        scale = min(RENDER_DPI / 72, MAX_SIDE / max(width, height, 1))
        rgb = np.asarray(page.render(scale=scale).to_pil().convert("RGB"))
        page.close()
    except pdfium.PdfiumError as err:
        raise ValueError("PDF page could not be rendered") from err
    finally:
        pdf.close()

    img = _crop_to_content(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("PDF page could not be converted to an image")
    return buf.tobytes()


def _crop_to_content(img: np.ndarray) -> np.ndarray:
    """Trim the white page around a scanned card; returns the image unchanged if it is all content."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray < 235)
    if len(xs) == 0:
        return img
    h, w = gray.shape
    pad = int(max(h, w) * MARGIN)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
    return img[y0:y1, x0:x1]
