"""Runs the real YuNet/SFace models on a public-domain test photo (scikit-image's
NASA astronaut portrait). Skipped when the models or scikit-image are missing."""

import cv2
import pytest

from app.services import face as face_service

skimage_data = pytest.importorskip("skimage.data")


def _jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


@pytest.fixture
def portrait():
    return cv2.cvtColor(skimage_data.astronaut(), cv2.COLOR_RGB2BGR)


def test_same_photo_matches_id_but_fails_liveness(models_available, portrait):
    frame = _jpeg(portrait)
    # A small "ID card" built from a downscaled copy of the same face.
    card = cv2.resize(portrait, (256, 256))
    card = cv2.copyMakeBorder(card, 40, 40, 300, 40, cv2.BORDER_CONSTANT, value=(220, 220, 200))
    steps = ["center", "left", "right", "up"]
    result = face_service.analyze_liveness({s: frame for s in steps}, steps, _jpeg(card))

    assert result["face_match"]["verdict"] == "match"
    assert result["face_match"]["score"] > 0.6
    assert result["same_person"] is True
    # One still photo replayed for every step: no head movement, identical frames.
    assert result["static_suspect"] is True
    assert result["liveness_pass"] is False
    assert not any(result["pose_checks"].values())


def test_rotated_id_photo_is_found(models_available, portrait):
    card = cv2.rotate(cv2.resize(portrait, (300, 300)), cv2.ROTATE_90_CLOCKWISE)
    engine = face_service.get_face_engine()
    assert engine.detect_document_face(face_service.decode_image(_jpeg(card))) is not None


def test_blank_image_has_no_face(models_available):
    import numpy as np

    blank = np.full((480, 640, 3), 128, np.uint8)
    steps = ["center", "left"]
    result = face_service.analyze_liveness({s: _jpeg(blank) for s in steps}, steps, None)
    assert result["liveness_pass"] is False
    assert any("No face" in i for i in result["issues"])
