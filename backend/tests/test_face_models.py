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


def _patterned_card(portrait, photo_w, contrast):
    """A card whose small portrait is faded and covered by a wavy security pattern."""
    import numpy as np

    face = portrait[20:300, 90:380]
    card = np.full((640, 1010, 3), (205, 190, 170), np.uint8)
    p = cv2.resize(face, (photo_w, int(photo_w * 1.25))).astype(np.float32)
    p = (p - p.mean()) * contrast + p.mean()
    ph, pw = p.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    p += (np.sin(xx / 3.0 + np.sin(yy / 7.0) * 3) * 18)[..., None]
    card[150:150 + ph, 60:60 + pw] = np.clip(p, 0, 255).astype(np.uint8)
    scene = np.full((1500, 2000, 3), (60, 80, 110), np.uint8)
    scene[400:1040, 500:1510] = card
    box = [(560) / 2000, (550) / 1500, (560 + pw) / 2000, (550 + ph) / 1500]
    return cv2.GaussianBlur(scene, (3, 3), 0), box


def test_faded_patterned_id_photo_is_found(models_available, portrait):
    scene, _ = _patterned_card(portrait, 140, 0.45)
    engine = face_service.get_face_engine()
    assert engine.detect_document_face(face_service.decode_image(_jpeg(scene), face_service.DOC_MAX_SIDE))


def test_tiny_faded_portrait_is_found_and_compared(models_available, portrait):
    scene, _ = _patterned_card(portrait, 90, 0.45)
    match = face_service.analyze_liveness({"center": _jpeg(portrait)}, ["center"], _jpeg(scene))["face_match"]
    assert match["verdict"] != "no_id_face" and match["score"] is not None


def test_photo_box_retry_matches_small_portrait(models_available, portrait):
    scene, box = _patterned_card(portrait, 90, 0.7)
    result = face_service.match_with_photo_box(_jpeg(scene), box, _jpeg(portrait))
    assert result is not None and result["verdict"] == "match", result
    assert face_service.match_with_photo_box(_jpeg(scene), [0.9, 0.9, 0.1, 0.1], _jpeg(portrait)) is None
