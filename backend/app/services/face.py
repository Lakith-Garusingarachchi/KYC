"""Face detection, head-pose estimate and face matching on CPU.

Models (OpenCV Zoo, ONNX, run on CPU — no GPU needed):
  * YuNet  face_detection_yunet_2023mar.onnx  — face box + 5 landmarks
  * SFace  face_recognition_sface_2021dec.onnx — 128-d face embedding

Head pose comes from the 5 landmarks of the *unmirrored* camera frame:
  yaw   = (nose_x - eye_mid_x) / eye_distance      > 0 when the person turns to THEIR left
  pitch = (nose_y - eye_mid_y) / (mouth_mid_y - eye_mid_y)  smaller when looking up
Each turned frame is compared with the person's own centre frame, so the
thresholds are relative and work for any face shape.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"

_MAX_SIDE = 1280
DOC_MAX_SIDE = 2400        # keep more detail when looking for the small ID portrait
_MAX_UPSCALED_SIDE = 3000


class FaceModelsMissingError(RuntimeError):
    pass


@dataclass
class DetectedFace:
    row: np.ndarray  # YuNet row: x, y, w, h, 10 landmark coords, score
    image: np.ndarray  # the (possibly resized/rotated) image the row refers to

    @property
    def score(self) -> float:
        return float(self.row[14])

    @property
    def area(self) -> float:
        return float(self.row[2] * self.row[3])

    @property
    def size(self) -> Tuple[int, int]:
        return int(self.row[2]), int(self.row[3])

    def pose(self) -> Tuple[float, float]:
        r = self.row
        right_eye, left_eye = np.array(r[4:6]), np.array(r[6:8])
        nose = np.array(r[8:10])
        mouth_r, mouth_l = np.array(r[10:12]), np.array(r[12:14])
        return pose_from_landmarks(right_eye, left_eye, nose, mouth_r, mouth_l)


def pose_from_landmarks(right_eye, left_eye, nose, mouth_r, mouth_l) -> Tuple[float, float]:
    eye_mid = (np.asarray(right_eye, float) + np.asarray(left_eye, float)) / 2
    mouth_mid = (np.asarray(mouth_r, float) + np.asarray(mouth_l, float)) / 2
    eye_dist = float(np.linalg.norm(np.asarray(left_eye, float) - np.asarray(right_eye, float))) or 1.0
    face_height = float(mouth_mid[1] - eye_mid[1]) or 1.0
    yaw = (float(nose[0]) - float(eye_mid[0])) / eye_dist
    pitch = (float(nose[1]) - float(eye_mid[1])) / face_height
    return round(yaw, 4), round(pitch, 4)


def decode_image(data: bytes, max_side: int = _MAX_SIDE) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # applies EXIF orientation
    if img is None:
        raise ValueError("not a readable image (use JPG, PNG or WEBP)")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel: brings out faces under laminate glare or guilloche."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l_chan)
    return cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


class FaceEngine:
    def __init__(self, models_dir: Path):
        det_path, rec_path = models_dir / YUNET_FILE, models_dir / SFACE_FILE
        missing = [p.name for p in (det_path, rec_path) if not p.is_file()]
        if missing:
            raise FaceModelsMissingError(
                f"Face models missing in {models_dir}: {', '.join(missing)}. "
                f"Run backend/scripts/download_models.sh"
            )
        self._detector = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), 0.7, 0.3, 5000)
        # ID photos are small, printed and covered by security patterns: accept lower scores.
        self._doc_detector = cv2.FaceDetectorYN.create(str(det_path), "", (320, 320), 0.5, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
        self._lock = threading.Lock()

    # -- detection ----------------------------------------------------------------
    def detect(self, img: np.ndarray, *, document: bool = False) -> List[DetectedFace]:
        h, w = img.shape[:2]
        detector = self._doc_detector if document else self._detector
        with self._lock:
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
        if faces is None:
            return []
        found = [DetectedFace(row=f, image=img) for f in faces]
        return sorted(found, key=lambda f: f.area, reverse=True)

    def detect_document_face(self, img: np.ndarray) -> Optional[DetectedFace]:
        """The portrait on an ID/passport.

        Tries the four rotations, a contrast-enhanced copy and an upscaled copy, and
        keeps the most confident detection (the largest face of each pass, so the
        small "ghost" portrait is not picked over the main photo).
        """
        rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        for rot in rotations:
            base = img if rot is None else cv2.rotate(img, rot)
            best: Optional[DetectedFace] = None
            for variant in (base, enhance_contrast(base)):
                for scale in (1.0, 1.8):
                    work = variant
                    if scale != 1.0:
                        h, w = variant.shape[:2]
                        factor = min(scale, _MAX_UPSCALED_SIDE / max(h, w))
                        if factor <= 1.05:
                            continue
                        work = cv2.resize(variant, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
                    faces = self.detect(work, document=True)
                    if faces and (best is None or faces[0].score > best.score):
                        best = faces[0]
                if best is not None and best.score >= 0.8:
                    return best
            if best is not None:
                return best
        return None

    # -- recognition ----------------------------------------------------------------
    def embed(self, face: DetectedFace) -> np.ndarray:
        with self._lock:
            aligned = self._recognizer.alignCrop(face.image, face.row)
            return self._recognizer.feature(aligned).copy()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        with self._lock:
            return round(float(self._recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE)), 4)


_engine: Optional[FaceEngine] = None
_engine_lock = threading.Lock()


def get_face_engine() -> FaceEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = FaceEngine(Path(get_settings().face_models_dir))
        return _engine


def set_face_engine(engine) -> None:  # tests
    global _engine
    _engine = engine


# --- liveness + match --------------------------------------------------------------
def _gray_thumb(img: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)


def _blur_score(img: np.ndarray) -> float:
    return round(float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 1)


def pose_step_passed(step: str, yaw: float, pitch: float, base_yaw: float, base_pitch: float,
                     yaw_delta: float, pitch_delta: float) -> bool:
    if step == "left":
        return yaw - base_yaw >= yaw_delta
    if step == "right":
        return base_yaw - yaw >= yaw_delta
    if step == "up":
        return base_pitch - pitch >= pitch_delta
    if step == "down":
        return pitch - base_pitch >= pitch_delta
    return True


def analyze_liveness(frames: Dict[str, bytes], steps: List[str], id_image: Optional[bytes]) -> Dict:
    """Check the guided-capture frames and match the centre frame with the ID photo.

    `frames` maps step name (center, left, right, up, down) -> JPEG bytes.
    """
    settings = get_settings()
    engine = get_face_engine()
    issues: List[str] = []
    per_frame: Dict[str, Dict] = {}
    embeddings: Dict[str, np.ndarray] = {}
    images: Dict[str, np.ndarray] = {}
    poses: Dict[str, Tuple[float, float]] = {}

    for step in steps:
        data = frames.get(step)
        info: Dict = {"received": data is not None}
        per_frame[step] = info
        if data is None:
            issues.append(f"Missing frame for '{step}'")
            continue
        try:
            img = decode_image(data)
        except ValueError:
            info["issue"] = "unreadable image"
            issues.append(f"Frame '{step}' is not a readable image")
            continue
        images[step] = img
        faces = engine.detect(img)
        info["faces"] = len(faces)
        info["sharpness"] = _blur_score(img)
        if not faces:
            info["issue"] = "no face"
            issues.append(f"No face found in the '{step}' frame")
            continue
        if len(faces) > 1 and faces[1].area > 0.25 * faces[0].area:
            info["issue"] = "more than one face"
            issues.append(f"More than one face in the '{step}' frame")
        face = faces[0]
        yaw, pitch = face.pose()
        poses[step] = (yaw, pitch)
        info.update({"yaw": yaw, "pitch": pitch, "detector_score": round(face.score, 3)})
        embeddings[step] = engine.embed(face)

    # Pose: every turned frame must move the right way relative to the centre frame.
    pose_results: Dict[str, bool] = {}
    if "center" in poses:
        base_yaw, base_pitch = poses["center"]
        for step in steps:
            if step == "center":
                continue
            if step not in poses:
                pose_results[step] = False
                continue
            ok = pose_step_passed(step, *poses[step], base_yaw, base_pitch,
                                  settings.liveness_yaw_delta, settings.liveness_pitch_delta)
            pose_results[step] = ok
            if not ok:
                issues.append(f"Head did not turn {step} as asked")
    else:
        issues.append("No usable centre frame")

    # Same person in every frame.
    same_person_scores: Dict[str, float] = {}
    if "center" in embeddings:
        for step, emb in embeddings.items():
            if step != "center":
                same_person_scores[step] = engine.similarity(embeddings["center"], emb)
    same_person_ok = bool(same_person_scores) and min(same_person_scores.values()) >= settings.face_match_threshold
    if same_person_scores and not same_person_ok:
        issues.append("The frames do not all show the same person")

    # Frames that are pixel-identical point to a still image being replayed.
    thumbs = [_gray_thumb(img) for img in images.values()]
    diffs = [float(np.mean(np.abs(thumbs[i] - thumbs[j])))
             for i in range(len(thumbs)) for j in range(i + 1, len(thumbs))]
    static_suspect = bool(diffs) and max(diffs) < 1.0
    if static_suspect:
        issues.append("Frames are identical (possible photo or replay)")

    liveness_pass = (
        len(poses) == len(steps)
        and all(pose_results.values())
        and same_person_ok
        and not static_suspect
        and not any(f.get("issue") == "more than one face" for f in per_frame.values())
    )

    # Face match against the ID photo.
    match: Dict = {"score": None, "verdict": "no_id_face"}
    if id_image is not None and "center" in embeddings:
        try:
            id_face = engine.detect_document_face(decode_image(id_image, DOC_MAX_SIDE))
        except ValueError:
            id_face = None
        if id_face is None:
            issues.append("No face found on the ID document")
        else:
            id_emb = engine.embed(id_face)
            scores = [engine.similarity(id_emb, emb) for emb in embeddings.values()]
            best = max(scores)
            match = {
                "score": best,
                "center_score": engine.similarity(id_emb, embeddings["center"]),
                "id_face_size": list(id_face.size),
                "verdict": ("match" if best >= settings.face_match_threshold
                            else "mismatch" if best < settings.face_reject_threshold else "uncertain"),
            }

    return {
        "steps": steps,
        "frames": per_frame,
        "pose_checks": pose_results,
        "same_person_scores": same_person_scores,
        "same_person": same_person_ok,
        "static_suspect": static_suspect,
        "liveness_pass": liveness_pass,
        "face_match": match,
        "issues": issues,
        "thresholds": {
            "match": settings.face_match_threshold,
            "reject": settings.face_reject_threshold,
            "yaw_delta": settings.liveness_yaw_delta,
            "pitch_delta": settings.liveness_pitch_delta,
        },
    }


def _verdict(score: float) -> str:
    settings = get_settings()
    if score >= settings.face_match_threshold:
        return "match"
    return "mismatch" if score < settings.face_reject_threshold else "uncertain"


def match_with_photo_box(id_image: bytes, box: List[float], frame: bytes) -> Optional[Dict]:
    """Second try at the face match, using the portrait box the LLM found on the ID.

    `box` is [left, top, right, bottom] as fractions of the ID image. The portrait
    is cropped with a margin and enlarged, which finds faces the full-image pass
    misses (small, low-contrast or patterned ID photos).
    """
    if not box or len(box) != 4:
        return None
    left, top, right, bottom = (float(v) for v in box)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1) or (right - left) * (bottom - top) < 0.002:
        return None
    engine = get_face_engine()
    img = decode_image(id_image, DOC_MAX_SIDE)
    h, w = img.shape[:2]
    mx, my = (right - left) * w * 0.3, (bottom - top) * h * 0.3
    x0, y0 = max(int(left * w - mx), 0), max(int(top * h - my), 0)
    x1, y1 = min(int(right * w + mx), w), min(int(bottom * h + my), h)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    target_w = 360
    crop = cv2.resize(crop, (target_w, max(1, int(crop.shape[0] * target_w / crop.shape[1]))),
                      interpolation=cv2.INTER_CUBIC)
    id_face = engine.detect_document_face(crop)
    frame_faces = engine.detect(decode_image(frame))
    if id_face is None or not frame_faces:
        return None
    score = engine.similarity(engine.embed(id_face), engine.embed(frame_faces[0]))
    return {"score": score, "center_score": score, "verdict": _verdict(score), "source": "id_photo_box"}
