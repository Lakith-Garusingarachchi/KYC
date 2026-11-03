"""KYC flow: documents + address -> guided face capture -> checks -> decision."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .. import storage
from ..config import get_settings
from ..database import get_db
from ..models import Submission, User
from ..redis_client import get_redis
from ..schemas import ChallengeOut, LivenessOut, SubmissionOut
from ..security import get_current_user
from ..services import face as face_service
from ..services import decision as decision_service
from ..services import image_quality, pipeline
from ..services.llm_service import Attachment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/kyc", tags=["kyc"])

OPEN_STATUSES = ("draft", "documents_uploaded")
MAX_LIVENESS_ATTEMPTS = 3
TURN_STEPS = ["left", "right", "up"]

_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _sniff(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:5] == b"%PDF-":
        return "application/pdf"
    return None


async def _read_upload(upload: UploadFile, label: str, allow_pdf: bool) -> tuple[bytes, str]:
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await upload.read(limit + 1)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{label} is empty")
    if len(data) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"{label} is larger than {get_settings().max_upload_mb} MB")
    mime = _sniff(data)
    allowed = _IMAGE_TYPES | ({"application/pdf"} if allow_pdf else set())
    if mime not in allowed:
        kinds = "JPG, PNG, WEBP" + (" or PDF" if allow_pdf else "")
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{label} must be {kinds}")
    return data, mime


async def _own_submission(submission_id: str, user: User, db: AsyncSession) -> Submission:
    sub = await db.get(Submission, submission_id)
    if sub is None or sub.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
    return sub


async def _to_out(sub: Submission) -> SubmissionOut:
    docs = sub.document_result or {}
    idd, bill = docs.get("identity") or {}, docs.get("bill") or {}
    extracted: Dict = {}
    if idd:
        extracted["identity"] = {k: idd.get(k) for k in (
            "document_type", "full_name", "id_number", "date_of_birth", "gender", "address",
            "expiry_date", "nationality", "authenticity_assessment")}
    if bill:
        extracted["bill"] = {k: bill.get(k) for k in (
            "issuer", "bill_type", "account_holder_name", "service_address", "bill_date",
            "authenticity_assessment")}
    for key in ("identity_error", "bill_error"):
        if docs.get(key):
            extracted[key] = docs[key]

    face_out: Dict = {}
    if sub.face_result:
        fr = sub.face_result
        face_out = {
            "liveness_pass": fr.get("liveness_pass"),
            "pose_checks": fr.get("pose_checks"),
            "face_match": fr.get("face_match"),
            "issues": fr.get("issues"),
            "ai_review": fr.get("ai_review"),
        }
    report = decision_service.evaluate(sub.checks) if sub.status == "completed" and sub.checks else {}
    return SubmissionOut(
        id=sub.id, status=sub.status, decision=report.get("decision", sub.decision),
        score=report.get("score"), pass_score=report.get("pass_score"), blockers=report.get("blockers", []),
        progress=await pipeline.get_progress(sub.id) if sub.status not in ("completed", "draft") else None,
        id_type=sub.id_type, address_input=sub.address_input,
        uploaded=sorted((sub.files or {}).keys()),
        quality=sub.quality or {}, extracted=extracted, face=face_out,
        checks=sub.checks or [], error=sub.error,
        created_at=_utc(sub.created_at), updated_at=_utc(sub.updated_at),
    )


def _utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes; they are stored in UTC.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


@router.post("/submissions", response_model=SubmissionOut)
async def start_submission(
    new: bool = False,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubmissionOut:
    """Return the user's open submission, or start a new one (`?new=true` always starts fresh)."""
    if not new:
        sub = await db.scalar(
            select(Submission).where(Submission.user_id == user.id, Submission.status.in_(OPEN_STATUSES))
            .order_by(Submission.created_at.desc()).limit(1)
        )
        if sub is not None:
            return await _to_out(sub)
    sub = Submission(user_id=user.id)
    db.add(sub)
    await db.commit()
    return await _to_out(sub)


@router.get("/submissions/current", response_model=SubmissionOut)
async def current_submission(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    sub = await db.scalar(
        select(Submission).where(Submission.user_id == user.id).order_by(Submission.created_at.desc()).limit(1)
    )
    if sub is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No submission yet")
    return await _to_out(sub)


@router.get("/submissions/{submission_id}", response_model=SubmissionOut)
async def get_submission(submission_id: str, user: User = Depends(get_current_user),
                         db: AsyncSession = Depends(get_db)) -> SubmissionOut:
    return await _to_out(await _own_submission(submission_id, user, db))


@router.post("/submissions/{submission_id}/documents", response_model=SubmissionOut)
async def upload_documents(
    submission_id: str,
    background: BackgroundTasks,
    id_type: str = Form(..., pattern="^(nic|passport)$"),
    address: str = Form(..., min_length=5, max_length=500),
    id_front: UploadFile = File(...),
    utility_bill: UploadFile = File(...),
    id_back: Optional[UploadFile] = File(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SubmissionOut:
    sub = await _own_submission(submission_id, user, db)
    if sub.status not in OPEN_STATUSES:
        raise HTTPException(status.HTTP_409_CONFLICT, "This submission is already being processed")

    front, front_mime = await _read_upload(id_front, "ID front", allow_pdf=False)
    bill, bill_mime = await _read_upload(utility_bill, "Utility bill", allow_pdf=True)
    back = back_mime = None
    if id_back is not None and id_back.filename:
        back, back_mime = await _read_upload(id_back, "ID back", allow_pdf=False)

    # Upright, metadata-free JPEGs so the LLM and the face models see the same image.
    front, front_mime = await asyncio.to_thread(image_quality.normalize_image, front), "image/jpeg"
    if back is not None:
        back, back_mime = await asyncio.to_thread(image_quality.normalize_image, back), "image/jpeg"
    if bill_mime != "application/pdf":
        bill, bill_mime = await asyncio.to_thread(image_quality.normalize_image, bill), "image/jpeg"

    files: Dict = {}
    quality: Dict = {}
    for key, data, mime, upload in (("id_front", front, front_mime, id_front),
                                     ("id_back", back, back_mime, id_back),
                                     ("bill", bill, bill_mime, utility_bill)):
        if data is None:
            continue
        files[key] = {
            "path": storage.save_file(sub.id, key, data),
            "mime_type": mime,
            "filename": (upload.filename or key)[:120],
            "sha256": storage.sha256(data),
            "size": len(data),
        }
        if mime in _IMAGE_TYPES:
            quality[key] = await asyncio.to_thread(image_quality.assess, data)

    sub.id_type = id_type
    sub.address_input = address.strip()
    sub.files = files
    sub.quality = quality
    sub.id_front_sha256 = files["id_front"]["sha256"]
    sub.document_result = None
    sub.status = "documents_uploaded"
    await db.commit()

    await pipeline.set_progress(sub.id, "Reading your ID document and utility bill")
    background.add_task(pipeline.process_documents, sub.id)
    return await _to_out(sub)


@router.post("/submissions/{submission_id}/liveness/challenge", response_model=ChallengeOut)
async def liveness_challenge(submission_id: str, user: User = Depends(get_current_user),
                             db: AsyncSession = Depends(get_db)) -> ChallengeOut:
    sub = await _own_submission(submission_id, user, db)
    if sub.status != "documents_uploaded":
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload your documents first")
    turns = TURN_STEPS[:]
    random.SystemRandom().shuffle(turns)
    steps = ["center"] + turns
    challenge_id = secrets.token_urlsafe(16)
    ttl = get_settings().liveness_challenge_ttl_seconds
    await get_redis().set(f"kyc:challenge:{challenge_id}",
                          json.dumps({"submission_id": sub.id, "steps": steps}), ex=ttl)
    return ChallengeOut(challenge_id=challenge_id, steps=steps, expires_in=ttl)


@router.post("/submissions/{submission_id}/liveness", response_model=LivenessOut)
async def submit_liveness(
    submission_id: str,
    request: Request,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LivenessOut:
    sub = await _own_submission(submission_id, user, db)
    if sub.status != "documents_uploaded":
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload your documents first")

    form = await request.form()
    challenge_id = str(form.get("challenge_id") or "")
    redis = get_redis()
    raw = await redis.getdel(f"kyc:challenge:{challenge_id}") if challenge_id else None
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Challenge expired or already used. Start the capture again.")
    challenge = json.loads(raw)
    if challenge["submission_id"] != sub.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Challenge does not belong to this submission")
    steps: List[str] = challenge["steps"]

    frames: Dict[str, bytes] = {}
    frame_mimes: Dict[str, str] = {}
    for step in steps:
        upload = form.get(f"frame_{step}")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Missing frame_{step}")
        frames[step], frame_mimes[step] = await _read_upload(upload, f"Frame {step}", allow_pdf=False)

    id_front = storage.load_file(sub.files["id_front"]["path"])
    try:
        result = await asyncio.to_thread(face_service.analyze_liveness, frames, steps, id_front)
    except face_service.FaceModelsMissingError as err:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(err))

    attempts_key = f"kyc:liveness_attempts:{sub.id}"
    attempts = int(await redis.incr(attempts_key))
    await redis.expire(attempts_key, 24 * 3600)
    attempts_left = max(MAX_LIVENESS_ATTEMPTS - attempts, 0)

    accepted = result["liveness_pass"] or attempts_left == 0
    if accepted:
        for step, data in frames.items():
            sub.files = {**(sub.files or {}), f"frame_{step}": {
                "path": storage.save_file(sub.id, f"frame_{step}", data), "mime_type": frame_mimes[step],
                "filename": f"{step}.jpg", "sha256": storage.sha256(data), "size": len(data)}}
        result["attempts"] = attempts
        sub.face_result = result
        sub.status = "processing"
        await db.commit()
        await pipeline.set_progress(sub.id, "Reviewing your face capture")
        attachments = [Attachment(data=frames[s], mime_type=frame_mimes[s], filename=f"{s}.jpg") for s in steps]
        background.add_task(pipeline.process_liveness_review, sub.id, attachments, steps)

    return LivenessOut(
        accepted=accepted,
        liveness_pass=result["liveness_pass"],
        attempts_left=attempts_left,
        issues=result["issues"],
        face_match=result["face_match"],
        submission_status=sub.status,
    )
