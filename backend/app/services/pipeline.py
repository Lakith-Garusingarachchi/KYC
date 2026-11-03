"""Background processing for a submission.

  documents uploaded  -> process_documents()      (LLM reads ID + bill)
  liveness accepted   -> process_liveness_review() (LLM reviews the frames)
  both finished       -> maybe_finalize()          (address check, rules, decision)

Progress text lives in Redis so the frontend can poll it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from sqlalchemy import select

from ..config import get_settings
from ..database import SessionLocal
from ..models import Submission
from ..redis_client import get_redis
from .. import storage
from . import decision, document_ai, storage_helpers
from . import face as face_service
from .llm_service import Attachment

logger = logging.getLogger(__name__)

PROGRESS_TTL = 3600


def progress_key(submission_id: str) -> str:
    return f"kyc:progress:{submission_id}"


async def set_progress(submission_id: str, text: str) -> None:
    try:
        await get_redis().set(progress_key(submission_id), text, ex=PROGRESS_TTL)
    except Exception:  # progress is cosmetic
        logger.debug("progress update failed", exc_info=True)


async def get_progress(submission_id: str) -> Optional[str]:
    try:
        return await get_redis().get(progress_key(submission_id))
    except Exception:
        return None


def _short_error(err: BaseException) -> str:
    text = f"{type(err).__name__}: {err}"
    return text[:300]


async def process_documents(submission_id: str) -> None:
    async with SessionLocal() as db:
        sub = await db.get(Submission, submission_id)
        if sub is None:
            return
        files = dict(sub.files or {})
        id_type = sub.id_type or "nic"
        front_sha = sub.id_front_sha256

    await set_progress(submission_id, "Reading your ID document and utility bill")
    front = storage_helpers.attachment(files, "id_front")
    back = storage_helpers.attachment(files, "id_back")
    bill = storage_helpers.attachment(files, "bill")

    identity, bill_data = await asyncio.gather(
        document_ai.extract_identity(id_type, front, back),
        document_ai.extract_bill(bill),
        return_exceptions=True,
    )
    result: Dict = {}
    if isinstance(identity, BaseException):
        logger.warning("identity extraction failed for %s: %s", submission_id, identity)
        result["identity_error"] = _short_error(identity)
    else:
        result["identity"] = identity
    if isinstance(bill_data, BaseException):
        logger.warning("bill extraction failed for %s: %s", submission_id, bill_data)
        result["bill_error"] = _short_error(bill_data)
    else:
        result["bill"] = bill_data

    async with SessionLocal() as db:
        sub = await db.get(Submission, submission_id)
        if sub is None or sub.id_front_sha256 != front_sha:
            return  # documents were replaced while we were reading them
        sub.document_result = result
        await db.commit()
    await set_progress(submission_id, "Documents read")
    await maybe_finalize(submission_id)


async def process_liveness_review(submission_id: str, frames: List[Attachment], steps: List[str]) -> None:
    await set_progress(submission_id, "Reviewing your face capture")
    review: Optional[Dict] = None
    error: Optional[str] = None
    try:
        review = await document_ai.review_liveness(frames, steps)
    except Exception as err:  # the AI review is an extra signal; never block on it
        logger.warning("liveness review failed for %s: %s", submission_id, err)
        error = _short_error(err)

    async with SessionLocal() as db:
        sub = await db.get(Submission, submission_id)
        if sub is None or sub.face_result is None:
            return
        face = dict(sub.face_result)
        face["ai_review"] = review
        face["ai_review_error"] = error
        face["ai_review_done"] = True
        sub.face_result = face
        await db.commit()
    await maybe_finalize(submission_id)


async def _retry_face_match(files: Dict, identity: Dict, current: Dict) -> Optional[Dict]:
    """Match again on the ID portrait the LLM located; keep it only if it scores higher."""
    box = identity.get("photo_box")
    if not box or "id_front" not in files or "frame_center" not in files:
        return None
    try:
        id_image = storage.load_file(files["id_front"]["path"])
        frame = storage.load_file(files["frame_center"]["path"])
        retry = await asyncio.to_thread(face_service.match_with_photo_box, id_image, box, frame)
    except Exception:
        logger.warning("face match retry failed", exc_info=True)
        return None
    if retry is None:
        return None
    if current.get("score") is None or retry["score"] > current["score"]:
        return retry
    return None


async def maybe_finalize(submission_id: str) -> None:
    """Run the decision once both the document reading and the face review are done.

    Both background tasks call this when they finish. A Redis lock makes sure only
    one runs at a time; a "pending" flag makes the lock holder re-check, so a task
    that finishes while the other holds the lock is never lost.
    """
    redis = get_redis()
    lock_key = f"kyc:finalize:{submission_id}"
    pending_key = f"kyc:finalize_pending:{submission_id}"
    await redis.set(pending_key, "1", ex=300)
    while True:
        if not await redis.set(lock_key, "1", nx=True, ex=300):
            return  # the holder will see our pending flag
        try:
            while await redis.delete(pending_key):
                if await _finalize_once(submission_id):
                    await redis.delete(pending_key)
                    return
        finally:
            await redis.delete(lock_key)
        # A task may have flagged us between our last check and the unlock.
        if not await redis.exists(pending_key):
            return


async def _finalize_once(submission_id: str) -> bool:
    """Returns True when the submission is completed (or failed)."""
    try:
        async with SessionLocal() as db:
            sub = await db.get(Submission, submission_id)
            if sub is None or sub.status in ("completed", "failed"):
                return True
            if sub.document_result is None or not (sub.face_result or {}).get("ai_review_done"):
                return False  # the other task will flag us again

            await set_progress(submission_id, "Comparing your address and running checks")
            docs = sub.document_result
            bill = docs.get("bill") or {}
            idd = docs.get("identity") or {}
            address_llm = None
            bill_addr = bill.get("service_address") or bill.get("billing_address")
            if bill_addr and sub.address_input:
                try:
                    address_llm = await document_ai.compare_addresses(
                        sub.address_input, bill_addr, idd.get("address", ""), get_settings().text_provider)
                except Exception as err:
                    logger.warning("address comparison failed for %s: %s", submission_id, err)

            face_result = dict(sub.face_result or {})
            better = await _retry_face_match(sub.files or {}, idd, face_result.get("face_match") or {})
            if better is not None:
                face_result["face_match"] = better
                sub.face_result = face_result

            duplicate = False
            if sub.id_front_sha256:
                other = await db.execute(
                    select(Submission.id).where(
                        Submission.id_front_sha256 == sub.id_front_sha256,
                        Submission.user_id != sub.user_id,
                    ).limit(1)
                )
                duplicate = other.first() is not None

            checks = decision.build_checks(
                id_type=sub.id_type or "nic",
                typed_address=sub.address_input or "",
                document_result=docs,
                face_result=face_result,
                quality=sub.quality or {},
                address_llm=address_llm,
                duplicate_of_other_user=duplicate,
            )
            sub.checks = checks
            sub.decision = decision.evaluate(checks)["decision"]
            sub.status = "completed"
            await db.commit()
        await set_progress(submission_id, "Done")
        return True
    except Exception as err:
        logger.exception("finalize failed for %s", submission_id)
        async with SessionLocal() as db:
            sub = await db.get(Submission, submission_id)
            if sub is not None:
                sub.status = "failed"
                sub.error = _short_error(err)
                await db.commit()
        return True
