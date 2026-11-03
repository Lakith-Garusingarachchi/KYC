"""Turn extracted data + face results into a list of checks and one decision.

Each check: {id, category, label, status: pass|warn|fail, detail}.
Decision: any fail -> rejected, any warn -> manual_review, else approved.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from ..config import get_settings
from . import id_rules, matching

PASS, WARN, FAIL = "pass", "warn", "fail"


def _check(checks: List[Dict], cid: str, category: str, label: str, status: str, detail: str = "") -> None:
    checks.append({"id": cid, "category": category, "label": label, "status": status, "detail": detail})


def _parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _authenticity_check(checks: List[Dict], cid: str, category: str, label: str, doc: Dict) -> None:
    """Major tamper signs (or a 'suspicious' verdict) need review; minor notes still pass."""
    signs = doc.get("tamper_signs") or []
    severity = doc.get("tamper_severity") or ("minor" if signs else "none")
    if doc.get("authenticity_assessment") == "suspicious" or severity == "major":
        _check(checks, cid, category, label, WARN, "; ".join(signs) or "Model flagged the document as suspicious")
    elif signs:
        _check(checks, cid, category, label, PASS, "Minor notes only: " + "; ".join(signs))
    else:
        _check(checks, cid, category, label, PASS, "No tamper signs found")


def _id_checks(checks: List[Dict], id_type: str, idd: Dict, quality: Dict, today: date) -> None:
    cat = "document"
    if not idd.get("is_identity_document"):
        _check(checks, "id_valid", cat, "ID document recognised", FAIL,
               idd.get("notes") or "The uploaded file does not look like an identity document")
        return
    legible = idd.get("text_legible", True)
    _check(checks, "id_valid", cat, "ID document recognised", PASS if legible else WARN,
           "Readable" if legible else "Some text is not legible")

    detected = idd.get("document_type") or "other"
    expected_nic = id_type == "nic"
    if (expected_nic and detected == "passport") or (not expected_nic and detected.startswith("nic")):
        _check(checks, "id_type", cat, "Document type matches selection", WARN,
               f"You selected {id_type.upper()} but the document looks like a {detected}")

    # Number / MRZ rules
    if expected_nic:
        nic = id_rules.parse_nic(idd.get("id_number", ""), today=today)
        if not idd.get("id_number"):
            _check(checks, "id_number", cat, "NIC number format", WARN, "NIC number not readable")
        elif not nic.valid:
            _check(checks, "id_number", cat, "NIC number format", FAIL, "; ".join(nic.errors))
        else:
            _check(checks, "id_number", cat, "NIC number format", PASS, f"{nic.format} format, {nic.number}")
            dob = _parse_date(idd.get("date_of_birth"))
            problems = []
            if dob and dob != nic.birth_date:
                problems.append(f"printed birth date {dob} vs {nic.birth_date} encoded in the number")
            if idd.get("gender") in ("M", "F") and idd["gender"] != nic.gender:
                problems.append(f"printed gender {idd['gender']} vs {nic.gender} encoded in the number")
            if problems:
                _check(checks, "nic_consistency", cat, "NIC number matches printed details", WARN, "; ".join(problems))
            elif dob:
                _check(checks, "nic_consistency", cat, "NIC number matches printed details", PASS,
                       "Birth date and gender agree with the number")
    else:
        mrz = id_rules.parse_td3(idd.get("mrz_line1", ""), idd.get("mrz_line2", ""), today=today)
        if not mrz["present"] or not mrz["valid_format"]:
            _check(checks, "mrz", cat, "Passport MRZ check digits", WARN, "MRZ not readable")
        elif mrz["errors"]:
            _check(checks, "mrz", cat, "Passport MRZ check digits", WARN, "; ".join(mrz["errors"]))
        else:
            _check(checks, "mrz", cat, "Passport MRZ check digits", PASS, "All check digits valid")
            mrz_name = f"{mrz.get('given_names', '')} {mrz.get('surname', '')}"
            score = matching.name_similarity(mrz_name, idd.get("full_name", ""))
            if score < get_settings().name_match_threshold:
                _check(checks, "mrz_name", cat, "MRZ name matches printed name", WARN,
                       f"MRZ '{mrz_name.strip()}' vs printed '{idd.get('full_name', '')}' ({score}%)")
            if not idd.get("expiry_date") and mrz.get("expiry_date"):
                idd["expiry_date"] = mrz["expiry_date"]

    expiry = _parse_date(idd.get("expiry_date"))
    if expiry:
        _check(checks, "id_expiry", cat, "Document not expired", PASS if expiry >= today else FAIL,
               f"Expires {expiry.isoformat()}")
    elif not expected_nic:
        _check(checks, "id_expiry", cat, "Document not expired", WARN, "Expiry date not readable")

    _authenticity_check(checks, "id_authenticity", cat, "No signs of tampering on ID", idd)

    issues = [i for q in quality.values() if isinstance(q, dict) for i in q.get("issues", [])]
    issues += idd.get("quality_issues") or []
    if issues:
        _check(checks, "upload_quality", cat, "Upload quality", WARN, "; ".join(dict.fromkeys(issues)))


def _bill_checks(checks: List[Dict], bill: Dict, idd: Dict, today: date) -> None:
    cat = "address"
    settings = get_settings()
    if not bill.get("is_utility_bill"):
        _check(checks, "bill_valid", cat, "Utility bill recognised", FAIL,
               bill.get("notes") or "The file does not look like a utility bill")
        return
    _check(checks, "bill_valid", cat, "Utility bill recognised", PASS,
           f"{bill.get('issuer') or 'Unknown issuer'} ({bill.get('bill_type')})")

    bill_date = _parse_date(bill.get("bill_date"))
    max_age = settings.bill_max_age_days
    if not bill_date:
        _check(checks, "bill_recent", cat, f"Bill issued in the last {max_age} days", WARN, "Bill date not readable")
    elif bill_date > today + timedelta(days=1):
        _check(checks, "bill_recent", cat, f"Bill issued in the last {max_age} days", WARN,
               f"Bill date {bill_date} is in the future")
    else:
        age = (today - bill_date).days
        status = PASS if age <= max_age else (WARN if age <= 2 * max_age else FAIL)
        _check(checks, "bill_recent", cat, f"Bill issued in the last {max_age} days", status,
               f"Dated {bill_date} ({age} days ago)")

    _authenticity_check(checks, "bill_authenticity", cat, "No signs of tampering on bill", bill)

    holder, id_name = bill.get("account_holder_name", ""), idd.get("full_name", "")
    if holder and id_name:
        score = matching.name_similarity(holder, id_name)
        ok = score >= settings.name_match_threshold
        _check(checks, "bill_name", cat, "Bill is in the ID holder's name", PASS if ok else WARN,
               f"'{holder}' vs '{id_name}' ({score}%)" + ("" if ok else " — bill may be in a family member's name"))
    else:
        _check(checks, "bill_name", cat, "Bill is in the ID holder's name", WARN, "Name missing on bill or ID")


def _address_checks(checks: List[Dict], typed: str, bill: Dict, idd: Dict, llm: Optional[Dict]) -> None:
    cat = "address"
    settings = get_settings()
    bill_addr = bill.get("service_address") or bill.get("billing_address") or ""
    if not bill_addr:
        _check(checks, "address_match", cat, "Typed address matches the bill", WARN, "Address not readable on bill")
        return
    score = matching.address_similarity(typed, bill_addr)
    numbers_conflict = matching.house_numbers_conflict(typed, bill_addr)
    llm_same = llm.get("same_address") if llm else None
    llm_conf = float(llm.get("confidence") or 0) if llm else 0.0
    detail = f"Similarity {score}%"
    if llm:
        detail += f"; AI: {'same' if llm_same else 'different'} ({llm_conf:.0%})"
        if llm.get("differences"):
            detail += " — " + "; ".join(llm["differences"])
    if numbers_conflict:
        detail += "; house/unit numbers differ"

    if (score >= settings.address_match_threshold and not numbers_conflict and llm_same is not False) or \
            (llm_same is True and llm_conf >= 0.7 and not numbers_conflict):
        status = PASS
    elif (score < 50 or numbers_conflict) and llm_same is False and llm_conf >= 0.8:
        status = FAIL
    else:
        status = WARN
    _check(checks, "address_match", cat, "Typed address matches the bill", status, detail)

    id_addr = idd.get("address") or ""
    if id_addr:
        id_score = matching.address_similarity(id_addr, bill_addr)
        _check(checks, "address_id", cat, "ID address matches the bill",
               PASS if id_score >= settings.address_match_threshold else WARN,
               f"Similarity {id_score}%" + ("" if id_score >= settings.address_match_threshold
                                            else " — the ID may show an older address"))


def _face_checks(checks: List[Dict], face: Dict, review: Optional[Dict]) -> None:
    poses = face.get("pose_checks") or {}
    failed = [s for s, ok in poses.items() if not ok]
    _check(checks, "liveness_pose", "liveness", "Head turned as instructed",
           PASS if poses and not failed else FAIL,
           "All movements detected" if poses and not failed else f"Not detected: {', '.join(failed) or 'all'}")
    _check(checks, "liveness_same_person", "liveness", "Same person in every frame",
           PASS if face.get("same_person") else FAIL,
           ", ".join(f"{k} {v:.2f}" for k, v in (face.get("same_person_scores") or {}).items()) or "No faces")
    if face.get("static_suspect"):
        _check(checks, "liveness_static", "liveness", "Frames are live, not a still image", FAIL,
               "All frames are identical")
    if review:
        signs = review.get("spoof_signs") or []
        ok = review.get("real_live_capture") and not signs
        _check(checks, "liveness_ai", "liveness", "AI review: live capture, no spoof signs", PASS if ok else WARN,
               "; ".join(signs + (review.get("pose_issues") or [])) or review.get("notes") or "Looks like a live capture")

    match = face.get("face_match") or {}
    verdict, score = match.get("verdict"), match.get("score")
    label = "Face matches the ID photo"
    if verdict == "match":
        _check(checks, "face_match", "face", label, PASS, f"Similarity {score:.2f}")
    elif verdict == "uncertain":
        _check(checks, "face_match", "face", label, WARN, f"Similarity {score:.2f} is borderline")
    elif verdict == "mismatch":
        _check(checks, "face_match", "face", label, FAIL, f"Similarity {score:.2f} is too low")
    else:
        _check(checks, "face_match", "face", label, WARN, "No face found on the ID photo")


def build_checks(
    *,
    id_type: str,
    typed_address: str,
    document_result: Dict,
    face_result: Dict,
    quality: Dict,
    address_llm: Optional[Dict] = None,
    duplicate_of_other_user: bool = False,
    today: Optional[date] = None,
) -> List[Dict]:
    today = today or date.today()
    checks: List[Dict] = []
    idd = document_result.get("identity") or {}
    bill = document_result.get("bill") or {}

    if document_result.get("identity_error"):
        _check(checks, "id_valid", "document", "ID document recognised", WARN,
               f"Automatic reading failed: {document_result['identity_error']}")
    else:
        _id_checks(checks, id_type, idd, quality, today)

    if document_result.get("bill_error"):
        _check(checks, "bill_valid", "address", "Utility bill recognised", WARN,
               f"Automatic reading failed: {document_result['bill_error']}")
    else:
        _bill_checks(checks, bill, idd, today)
        _address_checks(checks, typed_address, bill, idd, address_llm)

    if duplicate_of_other_user:
        _check(checks, "duplicate", "document", "Documents not used by another account", WARN,
               "The same ID image was uploaded by another account")

    _face_checks(checks, face_result, face_result.get("ai_review"))
    return checks


# --- score + decision ------------------------------------------------------------------
# How much each check counts toward the verification score.
WEIGHTS: Dict[str, int] = {
    "face_match": 20,
    "liveness_pose": 10,
    "liveness_same_person": 10,
    "liveness_static": 10,
    "liveness_ai": 5,
    "id_valid": 10,
    "id_number": 8,
    "mrz": 8,
    "nic_consistency": 6,
    "id_expiry": 6,
    "id_authenticity": 6,
    "mrz_name": 4,
    "id_type": 3,
    "upload_quality": 3,
    "address_match": 10,
    "bill_valid": 8,
    "bill_recent": 6,
    "bill_authenticity": 5,
    "bill_name": 4,
    "address_id": 3,
    "duplicate": 5,
}
_POINTS = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
# A fail here always needs a person, whatever the score.
HARD_FAILS = {"id_valid", "id_number", "id_expiry", "bill_valid", "bill_recent", "liveness_static"}


def score(checks: List[Dict]) -> int:
    """0-100: weighted share of checks passed (review counts half)."""
    total = sum(WEIGHTS.get(c["id"], 5) for c in checks)
    if not total:
        return 0
    got = sum(WEIGHTS.get(c["id"], 5) * _POINTS.get(c["status"], 0) for c in checks)
    return int(round(100 * got / total))


def evaluate(checks: List[Dict]) -> Dict:
    """Score the checks and decide: approved (passed) or manual_review.

    Manual review when the score is below KYC_PASS_SCORE, when a required check
    (KYC_REQUIRED_CHECKS) did not pass, or when a hard check failed.
    """
    settings = get_settings()
    by_id = {c["id"]: c for c in checks}
    blockers: List[str] = []
    for cid in settings.kyc_required_checks:
        check = by_id.get(cid)
        if check is None or check["status"] != PASS:
            blockers.append(check["label"] if check else f"{cid} (not run)")
    for c in checks:
        if c["status"] == FAIL and c["id"] in HARD_FAILS and c["label"] not in blockers:
            blockers.append(c["label"])
    pct = score(checks)
    below = pct < settings.kyc_pass_score
    return {
        "decision": "manual_review" if blockers or below else "approved",
        "score": pct,
        "pass_score": settings.kyc_pass_score,
        "blockers": blockers,
        "below_pass_score": below,
    }


def decide(checks: List[Dict]) -> str:
    return evaluate(checks)["decision"]
