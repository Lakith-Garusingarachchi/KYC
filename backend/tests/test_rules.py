from datetime import date

from app.services import id_rules, matching
from app.services.face import pose_from_landmarks, pose_step_passed


# --- Sri Lankan NIC ---------------------------------------------------------------
def test_old_nic_male():
    info = id_rules.parse_nic("853400937V", today=date(2026, 1, 1))
    assert info.valid and info.format == "old"
    assert info.birth_date == date(1985, 12, 5)  # day 340 with a 29-day February
    assert info.gender == "M"


def test_new_nic_female():
    info = id_rules.parse_nic("199578201234", today=date(2026, 1, 1))
    assert info.valid and info.format == "new"
    assert info.gender == "F"
    assert info.birth_date == date(1995, 10, 8)  # 782 - 500 = day 282


def test_nic_day_60_is_feb_29_only_in_leap_years():
    assert id_rules.parse_nic("200006012345").birth_date == date(2000, 2, 29)
    assert not id_rules.parse_nic("199906012345").valid  # 1999 has no Feb 29
    assert id_rules.parse_nic("199906112345").birth_date == date(1999, 3, 1)


def test_nic_rejects_bad_formats():
    for bad in ("12345", "85340093AV", "853400937", "199588801234", "203000112345"):
        assert not id_rules.parse_nic(bad, today=date(2026, 1, 1)).valid, bad


# --- Passport MRZ (ICAO 9303 specimen) -----------------------------------------------
L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_mrz_check_digits_specimen():
    assert id_rules.mrz_check_digit("L898902C3") == 6
    result = id_rules.parse_td3(L1, L2, today=date(2026, 1, 1))
    assert result["valid_format"] and not result["errors"]
    assert result["surname"] == "ERIKSSON" and result["given_names"] == "ANNA MARIA"
    assert result["birth_date"] == "1974-08-12" and result["expiry_date"] == "2012-04-15"


def test_mrz_detects_edited_number():
    edited = "L898902C46" + L2[10:]
    result = id_rules.parse_td3(L1, edited)
    assert result["checks"]["document_number"] is False
    assert result["errors"]


# --- matching --------------------------------------------------------------------
def test_name_matching_handles_initials_and_order():
    assert matching.name_similarity("Mr. A B Perera", "Ashan Bandara Perera") >= 85
    assert matching.name_similarity("PERERA ASHAN BANDARA", "Ashan Bandara Perera") == 100
    assert matching.name_similarity("Nimal Silva", "Ashan Perera") < 50


def test_address_matching_normalizes_abbreviations():
    a = "No. 12/3, Galle Rd, Colombo 03"
    b = "12/3 GALLE ROAD COLOMBO 3"
    assert matching.address_similarity(a, b) >= 90
    assert not matching.house_numbers_conflict(a, b)
    assert matching.house_numbers_conflict("45 Galle Road Colombo 3", "12 Galle Road Colombo 3")


# --- head pose ---------------------------------------------------------------------
def _pose(nose_x, nose_y=124.0):
    return pose_from_landmarks((200, 102), (247, 102), (nose_x, nose_y), (199, 143), (240, 143))


def test_pose_turns_relative_to_center():
    base_yaw, base_pitch = _pose(223.5)
    left_yaw, left_pitch = _pose(236)    # nose moves to image-right = the person's left
    right_yaw, right_pitch = _pose(211)
    up_yaw, up_pitch = _pose(223.5, nose_y=118)
    assert pose_step_passed("left", left_yaw, left_pitch, base_yaw, base_pitch, 0.1, 0.05)
    assert not pose_step_passed("left", right_yaw, right_pitch, base_yaw, base_pitch, 0.1, 0.05)
    assert pose_step_passed("right", right_yaw, right_pitch, base_yaw, base_pitch, 0.1, 0.05)
    assert pose_step_passed("up", up_yaw, up_pitch, base_yaw, base_pitch, 0.1, 0.05)
    assert not pose_step_passed("up", base_yaw, base_pitch, base_yaw, base_pitch, 0.1, 0.05)


# --- score + decision ---------------------------------------------------------------
from app.services import decision  # noqa: E402


def _c(cid, status, label=None):
    return {"id": cid, "category": "x", "label": label or cid, "status": status, "detail": ""}


REQUIRED_PASS = [_c("face_match", "pass"), _c("liveness_pose", "pass"),
                 _c("liveness_same_person", "pass"), _c("address_match", "pass")]


def test_score_weights_review_as_half():
    assert decision.score(REQUIRED_PASS) == 100
    half = [dict(c, status="warn") for c in REQUIRED_PASS]
    assert decision.score(half) == 50
    assert decision.score([]) == 0


def test_pass_score_threshold(monkeypatch):
    from app.config import get_settings
    checks = REQUIRED_PASS + [_c("id_authenticity", "warn"), _c("bill_name", "warn"), _c("bill_authenticity", "warn"),
                              _c("id_valid", "warn"), _c("bill_valid", "warn"), _c("duplicate", "fail"),
                              _c("upload_quality", "fail"), _c("liveness_ai", "fail")]
    result = decision.evaluate(checks)
    assert result["score"] < 100 and result["blockers"] == []
    monkeypatch.setattr(get_settings(), "kyc_pass_score", result["score"])
    assert decision.evaluate(checks)["decision"] == "approved"      # score == pass score -> pass
    monkeypatch.setattr(get_settings(), "kyc_pass_score", result["score"] + 1)
    assert decision.evaluate(checks)["decision"] == "manual_review"


def test_required_checks_block_a_pass_only_when_configured(monkeypatch):
    from app.config import get_settings
    checks = [dict(c) for c in REQUIRED_PASS] + [_c("id_expiry", "fail", "Document not expired")]
    checks[0]["status"] = "warn"   # face_match not passed
    assert decision.evaluate(checks)["decision"] == "approved"           # score decides by default
    monkeypatch.setattr(get_settings(), "kyc_required_checks", ["face_match"])
    result = decision.evaluate(checks)
    assert result["decision"] == "manual_review" and result["blockers"] == ["face_match"]
    assert "Document not expired" in result["issues"]


def test_missed_nod_is_review_but_missed_turn_is_fail():
    face = {"pose_checks": {"left": True, "right": True, "up": False}, "same_person": True,
            "frames": {"center": {"yaw": 0.0, "pitch": 0.55}, "up": {"yaw": 0.0, "pitch": 0.53}},
            "thresholds": {"yaw_delta": 0.1, "pitch_delta": 0.03}, "face_match": {"verdict": "match", "score": 0.5}}
    checks = []
    decision._face_checks(checks, face, None)
    pose = checks[0]
    assert pose["status"] == "warn" and "up +0.02 (needs 0.03)" in pose["detail"]
    face["pose_checks"]["left"] = False
    checks = []
    decision._face_checks(checks, face, None)
    assert checks[0]["status"] == "fail"
