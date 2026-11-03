"""End-to-end API flow with a fake LLM adapter and a stubbed face engine.

The fake adapter is registered under the "gemini" provider string, so the real
llm_service -> AdapterFactory -> adapter path runs; only the network call is fake.
"""

from datetime import date, timedelta

import cv2
import numpy as np
import pytest

from app.llm import AdapterFactory, Capabilities, LLMAdapter, LLMResponse
from app.services import face as face_service

TODAY = date.today()


class FakeAdapter(LLMAdapter):
    provider_name = "Fake"
    capabilities = Capabilities(text=True, image=True, document=True, structured_output=True)
    calls = []
    overrides = {}

    def __init__(self, config):
        self.config = config

    async def complete(self, request):
        FakeAdapter.calls.append((request.response_schema_name, len(request.media)))
        data = {
            "identity_document": {
                "is_identity_document": True, "document_type": "nic_new", "issuing_country": "LKA",
                "full_name": "Ashan Bandara Perera", "id_number": "199534001234",
                "date_of_birth": "1995-12-05", "gender": "M", "address": "12/3 Galle Road, Colombo 3",
                "expiry_date": "", "nationality": "Sri Lankan", "mrz_line1": "", "mrz_line2": "",
                "has_photo": True, "text_legible": True, "tamper_signs": [], "quality_issues": [],
                "authenticity_assessment": "looks_genuine", "notes": "",
            },
            "utility_bill": {
                "is_utility_bill": True, "bill_type": "electricity", "issuer": "CEB",
                "account_holder_name": "A B Perera", "service_address": "No. 12/3, Galle Rd, Colombo 03",
                "billing_address": "", "bill_date": (TODAY - timedelta(days=20)).isoformat(),
                "account_number": "1234567890", "tamper_signs": [],
                "authenticity_assessment": "looks_genuine", "notes": "",
            },
            "liveness_review": {
                "real_live_capture": True, "spoof_signs": [], "pose_matches_labels": True,
                "pose_issues": [], "image_quality_ok": True, "notes": "",
            },
            "address_comparison": {"same_address": True, "confidence": 0.95, "differences": [], "explanation": ""},
        }[request.response_schema_name]
        data = {**data, **FakeAdapter.overrides.get(request.response_schema_name, {})}
        return LLMResponse(text="", data=data, provider="Fake", model_used="fake-1")

    async def complete_multimodal(self, request):
        return await self.complete(request)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    monkeypatch.setitem(AdapterFactory._ADAPTER_REGISTRY, "gemini", FakeAdapter)
    FakeAdapter.calls = []
    FakeAdapter.overrides = {}
    yield FakeAdapter


@pytest.fixture
def face_passes(monkeypatch):
    def fake_analyze(frames, steps, id_image):
        return {"steps": steps, "frames": {}, "pose_checks": {s: True for s in steps if s != "center"},
                "same_person_scores": {s: 0.8 for s in steps if s != "center"}, "same_person": True,
                "static_suspect": False, "liveness_pass": True,
                "face_match": {"score": 0.62, "verdict": "match"}, "issues": []}
    monkeypatch.setattr(face_service, "analyze_liveness", fake_analyze)


def _jpeg(w=1000, h=640) -> bytes:
    rng = np.random.default_rng(1)
    img = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    cv2.putText(img, "TEST DOCUMENT", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
    return cv2.imencode(".jpg", img)[1].tobytes()


async def _register(client, email="ashan@example.com"):
    r = await client.post("/auth/register", json={"email": email, "password": "correct-horse-1",
                                                   "full_name": "Ashan Perera"})
    assert r.status_code == 201, r.text
    return r


async def _upload_docs(client, sid):
    files = {
        "id_front": ("front.jpg", _jpeg(), "image/jpeg"),
        "utility_bill": ("bill.pdf", b"%PDF-1.4\n% fake bill\n", "application/pdf"),
    }
    data = {"id_type": "nic", "address": "12/3 Galle Road, Colombo 3"}
    return await client.post(f"/kyc/submissions/{sid}/documents", data=data, files=files)


async def _do_liveness(client, sid):
    ch = (await client.post(f"/kyc/submissions/{sid}/liveness/challenge")).json()
    assert ch["steps"][0] == "center" and sorted(ch["steps"][1:]) == ["left", "right", "up"]
    files = {f"frame_{s}": (f"{s}.jpg", _jpeg(640, 480), "image/jpeg") for s in ch["steps"]}
    return await client.post(f"/kyc/submissions/{sid}/liveness", data={"challenge_id": ch["challenge_id"]},
                             files=files)


async def test_auth_password_flow(client):
    await _register(client, "flow@example.com")
    assert (await client.get("/auth/me")).json()["email"] == "flow@example.com"  # cookie session
    client.cookies.clear()
    assert (await client.get("/auth/me")).status_code == 401

    bad = await client.post("/auth/token", data={"username": "flow@example.com", "password": "nope-nope"})
    assert bad.status_code == 401
    ok = await client.post("/auth/token", data={"username": "FLOW@example.com", "password": "correct-horse-1"})
    assert ok.status_code == 200
    token = ok.json()["access_token"]
    client.cookies.clear()
    me = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert (await client.post("/auth/register", json={"email": "flow@example.com", "password": "x" * 9})).status_code == 409
    assert (await client.get("/auth/providers")).json() == {"password": True, "google": False}


async def test_full_kyc_flow_approved(client, face_passes):
    await _register(client)
    sub = (await client.post("/kyc/submissions")).json()
    r = await _upload_docs(client, sub["id"])
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "documents_uploaded"

    after_docs = (await client.get(f"/kyc/submissions/{sub['id']}")).json()
    assert after_docs["extracted"]["identity"]["full_name"] == "Ashan Bandara Perera"
    assert after_docs["extracted"]["bill"]["issuer"] == "CEB"

    live = await _do_liveness(client, sub["id"])
    assert live.status_code == 200, live.text
    assert live.json()["accepted"] is True

    final = (await client.get(f"/kyc/submissions/{sub['id']}")).json()
    assert final["status"] == "completed", final
    by_id = {c["id"]: c for c in final["checks"]}
    assert by_id["id_number"]["status"] == "pass"
    assert by_id["nic_consistency"]["status"] == "pass"
    assert by_id["address_match"]["status"] == "pass"
    assert by_id["bill_name"]["status"] == "pass"
    assert by_id["face_match"]["status"] == "pass"
    assert final["decision"] == "approved", [c for c in final["checks"] if c["status"] != "pass"]
    names = [c[0] for c in FakeAdapter.calls]
    assert sorted(names) == ["address_comparison", "identity_document", "liveness_review", "utility_bill"]


async def test_old_bill_and_wrong_address_are_flagged(client, face_passes, fake_llm):
    fake_llm.overrides = {
        "utility_bill": {"bill_date": (TODAY - timedelta(days=400)).isoformat(),
                         "service_address": "45 Kandy Road, Kurunegala"},
        "address_comparison": {"same_address": False, "confidence": 0.95, "differences": ["street", "city"]},
    }
    await _register(client, "old@example.com")
    sid = (await client.post("/kyc/submissions")).json()["id"]
    assert (await _upload_docs(client, sid)).status_code == 200
    assert (await _do_liveness(client, sid)).status_code == 200
    final = (await client.get(f"/kyc/submissions/{sid}")).json()
    by_id = {c["id"]: c for c in final["checks"]}
    assert by_id["bill_recent"]["status"] == "fail"
    assert by_id["address_match"]["status"] == "fail"
    assert final["decision"] == "rejected"


async def test_failed_liveness_allows_retry(client, monkeypatch):
    def failing(frames, steps, id_image):
        return {"steps": steps, "pose_checks": {s: False for s in steps if s != "center"}, "same_person": True,
                "same_person_scores": {}, "static_suspect": True, "liveness_pass": False,
                "face_match": {"score": 0.7, "verdict": "match"}, "issues": ["Head did not turn left as asked"]}
    monkeypatch.setattr(face_service, "analyze_liveness", failing)
    await _register(client, "retry@example.com")
    sid = (await client.post("/kyc/submissions")).json()["id"]
    await _upload_docs(client, sid)

    first = (await _do_liveness(client, sid)).json()
    assert first["accepted"] is False and first["attempts_left"] == 2
    assert (await client.get(f"/kyc/submissions/{sid}")).json()["status"] == "documents_uploaded"
    await _do_liveness(client, sid)
    third = (await _do_liveness(client, sid)).json()
    assert third["accepted"] is True and third["attempts_left"] == 0  # out of retries -> decided
    final = (await client.get(f"/kyc/submissions/{sid}")).json()
    assert final["decision"] == "rejected"


async def test_challenge_cannot_be_reused_and_uploads_are_validated(client, face_passes):
    await _register(client, "val@example.com")
    sid = (await client.post("/kyc/submissions")).json()["id"]
    bad = await client.post(f"/kyc/submissions/{sid}/documents",
                            data={"id_type": "nic", "address": "12 Galle Road"},
                            files={"id_front": ("x.txt", b"hello", "text/plain"),
                                   "utility_bill": ("b.pdf", b"%PDF-1.4", "application/pdf")})
    assert bad.status_code == 415
    # Liveness before documents is refused.
    assert (await client.post(f"/kyc/submissions/{sid}/liveness/challenge")).status_code == 409

    await _upload_docs(client, sid)
    ch = (await client.post(f"/kyc/submissions/{sid}/liveness/challenge")).json()
    files = {f"frame_{s}": (f"{s}.jpg", _jpeg(640, 480), "image/jpeg") for s in ch["steps"]}
    ok = await client.post(f"/kyc/submissions/{sid}/liveness", data={"challenge_id": ch["challenge_id"]}, files=files)
    assert ok.status_code == 200
    # Status moved on, and the challenge is gone either way.
    again = await client.post(f"/kyc/submissions/{sid}/liveness", data={"challenge_id": ch["challenge_id"]}, files=files)
    assert again.status_code in (400, 409)


async def test_other_users_cannot_read_submission(client):
    await _register(client, "owner@example.com")
    sid = (await client.post("/kyc/submissions")).json()["id"]
    client.cookies.clear()
    await _register(client, "intruder@example.com")
    assert (await client.get(f"/kyc/submissions/{sid}")).status_code == 404


async def test_text_only_provider_is_rejected_for_documents(client, face_passes, monkeypatch):
    from app.services import llm_service

    class TextOnly(FakeAdapter):
        capabilities = Capabilities(text=True, structured_output=True)

    monkeypatch.setitem(AdapterFactory._ADAPTER_REGISTRY, "gemini", TextOnly)
    await _register(client, "text@example.com")
    sid = (await client.post("/kyc/submissions")).json()["id"]
    await _upload_docs(client, sid)
    sub = (await client.get(f"/kyc/submissions/{sid}")).json()
    assert "cannot read" in sub["extracted"]["identity_error"]
    assert llm_service.normalize_provider("Vertex") == "gemini_vertex"
