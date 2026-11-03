"""LLM prompts and JSON schemas: read the ID and the utility bill, review the
liveness frames, and compare addresses.

Schemas stay flat (strings, booleans, numbers, string arrays) so the same schema
works with every adapter's structured-output mode (OpenAI strict, Gemini
responseSchema, Claude output_config).
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from .llm_service import Attachment, ask_json

SYSTEM_PROMPT = (
    "You are a careful KYC document examiner. You read identity documents and utility bills "
    "exactly as printed, never guess missing values, and report anything that looks edited, "
    "pasted, re-typed or inconsistent. Use an empty string when a field is not visible."
)

_ASSESSMENT = {"type": "string", "enum": ["looks_genuine", "suspicious", "cannot_tell"]}
_STR_LIST = {"type": "array", "items": {"type": "string"}}
# none: nothing found; minor: wear, glare, print quirks, normal handwriting;
# major: signs that a name, number, photo, address or date was changed.
_SEVERITY = {"type": "string", "enum": ["none", "minor", "major"]}

ID_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "is_identity_document": {"type": "boolean"},
        "document_type": {"type": "string", "enum": ["nic_old", "nic_new", "passport", "other"]},
        "issuing_country": {"type": "string"},
        "full_name": {"type": "string"},
        "id_number": {"type": "string"},
        "date_of_birth": {"type": "string", "description": "YYYY-MM-DD or empty"},
        "gender": {"type": "string", "enum": ["M", "F", "unknown"]},
        "address": {"type": "string"},
        "expiry_date": {"type": "string", "description": "YYYY-MM-DD or empty"},
        "nationality": {"type": "string"},
        "mrz_line1": {"type": "string"},
        "mrz_line2": {"type": "string"},
        "has_photo": {"type": "boolean"},
        "photo_box": {
            "type": "array", "items": {"type": "number"},
            "description": "Main portrait on image 1 as [left, top, right, bottom], fractions 0-1 of the "
                           "image width/height; empty if there is no portrait",
        },
        "text_legible": {"type": "boolean"},
        "tamper_signs": _STR_LIST,
        "tamper_severity": _SEVERITY,
        "quality_issues": _STR_LIST,
        "authenticity_assessment": _ASSESSMENT,
        "notes": {"type": "string"},
    },
    "required": [
        "is_identity_document", "document_type", "issuing_country", "full_name", "id_number",
        "date_of_birth", "gender", "address", "expiry_date", "nationality", "mrz_line1", "mrz_line2",
        "has_photo", "photo_box", "text_legible", "tamper_signs", "tamper_severity", "quality_issues",
        "authenticity_assessment", "notes",
    ],
}

BILL_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "is_utility_bill": {"type": "boolean"},
        "bill_type": {"type": "string", "enum": ["electricity", "water", "telecom", "internet", "gas", "other"]},
        "issuer": {"type": "string"},
        "account_holder_name": {"type": "string"},
        "service_address": {"type": "string"},
        "billing_address": {"type": "string"},
        "bill_date": {"type": "string", "description": "YYYY-MM-DD issue/statement date, or empty"},
        "account_number": {"type": "string"},
        "tamper_signs": _STR_LIST,
        "tamper_severity": _SEVERITY,
        "authenticity_assessment": _ASSESSMENT,
        "notes": {"type": "string"},
    },
    "required": [
        "is_utility_bill", "bill_type", "issuer", "account_holder_name", "service_address",
        "billing_address", "bill_date", "account_number", "tamper_signs", "tamper_severity",
        "authenticity_assessment", "notes",
    ],
}

LIVENESS_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "real_live_capture": {"type": "boolean"},
        "spoof_signs": _STR_LIST,
        "pose_matches_labels": {"type": "boolean"},
        "pose_issues": _STR_LIST,
        "image_quality_ok": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["real_live_capture", "spoof_signs", "pose_matches_labels", "pose_issues",
                 "image_quality_ok", "notes"],
}

ADDRESS_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "same_address": {"type": "boolean"},
        "confidence": {"type": "number"},
        "differences": _STR_LIST,
        "explanation": {"type": "string"},
    },
    "required": ["same_address", "confidence", "differences", "explanation"],
}


async def extract_identity(id_type: str, front: Attachment, back: Optional[Attachment]) -> Dict:
    doc = "national identity card (NIC)" if id_type == "nic" else "passport (photo/data page)"
    images = "Image 1 is the front." + (" Image 2 is the back." if back else "")
    prompt = (
        f"The customer says this is their {doc}. {images}\n"
        "Extract the printed fields. For a passport also copy both machine-readable zone (MRZ) lines "
        "exactly, character for character, using '<' for filler. Dates must be YYYY-MM-DD. "
        "Give the position of the main portrait photo on image 1 as photo_box.\n"
        "Then examine the document like a fraud analyst: is it a real identity document (not a "
        "screenshot, printout of a template, sample/specimen, or a photo of a screen)? List concrete "
        "tamper signs (mismatched fonts, misaligned text, a pasted or replaced photo, edited numbers, "
        "missing security features) and quality problems (blur, glare, cropping).\n"
        "Set tamper_severity: 'major' only for signs that the name, number, dates or photo were changed "
        "or that the document is fake; 'minor' for wear, laminate glare, printing or spacing quirks, "
        "different backgrounds for front and back, or anything you cannot confirm from a photo; "
        "'none' when there is nothing to note."
    )
    attachments = [front] + ([back] if back else [])
    return await ask_json(prompt, ID_SCHEMA, "identity_document", attachments, system_prompt=SYSTEM_PROMPT)


async def extract_bill(bill: Attachment) -> Dict:
    prompt = (
        f"This file is a utility bill submitted as proof of address (today is {date.today().isoformat()}).\n"
        "Extract the issuer, the account holder name, the service (supply) address, the billing "
        "address if different, the bill/statement date as YYYY-MM-DD and the account number. "
        "Copy addresses exactly as printed on one line. Then judge whether it is a real utility, "
        "telecom or internet bill and list any signs of editing (inconsistent fonts or alignment, "
        "overwritten names, addresses or dates).\n"
        "In Sri Lanka, water (NWSDB) and electricity (CEB/LECO) bills are often printed or filled in "
        "by hand on site by the meter reader, so handwriting, carbon copies, reading dates after the "
        "printed date and totals that differ because of payments or arrears are normal.\n"
        "Set tamper_severity: 'major' only when the name, address or bill date looks altered or the "
        "bill looks fabricated; 'minor' for the normal quirks above or anything you cannot confirm; "
        "'none' when there is nothing to note."
    )
    return await ask_json(prompt, BILL_SCHEMA, "utility_bill", [bill], system_prompt=SYSTEM_PROMPT)


async def review_liveness(frames: List[Attachment], steps: List[str]) -> Dict:
    labels = "; ".join(f"Image {i + 1}: head {'facing the camera' if s == 'center' else 'turned ' + s}"
                       for i, s in enumerate(steps))
    prompt = (
        "These images were captured from a phone or laptop camera during a guided liveness check. "
        f"The person was asked to move their head in this order: {labels}. "
        "'left' and 'right' are the person's own left and right; the images are not mirrored.\n"
        "Do not identify the person. Only judge the capture: does each image look like a live camera "
        "capture of a real person (not a printed photo, a phone/monitor screen, a mask or a video "
        "replay — look for screen moire, bezels, paper edges, reflections, flat lighting)? Does the head "
        "pose in each image roughly match its label? Is image quality good enough?"
    )
    return await ask_json(prompt, LIVENESS_SCHEMA, "liveness_review", frames,
                          system_prompt="You review camera captures for presentation-attack (spoof) signs.")


async def compare_addresses(typed: str, bill_address: str, id_address: str, provider: str) -> Dict:
    prompt = (
        "Decide whether these describe the same residential address. Ignore formatting, abbreviations "
        "(Rd/Road, Mw/Mawatha), letter case, the order of lines, and missing postal codes. Treat a "
        "different house/apartment number, street or city as different.\n"
        f"A (typed by customer): {typed or '-'}\n"
        f"B (utility bill): {bill_address or '-'}\n"
        f"C (identity document, may be empty or older): {id_address or '-'}\n"
        "Answer for A vs B. Mention C only if it conflicts with both."
    )
    return await ask_json(prompt, ADDRESS_SCHEMA, "address_comparison", provider=provider)
