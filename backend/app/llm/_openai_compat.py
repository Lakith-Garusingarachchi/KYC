"""
Shared helpers for OpenAI-compatible adapters (DeepSeek, Qwen, …).

These utilities are intentionally kept here — rather than imported from
``openai_adapter`` — so the third-party OpenAI-compatible adapters stay decoupled
from churn in the first-party OpenAI adapter (retry/backoff/redaction details
have varied across branches). Everything here is provider-agnostic.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

# HTTP status codes worth retrying. Everything else in the 4xx range is a
# deterministic client error (bad schema, auth, validation) and must NOT retry.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def preview_text(value: Any, limit: int = 500) -> str:
    """Truncate a value for safe logging (bounds PHI/base64 leakage at DEBUG)."""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… <+{len(text) - limit} chars redacted>"


def _redact_responses_input(items: List[Any]) -> List[Any]:
    """Redact the OpenAI Responses payload shape (input[].content[]).

    Same guarantees as the chat/messages branch — data URLs and inline file bytes
    become size markers, free text is truncated — for the part types the Responses
    API uses (input_text / input_image / input_file).
    """
    redacted: List[Any] = []
    for message in items:
        if not isinstance(message, dict):
            redacted.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            redacted.append(message)
            continue
        parts: List[Any] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(part)
                continue
            kind = part.get("type")
            if kind == "input_text":
                parts.append({**part, "text": preview_text(part.get("text", ""))})
            elif kind == "input_image":
                url = part.get("image_url", "")
                parts.append({**part, "image_url": (
                    f"<data-url {len(url)} chars redacted>"
                    if isinstance(url, str) and url.startswith("data:")
                    else preview_text(url, 120)
                )})
            elif kind == "input_file":
                # file_id / file_url / filename / detail are safe and useful to see;
                # inline base64 bytes are not.
                safe_part = {k: v for k, v in part.items() if k != "file_data"}
                if "file_data" in part:
                    data = part.get("file_data") or ""
                    safe_part["file_data"] = f"<file-data {len(data)} chars redacted>"
                parts.append(safe_part)
            else:
                parts.append(part)
        redacted.append({**message, "content": parts})
    return redacted


def redact_payload_for_log(payload: Any) -> Any:
    """
    Build a log-safe view of a chat or Responses payload.

    Image data URLs are replaced with size markers and free-text content is
    truncated, so DEBUG logs stay useful without spilling base64 image bytes or
    full medical document text (PHI).
    """
    if not isinstance(payload, dict):
        return preview_text(payload, 800)

    safe = {k: v for k, v in payload.items() if k not in ("messages", "input")}

    # Responses API shape: input[] instead of messages[]. Handled here so the
    # document path can log its payload with the same redaction guarantees.
    responses_input = payload.get("input")
    if isinstance(responses_input, list):
        safe["input"] = _redact_responses_input(responses_input)

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return safe

    redacted_messages: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            redacted_messages.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            redacted_messages.append({**message, "content": preview_text(content)})
        elif isinstance(content, list):
            parts: List[Any] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(part)
                    continue
                if part.get("type") == "image_url":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url", "")
                    marker = (
                        f"<data-url {len(url)} chars redacted>"
                        if isinstance(url, str) and url.startswith("data:")
                        else preview_text(url, 120)
                    )
                    # Keep the sibling keys (notably `detail`) — only the url is
                    # unsafe to log. Rebuilding the part from scratch used to drop
                    # them, which made the dump look like detail was never sent.
                    parts.append({**part, "image_url": {**image_url, "url": marker}})
                elif part.get("type") == "text":
                    parts.append({"type": "text", "text": preview_text(part.get("text", ""))})
                else:
                    parts.append(part)
            redacted_messages.append({**message, "content": parts})
        else:
            redacted_messages.append(message)

    safe["messages"] = redacted_messages
    return safe


def compute_backoff_seconds(attempt: int, retry_after: Optional[str] = None) -> float:
    """Exponential backoff with full jitter; honors a Retry-After header when present."""
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except (TypeError, ValueError):
            pass
    return min(0.5 * (2 ** attempt) + random.uniform(0, 0.5), 20.0)


def prepare_strict_schema(response_schema: Any) -> Dict[str, Any]:
    """
    Normalize a schema into a JSON Schema object with additionalProperties=false.

    Accepts either a legacy list-of-fields ([{name,type,required}, ...]) or a
    JSON-Schema dict.
    """
    schema = response_schema
    if isinstance(schema, list):
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for field in schema:
            if not isinstance(field, dict):
                continue
            fname = field.get("name")
            if not fname:
                continue
            properties[fname] = {"type": field.get("type", "string")}
            if field.get("required"):
                required.append(fname)
        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
    if isinstance(schema, dict):
        schema.setdefault("additionalProperties", False)
    return schema


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first complete, balanced JSON object from mixed text.

    Uses the JSON decoder to find the first syntactically valid object instead of
    a greedy ``\\{.*\\}`` regex (which can span multiple objects).
    """
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index != -1:
        try:
            obj, _ = decoder.raw_decode(text[index:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        index = text.find("{", index + 1)
    return None
