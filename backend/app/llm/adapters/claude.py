"""
Anthropic Claude adapter (text + image + document) on the official `anthropic` SDK.

Claude is NOT OpenAI-compatible (Messages API: top-level `system`, typed content
blocks, `output_config`), so like Gemini it implements the LLMAdapter interface
directly instead of subclassing the OpenAI-compatible base.

Attachments, mirroring the OpenAI adapter:
  * images      -> `image` blocks (base64, or a URL source for http(s) links);
                   formats Claude rejects and images over 5 MB are re-encoded
  * PDFs        -> native `document` blocks (Claude reads the text and the page images)
  * text files  -> plain-text `document` blocks titled with the filename
  * DOCX / XLSX / XLS -> Claude has no native Office input, so the text is
                   extracted locally and sent as a plain-text `document` block
Attachments go before the prompt, where Claude answers best.

Model features, resolved per model id by `claude_model_profile`:
  * adaptive thinking on the models that support it; the model config's
    `reasoning_effort` maps to `output_config.effort` (or to a thinking budget
    on older models that have no effort control)
  * temperature / top_p / top_k only where the model still accepts them and
    thinking is off
  * native structured output (`output_config.format`), with a prompt-guided
    retry when a model rejects the schema
  * server-side refusal fallback (`fallbacks: "default"`) where available;
    any other refusal raises `ClaudeRefusalError`

Every request is streamed and collected with `get_final_message()`, so a large
`max_tokens` never trips the SDK's non-streaming timeout guard.

`raw` is the Messages API response plus an OpenAI-style `choices` entry, so the
legacy callers that read `raw["choices"][0]["message"]["content"]`
(`FunctionExecutor._extract_direct_response`, Langfuse previews) work unchanged.
Token usage stays in the native `usage.input_tokens` / `output_tokens` shape,
which UsageExtractor and UsageStore already read.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import anthropic
import httpx

from ..config import ModelConfig
from ..contracts import (
    Capabilities,
    LLMRequest,
    LLMResponse,
    Media,
    MediaKind,
    legacy_media,
    media_from_legacy,
)
from ..interfaces import LLMAdapter
from ..media_normalizer import MediaNormalizer, ResolvedMedia
from ..usage_extractor import UsageExtractor
from .._openai_compat import extract_first_json_object, prepare_strict_schema, preview_text

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com"
_SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"minimal": _EFFORT_LEVELS[0]}  # OpenAI's lowest level, for configs shared across providers
# Older models (Sonnet 4.5, Haiku 4.5, Opus 4.1 …) take a thinking budget instead of an effort level.
_THINKING_BUDGETS = dict(zip(_EFFORT_LEVELS, (4096, 10000, 24000, 32000, 48000)))
_MIN_THINKING_BUDGET = 1024

_BLOCK_TEXT = "text"
_BLOCK_IMAGE = "image"
_BLOCK_DOCUMENT = "document"
_SOURCE_BASE64 = "base64"
_SOURCE_URL = "url"

_PDF_MIME = "application/pdf"
_PLAIN_TEXT_MIME = "text/plain"
_OCTET_STREAM_MIME = "application/octet-stream"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_XLS_MIME = "application/vnd.ms-excel"
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
_MAX_IMAGE_BASE64_CHARS = 5 * 1024 * 1024  # the API's per-image limit
_DOWNSCALE_EDGE = 2048                       # long edge for images over that limit

_TEXT_MIMES = {
    "application/json", "application/xml", "application/javascript",
    "application/csv", "application/x-yaml", "application/yaml",
    "application/sql", "application/x-sh",
}
_TEXT_EXTENSIONS = (
    '.txt', '.csv', '.tsv', '.json', '.xml', '.md', '.py', '.js', '.ts',
    '.html', '.htm', '.css', '.yaml', '.yml', '.sql', '.log', '.sh',
    '.env', '.ini', '.conf', '.toml', '.rtf', '.c', '.cpp', '.h',
    '.java', '.go', '.rs', '.bat', '.ps1'
)

# Claude stop_reason -> the OpenAI finish_reason legacy callers expect in `choices`.
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class ClaudeRefusalError(ValueError):
    """Claude declined the request (`stop_reason == "refusal"`) and no fallback rescued it."""


# ------------------------------------------------------------ model profile ---
@dataclass(frozen=True)
class ClaudeModelProfile:
    """What one Claude model accepts, derived from its model id."""

    recognized: bool = False
    adaptive_thinking: bool = False      # accepts adaptive thinking
    budget_thinking: bool = False        # older models: thinking with a token budget instead
    effort_levels: Tuple[str, ...] = ()  # accepted output_config.effort values
    sampling: bool = True                # temperature / top_p / top_k accepted
    server_fallback: bool = False        # fallbacks="default" accepted
    default_max_tokens: int = 8192
    context_window: int = 200_000


# claude-opus-4-6, claude-sonnet-4-5-20250929, claude-fable-5-1, anthropic.claude-opus-5-5 …
_MODEL_ID_RE = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:[-.](\d)(?!\d))?", re.IGNORECASE)
# Legacy naming: claude-3-5-sonnet-20241022, claude-3-7-sonnet-latest
_LEGACY_MODEL_ID_RE = re.compile(r"claude-(\d+)(?:[-.](\d)(?!\d))?-(opus|sonnet|haiku)", re.IGNORECASE)
_FRONTIER_FAMILIES = {"fable", "mythos"}
_SERVER_FALLBACK_MODELS = {("fable", (5, 1)), ("opus", (5, 0)), ("opus", (5, 5)), ("sonnet", (5, 5))}


def _parse_model_id(model_id: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    match = _MODEL_ID_RE.search(model_id or "")
    if match:
        return match.group(1).lower(), (int(match.group(2)), int(match.group(3) or 0))
    match = _LEGACY_MODEL_ID_RE.search(model_id or "")
    if match:
        return match.group(3).lower(), (int(match.group(1)), int(match.group(2) or 0))
    return None


def _effort_levels(family: str, version: Tuple[int, int], frontier: bool) -> Tuple[str, ...]:
    if frontier or version >= (4, 7):
        return _EFFORT_LEVELS
    if version >= (4, 6):
        return _EFFORT_LEVELS[:3] + _EFFORT_LEVELS[4:]  # no xhigh before 4.7
    if family == "opus" and version >= (4, 5):
        return _EFFORT_LEVELS[:3]                       # only the low, medium and high levels
    return ()


def _default_max_tokens(version: Tuple[int, int], frontier: bool) -> int:
    if frontier or version >= (4, 5):
        return 64000
    if version >= (4, 0):
        return 32000
    return 4096


def claude_model_profile(model_id: str) -> ClaudeModelProfile:
    """
    Request rules for a Claude model id. Unrecognized ids (proxy aliases, custom
    names) get a conservative profile: no thinking, no effort, sampling allowed.
    """
    parsed = _parse_model_id(model_id)
    if parsed is None:
        return ClaudeModelProfile()
    family, version = parsed
    frontier = family in _FRONTIER_FAMILIES
    adaptive = frontier or version >= ((5, 0) if family == "haiku" else (4, 6))

    return ClaudeModelProfile(
        recognized=True,
        adaptive_thinking=adaptive,
        budget_thinking=not adaptive and version >= (3, 7),
        effort_levels=_effort_levels(family, version, frontier),
        # Opus 4.7+, Sonnet 5+, Haiku 5.5 and Fable reject sampling parameters.
        sampling=not frontier and version < (4, 7),
        server_fallback=(family, version) in _SERVER_FALLBACK_MODELS,
        default_max_tokens=_default_max_tokens(version, frontier),
        context_window=1_000_000 if adaptive else 200_000,
    )


def resolve_anthropic_base_url(endpoint: Optional[str]) -> Optional[str]:
    """SDK base URL from a configured endpoint (the SDK appends `/v1/messages` itself)."""
    url = (endpoint or "").strip()
    if not url:
        return None
    if "://" not in url:
        url = f"https://{url}"
    url = url.rstrip("/")
    for suffix in ("/v1/messages", "/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


# ------------------------------------------------------------------ helpers ---
def _is_text_media(mime: str, fname: str) -> bool:
    if mime.startswith("text/") or mime in _TEXT_MIMES:
        return True
    return fname.endswith(_TEXT_EXTENSIONS)


def _text_document(text: str, title: str) -> Dict[str, Any]:
    return {
        "type": _BLOCK_DOCUMENT,
        "source": {
            "type": _BLOCK_TEXT,
            "media_type": _PLAIN_TEXT_MIME,
            # The API rejects an empty text source.
            "data": text if text.strip() else f"[{title} has no text content]",
        },
        "title": title,
    }


def _base64_chars(byte_count: int) -> int:
    return 4 * ((byte_count + 2) // 3)


def _fit_image(data: bytes) -> Tuple[bytes, str]:
    """Image bytes + media type Claude accepts: JPEG/PNG/GIF/WebP within the 5 MB limit."""
    mime = MediaNormalizer.sniff_mime(data, fallback=_OCTET_STREAM_MIME)
    oversized = _base64_chars(len(data)) > _MAX_IMAGE_BASE64_CHARS
    if mime in _IMAGE_MIMES and not oversized:
        return data, mime
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.load()
            if oversized:
                img.thumbnail((_DOWNSCALE_EDGE, _DOWNSCALE_EDGE))
            out = io.BytesIO()
            if img.mode in ("RGBA", "LA", "P"):
                img.save(out, format="PNG", optimize=True)
                new_mime = "image/png"
            else:
                img.convert("RGB").save(out, format="JPEG", quality=90)
                new_mime = "image/jpeg"
    except Exception as err:
        raise ValueError(f"Claude cannot read this image ({mime}, {len(data)} bytes) and it could not be converted: {err}") from err
    logger.info("Re-encoded image for Claude: %s %s bytes -> %s %s bytes", mime, len(data), new_mime, out.tell())
    return out.getvalue(), new_mime


def _sheet_lines(title: str, rows: Iterable[Iterable[Any]]) -> List[str]:
    lines = [f"## Sheet: {title}"]
    for row in rows:
        cells = ["" if value is None else str(value) for value in row]
        if any(cell.strip() for cell in cells):
            lines.append(" | ".join(cells))
    return lines


def _docx_text(data: bytes) -> str:
    import docx
    from docx.table import Table

    lines: List[str] = []
    # iter_inner_content keeps paragraphs and tables in document order.
    for block in docx.Document(io.BytesIO(data)).iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
        elif block.text.strip():
            lines.append(block.text)
    return "\n".join(lines)


def _xlsx_text(data: bytes) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        lines: List[str] = []
        for sheet in workbook.worksheets:
            lines.extend(_sheet_lines(sheet.title, sheet.iter_rows(values_only=True)))
        return "\n".join(lines)
    finally:
        workbook.close()


def _xls_text(data: bytes) -> str:
    import xlrd

    book = xlrd.open_workbook(file_contents=data)
    lines: List[str] = []
    for sheet in book.sheets():
        lines.extend(_sheet_lines(sheet.name, (sheet.row_values(r) for r in range(sheet.nrows))))
    return "\n".join(lines)


def _office_extractor(mime: str, fname: str) -> Optional[Callable[[bytes], str]]:
    if mime == _DOCX_MIME or fname.endswith(".docx"):
        return _docx_text
    if mime == _XLSX_MIME or fname.endswith((".xlsx", ".xlsm")):
        return _xlsx_text
    if mime == _XLS_MIME or fname.endswith(".xls"):
        return _xls_text
    return None


def _close_objects(node: Any) -> None:
    """Set additionalProperties=false on every object schema, as structured output requires."""
    if isinstance(node, list):
        for item in node:
            _close_objects(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        node.setdefault("additionalProperties", False)
    for key in ("properties", "$defs", "definitions"):
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _close_objects(child)
    for key in ("items", "anyOf", "oneOf", "allOf"):
        _close_objects(node.get(key))


def _claude_schema(response_schema: Any) -> Dict[str, Any]:
    schema = prepare_strict_schema(copy.deepcopy(response_schema))
    _close_objects(schema)
    return schema


def _guide_prompt_with_schema(prompt: str, schema: Any) -> str:
    """Fallback for models that reject output_config.format: the schema goes in the prompt."""
    try:
        schema_text = json.dumps(_claude_schema(schema), ensure_ascii=False)
    except Exception:
        schema_text = str(schema)
    return (
        f"{prompt}\n\nRespond ONLY with a valid JSON object matching this schema "
        f"(no prose, no markdown fences):\n{schema_text}"
    )


def _is_schema_rejection(err: anthropic.BadRequestError) -> bool:
    message = str(getattr(err, "message", "") or err).lower()
    return any(token in message for token in ("output_config", "output_format", "json_schema", "schema"))


def _refusal_message(raw: Dict[str, Any], model_used: Optional[str]) -> str:
    details = raw.get("stop_details") or {}
    message = f"{model_used} declined the request (category: {details.get('category') or 'unspecified'})"
    explanation = details.get("explanation")
    return f"{message}: {explanation}" if explanation else message


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """The JSON object in a structured-output reply (tolerates surrounding prose)."""
    if not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return extract_first_json_object(text)
    return parsed if isinstance(parsed, dict) else None


def _requested_effort(params: Dict[str, Any], config: ModelConfig) -> Optional[str]:
    value = params.get("reasoning_effort") or params.get("effort") or getattr(config, "reasoning_effort", None)
    if not value:
        return None
    effort = str(value).strip().lower()
    return _EFFORT_ALIASES.get(effort, effort)


def _redact_for_log(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Log-safe request view: attachment bytes become size markers and free text is truncated (PHI)."""
    safe = {k: v for k, v in kwargs.items() if k != "messages"}
    if isinstance(safe.get("system"), str):
        safe["system"] = preview_text(safe["system"])
    messages: List[Any] = []
    for message in kwargs.get("messages") or []:
        parts: List[Any] = []
        for part in message.get("content") or []:
            source = part.get("source") or {}
            if part.get("type") == _BLOCK_TEXT:
                parts.append({**part, "text": preview_text(part.get("text", ""))})
            elif isinstance(source.get("data"), str):
                parts.append({**part, "source": {**source, "data": f"<{len(source['data'])} chars redacted>"}})
            else:
                parts.append(part)
        messages.append({**message, "content": parts})
    safe["messages"] = messages
    return safe


# ------------------------------------------------------------------ adapter ---
class ClaudeAdapter(LLMAdapter):
    provider_name = "Anthropic"
    capabilities = Capabilities(text=True, image=True, document=True, structured_output=True)

    def __init__(self, config: ModelConfig):
        self.config = config
        self.profile = claude_model_profile(self.resolve_model_id())
        self.base_url = resolve_anthropic_base_url(config.endpoint_url or config.api_endpoint)
        self.effective_timeout = float(max(config.timeout or 180, 180))
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key or None,
            base_url=self.base_url,
            timeout=httpx.Timeout(self.effective_timeout, connect=30.0),
            # config.max_retries counts attempts, as in the other adapters; the SDK counts retries.
            max_retries=max((config.max_retries or 3) - 1, 0),
        )
        # Only for fetching remote attachments Claude can't take as a URL (text, Office files).
        self._http: Optional[httpx.AsyncClient] = None

    async def close(self) -> None:
        try:
            await self.client.close()
            if self._http is not None:
                await self._http.aclose()
        except Exception:  # pragma: no cover - best effort
            pass

    def resolve_model_id(self) -> str:
        """API model id. model_version is authoritative; model_name is display."""
        return (self.config.model_version or self.config.model_name or "").strip()

    # ------------------------------------------------------------ media -------
    async def _read_bytes(self, resolved: ResolvedMedia) -> bytes:
        if resolved.is_remote:
            if self._http is None:
                self._http = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.effective_timeout, connect=30.0), follow_redirects=True,
                )
            resp = await self._http.get(resolved.url)
            resp.raise_for_status()
            return resp.content
        try:
            return resolved.read_bytes()
        except ValueError as err:
            raise ValueError(
                f"Attachment '{resolved.filename}' has no readable bytes for Claude"
                f"{f' (provider file id {resolved.file_id})' if resolved.file_id else ''}"
            ) from err

    async def _image_block(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        if resolved.is_remote:
            return {"type": _BLOCK_IMAGE, "source": {"type": _SOURCE_URL, "url": resolved.url}}
        data, mime = _fit_image(await self._read_bytes(resolved))
        return {
            "type": _BLOCK_IMAGE,
            "source": {"type": _SOURCE_BASE64, "media_type": mime, "data": base64.b64encode(data).decode("ascii")},
        }

    async def _pdf_block(self, resolved: ResolvedMedia) -> Dict[str, Any]:
        if resolved.is_remote:
            source: Dict[str, Any] = {"type": _SOURCE_URL, "url": resolved.url}
        else:
            data = resolved.base64_data or base64.b64encode(await self._read_bytes(resolved)).decode("ascii")
            source = {"type": _SOURCE_BASE64, "media_type": _PDF_MIME, "data": data}
        return {"type": _BLOCK_DOCUMENT, "source": source, "title": resolved.filename}

    async def _media_block(self, media: Media) -> Dict[str, Any]:
        resolved = MediaNormalizer.resolve(media)
        mime = (resolved.mime_type or "").lower()
        fname = (resolved.filename or "attachment").lower()

        if _is_text_media(mime, fname):
            text = (await self._read_bytes(resolved)).decode("utf-8", errors="replace")
            return _text_document(text, resolved.filename)
        if mime == _PDF_MIME or fname.endswith(".pdf"):
            return await self._pdf_block(resolved)
        if resolved.kind == MediaKind.IMAGE or mime.startswith("image/"):
            return await self._image_block(resolved)

        extractor = _office_extractor(mime, fname)
        if extractor is None:
            raise ValueError(
                f"Claude cannot read '{resolved.filename}' ({mime or 'unknown type'}); "
                f"send it as a PDF, an image, a DOCX/XLSX/XLS file or plain text"
            )
        text = await asyncio.to_thread(extractor, await self._read_bytes(resolved))
        logger.info("%s: extracted %s chars of text from %s", self.provider_name, len(text), resolved.filename)
        return _text_document(text, resolved.filename)

    @staticmethod
    def _content(media_blocks: List[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
        # The API rejects empty text blocks, so a media-only request carries no prompt block.
        return [*media_blocks, {"type": _BLOCK_TEXT, "text": prompt}] if prompt else list(media_blocks)

    # ---------------------------------------------------------- payload -------
    def _output_effort(self, requested: Optional[str]) -> Optional[str]:
        if not requested:
            return None
        if not self.profile.recognized:
            # Trust an explicit, valid level for ids we can't classify (proxies, aliases).
            return requested if requested in _EFFORT_LEVELS else None
        levels = self.profile.effort_levels
        if requested in levels:
            return requested
        if levels and requested in _EFFORT_LEVELS:
            # Step down to the highest level the model accepts (xhigh -> high on 4.6, max -> high on 4.5).
            lower = [level for level in _EFFORT_LEVELS[:_EFFORT_LEVELS.index(requested)] if level in levels]
            if lower:
                return lower[-1]
        return None

    def _thinking(self, params: Dict[str, Any], requested_effort: Optional[str], max_tokens: int) -> Optional[Dict[str, Any]]:
        explicit = params.get("thinking")
        if isinstance(explicit, dict):
            return explicit
        if self.profile.adaptive_thinking:
            return {"type": "adaptive"}
        if self.profile.budget_thinking and requested_effort in _THINKING_BUDGETS:
            # budget_tokens must stay below max_tokens, leaving room for the answer.
            budget = min(_THINKING_BUDGETS[requested_effort], max_tokens - _MIN_THINKING_BUDGET)
            if budget >= _MIN_THINKING_BUDGET:
                return {"type": "enabled", "budget_tokens": budget}
        return None

    def _apply_sampling(self, kwargs: Dict[str, Any], params: Dict[str, Any]) -> None:
        requested = {k: params[k] for k in ("temperature", "top_p", "top_k") if params.get(k) is not None}
        if not requested:
            return
        if "thinking" in kwargs or not self.profile.sampling:
            reason = "thinking is on" if "thinking" in kwargs else "the model does not accept sampling parameters"
            logger.info("%s: not sending %s for %s (%s)", self.provider_name, sorted(requested), kwargs["model"], reason)
            return
        # Claude 4.x accepts temperature or top_p, not both.
        if "temperature" in requested:
            kwargs["temperature"] = requested["temperature"]
        elif "top_p" in requested:
            kwargs["top_p"] = requested["top_p"]
        if "top_k" in requested:
            kwargs["top_k"] = requested["top_k"]

    def _build_kwargs(self, request: LLMRequest, content: List[Dict[str, Any]], native_schema: bool) -> Dict[str, Any]:
        params = request.params or {}
        max_tokens = int(
            params.get("max_tokens")
            or params.get("max_completion_tokens")
            or self.config.max_tokens
            or self.profile.default_max_tokens
        )
        requested_effort = _requested_effort(params, self.config)

        kwargs: Dict[str, Any] = {
            "model": self.resolve_model_id(),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if self.config.system_prompt:
            kwargs["system"] = self.config.system_prompt
        thinking = self._thinking(params, requested_effort, max_tokens)
        if thinking:
            kwargs["thinking"] = thinking
        self._apply_sampling(kwargs, params)

        output_config: Dict[str, Any] = {}
        effort = self._output_effort(requested_effort)
        if effort:
            output_config["effort"] = effort
        if request.wants_structured and native_schema:
            output_config["format"] = {"type": "json_schema", "schema": _claude_schema(request.response_schema)}
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    def _use_server_fallback(self, params: Dict[str, Any]) -> bool:
        """`fallbacks: "default"` re-runs a refused request on Anthropic's recommended model (first-party API only)."""
        if params.get("fallbacks") is False:
            return False
        return self.profile.server_fallback and self.base_url in (None, ANTHROPIC_API_URL)

    # -------------------------------------------------------- transport -------
    async def _send(self, kwargs: Dict[str, Any], server_fallback: bool) -> Dict[str, Any]:
        output_config = kwargs.get("output_config") or {}
        # Confirmation log (visible in Dozzle) of what was actually sent.
        logger.info(
            "%s request model=%s max_tokens=%s thinking=%s effort=%s structured=%s attachments=%s fallbacks=%s",
            self.provider_name, kwargs["model"], kwargs["max_tokens"],
            (kwargs.get("thinking") or {}).get("type"), output_config.get("effort"),
            "format" in output_config, sum(1 for p in kwargs["messages"][0]["content"] if p.get("type") != _BLOCK_TEXT),
            server_fallback,
        )
        logger.debug("Payload: %s", _redact_for_log(kwargs))
        try:
            if server_fallback:
                stream_manager = self.client.beta.messages.stream(
                    **kwargs, betas=[_SERVER_FALLBACK_BETA], fallbacks="default",
                )
            else:
                stream_manager = self.client.messages.stream(**kwargs)
            async with stream_manager as stream:
                message = await stream.get_final_message()
        except anthropic.APIStatusError as err:
            logger.warning("%s error %s: %s (request_id=%s)", self.provider_name, err.status_code,
                           preview_text(err.message, 400), getattr(err, "request_id", None))
            raise
        except anthropic.APIConnectionError as err:
            logger.warning("%s connection error: %s: %s", self.provider_name,
                           type(err).__name__, str(err) or "<no message>")
            raise
        raw = message.to_dict(mode="json")
        logger.debug("%s response request_id=%s stop_reason=%s usage=%s", self.provider_name,
                     getattr(message, "_request_id", None), raw.get("stop_reason"), raw.get("usage"))
        return raw

    # ----------------------------------------------------------- parsing ------
    def _to_response(self, raw: Dict[str, Any], want_data: bool, max_tokens: int) -> LLMResponse:
        stop_reason = raw.get("stop_reason")
        model_used = UsageExtractor.extract_model(raw, self.resolve_model_id())
        if stop_reason == "refusal":
            raise ClaudeRefusalError(_refusal_message(raw, model_used))

        blocks = [b for b in raw.get("content") or [] if isinstance(b, dict)]
        self._log_fallbacks(blocks)
        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == _BLOCK_TEXT)
        if stop_reason == "max_tokens":
            logger.warning("%s output for %s hit max_tokens=%s and is truncated", self.provider_name, model_used, max_tokens)

        raw["choices"] = [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _FINISH_REASONS.get(stop_reason, stop_reason),
        }]
        return LLMResponse(
            text=text,
            data=_parse_json_object(text) if want_data else None,
            model_used=model_used,
            provider=self.provider_name,
            usage=UsageExtractor.extract(raw),
            finish_reason=stop_reason,
            raw=raw,
        )

    def _log_fallbacks(self, blocks: List[Dict[str, Any]]) -> None:
        """A `fallback` block marks where server-side fallback switched models mid-turn."""
        for block in blocks:
            if block.get("type") == "fallback":
                logger.info("%s: %s declined, %s continued", self.provider_name,
                            (block.get("from") or {}).get("model"), (block.get("to") or {}).get("model"))

    # ------------------------------------------------------ new interface -----
    async def complete(self, request: LLMRequest) -> LLMResponse:
        media_blocks = [await self._media_block(m) for m in request.media]
        server_fallback = self._use_server_fallback(request.params or {})
        kwargs = self._build_kwargs(request, self._content(media_blocks, request.prompt), native_schema=True)
        try:
            raw = await self._send(kwargs, server_fallback)
        except anthropic.BadRequestError as err:
            if not (request.wants_structured and _is_schema_rejection(err)):
                raise
            logger.warning("%s rejected native structured output for %s (%s); retrying with the schema in the prompt",
                           self.provider_name, kwargs["model"], preview_text(err.message, 300))
            guided = _guide_prompt_with_schema(request.prompt, request.response_schema)
            kwargs = self._build_kwargs(request, self._content(media_blocks, guided), native_schema=False)
            raw = await self._send(kwargs, server_fallback)
        return self._to_response(raw, want_data=request.wants_structured, max_tokens=kwargs["max_tokens"])

    async def complete_multimodal(self, request: LLMRequest) -> LLMResponse:
        # Same code path — attachments are extra content blocks on one message.
        return await self.complete(request)

    # ------------------------------------------------ backward-compat shims ---
    def _legacy_request(self, prompt: str, image_url: Optional[str], kwargs: Dict[str, Any]) -> LLMRequest:
        schema = kwargs.pop("response_schema", None)
        schema_name = kwargs.pop("response_schema_name", None)
        file_meta = kwargs.pop("file", None)
        mime = kwargs.pop("mime_type", None)
        media = legacy_media(image_url, file_meta, mime)
        return LLMRequest(
            prompt=prompt, config=self.config, media=media,
            response_schema=schema, response_schema_name=schema_name, params=dict(kwargs),
        )

    async def generate(self, prompt: str, image_url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Legacy: returns the raw provider dict (with an OpenAI-style `choices` entry)."""
        resp = await self.complete(self._legacy_request(prompt, image_url, kwargs))
        self.last_response = resp
        return resp.raw

    async def generate_structured(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Legacy: returns the parsed dict (or raw when parsing fails)."""
        kwargs["response_schema"] = response_schema
        kwargs["response_schema_name"] = response_schema_name
        resp = await self.complete(self._legacy_request(prompt, image_url, kwargs))
        # The parsed dict drops `usage`; keep the full response for usage tracking.
        self.last_response = resp
        return resp.data if resp.data is not None else resp.raw

    async def generate_multimodal(
        self,
        prompt: str,
        media: Optional[List[Dict[str, Any]]] = None,
        intro: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        response_schema_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Legacy multi-image/document entry point (returns raw provider dict)."""
        full_prompt = f"{intro}\n\n{prompt}" if intro else prompt
        request = LLMRequest(
            prompt=full_prompt,
            config=self.config,
            media=media_from_legacy(media),
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            params=dict(kwargs),
        )
        resp = await self.complete(request)
        self.last_response = resp
        return resp.raw
