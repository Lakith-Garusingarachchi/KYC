"""
Gemini on Vertex AI adapter (text + vision + documents).

Same Gemini `contents`/`parts` request/response shape as the AI-Studio adapter,
so it subclasses GeminiAdapter and reuses all payload building, media handling
(inline_data for images/PDFs) and response parsing. Only two things differ:

  1. Auth — Vertex uses a Google Cloud **service-account JSON** exchanged for an
     OAuth2 Bearer token (cloud-platform scope), instead of an AI-Studio API key.
  2. Endpoint — a project/region URL:
     https://{loc}-aiplatform.googleapis.com/v1/projects/{project}/locations/{loc}
       /publishers/google/models/{model}:generateContent
     (host is aiplatform.googleapis.com when location == "global").

Billing therefore runs through the customer's Vertex AI / GCP project.

Config (from ModelConfig, populated from the model's authentication record):
  * service_account_json : the SA key file contents (JSON string) — required
  * vertex_location      : region, default "us-central1"
  * vertex_project       : optional override; otherwise taken from the SA JSON

Requires the `google-auth` package (see requirements.txt).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from ..config import ModelConfig
from ..contracts import Capabilities
from .gemini import GeminiAdapter

logger = logging.getLogger(__name__)

_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GeminiVertexAdapter(GeminiAdapter):
    provider_name = "GeminiVertex"
    capabilities = Capabilities(text=True, image=True, document=True, structured_output=True)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._sa_info: Optional[Dict[str, Any]] = self._parse_sa(config.service_account_json)
        self._location = (getattr(config, "vertex_location", None) or "us-central1").strip()
        self._project = (
            getattr(config, "vertex_project", None)
            or (self._sa_info or {}).get("project_id")
        )
        self._creds = None  # lazily built google.oauth2 credentials

    @staticmethod
    def _parse_sa(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"service_account_json is not valid JSON: {exc}") from exc

    # -- endpoint (project/region, no ?key=) -----------------------------------
    def _endpoint(self) -> str:
        if self.config.endpoint_url:
            return self.config.endpoint_url
        if not self._project:
            raise ValueError(
                "Vertex Gemini requires a project id (from the service-account JSON "
                "or vertex_project)."
            )
        loc = self._location
        host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1/projects/{self._project}/locations/{loc}"
            f"/publishers/google/models/{self._model_id()}:generateContent"
        )

    # -- auth: service-account JSON -> OAuth2 Bearer ---------------------------
    async def _auth_headers(self) -> Dict[str, str]:
        token = await self._access_token()
        return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    async def _access_token(self) -> str:
        if not self._sa_info:
            raise ValueError(
                "Vertex Gemini requires a service-account JSON (upload it as the "
                "model's authentication 'service_account_json')."
            )
        # google-auth's refresh is blocking; run it off the event loop. Credentials
        # cache the token and only hit the network when expired.
        return await asyncio.to_thread(self._refresh_token_sync)

    def _refresh_token_sync(self) -> str:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "google-auth is required for the Vertex Gemini adapter. "
                "Add 'google-auth' (and 'requests') to requirements.txt."
            ) from exc

        if self._creds is None:
            self._creds = service_account.Credentials.from_service_account_info(
                self._sa_info, scopes=[_SCOPE]
            )
        if not self._creds.valid:
            self._creds.refresh(Request())
        return self._creds.token
