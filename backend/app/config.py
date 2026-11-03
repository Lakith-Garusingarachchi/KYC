"""Application settings, read from environment variables / backend/.env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- app -----------------------------------------------------------------
    app_name: str = "KYC MVP"
    environment: str = "development"
    log_level: str = "INFO"
    frontend_url: str = "http://localhost:3000"
    cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- storage -------------------------------------------------------------
    database_url: str = f"sqlite+aiosqlite:///{BACKEND_DIR / 'data' / 'kyc.db'}"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    # Fernet key; uploaded documents and face frames are encrypted at rest with it.
    file_encryption_key: str = ""
    max_upload_mb: int = 10

    # --- auth (OAuth2 password flow + JWT, optional Google OAuth) -------------
    auth_secret_key: str = ""
    auth_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 8
    auth_cookie_name: str = "kyc_token"
    auth_cookie_secure: bool = False
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:3000/api/auth/google/callback"

    # --- LLM adapter selection ------------------------------------------------
    # Vision-capable provider used to read documents and review face frames:
    #   gemini | gemini_vertex | openai | claude | deepseek (text only)
    llm_provider: str = "gemini"
    # Optional provider for text-only reasoning (address comparison). Defaults to llm_provider.
    llm_text_provider: Optional[str] = None
    llm_timeout_seconds: int = 180
    llm_max_tokens: int = 4096

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_endpoint: Optional[str] = None

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-5"
    claude_endpoint: Optional[str] = None

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    vertex_service_account_file: Optional[str] = None
    vertex_service_account_json: Optional[str] = None
    vertex_project_id: Optional[str] = None
    vertex_location: str = "us-central1"
    vertex_model: str = "gemini-2.5-flash"

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"

    # --- face / liveness --------------------------------------------------------
    face_models_dir: Path = BACKEND_DIR / "models"
    # SFace cosine similarity: >= match_threshold is a match (OpenCV's recommended 0.363),
    # below reject_threshold is a clear mismatch; in between goes to manual review.
    face_match_threshold: float = 0.363
    face_reject_threshold: float = 0.25
    liveness_challenge_ttl_seconds: int = 300
    liveness_yaw_delta: float = 0.10
    liveness_pitch_delta: float = 0.03

    # --- decision ---------------------------------------------------------------
    # Verification score (0-100) needed to pass; below it -> manual review.
    kyc_pass_score: int = 40
    # Optional: checks that must pass regardless of the score (comma-separated check ids).
    # Empty = the score alone decides.
    kyc_required_checks: Annotated[List[str], NoDecode] = Field(default_factory=list)

    # --- document rules ---------------------------------------------------------
    bill_max_age_days: int = 90
    name_match_threshold: int = 80
    address_match_threshold: int = 75

    @field_validator("cors_origins", "kyc_required_checks", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @property
    def text_provider(self) -> str:
        return (self.llm_text_provider or self.llm_provider).strip().lower()

    @property
    def google_oauth_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
