"""Central settings, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Optional at import time so /health works without a key; the LLM client
    # raises ConfigError when it is actually needed.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # Small output budget: the JSON answer is ~100-150 tokens. Keeping this low
    # leaves token-per-minute headroom and bounds cost.
    max_output_tokens: int = Field(400, ge=50, le=4096)
    # None = do not send (the SDK's messages.create() has no temperature argument;
    # when set, it is sent via extra_body).
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    request_timeout_s: float = Field(30.0, gt=0)

    # Retries (hard ceiling on attempts).
    retry_max_attempts: int = Field(4, ge=1, le=10)
    retry_base_delay_s: float = Field(1.0, ge=0)
    retry_max_delay_s: float = Field(20.0, ge=0)

    # At most this many model calls in flight at once (rate-limit headroom).
    max_concurrent_calls: int = Field(4, ge=1, le=64)

    # Prompt files (stem names under prompts/).
    triage_prompt: str = "triage_v1"
    repair_prompt: str = "repair_v1"
    prompts_dir: Path = BASE_DIR / "prompts"

    # Observability / privacy
    log_path: Path = BASE_DIR / "logs" / "triage.jsonl"
    log_raw_content: bool = False

    # Optional cost tracking (USD per million tokens). None = not configured.
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
