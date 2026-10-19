"""Runtime configuration loaded from environment variables (and a local `.env` file).

All settings use the ``CTVIZ_`` prefix except ``OPENAI_API_KEY``. Secrets are held in
``SecretStr`` so they never appear in reprs, logs, or error messages.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    """Typed service settings. Instantiate via :func:`get_settings`."""

    model_config = SettingsConfigDict(env_prefix="CTVIZ_", extra="ignore", populate_by_name=True)

    # Secrets / mode
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    offline: bool = False

    # Model routing (defaults restricted to models the project key can access)
    model_planner: str = "gpt-5.4-nano"
    model_synthesis: str = "gpt-5.4-nano"
    model_extract: str = "gpt-5.4-nano"
    model_analysis: str = "gpt-5.4-mini"
    model_escalation: str = "gpt-5.4-mini"
    chart_planner_enabled: bool = True  # one validated LLM call after retrieval picks the view (agents/chart_planner)
    model_max: str = "gpt-5.4"
    embeddings_enabled: bool = False
    embed_model: str = "text-embedding-3-small"
    llm_timeout_s: float = 8.0

    # Storage
    db_path: Path = PROJECT_ROOT / "ctviz.db"
    fixtures_dir: Path = PROJECT_ROOT / "tests" / "fixtures" / "ctgov"
    record: bool = False
    cache_dir: Path = PROJECT_ROOT / ".cache" / "ctgov"  # readable copy of every API response
    replay: bool = False  # serve ClinicalTrials.gov calls only from cache_dir (no network)
    runs_dir: Path = PROJECT_ROOT / "runs"  # per-request artifacts for inspection
    # Write runs/<request_id>/ artifacts. Off for the server unless enabled or CTVIZ_RUNS_DIR is set;
    # `ctviz ask` turns it on.
    runs_enabled: bool = False

    # ClinicalTrials.gov outbound
    ctgov_base: str = "https://clinicaltrials.gov/api/v2"
    # burst + rpm <= 50 keeps any 60 s window under ClinicalTrials.gov's ~50 requests/minute limit, while a
    # burst of 20 lets registry-count charts (one count per bucket) start immediately
    ctgov_rpm: int = Field(default=30, ge=1, le=600)
    ctgov_burst: int = Field(default=20, ge=1, le=100)
    ctgov_concurrency: int = Field(default=8, ge=1, le=32)
    max_studies: int = Field(default=25000, ge=1, le=25000)  # read every matching trial up to this many
    max_ctgov_calls_per_request: int = Field(default=40, ge=1, le=100)
    max_full_records: int = Field(default=25, ge=1, le=100)

    # Inbound
    inbound_rpm: int = Field(default=30, ge=1)
    inbound_burst: int = Field(default=10, ge=1)
    max_inflight: int = Field(default=16, ge=1)
    trust_proxy: bool = False
    # Keys that get their own inbound rate-limit bucket (JSON list in CTVIZ_API_KEYS); others are bucketed by IP.
    api_keys: list[SecretStr] = Field(default_factory=list)

    # Budgets
    request_deadline_s: float = 60.0  # quality over latency: full reads of up to 25k trials
    request_cost_cap_usd: float = 0.05
    max_llm_calls: int = 4

    # Cache TTLs (seconds)
    ttl_http_list_s: int = 6 * 3600
    ttl_http_study_s: int = 24 * 3600
    ttl_cohort_s: int = 24 * 3600  # cohort snapshots (refreshed daily by `ctviz warm`)
    ttl_response_s: int = 3600
    ttl_plan_s: int = 7 * 24 * 3600

    # Observability
    log_level: str = "INFO"
    include_trace: bool = True

    def require_llm_key(self) -> str:
        """Return the OpenAI key or raise a clear error. Never logs the key itself."""
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise ConfigError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and set OPENAI_API_KEY, "
                "or run with CTVIZ_OFFLINE=1 to use recorded fixtures and the rule-based planner."
            )
        return self.openai_api_key.get_secret_value()

    @property
    def runs_active(self) -> bool:
        """True when per-request run artifacts should be written."""
        return self.runs_enabled or "runs_dir" in self.model_fields_set

    @property
    def cohort_limit(self) -> int:
        """Trials read per cohort. Offline fixtures were recorded with ``CTVIZ_MAX_STUDIES=3000`` (``ctviz record``),
        so offline mode reads at most that many to request exactly the recorded pages."""
        return min(self.max_studies, OFFLINE_MAX_STUDIES) if self.offline and not self.replay else self.max_studies

    @property
    def llm_enabled(self) -> bool:
        return not self.offline and self.openai_api_key is not None


OFFLINE_MAX_STUDIES = 3000


def validate_settings(settings: Settings) -> Settings:
    """Fail fast at startup: online mode requires an OpenAI key."""
    if not settings.offline:
        settings.require_llm_key()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load `.env` (if present, without overriding real env vars) and return validated settings."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return validate_settings(Settings())
