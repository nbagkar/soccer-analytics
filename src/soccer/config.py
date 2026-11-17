"""Configuration. Env-driven, validated at startup, no secrets in the repo.

Precedence: environment variables > .env file > defaults. Every path is derived from
`data_dir` so a single override relocates the whole install.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from soccer.sources.registry import SourceId


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SOCCER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Credentials -------------------------------------------------------
    # football-data.org requires a free API token. TheSportsDB's documented free
    # key is "123"; it is not a secret, but it is overridable so a supporter key
    # can be swapped in without a code change.
    football_data_org_token: str | None = None
    thesportsdb_key: str = "123"
    # Optional LLM narrator for the assistant (soccer.intelligence). Off unless provider,
    # model and key are all set; see .env.example.
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_monthly_token_cap: int = Field(default=2_000_000, ge=1)
    llm_price_in_per_m: float | None = None
    """USD per million input tokens, from the provider's pricing page -- for cost estimates."""
    llm_price_out_per_m: float | None = None
    api_football_key: SecretStr | None = None
    """API-Football (api-sports.io) key. Unset = the source is off. SecretStr so it never
    lands in a log line or a repr."""

    # --- Source toggles ----------------------------------------------------
    enable_fpl: bool = False
    """Off by default: Premier League terms bar 'creating a database'. Enabling this
    is a decision the operator makes knowingly. See registry caveats."""

    # --- Usage tracking --------------------------------------------------
    usage_tracking: bool = True
    """Log page views, actions and assistant intents to the local live DB (see
    domain/usage.py). Local only -- nothing is sent anywhere. Set false to disable."""

    auto_refresh: bool = True
    """Refresh results and fixtures in the background when the dashboard opens on data more
    than a day old (see dashboard/autorefresh.py). Needs network; set false to stay manual."""

    # --- Politeness --------------------------------------------------------
    # Deliberately below each provider's published ceiling. football-data.org's real
    # limit is 10/min; we budget 8 to leave headroom for retries without tripping it.
    football_data_org_rpm: int = Field(default=8, ge=1, le=10)
    thesportsdb_rpm: int = Field(default=20, ge=1, le=30)
    # API-Football's free plan: 100 requests/day (resets 00:00 UTC), a per-minute cap
    # reported as 10 by its own docs. Paid plans raise both; set these to match.
    api_football_daily_limit: int = Field(default=100, ge=1)
    api_football_rpm: int = Field(default=10, ge=1)
    api_football_reserve: int = Field(default=5, ge=0)
    """Requests per day held back from automated jobs, so a manual check is always possible."""

    http_timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("data_dir")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    # --- Derived paths -----------------------------------------------------

    @property
    def raw_dir(self) -> Path:
        """Immutable provider snapshots. Never mutated, only appended to."""
        return self.data_dir / "raw"

    @property
    def live_db(self) -> Path:
        return self.data_dir / "live.sqlite"

    @property
    def usage_db(self) -> Path:
        """The local usage log -- kept out of live.sqlite so that logging a page view never
        changes the live store's mtime, which keys the dashboard's data caches."""
        return self.data_dir / "usage.sqlite"

    @property
    def analytics_db(self) -> Path:
        return self.data_dir / "analytics.duckdb"

    @property
    def quota_db(self) -> Path:
        """Paid-API request ledger -- its own file, like usage.sqlite, so counting a request
        never touches a data store's mtime (which keys the dashboard caches)."""
        return self.data_dir / "quota.sqlite"

    def is_enabled(self, source: SourceId) -> bool:
        """Whether a source may be used, honouring both its default and the override."""
        match source:
            case SourceId.OPENLIGADB:
                # Registry record of verified capabilities only -- no adapter is built.
                return False
            case SourceId.FPL:
                return self.enable_fpl
            case SourceId.API_FOOTBALL:
                return self.api_football_key is not None
            case SourceId.FOOTBALL_DATA_ORG:
                # Silently disabled without a token rather than failing at request time.
                return self.football_data_org_token is not None
            case _:
                return True

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None

# Absolute path of the .env to read, for processes started outside the project directory --
# `soccer dashboard` runs Streamlit from the package dir (so its theme loads), where a
# relative ".env" silently doesn't exist: no token, FPL off, and every refresh needing them
# quietly skipped.
ENV_FILE_VAR = "SOCCER_ENV_FILE"


def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    global _settings
    if _settings is None:
        env_file = os.environ.get(ENV_FILE_VAR)
        # `_env_file` is pydantic-settings' documented init override; mypy (without the
        # pydantic plugin) can't see it on the generated __init__.
        _settings = Settings(_env_file=env_file) if env_file else Settings()  # type: ignore[call-arg]
    return _settings
