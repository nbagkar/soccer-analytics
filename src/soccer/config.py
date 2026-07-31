"""Configuration. Env-driven, validated at startup, no secrets in the repo.

Precedence: environment variables > .env file > defaults. Every path is derived from
`data_dir` so a single override relocates the whole install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    # --- Source toggles ----------------------------------------------------
    enable_openligadb: bool = False
    enable_fpl: bool = False
    """Off by default: Premier League terms bar 'creating a database'. Enabling this
    is a decision the operator makes knowingly. See registry caveats."""

    # --- Politeness --------------------------------------------------------
    # Deliberately below each provider's published ceiling. football-data.org's real
    # limit is 10/min; we budget 8 to leave headroom for retries without tripping it.
    football_data_org_rpm: int = Field(default=8, ge=1, le=10)
    thesportsdb_rpm: int = Field(default=20, ge=1, le=30)

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
    def parquet_dir(self) -> Path:
        return self.data_dir / "parquet"

    @property
    def live_db(self) -> Path:
        return self.data_dir / "live.sqlite"

    @property
    def analytics_db(self) -> Path:
        return self.data_dir / "analytics.duckdb"

    def is_enabled(self, source: SourceId) -> bool:
        """Whether a source may be used, honouring both its default and the override."""
        match source:
            case SourceId.OPENLIGADB:
                return self.enable_openligadb
            case SourceId.FPL:
                return self.enable_fpl
            case SourceId.FOOTBALL_DATA_ORG:
                # Silently disabled without a token rather than failing at request time.
                return self.football_data_org_token is not None
            case _:
                return True

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.raw_dir, self.parquet_dir):
            path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
