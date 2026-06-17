"""Runtime configuration loaded from .env (API token, player tag, crawler tuning).

Importing this requires `pydantic-settings` (see backend/requirements.txt). The
pure-data reference layer deliberately does NOT import this, so it can run without
installing dependencies.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from bsdraft.constants import REPO_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets
    brawlstars_api_token: str = ""
    player_tag: str = ""

    # Crawler tuning
    crawl_rate_limit_per_sec: float = 5.0
    crawl_seed_countries: str = "global,US,DE,KR,BR,JP"
    crawl_ranked_only: bool = True

    @property
    def seed_countries(self):
        return [c.strip() for c in self.crawl_seed_countries.split(",") if c.strip()]

    @property
    def normalized_player_tag(self):
        """Player tag without a leading '#', uppercased (API path form)."""
        return self.player_tag.lstrip("#").strip().upper()


settings = Settings()
