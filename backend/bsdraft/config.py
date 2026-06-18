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
    # Re-scan a known player once their last fetch is older than this many hours, to pick up
    # ranked games played since (the API only exposes a player's last ~25 battles). 0 disables
    # re-scanning — each player is fetched at most once.
    crawl_revisit_hours: float = 12.0

    # Cloud serving / live refresh
    # URL of the published matches dataset (GitHub Release asset, gzipped). When set, the API
    # syncs it at startup and every `refresh_seconds`, rebuilding draft stats with no restart.
    data_url: str = ""
    # URL of the published win-prob model (winprob.npz Release asset). When set, the API syncs
    # it alongside the dataset and hot-swaps the reloaded model in without a restart — so a
    # retrain (e.g. after a balance shift) rolls out live instead of waiting for a redeploy.
    model_url: str = ""
    refresh_seconds: int = 600  # re-sync interval in seconds (0 disables the refresh loop)

    # Engine tuning
    stats_halflife_days: float = 21.0  # recency half-life (days) for empirical stats; <=0 disables decay
    # The frontend re-polls the roster so a long session picks up newly unlocked/upgraded
    # brawlers. Serve a cached roster for this many seconds so that polling (and multiple
    # tabs) don't each hit the live Supercell API. Keep it well below the poll interval so a
    # poll still refreshes. 0 disables caching (every request fetches live).
    roster_ttl_seconds: int = 90

    @property
    def seed_countries(self):
        return [c.strip() for c in self.crawl_seed_countries.split(",") if c.strip()]

    @property
    def normalized_player_tag(self):
        """Player tag without a leading '#', uppercased (API path form)."""
        return self.player_tag.lstrip("#").strip().upper()


settings = Settings()
