from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    database_url: str
    archive_dir: Path
    pitchapi_key: str | None
    strikerapi_key: str | None
    user_agent: str


def get_settings() -> Settings:
    archive = Path(os.getenv("ARCHIVE_DIR", "data/archive"))
    if not archive.is_absolute():
        archive = ROOT / archive
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://fpl:fpl@localhost:5432/fpl"),
        archive_dir=archive,
        pitchapi_key=os.getenv("PITCHAPI_API_KEY") or None,
        strikerapi_key=os.getenv("STRIKERAPI_API_KEY") or None,
        user_agent=os.getenv("HTTP_USER_AGENT", "fpl-prediction-engine/0.1"),
    )
