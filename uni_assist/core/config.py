"""Application configuration for Uni-Assist v2."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    """Central configuration shared across application layers."""

    project_root: Path
    data_dir: Path
    database_path: Path
    database_url: str
    default_user_id: str
    environment: str


def load_settings() -> Settings:
    """Load settings from environment variables with local defaults."""

    data_dir = Path(
        os.getenv("UNI_ASSIST_DATA_DIR", str(DEFAULT_DATA_DIR))
    ).expanduser().resolve()

    database_path = data_dir / "uni_assist.db"

    database_url = os.getenv(
        "UNI_ASSIST_DATABASE_URL",
        f"sqlite:///{database_path}",
    )

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        database_path=database_path,
        database_url=database_url,
        default_user_id=os.getenv(
            "UNI_ASSIST_DEFAULT_USER_ID",
            "default_user",
        ),
        environment=os.getenv(
            "UNI_ASSIST_ENVIRONMENT",
            "local",
        ),
    )


settings = load_settings()
