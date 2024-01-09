import os
from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///data/runtime/adapterlab.db")
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "artifacts")).resolve()
    )
    protected: bool = field(
        default_factory=lambda: os.getenv("PROTECTED_MODE", "false").lower() == "true"
    )
    operator_token: str = field(default_factory=lambda: os.getenv("OPERATOR_TOKEN", ""))
    model_path: str = MODEL_ID
    model_revision: str | None = MODEL_REVISION
    seed_demo: bool = True

    def validate(self):
        if self.protected and len(self.operator_token) < 32:
            raise ValueError("Protected mode requires an OPERATOR_TOKEN of at least 32 characters")
