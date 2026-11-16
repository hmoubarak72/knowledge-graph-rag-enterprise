"""Single source of configuration. Values come from the environment / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    extraction_model: str = Field(default="claude-opus-5-5", alias="KGRAG_EXTRACTION_MODEL")
    synthesis_model: str = Field(default="claude-opus-5-5", alias="KGRAG_SYNTHESIS_MODEL")
    router_model: str = Field(default="claude-haiku-5-5", alias="KGRAG_ROUTER_MODEL")

    budget_usd: float = Field(default=25.0, alias="KGRAG_BUDGET_USD")
    offline: bool = Field(default=False, alias="KGRAG_OFFLINE")

    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="kgrag-dev-password", alias="NEO4J_PASSWORD")
    pg_dsn: str = Field(default="postgresql://kgrag:kgrag@localhost:5432/kgrag", alias="PG_DSN")

    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="KGRAG_EMBED_MODEL")
    embed_dim: int = Field(default=384, alias="KGRAG_EMBED_DIM")

    er_threshold: float = Field(default=0.91, alias="KGRAG_ER_THRESHOLD")
    router_confidence: float = Field(default=0.70, alias="KGRAG_ROUTER_CONFIDENCE")

    # HNSW parameters (tuned in Phase 2 via kgrag.eval.recall)
    hnsw_m: int = Field(default=16, alias="KGRAG_HNSW_M")
    hnsw_ef_construction: int = Field(default=128, alias="KGRAG_HNSW_EF_CONSTRUCTION")
    hnsw_ef_search: int = Field(default=80, alias="KGRAG_HNSW_EF_SEARCH")
    vector_top_k: int = Field(default=8, alias="KGRAG_VECTOR_TOP_K")

    chunk_max_chars: int = Field(default=3500, alias="KGRAG_CHUNK_MAX_CHARS")
    chunk_overlap_chars: int = Field(default=300, alias="KGRAG_CHUNK_OVERLAP_CHARS")

    sec_user_agent: str = Field(default="", alias="SEC_USER_AGENT")

    data_dir: Path = Field(default=ROOT / "data", alias="KGRAG_DATA_DIR")

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def eval_dir(self) -> Path:
        """Labeled sets are versioned with the code, independent of the runtime data dir."""
        return ROOT / "data" / "eval"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.logs_dir, self.eval_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
