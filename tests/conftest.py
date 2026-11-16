from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def offline_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """Fresh offline settings with an isolated data dir (cache, ledger, logs, results)."""
    from kgrag import config, embeddings

    monkeypatch.setenv("KGRAG_OFFLINE", "1")
    monkeypatch.setenv("KGRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KGRAG_BUDGET_USD", "5")
    monkeypatch.setenv("KGRAG_ER_THRESHOLD", "0.92")
    config.get_settings.cache_clear()
    embeddings.get_embedder.cache_clear()
    yield config.get_settings()
    config.get_settings.cache_clear()
    embeddings.get_embedder.cache_clear()


@pytest.fixture()
def corpus() -> Path:
    return FIXTURES / "corpus"
