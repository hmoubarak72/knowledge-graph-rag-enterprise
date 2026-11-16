"""Full pipeline on the synthetic fixture corpus with the offline LLM + in-memory stores."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from kgrag.embeddings import get_embedder
from kgrag.ingest.chunker import chunk_document, load_document, split_sections
from kgrag.ingest.pipeline import run
from kgrag.llm import LLMClient
from kgrag.rag import HybridRAG, VectorOnlyRAG
from kgrag.stores.factory import make_stores


def test_chunker_is_section_aware_and_deterministic(corpus: Path) -> None:
    doc = load_document(corpus / "NVDA_10K_2025.txt")
    sections = [s for s, _ in split_sections(doc.text)]
    assert "Item 1 > Business" in sections and "Item 1A > Risk Factors" in sections
    a = chunk_document(doc, 600, 100)
    b = chunk_document(doc, 600, 100)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.chunk_id.startswith("NVDA-10K-2025:") for c in a)
    assert all(len(c.text) <= 600 + 100 for c in a)


def test_ingest_is_idempotent_and_answers(offline_settings, corpus: Path) -> None:  # type: ignore[no-untyped-def]
    est = run(corpus, offline_settings, estimate_only=True)
    assert est["chunks"] > 0
    r1 = run(corpus, offline_settings, check_idempotent=True)
    r2 = run(corpus, offline_settings)  # second full run: extraction is free, counts are stable
    assert r1["graph_counts"] == r2["graph_counts"]
    assert r2["cost_usd"]["extraction"] == 0.0  # type: ignore[index]
    assert r1["invariant"]["shared_chunk_ids"] == r1["chunks"]  # type: ignore[index]

    llm = LLMClient(offline_settings)
    graph, vectors = make_stores(offline_settings)
    names = {e["name"] for e in graph.entities()}
    assert "Taiwan Semiconductor Manufacturing Company Limited" in names
    hybrid = HybridRAG(offline_settings, llm, get_embedder(), vectors, graph)

    resp, tr = hybrid.ask("Who is the CEO of the company that acquired VMware?")
    assert resp.route_used == "GRAPH" and tr.query_type == "TWO_HOP"
    assert "Hock Tan" in resp.answer and resp.citations and resp.graph_paths
    assert {c.chunk_id for c in resp.citations} <= vectors.chunk_ids()

    resp, tr = hybrid.ask("Which suppliers do both NVIDIA and AMD rely on?")
    assert tr.query_type == "SHARED_NEIGHBORS" and "Taiwan Semiconductor" in resp.answer

    resp, _ = hybrid.ask("Write me a poem about semiconductors.")
    assert resp.refused and not resp.citations

    base = VectorOnlyRAG(offline_settings, llm, get_embedder(), vectors)
    resp, _ = base.ask("What restrictions apply to NVIDIA's sales to China?")
    assert resp.route_used == "VECTOR" and resp.citations

    log = (offline_settings.logs_dir / "routing.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 3 and '"outcome"' in log[0]


def test_api(offline_settings, corpus: Path) -> None:  # type: ignore[no-untyped-def]
    run(corpus, offline_settings)
    from kgrag.api.app import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ask", json={"question": "Who leads the company that acquired Xilinx?"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"answer", "citations", "route_used"} and "Lisa Su" in body["answer"]
        assert client.post("/ask", json={"question": ""}).status_code == 422
