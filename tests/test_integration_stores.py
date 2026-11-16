"""Real Neo4j + pgvector. Run with `docker compose up -d neo4j postgres` then `pytest -m integration`."""

from __future__ import annotations

import numpy as np
import pytest

from kgrag.config import Settings
from kgrag.models import Chunk, ResolvedEdge, ResolvedNode
from kgrag.ontology import EntityType, RelationType
from kgrag.retrieval.templates import QueryType, TemplateParams
from kgrag.stores.factory import verify_chunk_invariant
from kgrag.stores.graph import MemoryGraphStore, Neo4jGraphStore
from kgrag.stores.vector import PgVectorStore

pytestmark = pytest.mark.integration


def _data() -> tuple[list[ResolvedNode], list[ResolvedEdge], list[Chunk]]:
    c = [Chunk(chunk_id=f"IT-10K-2025:item-1:000{i}", document_id="IT-10K-2025", company="ItCo", section_path="Item 1",
               date="2025-01-01", ordinal=i, text=t) for i, t in enumerate(
        ["Broadcom acquired VMware.", "Hock Tan is CEO of Broadcom.", "TSMC supplies Broadcom and AMD."])]
    n = [ResolvedNode(node_id=f"Company:{x}", entity_type=EntityType.COMPANY, canonical_name=x.title(), aliases=[],
                      mention_count=1, chunk_ids=[c[0].chunk_id]) for x in ("broadcom", "vmware", "tsmc", "amd")]
    n.append(ResolvedNode(node_id="Person:hock tan", entity_type=EntityType.PERSON, canonical_name="Hock Tan",
                          aliases=[], mention_count=1, chunk_ids=[c[1].chunk_id]))
    def e(k: str, s: str, r: RelationType, t: str, ci: int) -> ResolvedEdge:
        return ResolvedEdge(edge_key=k, source_id=s, target_id=t, relationship_type=r,
                            source_chunk_ids=[c[ci].chunk_id], evidence=[c[ci].text], confidence=0.9)
    edges = [e("e1", "Company:broadcom", RelationType.ACQUIRED, "Company:vmware", 0),
             e("e2", "Company:broadcom", RelationType.EMPLOYS, "Person:hock tan", 1),
             e("e3", "Company:tsmc", RelationType.SUPPLIES, "Company:broadcom", 2),
             e("e4", "Company:tsmc", RelationType.SUPPLIES, "Company:amd", 2)]
    return n, edges, c


@pytest.fixture()
def stores():  # type: ignore[no-untyped-def]
    s = Settings()
    try:
        g = Neo4jGraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
        g.driver.verify_connectivity()
        v = PgVectorStore(s.pg_dsn, 4, 16, 64, 40)
    except Exception as err:  # pragma: no cover
        pytest.skip(f"stores not reachable: {err}")
    with g.driver.session() as sess:
        sess.run("MATCH (n) WHERE n.node_id IN $ids OR n.chunk_id STARTS WITH 'IT-' DETACH DELETE n",
                 ids=[x.node_id for x in _data()[0]])
    v.conn.execute("DROP TABLE IF EXISTS chunks")
    yield g, v
    g.close()
    v.close()


def test_merge_idempotent_and_templates_match_memory_store(stores) -> None:  # type: ignore[no-untyped-def]
    g, v = stores
    nodes, edges, chunks = _data()
    g.setup()
    v.setup()
    g.write(nodes, edges, chunks)
    first = g.counts()
    g.write(nodes, edges, chunks)
    assert g.counts() == first
    v.upsert(chunks, np.eye(4, dtype=np.float32)[:3], {})
    verify_chunk_invariant(g, v)

    mem = MemoryGraphStore()
    mem.write(nodes, edges, chunks)
    cases = [
        (QueryType.TWO_HOP, TemplateParams(entity_ids=["Company:vmware"], rel_types=[RelationType.ACQUIRED],
                                           direction="in", rel_types_2=[RelationType.EMPLOYS], direction_2="out")),
        (QueryType.SHARED_NEIGHBORS, TemplateParams(entity_ids=["Company:broadcom", "Company:amd"],
                                                    rel_types=[RelationType.SUPPLIES], direction="in")),
        (QueryType.AGGREGATE_DEGREE, TemplateParams(rel_types=[RelationType.SUPPLIES], direction="out")),
    ]
    for qt, p in cases:
        neo = sorted(tuple(n["node_id"] for n in x.nodes) for x in g.run(qt, p))
        py = sorted(tuple(n["node_id"] for n in x.nodes) for x in mem.run(qt, p))
        assert neo == py and neo, qt
