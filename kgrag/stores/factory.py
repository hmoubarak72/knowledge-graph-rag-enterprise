"""Build stores from settings. Offline mode uses the in-memory stores persisted under data/cache/."""

from __future__ import annotations

from kgrag.config import Settings
from kgrag.stores.graph import GraphStore, MemoryGraphStore, Neo4jGraphStore
from kgrag.stores.vector import MemoryVectorStore, PgVectorStore, VectorStore


class ChunkIdInvariantError(RuntimeError):
    """The shared chunk-ID key between graph and vector store is broken."""


def make_stores(s: Settings) -> tuple[GraphStore, VectorStore]:
    if s.offline:
        return (MemoryGraphStore(s.cache_dir / "memory_graph.json"),
                MemoryVectorStore(s.cache_dir / "memory_vectors.json", s.embed_dim))
    return (Neo4jGraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password),
            PgVectorStore(s.pg_dsn, s.embed_dim, s.hnsw_m, s.hnsw_ef_construction, s.hnsw_ef_search))


def verify_chunk_invariant(graph: GraphStore, vectors: VectorStore) -> dict[str, int]:
    """Both stores must hold exactly the same chunk IDs, and every edge must cite a stored chunk."""
    g, v = graph.chunk_ids(), vectors.chunk_ids()
    if g != v:
        only_g, only_v = sorted(g - v)[:5], sorted(v - g)[:5]
        raise ChunkIdInvariantError(f"chunk id sets differ: graph-only={only_g} vector-only={only_v}")
    edge_ids = graph.edge_chunk_ids()
    dangling = edge_ids - g
    if dangling:
        raise ChunkIdInvariantError(f"edges cite unknown chunks: {sorted(dangling)[:5]}")
    return {"shared_chunk_ids": len(g), "edge_cited_chunks": len(edge_ids)}
