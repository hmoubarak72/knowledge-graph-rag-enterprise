"""Vector store over the SAME chunk IDs as the graph. `PgVectorStore` (HNSW) + `MemoryVectorStore`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from kgrag.log import get_logger
from kgrag.models import Chunk, Passage

log = get_logger(__name__)


class VectorStore(Protocol):
    def setup(self) -> None: ...
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, entity_ids: dict[str, list[str]]) -> None: ...
    def search(self, qvec: np.ndarray, k: int, ef_search: int | None = None, exact: bool = False,
               company: str | None = None) -> list[Passage]: ...
    def get(self, chunk_ids: list[str]) -> list[Passage]: ...
    def chunk_ids(self) -> set[str]: ...
    def all_chunks(self) -> list[Chunk]: ...
    def close(self) -> None: ...


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, m: int, ef_construction: int, ef_search: int) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.dim, self.m, self.efc, self.ef_search = dim, m, ef_construction, ef_search

    def close(self) -> None:
        self.conn.close()

    def setup(self) -> None:
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id     TEXT PRIMARY KEY,
                document_id  TEXT NOT NULL,
                company      TEXT NOT NULL,
                section_path TEXT NOT NULL,
                date         TEXT,
                ordinal      INT,
                text         TEXT NOT NULL,
                entity_ids   TEXT[] NOT NULL DEFAULT '{{}}',
                embedding    vector({self.dim}) NOT NULL
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id)")
        self.build_index(self.m, self.efc, if_missing=True)

    def build_index(self, m: int, ef_construction: int, if_missing: bool = False) -> None:
        if not if_missing:
            self.conn.execute("DROP INDEX IF EXISTS chunks_hnsw")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
        )
        self.m, self.efc = m, ef_construction

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, entity_ids: dict[str, list[str]]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (chunk_id, document_id, company, section_path, date, ordinal, text,
                                       entity_ids, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (chunk_id) DO UPDATE SET document_id = EXCLUDED.document_id,
                     company = EXCLUDED.company, section_path = EXCLUDED.section_path, date = EXCLUDED.date,
                     ordinal = EXCLUDED.ordinal, text = EXCLUDED.text, entity_ids = EXCLUDED.entity_ids,
                     embedding = EXCLUDED.embedding""",
                [(c.chunk_id, c.document_id, c.company, c.section_path, c.date, c.ordinal, c.text,
                  sorted(entity_ids.get(c.chunk_id, [])), vectors[i]) for i, c in enumerate(chunks)],
            )

    def search(self, qvec: np.ndarray, k: int, ef_search: int | None = None, exact: bool = False,
               company: str | None = None) -> list[Passage]:
        with self.conn.transaction(), self.conn.cursor() as cur:
            if exact:
                cur.execute("SET LOCAL enable_indexscan = off")
            else:
                cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search or self.ef_search)}")
            where, args = ("WHERE company = %s", [company]) if company else ("", [])
            cur.execute(
                f"SELECT chunk_id, text, company, section_path, 1 - (embedding <=> %s) AS score "
                f"FROM chunks {where} ORDER BY embedding <=> %s LIMIT %s",
                [qvec, *args, qvec, k],
            )
            return [Passage(chunk_id=r[0], text=r[1], company=r[2], section_path=r[3], score=float(r[4]))
                    for r in cur.fetchall()]

    def get(self, chunk_ids: list[str]) -> list[Passage]:
        if not chunk_ids:
            return []
        rows = self.conn.execute(
            "SELECT chunk_id, text, company, section_path FROM chunks WHERE chunk_id = ANY(%s)", [chunk_ids]
        ).fetchall()
        by = {r[0]: Passage(chunk_id=r[0], text=r[1], company=r[2], section_path=r[3], score=1.0) for r in rows}
        return [by[c] for c in chunk_ids if c in by]

    def chunk_ids(self) -> set[str]:
        return {r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks").fetchall()}

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT chunk_id, document_id, company, section_path, date, ordinal, text FROM chunks"
        ).fetchall()
        return [Chunk(chunk_id=r[0], document_id=r[1], company=r[2], section_path=r[3], date=r[4] or "",
                      ordinal=r[5], text=r[6]) for r in rows]


class MemoryVectorStore:
    """Exact cosine search with numpy. Persisted to disk for offline runs."""

    def __init__(self, path: Path | None = None, dim: int = 384) -> None:
        self.path = path
        self.dim = dim
        self.chunks: dict[str, Chunk] = {}
        self.entity_ids: dict[str, list[str]] = {}
        self.vecs: dict[str, np.ndarray] = {}
        if path and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            for row in d:
                c = Chunk.model_validate(row["chunk"])
                self.chunks[c.chunk_id] = c
                self.entity_ids[c.chunk_id] = row["entity_ids"]
                self.vecs[c.chunk_id] = np.array(row["vec"], dtype=np.float32)

    def setup(self) -> None:
        return None

    def close(self) -> None:
        return None

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, entity_ids: dict[str, list[str]]) -> None:
        for i, c in enumerate(chunks):
            self.chunks[c.chunk_id] = c
            self.vecs[c.chunk_id] = vectors[i].astype(np.float32)
            self.entity_ids[c.chunk_id] = sorted(entity_ids.get(c.chunk_id, []))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([
                {"chunk": c.model_dump(), "entity_ids": self.entity_ids[cid], "vec": self.vecs[cid].tolist()}
                for cid, c in self.chunks.items()
            ]), encoding="utf-8")

    def search(self, qvec: np.ndarray, k: int, ef_search: int | None = None, exact: bool = False,
               company: str | None = None) -> list[Passage]:
        ids = [cid for cid, c in self.chunks.items() if not company or c.company == company]
        if not ids:
            return []
        m = np.stack([self.vecs[i] for i in ids])
        sims = m @ qvec
        order = np.argsort(-sims)[:k]
        return [Passage(chunk_id=ids[i], text=self.chunks[ids[i]].text, company=self.chunks[ids[i]].company,
                        section_path=self.chunks[ids[i]].section_path, score=float(sims[i])) for i in order]

    def get(self, chunk_ids: list[str]) -> list[Passage]:
        return [Passage(chunk_id=c, text=self.chunks[c].text, company=self.chunks[c].company,
                        section_path=self.chunks[c].section_path, score=1.0) for c in chunk_ids if c in self.chunks]

    def chunk_ids(self) -> set[str]:
        return set(self.chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks.values())
