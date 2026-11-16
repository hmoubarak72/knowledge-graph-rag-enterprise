"""Graph store. `Neo4jGraphStore` is the real one; `MemoryGraphStore` mirrors the template
semantics in Python for offline tests and dry runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from kgrag.log import get_logger
from kgrag.models import Chunk, GraphPath, ResolvedEdge, ResolvedNode
from kgrag.retrieval.templates import CYPHER, QueryType, TemplateParams, check_params

log = get_logger(__name__)


class GraphStore(Protocol):
    def setup(self) -> None: ...
    def write(self, nodes: list[ResolvedNode], edges: list[ResolvedEdge], chunks: list[Chunk]) -> None: ...
    def counts(self) -> dict[str, int]: ...
    def entities(self) -> list[dict[str, Any]]: ...
    def chunk_ids(self) -> set[str]: ...
    def edge_chunk_ids(self) -> set[str]: ...
    def entities_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]: ...
    def run(self, qt: QueryType, params: TemplateParams) -> list[GraphPath]: ...
    def close(self) -> None: ...


# =============================================================================== Neo4j


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        # Server notifications (cartesian-product hints, "type does not exist yet") are informational
        # and flood the logs before the graph is populated.
        self.driver = GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")

    def close(self) -> None:
        self.driver.close()

    def setup(self) -> None:
        stmts = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.aliases_text]",
        ]
        with self.driver.session() as s:
            for q in stmts:
                s.run(q)

    def write(self, nodes: list[ResolvedNode], edges: list[ResolvedEdge], chunks: list[Chunk]) -> None:
        """MERGE-only, batched. Re-running with the same input is a no-op on counts."""
        with self.driver.session() as s:
            for batch in _batches([c.model_dump(exclude={"text"}) for c in chunks], 500):
                s.run(
                    "UNWIND $rows AS row MERGE (c:Chunk {chunk_id: row.chunk_id}) "
                    "SET c.document_id = row.document_id, c.company = row.company, "
                    "c.section_path = row.section_path, c.date = row.date, c.ordinal = row.ordinal",
                    rows=batch,
                )
            # Type labels come from the EntityType enum (validated upstream), one statement per label.
            by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for n in nodes:
                by_type[n.entity_type.value].append({
                    "node_id": n.node_id, "name": n.canonical_name, "aliases": n.aliases,
                    "aliases_text": " | ".join(n.aliases), "mention_count": n.mention_count,
                    "chunk_ids": n.chunk_ids,
                })
            for label, rows in by_type.items():
                for batch in _batches(rows, 500):
                    s.run(
                        f"UNWIND $rows AS row MERGE (n:Entity {{node_id: row.node_id}}) SET n:`{label}`, "
                        "n.entity_type = $label, n.name = row.name, n.aliases = row.aliases, "
                        "n.aliases_text = row.aliases_text, n.mention_count = row.mention_count, "
                        "n.chunk_ids = row.chunk_ids "
                        "WITH n, row UNWIND row.chunk_ids AS cid MATCH (c:Chunk {chunk_id: cid}) "
                        "MERGE (n)-[:MENTIONED_IN]->(c)",
                        rows=batch, label=label,
                    )
            by_rel: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for e in edges:
                by_rel[e.relationship_type.value].append({
                    "edge_key": e.edge_key, "s": e.source_id, "t": e.target_id,
                    "source_chunk_ids": e.source_chunk_ids, "source_chunk_id": e.source_chunk_id,
                    "evidence": e.evidence, "confidence": e.confidence, "props": e.properties,
                })
            for rel, rows in by_rel.items():
                for batch in _batches(rows, 500):
                    s.run(
                        "UNWIND $rows AS row MATCH (a:Entity {node_id: row.s}), (b:Entity {node_id: row.t}) "
                        f"MERGE (a)-[r:`{rel}` {{edge_key: row.edge_key}}]->(b) "
                        "SET r.source_chunk_ids = row.source_chunk_ids, r.source_chunk_id = row.source_chunk_id, "
                        "r.evidence = row.evidence, r.confidence = row.confidence, r += row.props",
                        rows=batch,
                    )

    def counts(self) -> dict[str, int]:
        def one(q: str) -> int:
            rec = s.run(q).single()
            return int(rec["c"]) if rec else 0

        with self.driver.session() as s:
            return {"nodes": one("MATCH (n:Entity) RETURN count(n) AS c"),
                    "edges": one("MATCH (:Entity)-[r]->(:Entity) RETURN count(r) AS c"),
                    "chunks": one("MATCH (c:Chunk) RETURN count(c) AS c")}

    def entities(self) -> list[dict[str, Any]]:
        with self.driver.session() as s:
            return [dict(r) for r in s.run(
                "MATCH (n:Entity) RETURN n.node_id AS node_id, n.entity_type AS entity_type, "
                "n.name AS name, n.aliases AS aliases, n.mention_count AS mention_count"
            )]

    def chunk_ids(self) -> set[str]:
        with self.driver.session() as s:
            return {r["id"] for r in s.run("MATCH (c:Chunk) RETURN c.chunk_id AS id")}

    def edge_chunk_ids(self) -> set[str]:
        with self.driver.session() as s:
            return {r["id"] for r in s.run(
                "MATCH (:Entity)-[r]->(:Entity) UNWIND r.source_chunk_ids AS id RETURN DISTINCT id")}

    def entities_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        with self.driver.session() as s:
            rows = s.run("MATCH (n:Entity)-[:MENTIONED_IN]->(c:Chunk) WHERE c.chunk_id IN $ids "
                         "RETURN c.chunk_id AS cid, collect(n.node_id) AS ents", ids=chunk_ids)
            return {r["cid"]: r["ents"] for r in rows}

    def run(self, qt: QueryType, params: TemplateParams) -> list[GraphPath]:
        check_params(qt, params)
        with self.driver.session(default_access_mode="READ") as s:
            rows = s.run(CYPHER[qt], **params.driver_params())
            out = []
            for r in rows:
                edges = [_clean_edge(e) for e in r["edges"]]
                out.append(GraphPath(nodes=r["nodes"], edges=edges, aggregate=r["aggregate"]))
            return out


def _clean_edge(e: dict[str, Any]) -> dict[str, Any]:
    props = {k: v for k, v in e.items()
             if k not in {"rel_type", "start", "end", "source_chunk_ids", "source_chunk_id", "evidence",
                          "confidence", "edge_key"}}
    return {"rel_type": e["rel_type"], "start": e["start"], "end": e["end"],
            "source_chunk_ids": list(e.get("source_chunk_ids") or []), "evidence": list(e.get("evidence") or []),
            "confidence": e.get("confidence"), "properties": props}


def _batches(rows: list[Any], n: int) -> list[list[Any]]:
    return [rows[i:i + n] for i in range(0, len(rows), n)]


# =============================================================================== in-memory


class MemoryGraphStore:
    """Same semantics as the Cypher templates, in Python. Persisted as JSON for offline runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            self.nodes, self.edges, self.chunks = d["nodes"], d["edges"], d["chunks"]

    def setup(self) -> None:
        return None

    def close(self) -> None:
        return None

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"nodes": self.nodes, "edges": self.edges, "chunks": self.chunks}),
                                 encoding="utf-8")

    def write(self, nodes: list[ResolvedNode], edges: list[ResolvedEdge], chunks: list[Chunk]) -> None:
        for c in chunks:  # MERGE semantics: keyed upsert
            self.chunks[c.chunk_id] = c.model_dump(exclude={"text"})
        for n in nodes:
            self.nodes[n.node_id] = {"node_id": n.node_id, "entity_type": n.entity_type.value,
                                     "name": n.canonical_name, "aliases": n.aliases,
                                     "mention_count": n.mention_count, "chunk_ids": n.chunk_ids}
        for e in edges:
            self.edges[e.edge_key] = {"rel_type": e.relationship_type.value, "start": e.source_id,
                                      "end": e.target_id, "source_chunk_ids": e.source_chunk_ids,
                                      "evidence": e.evidence, "confidence": e.confidence,
                                      "properties": e.properties}
        self.save()

    def counts(self) -> dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges), "chunks": len(self.chunks)}

    def entities(self) -> list[dict[str, Any]]:
        return [{k: n[k] for k in ("node_id", "entity_type", "name", "aliases", "mention_count")}
                for n in self.nodes.values()]

    def chunk_ids(self) -> set[str]:
        return set(self.chunks)

    def edge_chunk_ids(self) -> set[str]:
        return {c for e in self.edges.values() for c in e["source_chunk_ids"]}

    def entities_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        want = set(chunk_ids)
        out: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes.values():
            for c in n["chunk_ids"]:
                if c in want:
                    out[c].append(n["node_id"])
        return dict(out)

    # ---------------------------------------------------------------- traversal helpers
    def _adj(self) -> dict[str, list[dict[str, Any]]]:
        adj: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in self.edges.values():
            adj[e["start"]].append(e)
            adj[e["end"]].append(e)
        return adj

    @staticmethod
    def _ok(e: dict[str, Any], anchor: str, rels: list[str], d: str) -> bool:
        if rels and e["rel_type"] not in rels:
            return False
        return d == "any" or (d == "out" and e["start"] == anchor) or (d == "in" and e["end"] == anchor)

    @staticmethod
    def _other(e: dict[str, Any], x: str) -> str:
        return str(e["end"] if e["start"] == x else e["start"])

    def _path(self, ids: list[str], edges: list[dict[str, Any]], agg: dict[str, Any] | None = None) -> GraphPath:
        nodes = [{k: self.nodes[i][k] for k in ("node_id", "name", "entity_type")} for i in ids]
        return GraphPath(nodes=nodes, edges=[dict(e) for e in edges], aggregate=agg)

    def run(self, qt: QueryType, params: TemplateParams) -> list[GraphPath]:  # noqa: C901
        check_params(qt, params)
        p = params.driver_params()
        adj = self._adj()
        rels = p["rel_types"]
        lim = int(p["limit"])
        out: list[GraphPath] = []
        a = str(p["entity_id"])

        if qt in (QueryType.NEIGHBORHOOD, QueryType.RELATION_LOOKUP):
            use = rels if qt == QueryType.RELATION_LOOKUP else []
            d = p["direction"] if qt == QueryType.RELATION_LOOKUP else "any"
            es = sorted((e for e in adj.get(a, []) if self._ok(e, a, use, str(d))),
                        key=lambda e: -e["confidence"])
            out = [self._path([a, self._other(e, a)], [e]) for e in es[:lim]]

        elif qt == QueryType.SHARED_NEIGHBORS:
            anchors: list[str] = p["entity_ids"]
            linked: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for x in anchors:
                for e in adj.get(x, []):
                    b = self._other(e, x)
                    if b not in anchors and self._ok(e, x, rels, str(p["direction"])):
                        linked[b].setdefault(x, e)
            for b, by_anchor in sorted(linked.items()):
                if len(by_anchor) == len(anchors):
                    for x, e in by_anchor.items():
                        out.append(self._path([x, b], [e]))
            out = out[:lim]

        elif qt in (QueryType.TWO_HOP, QueryType.THREE_HOP):
            hops = [(rels, p["direction"]), (p["rel_types_2"], p["direction_2"])]
            if qt == QueryType.THREE_HOP:
                hops.append((p["rel_types_3"], p["direction_3"]))
            frontier: list[tuple[list[str], list[dict[str, Any]]]] = [([a], [])]
            for hr, hd in hops:
                nxt = []
                for ids, es in frontier:
                    cur = ids[-1]
                    for e in adj.get(cur, []):
                        b = self._other(e, cur)
                        if b in ids or not self._ok(e, cur, hr, str(hd)):
                            continue
                        nxt.append((ids + [b], es + [e]))
                frontier = nxt
            frontier.sort(key=lambda f: -_prod(e["confidence"] for e in f[1]))
            out = [self._path(ids, es) for ids, es in frontier[:lim]]

        elif qt == QueryType.PATH_BETWEEN:
            b = str(p["b_id"])
            # BFS for all shortest paths <= 3 hops
            layer: list[tuple[list[str], list[dict[str, Any]]]] = [([a], [])]
            for _ in range(3):
                nxt_layer: list[tuple[list[str], list[dict[str, Any]]]] = []
                found: list[tuple[list[str], list[dict[str, Any]]]] = []
                for ids, es in layer:
                    for e in adj.get(ids[-1], []):
                        o = self._other(e, ids[-1])
                        if o in ids:
                            continue
                        (found if o == b else nxt_layer).append((ids + [o], es + [e]))
                if found:
                    out = [self._path(i, e) for i, e in found[:lim]]
                    break
                layer = nxt_layer

        elif qt == QueryType.AGGREGATE_DEGREE:
            restrict = set(p["entity_ids"])
            et = p["entity_type"]
            deg: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for e in self.edges.values():
                for x in (e["start"], e["end"]):
                    if et and self.nodes[x]["entity_type"] != et:
                        continue
                    if not self._ok(e, x, rels, str(p["direction"])):
                        continue
                    b = self._other(e, x)
                    if restrict and b not in restrict:
                        continue
                    deg[x].setdefault(b, e)
            top = sorted(deg.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
            for x, nb in top:
                for b, e in nb.items():
                    out.append(self._path([x, b], [e], {"node_id": x, "degree": len(nb)}))
            out = out[:lim]

        elif qt == QueryType.METRIC_COMPARE:
            src_ids = set(p["entity_ids"])
            mids = set(p["metric_ids"])
            period = str(p["period"]).lower()
            for e in self.edges.values():
                if e["rel_type"] != "REPORTS" or e["start"] not in src_ids:
                    continue
                if mids and e["end"] not in mids:
                    continue
                if period and period not in str(e["properties"].get("period", "")).lower():
                    continue
                out.append(self._path([e["start"], e["end"]], [e]))
            out = out[:lim]
        return out


def _prod(xs: Any) -> float:
    r = 1.0
    for x in xs:
        r *= float(x)
    return r
