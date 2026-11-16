"""Curated, parameterized Cypher template library — the ONLY Cypher that reaches the database.

The model never emits Cypher. The router chooses a `QueryType` and fills slots; the slots are
validated here (`TemplateParams`) against the ontology enums and resolved node IDs, and passed as
driver parameters. Relationship types are never string-interpolated: they are filtered with
`type(r) IN $rel_types`, and the values are checked against `RelationType` first.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from kgrag.ontology import EntityType, RelationType

Direction = Literal["out", "in", "any"]


class QueryType(StrEnum):
    NEIGHBORHOOD = "NEIGHBORHOOD"            # everything directly connected to one entity
    RELATION_LOOKUP = "RELATION_LOOKUP"      # X -[REL]-> ?   /   ? -[REL]-> X
    SHARED_NEIGHBORS = "SHARED_NEIGHBORS"    # ? connected to ALL of X, Y, (Z) by REL
    TWO_HOP = "TWO_HOP"                      # X -[R1]- ? -[R2]- ?
    THREE_HOP = "THREE_HOP"                  # X -[R1]- ? -[R2]- ? -[R3]- ?
    PATH_BETWEEN = "PATH_BETWEEN"            # how are X and Y connected (<= 3 hops)
    AGGREGATE_DEGREE = "AGGREGATE_DEGREE"    # which entity has the most REL edges
    METRIC_COMPARE = "METRIC_COMPARE"        # REPORTS values across companies


class TemplateParams(BaseModel):
    """Validated slot values. Built from router output + entity linking, never from raw text."""

    entity_ids: list[str] = Field(default_factory=list, max_length=5)
    rel_types: list[RelationType] = Field(default_factory=list)
    rel_types_2: list[RelationType] = Field(default_factory=list)
    rel_types_3: list[RelationType] = Field(default_factory=list)
    direction: Direction = "any"
    direction_2: Direction = "any"
    direction_3: Direction = "any"
    entity_type: EntityType | None = None
    metric_ids: list[str] = Field(default_factory=list)
    period: str = ""
    limit: int = Field(default=25, ge=1, le=100)

    @field_validator("entity_ids", "metric_ids")
    @classmethod
    def _ids_look_like_nodes(cls, v: list[str]) -> list[str]:
        for x in v:
            t = x.split(":", 1)[0]
            if t not in {e.value for e in EntityType}:
                raise ValueError(f"not a resolved node id: {x!r}")
        return v

    def driver_params(self) -> dict[str, Any]:
        return {
            "entity_ids": self.entity_ids,
            "entity_id": self.entity_ids[0] if self.entity_ids else "",
            "a_id": self.entity_ids[0] if self.entity_ids else "",
            "b_id": self.entity_ids[1] if len(self.entity_ids) > 1 else "",
            "rel_types": [r.value for r in self.rel_types],
            "rel_types_2": [r.value for r in self.rel_types_2],
            "rel_types_3": [r.value for r in self.rel_types_3],
            "direction": self.direction, "direction_2": self.direction_2, "direction_3": self.direction_3,
            "entity_type": self.entity_type.value if self.entity_type else "",
            "metric_ids": self.metric_ids,
            "period": self.period,
            "limit": self.limit,
        }


REQUIRED: dict[QueryType, tuple[int, bool]] = {
    # (min entity_ids, rel_types required)
    QueryType.NEIGHBORHOOD: (1, False),
    QueryType.RELATION_LOOKUP: (1, True),
    QueryType.SHARED_NEIGHBORS: (2, True),
    QueryType.TWO_HOP: (1, True),
    QueryType.THREE_HOP: (1, True),
    QueryType.PATH_BETWEEN: (2, False),
    QueryType.AGGREGATE_DEGREE: (0, True),
    QueryType.METRIC_COMPARE: (1, False),
}


def check_params(qt: QueryType, p: TemplateParams) -> None:
    n, needs_rel = REQUIRED[qt]
    if len(p.entity_ids) < n:
        raise ValueError(f"{qt.value} needs >= {n} resolved entities, got {len(p.entity_ids)}")
    if needs_rel and not p.rel_types:
        raise ValueError(f"{qt.value} needs rel_types")
    if qt == QueryType.TWO_HOP and not p.rel_types_2:
        raise ValueError("TWO_HOP needs rel_types_2")
    if qt == QueryType.THREE_HOP and not (p.rel_types_2 and p.rel_types_3):
        raise ValueError("THREE_HOP needs rel_types_2 and rel_types_3")


def _dir(r: str, anchor: str, d: str) -> str:
    return (f"(${d} = 'any' OR (${d} = 'out' AND startNode({r}) = {anchor}) "
            f"OR (${d} = 'in' AND endNode({r}) = {anchor}))")


_RETURN = """
RETURN [n IN nodes(p) | {node_id: n.node_id, name: n.name, entity_type: n.entity_type}] AS nodes,
       [x IN relationships(p) | x{.*, rel_type: type(x), start: startNode(x).node_id,
                                  end: endNode(x).node_id}] AS edges"""

CYPHER: dict[QueryType, str] = {
    QueryType.NEIGHBORHOOD: f"""
MATCH p = (a:Entity {{node_id: $entity_id}})-[r]-(b:Entity)
WITH p, r ORDER BY r.confidence DESC, b.mention_count DESC LIMIT $limit
{_RETURN}, null AS aggregate""",

    QueryType.RELATION_LOOKUP: f"""
MATCH p = (a:Entity {{node_id: $entity_id}})-[r]-(b:Entity)
WHERE type(r) IN $rel_types AND {_dir('r', 'a', 'direction')}
WITH p, r ORDER BY r.confidence DESC LIMIT $limit
{_RETURN}, null AS aggregate""",

    QueryType.SHARED_NEIGHBORS: f"""
MATCH (a:Entity)-[r]-(b:Entity)
WHERE a.node_id IN $entity_ids AND NOT b.node_id IN $entity_ids
  AND type(r) IN $rel_types AND {_dir('r', 'a', 'direction')}
WITH b, collect(DISTINCT a.node_id) AS linked
WHERE size(linked) = size($entity_ids)
MATCH p = (a:Entity)-[r]-(b)
WHERE a.node_id IN $entity_ids AND type(r) IN $rel_types AND {_dir('r', 'a', 'direction')}
WITH p LIMIT $limit
{_RETURN}, null AS aggregate""",

    QueryType.TWO_HOP: f"""
MATCH p = (a:Entity {{node_id: $entity_id}})-[r1]-(b:Entity)-[r2]-(c:Entity)
WHERE type(r1) IN $rel_types AND type(r2) IN $rel_types_2 AND c <> a
  AND {_dir('r1', 'a', 'direction')} AND {_dir('r2', 'b', 'direction_2')}
WITH p, r1, r2 ORDER BY r1.confidence * r2.confidence DESC LIMIT $limit
{_RETURN}, null AS aggregate""",

    QueryType.THREE_HOP: f"""
MATCH p = (a:Entity {{node_id: $entity_id}})-[r1]-(b:Entity)-[r2]-(c:Entity)-[r3]-(d:Entity)
WHERE type(r1) IN $rel_types AND type(r2) IN $rel_types_2 AND type(r3) IN $rel_types_3
  AND c <> a AND d <> b AND d <> a
  AND {_dir('r1', 'a', 'direction')} AND {_dir('r2', 'b', 'direction_2')} AND {_dir('r3', 'c', 'direction_3')}
WITH p, r1, r2, r3 ORDER BY r1.confidence * r2.confidence * r3.confidence DESC LIMIT $limit
{_RETURN}, null AS aggregate""",

    QueryType.PATH_BETWEEN: f"""
MATCH (a:Entity {{node_id: $a_id}}), (b:Entity {{node_id: $b_id}})
MATCH p = allShortestPaths((a)-[*..3]-(b))
WHERE all(n IN nodes(p) WHERE n:Entity)
WITH p LIMIT $limit
{_RETURN}, null AS aggregate""",

    QueryType.AGGREGATE_DEGREE: f"""
MATCH (a:Entity)-[r]-(b:Entity)
WHERE type(r) IN $rel_types AND ($entity_type = '' OR a.entity_type = $entity_type)
  AND {_dir('r', 'a', 'direction')}
  AND (size($entity_ids) = 0 OR b.node_id IN $entity_ids)
WITH a, count(DISTINCT b) AS degree ORDER BY degree DESC, a.node_id LIMIT 5
MATCH p = (a)-[r]-(b:Entity)
WHERE type(r) IN $rel_types AND {_dir('r', 'a', 'direction')}
  AND (size($entity_ids) = 0 OR b.node_id IN $entity_ids)
WITH p, a, degree LIMIT $limit
{_RETURN}, {{node_id: a.node_id, degree: degree}} AS aggregate""",

    QueryType.METRIC_COMPARE: f"""
MATCH p = (a:Entity)-[r:REPORTS]->(m:Entity)
WHERE a.node_id IN $entity_ids
  AND (size($metric_ids) = 0 OR m.node_id IN $metric_ids)
  AND ($period = '' OR toLower(coalesce(r.period, '')) CONTAINS toLower($period))
WITH p LIMIT $limit
{_RETURN}, null AS aggregate""",
}

DESCRIPTIONS: dict[QueryType, str] = {
    QueryType.NEIGHBORHOOD: "All facts directly attached to one entity (fallback / 'tell me about X's relationships').",
    QueryType.RELATION_LOOKUP: "One relationship from one entity, e.g. who supplies X (direction=in, SUPPLIES).",
    QueryType.SHARED_NEIGHBORS: "Entities linked to ALL of 2-5 entities by a relationship, e.g. suppliers X and Y share.",
    QueryType.TWO_HOP: "Two-step chain from one entity, e.g. CEO of the company X acquired (ACQUIRED out, EMPLOYS out).",
    QueryType.THREE_HOP: "Three-step chain from one entity.",
    QueryType.PATH_BETWEEN: "How two entities are connected (shortest paths up to 3 hops).",
    QueryType.AGGREGATE_DEGREE: "Rank entities by number of distinct neighbours over a relationship, e.g. most-shared supplier.",
    QueryType.METRIC_COMPARE: "Compare reported financial metrics across companies, optional period filter.",
}
