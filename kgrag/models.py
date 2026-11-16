"""Pydantic schemas shared across the pipeline. The extraction schema is the contract with the LLM."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kgrag.ontology import EDGE_PROPERTIES, EntityType, RelationType, is_valid_signature


class Chunk(BaseModel):
    """A unit of text shared by both stores. `chunk_id` is THE cross-store key."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    company: str
    section_path: str
    date: str  # filing date, ISO yyyy-mm-dd
    ordinal: int
    text: str


# ---------------------------------------------------------------- extraction contract


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    canonical_name: str = Field(min_length=1, max_length=200)
    source_chunk_id: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("canonical_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("canonical_name is blank")
        return v


class EdgeProperties(BaseModel):
    """Closed set of edge qualifiers (structured outputs forbid open-ended maps)."""

    model_config = ConfigDict(extra="forbid")

    role: str | None = None
    year: str | None = None
    amount: str | None = None
    value: str | None = None
    unit: str | None = None
    period: str | None = None
    share: str | None = None
    item: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v not in (None, "")}


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)
    source_type: EntityType
    relationship_type: RelationType
    target_name: str = Field(min_length=1)
    target_type: EntityType
    source_chunk_id: str
    evidence: str = Field(min_length=1, max_length=600, description="Verbatim supporting sentence.")
    confidence: float = Field(ge=0.0, le=1.0)
    properties: EdgeProperties = Field(default_factory=EdgeProperties)

    @model_validator(mode="after")
    def _check_signature(self) -> ExtractedRelation:
        if not is_valid_signature(self.relationship_type, self.source_type, self.target_type):
            raise ValueError(
                f"{self.relationship_type.value} does not allow "
                f"{self.source_type.value}->{self.target_type.value}"
            )
        allowed = set(EDGE_PROPERTIES.get(self.relationship_type, ()))
        unknown = set(self.properties.as_dict()) - allowed
        if unknown:
            raise ValueError(f"properties {sorted(unknown)} not allowed on {self.relationship_type.value}")
        return self


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelation] = Field(default_factory=list)

    def check_chunk_ids(self, expected_chunk_id: str) -> None:
        bad = [e.source_chunk_id for e in self.entities if e.source_chunk_id != expected_chunk_id]
        bad += [r.source_chunk_id for r in self.relationships if r.source_chunk_id != expected_chunk_id]
        if bad:
            raise ValueError(f"source_chunk_id must be exactly {expected_chunk_id!r}")

    def check_relation_endpoints(self) -> None:
        names = {(e.entity_type, e.canonical_name.lower()) for e in self.entities}
        for r in self.relationships:
            for t, n in ((r.source_type, r.source_name), (r.target_type, r.target_name)):
                if (t, n.lower()) not in names:
                    raise ValueError(f"relationship endpoint {t.value}:{n!r} is not in the entities list")


# ---------------------------------------------------------------- resolved graph objects


class ResolvedNode(BaseModel):
    node_id: str  # "<Type>:<normalized key>", stable across runs
    entity_type: EntityType
    canonical_name: str
    aliases: list[str]
    mention_count: int
    chunk_ids: list[str]


class ResolvedEdge(BaseModel):
    edge_key: str
    source_id: str
    target_id: str
    relationship_type: RelationType
    source_chunk_ids: list[str]
    evidence: list[str]
    confidence: float
    properties: dict[str, str] = Field(default_factory=dict)

    @property
    def source_chunk_id(self) -> str:
        return self.source_chunk_ids[0]


# ---------------------------------------------------------------- retrieval / answer


class Passage(BaseModel):
    chunk_id: str
    text: str
    company: str
    section_path: str
    score: float


class GraphPath(BaseModel):
    """One traversal result: an ordered list of (node, edge, node, ...) hops."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]  # each: rel_type, start, end, source_chunk_ids, evidence, properties
    aggregate: dict[str, Any] | None = None  # e.g. {"node_id": ..., "degree": 4} for AGGREGATE_DEGREE

    @property
    def chunk_ids(self) -> list[str]:
        out: list[str] = []
        for e in self.edges:
            for c in e.get("source_chunk_ids", []):
                if c not in out:
                    out.append(c)
        return out


class Citation(BaseModel):
    chunk_id: str
    quote: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    route_used: str
    graph_paths: list[list[str]] | None = None
    refused: bool = False
    latency_ms: float | None = None
    cost_usd: float | None = None
