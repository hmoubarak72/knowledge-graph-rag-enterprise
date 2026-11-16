"""Phase 3 router: one cheap small-model call -> {route, confidence, query_type, slots}.

The model only *fills parameters*: a QueryType from the enum, entity mentions (later resolved by
the deterministic linker), relationship types from the ontology enum, and directions. It never
sees or writes Cypher. Low confidence -> BOTH (run both paths and merge).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kgrag.config import Settings
from kgrag.llm import CallStats, LLMClient
from kgrag.ontology import EntityType, RelationType
from kgrag.retrieval.templates import DESCRIPTIONS, QueryType


class Route(StrEnum):
    GRAPH = "GRAPH"
    VECTOR = "VECTOR"
    BOTH = "BOTH"


class Mention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    entity_type: EntityType | None = None


class Hop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rel_types: list[RelationType]
    direction: Literal["out", "in", "any"] = "any"


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route
    confidence: float = Field(ge=0, le=1)
    query_type: QueryType | None = None
    mentions: list[Mention] = Field(default_factory=list)
    hops: list[Hop] = Field(default_factory=list, max_length=3)
    target_entity_type: EntityType | None = None
    metrics: list[str] = Field(default_factory=list)
    period: str = ""
    rationale: str = Field(default="", max_length=300)


_TEMPLATES = "\n".join(f"- {q.value}: {d}" for q, d in DESCRIPTIONS.items())
_RELS = ", ".join(r.value for r in RelationType)

SYSTEM_PROMPT = f"""You route questions about SEC 10-K filings of semiconductor companies to a retrieval path.

Routes:
- GRAPH: connection questions, multi-hop chains, cross-entity comparisons, aggregations over relationships
  ("Which suppliers do both X and Y share?", "Who leads the company X acquired?", "Which supplier serves the most companies?").
- VECTOR: definitions, policy lookups, single-fact questions answerable from one passage
  ("What does NVIDIA say about climate risk?", "How does Intel define gross margin?").
- BOTH: you are unsure, or the question needs both a relationship and narrative detail.
Out-of-scope questions (unrelated to these filings) -> VECTOR with confidence >= 0.8; the answer stage refuses them.

For GRAPH/BOTH also fill the template slots. Template library (pick one query_type):
{_TEMPLATES}

Relationship types: {_RELS}
SUPPLIES points supplier -> customer. EMPLOYS company -> person. ACQUIRED acquirer -> acquiree.
OFFERS company -> product. REPORTS company -> metric. direction is relative to the entity at that hop
("out" = the entity is the source). `hops` has one entry per hop, in order.
mentions: entity names exactly as written in the question, with a type when obvious.

Examples:
Q: Which suppliers do both NVIDIA and AMD rely on?
A: {{"route":"GRAPH","confidence":0.93,"query_type":"SHARED_NEIGHBORS","mentions":[{{"text":"NVIDIA","entity_type":"Company"}},{{"text":"AMD","entity_type":"Company"}}],"hops":[{{"rel_types":["SUPPLIES"],"direction":"in"}}],"target_entity_type":"Company","metrics":[],"period":"","rationale":"shared suppliers across two entities"}}
Q: Who is the CEO of the company that acquired VMware?
A: {{"route":"GRAPH","confidence":0.9,"query_type":"TWO_HOP","mentions":[{{"text":"VMware","entity_type":"Company"}}],"hops":[{{"rel_types":["ACQUIRED"],"direction":"in"}},{{"rel_types":["EMPLOYS"],"direction":"out"}}],"target_entity_type":"Person","metrics":[],"period":"","rationale":"acquirer then its executive"}}
Q: What is Qualcomm's policy on human capital and diversity?
A: {{"route":"VECTOR","confidence":0.9,"query_type":null,"mentions":[{{"text":"Qualcomm","entity_type":"Company"}}],"hops":[],"target_entity_type":null,"metrics":[],"period":"","rationale":"policy lookup"}}
Q: Which foundry supplies the most companies in the corpus?
A: {{"route":"GRAPH","confidence":0.88,"query_type":"AGGREGATE_DEGREE","mentions":[],"hops":[{{"rel_types":["SUPPLIES"],"direction":"out"}}],"target_entity_type":"Company","metrics":[],"period":"","rationale":"aggregation over SUPPLIES"}}
Q: Compare R&D expense of Intel and AMD in fiscal 2024.
A: {{"route":"BOTH","confidence":0.75,"query_type":"METRIC_COMPARE","mentions":[{{"text":"Intel","entity_type":"Company"}},{{"text":"AMD","entity_type":"Company"}}],"hops":[],"target_entity_type":"FinancialMetric","metrics":["Research and development expense"],"period":"2024","rationale":"metric comparison, values also in MD&A text"}}
Q: What will NVIDIA's share price be next year?
A: {{"route":"VECTOR","confidence":0.85,"query_type":null,"mentions":[{{"text":"NVIDIA","entity_type":"Company"}}],"hops":[],"target_entity_type":null,"metrics":[],"period":"","rationale":"speculative, out of scope; answer stage should refuse"}}
"""


class Router:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self.llm, self.settings = llm, settings

    def route(self, question: str) -> tuple[RouteDecision, bool, CallStats]:
        """Returns (decision, low_confidence_fallback_applied, stats)."""
        d, stats = self.llm.structured(
            model=self.settings.router_model, system=SYSTEM_PROMPT, user=f"Q: {question}\nA:",
            output=RouteDecision, purpose="router", max_tokens=1500, expected_output_tokens=200, effort="low",
        )
        fallback = False
        if d.confidence < self.settings.router_confidence and d.route != Route.BOTH:
            d = d.model_copy(update={"route": Route.BOTH})
            fallback = True
        return d, fallback, stats
