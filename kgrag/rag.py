"""Orchestration: HybridRAG (router -> graph and/or vector -> synthesis) and the VectorOnlyRAG baseline.

Both systems share the same embedder, vector store, synthesis prompt, model and citation validator,
so the benchmark isolates the effect of the graph path.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kgrag.config import Settings
from kgrag.embeddings import Embedder
from kgrag.llm import LLMClient
from kgrag.log import JsonlLog, get_logger
from kgrag.models import AskResponse, GraphPath, Passage
from kgrag.ontology import EntityType
from kgrag.retrieval.linker import EntityLinker, LinkedEntity
from kgrag.retrieval.router import Route, RouteDecision, Router
from kgrag.retrieval.templates import QueryType, TemplateParams, check_params
from kgrag.stores.graph import GraphStore
from kgrag.stores.vector import VectorStore
from kgrag.synthesis import Synthesizer, build_context

log = get_logger(__name__)


@dataclass
class Trace:
    request_id: str
    route: str = ""
    router_confidence: float | None = None
    low_confidence_fallback: bool = False
    query_type: str | None = None
    linked: list[dict[str, Any]] = field(default_factory=list)
    unlinked: list[str] = field(default_factory=list)
    template_error: str | None = None
    graph_fallback: str | None = None
    n_paths: int = 0
    n_passages: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class _Base:
    def __init__(self, settings: Settings, llm: LLMClient, embedder: Embedder, vectors: VectorStore) -> None:
        self.settings, self.llm, self.embedder, self.vectors = settings, llm, embedder, vectors
        self.synth = Synthesizer(llm, settings)

    def _vector_search(self, question: str) -> list[Passage]:
        return self.vectors.search(self.embedder.embed_query(question), self.settings.vector_top_k)


class VectorOnlyRAG(_Base):
    name = "vector_only"

    def ask(self, question: str) -> tuple[AskResponse, Trace]:
        t0 = time.perf_counter()
        spent0 = self.llm.spent()
        tr = Trace(uuid.uuid4().hex[:12], route="VECTOR")
        passages = self._vector_search(question)
        ctx = build_context([], passages, [])
        res = self.synth.answer(question, ctx)
        tr.n_passages = len(ctx.passages)
        tr.latency_ms = (time.perf_counter() - t0) * 1000
        tr.cost_usd = self.llm.spent() - spent0
        return AskResponse(answer=res.answer, citations=res.citations, route_used="VECTOR", graph_paths=None,
                           refused=res.refused, latency_ms=round(tr.latency_ms, 1),
                           cost_usd=round(tr.cost_usd, 6)), tr


class HybridRAG(_Base):
    name = "hybrid"

    def __init__(self, settings: Settings, llm: LLMClient, embedder: Embedder, vectors: VectorStore,
                 graph: GraphStore) -> None:
        super().__init__(settings, llm, embedder, vectors)
        self.graph = graph
        self.router = Router(llm, settings)
        self.linker = EntityLinker(graph.entities(), embedder)
        self.route_log = JsonlLog(settings.logs_dir / "routing.jsonl")

    # ---------------------------------------------------------------- graph path
    def _params(self, d: RouteDecision, tr: Trace) -> tuple[QueryType, TemplateParams, list[LinkedEntity]]:
        linked: list[LinkedEntity] = []
        for m in d.mentions:
            le = self.linker.link(m.text, m.entity_type.value if m.entity_type else None)
            if le is None and m.entity_type:  # type hint may be wrong; retry untyped
                le = self.linker.link(m.text)
            if le:
                linked.append(le)
            else:
                tr.unlinked.append(m.text)
        metric_ids = []
        for mt in d.metrics:
            le = self.linker.link(mt, EntityType.FINANCIAL_METRIC.value)
            if le:
                metric_ids.append(le.node_id)
        ids = list(dict.fromkeys(le.node_id for le in linked if le.entity_type != "FinancialMetric"))
        tr.linked = [vars(le) for le in linked]
        hops = d.hops + [type(d.hops[0])(rel_types=[], direction="any")] * 3 if d.hops else []
        qt = d.query_type or (QueryType.PATH_BETWEEN if len(ids) >= 2 else QueryType.NEIGHBORHOOD)
        p = TemplateParams(
            entity_ids=ids[:5],
            rel_types=hops[0].rel_types if hops else [], direction=hops[0].direction if hops else "any",
            rel_types_2=hops[1].rel_types if hops else [], direction_2=hops[1].direction if hops else "any",
            rel_types_3=hops[2].rel_types if hops else [], direction_3=hops[2].direction if hops else "any",
            entity_type=d.target_entity_type if qt == QueryType.AGGREGATE_DEGREE else None,
            metric_ids=metric_ids, period=d.period,
        )
        return qt, p, linked

    def _graph(self, d: RouteDecision, tr: Trace, passages: list[Passage]) -> list[GraphPath]:
        qt, p, _ = self._params(d, tr)
        tr.query_type = qt.value
        paths: list[GraphPath] = []
        try:
            check_params(qt, p)
            paths = self.graph.run(qt, p)
        except ValueError as err:
            tr.template_error = str(err)
        if not paths:
            anchors = p.entity_ids
            if not anchors and passages:  # passage -> graph neighbourhood via the shared chunk id
                ents = self.graph.entities_for_chunks([x.chunk_id for x in passages[:3]])
                freq = Counter(e for es in ents.values() for e in es if e.startswith("Company:"))
                anchors = [e for e, _ in freq.most_common(1)]
            for a in anchors[:2]:
                paths += self.graph.run(QueryType.NEIGHBORHOOD, TemplateParams(entity_ids=[a], limit=15))
            if paths:
                tr.graph_fallback = "NEIGHBORHOOD"
        return paths

    # ---------------------------------------------------------------- main
    def ask(self, question: str) -> tuple[AskResponse, Trace]:
        t0 = time.perf_counter()
        spent0 = self.llm.spent()
        tr = Trace(uuid.uuid4().hex[:12])
        d, fallback, _ = self.router.route(question)
        tr.route, tr.router_confidence, tr.low_confidence_fallback = d.route.value, d.confidence, fallback

        passages: list[Passage] = []
        paths: list[GraphPath] = []
        if d.route in (Route.VECTOR, Route.BOTH):
            passages = self._vector_search(question)
        if d.route in (Route.GRAPH, Route.BOTH):
            paths = self._graph(d, tr, passages)
            if d.route == Route.GRAPH and not paths:  # graph came back empty: don't answer blind
                passages = self._vector_search(question)
                tr.graph_fallback = "VECTOR"

        evidence_ids = list(dict.fromkeys(c for gp in paths for c in gp.chunk_ids))[:12]
        evidence = self.vectors.get(evidence_ids)  # graph path -> original text (shared chunk id)
        ctx = build_context(paths, passages, evidence)
        res = self.synth.answer(question, ctx)

        tr.n_paths, tr.n_passages = len(paths), len(ctx.passages)
        tr.latency_ms = (time.perf_counter() - t0) * 1000
        tr.cost_usd = self.llm.spent() - spent0
        graph_paths = [[_fmt_hop(gp, i) for i in range(len(gp.edges))] for gp in paths[:20]] or None
        resp = AskResponse(answer=res.answer, citations=res.citations, route_used=d.route.value,
                           graph_paths=graph_paths, refused=res.refused, latency_ms=round(tr.latency_ms, 1),
                           cost_usd=round(tr.cost_usd, 6))
        self.route_log.write({  # Phase 3: every routing decision + outcome, persisted
            "request_id": tr.request_id, "question": question, "route": d.route.value,
            "confidence": d.confidence, "low_confidence_fallback": fallback,
            "query_type": tr.query_type, "decision": d.model_dump(mode="json"), "linked": tr.linked,
            "unlinked": tr.unlinked, "template_error": tr.template_error, "graph_fallback": tr.graph_fallback,
            "outcome": {"n_paths": tr.n_paths, "n_passages": tr.n_passages, "refused": res.refused,
                        "refusal_reason": res.refusal_reason,
                        "citation_validation_failed": res.citation_validation_failed,
                        "n_citations": len(res.citations), "latency_ms": round(tr.latency_ms, 1),
                        "cost_usd": round(tr.cost_usd, 6)},
        })
        return resp, tr


def _fmt_hop(gp: GraphPath, i: int) -> str:
    names = {n["node_id"]: n["name"] for n in gp.nodes}
    e = gp.edges[i]
    return f"({names.get(e['start'], e['start'])})-[{e['rel_type']}]->({names.get(e['end'], e['end'])})"
