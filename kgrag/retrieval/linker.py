"""Resolve entity mentions in a question to graph node IDs (deterministic, no LLM).

Order: exact normalized name/alias -> fuzzy string (RapidFuzz token_set_ratio) -> embedding cosine.
Uses the same `normalize` as ingestion, so "Taiwan Semiconductor" and "TSMC" land on the same node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process

from kgrag.embeddings import Embedder
from kgrag.ingest.resolve import normalize
from kgrag.ontology import EntityType


@dataclass
class LinkedEntity:
    mention: str
    node_id: str
    name: str
    entity_type: str
    score: float
    method: str


class EntityLinker:
    def __init__(self, entities: list[dict[str, Any]], embedder: Embedder,
                 fuzzy_min: float = 88.0, embed_min: float = 0.82) -> None:
        self.entities = {e["node_id"]: e for e in entities}
        self.embedder = embedder
        self.fuzzy_min, self.embed_min = fuzzy_min, embed_min
        self.surface: list[str] = []
        self.surface_node: list[str] = []
        self.exact: dict[tuple[str, str], str] = {}
        for e in sorted(entities, key=lambda e: -(e.get("mention_count") or 0)):
            t = EntityType(e["entity_type"])
            for name in [e["name"], *(e.get("aliases") or [])]:
                self.exact.setdefault((t.value, normalize(name, t)), e["node_id"])
                self.exact.setdefault(("*", normalize(name, t)), e["node_id"])
                self.surface.append(name)
                self.surface_node.append(e["node_id"])
        self._vecs: np.ndarray | None = None

    def _embeddings(self) -> np.ndarray:
        if self._vecs is None:
            self._vecs = self.embedder.embed(self.surface) if self.surface else np.zeros((0, 1))
        return self._vecs

    def link(self, mention: str, type_hint: str | None = None) -> LinkedEntity | None:
        types = [type_hint] if type_hint else []
        for t in types + ["*"]:
            et = EntityType(t) if t != "*" else EntityType.COMPANY
            nid = self.exact.get((t, normalize(mention, et)))
            if nid:
                return self._mk(mention, nid, 1.0, "exact")
        if not self.surface:
            return None
        allowed = [i for i, n in enumerate(self.surface_node)
                   if not type_hint or self.entities[n]["entity_type"] == type_hint]
        choices = {i: self.surface[i] for i in allowed}
        best = process.extractOne(mention, choices, scorer=fuzz.token_set_ratio)
        if best and best[1] >= self.fuzzy_min and min(len(mention), len(best[0])) >= 3:
            return self._mk(mention, self.surface_node[best[2]], best[1] / 100, "fuzzy")
        vecs = self._embeddings()
        q = self.embedder.embed_query(mention)
        sims = vecs[allowed] @ q if allowed else np.array([])
        if sims.size:
            j = int(np.argmax(sims))
            if sims[j] >= self.embed_min:
                return self._mk(mention, self.surface_node[allowed[j]], float(sims[j]), "embedding")
        return None

    def _mk(self, mention: str, nid: str, score: float, method: str) -> LinkedEntity:
        e = self.entities[nid]
        return LinkedEntity(mention, nid, e["name"], e["entity_type"], score, method)
