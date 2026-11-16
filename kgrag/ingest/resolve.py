"""Entity resolution (mandatory). "Acme Corp", "Acme Corporation", "ACME" -> one node.

Stages, per entity type:
  1. Deterministic normalization (case, punctuation, '&', leading 'the', legal suffixes for companies).
     Mentions with the same normalized key are one candidate cluster.
  2. Curated alias seeds (data/aliases.json) — for well-known abbreviations embeddings can't see
     (e.g. "TSMC").
  3. Acronym rule for companies ("AMD" <-> "Advanced Micro Devices").
  4. Embedding similarity between cluster names >= threshold (tuned; see kgrag.eval.tune_er),
     with type-specific guards (persons must share a surname).
Union-find makes the result order-independent; canonical choice is deterministic, so node IDs
are stable across re-runs. Threshold and merge rate are logged, and every merge is written to
data/logs/er_merges.jsonl for audit.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kgrag.embeddings import Embedder
from kgrag.log import JsonlLog, get_logger
from kgrag.models import ExtractionResult, ResolvedEdge, ResolvedNode
from kgrag.ontology import EDGE_KEY_PROPERTIES, SYMMETRIC, EntityType

log = get_logger(__name__)

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "plc",
    "ag", "sa", "nv", "bv", "gmbh", "lp", "llp", "holdings", "holding", "group", "the",
}
# Words that don't distinguish two companies ("Micron Technology" == "Micron"). Any OTHER extra token
# blocks an embedding merge ("Arm Holdings" != "ARM Investments", "Intel" != "Intel Foundry").
GENERIC_COMPANY_WORDS = {
    "technology", "technologies", "tech", "semiconductor", "semiconductors", "electronics", "systems",
    "international", "global", "industries", "industry", "manufacturing", "precision", "solutions",
    "networks", "communications", "platforms", "enterprises", "worldwide",
}
# Trailing product descriptors stripped during normalization ("EPYC processors" -> "epyc").
PRODUCT_DESCRIPTORS = {
    "platform", "platforms", "processor", "processors", "gpu", "gpus", "chip", "chips", "series", "family",
    "products", "product", "architecture", "line",
}
MIN_CONFIDENCE = 0.6


def normalize(name: str, entity_type: EntityType) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[’'`]s\b", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = s.split()
    if toks and toks[0] == "the":
        toks = toks[1:]
    if entity_type == EntityType.COMPANY:
        while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
            toks.pop()
    if entity_type == EntityType.PRODUCT:
        while len(toks) > 1 and toks[-1] in PRODUCT_DESCRIPTORS:
            toks.pop()
    return " ".join(toks)


def has_distinguishing_token(a_norm: str, b_norm: str) -> bool:
    """True if either name has a non-generic token with no close spelling match in the other."""
    from rapidfuzz import fuzz

    ta, tb = set(a_norm.split()), set(b_norm.split())
    for extra, other in ((ta - tb, tb), (tb - ta, ta)):
        for tok in extra:
            if tok in GENERIC_COMPANY_WORDS:
                continue
            if not any(fuzz.ratio(tok, o) >= 85 for o in other):
                return True
    return False


def acronym(name: str) -> str:
    toks = [t for t in re.sub(r"[^A-Za-z0-9 ]+", " ", name).split() if t.lower() not in {"and", "of", "the"}]
    return "".join(t[0] for t in toks).lower()


def is_acronym_pair(a_norm: str, a_raw: str, b_norm: str, b_raw: str) -> bool:
    for short, long_raw, long_norm in ((a_norm, b_raw, b_norm), (b_norm, a_raw, a_norm)):
        if " " in short or len(short) < 3 or " " not in long_norm:
            continue
        stripped = " ".join(t for t in long_norm.split() if t not in LEGAL_SUFFIXES)
        if short in (acronym(long_raw), acronym(stripped)):
            return True
    return False


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[max(ra, rb)] = min(ra, rb)
        return True


@dataclass
class _Cluster:
    entity_type: EntityType
    key: str
    surface: Counter[str] = field(default_factory=Counter)
    chunk_ids: set[str] = field(default_factory=set)


@dataclass
class ResolutionReport:
    threshold: float
    mentions: int
    normalized_keys: int
    nodes: int
    edges: int
    merges_by_rule: dict[str, int]
    dropped_low_confidence: int
    dropped_self_loops: int

    @property
    def merge_rate(self) -> float:
        """Fraction of distinct normalized names collapsed by stages 2-4."""
        return 0.0 if not self.normalized_keys else 1 - self.nodes / self.normalized_keys


def load_alias_seeds(path: Path) -> dict[str, list[list[str]]]:
    if not path.exists():
        return {}
    data: dict[str, list[list[str]]] = json.loads(path.read_text(encoding="utf-8"))
    return data  # {EntityType: [[name, alias, ...], ...]}


def should_merge(
    a_norm: str, a_raw: str, b_norm: str, b_raw: str, sim: float, entity_type: EntityType, threshold: float
) -> str | None:
    """Pairwise decision used both by the resolver and by threshold tuning. Returns rule name or None."""
    if a_norm == b_norm:
        return "normalized"
    if entity_type == EntityType.COMPANY and is_acronym_pair(a_norm, a_raw, b_norm, b_raw):
        return "acronym"
    if sim >= threshold:
        if entity_type == EntityType.PERSON and a_norm.split()[-1] != b_norm.split()[-1]:
            return None
        # numbers/years must agree (e.g. "H100" vs "H200", "Fiscal 2024" vs "Fiscal 2025")
        if re.findall(r"\d+", a_norm) != re.findall(r"\d+", b_norm):
            return None
        if entity_type == EntityType.COMPANY and has_distinguishing_token(a_norm, b_norm):
            return None
        return "embedding"
    return None


class EntityResolver:
    def __init__(self, embedder: Embedder, threshold: float, alias_path: Path, audit_path: Path) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.seeds = load_alias_seeds(alias_path)
        self.audit = JsonlLog(audit_path)

    def resolve(self, results: dict[str, ExtractionResult]) -> tuple[list[ResolvedNode], list[ResolvedEdge],
                                                                       ResolutionReport]:
        # ---- stage 1: collect mentions into normalized clusters
        clusters: dict[tuple[EntityType, str], _Cluster] = {}
        mentions = 0

        def add(t: EntityType, name: str, chunk_id: str) -> tuple[EntityType, str]:
            nonlocal mentions
            key = normalize(name, t)
            if not key:
                key = name.lower()
            c = clusters.setdefault((t, key), _Cluster(t, key))
            c.surface[" ".join(name.split())] += 1
            c.chunk_ids.add(chunk_id)
            mentions += 1
            return (t, key)

        for cid in sorted(results):
            for e in results[cid].entities:
                if e.confidence >= MIN_CONFIDENCE:
                    add(e.entity_type, e.canonical_name, cid)

        by_type: dict[EntityType, list[_Cluster]] = defaultdict(list)
        for (t, _k), c in sorted(clusters.items(), key=lambda kv: (kv[0][0].value, kv[0][1])):
            by_type[t].append(c)

        merges: Counter[str] = Counter()
        key_to_node: dict[tuple[EntityType, str], str] = {}
        nodes: list[ResolvedNode] = []

        for t, cl in by_type.items():
            n = len(cl)
            uf = UnionFind(n)
            idx = {c.key: i for i, c in enumerate(cl)}
            raws = [c.surface.most_common(1)[0][0] for c in cl]

            # ---- stage 2: curated alias seeds
            for group in self.seeds.get(t.value, []):
                keys = [normalize(g, t) for g in group]
                present = [idx[k] for k in keys if k in idx]
                for j in present[1:]:
                    if uf.union(present[0], j):
                        merges["seed"] += 1
                        self.audit.write({"rule": "seed", "type": t.value, "a": cl[present[0]].key, "b": cl[j].key})

            # ---- stages 3+4: acronym + embedding similarity
            if n > 1:
                vecs = self.embedder.embed(raws)
                sims = vecs @ vecs.T
                cand: set[tuple[int, int]] = {
                    (int(i), int(j)) for i, j in zip(*np.where(np.triu(sims, 1) >= self.threshold), strict=True)
                }
                if t == EntityType.COMPANY:  # acronym candidates via lookup instead of O(n^2) scan
                    single = {c.key: i for i, c in enumerate(cl) if " " not in c.key}
                    for i, c in enumerate(cl):
                        if " " in c.key:
                            stripped = " ".join(x for x in c.key.split() if x not in LEGAL_SUFFIXES)
                            for ac in {acronym(raws[i]), acronym(stripped)}:
                                hit = single.get(ac)
                                if hit is not None and hit != i:
                                    cand.add((min(i, hit), max(i, hit)))
                for i, j in sorted(cand):
                    rule = should_merge(cl[i].key, raws[i], cl[j].key, raws[j], float(sims[i, j]), t,
                                        self.threshold)
                    if rule and uf.union(i, j):
                        merges[rule] += 1
                        self.audit.write({"rule": rule, "type": t.value, "a": raws[i], "b": raws[j],
                                          "sim": round(float(sims[i, j]), 4), "threshold": self.threshold})

            groups: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                groups[uf.find(i)].append(i)
            for members in groups.values():
                surface: Counter[str] = Counter()
                chunk_ids: set[str] = set()
                for i in members:
                    surface.update(cl[i].surface)
                    chunk_ids |= cl[i].chunk_ids
                # canonical: most frequent surface form; ties -> longest, then alphabetical
                canonical = sorted(surface.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]
                node_id = f"{t.value}:{normalize(canonical, t) or canonical.lower()}"
                for i in members:
                    key_to_node[(t, cl[i].key)] = node_id
                nodes.append(ResolvedNode(
                    node_id=node_id, entity_type=t, canonical_name=canonical,
                    aliases=sorted(set(surface) - {canonical}), mention_count=sum(surface.values()),
                    chunk_ids=sorted(chunk_ids),
                ))

        # ---- edges
        edges: dict[str, ResolvedEdge] = {}
        low_conf = self_loops = 0
        for cid in sorted(results):
            for r in results[cid].relationships:
                if r.confidence < MIN_CONFIDENCE:
                    low_conf += 1
                    continue
                s = key_to_node.get((r.source_type, normalize(r.source_name, r.source_type) or r.source_name.lower()))
                o = key_to_node.get((r.target_type, normalize(r.target_name, r.target_type) or r.target_name.lower()))
                if not s or not o:
                    low_conf += 1  # endpoint was itself below confidence
                    continue
                if s == o:
                    self_loops += 1
                    continue
                if r.relationship_type in SYMMETRIC and o < s:
                    s, o = o, s
                props = r.properties.as_dict()
                qual = "|".join(props.get(k, "") for k in EDGE_KEY_PROPERTIES.get(r.relationship_type, ()))
                ek = hashlib.sha1(f"{s}|{r.relationship_type.value}|{o}|{qual}".encode()).hexdigest()[:20]
                existing = edges.get(ek)
                if existing is None:
                    edges[ek] = ResolvedEdge(edge_key=ek, source_id=s, target_id=o,
                                             relationship_type=r.relationship_type, source_chunk_ids=[cid],
                                             evidence=[r.evidence], confidence=r.confidence, properties=props)
                else:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
                        existing.evidence.append(r.evidence)
                    existing.confidence = max(existing.confidence, r.confidence)
                    for k, v in props.items():
                        existing.properties.setdefault(k, v)

        report = ResolutionReport(
            threshold=self.threshold, mentions=mentions, normalized_keys=len(clusters), nodes=len(nodes),
            edges=len(edges), merges_by_rule=dict(merges), dropped_low_confidence=low_conf,
            dropped_self_loops=self_loops,
        )
        log.info("er.done", threshold=self.threshold, mentions=mentions, normalized_keys=len(clusters),
                 nodes=len(nodes), merge_rate=round(report.merge_rate, 4), merges=dict(merges), edges=len(edges))
        nodes.sort(key=lambda n: n.node_id)
        return nodes, sorted(edges.values(), key=lambda e: e.edge_key), report


def duplicate_canonicals(nodes: list[ResolvedNode]) -> list[tuple[str, list[str]]]:
    """Acceptance helper: nodes whose normalized canonical names collide (should be empty)."""
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    for n in nodes:
        seen[(n.entity_type.value, normalize(n.canonical_name, n.entity_type))].append(n.node_id)
    return [(f"{k[0]}:{k[1]}", v) for k, v in seen.items() if len(v) > 1]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
