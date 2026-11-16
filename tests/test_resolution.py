"""Entity resolution: the Acme trio collapses, near-misses don't, and results are order-independent."""

from __future__ import annotations

import random
from pathlib import Path

from kgrag.embeddings import HashEmbedder
from kgrag.ingest.resolve import EntityResolver, duplicate_canonicals, normalize, should_merge
from kgrag.models import ExtractionResult
from kgrag.ontology import EntityType

ROOT = Path(__file__).resolve().parent.parent


def ents(cid: str, *names: tuple[str, str]) -> ExtractionResult:
    return ExtractionResult.model_validate({"entities": [
        {"entity_type": t, "canonical_name": n, "source_chunk_id": cid, "confidence": 0.9} for n, t in names
    ], "relationships": []})


def resolver(tmp_path: Path, threshold: float = 0.92) -> EntityResolver:
    return EntityResolver(HashEmbedder(), threshold, ROOT / "data" / "aliases.json", tmp_path / "er.jsonl")


def test_normalize_strips_legal_suffixes_only_for_companies() -> None:
    assert normalize("Acme Corp.", EntityType.COMPANY) == "acme"
    assert normalize("ACME Corporation", EntityType.COMPANY) == "acme"
    assert normalize("The Acme Company, Ltd.", EntityType.COMPANY) == "acme"
    assert normalize("AT&T Inc.", EntityType.COMPANY) == "at and t"
    assert normalize("Group Policy", EntityType.REGULATION) == "group policy"


def test_acme_trio_collapses_to_one_node(tmp_path: Path) -> None:
    results = {"c1": ents("c1", ("Acme Corp", "Company")), "c2": ents("c2", ("Acme Corporation", "Company")),
               "c3": ents("c3", ("ACME", "Company"))}
    nodes, _, report = resolver(tmp_path).resolve(results)
    assert len(nodes) == 1
    assert set(nodes[0].aliases) | {nodes[0].canonical_name} == {"Acme Corp", "Acme Corporation", "ACME"}
    assert nodes[0].chunk_ids == ["c1", "c2", "c3"]
    assert report.mentions == 3


def test_seed_aliases_and_acronyms(tmp_path: Path) -> None:
    results = {"c1": ents("c1", ("Taiwan Semiconductor Manufacturing Company Limited", "Company"),
                          ("Advanced Micro Devices, Inc.", "Company")),
               "c2": ents("c2", ("TSMC", "Company"), ("AMD", "Company"))}
    nodes, _, report = resolver(tmp_path).resolve(results)
    assert len(nodes) == 2
    assert report.merges_by_rule.get("seed", 0) + report.merges_by_rule.get("acronym", 0) == 2


def test_distinct_entities_stay_separate(tmp_path: Path) -> None:
    results = {"c1": ents("c1", ("Intel Corporation", "Company"), ("Intel Foundry", "Company"),
                          ("H100", "Product"), ("H200", "Product"),
                          ("Hock Tan", "Person"), ("Lip-Bu Tan", "Person"))}
    nodes, _, _ = resolver(tmp_path, threshold=0.80).resolve(results)
    assert len(nodes) == 6


def test_person_guard_and_number_guard() -> None:
    assert should_merge("lisa su", "Lisa Su", "lisa smith", "Lisa Smith", 0.99, EntityType.PERSON, 0.9) is None
    assert should_merge("h100", "H100", "h200", "H200", 0.99, EntityType.PRODUCT, 0.9) is None
    assert should_merge("lisa t su", "Lisa T. Su", "lisa su", "Lisa Su", 0.95, EntityType.PERSON, 0.9) == "embedding"


def test_company_sibling_guard_and_product_descriptors() -> None:
    # Real finding: BGE-small scores "Arm Holdings" vs "ARM Investments" at ~0.94 cosine.
    c = EntityType.COMPANY
    a, b = normalize("Arm Holdings", c), normalize("ARM Investments", c)
    assert should_merge(a, "Arm Holdings", b, "ARM Investments", 0.99, c, 0.91) is None
    assert should_merge(normalize("Intel", c), "Intel", normalize("Intel Foundry", c), "Intel Foundry", 0.99, c, 0.91) is None
    m = normalize("Micron Technology", c)
    assert should_merge(m, "Micron Technology", normalize("Micron", c), "Micron", 0.95, c, 0.91) == "embedding"
    assert normalize("EPYC processors", EntityType.PRODUCT) == normalize("EPYC", EntityType.PRODUCT)


def test_resolution_is_order_independent_and_ids_stable(tmp_path: Path) -> None:
    names = [("Acme Corp", "Company"), ("Acme Corporation", "Company"), ("ACME", "Company"),
             ("NVIDIA Corporation", "Company"), ("NVIDIA", "Company"), ("Jensen Huang", "Person")]
    a = {f"c{i}": ents(f"c{i}", n) for i, n in enumerate(names)}
    keys = list(a)
    random.Random(3).shuffle(keys)
    b = {k: a[k] for k in keys}
    na, _, _ = resolver(tmp_path).resolve(a)
    nb, _, _ = resolver(tmp_path).resolve(b)
    assert [n.node_id for n in na] == [n.node_id for n in nb]
    assert not duplicate_canonicals(na)


def test_edges_aggregate_chunk_ids_and_symmetric_edges_dedupe(tmp_path: Path) -> None:
    def r(cid: str, s: str, o: str) -> ExtractionResult:
        return ExtractionResult.model_validate({
            "entities": [{"entity_type": "Company", "canonical_name": x, "source_chunk_id": cid, "confidence": 0.9}
                         for x in (s, o)],
            "relationships": [{"source_name": s, "source_type": "Company", "relationship_type": "COMPETES_WITH",
                               "target_name": o, "target_type": "Company", "source_chunk_id": cid,
                               "evidence": f"{s} competes with {o}.", "confidence": 0.9}]})
    nodes, edges, _ = resolver(tmp_path).resolve({"c1": r("c1", "NVIDIA", "AMD"), "c2": r("c2", "AMD", "NVIDIA Corporation")})
    assert len(edges) == 1 and edges[0].source_chunk_ids == ["c1", "c2"]
