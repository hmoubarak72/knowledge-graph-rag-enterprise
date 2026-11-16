"""Extraction contract: schema, ontology signatures, chunk-id and verbatim-evidence checks, retries."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kgrag.ingest.extract import make_validator
from kgrag.llm import BudgetExceededError, FakeBackend, LLMClient, LLMValidationError
from kgrag.models import Chunk, ExtractedRelation, ExtractionResult

CID = "TEST-10K-2025:item-1:0000"
CHUNK = Chunk(chunk_id=CID, document_id="TEST-10K-2025", company="Acme Corp", section_path="Item 1 > Business",
              date="2025-01-01", ordinal=0, text="Acme Corp acquired Widget Co in 2021. Jane Roe is our CEO.")


def rel(**kw: object) -> dict[str, object]:
    base = {"source_name": "Acme Corp", "source_type": "Company", "relationship_type": "ACQUIRED",
            "target_name": "Widget Co", "target_type": "Company", "source_chunk_id": CID,
            "evidence": "Acme Corp acquired Widget Co in 2021.", "confidence": 0.95, "properties": {"year": "2021"}}
    base.update(kw)
    return base


def good_result() -> dict[str, object]:
    return {"entities": [
        {"entity_type": "Company", "canonical_name": "Acme Corp", "source_chunk_id": CID, "confidence": 0.99},
        {"entity_type": "Company", "canonical_name": "Widget Co", "source_chunk_id": CID, "confidence": 0.97},
    ], "relationships": [rel()]}


def test_valid_result_parses_and_passes_post_validation() -> None:
    res = ExtractionResult.model_validate(good_result())
    make_validator(CHUNK)(res)


@pytest.mark.parametrize("bad", [
    {"relationship_type": "OWNS"},                                   # not in ontology
    {"target_type": "Person"},                                       # ACQUIRED is Company->Company
    {"confidence": 1.4},                                             # out of range
    {"properties": {"role": "CEO"}},                                 # property not allowed on ACQUIRED
    {"properties": {"colour": "red"}},                               # unknown property key
])
def test_schema_rejects_out_of_ontology_output(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExtractedRelation.model_validate(rel(**bad))


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExtractedRelation.model_validate({**rel(), "cypher": "MATCH (n) DETACH DELETE n"})


def test_post_validation_rejects_wrong_chunk_id_and_paraphrased_evidence() -> None:
    v = make_validator(CHUNK)
    wrong_id = good_result()
    wrong_id["relationships"] = [rel(source_chunk_id="OTHER:item-1:0000")]
    with pytest.raises(ValueError, match="source_chunk_id"):
        v(ExtractionResult.model_validate(wrong_id))
    para = good_result()
    para["relationships"] = [rel(evidence="Acme bought Widget.")]
    with pytest.raises(ValueError, match="verbatim"):
        v(ExtractionResult.model_validate(para))


def test_post_validation_rejects_dangling_endpoint() -> None:
    r = good_result()
    r["entities"] = r["entities"][:1]  # type: ignore[index]
    with pytest.raises(ValueError, match="endpoint"):
        make_validator(CHUNK)(ExtractionResult.model_validate(r))


def test_retry_attaches_validation_error(offline_settings: object) -> None:
    seen: list[str] = []
    replies = iter([json.dumps({"entities": [], "relationships": [rel(relationship_type="OWNS")]}),
                    json.dumps(good_result())])

    def responder(model: str, system: str, user: str) -> str:
        seen.append(user)
        return next(replies)

    llm = LLMClient(offline_settings, backend=FakeBackend(responder))  # type: ignore[arg-type]
    res, stats = llm.structured(model="claude-opus-5-5", system="sys", user="extract", output=ExtractionResult,
                                purpose="extraction", post_validate=make_validator(CHUNK))
    assert stats.attempts == 2
    assert "failed validation" in seen[1] and "OWNS" in seen[1]
    assert res.relationships[0].relationship_type.value == "ACQUIRED"


def test_gives_up_after_max_attempts(offline_settings: object) -> None:
    llm = LLMClient(offline_settings, backend=FakeBackend(lambda *a: "not json"))  # type: ignore[arg-type]
    with pytest.raises(LLMValidationError):
        llm.structured(model="claude-opus-5-5", system="s", user="u", output=ExtractionResult,
                       purpose="extraction", max_attempts=2)


def test_cache_makes_reruns_free(offline_settings: object) -> None:
    backend = FakeBackend(lambda *a: good_result())
    llm = LLMClient(offline_settings, backend=backend)  # type: ignore[arg-type]
    kw = dict(model="claude-opus-5-5", system="s", user="u", output=ExtractionResult, purpose="extraction")
    llm.structured(**kw)  # type: ignore[arg-type]
    spent = llm.spent()
    _, stats = llm.structured(**kw)  # type: ignore[arg-type]
    assert backend.calls == 1 and stats.cached and llm.spent() == spent


def test_budget_cap_aborts_before_calling(offline_settings: object) -> None:
    backend = FakeBackend(lambda *a: good_result())
    llm = LLMClient(offline_settings, backend=backend)  # type: ignore[arg-type]
    with pytest.raises(BudgetExceededError):
        llm.structured(model="claude-opus-5-5", system="s" * 10_000_000, user="u", output=ExtractionResult,
                       purpose="extraction")
    assert backend.calls == 0
