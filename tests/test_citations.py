"""Citation enforcement is a hard validation: invented IDs are rejected, regenerated once, then refused."""

from __future__ import annotations

from typing import Any

from kgrag.llm import FakeBackend, LLMClient
from kgrag.models import GraphPath, Passage
from kgrag.synthesis import Synthesizer, build_context, verbalize_edge

P1 = Passage(chunk_id="AVGO-10K-2024:item-1:0000", text="Broadcom completed the acquisition of VMware.",
             company="Broadcom Inc.", section_path="Item 1", score=0.9)
P2 = Passage(chunk_id="AVGO-10K-2024:item-1:0001", text="Hock Tan, our President and Chief Executive Officer.",
             company="Broadcom Inc.", section_path="Item 1", score=0.8)


def answer(cites: list[str]) -> dict[str, Any]:
    return {"refused": False, "refusal_reason": "", "sentences": [{"text": "Hock Tan leads Broadcom.", "citations": cites}]}


def synth(offline_settings: object, replies: list[dict[str, Any]]) -> tuple[Synthesizer, FakeBackend]:
    it = iter(replies)
    backend = FakeBackend(lambda *a: next(it))
    return Synthesizer(LLMClient(offline_settings, backend=backend), offline_settings), backend  # type: ignore[arg-type]


def test_valid_citations_pass(offline_settings: object) -> None:
    s, backend = synth(offline_settings, [answer([P2.chunk_id])])
    r = s.answer("Who leads Broadcom?", build_context([], [P1, P2], []))
    assert not r.refused and [c.chunk_id for c in r.citations] == [P2.chunk_id] and r.answer.endswith("[1]")
    assert backend.calls == 1


def test_invented_citation_is_regenerated_once(offline_settings: object) -> None:
    s, backend = synth(offline_settings, [answer(["MADE-UP:item-9:0042"]), answer([P1.chunk_id, P2.chunk_id])])
    r = s.answer("Who leads Broadcom?", build_context([], [P1, P2], []))
    assert backend.calls == 2 and not r.refused
    assert {c.chunk_id for c in r.citations} == {P1.chunk_id, P2.chunk_id}


def test_persistent_invention_becomes_refusal(offline_settings: object) -> None:
    s, backend = synth(offline_settings, [answer(["FAKE:1"]), answer(["FAKE:2"])])
    r = s.answer("Who leads Broadcom?", build_context([], [P1, P2], []))
    assert backend.calls == 2 and r.refused and r.citation_validation_failed and r.citations == []


def test_uncited_sentence_rejected(offline_settings: object) -> None:
    s, _ = synth(offline_settings, [answer([]), answer([])])
    r = s.answer("Who leads Broadcom?", build_context([], [P1], []))
    assert r.refused and r.citation_validation_failed


def test_no_context_refuses_without_calling_model(offline_settings: object) -> None:
    s, backend = synth(offline_settings, [])
    r = s.answer("anything", build_context([], [], []))
    assert r.refused and backend.calls == 0


def test_graph_facts_are_verbalized_and_only_citable_facts_survive() -> None:
    edge = {"rel_type": "EMPLOYS", "start": "Company:broadcom", "end": "Person:hock tan",
            "source_chunk_ids": [P2.chunk_id], "properties": {"role": "President and Chief Executive Officer"}}
    orphan = {"rel_type": "ACQUIRED", "start": "Company:broadcom", "end": "Company:vmware",
              "source_chunk_ids": ["NOT-LOADED:0"], "properties": {}}
    names = {"Company:broadcom": "Broadcom Inc.", "Person:hock tan": "Hock Tan", "Company:vmware": "VMware"}
    assert verbalize_edge(edge, names) == "Hock Tan is President and Chief Executive Officer of Broadcom Inc."
    nodes = [{"node_id": k, "name": v, "entity_type": k.split(":")[0]} for k, v in names.items()]
    ctx = build_context([GraphPath(nodes=nodes, edges=[edge, orphan])], [], [P2])
    assert [f.text for f in ctx.facts] == ["Hock Tan is President and Chief Executive Officer of Broadcom Inc."]
    rendered = ctx.render()
    assert "## GRAPH FACTS" in rendered and "## RETRIEVED PASSAGES" in rendered


def test_dedup_across_sources() -> None:
    ctx = build_context([], [P1, P2, P1], [P1])
    assert [p.chunk_id for p in ctx.passages] == [P1.chunk_id, P2.chunk_id]
