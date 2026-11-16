"""Phase 4: grounded answer synthesis with hard citation validation.

1. Graph paths are verbalized into readable statements (raw triples make awkward answers).
2. Facts and passages are deduplicated across both sources.
3. Context is assembled with explicit sections: GRAPH FACTS vs RETRIEVED PASSAGES.
4. The model returns sentences, each with chunk-ID citations. Validation (code, not prompt):
   every sentence must cite >= 1 chunk and every cited chunk must be in the retrieved context.
   On failure: reject and regenerate once with the error attached; if it fails again the system
   refuses rather than emit an unverifiable answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from kgrag.config import Settings
from kgrag.llm import CallStats, LLMClient, LLMValidationError
from kgrag.models import Citation, GraphPath, Passage

PHRASES: dict[str, str] = {
    "SUBSIDIARY_OF": "{a} is a subsidiary of {b}",
    "ACQUIRED": "{a} acquired {b}",
    "SUPPLIES": "{a} supplies {b}",
    "COMPETES_WITH": "{a} competes with {b}",
    "PARTNERS_WITH": "{a} partners with {b}",
    "EMPLOYS": "{b} works at {a}",
    "OFFERS": "{a} offers {b}",
    "HAS_SEGMENT": "{a} has a business segment called {b}",
    "OPERATES_IN": "{a} operates in {b}",
    "REGULATED_BY": "{a} is regulated by {b}",
    "SUBJECT_TO": "{a} is subject to {b}",
    "PARTY_TO": "{a} is a party to {b}",
    "REPORTS": "{a} reported {b}",
    "CUSTOMER_CONCENTRATION": "{b} is a major customer of {a}",
}


def verbalize_edge(edge: dict[str, object], names: dict[str, str]) -> str:
    a, b = names.get(str(edge["start"]), str(edge["start"])), names.get(str(edge["end"]), str(edge["end"]))
    rel = str(edge["rel_type"])
    s = PHRASES.get(rel, "{a} " + rel.lower().replace("_", " ") + " {b}").format(a=a, b=b)
    p: dict[str, str] = dict(edge.get("properties") or {})  # type: ignore[call-overload]
    if rel == "EMPLOYS" and p.get("role"):
        s = f"{b} is {p['role']} of {a}"
    elif rel == "REPORTS" and p.get("value"):
        s = f"{a} reported {b} of {p['value']}{(' ' + p['unit']) if p.get('unit') else ''}"
        if p.get("period"):
            s += f" for {p['period']}"
    elif rel == "ACQUIRED" and p.get("year"):
        s += f" in {p['year']}"
    elif rel == "CUSTOMER_CONCENTRATION" and p.get("share"):
        s += f" ({p['share']} of revenue{', ' + p['period'] if p.get('period') else ''})"
    elif rel == "SUPPLIES" and p.get("item"):
        s += f" ({p['item']})"
    return s.rstrip(".") + "."


@dataclass
class Fact:
    text: str
    chunk_ids: list[str]


@dataclass
class Context:
    facts: list[Fact] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    aggregates: list[str] = field(default_factory=list)

    @property
    def allowed_chunk_ids(self) -> set[str]:
        return {p.chunk_id for p in self.passages}

    def render(self) -> str:
        out = []
        if self.facts or self.aggregates:
            out.append("## GRAPH FACTS (from the knowledge graph; each lists the chunk(s) that justify it)")
            out += [f"- {a}" for a in self.aggregates]
            out += [f"F{i + 1}. {f.text} [sources: {', '.join(f.chunk_ids)}]" for i, f in enumerate(self.facts)]
        if self.passages:
            out.append("\n## RETRIEVED PASSAGES (verbatim filing text)")
            for p in self.passages:
                out.append(f'<passage chunk_id="{p.chunk_id}" company="{p.company}" section="{p.section_path}">\n'
                           f"{p.text}\n</passage>")
        return "\n".join(out) if out else "(no context retrieved)"


def build_context(paths: list[GraphPath], passages: list[Passage], evidence_passages: list[Passage],
                  max_facts: int = 40, max_passages: int = 12) -> Context:
    ctx = Context()
    seen_facts: set[str] = set()
    for p in paths:
        names = {n["node_id"]: n["name"] for n in p.nodes}
        if p.aggregate:
            agg = f"{names.get(p.aggregate['node_id'], p.aggregate['node_id'])} has {p.aggregate['degree']} distinct connections of this type"
            if agg not in ctx.aggregates:
                ctx.aggregates.append(agg)
        for e in p.edges:
            t = verbalize_edge(e, names)
            if t in seen_facts:
                continue
            seen_facts.add(t)
            ctx.facts.append(Fact(t, list(e.get("source_chunk_ids", []))[:3]))
    ctx.facts = ctx.facts[:max_facts]
    # passages: graph evidence chunks first (so every fact's source is in context), then vector hits
    cited_by_facts = {c for f in ctx.facts for c in f.chunk_ids}
    seen: set[str] = set()
    for psg in [*evidence_passages, *passages]:
        if psg.chunk_id in seen:
            continue
        if len(ctx.passages) >= max_passages and psg.chunk_id not in cited_by_facts:
            continue
        seen.add(psg.chunk_id)
        ctx.passages.append(psg)
    # facts whose evidence chunk could not be loaded are dropped (they'd be uncitable)
    allowed = ctx.allowed_chunk_ids
    for f in ctx.facts:
        f.chunk_ids = [c for c in f.chunk_ids if c in allowed]
    ctx.facts = [f for f in ctx.facts if f.chunk_ids]
    return ctx


class AnswerSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    citations: list[str]


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refused: bool
    refusal_reason: str = ""
    sentences: list[AnswerSentence] = Field(default_factory=list)


SYSTEM_PROMPT = """You answer questions using ONLY the provided context from SEC 10-K filings.

- Every sentence of your answer must cite at least one chunk_id from the context in `citations`.
  Cite the passage chunk_id(s) that support the sentence; graph facts list their source chunk_ids.
- Use only chunk_ids that appear in the context. Never invent or alter an id.
- If the context does not contain the information needed, or the question is outside the scope of
  these filings (other companies, predictions, opinions, unrelated topics), set refused=true,
  explain briefly in refusal_reason, and return no sentences. Do not answer partially from memory.
- Be concise: 1-5 sentences. For list questions, name every item supported by the context.
"""


def _validator(allowed: set[str]):  # type: ignore[no-untyped-def]
    def check(out: SynthesisOutput) -> None:
        if out.refused:
            return
        if not out.sentences:
            raise ValueError("refused=false but no sentences")
        for s in out.sentences:
            if not s.citations:
                raise ValueError(f"sentence has no citation: {s.text[:80]!r}")
            bad = [c for c in s.citations if c not in allowed]
            if bad:
                raise ValueError(f"citations not in retrieved context: {bad}. Allowed ids: {sorted(allowed)}")
    return check


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str
    citation_validation_failed: bool
    stats: CallStats | None


class Synthesizer:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self.llm, self.settings = llm, settings

    def answer(self, question: str, ctx: Context) -> SynthesisResult:
        allowed = ctx.allowed_chunk_ids
        if not allowed:
            return SynthesisResult("I can't answer that from the indexed filings.", [], True,
                                   "no context retrieved", False, None)
        user = f"<context>\n{ctx.render()}\n</context>\n\nQuestion: {question}"
        try:
            out, stats = self.llm.structured(
                model=self.settings.synthesis_model, system=SYSTEM_PROMPT, user=user, output=SynthesisOutput,
                purpose="synthesis", max_tokens=4000, expected_output_tokens=400, effort="low",
                max_attempts=2,  # one regenerate on citation failure
                post_validate=_validator(allowed),
            )
        except LLMValidationError as err:
            return SynthesisResult("I couldn't produce an answer with verifiable citations.", [], True,
                                   f"citation validation failed: {err}"[:300], True, None)
        if out.refused:
            return SynthesisResult(out.refusal_reason or "I can't answer that from the indexed filings.", [], True,
                                   out.refusal_reason, False, stats)
        by_id = {p.chunk_id: p for p in ctx.passages}
        order: list[str] = []
        parts: list[str] = []
        for s in out.sentences:
            marks = []
            for c in s.citations:
                if c not in order:
                    order.append(c)
                marks.append(str(order.index(c) + 1))
            parts.append(f"{s.text.strip()} [{', '.join(marks)}]")
        cites = [Citation(chunk_id=c, quote=by_id[c].text[:240]) for c in order]
        return SynthesisResult(" ".join(parts), cites, False, "", False, stats)
