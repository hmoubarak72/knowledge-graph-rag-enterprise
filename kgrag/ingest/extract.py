"""Phase 1 extraction: one Claude call per chunk, strict schema, validated, cached by document hash."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from kgrag.config import Settings
from kgrag.ingest.chunker import Document
from kgrag.llm import LLMClient, estimate_cost
from kgrag.log import get_logger
from kgrag.models import Chunk, ExtractionResult
from kgrag.ontology import ontology_prompt_block

log = get_logger(__name__)

EXPECTED_OUTPUT_TOKENS = 1800  # empirical average per 3.5k-char chunk; used for pre-run estimates

SYSTEM_PROMPT = f"""You extract a knowledge graph from SEC 10-K filing text.

Use ONLY the ontology below. Never invent entity or relationship types.

{ontology_prompt_block()}

Rules:
- Extract only facts stated in the chunk. No outside knowledge, no inference beyond the text.
- canonical_name: the most complete proper name used in the text (e.g. "Taiwan Semiconductor
  Manufacturing Company Limited" rather than "our foundry"). Resolve "we", "our", "the Company"
  to the filer named in the context header.
- FinancialMetric names are generic measures ("Revenue", "Research and development expense");
  put numbers, units and periods in the relationship properties (value, unit, period), never in the name.
- Every relationship endpoint must also appear in `entities` with the same name and type.
- evidence: copy the supporting sentence VERBATIM from the chunk (it is checked by string match).
- source_chunk_id: exactly the chunk id given in the header, on every entity and relationship.
- confidence: 0.9+ explicit statement, 0.6-0.9 clearly implied, below 0.6 omit the fact.
- Do not extract generic risk-factor hypotheticals ("we may be subject to...") as relationships,
  except SUBJECT_TO / REGULATED_BY for named regulations and regulators.
- If the chunk has no extractable facts, return empty lists.
"""


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def build_user_prompt(chunk: Chunk) -> str:
    return (
        f"<context>\nchunk_id: {chunk.chunk_id}\nfiler: {chunk.company}\n"
        f"section: {chunk.section_path}\nfiling_date: {chunk.date}\n</context>\n\n"
        f"<chunk>\n{chunk.text}\n</chunk>"
    )


def make_validator(chunk: Chunk):  # type: ignore[no-untyped-def]
    haystack = _norm_ws(chunk.text)

    def _validate(res: ExtractionResult) -> None:
        res.check_chunk_ids(chunk.chunk_id)
        res.check_relation_endpoints()
        missing = [r.evidence[:80] for r in res.relationships if _norm_ws(r.evidence) not in haystack]
        if missing:
            raise ValueError(f"evidence must be verbatim from the chunk; not found: {missing[:3]}")

    return _validate


@dataclass
class DocEstimate:
    document_id: str
    chunks: int
    est_usd: float
    cached: bool


class Extractor:
    def __init__(self, llm: LLMClient, settings: Settings, workers: int = 4) -> None:
        self.llm = llm
        self.settings = settings
        self.workers = workers
        self.cache_dir = settings.cache_dir / "extractions"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.failed: set[str] = set()  # chunk ids whose extraction failed validation after retries

    # -------------------------------------------------- document-hash cache (re-runs are free)
    def _cache_path(self, doc: Document) -> Path:
        return self.cache_dir / f"{doc.sha256}.json"

    def cached(self, doc: Document) -> dict[str, ExtractionResult] | None:
        p = self._cache_path(doc)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("model") != self.settings.extraction_model:
            return None
        return {cid: ExtractionResult.model_validate(r) for cid, r in data["results"].items()}

    # -------------------------------------------------- cost guardrail
    def estimate(self, doc: Document, chunks: list[Chunk]) -> DocEstimate:
        if self.cached(doc) is not None:
            return DocEstimate(doc.document_id, len(chunks), 0.0, True)
        usd = sum(
            estimate_cost(self.settings.extraction_model, len(SYSTEM_PROMPT) + len(build_user_prompt(c)),
                          EXPECTED_OUTPUT_TOKENS)
            for c in chunks
        )
        return DocEstimate(doc.document_id, len(chunks), usd, False)

    # -------------------------------------------------- run
    def extract_document(self, doc: Document, chunks: list[Chunk]) -> dict[str, ExtractionResult]:
        hit = self.cached(doc)
        if hit is not None:
            log.info("extract.cache_hit", document=doc.document_id, chunks=len(hit))
            return hit

        def one(c: Chunk) -> tuple[str, ExtractionResult]:
            try:
                res, stats = self.llm.structured(
                    model=self.settings.extraction_model, system=SYSTEM_PROMPT, user=build_user_prompt(c),
                    output=ExtractionResult, purpose="extraction", max_tokens=12000,
                    expected_output_tokens=EXPECTED_OUTPUT_TOKENS, effort="low",
                    post_validate=make_validator(c),
                )
                return c.chunk_id, res
            except Exception as err:  # keep the run going; failures are logged and counted
                if err.__class__.__name__ == "BudgetExceededError":
                    raise
                log.error("extract.chunk_failed", chunk=c.chunk_id, error=str(err)[:300])
                self.failed.add(c.chunk_id)
                return c.chunk_id, ExtractionResult()

        with ThreadPoolExecutor(self.workers) as pool:
            results = dict(pool.map(one, chunks))

        if any(c.chunk_id in self.failed for c in chunks):
            # don't cache a partially failed document: a re-run should retry the failed chunks
            # (successful chunks are still free via the per-call LLM cache)
            log.warning("extract.not_cached_due_to_failures", document=doc.document_id)
            return results
        self._cache_path(doc).write_text(json.dumps({
            "document_id": doc.document_id, "sha256": doc.sha256, "model": self.settings.extraction_model,
            "results": {cid: r.model_dump(mode="json") for cid, r in results.items()},
        }), encoding="utf-8")
        n_e = sum(len(r.entities) for r in results.values())
        n_r = sum(len(r.relationships) for r in results.values())
        log.info("extract.done", document=doc.document_id, chunks=len(chunks), entities=n_e, relations=n_r)
        return results
