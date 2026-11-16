"""Section-aware chunking for 10-K text.

10-Ks have a fixed item structure ("Item 1. Business", "Item 1A. Risk Factors", "Item 7. MD&A").
We split on those headings first, then pack paragraphs into chunks of <= max_chars with a
paragraph-aligned overlap. Chunk IDs are deterministic: `<doc_id>:<section>:<ordinal>`, so
re-chunking the same file yields the same IDs (required for idempotent MERGE writes).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from kgrag.models import Chunk

ITEM_RE = re.compile(
    r"^\s*(?:PART\s+[IV]+\s*[,.\-–—]?\s*)?ITEM\s+(\d{1,2}[A-C]?)\s*[.:\-–—]?\s*(.{0,120})$",
    re.IGNORECASE,
)
ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity", "6": "Reserved",
    "7": "Management's Discussion and Analysis", "7A": "Market Risk", "8": "Financial Statements",
    "9": "Changes in and Disagreements with Accountants", "9A": "Controls and Procedures",
    "9B": "Other Information", "9C": "Foreign Jurisdictions", "10": "Directors and Executive Officers",
    "11": "Executive Compensation", "12": "Security Ownership", "13": "Relationships",
    "14": "Accountant Fees", "15": "Exhibits", "16": "Form 10-K Summary",
}


@dataclass
class Document:
    document_id: str
    path: Path
    company: str
    ticker: str
    date: str
    fiscal_year: str
    text: str
    sha256: str


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    ticker = meta.get("ticker") or path.stem.split("_")[0].upper()
    fy = str(meta.get("fiscal_year") or "")
    return Document(
        document_id=f"{ticker}-10K-{fy}" if fy else path.stem,
        path=path,
        company=meta.get("company", ticker),
        ticker=ticker,
        date=meta.get("filing_date", ""),
        fiscal_year=fy,
        text=raw,
        sha256=sha,
    )


def discover(corpus: Path) -> list[Path]:
    if corpus.is_file():
        return [corpus]
    return sorted(p for p in corpus.glob("*.txt") if not p.name.startswith("_"))


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_path, body)]. The table of contents is skipped by keeping the *last*
    occurrence of each item heading that is followed by real content."""
    lines = text.splitlines()
    marks: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        m = ITEM_RE.match(ln)
        if m and len(ln) < 160:
            marks.append((i, m.group(1).upper()))
    if not marks:
        return [("document", text)]

    sections: list[tuple[str, str]] = []
    preamble = "\n".join(lines[: marks[0][0]]).strip()
    if preamble:
        sections.append(("cover", preamble))
    for idx, (start, item) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        if len(body) < 200:  # TOC entry or stub heading
            continue
        title = ITEM_TITLES.get(item, "")
        sections.append((f"Item {item}" + (f" > {title}" if title else ""), body))
    # Merge duplicates (same item split by a page break heading)
    merged: dict[str, list[str]] = {}
    for sp, body in sections:
        merged.setdefault(sp, []).append(body)
    return [(sp, "\n\n".join(bodies)) for sp, bodies in merged.items()]


def _slug(section_path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section_path.split(">")[0].lower()).strip("-") or "doc"


def chunk_document(doc: Document, max_chars: int = 3500, overlap_chars: int = 300) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_path, body in split_sections(doc.text):
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        # hard-split very long paragraphs on sentence boundaries
        units: list[str] = []
        for p in paras:
            p = " ".join(p.split())
            while len(p) > max_chars:
                cut = p.rfind(". ", 0, max_chars)
                cut = cut + 1 if cut > max_chars // 2 else max_chars
                units.append(p[:cut].strip())
                p = p[cut:].strip()
            if p:
                units.append(p)

        buf: list[str] = []
        size = 0
        ordinal = 0
        fresh = 0  # units added since the last flush (overlap units don't count)

        def flush(section_path: str = section_path) -> None:
            nonlocal buf, size, ordinal, fresh
            if not buf or fresh == 0:
                return
            text = "\n\n".join(buf)
            chunks.append(Chunk(
                chunk_id=f"{doc.document_id}:{_slug(section_path)}:{ordinal:04d}",
                document_id=doc.document_id, company=doc.company, section_path=section_path,
                date=doc.date, ordinal=ordinal, text=text,
            ))
            ordinal += 1
            # paragraph-aligned overlap
            keep: list[str] = []
            kept = 0
            for u in reversed(buf):
                if kept + len(u) > overlap_chars:
                    break
                keep.insert(0, u)
                kept += len(u)
            buf, size, fresh = keep, kept, 0

        for u in units:
            if size + len(u) > max_chars and fresh:
                flush()
            buf.append(u)
            size += len(u)
            fresh += 1
        flush()
    _assert_unique(chunks)
    return chunks


def _assert_unique(chunks: list[Chunk]) -> None:
    seen: set[str] = set()
    for c in chunks:
        if c.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {c.chunk_id}")
        seen.add(c.chunk_id)
