"""Deterministic stand-in for the LLM (KGRAG_OFFLINE=1). Used by tests and dry runs ONLY.

It is a crude rule-based extractor/router/synthesizer so the whole pipeline (schemas, validation,
retries, MERGE writes, templates, citation checks, API) can be exercised without an API key.
Nothing it produces is a benchmark result; the benchmark refuses to publish offline numbers.
"""

from __future__ import annotations

import re
from typing import Any

KNOWN = [
    ("Taiwan Semiconductor Manufacturing Company Limited", "Company"), ("TSMC", "Company"),
    ("NVIDIA Corporation", "Company"), ("NVIDIA", "Company"), ("Advanced Micro Devices, Inc.", "Company"),
    ("AMD", "Company"), ("Intel Corporation", "Company"), ("Intel", "Company"), ("Qualcomm", "Company"),
    ("Broadcom Inc.", "Company"), ("Broadcom", "Company"), ("VMware", "Company"), ("Xilinx", "Company"),
    ("Mellanox", "Company"), ("Samsung", "Company"), ("Micron", "Company"), ("SK hynix", "Company"),
    ("Apple", "Company"), ("Arm", "Company"),
    ("Export Administration Regulations", "Regulation"),
    ("Bureau of Industry and Security", "RegulatoryBody"), ("U.S. Department of Commerce", "RegulatoryBody"),
]
OUT_OF_SCOPE = re.compile(r"poem|stock price|should i buy|world cup|capital of|home address|tesla|win the|"
                          r"apple's own|samsung's operating", re.I)


def _found(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    taken: list[tuple[int, int]] = []
    for name, t in KNOWN:
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            if (name, t) not in out:
                out.append((name, t))
    return out


def _extract(user: str) -> dict[str, Any]:
    cid = re.search(r"chunk_id: (\S+)", user).group(1)  # type: ignore[union-attr]
    filer = re.search(r"filer: (.+)", user).group(1).strip()  # type: ignore[union-attr]
    body = user.split("<chunk>", 1)[1].rsplit("</chunk>", 1)[0]
    ents: dict[tuple[str, str], dict[str, Any]] = {}
    rels: list[dict[str, Any]] = []

    def ent(name: str, t: str) -> None:
        ents.setdefault((name.lower(), t), {"entity_type": t, "canonical_name": name, "source_chunk_id": cid,
                                            "confidence": 0.95})

    def rel(s: str, st: str, r: str, o: str, ot: str, ev: str, props: dict[str, str] | None = None) -> None:
        ent(s, st)
        ent(o, ot)
        rels.append({"source_name": s, "source_type": st, "relationship_type": r, "target_name": o,
                     "target_type": ot, "source_chunk_id": cid, "evidence": ev, "confidence": 0.9,
                     "properties": props or {}})

    for sent in re.split(r"(?<=[.!?])\s+", " ".join(body.split())):
        names = [n for n, t in _found(sent) if t == "Company" and n.lower() != filer.lower()]
        low = sent.lower()
        if m := re.search(r"([A-Z][a-z]+ (?:[A-Z]\. )?[A-Z][a-z]+), (?:our )?(?:[A-Z][a-z]+ and )?Chief Executive Officer",
                          sent):
            rel(filer, "Company", "EMPLOYS", m.group(1), "Person", sent, {"role": "Chief Executive Officer"})
        if "acquired" in low or "acquisition of" in low:
            for n in names:
                rel(filer, "Company", "ACQUIRED", n, "Company", sent)
        elif re.search(r"manufactur|foundr|wafer", low):
            for n in names:
                rel(n, "Company", "SUPPLIES", filer, "Company", sent, {"item": "wafers"})
        elif "compet" in low:
            for n in names:
                rel(filer, "Company", "COMPETES_WITH", n, "Company", sent)
        if "Export Administration Regulations" in sent:
            rel(filer, "Company", "SUBJECT_TO", "Export Administration Regulations", "Regulation", sent)
    return {"entities": list(ents.values()), "relationships": rels}


def _route(user: str) -> dict[str, Any]:
    qtext = user.split("Q:", 1)[-1].rsplit("A:", 1)[0]
    ments = [{"text": n, "entity_type": t} for n, t in _found(qtext)]
    low = qtext.lower()
    base = {"mentions": ments, "target_entity_type": None, "metrics": [], "period": "", "rationale": "offline rules"}
    if OUT_OF_SCOPE.search(qtext):
        return {**base, "route": "VECTOR", "confidence": 0.9, "query_type": None, "hops": []}
    if "acquired" in low and re.search(r"ceo|leads", low):
        return {**base, "route": "GRAPH", "confidence": 0.9, "query_type": "TWO_HOP",
                "hops": [{"rel_types": ["ACQUIRED"], "direction": "in"}, {"rel_types": ["EMPLOYS"], "direction": "out"}]}
    if "both" in low and len(ments) >= 2:
        rt = "COMPETES_WITH" if "compet" in low else "SUPPLIES"
        return {**base, "route": "GRAPH", "confidence": 0.9, "query_type": "SHARED_NEIGHBORS",
                "hops": [{"rel_types": [rt], "direction": "in" if rt == "SUPPLIES" else "any"}]}
    if "most" in low or "how many" in low:
        return {**base, "route": "GRAPH", "confidence": 0.85, "query_type": "AGGREGATE_DEGREE",
                "hops": [{"rel_types": ["SUPPLIES"], "direction": "out"}]}
    if re.search(r"\bceo\b", low) and ments:
        return {**base, "route": "GRAPH", "confidence": 0.8, "query_type": "RELATION_LOOKUP",
                "hops": [{"rel_types": ["EMPLOYS"], "direction": "out"}]}
    return {**base, "route": "VECTOR", "confidence": 0.6, "query_type": None, "hops": []}


def _synthesize(user: str) -> dict[str, Any]:
    question = user.rsplit("Question:", 1)[-1].strip()
    if OUT_OF_SCOPE.search(question):
        return {"refused": True, "refusal_reason": "outside the scope of the indexed filings", "sentences": []}
    facts = re.findall(r"^F\d+\. (.+?) \[sources: ([^\]]+)\]", user, re.M)
    if facts:
        return {"refused": False, "refusal_reason": "",
                "sentences": [{"text": t, "citations": [c.strip() for c in ids.split(",")][:2]} for t, ids in facts[:5]]}
    m = re.search(r'<passage chunk_id="([^"]+)"[^>]*>\n(.+?)\n</passage>', user, re.S)
    if not m:
        return {"refused": True, "refusal_reason": "no supporting context", "sentences": []}
    first = re.split(r"(?<=[.!?])\s+", m.group(2).strip())[0][:300]
    return {"refused": False, "refusal_reason": "", "sentences": [{"text": first, "citations": [m.group(1)]}]}


def _judge(user: str) -> dict[str, Any]:
    gold = re.search(r"Gold answer: (.+)", user).group(1)  # type: ignore[union-attr]
    ans = re.search(r"System answer: (.+)", user, re.S).group(1).lower()  # type: ignore[union-attr]
    key = [w for w in re.findall(r"[A-Za-z]{4,}", gold)][:2]
    ok = all(k.lower() in ans for k in key)
    return {"correct": ok, "reason": "offline keyword judge"}


def offline_responder(model: str, system: str, user: str) -> Any:
    if system.startswith("You extract a knowledge graph"):
        return _extract(user)
    if system.startswith("You route questions"):
        return _route(user)
    if system.startswith("You answer questions"):
        return _synthesize(user)
    if system.startswith("You grade answers"):
        return _judge(user)
    raise ValueError("offline responder: unknown prompt")
