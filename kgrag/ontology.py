"""The ontology, hard-coded as enums. Defined before any extraction code (Phase 1).

Domain: semiconductor 10-K filings (NVIDIA, AMD, Intel, Qualcomm, Broadcom by default).
The extraction prompt, the Pydantic schema, the Neo4j labels and the Cypher templates
all derive from this file. Nothing outside these enums can be written to the graph.
"""

from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    COMPANY = "Company"                  # issuers, suppliers, customers, competitors, acquirees
    PERSON = "Person"                    # executives, directors
    PRODUCT = "Product"                  # product lines / platforms (e.g. "H100", "EPYC")
    SEGMENT = "Segment"                  # reportable business segments / end markets
    LOCATION = "Location"                # countries, regions, sites
    REGULATORY_BODY = "RegulatoryBody"   # SEC, BIS, EU Commission, SAMR ...
    REGULATION = "Regulation"            # export controls, CHIPS Act, GDPR ...
    LEGAL_PROCEEDING = "LegalProceeding" # named lawsuits / investigations
    FINANCIAL_METRIC = "FinancialMetric" # revenue, R&D expense, ... (value lives on the edge)


class RelationType(StrEnum):
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    ACQUIRED = "ACQUIRED"
    SUPPLIES = "SUPPLIES"            # supplier -> customer
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    EMPLOYS = "EMPLOYS"              # company -> person (role on edge)
    OFFERS = "OFFERS"                # company -> product
    HAS_SEGMENT = "HAS_SEGMENT"      # company -> segment
    OPERATES_IN = "OPERATES_IN"      # company -> location
    REGULATED_BY = "REGULATED_BY"    # company -> regulatory body
    SUBJECT_TO = "SUBJECT_TO"        # company/product -> regulation
    PARTY_TO = "PARTY_TO"            # company -> legal proceeding
    REPORTS = "REPORTS"              # company -> financial metric (value, unit, period on edge)
    CUSTOMER_CONCENTRATION = "CUSTOMER_CONCENTRATION"  # company -> major customer (share on edge)


E, R = EntityType, RelationType

# Allowed (source_type, target_type) pairs per relationship. Anything else is rejected
# at validation time — this is what keeps the graph queryable.
RELATION_SIGNATURES: dict[RelationType, set[tuple[EntityType, EntityType]]] = {
    R.SUBSIDIARY_OF: {(E.COMPANY, E.COMPANY)},
    R.ACQUIRED: {(E.COMPANY, E.COMPANY)},
    R.SUPPLIES: {(E.COMPANY, E.COMPANY)},
    R.COMPETES_WITH: {(E.COMPANY, E.COMPANY), (E.PRODUCT, E.PRODUCT)},
    R.PARTNERS_WITH: {(E.COMPANY, E.COMPANY)},
    R.EMPLOYS: {(E.COMPANY, E.PERSON)},
    R.OFFERS: {(E.COMPANY, E.PRODUCT)},
    R.HAS_SEGMENT: {(E.COMPANY, E.SEGMENT)},
    R.OPERATES_IN: {(E.COMPANY, E.LOCATION)},
    R.REGULATED_BY: {(E.COMPANY, E.REGULATORY_BODY)},
    R.SUBJECT_TO: {(E.COMPANY, E.REGULATION), (E.PRODUCT, E.REGULATION)},
    R.PARTY_TO: {(E.COMPANY, E.LEGAL_PROCEEDING)},
    R.REPORTS: {(E.COMPANY, E.FINANCIAL_METRIC), (E.SEGMENT, E.FINANCIAL_METRIC)},
    R.CUSTOMER_CONCENTRATION: {(E.COMPANY, E.COMPANY)},
}

# Symmetric relationships are stored once with endpoints in sorted order.
SYMMETRIC: frozenset[RelationType] = frozenset({R.COMPETES_WITH, R.PARTNERS_WITH})

# Edge properties the extractor may fill, per relationship. Everything else is dropped.
EDGE_PROPERTIES: dict[RelationType, tuple[str, ...]] = {
    R.EMPLOYS: ("role",),
    R.ACQUIRED: ("year", "amount"),
    R.REPORTS: ("value", "unit", "period"),
    R.CUSTOMER_CONCENTRATION: ("share", "period"),
    R.SUPPLIES: ("item",),
}

# Qualifier properties that are part of the edge identity (so FY2023 and FY2024 revenue
# are different edges rather than one overwritten edge).
EDGE_KEY_PROPERTIES: dict[RelationType, tuple[str, ...]] = {
    R.REPORTS: ("period",),
    R.CUSTOMER_CONCENTRATION: ("period",),
}

ENTITY_DESCRIPTIONS: dict[EntityType, str] = {
    E.COMPANY: "A legal business entity (issuer, supplier, customer, competitor, acquiree).",
    E.PERSON: "A named individual, typically an executive officer or director.",
    E.PRODUCT: "A named product, product family or platform.",
    E.SEGMENT: "A reportable business segment or named end market of a company.",
    E.LOCATION: "A country, region, or named site.",
    E.REGULATORY_BODY: "A government agency or regulator.",
    E.REGULATION: "A named law, rule, or control regime.",
    E.LEGAL_PROCEEDING: "A named lawsuit, investigation or arbitration.",
    E.FINANCIAL_METRIC: "A named financial measure such as Revenue or R&D expense (no numbers in the name).",
}


def is_valid_signature(rel: RelationType, src: EntityType, tgt: EntityType) -> bool:
    return (src, tgt) in RELATION_SIGNATURES[rel]


def ontology_prompt_block() -> str:
    """Render the ontology for inclusion in the extraction system prompt."""
    lines = ["ENTITY TYPES:"]
    lines += [f"- {t.value}: {ENTITY_DESCRIPTIONS[t]}" for t in EntityType]
    lines.append("\nRELATIONSHIP TYPES (allowed source -> target):")
    for rel, sigs in RELATION_SIGNATURES.items():
        pairs = ", ".join(f"{s.value}->{t.value}" for s, t in sorted(sigs))
        props = EDGE_PROPERTIES.get(rel)
        extra = f"  properties: {', '.join(props)}" if props else ""
        lines.append(f"- {rel.value}: {pairs}{extra}")
    return "\n".join(lines)
