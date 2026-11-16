"""The Cypher library is closed: only enum-validated parameters, never model-written query text."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from kgrag.retrieval.templates import CYPHER, QueryType, TemplateParams, check_params


def test_every_query_type_has_a_template() -> None:
    assert set(CYPHER) == set(QueryType)


def test_templates_take_values_only_as_driver_parameters() -> None:
    allowed = set(TemplateParams().driver_params())
    for qt, cypher in CYPHER.items():
        # no unrendered format placeholders, and every literal map uses only $params / node properties
        assert not re.search(r"\{\s*[A-Za-z_]+\s*\}", cypher), qt
        for m in re.findall(r"\{(node_id: [^}]*)\}", cypher):
            assert m.startswith("node_id: $") or m.startswith("node_id: n.") or m.startswith("node_id: a."), (qt, m)
        used = set(re.findall(r"\$([a-z_0-9]+)", cypher))
        assert used <= allowed, (qt, used - allowed)
        assert not re.search(r"\b(CREATE|MERGE|DELETE|SET|REMOVE|CALL|LOAD)\b", cypher), qt


def test_params_reject_free_text() -> None:
    with pytest.raises(ValidationError):
        TemplateParams(rel_types=["SUPPLIES]->() DETACH DELETE n //"])  # type: ignore[list-item]
    with pytest.raises(ValidationError):
        TemplateParams(entity_ids=["x' OR 1=1"])
    with pytest.raises(ValidationError):
        TemplateParams(direction="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TemplateParams(limit=10_000)


def test_check_params_enforces_slots() -> None:
    with pytest.raises(ValueError):
        check_params(QueryType.SHARED_NEIGHBORS, TemplateParams(entity_ids=["Company:nvidia"], rel_types=["SUPPLIES"]))
    with pytest.raises(ValueError):
        check_params(QueryType.TWO_HOP, TemplateParams(entity_ids=["Company:vmware"], rel_types=["ACQUIRED"]))
    check_params(QueryType.TWO_HOP, TemplateParams(entity_ids=["Company:vmware"], rel_types=["ACQUIRED"],
                                                   rel_types_2=["EMPLOYS"]))
