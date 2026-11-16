"""FastAPI service. POST /ask {question} -> {answer, citations[], route_used, graph_paths[]?}."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from kgrag.config import get_settings
from kgrag.embeddings import get_embedder
from kgrag.llm import BudgetExceededError, LLMClient, LLMRefusalError
from kgrag.models import AskRequest, AskResponse
from kgrag.rag import HybridRAG
from kgrag.stores.factory import make_stores

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    graph, vectors = make_stores(s)
    state["rag"] = HybridRAG(s, LLMClient(s), get_embedder(), vectors, graph)
    state["stores"] = (graph, vectors)
    yield
    graph.close()
    vectors.close()


app = FastAPI(title="Knowledge Graph RAG for Enterprise Data", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    graph, _ = state["stores"]
    return {"status": "ok", "graph": graph.counts(), "offline": get_settings().offline}


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(req: AskRequest) -> AskResponse:
    try:
        resp, _trace = state["rag"].ask(req.question)
    except BudgetExceededError as err:
        raise HTTPException(status_code=402, detail=str(err)) from err
    except LLMRefusalError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return resp  # type: ignore[no-any-return]
