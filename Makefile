.PHONY: up down fetch estimate ingest recall run eval test lint

CORPUS ?= data/corpus

up:        ## start Neo4j + Postgres/pgvector
	docker compose up -d neo4j postgres

down:
	docker compose down

fetch:     ## download 10-Ks from SEC EDGAR (needs SEC_USER_AGENT in .env)
	python -m kgrag.corpus.sec --out $(CORPUS)

estimate:  ## dry-run: chunk + estimate extraction cost, no API calls
	python -m ingest --path $(CORPUS) --estimate-only

ingest:    ## Phase 1+2: extract, resolve, write Neo4j + pgvector (idempotent)
	python -m ingest --path $(CORPUS)

recall:    ## Phase 2 gate: recall@k over HNSW configs
	python -m kgrag.eval.recall

run:       ## FastAPI on :8000
	uvicorn kgrag.api.app:app --port 8000

eval:      ## Phase 5 benchmark: hybrid vs vector-only
	python -m kgrag.eval.benchmark

test:
	pytest -q

lint:
	ruff check kgrag ingest tests && mypy kgrag
