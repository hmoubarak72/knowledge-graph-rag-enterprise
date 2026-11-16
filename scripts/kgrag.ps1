# Windows equivalent of the Makefile (no `make` needed).
# Usage: ./scripts/kgrag.ps1 <up|down|fetch|estimate|ingest|recall|run|eval|test>
param([Parameter(Mandatory = $true)][string]$Task, [string]$Corpus = "data/corpus")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
switch ($Task) {
    "up"       { docker compose up -d neo4j postgres }
    "down"     { docker compose down }
    "fetch"    { python -m kgrag.corpus.sec --out $Corpus }
    "estimate" { python -m ingest --path $Corpus --estimate-only }
    "ingest"   { python -m ingest --path $Corpus }
    "recall"   { python -m kgrag.eval.recall }
    "run"      { uvicorn kgrag.api.app:app --port 8000 }
    "eval"     { python -m kgrag.eval.benchmark }
    "test"     { pytest -q }
    default    { throw "Unknown task '$Task'" }
}
