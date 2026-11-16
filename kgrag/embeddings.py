"""Embedders. Default is local fastembed (BGE-small, ONNX/CPU); `HashEmbedder` is for offline tests."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from kgrag.config import get_settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.array(list(self.model.embed(texts, batch_size=32)), dtype=np.float32)
        return _l2(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE recommends an instruction prefix for retrieval queries.
        vec = np.array(list(self.model.query_embed(text)), dtype=np.float32)
        out: np.ndarray = _l2(vec)[0]
        return out


class HashEmbedder:
    """Character-trigram hashing embedder: deterministic, dependency-free, offline."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f"  {re.sub(r'[^a-z0-9 ]', ' ', text.lower())}  "
        for i in range(len(t) - 2):
            h = int.from_bytes(hashlib.blake2b(t[i:i + 3].encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        out: np.ndarray = self.embed([text])[0]
        return out


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    out: np.ndarray = (m / n).astype(np.float32)
    return out


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    if s.offline:
        return HashEmbedder(s.embed_dim)
    return FastEmbedder(s.embed_model, s.embed_dim)
