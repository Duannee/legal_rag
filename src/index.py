"""Index building for hybrid retrieval (BM25 + semantic cosine similarity)."""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import chunk_documents
from .config import SETTINGS
from .schemas import Chunk, DocumentPage
from .utils import read_jsonl, tokenize, write_jsonl


class SemanticEncoder:
    """Sentence-transformer encoder with deterministic lexical fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.kind = "fallback"
        self.model = None
        self.dim = 256
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.kind = "sentence_transformer"
            self.dim = self.model.get_sentence_embedding_dimension()
        except Exception:
            self.model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "sentence_transformer" and self.model is not None:
            arr = self.model.encode(texts, normalize_embeddings=True)
            return np.asarray(arr, dtype=np.float32)

        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                idx = hash(tok) % self.dim
                mat[i, idx] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norms


def build_indexes() -> None:
    docs = [DocumentPage(**d) for d in read_jsonl(SETTINGS.documents_path)]
    chunks = chunk_documents(docs)
    write_jsonl(SETTINGS.chunks_path, [c.to_dict() for c in chunks])

    tokenized = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with SETTINGS.bm25_index_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, f)

    encoder = SemanticEncoder(SETTINGS.embedding_model)
    embeddings = encoder.encode([c.text for c in chunks])
    np.savez_compressed(
        SETTINGS.semantic_index_path,
        embeddings=embeddings,
        chunk_ids=np.array([c.chunk_id for c in chunks]),
        kind=np.array([encoder.kind]),
        model_name=np.array([SETTINGS.embedding_model]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval indexes from ingested docs.")
    _ = parser.parse_args()
    build_indexes()
    print(f"Wrote chunks -> {SETTINGS.chunks_path}")
    print(f"Wrote BM25 -> {SETTINGS.bm25_index_path}")
    print(f"Wrote semantic -> {SETTINGS.semantic_index_path}")


if __name__ == "__main__":
    main()
