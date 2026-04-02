"""Hybrid retrieval layer for narrow legal questions."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
import re

import numpy as np

from .config import SETTINGS
from .schemas import Chunk, RetrievalHit
from .utils import read_jsonl, tokenize


@dataclass
class RetrievalBundle:
    hits: list[RetrievalHit]
    weak_retrieval: bool


DOCTRINAL_QUERY_HINTS = [
    "legal standard",
    "holding",
    "elements",
    "test",
    "intent",
    "required",
    "two ways",
]


def _authority_rank(section_label: str) -> int:
    label = (section_label or "").lower()
    if "majority" in label or "opinion of the court" in label:
        return 3
    if "syllabus" in label:
        return 2
    if "concurr" in label or "dissent" in label:
        return 1
    return 1


def _authority_bonus(section_label: str, doctrinal_q: bool) -> float:
    label = (section_label or "").lower()
    if "majority" in label or "opinion of the court" in label:
        return 0.12 if doctrinal_q else 0.03
    if "syllabus" in label:
        return 0.08 if doctrinal_q else 0.02
    if "concurr" in label:
        return 0.01 if doctrinal_q else 0.005
    if "dissent" in label:
        return 0.0
    return 0.005


def _is_narrow_doctrinal_question(question: str) -> bool:
    q = question.lower()
    return any(h in q for h in DOCTRINAL_QUERY_HINTS)


def _doctrinal_support_score(question: str, chunk_text: str) -> float:
    q_toks = set(tokenize(question))
    c_toks = set(tokenize(chunk_text))
    overlap = len(q_toks & c_toks) / max(1, len(q_toks))

    lowered = chunk_text.lower()
    phrase_hits = 0
    for phrase in [
        "the court holds",
        "the court says",
        "required intent",
        "intent can be shown",
        "intent is shown",
        "two ways",
        "first",
        "second",
        "affirmatively induced",
        "tailored to infringement",
    ]:
        if phrase in lowered:
            phrase_hits += 1

    # Sentences with modal/legal formulation markers often carry the operative rule.
    modal_hits = len(re.findall(r"\b(can|must|requires?|shown|liable|enough)\b", lowered))
    return overlap + (0.12 * phrase_hits) + (0.01 * min(modal_hits, 8))


class HybridRetriever:
    def __init__(self) -> None:
        chunk_rows = read_jsonl(SETTINGS.chunks_path)
        for row in chunk_rows:
            row.setdefault("section_label", row.get("section_path", "unknown"))
        self.chunks = [Chunk(**row) for row in chunk_rows]
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}

        with SETTINGS.bm25_index_path.open("rb") as f:
            bm25_blob = pickle.load(f)
        self.bm25 = bm25_blob["bm25"]
        self.bm25_chunk_ids = bm25_blob["chunk_ids"]
        self.bm25_pos = {cid: i for i, cid in enumerate(self.bm25_chunk_ids)}

        sem = np.load(SETTINGS.semantic_index_path, allow_pickle=True)
        self.semantic_embeddings = sem["embeddings"].astype(np.float32)
        self.semantic_chunk_ids = sem["chunk_ids"].tolist()
        self.semantic_pos = {cid: i for i, cid in enumerate(self.semantic_chunk_ids)}
        kind_arr = sem.get("kind")
        self.semantic_kind = str(kind_arr[0]) if kind_arr is not None else "fallback"
        self.embedding_model_name = str(sem.get("model_name", np.array([SETTINGS.embedding_model]))[0])
        self._st_model = None
        if self.semantic_kind == "sentence_transformer":
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.embedding_model_name)
            except Exception:
                self.semantic_kind = "fallback"

    def _query_embedding(self, text: str) -> np.ndarray:
        if self.semantic_kind == "sentence_transformer" and self._st_model is not None:
            emb = self._st_model.encode([text], normalize_embeddings=True)
            return np.asarray(emb[0], dtype=np.float32)
        dim = self.semantic_embeddings.shape[1]
        vec = np.zeros(dim, dtype=np.float32)
        for tok in tokenize(text):
            vec[hash(tok) % dim] += 1.0
        norm = float(np.linalg.norm(vec)) + 1e-8
        return vec / norm

    def retrieve(
        self,
        question: str,
        top_k: int = 6,
        metadata_filter: dict[str, str] | None = None,
    ) -> RetrievalBundle:
        q_toks = tokenize(question)
        bm25_scores = np.asarray(self.bm25.get_scores(q_toks), dtype=np.float32)

        q_vec = self._query_embedding(question)
        sem_scores = self.semantic_embeddings @ q_vec

        lexical_rank = np.argsort(-bm25_scores)
        semantic_rank = np.argsort(-sem_scores)

        # Reciprocal-rank fusion: robust merge of lexical exactness and semantic recall
        fused: dict[str, float] = {}
        k_rrf = 50
        for rank, idx in enumerate(lexical_rank[: top_k * 5], start=1):
            cid = self.bm25_chunk_ids[idx]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank)
        for rank, idx in enumerate(semantic_rank[: top_k * 5], start=1):
            cid = self.semantic_chunk_ids[idx]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k_rrf + rank)

        ranked_ids = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]
        scored_hits: list[RetrievalHit] = []
        doctrinal_q = _is_narrow_doctrinal_question(question)
        for cid in ranked_ids:
            chunk = self.chunk_by_id.get(cid)
            if chunk is None:
                continue
            if metadata_filter:
                if any(getattr(chunk, k, None) != v for k, v in metadata_filter.items()):
                    continue

            b_idx = self.bm25_pos.get(cid, -1)
            s_idx = self.semantic_pos.get(cid, -1)
            lexical = float(bm25_scores[b_idx]) if b_idx >= 0 else 0.0
            semantic = float(sem_scores[s_idx]) if s_idx >= 0 else 0.0
            fused_score = float(fused[cid])
            authority_rank = _authority_rank(chunk.section_label)
            direct_support = _doctrinal_support_score(question, chunk.text)

            # Lightweight legal reranker: prioritize direct rule language and stronger authority.
            authority_weight = _authority_bonus(chunk.section_label, doctrinal_q)
            direct_weight = 0.03 * direct_support if doctrinal_q else 0.01 * direct_support
            rerank_score = fused_score + authority_weight + direct_weight

            scored_hits.append(
                RetrievalHit(
                    chunk_id=cid,
                    lexical_score=lexical,
                    semantic_score=semantic,
                    fused_score=fused_score,
                    rerank_score=rerank_score,
                    authority_rank=authority_rank,
                    direct_support_score=direct_support,
                    chunk=chunk,
                )
            )
            if len(scored_hits) >= top_k * 5:
                break

        hits = sorted(
            scored_hits,
            key=lambda h: (h.rerank_score, h.authority_rank, h.direct_support_score, h.fused_score),
            reverse=True,
        )[:top_k]
        weak_retrieval = not hits or hits[0].fused_score < 0.02
        return RetrievalBundle(hits=hits, weak_retrieval=weak_retrieval)
