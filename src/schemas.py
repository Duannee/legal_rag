"""Core data schemas used across ingestion, retrieval, and answering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class DocumentPage:
    doc_id: str
    title: str
    source_path: str
    source_type: str
    court: str
    jurisdiction: str
    date: str
    document_type: str
    page_start: int
    page_end: int
    section_label: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_title: str
    section_path: str
    section_label: str
    page_span: str
    paragraph_span: str
    char_start: int
    char_end: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk_id: str
    lexical_score: float
    semantic_score: float
    fused_score: float
    rerank_score: float
    authority_rank: int
    direct_support_score: float
    chunk: Chunk


@dataclass
class Citation:
    source_title: str
    doc_id: str
    chunk_id: str
    section_label: str
    page_span: str
    evidence_snippet: str


@dataclass
class ClaimCheck:
    claim: str
    status: Literal["supported", "partially_supported", "unsupported"]
    supporting_chunk_ids: list[str]


@dataclass
class ClaimVerification:
    status: Literal["supported", "partially_supported", "unsupported"]
    checked_claims: list[ClaimCheck]


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]
    failure_reason: str | None
    needs_clarification: bool
    claim_verification: ClaimVerification | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
            "needs_clarification": self.needs_clarification,
            "claim_verification": asdict(self.claim_verification) if self.claim_verification else None,
        }
