"""Runtime configuration for the legal research prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    raw_pdf_path: Path = project_root / "data" / "raw" / "cox_v_sony.pdf"
    processed_dir: Path = project_root / "data" / "processed"
    documents_path: Path = processed_dir / "documents.jsonl"
    chunks_path: Path = processed_dir / "chunks.jsonl"
    bm25_index_path: Path = processed_dir / "bm25_index.pkl"
    semantic_index_path: Path = processed_dir / "semantic_index.npz"
    prompt_charter_path: Path = project_root / "prompts" / "prompt_charter_v1.yaml"
    eval_cases_path: Path = project_root / "eval_cases.json"

    top_k: int = int(os.getenv("TOP_K", "6"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


SETTINGS = Settings()
