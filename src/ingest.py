"""Ingestion pipeline for legal PDF documents.

This module reads the source PDF page-by-page, preserves metadata, detects coarse
legal sections (syllabus/opinion/concurrence), and writes normalized JSONL output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from pypdf import PdfReader

from .config import SETTINGS
from .schemas import DocumentPage
from .utils import normalize_whitespace, stable_id, write_jsonl


def detect_section(text: str, current: str) -> str:
    lowered = text.lower()
    header = lowered[:700]
    heading = lowered[:260]

    # Section switches should only happen on strong heading cues, not topical words.
    if "concurring in judgment" in heading or re.search(r"\b[a-z]+,\s*j\.,\s*concurring\b", heading):
        return "concurrence"
    if re.search(r"\b[a-z]+,\s*j\.,\s*dissenting\b", heading):
        return "dissent"
    if "syllabus" in header and current in {"unknown", "syllabus"}:
        return "syllabus"
    if "opinion of the court" in header or "delivered the opinion of the court" in header:
        return "majority_opinion"
    return current


def ingest_pdf(pdf_path: Path) -> list[DocumentPage]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing source PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    title = "Cox Communications, Inc. v. Sony Music Entertainment, 607 U.S. ___ (2026)"
    base_doc_id = stable_id("doc", f"{pdf_path}:{title}")

    rows: list[DocumentPage] = []
    current_section = "unknown"

    for i, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_whitespace(raw_text)
        if not text:
            continue

        current_section = detect_section(text, current_section)
        page_doc_id = f"{base_doc_id}_p{i:03d}"

        rows.append(
            DocumentPage(
                doc_id=page_doc_id,
                title=title,
                source_path=str(pdf_path),
                source_type="pdf",
                court="Supreme Court of the United States",
                jurisdiction="United States",
                date="2026",
                document_type="judicial_opinion",
                page_start=i,
                page_end=i,
                section_label=current_section,
                text=text,
            )
        )

    return rows


def run_ingest(pdf_path: Path = SETTINGS.raw_pdf_path) -> Path:
    docs = ingest_pdf(pdf_path)
    write_jsonl(SETTINGS.documents_path, [d.to_dict() for d in docs])
    return SETTINGS.documents_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal PDF into normalized JSONL.")
    parser.add_argument("--pdf", type=Path, default=SETTINGS.raw_pdf_path)
    args = parser.parse_args()

    out = run_ingest(args.pdf)
    print(f"Ingested PDF -> {out}")


if __name__ == "__main__":
    main()
