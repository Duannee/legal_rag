"""Legal-structure-aware chunking.

The strategy prefers section continuity and paragraph boundaries before token size.
This improves legal citation precision over naive fixed-width chunking.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .schemas import Chunk, DocumentPage
from .utils import paragraph_split, stable_id


def chunk_documents(
    docs: Iterable[DocumentPage],
    target_chars: int = 1200,
    overlap_chars: int = 180,
) -> list[Chunk]:
    by_section: dict[str, list[DocumentPage]] = defaultdict(list)
    for d in docs:
        key = d.section_label or "unknown"
        by_section[key].append(d)

    chunks: list[Chunk] = []

    for section, pages in by_section.items():
        pages = sorted(pages, key=lambda x: x.page_start)
        para_items: list[tuple[int, int, str]] = []
        for p in pages:
            paragraphs = paragraph_split(p.text)
            for pi, para in enumerate(paragraphs, start=1):
                para_items.append((p.page_start, pi, para))

        i = 0
        while i < len(para_items):
            start_i = i
            total = 0
            text_parts: list[str] = []
            page_start = para_items[i][0]
            para_start = para_items[i][1]

            while i < len(para_items):
                para = para_items[i][2]
                added = len(para) + 2
                if text_parts and total + added > target_chars:
                    break
                text_parts.append(para)
                total += added
                i += 1

            page_end = para_items[i - 1][0]
            para_end = para_items[i - 1][1]
            text = "\n\n".join(text_parts)

            chunk_id = stable_id(
                "chunk",
                f"{pages[0].doc_id}:{section}:{page_start}:{para_start}:{len(text)}",
            )
            doc_id = pages[0].doc_id.rsplit("_p", 1)[0]

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_title=pages[0].title,
                    section_path=section,
                    section_label=section,
                    page_span=f"{page_start}-{page_end}",
                    paragraph_span=f"{para_start}-{para_end}",
                    char_start=0,
                    char_end=len(text),
                    text=text,
                )
            )

            if i >= len(para_items):
                break

            # Paragraph-aware overlap preserves legal reasoning continuity
            overlap_budget = overlap_chars
            j = i - 1
            while j > start_i and overlap_budget > 0:
                overlap_budget -= len(para_items[j][2])
                j -= 1
            i = max(j + 1, start_i + 1)

    chunks.sort(key=lambda c: (c.section_path, c.page_span, c.paragraph_span))
    return chunks
