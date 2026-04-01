# NextLex Legal RAG Prototype (Cox v. Sony)

Compact, production-minded local Python prototype for legal research QA on **Cox Communications, Inc. v. Sony Music Entertainment, 607 U.S. ___ (2026)**.

This repository is intentionally scoped for a senior-engineer take-home: no UI, no auth, no cloud infra; strong focus on retrieval quality, grounding, prompt governance, and graceful failure behavior.

## Architecture

Pipeline:
1. **Ingest (`src/ingest.py`)**: parse PDF page-by-page, normalize text, preserve source metadata, detect coarse legal sections.
2. **Chunk (`src/chunking.py`)**: split by paragraph with section continuity and overlap; preserve legal context in chunk metadata.
3. **Index (`src/index.py`)**: build hybrid retrieval indexes:
   - BM25 lexical index
   - semantic embedding index (sentence-transformers when available, deterministic fallback otherwise)
4. **Retrieve (`src/retrieve.py`)**: hybrid retrieval + reciprocal-rank fusion, score exposure, metadata filter hooks.
5. **Answer (`src/answer.py`)**: prompt-charter-governed, citation-grounded, structured output with failure/ambiguity handling.
6. **Evaluate (`src/eval.py`)**: lightweight regression harness over curated legal questions.

## Repository Layout

```
.
├── data/
│   ├── raw/
│   │   └── cox_v_sony.pdf
│   └── processed/
├── prompts/
│   └── prompt_charter_v1.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── index.py
│   ├── retrieve.py
│   ├── answer.py
│   ├── llm.py
│   ├── eval.py
│   ├── utils.py
│   └── main.py
├── examples/
│   └── sample_outputs.md
├── tests/
│   └── test_eval_smoke.py
├── eval_cases.json
├── README.md
├── README_EXPLAINED.md
├── DESIGN_NOTE.md
├── requirements.txt
├── .env.example
└── Makefile
```

## Setup

Prereqs: Python 3.11+

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
```

Place source corpus at:
- `data/raw/cox_v_sony.pdf`

Optional (better semantic retrieval, heavier install):
```bash
pip install sentence-transformers
```

## Run

```bash
make ingest
make index
python -m src.main --question "What are the two ways the Court says intent can be shown for contributory copyright infringement in Cox v. Sony?"
make eval
make test
```

## Corpus and Metadata

Ingestion creates normalized JSONL (`data/processed/documents.jsonl`) with fields including:
- `doc_id`, `title`, `source_path`, `source_type`
- `court`, `jurisdiction`, `date`, `document_type`
- `page_start`, `page_end`, `section_label`, `text`

Chunking creates `data/processed/chunks.jsonl` with legal-aware metadata:
- `chunk_id`, `doc_id`, `section_path`
- `page_span`, `paragraph_span`
- `char_start`, `char_end`, `text`

## Retrieval Strategy

Hybrid retrieval is used for narrow legal QA:
- BM25 catches exact statutory/doctrinal terms
- semantic retrieval improves recall for paraphrased legal phrasing
- reciprocal-rank fusion merges both signals

Returned hits expose:
- `lexical_score`
- `semantic_score`
- `fused_score`

A metadata-aware filter hook supports future constraints (e.g., section-level retrieval only).

## Prompt Charter

`prompts/prompt_charter_v1.yaml` separates prompt governance from code:
- input/output contracts
- citation policy
- guardrails
- ambiguity handling
- insufficient/conflicting evidence behavior

This makes behavior explicit, versionable, and interview-auditable.

## Output Contract

Structured response schema:
- `answer: string`
- `citations: list[{source_title, doc_id, chunk_id, page_span, evidence_snippet}]`
- `confidence: high|medium|low`
- `failure_reason: optional string`
- `needs_clarification: bool`

## Failure Modes

Implemented graceful behavior:
- ambiguous question -> asks for narrowing (`needs_clarification=true`)
- weak retrieval / insufficient support -> declines with explanation
- low-support cases -> reduced confidence

## Evaluation

`eval_cases.json` includes the six narrow assignment questions plus one ambiguous question.

`src/eval.py` reports:
- pass/fail per case
- answer/decline/clarification behavior
- citation presence
- retrieval-hit check (when expected doc IDs are provided)

## Limitations

- single-case starter corpus
- coarse section detection based on textual hints
- fallback semantic embeddings are lightweight, not SOTA
- no learned reranker yet
- no long-context quote pinning / citation span alignment beyond chunk snippets

## Productionization Notes

If extended to production:
- multi-document parser normalization + citation parser
- stronger legal section segmentation (layout-aware parsing)
- reranker tuned for legal answer-bearing evidence
- quote-level citation anchors with exact offsets
- provenance/audit logs + model/prompt version pinning
- tenant-aware indexing, auth, and observability
# legal_rag
