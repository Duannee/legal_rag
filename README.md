# NextLex Legal RAG Prototype (Cox v. Sony)

Compact, production-minded local Python prototype for legal research QA on **Cox Communications, Inc. v. Sony Music Entertainment, 607 U.S. ___ (2026)**.

This repository is intentionally scoped for a senior-engineer take-home: no UI, no auth, no cloud infra; strong focus on retrieval quality, grounding, prompt governance, and graceful failure behavior.

Stretch improvement included: a lightweight post-generation claim-to-evidence verification step so citations are not only present, but checked for support of the material answer claim.

## Architecture

Pipeline:
1. **Ingest (`src/ingest.py`)**: parse PDF page-by-page, normalize text, preserve source metadata, detect coarse legal sections.
2. **Chunk (`src/chunking.py`)**: split by paragraph with section continuity and overlap.
3. **Index (`src/index.py`)**: build hybrid retrieval indexes:
   - BM25 lexical index
   - semantic embedding index (sentence-transformers when available, deterministic fallback otherwise)
4. **Retrieve (`src/retrieve.py`)**: hybrid retrieval + reciprocal-rank fusion + lightweight legal reranking.
5. **Answer (`src/answer.py`)**: citation-grounded answer generation.
6. **Verify (`src/answer.py`)**: lightweight claim-to-evidence check over generated claim sentence(s) vs cited snippets (`supported` / `partially_supported` / `unsupported`).
7. **Evaluate (`src/eval.py`)**: regression harness over curated legal questions, including verification status checks.

## Repository Layout

```text
.
├── data/
│   ├── raw/
│   │   └── cox_v_sony_ocr.pdf
│   └── processed/
├── prompts/
│   └── prompt_charter_v1.yaml
├── src/
│   ├── __init__.py
│   ├── answer.py
│   ├── chunking.py
│   ├── config.py
│   ├── eval.py
│   ├── index.py
│   ├── ingest.py
│   ├── llm.py
│   ├── main.py
│   ├── retrieve.py
│   ├── schemas.py
│   └── utils.py
├── tests/
│   └── test_eval_smoke.py
├── examples/
│   └── sample_outputs.md
├── .env.example
├── DESIGN_NOTE.md
├── Makefile
├── eval_cases.json
├── README.md
└── requirements.txt
```

## Setup

Prereqs: Python 3.11+

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env
```

Optional (better semantic retrieval, heavier install):

```bash
pip install sentence-transformers
```

## Run

### Option A: use your current corpus filename (`cox_v_sony_ocr.pdf`)

```bash
python -m src.ingest --pdf data/raw/cox_v_sony_ocr.pdf
python -m src.index
python -m src.main --question "What are the two ways the Court says intent can be shown for contributory copyright infringement in Cox v. Sony?"
python -m src.eval
python -m pytest -q
```

### Option B: use default ingest path from code

`src/config.py` defaults to `data/raw/cox_v_sony.pdf`.
If you want to use `make ingest` without arguments, rename or copy your PDF to that path.

## Makefile Commands

```bash
make install
make ingest
make index
make ask Q="What are the two ways intent can be shown?"
make eval
make test
```

Note: `make ingest` uses the default path from `src/config.py`.

## Environment Variables

Defined in `.env.example`:

- `LLM_PROVIDER` (`mock` or `openai`; default code fallback is `mock`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`)
- `TOP_K` (default `6`)

`LLM_PROVIDER=openai` is only used when `OPENAI_API_KEY` is set; otherwise the system falls back to mock generation.

## Output Contract

`src/main.py` prints JSON in this shape:

- `answer: string`
- `citations: list[{source_title, doc_id, chunk_id, section_label, page_span, evidence_snippet}]`
- `confidence: high|medium|low`
- `failure_reason: null|ambiguous_question|insufficient_evidence|out_of_scope|conflicting_evidence|unsupported_claim`
- `needs_clarification: bool`
- `claim_verification: null|{status, checked_claims[]}`
  - `status: supported|partially_supported|unsupported`
  - `checked_claims[]: {claim, status, supporting_chunk_ids[]}`

Behavioral guardrail:
- `supported`: normal answer path
- `partially_supported`: answer kept, confidence downgraded one level
- `unsupported`: graceful decline (`failure_reason=unsupported_claim`, low confidence)

## Evaluation

`eval_cases.json` currently includes:

- doctrinal QA cases (`q1` to `q6`)
- one ambiguity case (`q7_ambiguous`)
- one out-of-scope decline case (`q8_out_of_scope`)
- one verification guardrail case where a mismatched doctrinal claim should not pass as supported (`q9_verification_guardrail`)

`src/eval.py` reports `total`, `passed`, `failed`, and per-case diagnostics.

## Notes and Limitations

- Single-case starter corpus
- Coarse section detection from textual cues
- Optional semantic encoder dependency (`sentence-transformers`)
- No learned reranker or quote-level citation offsets yet
- Verification is intentionally lightweight (heuristic overlap + legal term cues), not a full entailment subsystem
