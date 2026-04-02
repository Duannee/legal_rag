# Design Note

## Chunking Strategy

I use section-aware, paragraph-first chunking with bounded overlap. Legal text often relies on nearby paragraphs and subsection context; preserving those boundaries improves answer-bearing retrieval and citation defensibility compared to fixed windows.

## Retrieval Judgment

I chose hybrid retrieval (BM25 + semantic + RRF fusion). BM25 handles exact legal phrasing; semantic retrieval catches paraphrases. RRF is simple, robust, and easy to explain in an interview.

## Citation Strategy

Each answer carries citation objects containing source title, doc/chunk IDs, page span, and evidence snippet. This creates a transparent evidence trail and supports deterministic downstream checks.

## Claim-to-Evidence Verification (Lightweight Stretch)

In legal QA, returning citations is necessary but not sufficient: the material claim still needs a support check against those citations. I added a lightweight post-generation verifier in `src/answer.py` that:
- extracts 1-2 candidate claim sentences from the final answer
- compares each claim against cited evidence snippets using lexical overlap + legal-term overlap + a small phrase bonus
- labels each claim as `supported`, `partially_supported`, or `unsupported`
- emits a compact structured field: `claim_verification`

Output behavior is then calibrated:
- `supported`: keep normal behavior
- `partially_supported`: downgrade confidence by one level
- `unsupported`: fail safely with `failure_reason=unsupported_claim` and low confidence

This is intentionally explainable and interview-friendly, while adding a practical trustworthiness guardrail.

## Failure Handling

The system explicitly handles:
- ambiguous question -> request clarification
- insufficient evidence / weak retrieval -> decline with low confidence
- structured failure reasons for machine-readable behavior

This avoids silent hallucination and makes behavior predictable.

## Production Hardening Gaps

Main gaps for production:
- stronger verifier (claim decomposition, contradiction checks, calibrated entailment model)
- stronger legal section segmentation (layout/OCR-aware parsing)
- reranker tuned on legal QA labels
- exact quote offsets and richer provenance lineage
- monitoring for retrieval drift and citation quality
- multi-document scaling, tenancy, authz, and auditability
