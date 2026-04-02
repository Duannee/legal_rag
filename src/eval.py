"""Lightweight regression harness for retrieval + answer behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import LegalQASystem
from .config import SETTINGS


def _is_readable_snippet(snippet: str) -> bool:
    if not snippet or len(snippet.split()) < 6:
        return False
    # Heuristic: unreadable OCR spill often has very low whitespace density.
    if len(snippet) > 40 and snippet.count(" ") < 4:
        return False
    return True


def _has_rule_language(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def run_eval(cases_path: Path | None = None, qa: LegalQASystem | None = None) -> dict:
    path = cases_path or SETTINGS.eval_cases_path
    with path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    qa_system = qa or LegalQASystem()
    results = []

    for case in cases:
        out = qa_system.answer(case["question"])
        should_answer = case.get("should_answer", False)
        should_decline = case.get("should_decline", False)
        should_clarify = case.get("should_request_clarification", False)

        got_answer = not out.get("failure_reason")
        got_decline = out.get("failure_reason") in {"insufficient_evidence", "out_of_scope", "unsupported_claim"}
        got_clarify = bool(out.get("needs_clarification"))

        has_citations = len(out.get("citations", [])) > 0
        citation_ok = (not case.get("must_include_citations", False)) or has_citations

        expected_doc_ids = set(case.get("expected_doc_ids", []))
        found_doc_ids = {c.get("doc_id") for c in out.get("citations", [])}
        retrieval_hit = True
        if expected_doc_ids:
            retrieval_hit = expected_doc_ids.issubset(found_doc_ids)

        preferred_sections = {s.lower() for s in case.get("preferred_section_labels", [])}
        found_sections = {
            str(c.get("section_label", "")).lower()
            for c in out.get("citations", [])
            if c.get("section_label")
        }
        preferred_authority_ok = True
        if preferred_sections:
            preferred_authority_ok = bool(preferred_sections & found_sections)

        snippet_quality_ok = True
        if case.get("require_readable_snippets", False):
            snippet_quality_ok = has_citations and all(
                _is_readable_snippet(str(c.get("evidence_snippet", ""))) for c in out.get("citations", [])[:2]
            )

        doctrinal_terms = case.get("required_rule_terms", [])
        doctrinal_language_ok = True
        if doctrinal_terms:
            doctrinal_language_ok = _has_rule_language(str(out.get("answer", "")), doctrinal_terms)

        confidence_ok = True
        allowed_confidences = case.get("allowed_confidence", [])
        if allowed_confidences:
            confidence_ok = str(out.get("confidence")) in set(allowed_confidences)

        failure_reason_ok = True
        expected_failure_reasons = set(case.get("expected_failure_reasons", []))
        if expected_failure_reasons:
            failure_reason_ok = str(out.get("failure_reason")) in expected_failure_reasons

        verification_ok = True
        expected_verification_statuses = set(case.get("expected_verification_statuses", []))
        verification = out.get("claim_verification") or {}
        got_verification_status = verification.get("status")
        if expected_verification_statuses:
            verification_ok = str(got_verification_status) in expected_verification_statuses

        decline_citations_ok = True
        if should_decline:
            # Unsupported-claim declines may still return citations to show why support failed.
            if out.get("failure_reason") in {"unsupported_claim"}:
                decline_citations_ok = True
            else:
                decline_citations_ok = len(out.get("citations", [])) == 0

        passed = (
            ((not should_answer) or got_answer)
            and ((not should_decline) or got_decline)
            and ((not should_clarify) or got_clarify)
            and citation_ok
            and decline_citations_ok
            and retrieval_hit
            and preferred_authority_ok
            and snippet_quality_ok
            and doctrinal_language_ok
            and confidence_ok
            and failure_reason_ok
            and verification_ok
        )

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "passed": passed,
                "got_failure_reason": out.get("failure_reason"),
                "needs_clarification": out.get("needs_clarification"),
                "citations": len(out.get("citations", [])),
                "decline_citations_ok": decline_citations_ok,
                "retrieval_hit": retrieval_hit,
                "preferred_authority_ok": preferred_authority_ok,
                "snippet_quality_ok": snippet_quality_ok,
                "doctrinal_language_ok": doctrinal_language_ok,
                "confidence_ok": confidence_ok,
                "failure_reason_ok": failure_reason_ok,
                "got_verification_status": got_verification_status,
                "verification_ok": verification_ok,
            }
        )

    passed_count = sum(1 for r in results if r["passed"])
    report = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation cases.")
    _ = parser.parse_args()
    try:
        report = run_eval()
        print(json.dumps(report, indent=2))
    except FileNotFoundError as exc:
        print(
            json.dumps(
                {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "results": [],
                    "error": str(exc),
                    "hint": "Run `python -m src.ingest` then `python -m src.index` first.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
