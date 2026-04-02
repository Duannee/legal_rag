import json
from pathlib import Path

from src.eval import run_eval


class _StubQA:
    def answer(self, question: str) -> dict:
        if "legal standard" in question.lower():
            return {
                "answer": "Please narrow your question.",
                "citations": [],
                "confidence": "low",
                "failure_reason": "ambiguous_question",
                "needs_clarification": True,
            }
        return {
            "answer": (
                "The Court says intent can be shown in two ways: first, affirmatively induced infringement; "
                "second, a service tailored to infringement."
            ),
            "citations": [
                {
                    "source_title": "Stub Source",
                    "doc_id": "doc_stub",
                    "chunk_id": "chunk_stub",
                    "section_label": "majority_opinion",
                    "page_span": "1-1",
                    "evidence_snippet": "The Court says intent can be shown in two ways. First, affirmatively induced. Second, tailored to infringement.",
                }
            ],
            "confidence": "high",
            "failure_reason": None,
            "needs_clarification": False,
        }


def test_eval_smoke_runs_and_has_expected_keys() -> None:
    cases = [
        {
            "id": "case_1",
            "question": "What are the two ways intent can be shown?",
            "should_answer": True,
            "must_include_citations": True,
            "preferred_section_labels": ["majority_opinion", "syllabus"],
            "require_readable_snippets": True,
            "required_rule_terms": ["two ways", "affirmatively induced", "tailored to infringement"],
            "allowed_confidence": ["high", "medium"],
        },
        {
            "id": "case_2",
            "question": "What is the legal standard in this case?",
            "should_request_clarification": True,
        },
    ]
    path = Path("tests/_tmp_eval_cases.json")
    path.write_text(json.dumps(cases), encoding="utf-8")
    report = run_eval(cases_path=path, qa=_StubQA())  # type: ignore[arg-type]
    path.unlink(missing_ok=True)

    assert set(report.keys()) == {"total", "passed", "failed", "results"}
    assert report["total"] >= 1

    first = report["results"][0]
    for key in [
        "id",
        "question",
        "passed",
        "got_failure_reason",
        "needs_clarification",
        "citations",
        "retrieval_hit",
    ]:
        assert key in first
