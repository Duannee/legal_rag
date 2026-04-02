"""CLI entrypoint for question answering."""

from __future__ import annotations

import argparse
import json

from .answer import LegalQASystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded legal question.")
    parser.add_argument("--question", required=True, help="Question about Cox v. Sony")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k retrieved chunks")
    args = parser.parse_args()

    try:
        qa = LegalQASystem()
        result = qa.answer(args.question, top_k=args.top_k)
        print(json.dumps(result, indent=2))
    except FileNotFoundError as exc:
        print(
            json.dumps(
                {
                    "answer": "",
                    "citations": [],
                    "confidence": "low",
                    "failure_reason": "pipeline_not_initialized",
                    "needs_clarification": False,
                    "claim_verification": None,
                    "error": str(exc),
                    "hint": "Run `python -m src.ingest` then `python -m src.index` first.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
