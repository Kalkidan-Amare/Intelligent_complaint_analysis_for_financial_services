from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.rag.pipeline import RAGPipeline


def _shorten(text: str, max_len: int = 240) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: qualitative RAG evaluation runner")
    parser.add_argument("--questions", type=Path, required=True, help="JSON file containing a list of questions")
    parser.add_argument("--out", type=Path, default=Path("data/processed/rag_eval.md"), help="Markdown output path")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--collection", type=str, default="complaints_full")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    rag = RAGPipeline(collection_name=args.collection)

    rows: list[list[str]] = []
    for q in questions:
        res = rag.ask(q, top_k=args.top_k)
        src = res.sources[0] if res.sources else None
        src_preview = ""
        if src:
            meta = src.metadata or {}
            src_preview = f"{meta.get('product_category')} | {meta.get('complaint_id')} | {_shorten(src.text)}"

        rows.append([
            q,
            _shorten(res.answer, 400),
            src_preview,
            "",  # Quality score (1-5) - fill manually
            "",  # Comments - fill manually
        ])

    header = ["Question", "Generated Answer", "Retrieved Source (example)", "Quality Score (1-5)", "Comments"]

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        safe = [c.replace("|", "\\|") for c in r]
        lines.append("| " + " | ".join(safe) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
