from __future__ import annotations

import os
from typing import Tuple

import gradio as gr

from src.rag.pipeline import RAGPipeline


def _format_sources(sources) -> str:
    if not sources:
        return ""

    lines = ["## Sources"]
    for s in sources:
        meta = s.metadata or {}
        header = (
            f"- **{meta.get('product_category')}** | "
            f"complaint_id={meta.get('complaint_id')} | issue={meta.get('issue')}"
        )
        excerpt = (s.text or "").replace("\n", " ").strip()
        if len(excerpt) > 350:
            excerpt = excerpt[:347] + "..."
        lines.append(header)
        lines.append(f"  - {excerpt}")
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    collection = os.getenv("CHROMA_COLLECTION", "complaints_full")
    rag = RAGPipeline(collection_name=collection)

    with gr.Blocks() as demo:
        gr.Markdown("# Complaint Analyst (RAG Chatbot)")

        question = gr.Textbox(label="Question", placeholder="Ask a question about customer complaints...")
        ask_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

        answer_out = gr.Markdown(label="Answer")
        sources_out = gr.Markdown(label="Sources")

        def on_ask(q: str) -> Tuple[str, str]:
            q = (q or "").strip()
            if not q:
                return "Please enter a question.", ""
            res = rag.ask(q, top_k=5)
            return res.answer, _format_sources(res.sources)

        ask_btn.click(fn=on_ask, inputs=[question], outputs=[answer_out, sources_out])
        question.submit(fn=on_ask, inputs=[question], outputs=[answer_out, sources_out])

        def on_clear() -> Tuple[str, str, str]:
            return "", "", ""

        clear_btn.click(fn=on_clear, inputs=[], outputs=[question, answer_out, sources_out])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
