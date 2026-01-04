from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from src.config import settings
from src.rag.prompt import PROMPT_TEMPLATE
from src.rag.vector_store import ChromaVectorStore, RetrievedChunk


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    """Minimal RAG pipeline (retrieve top-k chunks, generate grounded answer)."""

    def __init__(
        self,
        *,
        chroma_dir: Path | None = None,
        collection_name: str = settings.full_chroma_collection,
        embedding_model: str = settings.embedding_model,
        llm_model: str = settings.llm_model,
        llm_max_new_tokens: int = settings.llm_max_new_tokens,
    ):
        self.embedder = SentenceTransformer(embedding_model)

        chroma_dir = Path(chroma_dir) if chroma_dir is not None else settings.full_chroma_dir
        self.store = ChromaVectorStore(persist_dir=chroma_dir, collection_name=collection_name)

        # text2text-generation works for Flan-T5 and many instruction-tuned seq2seq models
        self.generator = pipeline(
            task="text2text-generation",
            model=llm_model,
        )
        self.llm_max_new_tokens = llm_max_new_tokens

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        product_category: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        q_emb = self.embedder.encode([question], normalize_embeddings=True)[0].tolist()
        where: Optional[dict[str, Any]] = None
        if product_category:
            where = {"product_category": product_category}
        return self.store.query(query_embedding=q_emb, top_k=top_k, where=where)

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for i, c in enumerate(chunks, start=1):
            meta = c.metadata or {}
            header = (
                f"[Source {i}] product_category={meta.get('product_category')} "
                f"complaint_id={meta.get('complaint_id')} issue={meta.get('issue')}"
            )
            parts.append(header)
            parts.append(c.text)
        return "\n\n".join(parts)

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        product_category: Optional[str] = None,
    ) -> RAGAnswer:
        sources = self.retrieve(question, top_k=top_k, product_category=product_category)
        if not sources:
            return RAGAnswer(
                answer="I don't have enough information in the retrieved complaints to answer that question.",
                sources=[],
            )

        context = self._format_context(sources)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        out = self.generator(prompt, max_new_tokens=self.llm_max_new_tokens, do_sample=False)
        answer = (out[0].get("generated_text") or "").strip()
        if not answer:
            answer = "I don't have enough information in the retrieved complaints to answer that question."

        return RAGAnswer(answer=answer, sources=sources)
