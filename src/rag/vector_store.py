from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


class ChromaVectorStore:
    def __init__(self, *, persist_dir: Path, collection_name: str):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_collection(name=self.collection_name)

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: list[RetrievedChunk] = []
        for i in range(len(docs)):
            out.append(RetrievedChunk(text=docs[i], metadata=metas[i] or {}, distance=dists[i] if dists else None))
        return out
