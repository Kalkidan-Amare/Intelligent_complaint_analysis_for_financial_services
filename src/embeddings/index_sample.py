from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from src.config import settings


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    text: str
    metadata: dict[str, Any]


def stratified_sample(
    df: pd.DataFrame,
    *,
    label_col: str = "product_category",
    sample_size: int = 12_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a stratified sample with (approximately) proportional representation."""
    if label_col not in df.columns:
        raise KeyError(f"Missing column: {label_col}")

    df = df.copy()
    df[label_col] = df[label_col].astype(str)

    counts = df[label_col].value_counts()
    total = int(counts.sum())
    if total == 0:
        return df.head(0)

    # Initial allocation by proportion
    alloc = (counts / total * sample_size).round().astype(int)

    # Ensure at least 1 from each class if possible
    for k in alloc.index:
        if counts[k] > 0 and alloc[k] == 0:
            alloc[k] = 1

    # Adjust to match target size
    diff = int(sample_size - alloc.sum())
    if diff != 0:
        # Sort labels by remaining capacity (descending) for +diff, or by allocation (descending) for -diff
        if diff > 0:
            order = (counts - alloc).sort_values(ascending=False).index.tolist()
            i = 0
            while diff > 0 and i < len(order) * 10:
                label = order[i % len(order)]
                if alloc[label] < counts[label]:
                    alloc[label] += 1
                    diff -= 1
                i += 1
        else:
            order = alloc.sort_values(ascending=False).index.tolist()
            i = 0
            while diff < 0 and i < len(order) * 10:
                label = order[i % len(order)]
                if alloc[label] > 1:
                    alloc[label] -= 1
                    diff += 1
                i += 1

    parts: list[pd.DataFrame] = []
    for label, n in alloc.items():
        n = int(min(n, counts[label]))
        if n <= 0:
            continue
        parts.append(df[df[label_col] == label].sample(n=n, random_state=seed))

    if not parts:
        return df.head(0)

    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def chunk_complaints(
    df: pd.DataFrame,
    *,
    text_col: str = "narrative",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    id_col: str = "Complaint ID",
) -> list[ChunkRecord]:
    if text_col not in df.columns:
        raise KeyError(f"Missing column: {text_col}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_records: list[ChunkRecord] = []

    for _, row in df.iterrows():
        complaint_id = str(row.get(id_col, ""))
        narrative = str(row.get(text_col, "") or "")
        if not narrative.strip():
            continue

        chunks = splitter.split_text(narrative)
        total_chunks = len(chunks)

        for chunk_index, chunk_text in enumerate(chunks):
            doc_id = f"{complaint_id}_{chunk_index}" if complaint_id else f"row_{len(chunk_records)}"
            metadata = {
                "complaint_id": complaint_id,
                "product_category": row.get("product_category"),
                "product": row.get("Product"),
                "issue": row.get("Issue"),
                "sub_issue": row.get("Sub-issue"),
                "company": row.get("Company"),
                "state": row.get("State"),
                "date_received": row.get("Date received"),
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            }
            chunk_records.append(ChunkRecord(doc_id=doc_id, text=chunk_text, metadata=metadata))

    return chunk_records


def build_sample_index(
    df_filtered: pd.DataFrame,
    *,
    persist_dir: Path,
    collection_name: str = "complaints_sample",
    sample_size: int = 12_000,
    seed: int = 42,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = settings.embedding_model,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Build a persisted ChromaDB index on a stratified sample."""

    sample_df = stratified_sample(df_filtered, sample_size=sample_size, seed=seed)
    chunks = chunk_complaints(sample_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Fresh collection to avoid duplicating entries across runs
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer(embedding_model)

    ids = [c.doc_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        batch_emb = model.encode(batch_docs, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(batch_emb.tolist())

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    return {
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "sample_rows": int(sample_df.shape[0]),
        "chunks": int(len(chunks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: chunk + embed + index a stratified sample into ChromaDB")
    parser.add_argument("--input", type=Path, required=True, help="Path to filtered CSV from Task 1")
    parser.add_argument("--persist_dir", type=Path, default=settings.sample_chroma_dir, help="Chroma persist directory")
    parser.add_argument("--collection", type=str, default="complaints_sample", help="Chroma collection name")
    parser.add_argument("--sample_size", type=int, default=settings.sample_size)
    parser.add_argument("--seed", type=int, default=settings.sample_seed)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--embedding_model", type=str, default=settings.embedding_model)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    info = build_sample_index(
        df,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
    )

    print("Built sample index:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
