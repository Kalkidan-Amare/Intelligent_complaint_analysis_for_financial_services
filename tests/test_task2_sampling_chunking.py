import pandas as pd

from src.embeddings.index_sample import stratified_sample, chunk_complaints


def test_stratified_sample_size_and_labels():
    df = pd.DataFrame(
        {
            "product_category": ["A"] * 80 + ["B"] * 20,
            "narrative": ["x"] * 100,
            "Complaint ID": list(range(100)),
        }
    )
    s = stratified_sample(df, sample_size=20, seed=0)
    assert s.shape[0] == 20
    assert set(s["product_category"].unique()) == {"A", "B"}


def test_chunk_complaints_splits_long_text():
    long_text = "a" * 1200
    df = pd.DataFrame(
        {
            "product_category": ["A"],
            "narrative": [long_text],
            "Complaint ID": ["123"],
        }
    )
    chunks = chunk_complaints(df, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert all(c.metadata["complaint_id"] == "123" for c in chunks)
