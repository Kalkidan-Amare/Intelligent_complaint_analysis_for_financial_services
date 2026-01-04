from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.schema import (
    COL_COMPLAINT_ID,
    COL_DATE_RECEIVED,
    COL_PRODUCT,
    COL_NARRATIVE,
    COL_ISSUE,
    COL_SUB_ISSUE,
    COL_COMPANY,
    COL_STATE,
)
from src.utils.text import clean_narrative


@dataclass(frozen=True)
class PreprocessResult:
    df: pd.DataFrame
    stats: dict


def _normalize_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def map_product_category(product: object) -> str | None:
    """Map raw CFPB product strings into the project product categories.

    The CFPB dataset has multiple product labels; we map by keyword to keep this
    robust across versions.

    Returns one of:
    - "Credit Cards"
    - "Personal Loans"
    - "Savings Accounts"
    - "Money Transfers"

    or None if not in-scope.
    """
    p = _normalize_str(product).lower()

    if not p:
        return None

    # Credit cards
    if "credit card" in p:
        return "Credit Cards"

    # Personal loans
    if "personal loan" in p:
        return "Personal Loans"

    # Savings accounts
    if "savings" in p and "account" in p:
        return "Savings Accounts"

    # Money transfers (CFPB commonly uses longer variants)
    if "money transfer" in p or "money service" in p or "remittance" in p:
        return "Money Transfers"

    return None


def load_cfpb(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input file type: {path}")


def preprocess_complaints(
    df: pd.DataFrame,
    *,
    allowed_categories: Iterable[str] = (
        "Credit Cards",
        "Personal Loans",
        "Savings Accounts",
        "Money Transfers",
    ),
) -> PreprocessResult:
    df = df.copy()

    if COL_PRODUCT not in df.columns:
        raise KeyError(f"Missing required column: {COL_PRODUCT}")
    if COL_NARRATIVE not in df.columns:
        raise KeyError(f"Missing required column: {COL_NARRATIVE}")

    df["product_category"] = df[COL_PRODUCT].map(map_product_category)

    allowed = set(allowed_categories)
    df = df[df["product_category"].isin(allowed)].copy()

    df["narrative_raw"] = df[COL_NARRATIVE].fillna("").astype(str)
    df["narrative"] = df["narrative_raw"].map(clean_narrative)

    df["has_narrative"] = df["narrative"].str.len().fillna(0).astype(int) > 0
    df = df[df["has_narrative"]].copy()

    df["narrative_word_count"] = df["narrative"].str.split().map(len)

    # Keep canonical metadata (plus anything else present)
    keep_cols = [
        c
        for c in [
            COL_COMPLAINT_ID,
            COL_PRODUCT,
            "product_category",
            COL_ISSUE,
            COL_SUB_ISSUE,
            COL_COMPANY,
            COL_STATE,
            COL_DATE_RECEIVED,
            "narrative",
            "narrative_word_count",
        ]
        if c in df.columns
    ]

    df_out = df[keep_cols].copy()

    stats = {
        "rows_after_filter": int(df_out.shape[0]),
        "product_counts": df_out["product_category"].value_counts(dropna=False).to_dict(),
        "narrative_word_count_summary": df_out["narrative_word_count"].describe().to_dict(),
    }

    return PreprocessResult(df=df_out, stats=stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: EDA + preprocessing for CFPB complaints")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CFPB dataset (.csv or .parquet)")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path (filtered + cleaned)")
    args = parser.parse_args()

    df_raw = load_cfpb(args.input)
    result = preprocess_complaints(df_raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(args.output, index=False)

    # Print a small EDA summary for quick checks.
    print("Saved:", args.output)
    print("Rows:", result.stats["rows_after_filter"])
    print("Product counts:")
    for k, v in result.stats["product_counts"].items():
        print(f"  - {k}: {v}")
    print("Narrative word count summary:")
    for k, v in result.stats["narrative_word_count_summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
