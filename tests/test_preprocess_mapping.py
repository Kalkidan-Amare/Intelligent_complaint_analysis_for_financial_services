import pandas as pd

from src.data.preprocess import map_product_category, preprocess_complaints
from src.data.schema import COL_PRODUCT, COL_NARRATIVE


def test_map_product_category_keywords():
    assert map_product_category("Credit card") == "Credit Cards"
    assert map_product_category("Credit card or prepaid card") == "Credit Cards"
    assert map_product_category("Personal loan") == "Personal Loans"
    assert map_product_category("Savings account") == "Savings Accounts"
    assert map_product_category("Money transfer, virtual currency, or money service") == "Money Transfers"


def test_preprocess_filters_and_cleans():
    df = pd.DataFrame(
        {
            COL_PRODUCT: ["Credit card", "Mortgage"],
            COL_NARRATIVE: ["Hello there", ""],
        }
    )
    out = preprocess_complaints(df).df
    assert out.shape[0] == 1
    assert out.iloc[0]["product_category"] == "Credit Cards"
