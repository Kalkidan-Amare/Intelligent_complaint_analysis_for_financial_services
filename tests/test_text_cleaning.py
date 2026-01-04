from src.utils.text import clean_narrative


def test_clean_narrative_basic():
    text = "I am writing to file a complaint!!!   My card was charged.   "
    cleaned = clean_narrative(text)
    assert "i am writing to file a complaint" not in cleaned
    assert "my card was charged" in cleaned
    assert "  " not in cleaned


def test_clean_narrative_none():
    assert clean_narrative(None) == ""
