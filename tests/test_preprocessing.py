import pytest
from src.preprocessing.cleaner import clean_text, preprocess_for_ml, clean_comments
from src.preprocessing.validator import validate_comment, validate_comments_batch


def test_clean_text_removes_urls_mentions_hashtags():
    text = "Check this out https://example.com/deal @Amazon #BestService"
    cleaned = clean_text(text)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "@amazon" not in cleaned
    assert "#" not in cleaned
    assert "check this out" in cleaned


def test_clean_text_repeated_chars_and_html_entities():
    text = "Sooooo goooood &amp; fast!!!"
    cleaned = clean_text(text)
    assert "soooo" not in cleaned
    assert "gooood" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" not in cleaned
    assert "good" in cleaned


def test_clean_text_edge_cases():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_clean_text_very_short_comments():
    assert clean_text("ok") == "ok"
    assert clean_text("no!") == "no"
    assert clean_text("A") == "a"


def test_preprocess_for_ml():
    text = "The delivery drivers were running late and orders were delayed"
    processed = preprocess_for_ml(text)
    assert isinstance(processed, str)
    assert len(processed) > 0
    # Stopwords like "the", "were", "and" should be removed
    tokens = processed.split()
    assert "the" not in tokens
    assert "and" not in tokens


def test_preprocess_for_ml_empty():
    assert preprocess_for_ml("") == ""
    assert preprocess_for_ml("    ") == ""


def test_clean_comments_list():
    raw_comments = [
        {"id": "1", "text": "Great service! @brand", "platform": "twitter"},
        {"id": "2", "text": "Worst app ever #bug https://fail.com", "platform": "reddit"}
    ]
    cleaned = clean_comments(raw_comments)
    assert len(cleaned) == 2
    assert "cleaned_text" in cleaned[0]
    assert "@brand" not in cleaned[0]["cleaned_text"]
    assert "https://fail.com" not in cleaned[1]["cleaned_text"]


def test_validate_comment_valid():
    valid, msg = validate_comment({"text": "Hello world", "platform": "twitter"})
    assert valid is True
    assert msg == "Valid"


def test_validate_comment_missing_or_empty_text():
    valid, msg = validate_comment({})
    assert valid is False

    valid, msg = validate_comment({"text": ""})
    assert valid is False

    valid, msg = validate_comment({"text": "   "})
    assert valid is False

    valid, msg = validate_comment({"text": 12345})
    assert valid is False

    valid, msg = validate_comment("not a dict")
    assert valid is False


def test_validate_comments_batch():
    batch = [
        {"text": "Valid comment 1"},
        {"text": ""},
        {"text": "Valid comment 2"},
        {"wrong_field": "Missing text"},
        None
    ]
    valid, invalid = validate_comments_batch(batch)
    assert len(valid) == 2
    assert len(invalid) == 3
