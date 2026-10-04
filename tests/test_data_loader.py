import os
import json
import pytest
from src.data.loader import (
    load_comments,
    load_comments_from_string,
    load_comments_from_dicts,
    save_comments,
    get_data_summary,
)


def test_load_comments_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("id,text,platform,brand\n1,Great service,twitter,Amazon\n2,Late delivery,reddit,Flipkart\n", encoding="utf-8")
    comments = load_comments(str(csv_file))
    assert len(comments) == 2
    assert comments[0]["text"] == "Great service"
    assert comments[1]["brand"] == "Flipkart"


def test_load_comments_json(tmp_path):
    json_file = tmp_path / "test.json"
    data = [
        {"id": "1", "text": "Awesome app", "platform": "instagram"},
        {"id": "2", "text": "Crashing constantly", "platform": "facebook"}
    ]
    json_file.write_text(json.dumps(data), encoding="utf-8")
    comments = load_comments(str(json_file))
    assert len(comments) == 2
    assert comments[0]["text"] == "Awesome app"


def test_load_comments_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_comments("non_existent_file_12345.csv")


def test_load_comments_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_comments(str(empty_file))


def test_load_comments_from_string():
    csv_str = "id,text\n1,Fast delivery\n2,Wrong item\n"
    comments = load_comments_from_string(csv_str)
    assert len(comments) == 2
    assert comments[0]["text"] == "Fast delivery"

    with pytest.raises(ValueError):
        load_comments_from_string("")


def test_load_comments_from_dicts():
    valid = [{"text": "Hello"}, {"text": "World"}]
    assert load_comments_from_dicts(valid) == valid

    with pytest.raises(ValueError):
        load_comments_from_dicts("not a list")

    with pytest.raises(ValueError):
        load_comments_from_dicts(["not a dict"])


def test_save_comments_csv_and_json(tmp_path):
    data = [{"text": "Sample 1", "platform": "twitter"}, {"text": "Sample 2", "platform": "reddit"}]

    csv_dest = tmp_path / "saved.csv"
    save_comments(data, str(csv_dest))
    assert csv_dest.exists()
    reloaded_csv = load_comments(str(csv_dest))
    assert len(reloaded_csv) == 2

    json_dest = tmp_path / "saved.json"
    save_comments(data, str(json_dest))
    assert json_dest.exists()
    reloaded_json = load_comments(str(json_dest))
    assert len(reloaded_json) == 2

    with pytest.raises(ValueError):
        save_comments([], str(tmp_path / "empty_out.csv"))


def test_get_data_summary():
    data = [
        {"text": "Good", "platform": "twitter", "brand": "Amazon"},
        {"text": "Bad", "platform": "twitter", "brand": "Amazon"},
        {"text": "Neutral", "platform": "reddit", "brand": "Swiggy"},
    ]
    summary = get_data_summary(data)
    assert summary["count"] == 3
    assert set(summary["platforms"]) == {"twitter", "reddit"}
    assert set(summary["brands"]) == {"Amazon", "Swiggy"}
