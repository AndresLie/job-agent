import json
from pathlib import Path

import pytest

from career_copilot.wizard import parse_choice, parse_yes_no, write_json


def test_parse_yes_no_defaults_and_values():
    assert parse_yes_no("", default=True) is True
    assert parse_yes_no("n", default=True) is False
    assert parse_yes_no("yes") is True


def test_parse_yes_no_rejects_invalid():
    with pytest.raises(ValueError):
        parse_yes_no("maybe")


def test_parse_choice_defaults_and_validates():
    assert parse_choice("", {"text", "file"}, "text") == "text"
    assert parse_choice("file", {"text", "file"}, "text") == "file"
    with pytest.raises(ValueError):
        parse_choice("url", {"text", "file"}, "text")


def test_write_json(tmp_path: Path):
    path = write_json(tmp_path / "out" / "brief.json", {"ok": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
