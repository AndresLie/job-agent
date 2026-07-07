import os

from career_copilot.config import load_env_file


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text('NVIDIA_API_KEY="secret"\n', encoding="utf-8")
    load_env_file(path)
    assert os.environ["NVIDIA_API_KEY"] == "secret"


def test_load_env_file_does_not_override_existing_values(tmp_path, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "from-shell")
    path = tmp_path / ".env"
    path.write_text("EXA_API_KEY=from-file\n", encoding="utf-8")
    load_env_file(path)
    assert os.environ["EXA_API_KEY"] == "from-shell"
