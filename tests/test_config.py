"""Phase 1 tests: settings load from env vars and fail loudly when missing."""
import pytest

from docuquery.config import ConfigError, load_settings


def test_loads_settings_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("EMBED_MODEL", "embed-model-x")
    monkeypatch.setenv("GEN_MODEL", "gen-model-y")
    s = load_settings(load_env_file=False)
    assert s.api_key == "fake-key"
    assert s.embed_model == "embed-model-x"
    assert s.gen_model == "gen-model-y"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("EMBED_MODEL", "a")
    monkeypatch.setenv("GEN_MODEL", "b")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings(load_env_file=False)
