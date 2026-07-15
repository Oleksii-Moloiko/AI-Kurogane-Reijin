import httpx
import pytest

from backend.llm.errors import ModelUnavailableError, ProviderUnavailableError
from backend.llm.providers.ollama_provider import OllamaProvider


def test_ollama_healthcheck_validates_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://localhost:11434")
    monkeypatch.setattr(provider, "list_models", lambda: ["llama3.2"])

    with pytest.raises(ModelUnavailableError):
        provider.healthcheck("gemma3")


def test_ollama_list_models_wraps_httpx_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://localhost:11434")

    def fail() -> None:
        request = httpx.Request("GET", "http://localhost:11434/api/tags")
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(provider.client, "list", fail)

    with pytest.raises(ProviderUnavailableError):
        provider.list_models()


def test_ollama_healthcheck_accepts_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider("http://localhost:11434")
    monkeypatch.setattr(provider, "list_models", lambda: ["gemma3:latest"])

    provider.healthcheck("gemma3")
