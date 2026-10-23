import pytest

from app.azure_auth import create_credential
from azure.identity import AzureCliCredential, DefaultAzureCredential


def test_local_auth_remains_cli(monkeypatch):
    monkeypatch.delenv("RAG_AZURE_CREDENTIAL", raising=False)
    assert isinstance(create_credential(), AzureCliCredential)


def test_container_auth_supports_default_chain(monkeypatch):
    monkeypatch.setenv("RAG_AZURE_CREDENTIAL", "default")
    with create_credential() as credential:
        assert isinstance(credential, DefaultAzureCredential)


def test_invalid_auth_mode_fails_early(monkeypatch):
    monkeypatch.setenv("RAG_AZURE_CREDENTIAL", "typo")
    with pytest.raises(ValueError, match="RAG_AZURE_CREDENTIAL"):
        create_credential()
