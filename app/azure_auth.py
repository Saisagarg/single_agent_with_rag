"""Select Azure credentials without requiring Azure CLI inside containers."""
import os

from azure.identity import AzureCliCredential, DefaultAzureCredential


def create_credential():
    mode = os.getenv("RAG_AZURE_CREDENTIAL", "cli").strip().lower()
    if mode == "cli":
        return AzureCliCredential()
    if mode == "default":
        return DefaultAzureCredential()
    raise ValueError("RAG_AZURE_CREDENTIAL must be 'cli' or 'default'")
