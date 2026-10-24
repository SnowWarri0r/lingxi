import pytest


@pytest.fixture(autouse=True)
def _no_real_keychain(monkeypatch):
    """Credential sync also reads the macOS Keychain; tests see only their files."""
    monkeypatch.setattr(
        "lingxi.auth.external_sync.ExternalCredentialSync._read_claude_keychain",
        staticmethod(lambda: None))
