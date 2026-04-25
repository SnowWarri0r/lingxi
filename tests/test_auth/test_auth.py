"""Tests for the authentication system."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from persona_agent.auth.manager import AuthManager, AuthError
from persona_agent.auth.models import (
    AuthConfig,
    AuthMethod,
    AuthProfile,
    CredentialType,
    FailureCooldown,
    SecretRef,
    SecretRefSource,
    TokenInfo,
)
from persona_agent.auth.profile_store import ProfileStore
from persona_agent.auth.external_sync import ExternalCredentialSync
from persona_agent.auth.token_store import TokenStore


# ---------------------------------------------------------------------------
# TokenInfo
# ---------------------------------------------------------------------------

class TestTokenInfo:
    def test_not_expired(self):
        token = TokenInfo(access_token="abc", expires_at=datetime.now() + timedelta(hours=1))
        assert not token.is_expired

    def test_expired(self):
        token = TokenInfo(access_token="abc", expires_at=datetime.now() - timedelta(hours=1))
        assert token.is_expired

    def test_no_expiry(self):
        token = TokenInfo(access_token="abc")
        assert not token.is_expired


# ---------------------------------------------------------------------------
# SecretRef
# ---------------------------------------------------------------------------

class TestSecretRef:
    def test_resolve_env(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_123", "my-secret")
        ref = SecretRef(source=SecretRefSource.ENV, ref="TEST_SECRET_123")
        assert ref.resolve() == "my-secret"

    def test_resolve_env_missing(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        ref = SecretRef(source=SecretRefSource.ENV, ref="NONEXISTENT_VAR")
        assert ref.resolve() is None

    def test_resolve_file(self, tmp_path):
        secret_file = tmp_path / "secret.key"
        secret_file.write_text("file-secret-value\n")
        ref = SecretRef(source=SecretRefSource.FILE, ref=str(secret_file))
        assert ref.resolve() == "file-secret-value"

    def test_resolve_file_missing(self):
        ref = SecretRef(source=SecretRefSource.FILE, ref="/nonexistent/path.key")
        assert ref.resolve() is None

    def test_resolve_exec(self):
        ref = SecretRef(source=SecretRefSource.EXEC, ref="echo exec-secret")
        assert ref.resolve() == "exec-secret"

    def test_resolve_exec_failure(self):
        ref = SecretRef(source=SecretRefSource.EXEC, ref="false")
        assert ref.resolve() is None


# ---------------------------------------------------------------------------
# FailureCooldown
# ---------------------------------------------------------------------------

class TestFailureCooldown:
    def test_no_failures(self):
        cd = FailureCooldown()
        assert not cd.is_cooled_down
        assert cd.cooldown_remaining is None

    def test_first_failure_1min(self):
        cd = FailureCooldown()
        cd.record_failure()
        assert cd.consecutive_failures == 1
        assert cd.is_cooled_down  # within 60s window

    def test_cooldown_expires(self):
        cd = FailureCooldown(
            consecutive_failures=1,
            last_failure_at=datetime.now() - timedelta(seconds=61),
        )
        assert not cd.is_cooled_down

    def test_exponential_backoff(self):
        # n=2 -> 300s cooldown
        cd = FailureCooldown(
            consecutive_failures=2,
            last_failure_at=datetime.now() - timedelta(seconds=200),
        )
        assert cd.is_cooled_down  # still within 300s

        cd2 = FailureCooldown(
            consecutive_failures=2,
            last_failure_at=datetime.now() - timedelta(seconds=301),
        )
        assert not cd2.is_cooled_down

    def test_max_cooldown_1hour(self):
        cd = FailureCooldown(
            consecutive_failures=10,
            last_failure_at=datetime.now(),
        )
        remaining = cd.cooldown_remaining
        assert remaining is not None
        assert remaining.total_seconds() <= 3600

    def test_reset(self):
        cd = FailureCooldown()
        cd.record_failure()
        cd.record_failure()
        cd.reset()
        assert cd.consecutive_failures == 0
        assert not cd.is_cooled_down


# ---------------------------------------------------------------------------
# AuthProfile
# ---------------------------------------------------------------------------

class TestAuthProfile:
    def test_resolve_api_key_inline(self):
        p = AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="test",
            api_key="sk-123",
        )
        assert p.resolve_secret() == "sk-123"

    def test_resolve_api_key_ref(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "ref-key")
        p = AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="test",
            api_key="inline-key",
            api_key_ref=SecretRef(source=SecretRefSource.ENV, ref="MY_KEY"),
        )
        # SecretRef takes priority
        assert p.resolve_secret() == "ref-key"

    def test_resolve_oauth(self):
        p = AuthProfile(
            credential_type=CredentialType.OAUTH,
            provider="test",
            oauth_token=TokenInfo(
                access_token="oauth-token",
                expires_at=datetime.now() + timedelta(hours=1),
            ),
        )
        assert p.resolve_secret() == "oauth-token"

    def test_resolve_oauth_expired(self):
        p = AuthProfile(
            credential_type=CredentialType.OAUTH,
            provider="test",
            oauth_token=TokenInfo(
                access_token="expired",
                expires_at=datetime.now() - timedelta(hours=1),
            ),
        )
        assert p.resolve_secret() is None

    def test_sanitize_strips_inline(self):
        p = AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="test",
            api_key="should-be-stripped",
            api_key_ref=SecretRef(source=SecretRefSource.ENV, ref="KEY"),
        )
        sanitized = p.sanitize_for_save()
        assert sanitized.api_key is None
        assert sanitized.api_key_ref is not None

    def test_profile_key(self):
        p = AuthProfile(credential_type=CredentialType.API_KEY, provider="anthropic", label="work")
        assert p.profile_key == "anthropic:work"


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------

class TestProfileStore:
    def test_upsert_and_get(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        profile = AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic",
            api_key="sk-test",
        )
        store.upsert(profile)

        loaded = store.get("anthropic")
        assert loaded is not None
        assert loaded.api_key == "sk-test"

    def test_list_profiles(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY, provider="anthropic", api_key="a",
        ))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY, provider="openai", api_key="b",
        ))

        all_profiles = store.list_profiles()
        assert len(all_profiles) == 2

        anthropic_only = store.list_profiles("anthropic")
        assert len(anthropic_only) == 1

    def test_delete(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY, provider="test", api_key="x",
        ))
        assert store.delete("test") is True
        assert store.get("test") is None
        assert store.delete("test") is False

    def test_ordered_skips_cooled_down(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))

        # Active profile
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY, provider="test", label="active", api_key="a",
        ))
        # Cooled-down profile
        store.upsert(AuthProfile(
            credential_type=CredentialType.OAUTH, provider="test", label="cooled",
            oauth_token=TokenInfo(access_token="t"),
            cooldown=FailureCooldown(consecutive_failures=1, last_failure_at=datetime.now()),
        ))

        ordered = store.get_ordered_for_provider("test")
        assert len(ordered) == 2
        # Active first, cooled-down last
        assert ordered[0].label == "active"
        assert ordered[1].label == "cooled"

    def test_record_and_reset_cooldown(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY, provider="test", api_key="x",
        ))

        store.record_failure("test")
        p = store.get("test")
        assert p.cooldown.consecutive_failures == 1

        store.reset_cooldown("test")
        p = store.get("test")
        assert p.cooldown.consecutive_failures == 0

    def test_migration_from_token_store(self, tmp_path):
        # Create old-style token files
        tokens_dir = tmp_path / "tokens"
        tokens_dir.mkdir()
        token_data = TokenInfo(
            access_token="old-token",
            refresh_token="old-refresh",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        with open(tokens_dir / "anthropic.json", "w") as f:
            json.dump(token_data.model_dump(mode="json"), f, default=str)

        # ProfileStore should migrate
        store = ProfileStore(store_path=str(tmp_path / "auth-profiles.json"))
        profiles = store.list_profiles("anthropic")
        assert len(profiles) == 1
        assert profiles[0].source == "migrated"
        assert profiles[0].oauth_token.access_token == "old-token"

    def test_sanitize_on_save(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="test",
            api_key="inline-secret",
            api_key_ref=SecretRef(source=SecretRefSource.ENV, ref="KEY"),
        ))

        # Read raw JSON to verify inline key was stripped
        with open(tmp_path / "profiles.json") as f:
            data = json.load(f)
        saved = data["profiles"]["test:default"]
        assert saved["api_key"] is None
        assert saved["api_key_ref"]["ref"] == "KEY"


# ---------------------------------------------------------------------------
# ExternalCredentialSync
# ---------------------------------------------------------------------------

class TestExternalSync:
    def test_sync_claude_code(self, tmp_path, monkeypatch):
        # Create fake Claude Code credentials
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        cred_data = {
            "claudeAiOauth": {
                "accessToken": "claude-access-token",
                "refreshToken": "claude-refresh",
                "expiresAt": (datetime.now() + timedelta(hours=1)).timestamp(),
            }
        }
        with open(claude_dir / ".credentials.json", "w") as f:
            json.dump(cred_data, f)

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        sync = ExternalCredentialSync()
        profiles = sync._sync_claude_code()
        assert len(profiles) == 1
        assert profiles[0].provider == "anthropic"
        assert profiles[0].label == "claude_code"
        assert profiles[0].oauth_token.access_token == "claude-access-token"

    def test_sync_codex_cli(self, tmp_path, monkeypatch):
        # Create fake Codex CLI auth
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        auth_data = {
            "access_token": "codex-access-token",
            "refresh_token": "codex-refresh",
            "expires_at": (datetime.now() + timedelta(hours=1)).timestamp(),
        }
        with open(codex_dir / "auth.json", "w") as f:
            json.dump(auth_data, f)

        monkeypatch.setenv("CODEX_HOME", str(codex_dir))

        sync = ExternalCredentialSync()
        sync.invalidate_cache()
        profiles = sync._sync_codex_cli()
        assert len(profiles) == 1
        assert profiles[0].provider == "openai"
        assert profiles[0].source == "codex_cli"

    def test_missing_files_graceful(self, tmp_path, monkeypatch):
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: empty_home))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
        # Prevent macOS Keychain access
        monkeypatch.setattr(
            ExternalCredentialSync, "_read_claude_keychain", staticmethod(lambda: None)
        )

        sync = ExternalCredentialSync()
        sync.invalidate_cache()
        profiles = sync.sync_all()
        assert profiles == []

    def test_cache_ttl(self, tmp_path, monkeypatch):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        with open(claude_dir / ".credentials.json", "w") as f:
            json.dump({"accessToken": "cached-token"}, f)

        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        sync = ExternalCredentialSync()
        p1 = sync._sync_claude_code()
        p2 = sync._sync_claude_code()  # Should hit cache
        assert p1 == p2


# ---------------------------------------------------------------------------
# TokenStore (legacy, backward compat)
# ---------------------------------------------------------------------------

class TestTokenStore:
    def test_save_and_load(self, tmp_path):
        store = TokenStore(store_dir=str(tmp_path / "tokens"))
        token = TokenInfo(
            access_token="test-token-123",
            refresh_token="refresh-456",
            expires_at=datetime.now() + timedelta(hours=1),
        )
        store.save("test-provider", token)
        loaded = store.load("test-provider")
        assert loaded is not None
        assert loaded.access_token == "test-token-123"

    def test_load_nonexistent(self, tmp_path):
        store = TokenStore(store_dir=str(tmp_path / "tokens"))
        assert store.load("nonexistent") is None

    def test_delete(self, tmp_path):
        store = TokenStore(store_dir=str(tmp_path / "tokens"))
        store.save("deleteme", TokenInfo(access_token="to-delete"))
        assert store.delete("deleteme") is True
        assert store.load("deleteme") is None

    def test_list_providers(self, tmp_path):
        store = TokenStore(store_dir=str(tmp_path / "tokens"))
        store.save("provider-a", TokenInfo(access_token="a"))
        store.save("provider-b", TokenInfo(access_token="b"))
        assert set(store.list_providers()) == {"provider-a", "provider-b"}


# ---------------------------------------------------------------------------
# AuthManager (with ProfileStore)
# ---------------------------------------------------------------------------

class TestAuthManager:
    @pytest.mark.asyncio
    async def test_resolve_from_profile_api_key(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic",
            api_key="profile-key",
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        assert result == "profile-key"

    @pytest.mark.asyncio
    async def test_resolve_from_profile_oauth(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.OAUTH,
            provider="anthropic",
            oauth_token=TokenInfo(
                access_token="oauth-key",
                expires_at=datetime.now() + timedelta(hours=1),
            ),
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        assert result == "oauth-key"

    @pytest.mark.asyncio
    async def test_resolve_from_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        assert result == "env-key-123"

    @pytest.mark.asyncio
    async def test_resolve_from_config_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY, api_key="cfg-key")
        result = await auth.resolve_credentials(config)
        assert result == "cfg-key"

    @pytest.mark.asyncio
    async def test_resolve_raises_when_no_credentials(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        with pytest.raises(AuthError, match="未找到"):
            await auth.resolve_credentials(config)

    @pytest.mark.asyncio
    async def test_profile_priority_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic",
            api_key="profile-key",
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        assert result == "profile-key"

    @pytest.mark.asyncio
    async def test_cooled_down_profile_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-fallback")
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic",
            api_key="cooled-key",
            cooldown=FailureCooldown(consecutive_failures=1, last_failure_at=datetime.now()),
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        # Should fall through to env var since profile is cooled down
        assert result == "env-fallback"

    @pytest.mark.asyncio
    async def test_target_specific_profile(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic", label="default", api_key="default-key",
        ))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic", label="work", api_key="work-key",
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY, profile_label="work")
        result = await auth.resolve_credentials(config)
        assert result == "work-key"

    @pytest.mark.asyncio
    async def test_external_sync_on_init(self, tmp_path, monkeypatch):
        # Create fake Claude Code credentials
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        with open(claude_dir / ".credentials.json", "w") as f:
            json.dump({"accessToken": "synced-token"}, f)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))

        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        auth = AuthManager(profile_store=store, auto_sync_external=True)

        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        assert result == "synced-token"

    def test_list_authenticated(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic",
            api_key="valid-key",
        ))
        store.upsert(AuthProfile(
            credential_type=CredentialType.OAUTH,
            provider="openai",
            oauth_token=TokenInfo(
                access_token="expired",
                expires_at=datetime.now() - timedelta(hours=1),
            ),
            source="codex_cli",
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        result = auth.list_authenticated()
        assert len(result) == 2

        by_provider = {r["provider"]: r for r in result}
        assert by_provider["anthropic"]["has_valid_credential"] is True
        assert by_provider["openai"]["has_valid_credential"] is False
        assert by_provider["openai"]["source"] == "codex_cli"

    @pytest.mark.asyncio
    async def test_logout(self, tmp_path):
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic", label="default", api_key="a",
        ))
        store.upsert(AuthProfile(
            credential_type=CredentialType.API_KEY,
            provider="anthropic", label="work", api_key="b",
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        await auth.logout("anthropic")
        assert store.list_profiles("anthropic") == []

    @pytest.mark.asyncio
    async def test_resolve_custom_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_KEY", "custom-123")
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(
            provider="custom", method=AuthMethod.API_KEY, api_key_env_var="MY_CUSTOM_KEY",
        )
        result = await auth.resolve_credentials(config)
        assert result == "custom-123"

    @pytest.mark.asyncio
    async def test_expired_oauth_falls_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        store = ProfileStore(store_path=str(tmp_path / "profiles.json"))
        store.upsert(AuthProfile(
            credential_type=CredentialType.OAUTH,
            provider="anthropic",
            oauth_token=TokenInfo(
                access_token="expired",
                expires_at=datetime.now() - timedelta(hours=1),
                # No refresh token -> can't refresh
            ),
        ))

        auth = AuthManager(profile_store=store, auto_sync_external=False)
        config = AuthConfig(provider="anthropic", method=AuthMethod.API_KEY)
        result = await auth.resolve_credentials(config)
        assert result == "env-key"
