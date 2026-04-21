"""OAuth 2.0 Device Authorization Grant (RFC 8628) implementation.

This is the same flow used by Claude Code (`claude login`) and Codex CLI:
1. Client requests a device code from the authorization server
2. User visits a URL and enters the code (or uses a direct link)
3. Client polls the token endpoint until the user completes authorization
"""

from __future__ import annotations

import asyncio
import sys
import webbrowser

import httpx

from persona_agent.auth.models import DeviceCodeResponse, TokenInfo


class DeviceFlowError(Exception):
    """Errors during device flow authentication."""


class DeviceFlowAuth:
    """Handles OAuth 2.0 Device Authorization Grant."""

    def __init__(
        self,
        client_id: str,
        device_auth_url: str,
        token_url: str,
        scopes: list[str] | None = None,
        client_secret: str | None = None,
    ):
        self.client_id = client_id
        self.device_auth_url = device_auth_url
        self.token_url = token_url
        self.scopes = scopes or []
        self.client_secret = client_secret

    async def request_device_code(self) -> DeviceCodeResponse:
        """Step 1: Request a device code from the authorization server."""
        data: dict[str, str] = {"client_id": self.client_id}
        if self.scopes:
            data["scope"] = " ".join(self.scopes)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.device_auth_url,
                data=data,
                headers={"Accept": "application/json"},
            )

            if response.status_code != 200:
                raise DeviceFlowError(
                    f"Failed to request device code: {response.status_code} {response.text}"
                )

            return DeviceCodeResponse.model_validate(response.json())

    async def poll_for_token(
        self,
        device_code: str,
        interval: int = 5,
        expires_in: int = 900,
    ) -> TokenInfo:
        """Step 3: Poll the token endpoint until the user completes authorization."""
        from datetime import datetime, timedelta

        deadline = datetime.now() + timedelta(seconds=expires_in)

        data: dict[str, str] = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code,
            "client_id": self.client_id,
        }
        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with httpx.AsyncClient() as client:
            while datetime.now() < deadline:
                response = await client.post(
                    self.token_url,
                    data=data,
                    headers={"Accept": "application/json"},
                )

                body = response.json()

                if response.status_code == 200 and "access_token" in body:
                    expires_at = None
                    if "expires_in" in body:
                        expires_at = datetime.now() + timedelta(seconds=body["expires_in"])

                    return TokenInfo(
                        access_token=body["access_token"],
                        refresh_token=body.get("refresh_token"),
                        token_type=body.get("token_type", "Bearer"),
                        expires_at=expires_at,
                        scope=body.get("scope"),
                    )

                error = body.get("error", "")

                if error == "authorization_pending":
                    await asyncio.sleep(interval)
                    continue
                elif error == "slow_down":
                    interval = min(interval + 5, 30)
                    await asyncio.sleep(interval)
                    continue
                elif error == "expired_token":
                    raise DeviceFlowError("Device code expired. Please try again.")
                elif error == "access_denied":
                    raise DeviceFlowError("Authorization denied by user.")
                else:
                    raise DeviceFlowError(f"Token error: {error} - {body.get('error_description', '')}")

        raise DeviceFlowError("Device code expired (timeout).")

    async def login_interactive(self) -> TokenInfo:
        """Run the full interactive device flow login.

        Displays the code to the user and opens the browser.
        """
        # Step 1: Get device code
        device_resp = await self.request_device_code()

        # Step 2: Display to user and open browser
        print(f"\n{'='*50}")
        print(f"  请在浏览器中完成登录")
        print(f"{'='*50}")

        if device_resp.verification_uri_complete:
            print(f"\n  打开链接: {device_resp.verification_uri_complete}")
        else:
            print(f"\n  1. 打开: {device_resp.verification_uri}")
            print(f"  2. 输入验证码: {device_resp.user_code}")

        print(f"\n  等待授权中... (超时: {device_resp.expires_in}秒)")
        print(f"{'='*50}\n")

        # Try to open browser
        url = device_resp.verification_uri_complete or device_resp.verification_uri
        try:
            webbrowser.open(url)
        except Exception:
            pass  # Browser open is best-effort

        # Step 3: Poll for token
        token = await self.poll_for_token(
            device_code=device_resp.device_code,
            interval=device_resp.interval,
            expires_in=device_resp.expires_in,
        )

        print("  ✓ 登录成功！\n")
        return token


async def refresh_token(
    token_url: str,
    client_id: str,
    refresh_token: str,
    client_secret: str | None = None,
) -> TokenInfo:
    """Refresh an expired OAuth token."""
    from datetime import datetime, timedelta

    data: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data=data,
            headers={"Accept": "application/json"},
        )

        if response.status_code != 200:
            raise DeviceFlowError(f"Token refresh failed: {response.status_code} {response.text}")

        body = response.json()
        expires_at = None
        if "expires_in" in body:
            expires_at = datetime.now() + timedelta(seconds=body["expires_in"])

        return TokenInfo(
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token", refresh_token),
            token_type=body.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=body.get("scope"),
        )
