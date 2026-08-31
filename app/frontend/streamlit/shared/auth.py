"""HTTP client for portal APIs. No ML here."""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_BACKEND = "http://127.0.0.1:8000"


class PortalClient:
    def __init__(self, base: str = DEFAULT_BACKEND, token: str | None = None):
        self.base = base.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, path: str, **kwargs) -> Any:
        r = requests.get(self.base + path, headers=self._headers(), timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, **kwargs) -> Any:
        r = requests.post(self.base + path, headers=self._headers(), timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()

    def put(self, path: str, **kwargs) -> Any:
        r = requests.put(self.base + path, headers=self._headers(), timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str) -> Any:
        r = requests.delete(self.base + path, headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()
