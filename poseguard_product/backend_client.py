from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class BackendRequestError(RuntimeError):
    """Raised when the FastAPI backend cannot be reached or returns an error."""


@dataclass
class PoseGuardBackendClient:
    base_url: str
    timeout: float = 4.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except Exception as exc:  # pragma: no cover - defensive
            raise BackendRequestError(
                f"Backend returned non-JSON response ({response.status_code})."
            ) from exc

        if response.status_code >= 400:
            detail = payload.get("detail") if isinstance(payload, dict) else payload
            raise BackendRequestError(f"{response.status_code}: {detail}")
        return payload

    def get(self, path: str, **params: Any) -> Dict[str, Any]:
        try:
            response = self.session.get(self._url(path), params=params or None, timeout=self.timeout)
        except requests.RequestException as exc:
            raise BackendRequestError(f"Unable to reach backend at {self.base_url}.") from exc
        return self._handle_response(response)

    def post(self, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            response = self.session.post(self._url(path), json=json_body or {}, timeout=self.timeout)
        except requests.RequestException as exc:
            raise BackendRequestError(f"Unable to reach backend at {self.base_url}.") from exc
        return self._handle_response(response)

    def get_status(self) -> Dict[str, Any]:
        return self.get("/api/v1/status")

    def get_events(self, limit: int = 50) -> Dict[str, Any]:
        return self.get("/api/v1/events", limit=limit)

    def ack_event(self, event_id: str) -> Dict[str, Any]:
        return self.post(f"/api/v1/events/{event_id}/ack")

    def dismiss_event(self, event_id: str) -> Dict[str, Any]:
        return self.post(f"/api/v1/events/{event_id}/dismiss")
