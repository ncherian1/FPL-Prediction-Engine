"""Shared HTTP layer: rate limiting, retries with exponential backoff, raw payload
archiving (raw.source_snapshots + gzipped content-addressed files) and TTL caching."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import psycopg
from psycopg.types.json import Jsonb
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import get_settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, FetchError) and exc.status in RETRYABLE_STATUS


@dataclass
class Response:
    body: bytes
    status: int
    snapshot_id: int
    from_cache: bool
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body)

    def text(self, encodings: tuple[str, ...] = ("utf-8-sig", "latin-1")) -> str:
        for enc in encodings:
            try:
                return self.body.decode(enc)
            except UnicodeDecodeError:
                continue
        return self.body.decode("utf-8", errors="replace")


class Fetcher:
    """One Fetcher per provider. Every request is recorded in raw.source_snapshots."""

    def __init__(
        self,
        conn: psycopg.Connection,
        source: str,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        min_interval: float = 1.0,
        timeout: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        settings = get_settings()
        self.conn = conn
        self.source = source
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.archive_dir = settings.archive_dir / source
        self._last_request = 0.0
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent, **(headers or {})},
        )

    # -- public -----------------------------------------------------------------
    def get(self, path: str, params: dict | None = None, *, cache_ttl: timedelta | None = None) -> Response:
        return self.request("GET", path, params=params, cache_ttl=cache_ttl)

    def post(self, path: str, body: dict, *, cache_ttl: timedelta | None = None) -> Response:
        return self.request("POST", path, json_body=body, cache_ttl=cache_ttl)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
        cache_ttl: timedelta | None = None,
    ) -> Response:
        endpoint = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        snap_params = {k: v for k, v in {"query": params, "body": json_body}.items() if v}
        if cache_ttl is not None:
            cached = self._from_cache(endpoint, snap_params, cache_ttl)
            if cached is not None:
                return cached
        try:
            resp = self._send(method, endpoint, params, json_body)
        except FetchError as exc:
            self._record(endpoint, snap_params, exc.status, False, str(exc)[:2000], None)
            raise
        except httpx.HTTPError as exc:
            self._record(endpoint, snap_params, None, False, repr(exc)[:2000], None)
            raise FetchError(f"{method} {endpoint} failed: {exc!r}") from exc
        snap_id = self._record(endpoint, snap_params, resp.status_code, True, None, resp.content)
        return Response(resp.content, resp.status_code, snap_id, False, dict(resp.headers))

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internals ----------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _send(self, method: str, url: str, params: dict | None, json_body: dict | None) -> httpx.Response:
        @retry(
            retry=retry_if_exception(_retryable),
            wait=wait_exponential_jitter(initial=2, max=120),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
        )
        def attempt() -> httpx.Response:
            self._throttle()
            r = self.client.request(method, url, params=params, json=json_body)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 300))
            if r.status_code >= 400:
                raise FetchError(f"{method} {url} -> HTTP {r.status_code}", r.status_code, r.text[:500])
            return r

        return attempt()

    def _archive(self, body: bytes) -> tuple[str, str]:
        sha = hashlib.sha256(body).hexdigest()
        path = self.archive_dir / sha[:2] / f"{sha}.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(gzip.compress(body))
            tmp.replace(path)
        return sha, str(path)

    def _record(
        self,
        endpoint: str,
        params: dict,
        status: int | None,
        success: bool,
        error: str | None,
        body: bytes | None,
        from_cache: bool = False,
    ) -> int:
        sha = path = None
        if body is not None:
            sha, path = self._archive(body)
        return self.conn.execute(
            """INSERT INTO raw.source_snapshots
               (source_name, endpoint, params, http_status, success, error_message,
                content_sha256, archive_path, bytes, from_cache)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (self.source, endpoint, Jsonb(params) if params else None, status, success, error,
             sha, path, len(body) if body is not None else None, from_cache),
        ).fetchone()[0]

    def _from_cache(self, endpoint: str, params: dict, ttl: timedelta) -> Response | None:
        row = self.conn.execute(
            """SELECT id, archive_path, http_status FROM raw.source_snapshots
               WHERE source_name = %s AND endpoint = %s AND success AND NOT from_cache
                 AND params IS NOT DISTINCT FROM %s
                 AND snapshot_timestamp > now() - %s
               ORDER BY snapshot_timestamp DESC LIMIT 1""",
            (self.source, endpoint, Jsonb(params) if params else None, ttl),
        ).fetchone()
        if not row or not row[1] or not Path(row[1]).exists():
            return None
        body = gzip.decompress(Path(row[1]).read_bytes())
        return Response(body, row[2], row[0], True, {})


def load_archived(snapshot_path: str) -> bytes:
    return gzip.decompress(Path(snapshot_path).read_bytes())
