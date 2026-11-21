"""Per-account reusable Noise sessions (one asyncio loop thread each).

Design goals:
- Reuse bootstrap HTTP TLS via requests.Session
- Reuse an established Noise WS + single shared chat.subscribe
- Serialize Noise encrypt/send with an asyncio lock (nonce order)
- Fan-out subscribe NDJSON events to per-task listeners by node_id
- Idle close; never cache hatch auth_token across new handshakes
- Do not fall back to CDP
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.muse_engine.engine import MuseGenerationError
from app.muse_engine.protocol_client import HatchProtocolClient

logger = logging.getLogger("muse-proxy.protocol_pool")


def _client_is_alive(client: Any) -> bool:
    """Return True when a pooled client holds an open, live WebSocket session."""
    if client is None or getattr(client, "closing", False):
        return False
    if hasattr(client, "is_alive"):
        return bool(client.is_alive())
    return bool(getattr(client, "ws", None))

_HUB_GUARD = threading.Lock()
_WORKERS: dict[str, "AccountNoiseWorker"] = {}


def _cookie_fingerprint(cookies: dict[str, str]) -> str:
    items = sorted((str(k), str(v)[:12]) for k, v in (cookies or {}).items() if v)
    return str(hash(tuple(items)))


@dataclass
class TaskListener:
    node_id: str
    queue: asyncio.Queue
    seen: set[Any] = field(default_factory=set)


class AccountNoiseWorker:
    """Owns one event loop + optional live HatchProtocolClient for an account."""

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop, name=f"muse-noise-{account_id[:8]}", daemon=True
        )
        self.thread.start()
        self._state_lock = threading.Lock()
        self.client: HatchProtocolClient | None = None
        self.cookie_fp: str | None = None
        self.proxy_key: str | None = None
        self.subscription_id: int | None = None
        self.listeners: dict[str, TaskListener] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._scan_offset = 0
        self._refs = 0
        self._last_used = time.monotonic()
        self._closed = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro, timeout: float = 600):
        if self._closed:
            raise MuseGenerationError("protocol worker is closed")
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return fut.result(timeout=timeout)

    def _proxy_key(self, proxy: dict | None) -> str:
        if not proxy:
            return ""
        return f"{proxy.get('server')}|{proxy.get('username') or ''}|{bool(proxy.get('password'))}"

    async def _ensure_client(self, cookies: dict, proxy: dict | None) -> HatchProtocolClient:
        fp = _cookie_fingerprint(cookies)
        pk = self._proxy_key(proxy)
        client = self.client
        alive = bool(
            _client_is_alive(client)
            and self.cookie_fp == fp
            and self.proxy_key == pk
            and self.subscription_id is not None
            and self._dispatch_task is not None
            and not self._dispatch_task.done()
        )
        if alive:
            client.reset_timings()
            client._mark("reuse_hit")
            return client

        if client is not None:
            await self.invalidate_client(client)

        client = HatchProtocolClient(cookies, proxy=proxy)
        client._mark("connect_start")
        await client.connect()
        subscription_id, _ack = await client.subscribe()
        self.client = client
        self.cookie_fp = fp
        self.proxy_key = pk
        self.subscription_id = subscription_id
        self._scan_offset = 0
        if self._dispatch_task is None or self._dispatch_task.done():
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        return client

    def _notify_listeners_error(self, error: str) -> None:
        for listener in list(self.listeners.values()):
            try:
                listener.queue.put_nowait({"type": "pool_error", "error": error})
            except Exception:
                pass

    async def _dispatch_loop(self) -> None:
        """Single reader for the shared subscribe stream; fan-out by device_id."""
        broken_client: HatchProtocolClient | None = None
        try:
            while True:
                client = self.client
                sub_id = self.subscription_id
                if not client or sub_id is None:
                    await asyncio.sleep(0.05)
                    continue
                if (hasattr(client, "is_alive") and not client.is_alive()) or getattr(
                    client, "closing", False
                ):
                    broken_client = client
                    self._notify_listeners_error("noise_closed")
                    break
                response = client.responses.get(sub_id)
                if response is None:
                    await asyncio.sleep(0.05)
                    continue
                raw = bytes(response.body)
                complete = raw.rfind(b"\n", self._scan_offset)
                if complete >= self._scan_offset:
                    chunk = raw[self._scan_offset:complete + 1]
                    self._scan_offset = complete + 1
                    for line in chunk.splitlines():
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            continue
                        payload = event.get("payload") if isinstance(event, dict) else None
                        device = None
                        if isinstance(payload, dict):
                            ctx = payload.get("chat_context") or {}
                            if isinstance(ctx, dict):
                                device = ctx.get("device_id")
                        # Deliver device-targeted events strictly to that listener (dropping
                        # stale/foreign device_ids from subscribe replay or other runs);
                        # only device-less events broadcast to active listeners.
                        targets: list[TaskListener]
                        if device:
                            targets = [self.listeners[device]] if device in self.listeners else []
                        else:
                            targets = list(self.listeners.values())
                        for listener in targets:
                            seq = event.get("seq") if isinstance(event, dict) else None
                            if seq is not None:
                                if seq in listener.seen:
                                    continue
                                listener.seen.add(seq)
                            try:
                                listener.queue.put_nowait(event)
                            except asyncio.QueueFull:
                                pass
                try:
                    await client.wait_update(sub_id, 0.5)
                except asyncio.TimeoutError:
                    pass
                except Exception as exc:
                    # Connection died — fail listeners, detach stale client, and exit.
                    broken_client = client
                    self._notify_listeners_error(str(exc) or "noise_closed")
                    break
        except asyncio.CancelledError:
            raise
        finally:
            if broken_client is not None:
                await self.invalidate_client(broken_client)

    async def acquire(self, cookies: dict, proxy: dict | None) -> HatchProtocolClient:
        self._refs += 1
        self._last_used = time.monotonic()
        return await self._ensure_client(cookies, proxy)

    async def release(self) -> None:
        self._refs = max(0, self._refs - 1)
        self._last_used = time.monotonic()

    async def register_listener(self, node_id: str) -> TaskListener:
        listener = TaskListener(node_id=node_id, queue=asyncio.Queue(maxsize=2000))
        self.listeners[node_id] = listener
        return listener

    async def unregister_listener(self, node_id: str) -> None:
        self.listeners.pop(node_id, None)

    async def invalidate_client(
        self, expected_client: HatchProtocolClient | None = None
    ) -> None:
        """Detach and close a dead or stale client without touching a newer one."""
        client = self.client
        if expected_client is not None and client is not expected_client:
            return
        dispatch = self._dispatch_task
        self._dispatch_task = None
        self.client = None
        self.subscription_id = None
        self._scan_offset = 0
        if dispatch and dispatch is not asyncio.current_task():
            dispatch.cancel()
            await asyncio.gather(dispatch, return_exceptions=True)
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    async def close_client(self) -> None:
        await self.invalidate_client()
        self.listeners.clear()

    def shutdown(self) -> None:
        self._closed = True
        try:
            asyncio.run_coroutine_threadsafe(self.close_client(), self.loop).result(timeout=15)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        try:
            self.loop.close()
        except Exception:
            pass


def get_worker(account_id: str) -> AccountNoiseWorker:
    key = str(account_id or "default")
    with _HUB_GUARD:
        worker = _WORKERS.get(key)
        if worker is None or worker._closed:
            worker = AccountNoiseWorker(key)
            _WORKERS[key] = worker
        return worker


def drop_worker(account_id: str) -> None:
    key = str(account_id or "default")
    with _HUB_GUARD:
        worker = _WORKERS.pop(key, None)
    if worker:
        worker.shutdown()


def has_warm_worker(account_id: str | None) -> bool:
    """True when this account already holds a live Noise + subscribe session."""
    key = str(account_id or "")
    if not key:
        return False
    with _HUB_GUARD:
        worker = _WORKERS.get(key)
    if worker is None or worker._closed:
        return False
    client = worker.client
    if worker.subscription_id is None or not _client_is_alive(client):
        return False
    dispatch = worker._dispatch_task
    if dispatch is not None and dispatch.done():
        return False
    return True


def warm_account_ids() -> list[str]:
    with _HUB_GUARD:
        items = list(_WORKERS.items())
    out = []
    for key, worker in items:
        if has_warm_worker(key):
            out.append(key)
    return out
