"""Offline unit checks for protocol helpers and proxy parsing (no network)."""
from __future__ import annotations

import unittest

from app.muse_engine.browser_proxy import (
    normalize_proxy_dict,
    parse_proxy_string,
    public_proxy,
    request_proxies,
)
from app.muse_engine.protocol_engine import (
    PNG_MAGIC,
    _media_candidates,
    _resource_kind,
    image_magic_mime,
    should_use_file_ref,
    to_png_bytes,
)
from app.signup.protocol import normalize_proxy_entry


class ProtocolHelperTests(unittest.TestCase):
    def test_should_use_file_ref_image_inline(self):
        self.assertFalse(should_use_file_ref(20 * 1024 * 1024, "image/webp", []))

    def test_should_use_file_ref_video_without_capability(self):
        self.assertFalse(should_use_file_ref(20 * 1024 * 1024, "video/mp4", []))
        self.assertTrue(should_use_file_ref(60 * 1024 * 1024, "video/mp4", []))

    def test_should_use_file_ref_video_with_capability(self):
        caps = ["video_file_ref_ingest"]
        self.assertTrue(should_use_file_ref(10 * 1024 * 1024, "video/mp4", caps))
        self.assertFalse(should_use_file_ref(9 * 1024 * 1024, "video/mp4", caps))

    def test_media_from_presentation_without_device(self):
        payload = {
            "kind": "image",
            "data": {
                "images": [
                    {
                        "path": "sandbox://workspace/imagine_media/demo.webp",
                        "mime": "image/webp",
                        "width": 1600,
                        "height": 1600,
                    }
                ]
            },
        }
        out: list = []
        _media_candidates(payload, out)
        self.assertEqual(len(out), 1)
        self.assertEqual(_resource_kind(out[0]["resource"]), "image")

    def test_media_from_resources_metadata_json_string(self):
        payload = {
            "resources": [
                {
                    "kind": "video",
                    "metadata": '{"path":"workspace/imagine_media/clip.mp4","mime_type":"video/mp4","size_bytes":123}',
                }
            ]
        }
        out: list = []
        _media_candidates(payload, out)
        self.assertTrue(out)
        self.assertEqual(_resource_kind(out[0]["resource"]), "video")

    def test_skips_user_upload_paths(self):
        payload = {"path": "sandbox://workspace/user/files/ref.mp4", "mime": "video/mp4"}
        out: list = []
        _media_candidates(payload, out)
        self.assertEqual(out, [])


class ProxyParseTests(unittest.TestCase):
    def test_host_port_user_pass_default_http(self):
        # Fictional placeholder credentials only.
        parsed = parse_proxy_string("proxy.example.test:8080:alice:p@ss:with:colons")
        self.assertEqual(parsed["server"], "http://proxy.example.test:8080")
        self.assertEqual(parsed["username"], "alice")
        self.assertEqual(parsed["password"], "p@ss:with:colons")

    def test_exact_five_field_scheme(self):
        parsed = parse_proxy_string("proxy.example.test:1080:bob:secret:socks5")
        self.assertEqual(parsed["server"], "socks5://proxy.example.test:1080")
        self.assertEqual(parsed["username"], "bob")
        self.assertEqual(parsed["password"], "secret")

    def test_url_userinfo_not_reencoded_in_server(self):
        parsed = parse_proxy_string("http://al%2Fice:s%3Fecret@proxy.example.test:8080")
        self.assertEqual(parsed["server"], "http://proxy.example.test:8080")
        self.assertEqual(parsed["username"], "al/ice")
        self.assertEqual(parsed["password"], "s?ecret")
        # request_proxies quotes once for requests.
        req = request_proxies(parsed)
        url = req["proxies"]["http"]
        self.assertIn("al%2Fice:s%3Fecret@", url)
        self.assertTrue(url.startswith("http://"))

    def test_ipv6_bracket_form(self):
        parsed = parse_proxy_string("[2001:db8::1]:8080:carol:pw")
        self.assertEqual(parsed["server"], "http://[2001:db8::1]:8080")
        self.assertEqual(parsed["username"], "carol")
        self.assertEqual(parsed["password"], "pw")

    def test_ipv6_unbracketed_errors(self):
        with self.assertRaises(ValueError):
            parse_proxy_string("2001:db8::1:8080:carol:pw")

    def test_public_proxy_hides_password(self):
        pub = public_proxy(
            {"server": "http://proxy.example.test:8080", "username": "u", "password": "secret"}
        )
        self.assertEqual(pub["username"], "u")
        self.assertTrue(pub["has_password"])
        self.assertNotIn("password", pub)

    def test_normalize_dict_colon_server(self):
        out = normalize_proxy_dict({"server": "proxy.example.test:8080:dave:se:cret"})
        self.assertEqual(out["server"], "http://proxy.example.test:8080")
        self.assertEqual(out["username"], "dave")
        self.assertEqual(out["password"], "se:cret")

    def test_signup_normalize_entry_default_socks(self):
        url = normalize_proxy_entry("proxy.example.test:8080:erin:pw:with:colon")
        self.assertTrue(url.startswith("socks5h://"))
        self.assertIn("@proxy.example.test:8080", url)
        # password with colons must be percent-encoded once in the synthesized URL
        self.assertIn("erin:", url)

    def test_signup_normalize_keeps_sid_placeholder(self):
        url = normalize_proxy_entry("user-sid-{sid}-t-5:secret@us.example.test:3010")
        self.assertTrue(url.startswith("socks5h://"))
        self.assertIn("{sid}", url)
        self.assertNotIn("%7B", url)

    def test_signup_normalize_explicit_http(self):
        url = normalize_proxy_entry("proxy.example.test:8080:erin:pw:http")
        self.assertTrue(url.startswith("http://"))
        self.assertIn("@proxy.example.test:8080", url)


if __name__ == "__main__":
    unittest.main()


class PngDeliveryTests(unittest.TestCase):
    def test_png_magic_and_conversion(self):
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(0, 80, 200)).save(buf, format="WEBP")
        webp = buf.getvalue()
        self.assertEqual(image_magic_mime(webp), "image/webp")
        png = to_png_bytes(webp)
        self.assertTrue(png.startswith(PNG_MAGIC))
        self.assertEqual(image_magic_mime(png), "image/png")
        # extension/MIME/magic must agree for API delivery
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertNotEqual(png[:4], webp[:4])

    def test_err_text_object_not_used_in_python(self):
        # sanity: conversion rejects empty
        with self.assertRaises(Exception):
            to_png_bytes(b"not-an-image")


class ApiErrShapeTests(unittest.TestCase):
    def test_nested_openai_detail_error_object(self):
        """Mirror of static/app.js errText/apiErr for OpenAI-style 401 bodies."""
        def err_text(v):
            if v is None or v == "":
                return ""
            if isinstance(v, str):
                return v
            if not isinstance(v, dict):
                return str(v)
            if isinstance(v.get("message"), str) and v["message"]:
                return v["message"]
            if isinstance(v.get("error"), str) and v["error"]:
                return v["error"]
            if isinstance(v.get("error"), dict):
                return err_text(v["error"])
            if isinstance(v.get("detail"), str) and v["detail"]:
                return v["detail"]
            if isinstance(v.get("detail"), dict):
                return err_text(v["detail"])
            return ""

        body = {
            "detail": {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            }
        }
        msg = err_text(body) or err_text(body.get("detail")) or err_text(body.get("error"))
        self.assertEqual(msg, "Invalid API key")
        self.assertNotIn("object Object", msg.lower())


class TimingDeltaTests(unittest.TestCase):
    def test_exclusive_stage_math(self):
        from app.muse_engine.protocol_engine import timing_deltas
        marks = {
            "session_start": 0.0,
            "session": 1.0,
            "token_start": 1.0,
            "token": 2.5,
            "noise_start": 2.5,
            "noise": 3.0,
            "subscribe_start": 3.0,
            "subscribe": 4.0,
            "connect_start": 0.0,
        }
        d = timing_deltas(marks)
        self.assertEqual(d["session_ms"], 1000.0)
        self.assertEqual(d["token_ms"], 1500.0)
        self.assertEqual(d["noise_ms"], 500.0)
        self.assertEqual(d["subscribe_ms"], 1000.0)
        self.assertEqual(d["bootstrap_ms"], 4000.0)

    def test_reuse_bootstrap(self):
        from app.muse_engine.protocol_engine import timing_deltas
        marks = {"reuse_hit": 10.0, "stream_start": 10.2}
        d = timing_deltas(marks)
        self.assertEqual(d["reuse"], 1.0)
        self.assertAlmostEqual(d["warm_to_stream_ms"], 200.0)


class ProtocolVmStateTests(unittest.TestCase):
    def test_unknown_vm_state_proceeds_to_token_while_stopped_raises(self):
        import asyncio
        from unittest import mock
        from app.muse_engine.engine import MuseGenerationError
        from app.muse_engine.protocol_client import HatchProtocolClient

        client = HatchProtocolClient({"hatch_sess": "demo"})
        calls = []

        def fake_http(method, url, body=None):
            calls.append((method, url))
            if "api/session" in url:
                return {
                    "status": "assigned",
                    "vm_id": "vm-1",
                    "vm_name": "vm-1",
                    "endpoint_url": "wss://vm-1.example.test/",
                    "vm_state": "UNKNOWN",
                    "vm_type": "hatch_vm",
                }
            if "api/hatch/token" in url:
                # Stop right after token request succeeds (avoid real websocket dial)
                raise MuseGenerationError("stop_after_token_probe")
            raise AssertionError(f"unexpected http call: {method} {url}")

        client._http = fake_http  # type: ignore[method-assign]
        with self.assertRaises(MuseGenerationError) as ctx:
            asyncio.run(client.connect())
        self.assertIn("stop_after_token_probe", str(ctx.exception))
        self.assertEqual(
            [u.split("?")[0] for _, u in calls],
            ["https://muse.ai/api/session", "https://muse.ai/api/hatch/token"],
        )

        # STOPPED should try wake and fail if still STOPPED
        client2 = HatchProtocolClient({"hatch_sess": "demo"})
        calls2 = []

        def fake_stopped(method, url, body=None):
            calls2.append((method, url))
            if "api/session" in url:
                return {
                    "status": "assigned",
                    "vm_id": "vm-1",
                    "endpoint_url": "wss://vm-1.example.test/",
                    "vm_state": "STOPPED",
                    "vm_type": "hatch_vm",
                }
            if "vm/wake" in url:
                return {"status": "wake_requested"}
            raise AssertionError(f"unexpected http call: {method} {url}")

        client2._http = fake_stopped  # type: ignore[method-assign]
        with self.assertRaises(MuseGenerationError) as ctx2:
            asyncio.run(client2.connect())
        self.assertIn("state=STOPPED", str(ctx2.exception))
        self.assertTrue(any("vm/wake" in u for _, u in calls2))

    def test_rotation_preserves_real_error_instead_of_empty_pool(self):
        from unittest import mock
        from app.muse_engine import service
        from app.muse_engine.engine import MuseGenerationError

        picked = []

        def fake_pick(*, force_rotate=False, exclude_ids=None):
            if exclude_ids and "acc1" in exclude_ids:
                raise MuseGenerationError(service.EMPTY_POOL_MSG)
            picked.append("acc1")
            return {"id": "acc1", "cookies": {"hatch_sess": "x"}}

        with mock.patch.object(service, "_pick", side_effect=fake_pick), \
             mock.patch.object(service, "_renew_if_needed", side_effect=MuseGenerationError("Muse real error")):
            with self.assertRaises(MuseGenerationError) as ctx:
                service.run_generation("hello", "image", timeout=10)
        self.assertEqual(str(ctx.exception), "Muse real error")


class MediaHostTests(unittest.TestCase):
    def test_request_public_base_prefers_external_host_and_falls_back_on_loopback(self):
        from types import SimpleNamespace
        from app.config import Settings
        from app.media_rewrite import request_public_base, rewrite_media_string

        s = Settings(public_base="http://23.106.46.111:8090")
        req_ext = SimpleNamespace(
            headers={"host": "23.106.46.111:8090"},
            url=SimpleNamespace(scheme="http", netloc="23.106.46.111:8090"),
        )
        self.assertEqual(request_public_base(req_ext, s), "http://23.106.46.111:8090")

        req_proxy = SimpleNamespace(
            headers={
                "host": "127.0.0.1:8090",
                "x-forwarded-host": "muse.example.com",
                "x-forwarded-proto": "https",
            },
            url=SimpleNamespace(scheme="http", netloc="127.0.0.1:8090"),
        )
        self.assertEqual(request_public_base(req_proxy, s), "https://muse.example.com")

        req_local = SimpleNamespace(
            headers={"host": "127.0.0.1:8090"},
            url=SimpleNamespace(scheme="http", netloc="127.0.0.1:8090"),
        )
        self.assertEqual(request_public_base(req_local, s), "http://23.106.46.111:8090")

        req_docker = SimpleNamespace(
            headers={"host": "172.17.0.1:8090"},
            url=SimpleNamespace(scheme="http", netloc="172.17.0.1:8090"),
        )
        self.assertEqual(request_public_base(req_docker, s), "http://23.106.46.111:8090")

        rewritten = rewrite_media_string(
            "http://127.0.0.1:8090/v1/media/demo.png",
            s,
            base_url="https://muse.example.com",
        )
        self.assertEqual(rewritten, "https://muse.example.com/v1/media/demo.png")

    def test_media_public_access_without_auth_by_default(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.config import Settings
        from app.routers.v1 import router as v1_router

        self.assertFalse(Settings(_env_file=None).media_auth_required)
        mini_app = FastAPI()
        mini_app.include_router(v1_router)
        with tempfile.TemporaryDirectory() as td:
            img = Path(td) / "demo.png"
            vid = Path(td) / "demo.mp4"
            img.write_bytes(PNG_MAGIC + b"demo")
            vid.write_bytes(b"\x00\x00\x00\x18ftypmp42")
            public_cfg = Settings(_env_file=None, media_auth_required=False)
            with (
                patch("app.routers.v1.get_settings", return_value=public_cfg),
                patch(
                    "app.muse_engine.service.media_path",
                    side_effect=lambda name: str(Path(td) / name),
                ),
            ):
                client = TestClient(mini_app)
                r_img = client.get("/v1/media/demo.png")
                r_vid = client.get("/v1/media/demo.mp4")
        self.assertEqual(r_img.status_code, 200)
        self.assertEqual(r_img.headers["content-type"], "image/png")
        self.assertEqual(r_vid.status_code, 200)
        self.assertEqual(r_vid.headers["content-type"], "video/mp4")


class StreamIsolationTests(unittest.TestCase):
    def test_pooled_dispatch_drops_foreign_device_events_and_stream_ignores_replay(self):
        import asyncio
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        from app.muse_engine.config import EngineConfig
        from app.muse_engine.protocol_engine import ProtocolEngine, _scoped_media_candidates
        from app.muse_engine.protocol_pool import AccountNoiseWorker

        # 1) _scoped_media_candidates skips older turns in delta.message_done transcript
        done_payload = {
            "chat_context": {"device_id": "node-new"},
            "transcript": {
                "messages": [
                    {
                        "role": "user",
                        "chat_context": {"device_id": "node-old"},
                        "content": [{"type": "text", "text": "old bird"}],
                    },
                    {
                        "role": "assistant",
                        "chat_context": {"device_id": "node-old"},
                        "path": "sandbox://workspace/imagine_media/old_bird.webp",
                    },
                    {
                        "role": "user",
                        "chat_context": {"device_id": "node-new"},
                        "content": [{"type": "text", "text": "new star"}],
                    },
                    {
                        "role": "assistant",
                        "chat_context": {"device_id": "node-new"},
                        "path": "sandbox://workspace/imagine_media/new_star.webp",
                    },
                ]
            },
        }
        scoped: list[dict] = []
        _scoped_media_candidates(
            "delta.message_done",
            done_payload,
            node_id="node-new",
            marker=None,
            output=scoped,
        )
        self.assertEqual(
            [c["resource"]["path"] for c in scoped],
            ["sandbox://workspace/imagine_media/new_star.webp"],
        )

        # 2) Unpooled _stream ignores replayed historical events from foreign device_id
        class FakeClient:
            def __init__(self):
                self.node_id = "node-new"
                self.timings = {}
                self.responses = {
                    1: SimpleNamespace(
                        body=bytearray(
                            (
                                json.dumps(
                                    {
                                        "event": "delta.presentation",
                                        "seq": 5,
                                        "payload": {
                                            "chat_context": {"device_id": "node-old"},
                                            "data": {
                                                "images": [
                                                    {
                                                        "path": "sandbox://workspace/imagine_media/old_bird.webp",
                                                        "mime": "image/webp",
                                                    }
                                                ]
                                            },
                                        },
                                    }
                                )
                                + "\n"
                            ).encode()
                        )
                    )
                }
                self.downloaded: list[str] = []

            def _mark(self, name):
                self.timings[name] = 1.0

            async def connect(self):
                return None

            async def subscribe(self):
                return 1, {"status": "ok", "payload": {"capabilities": []}}

            async def request(self, method, path, body, timeout):
                # Append the live event for node-new only after chat.stream is sent
                self.responses[1].body.extend(
                    (
                        json.dumps(
                            {
                                "event": "delta.presentation",
                                "seq": 16,
                                "payload": {
                                    "chat_context": {"device_id": "node-new"},
                                    "data": {
                                        "images": [
                                            {
                                                "path": "sandbox://workspace/imagine_media/new_star.webp",
                                                "mime": "image/webp",
                                            }
                                        ]
                                    },
                                },
                            }
                        )
                        + "\n"
                    ).encode()
                )
                return SimpleNamespace(status=200, json=lambda: {"message_id": "m2", "session_id": "s1"})

            async def wait_update(self, sub_id, timeout):
                await asyncio.sleep(0.01)

            async def download_bytes(self, resource):
                self.downloaded.append(resource["path"])
                return b"new-star-bytes"

            async def close(self):
                return None

        fake = FakeClient()
        eng = ProtocolEngine(EngineConfig())

        async def run_stream():
            events = []
            async for ev in eng._stream(
                {"hatch_sess": "h"},
                "draw star",
                expect="image",
                timeout=10,
                reference_images=None,
                proxy=None,
                use_pool=False,
            ):
                events.append(ev)
            return events

        with patch("app.muse_engine.protocol_engine.HatchProtocolClient", return_value=fake):
            out = asyncio.run(run_stream())
        self.assertEqual(fake.downloaded, ["sandbox://workspace/imagine_media/new_star.webp"])
        self.assertEqual(out[-1]["data"], b"new-star-bytes")

        # 3) Pooled _dispatch_loop must not deliver foreign device_id to active listeners
        worker = AccountNoiseWorker("test-acc-isolation")
        try:
            async def test_pool_dispatch():
                listener = await worker.register_listener("node-new")
                foreign = json.dumps(
                    {
                        "event": "delta.presentation",
                        "seq": 5,
                        "payload": {"chat_context": {"device_id": "node-old"}},
                    }
                )
                mine = json.dumps(
                    {
                        "event": "delta.presentation",
                        "seq": 16,
                        "payload": {"chat_context": {"device_id": "node-new"}},
                    }
                )
                worker.client = SimpleNamespace(
                    responses={1: SimpleNamespace(body=bytearray((foreign + "\n" + mine + "\n").encode()))},
                    wait_update=lambda *a, **kw: asyncio.sleep(0.05),
                )
                worker.subscription_id = 1
                task = asyncio.create_task(worker._dispatch_loop())
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                got = []
                while not listener.queue.empty():
                    got.append(listener.queue.get_nowait())
                return [e["seq"] for e in got]

            seqs = worker.call(test_pool_dispatch(), timeout=5)
            self.assertEqual(seqs, [16])
        finally:
            worker.client = None
            worker.shutdown()


class NoiseDisconnectRecoveryTests(unittest.TestCase):
    def test_client_liveness_and_send_disconnect_conversion(self):
        import asyncio
        from types import SimpleNamespace
        from websockets.exceptions import ConnectionClosedError
        from websockets.protocol import State
        from app.muse_engine.engine import MuseGenerationError
        from app.muse_engine.protocol_client import (
            HatchProtocolClient,
            MuseNoiseClosedError,
            Cipher,
        )

        client = HatchProtocolClient({"hatch_sess": "h"})
        client.send_cipher = Cipher(b"\x01" * 32)
        client.ws = SimpleNamespace(
            state=State.OPEN,
            close_code=None,
            closed=False,
        )
        self.assertTrue(client.is_alive())

        # Closed state or close_code marks client dead immediately.
        client.ws.state = State.CLOSED
        self.assertFalse(client.is_alive())
        client.ws.state = State.OPEN
        client.ws.close_code = 1006
        self.assertFalse(client.is_alive())

        # When ws.send raises ConnectionClosedError("no close frame received or sent"),
        # send_request must convert it into MuseNoiseClosedError (MuseGenerationError) and mark dead.
        async def run_send_disconnect():
            async def boom(_data):
                raise ConnectionClosedError(None, None)

            client.ws = SimpleNamespace(
                state=State.OPEN,
                close_code=None,
                closed=False,
                send=boom,
            )
            client.closing = False
            client._closed_error = None
            with self.assertRaises(MuseNoiseClosedError) as ctx:
                await client.send_request("POST", "/chat/stream", {"items": []})
            self.assertIsInstance(ctx.exception, MuseGenerationError)
            self.assertIn("no close frame received or sent", str(ctx.exception))
            self.assertFalse(client.is_alive())
            with self.assertRaises(MuseNoiseClosedError):
                await client.wait_update(1, 0.2)

        asyncio.run(run_send_disconnect())

    def test_pool_dead_worker_not_warm_and_chat_stream_recovers(self):
        import asyncio
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from websockets.exceptions import ConnectionClosedError
        from websockets.protocol import State
        from app.muse_engine.config import EngineConfig
        from app.muse_engine.engine import MuseGenerationError
        from app.muse_engine.protocol_client import Cipher, Response, MuseNoiseClosedError
        from app.muse_engine.protocol_engine import ProtocolEngine
        from app.muse_engine.protocol_pool import (
            AccountNoiseWorker,
            _HUB_GUARD,
            _WORKERS,
            drop_worker,
            has_warm_worker,
        )

        acc_id = "test-noise-recover-acc"
        drop_worker(acc_id)

        class FakePooledClient:
            instances: list["FakePooledClient"] = []

            def __init__(self, cookies, *, proxy=None):
                self.cookies = cookies
                self.proxy = proxy
                self.ws = SimpleNamespace(state=State.OPEN, close_code=None, closed=False)
                self.closing = False
                self._closed_error = None
                self.timings = {}
                self.node_id = "node-init"
                self.responses = {1: Response(status=200, body=bytearray())}
                self.waiters = {}
                self.receiver_task = None
                self.heartbeat_task = None
                self.send_cipher = Cipher(b"\x02" * 32)
                self.reqs = []
                FakePooledClient.instances.append(self)

            def is_alive(self):
                return (
                    self.ws is not None
                    and not self.closing
                    and getattr(self.ws, "close_code", None) is None
                    and getattr(self.ws, "state", State.OPEN) == State.OPEN
                )

            def _mark(self, name):
                self.timings[name] = 1.0

            def reset_timings(self):
                self.timings = {}

            def new_task_node(self):
                self.node_id = f"node-{len(self.reqs) + 1}"
                return self.node_id

            async def connect(self):
                return None

            async def subscribe(self):
                return 1, {"status": "ok", "payload": {"capabilities": []}}

            async def wait_update(self, stream_id, timeout=30):
                if self.closing:
                    raise MuseNoiseClosedError("Noise connection closed")
                await asyncio.sleep(min(timeout, 0.02))

            async def request(self, method, path, body=None, timeout=40):
                self.reqs.append((method, path, len(FakePooledClient.instances)))
                # First instance simulates stale reused WS dropping on second request!
                if len(FakePooledClient.instances) == 1 and len(self.reqs) >= 2:
                    self.closing = True
                    raise ConnectionClosedError(None, None)
                node_id = (body or {}).get("node_id") or self.node_id
                ev1 = json.dumps({
                    "event": "delta.text_append",
                    "seq": 100 + len(self.reqs),
                    "payload": {"text": f"ok-conn-{len(FakePooledClient.instances)}", "chat_context": {"device_id": node_id}},
                })
                ev2 = json.dumps({
                    "event": "delta.message_done",
                    "seq": 200 + len(self.reqs),
                    "payload": {"status": "completed", "chat_context": {"device_id": node_id}},
                })
                self.responses[1].body.extend((ev1 + "\n" + ev2 + "\n").encode())
                return Response(status=200, body=bytearray(b"{}"), ended=True)

            async def close(self):
                self.closing = True
                self.ws = None

        eng = ProtocolEngine(EngineConfig())
        try:
            with patch("app.muse_engine.protocol_pool.HatchProtocolClient", FakePooledClient):
                # 1) Initial pooled call warms connection #1
                out1 = "".join(
                    eng.chat_stream(
                        {"hatch_sess": "h1"},
                        "hello 1",
                        account_id=acc_id,
                        use_pool=True,
                    )
                )
                self.assertEqual(out1, "ok-conn-1")
                self.assertTrue(has_warm_worker(acc_id))

                # 2) Second pooled call hits reuse_hit on conn #1, gets ConnectionClosedError
                # ("no close frame received or sent"), invalidates conn #1, reconnects conn #2,
                # and succeeds transparently!
                out2 = "".join(
                    eng.chat_stream(
                        {"hatch_sess": "h1"},
                        "hello 2",
                        account_id=acc_id,
                        use_pool=True,
                    )
                )
                self.assertEqual(out2, "ok-conn-2")
                self.assertEqual(len(FakePooledClient.instances), 2)
                self.assertTrue(has_warm_worker(acc_id))

                # 3) If underlying socket dies while idle, has_warm_worker immediately reports False.
                FakePooledClient.instances[-1].ws.close_code = 1006
                self.assertFalse(has_warm_worker(acc_id))

                # 4) If every connection drops with ConnectionClosedError("no close frame received or sent"),
                # chat_stream raises MuseNoiseClosedError (MuseGenerationError), never raw ConnectionClosedError.
                class AlwaysDropClient(FakePooledClient):
                    async def request(self, method, path, body=None, timeout=40):
                        raise ConnectionClosedError(None, None)

                drop_worker(acc_id)
                with patch("app.muse_engine.protocol_pool.HatchProtocolClient", AlwaysDropClient):
                    with self.assertRaises(MuseNoiseClosedError) as ctx:
                        list(
                            eng.chat_stream(
                                {"hatch_sess": "h1"},
                                "hello 3",
                                account_id=acc_id,
                                use_pool=True,
                            )
                        )
                    self.assertIsInstance(ctx.exception, MuseGenerationError)
                    self.assertFalse(has_warm_worker(acc_id))
        finally:
            drop_worker(acc_id)
