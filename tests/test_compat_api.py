"""Offline compatibility tests for /v1/responses, /responses, /v1/messages, and count_tokens."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = ""
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:") :].strip()
        if data_line:
            events.append((ev_name, json.loads(data_line)))
    return events


class CompatAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_file = Path(self._tmp.name) / "test-compat.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
        os.environ["MUSE_PROXY_KEY"] = "test-compat-key"

        from app.config import reload_settings
        from app.db import init_db
        from app.main import app
        from fastapi.testclient import TestClient

        reload_settings()
        init_db()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        from app.config import reload_settings

        self._tmp.cleanup()
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("MUSE_PROXY_KEY", None)
        reload_settings()

    def test_models_and_single_model_lookup(self) -> None:
        h = {"Authorization": "Bearer test-compat-key"}
        r_list = self.client.get("/models", headers=h)
        self.assertEqual(r_list.status_code, 200)
        self.assertTrue(any(m["id"] == "muse-spark" for m in r_list.json()["data"]))

        r_one = self.client.get("/v1/models/muse-spark", headers={"x-api-key": "test-compat-key"})
        self.assertEqual(r_one.status_code, 200)
        self.assertEqual(r_one.json()["id"], "muse-spark")

        r_miss = self.client.get("/v1/models/no-such-model", headers=h)
        self.assertEqual(r_miss.status_code, 404)

    def test_responses_conversion_and_non_stream(self) -> None:
        from app.compat_api import responses_to_chat

        chat = responses_to_chat(
            {
                "model": "gpt-5.4",
                "instructions": "Be brief",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Follow style"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": {"q": "hi"},
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": [{"type": "output_text", "text": "found"}],
                    },
                    {"type": "reasoning", "summary": []},
                    {"type": "message", "role": "user", "content": "summarize"},
                ],
            }
        )
        roles = [m["role"] for m in chat["messages"]]
        self.assertEqual(roles, ["system", "system", "assistant", "tool", "user"])

        seen_prompts: list[str] = []

        def fake_stream(prompt: str, **kwargs):
            seen_prompts.append(prompt)
            yield "Hello "
            yield "Codex"

        with patch("app.muse_engine.service._sync_chat_stream", side_effect=fake_stream):
            res = self.client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-compat-key"},
                json={"model": "gpt-5.4", "input": "ping", "stream": False},
            )
            res_root = self.client.post(
                "/responses",
                headers={"x-api-key": "test-compat-key"},
                json={"model": "gpt-5.4", "input": "ping2", "stream": False},
            )
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["model"], "gpt-5.4")
        self.assertEqual(body["output"][0]["content"][0]["type"], "output_text")
        self.assertEqual(body["output"][0]["content"][0]["text"], "Hello Codex")
        self.assertGreater(body["usage"]["total_tokens"], 0)
        self.assertEqual(res_root.status_code, 200)
        self.assertEqual(seen_prompts, ["ping", "ping2"])

        bad = self.client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-compat-key"},
            json={"model": "gpt-5.4"},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertIn("error", bad.json())

    def test_responses_streaming_sse(self) -> None:
        def fake_stream(prompt: str, **kwargs):
            yield "chunk1-"
            yield "chunk2"

        with patch("app.muse_engine.service._sync_chat_stream", side_effect=fake_stream):
            res = self.client.post(
                "/v1/responses",
                headers={"Authorization": "Bearer test-compat-key"},
                json={"model": "gpt-5.4", "input": "stream it", "stream": True},
            )
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers["content-type"])
        events = _parse_sse(res.text)
        names = [e[0] for e in events]
        self.assertEqual(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        seqs = [p["sequence_number"] for _, p in events]
        self.assertEqual(seqs, list(range(1, len(events) + 1)))
        completed = events[-1][1]["response"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "chunk1-chunk2")

    def test_anthropic_messages_x_api_key_non_stream_and_stream_and_count_tokens(self) -> None:
        from app.compat_api import anthropic_messages_to_chat

        converted = anthropic_messages_to_chat(
            {
                "model": "claude-opus-4-6",
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.0;"},
                    {"type": "text", "text": "You are Claude Code"},
                ],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "run check"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "internal"},
                            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
                    },
                ],
            }
        )
        self.assertEqual(converted["messages"][0]["role"], "system")
        self.assertEqual(converted["messages"][0]["content"], "You are Claude Code")
        self.assertIn("[tool_use bash", converted["messages"][2]["content"])
        self.assertIn("[tool_result t1]: ok", converted["messages"][3]["content"])

        def fake_stream(prompt: str, **kwargs):
            yield "Anthropic "
            yield "reply"

        with patch("app.muse_engine.service._sync_chat_stream", side_effect=fake_stream):
            res = self.client.post(
                "/v1/messages",
                headers={"x-api-key": "test-compat-key"},
                json={
                    "model": "claude-opus-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            res_stream = self.client.post(
                "/messages",
                headers={"x-api-key": "test-compat-key"},
                json={
                    "model": "claude-opus-4-6",
                    "max_tokens": 1024,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        self.assertEqual(res.status_code, 200, res.text)
        msg = res.json()
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["model"], "claude-opus-4-6")
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertEqual(msg["content"], [{"type": "text", "text": "Anthropic reply"}])

        self.assertEqual(res_stream.status_code, 200)
        events = _parse_sse(res_stream.text)
        self.assertEqual(
            [e[0] for e in events],
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ],
        )
        self.assertEqual(events[2][1]["delta"]["text"], "Anthropic ")
        self.assertEqual(events[3][1]["delta"]["text"], "reply")
        self.assertEqual(events[5][1]["delta"]["stop_reason"], "end_turn")

        tok = self.client.post(
            "/v1/messages/count_tokens",
            headers={"x-api-key": "test-compat-key"},
            json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hello there"}]},
        )
        self.assertEqual(tok.status_code, 200)
        self.assertGreater(tok.json()["input_tokens"], 0)

        bad = self.client.post(
            "/v1/messages",
            headers={"x-api-key": "test-compat-key"},
            json={"model": "claude-opus-4-6", "messages": []},
        )
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()["type"], "error")


if __name__ == "__main__":
    unittest.main()
