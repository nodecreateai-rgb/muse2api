"""Compatibility adapters for OpenAI Responses API and Anthropic Messages API."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

_CLIENT_BILLING_MARKER_RE = re.compile(r"(?m)^x-anthropic-billing-header:[^\n]*\n?")


def estimate_tokens(*objs: Any) -> int:
    """Rough local token count matching runbios2api (~4 UTF-8 bytes per token)."""
    total_bytes = 0
    for obj in objs:
        if obj is None:
            continue
        raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        total_bytes += len(raw)
    return (total_bytes + 3) // 4


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return (len(text.encode("utf-8")) + 3) // 4


def _sse_frame(event: str, payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {raw}\n\n".encode("utf-8")


# ---------------- OpenAI Responses API (/v1/responses, /responses) ----------------


def _resp_image_url(part: dict[str, Any]) -> str:
    val = part.get("image_url")
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        return str(val.get("url") or "").strip()
    return str(part.get("url") or "").strip()


def _resp_content_from(val: Any) -> str | list[dict[str, Any]] | None:
    if isinstance(val, str):
        return val if val.strip() else None
    if not isinstance(val, list):
        return None
    parts: list[dict[str, Any]] = []
    texts: list[str] = []
    has_images = False
    for part in val:
        if isinstance(part, str):
            if part:
                texts.append(part)
                parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        typ = str(part.get("type") or "")
        if typ in ("", "text", "input_text", "output_text"):
            t = part.get("text")
            if isinstance(t, str) and t:
                texts.append(t)
                parts.append({"type": "text", "text": t})
        elif typ in ("input_image", "image_url"):
            url = _resp_image_url(part)
            if url:
                has_images = True
                parts.append({"type": "image_url", "image_url": {"url": url}})
        elif typ == "refusal":
            t = part.get("refusal")
            if isinstance(t, str) and t:
                texts.append(t)
                parts.append({"type": "text", "text": t})
    if not parts:
        return None
    if not has_images and texts:
        return "".join(texts)
    return parts


def _resp_output_text(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        chunks: list[str] = []
        for part in val:
            if not isinstance(part, dict):
                continue
            t = part.get("text")
            if isinstance(t, str) and t:
                chunks.append(t)
                continue
            o = part.get("output")
            if isinstance(o, str) and o:
                chunks.append(o)
        if chunks:
            return "".join(chunks)
    if val is None:
        return ""
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)


def _resp_message_from(item: dict[str, Any]) -> dict[str, Any] | None:
    role = str(item.get("role") or "").strip() or "user"
    if role == "developer":
        role = "system"
    content = _resp_content_from(item.get("content"))
    if content is None:
        txt = str(item.get("text") or "").strip()
        if txt:
            content = txt
    if content is None:
        return None
    return {"role": role, "content": content}


def _resp_tool_call_from(item: dict[str, Any]) -> dict[str, Any]:
    args = item.get("arguments")
    if isinstance(args, str):
        args_str = args if args.strip() else "{}"
    elif args is None:
        args_str = "{}"
    else:
        args_str = json.dumps(args, ensure_ascii=False)
    return {
        "id": str(item.get("call_id") or ""),
        "type": "function",
        "function": {
            "name": str(item.get("name") or ""),
            "arguments": args_str,
        },
    }


def _resp_items_to_messages(items: list[Any]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if not pending_calls:
            return
        msgs.append({"role": "assistant", "content": "", "tool_calls": list(pending_calls)})
        pending_calls.clear()

    for item in items:
        if isinstance(item, str):
            flush_calls()
            if item.strip():
                msgs.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        typ = str(item.get("type") or "")
        if typ in ("", "message"):
            flush_calls()
            msg = _resp_message_from(item)
            if msg:
                msgs.append(msg)
        elif typ == "function_call":
            pending_calls.append(_resp_tool_call_from(item))
        elif typ == "function_call_output":
            flush_calls()
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _resp_output_text(item.get("output")),
                }
            )
        elif typ == "reasoning":
            continue
        elif "role" in item:
            flush_calls()
            msg = _resp_message_from(item)
            if msg:
                msgs.append(msg)
    flush_calls()
    return msgs


def responses_to_chat(body: Any) -> dict[str, Any]:
    """Convert an OpenAI Responses request body into a Chat Completions body."""
    if not isinstance(body, dict):
        raise ValueError("responses: invalid json body")
    model = str(body.get("model") or "").strip() or "muse-spark"
    msgs: list[dict[str, Any]] = []
    inst = str(body.get("instructions") or "").strip()
    if inst:
        msgs.append({"role": "system", "content": inst})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        if raw_input.strip():
            msgs.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        msgs.extend(_resp_items_to_messages(raw_input))
    elif raw_input is not None:
        raise ValueError("responses: input must be a string or array")

    if not msgs:
        raise ValueError("responses: input required")

    chat: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "stream": bool(body.get("stream")),
    }
    for key in ("temperature", "top_p", "timeout"):
        if key in body and body[key] is not None:
            chat[key] = body[key]
    if body.get("max_output_tokens") is not None:
        chat["max_completion_tokens"] = body["max_output_tokens"]
    return chat


def _responses_envelope(
    *,
    resp_id: str,
    created_at: int,
    status: str,
    model: str,
    output: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "error": error,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def responses_non_stream_payload(
    *,
    model: str,
    text: str,
    input_tokens: int,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp_id = "resp_" + uuid.uuid4().hex[:16]
    msg_id = "msg_" + uuid.uuid4().hex[:16]
    created_at = int(time.time())
    output_tokens = estimate_text_tokens(text)
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {
        "id": msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [part],
    }
    payload = _responses_envelope(
        resp_id=resp_id,
        created_at=created_at,
        status="completed",
        model=model,
        output=[item],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if timing is not None:
        payload["muse_timing"] = timing
    return payload


class ResponsesStreamSerializer:
    """Emit OpenAI Responses API SSE events matching runbios2api/internal/responses."""

    def __init__(self, *, model: str, input_tokens: int = 0) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.resp_id = "resp_" + uuid.uuid4().hex[:16]
        self.msg_id = "msg_" + uuid.uuid4().hex[:16]
        self.created_at = int(time.time())
        self.seq = 0
        self.started = False
        self.message_started = False
        self.text_parts: list[str] = []

    def _emit(self, event: str, payload: dict[str, Any]) -> bytes:
        self.seq += 1
        data = {**payload, "sequence_number": self.seq}
        return _sse_frame(event, data)

    def _ensure_started(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        env = _responses_envelope(
            resp_id=self.resp_id,
            created_at=self.created_at,
            status="in_progress",
            model=self.model,
            output=[],
            input_tokens=self.input_tokens,
            output_tokens=0,
        )
        return [
            self._emit("response.created", {"type": "response.created", "response": env}),
            self._emit("response.in_progress", {"type": "response.in_progress", "response": env}),
        ]

    def _ensure_message(self) -> list[bytes]:
        out = self._ensure_started()
        if self.message_started:
            return out
        self.message_started = True
        out.append(
            self._emit(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": self.msg_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
            )
        )
        out.append(
            self._emit(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": self.msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        )
        return out

    def on_start(self) -> list[bytes]:
        return self._ensure_started()

    def on_delta(self, delta: str) -> list[bytes]:
        if not delta:
            return []
        out = self._ensure_message()
        self.text_parts.append(delta)
        out.append(
            self._emit(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": self.msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                },
            )
        )
        return out

    def on_done(self, full_text: str, timing: dict[str, Any] | None = None) -> list[bytes]:
        out = self._ensure_message()
        text = full_text or "".join(self.text_parts)
        part = {"type": "output_text", "text": text, "annotations": []}
        out.append(
            self._emit(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": self.msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
            )
        )
        out.append(
            self._emit(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": self.msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": part,
                },
            )
        )
        item = {
            "id": self.msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        }
        out.append(
            self._emit(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": item,
                },
            )
        )
        env = _responses_envelope(
            resp_id=self.resp_id,
            created_at=self.created_at,
            status="completed",
            model=self.model,
            output=[item],
            input_tokens=self.input_tokens,
            output_tokens=estimate_text_tokens(text),
        )
        if timing is not None:
            env["muse_timing"] = timing
        out.append(
            self._emit(
                "response.completed",
                {"type": "response.completed", "response": env},
            )
        )
        return out

    def on_error(self, message: str) -> list[bytes]:
        out = self._ensure_started()
        err = {"code": "muse_engine_error", "message": message}
        env = _responses_envelope(
            resp_id=self.resp_id,
            created_at=self.created_at,
            status="failed",
            model=self.model,
            output=[],
            input_tokens=self.input_tokens,
            output_tokens=estimate_text_tokens("".join(self.text_parts)),
            error=err,
        )
        out.append(self._emit("response.failed", {"type": "response.failed", "response": env}))
        out.append(self._emit("error", {"type": "error", "error": err}))
        return out


# ---------------- Anthropic Messages API (/v1/messages, /v1/messages/count_tokens) ----------------


def _strip_billing_marker(text: str) -> str:
    return _CLIENT_BILLING_MARKER_RE.sub("", text).strip()


def _anthropic_system_text(system: Any) -> str:
    if isinstance(system, str):
        return _strip_billing_marker(system)
    if isinstance(system, list):
        chunks: list[str] = []
        for blk in system:
            if isinstance(blk, str):
                cleaned = _strip_billing_marker(blk)
                if cleaned:
                    chunks.append(cleaned)
            elif isinstance(blk, dict):
                txt = str(blk.get("text") or "")
                cleaned = _strip_billing_marker(txt)
                if cleaned:
                    chunks.append(cleaned)
        return "\n".join(chunks)
    return ""


def _anthropic_block_to_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    typ = str(block.get("type") or "")
    if typ in ("", "text"):
        return str(block.get("text") or "")
    if typ == "thinking":
        return ""
    if typ in ("tool_use", "server_tool_use", "mcp_tool_use"):
        name = str(block.get("name") or "")
        inp = block.get("input")
        try:
            inp_s = json.dumps(inp if inp is not None else {}, ensure_ascii=False)
        except Exception:
            inp_s = str(inp)
        return f"[tool_use {name}({inp_s})]"
    if typ == "tool_result":
        raw_content = block.get("content")
        if isinstance(raw_content, str):
            inner = raw_content
        elif isinstance(raw_content, list):
            inner = "\n".join(
                _anthropic_block_to_text(sub) for sub in raw_content if _anthropic_block_to_text(sub)
            )
        else:
            inner = "" if raw_content is None else str(raw_content)
        tid = str(block.get("tool_use_id") or "")
        return f"[tool_result {tid}]: {inner}" if tid else inner
    if typ == "document":
        src = block.get("source")
        if isinstance(src, dict) and src.get("type") == "text":
            return str(src.get("data") or "")
        return str(block.get("text") or "")
    return ""


def anthropic_messages_to_chat(body: Any) -> dict[str, Any]:
    """Convert an Anthropic Messages request into internal Chat Completions shape."""
    if not isinstance(body, dict):
        raise ValueError("messages: invalid json body")
    model = str(body.get("model") or "").strip() or "muse-spark"
    raw_msgs = body.get("messages")
    if not isinstance(raw_msgs, list) or not raw_msgs:
        raise ValueError("messages: messages required")

    msgs: list[dict[str, Any]] = []
    sys_text = _anthropic_system_text(body.get("system"))
    if sys_text:
        msgs.append({"role": "system", "content": sys_text})

    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").strip() or "user"
        content = m.get("content")
        if isinstance(content, list):
            parts = [_anthropic_block_to_text(b) for b in content]
            text = "\n".join(p for p in parts if p)
        elif content is None:
            text = ""
        else:
            text = str(content)
        if text.strip():
            msgs.append({"role": role, "content": text})

    if not msgs:
        raise ValueError("messages: messages required")

    out: dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "stream": bool(body.get("stream")),
    }
    for key in ("temperature", "top_p", "timeout"):
        if key in body and body[key] is not None:
            out[key] = body[key]
    return out


def anthropic_count_tokens(body: Any) -> dict[str, int]:
    """Match runbios2api's local /v1/messages/count_tokens byte-length estimate."""
    if not isinstance(body, dict):
        raise ValueError("messages: invalid json body")
    return {
        "input_tokens": estimate_tokens(
            body.get("system"),
            body.get("messages"),
            body.get("tools"),
        )
    }


def anthropic_non_stream_payload(
    *,
    model: str,
    text: str,
    input_tokens: int,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": estimate_text_tokens(text),
        },
    }
    if timing is not None:
        payload["muse_timing"] = timing
    return payload


class AnthropicStreamSerializer:
    """Emit Anthropic Messages SSE events matching runbios2api / Claude Code expectations."""

    def __init__(self, *, model: str, input_tokens: int = 0) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.msg_id = "msg_" + uuid.uuid4().hex[:24]
        self.started = False
        self.text_parts: list[str] = []

    def _ensure_started(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        return [
            _sse_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": 0,
                        },
                    },
                },
            ),
            _sse_frame(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]

    def on_start(self) -> list[bytes]:
        return self._ensure_started()

    def on_delta(self, delta: str) -> list[bytes]:
        if not delta:
            return []
        out = self._ensure_started()
        self.text_parts.append(delta)
        out.append(
            _sse_frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": delta},
                },
            )
        )
        return out

    def on_done(self, full_text: str, timing: dict[str, Any] | None = None) -> list[bytes]:
        out = self._ensure_started()
        text = full_text or "".join(self.text_parts)
        out.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
        msg_delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": estimate_text_tokens(text)},
        }
        if timing is not None:
            msg_delta["muse_timing"] = timing
        out.append(_sse_frame("message_delta", msg_delta))
        out.append(_sse_frame("message_stop", {"type": "message_stop"}))
        return out

    def on_error(self, message: str) -> list[bytes]:
        return [
            _sse_frame(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": message},
                },
            )
        ]
