"""Public OpenAI-compatible /v1 routes (models, chat, images, videos, media)."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app.auth_admin import get_session_user
from app.auth_api import (
    extract_api_key,
    extract_bearer,
    openai_http_error,
    require_bearer_key,
    source_for_request,
    validate_bearer_key_detailed,
)
from app.call_log import write_call_log
from app.config import get_settings
from app.media_rewrite import is_safe_media_name, request_public_base

router = APIRouter(prefix="/v1", tags=["v1"])
compat_root_router = APIRouter(tags=["v1-compat"])



def _map_engine_error(exc: Exception):
    from app.muse_engine.engine import MuseAuthError, MuseGenerationError, MuseProxyError
    from app.muse_engine.prompts import PromptError
    from app.muse_engine.service import EMPTY_POOL_MSG

    msg = str(exc)
    if isinstance(exc, PromptError):
        raise openai_http_error(400, msg, code="invalid_request") from exc
    if isinstance(exc, MuseAuthError):
        raise openai_http_error(401, msg, code="muse_auth_error") from exc
    if isinstance(exc, MuseGenerationError) and EMPTY_POOL_MSG[:8] in msg:
        raise openai_http_error(502, msg, code="empty_pool") from exc
    if isinstance(exc, MuseGenerationError) and "Idempotency-Key reused" in msg:
        raise openai_http_error(409, msg, code="idempotency_conflict") from exc
    if isinstance(exc, MuseGenerationError) and "queue is full" in msg.lower():
        raise openai_http_error(429, msg, code="queue_full") from exc
    if isinstance(exc, (MuseGenerationError, MuseProxyError)):
        raise openai_http_error(502, msg, code="muse_engine_error") from exc
    raise openai_http_error(502, msg, code="muse_engine_error") from exc


def _task_id_of(resp: Any) -> str | None:
    """queue_image_task returns a JSONResponse; videos return a dict."""
    if isinstance(resp, dict):
        tid = resp.get("id") or resp.get("task_id")
        return str(tid) if tid else None
    body = getattr(resp, "body", None)
    if body:
        try:
            data = json.loads(body)
            tid = data.get("id") or data.get("task_id")
            return str(tid) if tid else None
        except Exception:
            return None
    return None


def _collect_image_refs(body: dict[str, Any]) -> list[str]:
    from app.muse_engine.prompts import reference_list

    extra = {
        "videos": body.get("videos"),
        "audios": body.get("audios"),
        "files": body.get("files"),
    }
    return reference_list(
        body.get("images"),
        [body.get("reference_image"), body.get("image")],
        "images",
        extra,
    )


def _collect_video_refs(body: dict[str, Any]) -> list[str]:
    from app.muse_engine.prompts import reference_list

    extra = {
        "videos": body.get("videos"),
        "audios": body.get("audios"),
        "files": body.get("files"),
    }
    return reference_list(
        body.get("images"),
        [body.get("reference_image"), body.get("image_url"), body.get("image")],
        "images",
        extra,
    )


def _compat_prompt_preview(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    if isinstance(body.get("prompt"), str) and body["prompt"].strip():
        return body["prompt"].strip()[:160]
    msgs = body.get("messages")
    if isinstance(msgs, list) and msgs:
        last = msgs[-1]
        if isinstance(last, dict):
            c = last.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()[:160]
            if isinstance(c, list):
                texts = [str(p.get("text") or "") for p in c if isinstance(p, dict) and p.get("text")]
                if texts:
                    return " ".join(texts).strip()[:160]
    raw_in = body.get("input")
    if isinstance(raw_in, str) and raw_in.strip():
        return raw_in.strip()[:160]
    if isinstance(raw_in, list) and raw_in:
        for item in reversed(raw_in):
            if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip():
                return item["content"].strip()[:160]
            if isinstance(item, dict) and isinstance(item.get("output"), str) and item["output"].strip():
                return item["output"].strip()[:160]
    return None


async def _run_text_compat_endpoint(
    request: Request,
    key: str,
    *,
    log_path: str,
    runner,
    anthropic_errors: bool = False,
):
    from app.muse_engine.engine import MuseAuthError, MuseGenerationError, MuseProxyError

    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        if anthropic_errors:
            raise HTTPException(
                status_code=400,
                detail={"type": "error", "error": {"type": "invalid_request_error", "message": "invalid json body"}},
            )
        raise openai_http_error(400, "invalid json body", code="invalid_request")

    t0 = time.time()
    prompt = _compat_prompt_preview(body)
    model = body.get("model")
    stream = bool(body.get("stream"))
    try:
        resp = await runner(body)
        write_call_log(
            path=log_path,
            method="POST",
            api_type="chat",
            model=model,
            stream=stream,
            status=200,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            prompt_preview=prompt,
        )
        return resp
    except ValueError as exc:
        write_call_log(
            path=log_path,
            method="POST",
            api_type="chat",
            model=model,
            stream=stream,
            status=400,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            error=str(exc),
            prompt_preview=prompt,
        )
        if anthropic_errors:
            raise HTTPException(
                status_code=400,
                detail={"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
            ) from exc
        raise openai_http_error(400, str(exc), code="invalid_request") from exc
    except (MuseGenerationError, MuseAuthError, MuseProxyError, Exception) as exc:
        write_call_log(
            path=log_path,
            method="POST",
            api_type="chat",
            model=model,
            stream=stream,
            status=502,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            error=str(exc),
            prompt_preview=prompt,
        )
        if anthropic_errors:
            raise HTTPException(
                status_code=502,
                detail={"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            ) from exc
        _map_engine_error(exc)


@compat_root_router.get("/models")
@router.get("/models")
async def list_models(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import local_models

    return local_models()


@compat_root_router.get("/models/{model_id:path}")
@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import local_models

    for row in local_models().get("data") or []:
        if row.get("id") == model_id:
            return row
    raise openai_http_error(404, "model not found", code="model_not_found")


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import chat_completions as local_chat

    return await _run_text_compat_endpoint(
        request,
        key,
        log_path="/v1/chat/completions",
        runner=local_chat,
    )


@compat_root_router.post("/responses")
@router.post("/responses")
async def responses_create(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import responses_completions

    return await _run_text_compat_endpoint(
        request,
        key,
        log_path="/v1/responses",
        runner=responses_completions,
    )


@compat_root_router.post("/messages")
@router.post("/messages")
async def messages_create(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import anthropic_messages

    return await _run_text_compat_endpoint(
        request,
        key,
        log_path="/v1/messages",
        runner=anthropic_messages,
        anthropic_errors=True,
    )


@compat_root_router.post("/messages/count_tokens")
@router.post("/messages/count_tokens")
async def messages_count_tokens(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.compat_api import anthropic_count_tokens

    try:
        body = await request.json()
        return anthropic_count_tokens(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"type": "error", "error": {"type": "invalid_request_error", "message": "bad body"}},
        ) from exc


@router.post("/images/generations")
async def images_generations(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.prompts import PromptError, build_image_prompt, reference_kinds
    from app.muse_engine.service import (
        image_response,
        prepare_reference_images,
        queue_image_task,
        run_generation,
    )

    body = await request.json()
    if not isinstance(body, dict):
        raise openai_http_error(400, "invalid json body", code="invalid_request")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise openai_http_error(400, "prompt required", code="invalid_request")
    t0 = time.time()
    acc_label = ""
    _acc = None
    try:
        refs_raw = _collect_image_refs(body)
        refs = await asyncio.to_thread(prepare_reference_images, refs_raw)
        kinds = reference_kinds(refs)
        api_prompt = build_image_prompt(
            prompt,
            size=body.get("size"),
            aspect_ratio=body.get("aspect_ratio"),
            extra=body.get("extra"),
            kinds=kinds,
            ref_count=len(refs),
        )
        timeout = body.get("timeout")
        async_mode = bool(body.get("async") or body.get("async_"))
        base_url = request_public_base(request)
        if async_mode:
            resp = queue_image_task(
                prompt=prompt,
                api_prompt=api_prompt,
                reference_images=refs,
                size=body.get("size"),
                response_format=body.get("response_format") or "url",
                timeout=int(timeout) if timeout else None,
                base_url=base_url,
            )
            write_call_log(
                path="/v1/images/generations",
                method="POST",
                api_type="image",
                model=body.get("model"),
                status=202,
                latency_ms=int((time.time() - t0) * 1000),
                source=source_for_request(request),
                key=key,
                prompt_preview=prompt[:160],
            )
            return resp
        res, _acc = await asyncio.to_thread(
            run_generation,
            api_prompt,
            "image",
            int(timeout) if timeout else None,
            reference_images=refs,
        )
        acc_label = ""
        try:
            from app.pool import store as pool_store

            _row = pool_store.get(str(_acc)) or {}
            acc_label = _row.get("label") or ""
        except Exception:
            acc_label = ""
        out = image_response(
            prompt,
            res,
            size=body.get("size"),
            response_format=body.get("response_format") or "url",
            base_url=base_url,
        )
        media = [out["data"][0].get("url")] if out.get("data") else []
        write_call_log(
            path="/v1/images/generations",
            method="POST",
            api_type="image",
            model=body.get("model"),
            status=200,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            prompt_preview=prompt[:160],
            media_urls=media,
            account=acc_label,
            account_id=str(_acc) if _acc else None,
            log=res.get("_engine_log"),
            attempts=res.get("_engine_attempts"),
        )
        return out
    except PromptError as exc:
        raise openai_http_error(400, str(exc), code="invalid_request") from exc
    except Exception as exc:  # noqa: BLE001
        write_call_log(
            path="/v1/images/generations",
            method="POST",
            api_type="image",
            model=body.get("model"),
            status=502,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            error=str(exc),
            prompt_preview=prompt[:160],
        )
        _map_engine_error(exc)


@router.post("/images/edits")
async def images_edits(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.engine import MAX_REFERENCE_IMAGES
    from app.muse_engine.prompts import PromptError, build_image_prompt, reference_kinds
    from app.muse_engine.service import (
        image_response,
        prepare_reference_images,
        queue_image_task,
        run_generation,
    )

    t0 = time.time()
    content_type = (request.headers.get("content-type") or "").lower()
    prompt = ""
    model = "muse-image"
    size = None
    aspect_ratio = None
    response_format = "url"
    timeout = None
    extra = None
    async_mode = False
    refs_raw: list[str] = []

    try:
        if "multipart/form-data" in content_type:
            form = await request.form(max_files=MAX_REFERENCE_IMAGES, max_fields=40)
            async_mode = str(form.get("async") or "").lower() in ("1", "true", "yes")
            prompt = str(form.get("prompt") or "")
            model = str(form.get("model") or "muse-image")
            size = form.get("size")
            aspect_ratio = form.get("aspect_ratio")
            response_format = str(form.get("response_format") or "url")
            extra = form.get("extra")
            if form.get("timeout") not in (None, ""):
                timeout = int(form.get("timeout"))  # type: ignore[arg-type]
            # collect UploadFile / string refs
            for key_name in ("image", "images", "mask"):
                vals = form.getlist(key_name) if hasattr(form, "getlist") else [form.get(key_name)]
                for v in vals:
                    if v is None:
                        continue
                    if hasattr(v, "read"):
                        data = await v.read()
                        mime = getattr(v, "content_type", None) or "application/octet-stream"
                        refs_raw.append(
                            f"data:{mime};base64,{base64.b64encode(data).decode()}"
                        )
                    elif isinstance(v, str) and v.strip():
                        refs_raw.append(v.strip())
        else:
            body = await request.json()
            if not isinstance(body, dict):
                raise openai_http_error(400, "invalid json body", code="invalid_request")
            prompt = str(body.get("prompt") or "")
            model = body.get("model") or "muse-image"
            size = body.get("size")
            aspect_ratio = body.get("aspect_ratio")
            response_format = body.get("response_format") or "url"
            timeout = body.get("timeout")
            extra = body.get("extra")
            async_mode = bool(body.get("async") or body.get("async_"))
            refs_raw = _collect_image_refs(body)

        prompt = prompt.strip()
        if not prompt:
            raise openai_http_error(400, "prompt required", code="invalid_request")
        if not refs_raw:
            raise openai_http_error(400, "images/edits 需要参考图", code="invalid_request")

        refs = await asyncio.to_thread(prepare_reference_images, refs_raw)
        kinds = reference_kinds(refs)
        api_prompt = build_image_prompt(
            prompt,
            size=size,
            aspect_ratio=aspect_ratio,
            extra=extra,
            kinds=kinds,
            ref_count=len(refs),
        )
        base_url = request_public_base(request)
        if async_mode:
            resp = queue_image_task(
                prompt=prompt,
                api_prompt=api_prompt,
                reference_images=refs,
                size=size,
                response_format=response_format,
                timeout=int(timeout) if timeout else None,
                base_url=base_url,
            )
            write_call_log(
                path="/v1/images/edits",
                method="POST",
                api_type="image",
                model=model,
                status=202,
                upstream_task_id=_task_id_of(resp),
                latency_ms=int((time.time() - t0) * 1000),
                source=source_for_request(request),
                key=key,
                prompt_preview=prompt[:160],
            )
            return resp
        res, _acc = await asyncio.to_thread(
            run_generation,
            api_prompt,
            "image",
            int(timeout) if timeout else None,
            reference_images=refs,
        )
        acc_label = ""
        try:
            from app.pool import store as pool_store

            _row = pool_store.get(str(_acc)) or {}
            acc_label = _row.get("label") or ""
        except Exception:
            acc_label = ""
        out = image_response(
            prompt, res, size=size, response_format=response_format, base_url=base_url
        )
        write_call_log(
            path="/v1/images/edits",
            method="POST",
            api_type="image",
            model=model,
            status=200,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            prompt_preview=prompt[:160],
            media_urls=[out["data"][0].get("url")] if out.get("data") else [],
            account=acc_label,
            account_id=str(_acc) if _acc else None,
            log=res.get("_engine_log"),
            attempts=res.get("_engine_attempts"),
        )
        return out
    except PromptError as exc:
        raise openai_http_error(400, str(exc), code="invalid_request") from exc
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "status_code", None):
            raise
        write_call_log(
            path="/v1/images/edits",
            method="POST",
            api_type="image",
            model=model,
            status=502,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            error=str(exc),
            prompt_preview=(prompt or "")[:160],
        )
        _map_engine_error(exc)


@router.post("/images/tasks")
async def images_tasks_create(
    request: Request,
    key: str = Depends(require_bearer_key),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    from app.muse_engine.prompts import PromptError, build_image_prompt, reference_kinds
    from app.muse_engine.service import prepare_reference_images, queue_image_task

    body = await request.json()
    if not isinstance(body, dict):
        raise openai_http_error(400, "invalid json body", code="invalid_request")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise openai_http_error(400, "prompt required", code="invalid_request")
    t0 = time.time()
    acc_label = ""
    _acc = None
    try:
        refs_raw = _collect_image_refs(body)
        refs = await asyncio.to_thread(prepare_reference_images, refs_raw)
        kinds = reference_kinds(refs)
        api_prompt = build_image_prompt(
            prompt,
            size=body.get("size"),
            aspect_ratio=body.get("aspect_ratio"),
            extra=body.get("extra"),
            kinds=kinds,
            ref_count=len(refs),
        )
        resp = queue_image_task(
            prompt=prompt,
            api_prompt=api_prompt,
            reference_images=refs,
            size=body.get("size"),
            response_format=body.get("response_format") or "url",
            timeout=int(body["timeout"]) if body.get("timeout") else None,
            idempotency_key=idempotency_key,
            base_url=request_public_base(request),
        )
        write_call_log(
            path="/v1/images/tasks",
            method="POST",
            api_type="image",
            model=body.get("model"),
            status=202,
            upstream_task_id=_task_id_of(resp),
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            prompt_preview=prompt[:160],
        )
        return resp
    except PromptError as exc:
        raise openai_http_error(400, str(exc), code="invalid_request") from exc
    except Exception as exc:  # noqa: BLE001
        write_call_log(
            path="/v1/images/tasks",
            method="POST",
            api_type="image",
            model=body.get("model"),
            status=502,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            error=str(exc),
            prompt_preview=prompt[:160],
        )
        _map_engine_error(exc)


@router.get("/images/tasks/{task_id}")
async def images_tasks_get(
    task_id: str,
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import get_image_task_view

    out = get_image_task_view(task_id, base_url=request_public_base(request))
    if not out:
        raise openai_http_error(404, "image task 不存在", code="not_found")
    return out


@router.post("/videos")
@router.post("/videos/generations")
async def videos_create(
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.prompts import PromptError, build_video_prompt, reference_kinds
    from app.muse_engine.service import prepare_reference_images, queue_video_task

    body = await request.json()
    if not isinstance(body, dict):
        raise openai_http_error(400, "invalid json body", code="invalid_request")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise openai_http_error(400, "prompt required", code="invalid_request")
    t0 = time.time()
    try:
        refs_raw = _collect_video_refs(body)
        refs = await asyncio.to_thread(prepare_reference_images, refs_raw)
        kinds = reference_kinds(refs)
        api_prompt = build_video_prompt(
            prompt,
            duration=body.get("duration"),
            size=body.get("size"),
            aspect_ratio=body.get("aspect_ratio"),
            resolution=body.get("resolution"),
            extra=body.get("extra"),
            kinds=kinds,
            ref_count=len(refs),
        )
        out = queue_video_task(
            prompt=prompt,
            api_prompt=api_prompt,
            reference_images=refs,
            timeout=int(body["timeout"]) if body.get("timeout") else None,
            base_url=request_public_base(request),
        )
        write_call_log(
            path="/v1/videos",
            method="POST",
            api_type="video",
            model=body.get("model"),
            status=200,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            prompt_preview=prompt[:160],
            upstream_task_id=out.get("id"),
        )
        return out
    except PromptError as exc:
        raise openai_http_error(400, str(exc), code="invalid_request") from exc
    except Exception as exc:  # noqa: BLE001
        write_call_log(
            path="/v1/videos",
            method="POST",
            api_type="video",
            model=body.get("model"),
            status=502,
            latency_ms=int((time.time() - t0) * 1000),
            source=source_for_request(request),
            key=key,
            error=str(exc),
            prompt_preview=prompt[:160],
        )
        _map_engine_error(exc)


@router.get("/videos/{video_id}")
@router.get("/videos/generations/{video_id}")
async def videos_get(
    video_id: str,
    request: Request,
    key: str = Depends(require_bearer_key),
):
    from app.muse_engine.service import get_video_task_view

    out = get_video_task_view(video_id, base_url=request_public_base(request))
    if not out:
        raise openai_http_error(404, "task 不存在", code="not_found")
    return out


@router.get("/media/{name}")
async def get_media(
    name: str,
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    settings = get_settings()
    if not is_safe_media_name(name):
        raise openai_http_error(400, "非法文件名", code="invalid_media_name")

    key: str | None = None
    source = "external"
    if settings.media_auth_required:
        token = extract_api_key(authorization, x_api_key)
        if token:
            key, kind, row = validate_bearer_key_detailed(authorization, x_api_key)
            request.state.key_kind = kind
            request.state.key_id = int(row["id"]) if row else None
            request.state.key_row = row
            request.state.charge_key = False
            source = "admin_console" if kind == "console" else "external"
        elif get_session_user(request) and settings.console_generate_enabled:
            request.state.key_kind = "console"
            source = "admin_console"
            key = None
        else:
            raise openai_http_error(
                401,
                "Missing API key (Authorization: Bearer <key> or x-api-key)",
                code="missing_api_key",
            )
    else:
        key = extract_api_key(authorization, x_api_key)

    from app.muse_engine.service import media_path

    path = Path(media_path(name))
    if path.is_file():
        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
        }
        return FileResponse(path, media_type=media_types.get(suffix))
    raise openai_http_error(404, "media not found", code="not_found")
