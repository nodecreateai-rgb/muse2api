"""Rewrite upstream /v1/media URLs to PUBLIC_BASE."""

from __future__ import annotations

import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlparse, urlsplit

from app.config import Settings, get_settings

_MEDIA_PATH_RE = re.compile(r"^(?P<prefix>https?://[^/]+)?(?P<path>/v1/media/(?P<name>[^/?#]+))(?P<rest>[^\"']*)?$")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]"}


def _is_loopback_or_docker_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower().strip("[]")
    if not host:
        return True
    if host in _LOOPBACK_HOSTS or host == "host.docker.internal":
        return True
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_unspecified or ip.is_link_local:
            return True
        # Docker bridge networks (172.16.0.0/12) are unreachable from external browsers.
        if ip.version == 4 and ip in ipaddress.ip_network("172.16.0.0/12"):
            return True
    except ValueError:
        pass
    return False


def _configured_nonlocal_base(settings: Settings) -> str:
    raw = (getattr(settings, "public_base", None) or "").strip().rstrip("/")
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
        if _is_loopback_or_docker_host(parsed.hostname or ""):
            return ""
    except Exception:
        return ""
    return raw


def request_public_base(request: Any, settings: Settings | None = None) -> str:
    """Resolve the outward base URL from the incoming request Host/X-Forwarded-* headers.

    If the caller connected via a loopback or Docker-internal host (e.g. 127.0.0.1
    or 172.17.0.1), prefer configured non-local PUBLIC_BASE when present.
    """
    settings = settings or get_settings()
    configured = (getattr(settings, "public_base", None) or "").strip().rstrip("/")
    nonlocal_cfg = _configured_nonlocal_base(settings)
    if request is None:
        return configured

    headers = getattr(request, "headers", None) or {}
    xf_host = str(headers.get("x-forwarded-host") or "").split(",")[0].strip()
    host = xf_host or str(headers.get("host") or "").strip()
    if not host:
        url = getattr(request, "url", None)
        host = str(getattr(url, "netloc", "") or "").strip()
    if not host or any(ch in host for ch in ("/", "\\", "@", "?", "#", " ", "\t", "\r", "\n")):
        return configured

    try:
        hostname = urlsplit(f"//{host}").hostname or ""
    except ValueError:
        return configured
    if not hostname:
        return configured

    if _is_loopback_or_docker_host(hostname) and nonlocal_cfg:
        return nonlocal_cfg

    xf_proto = str(headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if xf_proto in ("http", "https"):
        scheme = xf_proto
    else:
        url = getattr(request, "url", None)
        scheme = str(getattr(url, "scheme", "") or "http").lower()
        if scheme not in ("http", "https"):
            scheme = "http"
    return f"{scheme}://{host}".rstrip("/")


def is_safe_media_name(name: str) -> bool:
    if not name or len(name) > 200:
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    if name.startswith("."):
        return False
    return True


def rewrite_media_string(value: str, settings: Settings, *, base_url: str | None = None) -> str:
    if "/v1/media/" not in value:
        return value
    public = (base_url or settings.public_base).rstrip("/")

    # Relative
    if value.startswith("/v1/media/"):
        return public + value

    # Any host …/v1/media/name
    parsed = urlparse(value)
    if parsed.path.startswith("/v1/media/"):
        name = parsed.path[len("/v1/media/") :]
        if is_safe_media_name(name.split("/")[0]):
            q = f"?{parsed.query}" if parsed.query else ""
            return f"{public}/v1/media/{name.split('/')[0]}{q}"
    return value


def rewrite_media_urls(obj: Any, settings: Settings, *, base_url: str | None = None) -> Any:
    if isinstance(obj, str):
        return rewrite_media_string(obj, settings, base_url=base_url)
    if isinstance(obj, list):
        return [rewrite_media_urls(x, settings, base_url=base_url) for x in obj]
    if isinstance(obj, dict):
        return {k: rewrite_media_urls(v, settings, base_url=base_url) for k, v in obj.items()}
    return obj


def maybe_rewrite_json_bytes(
    raw: bytes, settings: Settings, *, base_url: str | None = None
) -> tuple[bytes, Any | None]:
    """Return (bytes, parsed_or_none). Parsed is post-rewrite if JSON object/list."""
    if not settings.rewrite_media_urls or not raw:
        try:
            return raw, json.loads(raw)
        except Exception:
            return raw, None
    try:
        parsed = json.loads(raw)
    except Exception:
        return raw, None
    rewritten = rewrite_media_urls(parsed, settings, base_url=base_url)
    return (
        json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        rewritten,
    )
