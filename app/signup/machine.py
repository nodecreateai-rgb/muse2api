"""Signup state machine — no checkout / card steps.

Paths:
  OTP register (default): restart → send-otp → confirm-otp → register-completion
                         → bootstrap → awaiting_external_payment → resume → tos/wake → pool
  Recovery set-password: recovery-start → confirm-code → recovery-complete → …
  Password login:        restart → send-otp (pla=true) → confirm-password → …
  Invite:                redeem / get / create after hatch_sess exists
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

from app import pool_meta
from app.pool import store as pool_store
from app.signup import invites, store
from app.signup.identity import (
    ensure_profile_identity,
    password_preview,
    random_password,
)
from app.signup.protocol import (
    MuseSession,
    MuseProtocolError,
    fresh_sticky_proxy,
    get_muse_client,
    mask_proxy_url,
    pick_proxy,
)
from app.config import get_settings

logger = logging.getLogger("muse-proxy.signup")

_STOP = False


def request_stop() -> None:
    global _STOP
    _STOP = True


def clear_stop() -> None:
    global _STOP
    _STOP = False


def stopped() -> bool:
    return _STOP


def _sess_from_meta(meta: dict[str, Any]) -> MuseSession:
    return MuseSession(
        waterfall_id=meta.get("waterfall_id") or "",
        csrf_token=meta.get("csrf_token") or "",
        cookies=dict(meta.get("cookies") or {}),
        tos_content_version=meta.get("tos_content_version") or "mock-tos-v1",
        frl_access_token=meta.get("frl_access_token") or "",
        frl_account_id=meta.get("frl_account_id") or "",
        hatch_sess=meta.get("hatch_sess") or "",
        last_otp=meta.get("last_otp") or "",
        last_recovery_code=meta.get("last_recovery_code") or "",
        email=meta.get("email") or "",
        password_encryption=dict(meta.get("password_encryption") or {}),
        password_login_available=bool(meta.get("password_login_available")),
        flow=meta.get("flow") or "",
        has_muse_account=bool(meta.get("has_muse_account")),
        has_password=bool(meta.get("has_password")),
        already_activated=bool(meta.get("already_activated")),
        fingerprint=dict(meta.get("fingerprint") or {}),
        last_request_headers=dict(meta.get("last_request_headers") or {}),
        proxy_url=str(meta.get("proxy_url") or ""),
        device_id=str(meta.get("device_id") or ""),
        checkout=dict(meta.get("checkout") or {}),
    )


def _meta_from_sess(sess: MuseSession) -> dict[str, Any]:
    return {
        "waterfall_id": sess.waterfall_id,
        "csrf_token": sess.csrf_token,
        "cookies": sess.cookies,
        "tos_content_version": sess.tos_content_version,
        "frl_access_token": sess.frl_access_token,
        "frl_account_id": sess.frl_account_id,
        "hatch_sess": sess.hatch_sess,
        "last_otp": sess.last_otp,
        "last_recovery_code": sess.last_recovery_code,
        "email": sess.email,
        "password_encryption": sess.password_encryption,
        "password_login_available": sess.password_login_available,
        "flow": sess.flow,
        "has_muse_account": sess.has_muse_account,
        "has_password": sess.has_password,
        "already_activated": sess.already_activated,
        "fingerprint": sess.fingerprint,
        # Persist non-secret fingerprint headers snapshot for debugging
        "last_request_headers": {
            k: v
            for k, v in (sess.last_request_headers or {}).items()
            if k.lower() not in ("cookie", "authorization")
        },
        "proxy_url": sess.proxy_url or "",
        "device_id": sess.device_id,
        "checkout": sess.checkout,
    }


def _log(item_id: str, msg: str, level: str = "info") -> None:
    store.add_log(item_id, level, msg)
    logger.info("[%s] %s", item_id, msg)


MAX_GEO_PROXY_RETRIES = 6


def _is_geo_blocked_error(exc: MuseProtocolError) -> bool:
    """Return True when Muse rejected this exit IP due to geo-gate / 403 block."""
    msg = (exc.message or "").lower()
    detail_text = json.dumps(exc.detail, ensure_ascii=False, default=str).lower() if exc.detail else ""
    blob = f"{msg} {detail_text}"
    if "csrf" in blob:
        return False
    if msg == "bootstrap_blocked" or exc.status_code == 403:
        return True
    return any(
        kw in blob
        for kw in (
            "geo_gated",
            "isgeogated",
            "geo-gate",
            "region",
            "waitlist",
            "content isn't available",
            "not available in your",
            "unsupported country",
        )
    )


def _bind_session_proxy(
    item: dict[str, Any],
    sess: MuseSession,
    *,
    force_rotate: bool = False,
    exclude: set[str] | None = None,
) -> str:
    """Attach a proxy exit to the session; return masked label for logs.

    When `force_rotate=True` (or the configured proxy is a `{sid}` template),
    pick a fresh exit that is not in `exclude`.
    """
    from app.signup.protocol import expand_proxy_sid

    settings = get_settings()
    pool_raw = getattr(settings, "proxies", "") or ""
    ex = set(exclude or ()) - {""}
    cur = (sess.proxy_url or "").strip()
    if force_rotate and cur:
        ex.add(cur)

    explicit = (item.get("proxy") or "").strip() or None
    preferred = explicit if explicit else (None if force_rotate else (cur or None))
    if "{sid}" in pool_raw and (
        force_rotate or not preferred or "-sid-" in preferred or "arxlabs" in preferred
    ):
        chosen = expand_proxy_sid(pick_proxy(pool_raw, exclude=ex) or "")
    else:
        chosen = expand_proxy_sid(pick_proxy(pool_raw, preferred=preferred, exclude=ex) or "")
    sess.proxy_url = chosen or ""
    return mask_proxy_url(sess.proxy_url) or "(direct)"


def _rotate_session_proxy(sess: MuseSession, tried: set[str]) -> str | None:
    """Switch `sess.proxy_url` to an untried proxy from settings.proxies; return masked label or None."""
    cur = (sess.proxy_url or "").strip()
    if cur:
        tried.add(cur)
    new_proxy = fresh_sticky_proxy(exclude=tried)
    if not new_proxy or new_proxy in tried:
        return None
    sess.proxy_url = new_proxy
    tried.add(new_proxy)
    return mask_proxy_url(new_proxy) or "(proxy)"


def _mask_code(code: str) -> str:
    raw = (code or "").strip().upper()
    if len(raw) <= 2:
        return raw or "—"
    return raw[:2] + "···"


async def _pull_owner_code(client: Any, sess: MuseSession) -> dict[str, Any] | None:
    """GET the account's invite, and create one when the panel has none yet."""
    data = await client.invite_get(sess)
    if invites.parse_invite_payload(data):
        return data if isinstance(data, dict) else None
    created = await client.invite_create(sess)
    return created if isinstance(created, dict) else None


async def _stock_session_invite(
    item_id: str,
    sess: MuseSession,
    client: Any,
    *,
    owner_email: str,
    owner_ref: str,
) -> str | None:
    try:
        data = await _pull_owner_code(client, sess)
    except MuseProtocolError as exc:
        _log(item_id, f"邀请码没放进库: {exc.status_code} {exc.message}"[:180], "warn")
        return None
    except Exception as exc:  # noqa: BLE001
        _log(item_id, f"邀请码没放进库: {exc}"[:180], "warn")
        return None
    code = invites.remember(data, owner_email=owner_email, owner_ref=owner_ref)
    if not code:
        _log(item_id, "这个号还没有自己的邀请码", "warn")
        return None
    parsed = invites.parse_invite_payload(data) or {}
    _log(
        item_id,
        f"邀请码已入库 {_mask_code(code)}，还能带 {parsed.get('uses_remaining', '?')} 个新号",
    )
    return code


def _donor_session(donor: dict[str, str]) -> MuseSession | None:
    if donor.get("kind") == "item":
        item = store.get_item(donor["id"]) or {}
        sess = _sess_from_meta(store.get_session_meta(donor["id"]))
        if not sess.hatch_sess:
            blob = item.get("cookie_blob") or ""
            _restore_sess_cookie(item, sess)
            if not sess.hatch_sess and blob:
                _restore_sess_cookie({"cookie_blob": blob}, sess)
        if not sess.csrf_token:
            sess.csrf_token = invites.csrf_from_cookies(sess.cookies)
        if not sess.hatch_sess or not sess.csrf_token:
            return None
        _bind_session_proxy(item, sess)
        return sess
    cookies, _exp = pool_store.get_cookies(donor["id"])
    if not cookies.get("hatch_sess"):
        return None
    sess = MuseSession(
        email=donor.get("email") or "",
        cookies=dict(cookies),
        hatch_sess=str(cookies.get("hatch_sess") or ""),
        csrf_token=invites.csrf_from_cookies(cookies),
    )
    if not sess.csrf_token:
        return None
    _bind_session_proxy({}, sess)
    return sess


async def _ensure_invite_stock(item_id: str, exclude_email: str) -> None:
    """Pull one invite code from every successful account that is not stocked yet."""
    missing = invites.missing_donors(exclude_email)
    if not missing:
        return
    _log(item_id, f"还有 {len(missing)} 个成功号的邀请码没入库，开始收")
    for donor in missing:
        sess = _donor_session(donor)
        if sess is None:
            _log(item_id, "有个成功号没有可用会话，这张码先跳过", "warn")
            continue
        client = get_muse_client(proxy=sess.proxy_url or None)
        owner_ref = donor.get("signup_item_id") or donor.get("id") or ""
        await _stock_session_invite(
            item_id,
            sess,
            client,
            owner_email=donor.get("email") or sess.email or "",
            owner_ref=owner_ref,
        )


async def _redeem_library_invite(item_id: str, sess: MuseSession, client: Any) -> bool:
    """Admit this new account with a code owned by an earlier success."""
    if (sess.checkout or {}).get("invite_redeemed"):
        return True
    email = sess.email or ""
    await _ensure_invite_stock(item_id, email)
    for _nth in range(3):
        code = invites.claim(email)
        if not code:
            _log(item_id, "没有可用的邀请码，这位先按原路继续", "warn")
            return False
        try:
            data = await client.invite_redeem(sess, code)
        except MuseProtocolError as exc:
            kind = invites.classify_redeem_error(exc.status_code, exc.message, exc.detail)
            if kind == "already":
                invites.release(code)
                sess.checkout = dict(sess.checkout or {})
                sess.checkout["invite_redeemed"] = True
                _log(item_id, "这个号已经用过邀请码")
                return True
            if kind == "dead":
                invites.burn(code)
                _log(item_id, f"邀请码 {_mask_code(code)} 失效了，换下一张", "warn")
                continue
            invites.release(code)
            _log(item_id, f"邀请码没兑成: {exc.status_code} {exc.message}"[:180], "warn")
            return False
        sess.checkout = dict(sess.checkout or {})
        sess.checkout["invite_redeemed"] = True
        admitted = ""
        if isinstance(data, dict):
            admitted = str(data.get("waitlistStatus") or data.get("waitlist_status") or "")
        _log(
            item_id,
            f"用老号的邀请码 {_mask_code(code)} 带入"
            + (f" · {admitted}" if admitted else ""),
        )
        return True
    _log(item_id, "连续几张邀请码都没兑上", "warn")
    return False


def _format_protocol_error(exc: MuseProtocolError) -> str:
    detail = exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail}
    bits = [exc.message, f"status={exc.status_code}"]
    if isinstance(detail, dict):
        if detail.get("proxy") is not None:
            bits.append(f"proxy={detail.get('proxy') or '(direct)'}")
        if "proxy_used" in detail:
            bits.append(f"proxy_used={detail.get('proxy_used')}")
        hdrs = detail.get("headers")
        if isinstance(hdrs, dict) and hdrs:
            # keep short
            shown = ",".join(f"{k}={v}" for k, v in list(hdrs.items())[:6])
            bits.append(f"headers[{shown}]")
        err = detail.get("error")
        if err:
            bits.append(f"err={str(err)[:180]}")
        raw = detail.get("raw")
        if raw and not err:
            bits.append(f"body={str(raw)[:120]}")
    return " · ".join(bits)




def _cookie_header(sess: MuseSession) -> str:
    parts = []
    if sess.hatch_sess:
        parts.append(f"hatch_sess={sess.hatch_sess}")
    for k, v in sess.cookies.items():
        if k == "hatch_sess":
            continue
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def _preview(cookie_header: str) -> str:
    if len(cookie_header) <= 24:
        return cookie_header[:4] + "…" if cookie_header else ""
    return cookie_header[:12] + "…" + cookie_header[-8:]


def _restore_sess_cookie(item: dict[str, Any], sess: MuseSession) -> None:
    if sess.hatch_sess:
        return
    blob = item.get("cookie_blob") or ""
    for part in str(blob).split(";"):
        part = part.strip()
        if part.startswith("hatch_sess="):
            sess.hatch_sess = part.split("=", 1)[1]
            sess.cookies["hatch_sess"] = sess.hatch_sess
            break


async def _push_pool(item_id: str, email: str, cookie_header: str) -> str | None:
    """注册成功后写入本库号池（Cookie 密文权威，不再推 19080）。"""
    item = store.get_item(item_id) or {}
    meta = store.get_session_meta(item_id)
    secret = store.get_secret(item_id)
    fp = meta.get("fingerprint") if isinstance(meta.get("fingerprint"), dict) else {}
    fp_id = (fp or {}).get("profile_id") or "?"
    profile = {}
    try:
        profile = json.loads(item.get("profile_json") or "{}")
    except json.JSONDecodeError:
        profile = {}
    display_name = (
        profile.get("display_name")
        or " ".join(
            x
            for x in [profile.get("first_name") or "", profile.get("last_name") or ""]
            if x
        ).strip()
        or (email.split("@")[0] if email else item_id)
    )
    label = (email or "").strip() or item_id
    checkout = meta.get("checkout") if isinstance(meta.get("checkout"), dict) else {}
    vm_assigned = checkout.get("vm_assigned")
    assigned_flag = vm_assigned if isinstance(vm_assigned, bool) else None
    _log(item_id, f"准备入本池：账号={label} 名字={display_name} 指纹={fp_id}")
    try:
        acc = pool_store.add_account(
            cookie_header=cookie_header,
            label=label,
            email=email or "",
            name=display_name,
            note=f"补号入池 · {display_name}",
            password=secret.get("muse_login_password") or None,
            fingerprint=fp or {"profile_id": fp_id},
            signup_item_id=item_id,
            proxy={"server": item.get("proxy")} if item.get("proxy") else None,
            assigned=assigned_flag,
        )
    except ValueError as exc:
        msg = str(exc)
        _log(item_id, f"入池失败: {msg}", "error")
        store.update_item(item_id, last_error=msg, message="入池失败")
        return None
    except Exception as exc:  # noqa: BLE001
        _log(item_id, f"入池失败: {exc}", "error")
        store.update_item(item_id, last_error=str(exc), message="入池失败")
        return None
    aid = (acc or {}).get("id")
    store.update_item(
        item_id,
        status="pooled",
        upstream_account_id=aid,
        cookie_preview=_preview(cookie_header),
        cookie_blob=None,
        message="注册成功，已导入号池",
        activated_at=store._now(),
    )
    if aid:
        pool_meta.upsert(
            str(aid),
            email=email or "",
            name=display_name,
            password=secret.get("muse_login_password") or None,
            fingerprint=fp or {"profile_id": fp_id},
            signup_item_id=item_id,
        )
    has_pwd = bool(secret.get("muse_login_password"))
    _log(
        item_id,
        f"注册成功 · {'已设密' if has_pwd else '未设密'} · 已获取Cookie · 指纹={fp_id} · 已导入本库号池 id={aid}",
    )
    if aid and assigned_flag is False:
        try:
            from app.jobs_keepalive import kick_assign

            kick_assign()
        except Exception:
            pass
    return aid


def _hard_decline(exc: MuseProtocolError) -> bool:
    """A decline means the card itself is bad → mark dead instead of retrying it."""
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    blob = " ".join(
        [
            str(exc.message or ""),
            str(detail.get("type") or ""),
            str(detail.get("title") or ""),
            str(detail.get("detail") or ""),
        ]
    ).lower()
    if exc.status_code in (401, 403):
        return False
    if exc.status_code == 402:
        return True
    return any(
        w in blob
        for w in ("declined", "invalidcard", "invalid_card", "cardinal", "do not honor", "expired")
    )


def _transient_failure(exc: MuseProtocolError) -> bool:
    """Network / CSRF / admission / server errors: the card itself is not at fault."""
    if exc.status_code in (0, 401, 403) or exc.status_code >= 500:
        return True
    blob = f"{exc.message} {exc.detail}".lower()
    return any(
        w in blob
        for w in ("csrf", "transport", "proxy", "ptt_unavailable", "admissioninvalidated", "timeout")
    )


_CARD_WAIT_LOOP: asyncio.AbstractEventLoop | None = None
_CARD_WAIT_COND: asyncio.Condition | None = None
_CARD_WAIT_TIMEOUT_SEC = 600.0
_CARD_STALE_IN_USE_SEC = 300


def _get_card_cond() -> asyncio.Condition:
    global _CARD_WAIT_LOOP, _CARD_WAIT_COND
    loop = asyncio.get_running_loop()
    if _CARD_WAIT_COND is None or _CARD_WAIT_LOOP is not loop:
        _CARD_WAIT_LOOP = loop
        _CARD_WAIT_COND = asyncio.Condition()
    return _CARD_WAIT_COND


async def _notify_card_released() -> None:
    cond = _get_card_cond()
    async with cond:
        cond.notify_all()


async def _acquire_card_queued(
    item_id: str,
    *,
    card_id: str | None = None,
    wait_timeout_sec: float = _CARD_WAIT_TIMEOUT_SEC,
) -> dict[str, Any] | None:
    """Claim a card, waiting in FIFO order while occupied cards keep making progress."""
    from app.cards import store as cards_store

    idle_timeout = max(5.0, wait_timeout_sec)
    last_progress = time.monotonic()
    last_snap: tuple[tuple[str, int, str, str], ...] = ()
    logged_wait = False
    while not stopped():
        card = (
            cards_store.claim_card_by_id(card_id, item_id)
            if card_id
            else cards_store.claim_card(item_id)
        )
        if card:
            return card
        snap = cards_store.in_use_snapshot(card_id=card_id)
        in_use = len(snap)
        if in_use <= 0:
            return None
        if snap != last_snap:
            last_snap = snap
            last_progress = time.monotonic()
        if time.monotonic() - last_progress >= idle_timeout:
            cards_store.recover_stale_in_use(max_age_sec=_CARD_STALE_IN_USE_SEC)
            return (
                cards_store.claim_card_by_id(card_id, item_id)
                if card_id
                else cards_store.claim_card(item_id)
            )
        if not logged_wait:
            logged_wait = True
            wait_msg = f"卡占用中（{in_use} 张使用中），排队等待空闲卡…"
            store.update_item(item_id, status="awaiting_card", message=wait_msg)
            _log(item_id, wait_msg)
        cards_store.recover_stale_in_use(max_age_sec=_CARD_STALE_IN_USE_SEC)
        cond = _get_card_cond()
        async with cond:
            try:
                await asyncio.wait_for(cond.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                pass
    return None


def _card_label(card: dict[str, Any]) -> str:
    return f"{card.get('brand') or 'Card'} •••• {card.get('last4') or '????'}"


async def _charge_card(
    item_id: str,
    sess: MuseSession,
    card: dict[str, Any],
    into_pool: bool,
    email: str,
) -> dict[str, Any]:
    """Run one checkout with an already-claimed card, then release it immediately."""
    from app.cards import store as cards_store

    client = get_muse_client(proxy=sess.proxy_url or None)
    label = _card_label(card)
    store.update_item(
        item_id,
        status="paying",
        card_id=card["id"],
        message=f"checkout 支付中 · {label}",
    )
    try:
        try:
            result = await client.run_checkout(sess, card)
            for step in result.get("steps") or []:
                _log(item_id, step)
        except asyncio.CancelledError:
            cards_store.release_card(card["id"], status="available")
            raise
        except MuseProtocolError as exc:
            msg = _format_protocol_error(exc)
            # 卡池策略：失败一律计数并放回卡池复用，超过 3 次才自动禁用
            hard = False
            transient = _transient_failure(exc)
            if transient:
                cards_store.release_card(card["id"], status="available")
                state = "放回卡池（不算失败）"
            else:
                new_state = cards_store.record_failure(
                    card["id"], msg, hard=hard, item_id=item_id
                )
                info = cards_store.get_card(card["id"]) or {}
                fails = info.get("fail_count") or 0
                state = (
                    f"已禁用（失败 {fails}/{cards_store.MAX_CARD_FAILURES}）"
                    if new_state == "disabled"
                    else f"放回卡池（失败 {fails}/{cards_store.MAX_CARD_FAILURES}）"
                )
            store.save_session_meta(item_id, _meta_from_sess(sess))
            store.update_item(item_id, last_error=msg[:800])
            _log(item_id, f"卡 {label} {state}: {msg[:300]}", "error")
            return {"ok": False, "error": msg, "label": label}
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"[:400]
            low = msg.lower()
            if any(w in low for w in ("timeout", "proxy", "connect", "curl", "socket", "eof", "reset")):
                cards_store.release_card(card["id"], status="available", error=msg)
                state = "网络波动，放回卡池（不算失败）"
            else:
                new_state = cards_store.record_failure(
                    card["id"], msg, hard=False, item_id=item_id
                )
                info = cards_store.get_card(card["id"]) or {}
                fails = info.get("fail_count") or 0
                state = (
                    f"已禁用（失败 {fails}/{cards_store.MAX_CARD_FAILURES}）"
                    if new_state == "disabled"
                    else f"放回卡池（失败 {fails}/{cards_store.MAX_CARD_FAILURES}）"
                )
            store.update_item(item_id, last_error=msg[:800])
            _log(item_id, f"卡 {label} 异常 · {state}: {msg}", "error")
            return {"ok": False, "error": msg, "label": label}

        cards_store.mark_used(card["id"], item_id)
    finally:
        cards_store.release_if_in_use(card["id"], item_id)
        await _notify_card_released()

    ptt = result.get("ptt") or {}
    _log(
        item_id,
        "checkout SUCCESS · "
        f"{label}（已释放回卡池） · order={result.get('order_id')} · "
        f"credential={result.get('credential_id')} · "
        f"ptt_key={ptt.get('id') or '?'} chain={ptt.get('chain_len', 0)}",
    )
    store.save_session_meta(item_id, _meta_from_sess(sess))
    fin = await _finish_activation(item_id, sess, into_pool, email)
    return {"ok": True, "result": fin, "checkout": result, "label": label}


async def _pay_with_card(
    item_id: str,
    sess: MuseSession,
    into_pool: bool,
    email: str,
    *,
    allow_fallback: bool = False,
) -> dict[str, Any] | None:
    """ age verification: pull one card from the pool and drive the protocol.

    `allow_fallback` returns `None` when the server answers checkout/init
    with 403 (account needs no card / not admitted) so the caller can continue
    with accept-tos instead of marking the item as failed.
    """
    from app.cards import store as cards_store

    settings = get_settings()
    max_tries = max(1, int(getattr(settings, "checkout_max_card_tries", 3) or 1))
    client = get_muse_client(proxy=sess.proxy_url or None)

    # 1) does this account actually need the card step?
    try:
        init = await client.checkout_init(sess)
        _log(
            item_id,
            "checkout/init OK → 需要  年龄验证 · order="
            f"{init.get('orderId')} · {init.get('receiverName')}",
        )
    except MuseProtocolError as exc:
        det = exc.detail if isinstance(exc.detail, dict) else {}
        _log(
            item_id,
            f"checkout/init {exc.status_code} {det.get('error') or det.get('type') or ''} "
            "→ 该账号不需要卡验证，改走 accept-tos",
            "warn",
        )
        if allow_fallback and exc.status_code in (403, 404):
            store.save_session_meta(item_id, _meta_from_sess(sess))
            return None
        msg = f"checkout 不可用：{exc.status_code} {det.get('error') or exc.message}"
        store.update_item(item_id, status="awaiting_card", message=msg, last_error=msg)
        return {"ok": False, "status": "awaiting_card", "needs": ["card"]}

    # 2) card pool (if all cards are currently in_use, wait one by one)
    errors: list[str] = []
    for attempt in range(1, max_tries + 1):
        if attempt > 1:
            await asyncio.sleep(1.0)
        card = await _acquire_card_queued(item_id)
        if not card:
            if stopped():
                store.update_item(item_id, status="stopped", message="已停止")
                return {"ok": True, "status": "stopped"}
            store.update_item(
                item_id,
                status="awaiting_card",
                message="卡池无可用卡：请先在「卡池」导入或启用卡，再点「继续支付」",
            )
            _log(item_id, "卡池无可用卡 → awaiting_card", "warn")
            return {"ok": True, "status": "awaiting_card", "needs": ["card"]}

        _log(item_id, f"卡池取卡 {_card_label(card)}（{attempt}/{max_tries}）")
        outcome = await _charge_card(item_id, sess, card, into_pool, email)
        if outcome.get("ok"):
            return outcome["result"]
        errors.append(f"{outcome.get('label')} → {str(outcome.get('error'))[:180]}")

    tail = errors[-1] if errors else "无可用卡"
    store.update_item(
        item_id,
        status="awaiting_card",
        message=f"本次 {max_tries} 张卡全部失败，可换卡后继续支付 · {tail[:200]}",
    )
    _log(item_id, f"checkout 全部失败：{tail}", "error")
    return {"ok": False, "status": "awaiting_card", "needs": ["card"], "errors": errors}


async def _after_session(
    item_id: str,
    sess: MuseSession,
    *,
    into_pool: bool,
    email: str,
    skip_payment_if_activated: bool = False,
    force_rotate_proxy: bool = False,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """bootstrap →  年龄验证（卡池自动取卡填协议） → accept-tos → 设密 → 入池.

    Only accounts the server flags as `ccvStatus: not_required` skip the card
    step (checkout/init 403); everything else pulls a card from the pool and
    fills the checkout protocol (PTT JWE → credential → finalize). If an IP is
    geo-gated (403), automatically rotate to a fresh proxy from the pool and
    retry before marking the item `geo_gated`.
    """
    item = store.get_item(item_id) or {}
    tried_proxies: set[str] = set(exclude or ()) - {""}
    if sess.proxy_url:
        tried_proxies.add(sess.proxy_url)
    proxy_label = _bind_session_proxy(
        item,
        sess,
        force_rotate=force_rotate_proxy,
        exclude=tried_proxies if force_rotate_proxy else None,
    )
    if force_rotate_proxy and sess.proxy_url:
        _log(item_id, f"切换新代理继续：proxy={proxy_label}")
    if sess.proxy_url:
        tried_proxies.add(sess.proxy_url)

    settings = get_settings()
    last_geo_exc: MuseProtocolError | None = None

    for geo_round in range(1, MAX_GEO_PROXY_RETRIES + 1):
        if stopped():
            store.update_item(item_id, status="stopped", message="已停止")
            return {"ok": True, "status": "stopped"}

        client = get_muse_client(proxy=sess.proxy_url or None)
        if geo_round == 1 or not sess.hatch_sess:
            try:
                await client.bootstrap(sess)
            except MuseProtocolError as exc:
                if _is_geo_blocked_error(exc) and geo_round < MAX_GEO_PROXY_RETRIES:
                    next_label = _rotate_session_proxy(sess, tried_proxies)
                    if next_label:
                        store.save_session_meta(item_id, _meta_from_sess(sess))
                        _log(
                            item_id,
                            f"bootstrap 遇到地区受限 / {exc.status_code}，换代理重试（第 {geo_round + 1}/{MAX_GEO_PROXY_RETRIES} 次 · proxy={next_label}）",
                            "warn",
                        )
                        await asyncio.sleep(1.0)
                        continue
                if _is_geo_blocked_error(exc):
                    last_geo_exc = exc
                    break
                raise

        if skip_payment_if_activated and sess.already_activated:
            _log(item_id, "已激活账号，跳过 checkout")
            return await _finish_activation(item_id, sess, into_pool, email)
        store.save_session_meta(item_id, _meta_from_sess(sess))

        if bool(getattr(settings, "checkout_enabled", False)):
            paid = await _pay_with_card(
                item_id, sess, into_pool, email, allow_fallback=True
            )
            if paid is not None:
                return paid
        else:
            store.update_item(
                item_id,
                status="awaiting_external_payment",
                message="等待外部支付（提交 payment_ref 后 resume）",
                cookie_preview=_preview(_cookie_header(sess)) if sess.hatch_sess else None,
                cookie_blob=_cookie_header(sess) if sess.hatch_sess else None,
            )
            _log(item_id, "到达 awaiting_external_payment（checkout 已关闭）")
            return {"ok": True, "status": "awaiting_external_payment", "needs": ["payment_external"]}

        # fallback: checkout refused this account (not admitted yet, or exit IP is geo-gated).
        # Spend one code from a successful account, then accept the terms.
        tos_ok = False
        last_exc: MuseProtocolError | None = None
        for attempt in (1, 2):
            if attempt == 2 and (sess.checkout or {}).get("invite_redeemed"):
                break
            await _redeem_library_invite(item_id, sess, client)
            try:
                await client.accept_tos(sess)
                _log(item_id, "accept-tos OK（该账号无需卡验证）")
                tos_ok = True
                break
            except MuseProtocolError as exc:
                last_exc = exc
                if (sess.checkout or {}).get("invite_redeemed"):
                    break

        if tos_ok:
            store.save_session_meta(item_id, _meta_from_sess(sess))
            return await _finish_activation(item_id, sess, into_pool, email, tos_done=True)

        last_geo_exc = last_exc or MuseProtocolError(403, "accept_tos_failed", {})
        if _is_geo_blocked_error(last_geo_exc) and geo_round < MAX_GEO_PROXY_RETRIES:
            next_label = _rotate_session_proxy(sess, tried_proxies)
            if next_label:
                store.save_session_meta(item_id, _meta_from_sess(sess))
                _log(
                    item_id,
                    f"检测到地区受限 / {last_geo_exc.status_code}，切换代理重试（第 {geo_round + 1}/{MAX_GEO_PROXY_RETRIES} 次 · proxy={next_label}）",
                    "warn",
                )
                await asyncio.sleep(1.0)
                continue
        break

    exc = last_geo_exc or MuseProtocolError(403, "accept_tos_failed", {})
    det = exc.detail if isinstance(exc.detail, dict) else {}
    msg = (
        f"已连续尝试 {max(1, len(tried_proxies))} 个代理仍地区受限（checkout 不可用且 accept-tos 被拒）："
        f"{exc.status_code} {det.get('type') or det.get('error') or ''}"
    ).strip()
    store.save_session_meta(item_id, _meta_from_sess(sess))
    store.update_item(item_id, status="geo_gated", message=msg, last_error=msg)
    _log(item_id, msg, "error")
    return {"ok": False, "status": "geo_gated", "needs": ["us_residential_proxy"]}


def _sync_mail_token(client: Any, item_id: str) -> None:
    """Persist a rotated Microsoft refresh_token (mail material) back to the item."""
    token = str(getattr(client, "rotated_refresh_token", "") or "")
    if not token:
        return
    store.merge_secret(item_id, {"refresh_token": token})
    try:
        client.rotated_refresh_token = ""
    except Exception:
        pass
    _log(item_id, "邮箱 refresh_token 已轮换并保存")


def _auto_password_enabled() -> bool:
    try:
        return bool(getattr(get_settings(), "signup_auto_password", True))
    except Exception:
        return True


async def _auto_set_password(
    item_id: str, sess: MuseSession, client: Any, *, logout_other_sessions: bool = False
) -> bool:
    """recovery-start → 邮箱恢复码 → recovery-complete(随机强密码) → 加密保存."""
    secret = store.get_secret(item_id)
    if secret.get("muse_login_password"):
        return True
    if not _auto_password_enabled():
        _log(item_id, "跳过自动设密（SIGNUP_AUTO_PASSWORD=false）")
        return False
    email = sess.email or ""
    if not email:
        item = store.get_item(item_id) or {}
        email = item.get("email") or ""
    if not email:
        _log(item_id, "自动设密跳过：无邮箱", "warn")
        return False
    pwd = random_password()
    try:
        await client.restart(sess, resume=False)
        await client.recovery_start(sess, email)
        _log(item_id, "recovery-start OK，自动取恢复码…")
        code = await client.fetch_otp_from_mail(
            email, secret, lengths=(6, 8, 4), since_ts=time.time() - 10
        )
        _sync_mail_token(client, item_id)
        if not code:
            _log(item_id, "自动设密失败：未取到恢复码（检查邮箱物料/授权）", "warn")
            return False
        await client.recovery_confirm_code(sess, code)
        await client.recovery_complete(
            sess, pwd, logout_other_sessions=logout_other_sessions
        )
    except MuseProtocolError as exc:
        _log(item_id, f"自动设密失败: {_format_protocol_error(exc)[:300]}", "warn")
        return False
    except Exception as exc:  # noqa: BLE001
        _log(item_id, f"自动设密异常: {exc}"[:300], "warn")
        return False
    sess.has_password = True
    store.merge_secret(
        item_id,
        {"muse_login_password": pwd, "muse_login_password_generated": True},
    )
    _log(item_id, f"自动设密成功 · 密码已加密保存 · preview={password_preview(pwd)}")
    return True


async def _finish_activation(
    item_id: str,
    sess: MuseSession,
    into_pool: bool,
    email: str,
    *,
    tos_done: bool = False,
) -> dict[str, Any]:
    item = store.get_item(item_id) or {}
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    sess.checkout = dict(sess.checkout or {})
    already_post_tos = bool(
        tos_done
        or sess.checkout.get("tos_accepted")
        or (item.get("status") == "activating" and item.get("message") == "设置登录密码…")
    )
    store.save_session_meta(item_id, _meta_from_sess(sess))
    if not already_post_tos:
        store.update_item(item_id, status="activating", message="accept-tos / wake…")
        await client.accept_tos(sess)
        sess.checkout["tos_accepted"] = True
        store.save_session_meta(item_id, _meta_from_sess(sess))
        _log(item_id, "accept-tos OK")
        # 激活后才能租工作区：lease-vm 在未验卡/未 accept-tos 时会 403
        try:
            lease = await client.lease_vm(sess, "standard")
            leased_ok = bool(
                isinstance(lease, dict)
                and (
                    str(lease.get("status") or "") == "assigned"
                    or lease.get("vmName")
                    or lease.get("gatewayUrl")
                )
            )
            if leased_ok:
                sess.checkout["vm_assigned"] = True
            store.save_session_meta(item_id, _meta_from_sess(sess))
            _log(
                item_id,
                f"工作区已租用 vm={(lease or {}).get('vmName')} "
                f"gateway={str((lease or {}).get('gatewayUrl'))[:50]}",
            )
        except MuseProtocolError as exc:
            _log(item_id, f"工作区租用失败: {exc.status_code} {exc.message}"[:200], "warn")
    else:
        if item.get("status") == "activating" and item.get("message") == "设置登录密码…":
            sess.checkout["vm_assigned"] = True
        store.update_item(item_id, status="activating", message="wake…")
    # activation order from the capture: accept-tos → stateful-sharing → wake
    try:
        await client._request("POST", "/api/hatch/stateful-sharing", sess, {})
        _log(item_id, "stateful-sharing OK")
    except MuseProtocolError as exc:
        _log(item_id, f"stateful-sharing 跳过: {exc.status_code} {exc.message}"[:160], "warn")
    try:
        await client.vm_wake(sess)
        _log(item_id, "vm/wake OK")
    except MuseProtocolError as exc:
        # Some accounts (no CVM provisioned yet) answer 404 "VM wake unavailable";
        # the session is still valid for the API, so don't fail the account.
        _log(item_id, f"vm/wake 跳过: {exc.status_code} {exc.message}"[:200], "warn")
    store.update_item(item_id, status="activating", message="设置登录密码…")
    # recovery-complete issues its own hatch session; the admitted one (post-TOS)
    # is the one that works for the API, so snapshot it and keep it alive.
    admitted_cookies = dict(sess.cookies)
    await _auto_set_password(item_id, sess, client, logout_other_sessions=False)
    if admitted_cookies.get("hatch_sess"):
        sess.cookies = dict(admitted_cookies)
        sess.hatch_sess = admitted_cookies.get("hatch_sess", "")
        _log(item_id, "已恢复激活会话（设密不顶掉入池 Cookie）")
    try:
        chk = await client._request("POST", "/api/auth/check", sess, {})
        _log(item_id, f"auth/check: ok={chk.get('ok')} outcome={chk.get('outcome')}")
    except MuseProtocolError as exc:
        _log(item_id, f"auth/check 未通过: {exc.status_code} {exc.message}"[:160], "warn")
    cookie_header = _cookie_header(sess)
    fp_id = (sess.fingerprint or {}).get("profile_id") or "?"
    has_pwd = bool((store.get_secret(item_id) or {}).get("muse_login_password")) or bool(
        sess.has_password
    )
    _log(
        item_id,
        f"注册成功：已获取 Cookie（{_preview(cookie_header)}）· 指纹={fp_id}"
        + (" · 已设密" if has_pwd else ""),
    )
    await _stock_session_invite(
        item_id, sess, client, owner_email=email or sess.email or "", owner_ref=item_id
    )
    store.update_item(
        item_id,
        status="activated",
        cookie_preview=_preview(cookie_header),
        cookie_blob=cookie_header if into_pool else None,
        message="注册成功" + ("，准备入池" if into_pool else ""),
        activated_at=store._now(),
    )
    if into_pool and cookie_header:
        aid = await _push_pool(item_id, email, cookie_header)
        return {"ok": True, "status": "pooled" if aid else "activated", "upstream_account_id": aid}
    store.clear_cookie_blob(item_id)
    return {"ok": True, "status": "activated"}


async def run_until_blocked(item_id: str, *, otp_code: str | None = None) -> dict[str, Any]:
    """Advance OTP-register path until human input / payment gate / terminal."""
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}

    from app.signup.protocol import current_item_id

    current_item_id.set(item_id)
    secret = store.get_secret(item_id)
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    profile: dict[str, Any] = {}
    try:
        profile = json.loads(item.get("profile_json") or "{}")
    except json.JSONDecodeError:
        profile = {}

    status = item.get("status") or "pending"
    region = item.get("region") or "US"
    email = item.get("email") or ""
    into_pool = bool(item.get("into_pool"))
    force_rotate_start = status in ("failed", "stopped", "geo_gated")
    tried_start_proxies: set[str] = ({sess.proxy_url} if sess.proxy_url else set())
    proxy_label = _bind_session_proxy(
        item,
        sess,
        force_rotate=force_rotate_start,
        exclude=tried_start_proxies if force_rotate_start else None,
    )
    client = get_muse_client(proxy=sess.proxy_url or None)
    if sess.proxy_url:
        tried_start_proxies.add(sess.proxy_url)

    def persist(status_: str, **extra: Any) -> None:
        store.save_session_meta(item_id, _meta_from_sess(sess))
        store.update_item(item_id, status=status_, **extra)

    # 防关联：同号两次提交之间至少间隔 acct_min_interval_sec（含随机抖动）
    try:
        from datetime import datetime, timezone

        from app.config import get_settings as _gs

        _s = _gs()
        gap = int(getattr(_s, "acct_min_interval_sec", 0) or 0)
        jitter_pct = int(getattr(_s, "acct_interval_jitter_pct", 0) or 0)
        last = item.get("updated_at")
        if gap and last and ((item.get("tries") or 0) > 0 or status in ("failed", "stopped")):
            last_ts = datetime.strptime(str(last)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            need = gap * (1 + random.random() * max(0, jitter_pct) / 100)
            wait = need - (time.time() - last_ts)
            if wait > 0:
                wait = min(wait, 600)
                _log(item_id, f"防关联：同号间隔等待 {wait:.0f}s")
                await asyncio.sleep(wait)
    except Exception:
        pass

    _log(item_id, f"开始补号：{email}（当前状态 {status}）")
    try:
        if status == "geo_gated":
            _restore_sess_cookie(item, sess)
            if sess.hatch_sess:
                _log(item_id, f"检测到已有会话（geo_gated），换新代理重试验证/激活（proxy={proxy_label}）")
                return await _after_session(
                    item_id,
                    sess,
                    into_pool=into_pool,
                    email=email,
                    force_rotate_proxy=False,
                    exclude=tried_start_proxies,
                )
            status = "pending"

        if status in ("pending", "failed", "stopped", "restarting"):
            if stopped():
                persist("stopped", message="已停止")
                return {"ok": True, "status": "stopped"}
            persist("restarting", message="restart…", tries=(item.get("tries") or 0) + 1)
            otp_resp: dict[str, Any] | None = None
            for pre_round in range(1, MAX_GEO_PROXY_RETRIES + 1):
                client = get_muse_client(proxy=sess.proxy_url or None)
                try:
                    await client.restart(sess, resume=False)
                    fp_id = (sess.fingerprint or {}).get("profile_id") or "?"
                    _log(
                        item_id,
                        f"restart OK（真站 {getattr(client, 'base', '?')} · proxy={proxy_label} · CSRF + password_encryption + fingerprint={fp_id}）",
                    )
                    otp_resp = await client.send_otp(sess, email, region)
                    break
                except MuseProtocolError as exc:
                    if _is_geo_blocked_error(exc) and pre_round < MAX_GEO_PROXY_RETRIES:
                        next_label = _rotate_session_proxy(sess, tried_start_proxies)
                        if next_label:
                            proxy_label = next_label
                            client = get_muse_client(proxy=sess.proxy_url or None)
                            _log(
                                item_id,
                                f"登录前步骤遇到地区受限 / {exc.status_code}，换代理重试（第 {pre_round + 1}/{MAX_GEO_PROXY_RETRIES} 次 · proxy={next_label}）",
                                "warn",
                            )
                            await asyncio.sleep(1.0)
                            continue
                    raise
            assert otp_resp is not None
            _log(
                item_id,
                f"send-otp OK flow={otp_resp.get('flow')} pla={otp_resp.get('password_login_available')}",
            )
            # Default activate stays on OTP register path even if pla=true
            otp_meta = {
                "masked_contact": (email[0] + "***@" + email.split("@")[-1]) if "@" in email else "***",
                "sent_at": store._now(),
                "password_login_available": bool(otp_resp.get("password_login_available")),
                "flow": otp_resp.get("flow"),
            }
            persist(
                "awaiting_otp",
                otp_meta_json=json.dumps(otp_meta, ensure_ascii=False),
                message="等待 OTP（也可改走设密 / 密码登录）",
            )
            status = "awaiting_otp"

        if status in ("otp_sent", "awaiting_otp"):
            code = otp_code
            if not code:
                code = await client.fetch_otp_from_mail(email, secret)
                _sync_mail_token(client, item_id)
                if code:
                    _log(item_id, f"邮箱自动取码成功（{len(code)} 位）")
                else:
                    _log(item_id, "邮箱暂未取到验证码（可手工提交或稍后重试）", "warn")
            if not code:
                persist("awaiting_otp", message="等待 OTP（可自动取信或手工提交）")
                return {"ok": True, "status": "awaiting_otp", "needs": ["otp_code"]}
            try:
                data = await client.confirm_otp(
                    sess, code, remember_device=bool(profile.get("remember_device", True))
                )
            except MuseProtocolError as exc:
                if exc.status_code == 403 or "csrf" in str(exc.detail).lower() or "csrf" in exc.message.lower():
                    _log(item_id, "confirm-otp csrfFailed → restart 自愈", "warn")
                    await client.restart(sess, resume=False)
                    await client.send_otp(sess, email, region)
                    persist("awaiting_otp", message="CSRF 自愈后请重新提交 OTP")
                    return {"ok": True, "status": "awaiting_otp", "needs": ["otp_code"]}
                raise
            tos_ver = None
            if isinstance(data, dict):
                tos = data.get("tos") or {}
                tos_ver = tos.get("content_version")
            if tos_ver:
                sess.tos_content_version = str(tos_ver)
                profile["tos_content_version"] = str(tos_ver)
            persist(
                "awaiting_profile",
                profile_json=json.dumps(profile, ensure_ascii=False),
                message="等待资料（生日等）",
            )
            status = "awaiting_profile"
            _log(item_id, "confirm-otp OK → awaiting_profile")

        if status in ("completion_required", "awaiting_profile"):
            # 随机名字 + 合规生日（可被手工 profile 覆盖）；对齐小 AD 补号自动填资料
            profile = ensure_profile_identity(profile)
            store.update_item(
                item_id, profile_json=json.dumps(profile, ensure_ascii=False)
            )
            dob = profile.get("date_of_birth")
            _log(
                item_id,
                "profile 自动填充 "
                f"name={profile.get('first_name')} {profile.get('last_name')} "
                f"dob={dob}",
            )
            for reg_round in range(1, MAX_GEO_PROXY_RETRIES + 1):
                try:
                    await client.register_completion(
                        sess,
                        str(dob),
                        profile.get("tos_content_version") or sess.tos_content_version,
                        first_name=str(profile.get("first_name") or ""),
                        last_name=str(profile.get("last_name") or ""),
                        display_name=str(profile.get("display_name") or ""),
                    )
                    break
                except MuseProtocolError as exc:
                    if _is_geo_blocked_error(exc) and reg_round < MAX_GEO_PROXY_RETRIES:
                        next_label = _rotate_session_proxy(sess, tried_start_proxies)
                        if next_label:
                            proxy_label = next_label
                            client = get_muse_client(proxy=sess.proxy_url or None)
                            _log(
                                item_id,
                                f"register-completion 遇到地区受限 / {exc.status_code}，换代理重试（第 {reg_round + 1}/{MAX_GEO_PROXY_RETRIES} 次 · proxy={next_label}）",
                                "warn",
                            )
                            await asyncio.sleep(1.0)
                            continue
                    raise
            _log(item_id, "register-completion OK")
            sess.checkout = dict(sess.checkout or {})
            try:
                leased = await client.lease_vm(sess)
                vm = ""
                gw = ""
                if isinstance(leased, dict):
                    vm = str(leased.get("vmName") or leased.get("vm_name") or "")
                    gw = str(leased.get("gatewayUrl") or leased.get("gateway_url") or "")
                leased_ok = bool(
                    isinstance(leased, dict)
                    and (str(leased.get("status") or "") == "assigned" or vm or gw)
                )
                sess.checkout["vm_assigned"] = leased_ok
                if leased_ok:
                    _log(item_id, f"工作区已租用 vm={vm or '—'} gateway={gw or '—'}")
                else:
                    _log(item_id, "工作区还没租到，入池后会自动再分配", "warn")
            except Exception as exc:  # noqa: BLE001
                sess.checkout["vm_assigned"] = False
                _log(item_id, f"租用工作区未成功，注册继续，入池后会自动再分配：{exc}"[:200], "warn")
            return await _after_session(
                item_id,
                sess,
                into_pool=into_pool,
                email=email,
                exclude=tried_start_proxies,
            )

        if status == "awaiting_external_payment":
            return {
                "ok": True,
                "status": "awaiting_external_payment",
                "needs": ["payment_external"],
            }

        if status in ("awaiting_card", "paying"):
            _restore_sess_cookie(item, sess)
            if sess.hatch_sess:
                return await _pay_with_card(item_id, sess, into_pool, email)
            return {"ok": True, "status": "awaiting_card", "needs": ["card"]}

        if status in ("awaiting_password_login",):
            return {
                "ok": True,
                "status": "awaiting_password_login",
                "needs": ["password"],
            }

        if status in ("awaiting_recovery_code",):
            return {
                "ok": True,
                "status": "awaiting_recovery_code",
                "needs": ["recovery_code"],
            }

        if status in ("awaiting_set_password",):
            return {
                "ok": True,
                "status": "awaiting_set_password",
                "needs": ["new_password"],
            }

        if status in ("post_payment", "activating"):
            _restore_sess_cookie(item, sess)
            return await _finish_activation(item_id, sess, into_pool, email)

        if status in ("activated", "pooled"):
            return {"ok": True, "status": status}

        return {"ok": True, "status": status}
    except MuseProtocolError as exc:
        msg = _format_protocol_error(exc)
        fail_status = "geo_gated" if _is_geo_blocked_error(exc) else "failed"
        fail_msg = f"地区受限: {msg[:400]}" if fail_status == "geo_gated" else f"失败: {msg[:400]}"
        persist(fail_status, last_error=msg[:800], message=fail_msg)
        _log(item_id, f"协议错误: {msg}", "error")
        return {
            "ok": False,
            "status": fail_status,
            "error": exc.message,
            "status_code": exc.status_code,
            "detail": exc.detail if isinstance(exc.detail, dict) else {"detail": exc.detail},
        }
    except Exception as exc:  # noqa: BLE001
        persist("failed", last_error=str(exc)[:500], message="异常失败")
        _log(item_id, f"异常: {exc}", "error")
        return {"ok": False, "status": "failed", "error": str(exc)}


async def apply_profile(item_id: str, body: dict[str, Any]) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    profile: dict[str, Any] = {}
    try:
        profile = json.loads(item.get("profile_json") or "{}")
    except json.JSONDecodeError:
        profile = {}
    # accept birthday alias
    if body.get("date_of_birth") is None and body.get("birthday"):
        body = dict(body)
        body["date_of_birth"] = body.get("birthday")
    for k in (
        "date_of_birth",
        "tos_content_version",
        "remember_device",
        "note",
        "display_name",
        "first_name",
        "last_name",
    ):
        if k in body and body[k] is not None:
            profile[k] = body[k]
    # If still missing identity fields, fill randomly (caller can omit DOB)
    if body.get("auto_fill", True):
        profile = ensure_profile_identity(profile)
    updates: dict[str, Any] = {
        "profile_json": json.dumps(profile, ensure_ascii=False),
        "message": "资料已更新",
    }
    if body.get("region"):
        updates["region"] = body["region"]
        updates["country"] = body.get("country") or body["region"]
    if body.get("country"):
        updates["country"] = body["country"]
    if "proxy" in body:
        updates["proxy"] = body["proxy"]
    store.update_item(item_id, **updates)
    _log(item_id, "profile 已写入")
    st = item.get("status")
    if st in ("awaiting_profile", "completion_required") and profile.get("date_of_birth"):
        return await run_until_blocked(item_id)
    return {"ok": True, "item": store.row_to_public(store.get_item(item_id))}  # type: ignore[arg-type]


async def apply_otp(item_id: str, otp_code: str, *, remember_device: bool = True) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    profile: dict[str, Any] = {}
    try:
        profile = json.loads(item.get("profile_json") or "{}")
    except json.JSONDecodeError:
        pass
    profile["remember_device"] = remember_device
    store.update_item(item_id, profile_json=json.dumps(profile, ensure_ascii=False))
    return await run_until_blocked(item_id, otp_code=otp_code)


async def apply_resume(
    item_id: str, payment_ref: str, provider: str = "external", note: str | None = None
) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    st = item.get("status")
    if st not in ("awaiting_external_payment", "post_payment"):
        return {
            "ok": False,
            "error": "invalid_status",
            "message": f"当前状态 {st} 不可 resume",
            "status": st,
        }
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _restore_sess_cookie(item, sess)
    store.update_item(
        item_id,
        status="post_payment",
        payment_ref=payment_ref,
        payment_provider=provider or "external",
        payment_resumed_at=store._now(),
        message=note or "外部支付已确认，继续激活",
    )
    _log(item_id, f"resume payment_ref={payment_ref[:32]}…")
    return await _finish_activation(
        item_id, sess, bool(item.get("into_pool")), item.get("email") or ""
    )


async def apply_card_retry(item_id: str, card_id: str | None = None) -> dict[str, Any]:
    """Retry the checkout with a card from the pool (awaiting_card → …).

    ``card_id`` pins one specific card; otherwise the oldest available card wins.
    """
    from app.cards import store as cards_store

    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _restore_sess_cookie(item, sess)
    if not sess.hatch_sess:
        return {
            "ok": False,
            "error": "no_session",
            "message": "该任务没有 hatch_sess，无法继续 checkout；请重置后重跑注册",
        }
    _bind_session_proxy(item, sess)
    into_pool = bool(item.get("into_pool"))
    email = item.get("email") or ""
    if card_id:
        card = await _acquire_card_queued(item_id, card_id=card_id)
        if not card:
            return {"ok": False, "error": "card_unavailable", "message": "该卡不可用（状态或有效期）"}
        _log(item_id, f"手工指定卡 {_card_label(card)} 支付")
        outcome = await _charge_card(item_id, sess, card, into_pool, email)
        if not outcome.get("ok"):
            store.update_item(item_id, status="awaiting_card", message=f"指定卡失败：{str(outcome.get('error'))[:200]}")
        return outcome.get("result") if outcome.get("ok") else outcome
    _log(item_id, "手工触发卡池支付重试")
    return await _pay_with_card(item_id, sess, into_pool, email)


async def start_recovery(item_id: str) -> dict[str, Any]:
    """recovery-start → awaiting_recovery_code."""
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    email = item.get("email") or ""
    try:
        if not sess.csrf_token:
            await client.restart(sess, resume=False)
            _log(item_id, "recovery 前 restart OK")
        data = await client.recovery_start(sess, email)
        store.save_session_meta(item_id, _meta_from_sess(sess))
        store.update_item(
            item_id,
            status="awaiting_recovery_code",
            message="恢复码已发送，请提交 recovery code",
            otp_meta_json=json.dumps(
                {
                    "masked_contact": data.get("display_contact_point"),
                    "sent_at": store._now(),
                    "kind": "recovery",
                },
                ensure_ascii=False,
            ),
        )
        _log(item_id, "recovery-start OK")
        return {
            "ok": True,
            "status": "awaiting_recovery_code",
            "needs": ["recovery_code"],
            "item": store.row_to_public(store.get_item(item_id)),  # type: ignore[arg-type]
        }
    except MuseProtocolError as exc:
        msg = _format_protocol_error(exc)
        store.update_item(item_id, status="failed", last_error=msg[:800], message=f"失败: {msg[:400]}")
        _log(item_id, f"recovery-start 失败: {msg}", "error")
        return {"ok": False, "error": exc.message, "status_code": exc.status_code, "detail": exc.detail}


async def apply_recovery_code(item_id: str, code: str) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    try:
        if item.get("status") not in ("awaiting_recovery_code", "awaiting_otp", "pending"):
            # allow from otp wait as "换一种方式"
            if item.get("status") not in ("awaiting_otp", "otp_sent", "awaiting_recovery_code"):
                pass
        if not sess.csrf_token:
            await client.restart(sess, resume=False)
            await client.recovery_start(sess, item.get("email") or "")
        data = await client.recovery_confirm_code(sess, code.strip())
        store.save_session_meta(item_id, _meta_from_sess(sess))
        store.update_item(
            item_id,
            status="awaiting_set_password",
            message="恢复码已确认，自动随机设密…",
        )
        _log(item_id, f"recovery-confirm-code OK next={data.get('next_step')} → auto set-password")
        # 默认随机设密，无需前端提交明文
        return await apply_set_password(item_id, None, logout_other_sessions=True)
    except MuseProtocolError as exc:
        store.update_item(item_id, last_error=exc.message, message=f"恢复码无效: {exc.message}")
        _log(item_id, f"recovery-confirm-code 失败: {exc.message}", "error")
        return {"ok": False, "error": exc.message, "status": item.get("status")}


async def apply_set_password(
    item_id: str, password: str | None = None, *, logout_other_sessions: bool = True
) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    if item.get("status") not in ("awaiting_set_password",):
        return {
            "ok": False,
            "error": "invalid_status",
            "message": f"当前状态 {item.get('status')} 不可设密",
        }
    # 空密码 → 服务端随机强密码（不落明文到任务日志）
    generated = False
    pwd = (password or "").strip()
    if not pwd:
        pwd = random_password()
        generated = True
    if len(pwd) < 6:
        return {"ok": False, "error": "weak_password", "message": "密码至少 6 位"}
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    email = item.get("email") or ""
    into_pool = bool(item.get("into_pool"))
    try:
        await client.recovery_complete(
            sess, pwd, logout_other_sessions=logout_other_sessions
        )
        sess.has_password = True
        # Persist login password into encrypted secret (not public API / logs)
        store.merge_secret(
            item_id,
            {
                "muse_login_password": pwd,
                "muse_login_password_generated": generated,
            },
        )
        _log(
            item_id,
            f"已设置密码（generated={generated} preview={password_preview(pwd)}）",
        )
        store.save_session_meta(item_id, _meta_from_sess(sess))
        return await _after_session(
            item_id,
            sess,
            into_pool=into_pool,
            email=email,
            skip_payment_if_activated=True,
        )
    except MuseProtocolError as exc:
        store.update_item(item_id, status="failed", last_error=exc.message, message=f"设密失败: {exc.message}")
        return {"ok": False, "error": exc.message}


async def start_password_login(item_id: str) -> dict[str, Any]:
    """restart → send-otp; if pla → awaiting_password_login else error hint."""
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    email = item.get("email") or ""
    region = item.get("region") or "US"
    try:
        await client.restart(sess, resume=False)
        data = await client.send_otp(sess, email, region)
        store.save_session_meta(item_id, _meta_from_sess(sess))
        pla = bool(data.get("password_login_available"))
        if not pla:
            store.update_item(
                item_id,
                status="awaiting_otp",
                message="尚不可密码登录（password_login_available=false），请先设密或走 OTP",
                otp_meta_json=json.dumps(
                    {
                        "masked_contact": data.get("masked_contact"),
                        "sent_at": store._now(),
                        "password_login_available": False,
                        "flow": data.get("flow"),
                    },
                    ensure_ascii=False,
                ),
            )
            return {
                "ok": False,
                "error": "password_login_unavailable",
                "message": "该账号尚未设密或不可密码登录，请先走「设密」",
                "status": "awaiting_otp",
                "item": store.row_to_public(store.get_item(item_id)),  # type: ignore[arg-type]
            }
        store.update_item(
            item_id,
            status="awaiting_password_login",
            message="请提交密码（免 OTP）",
            otp_meta_json=json.dumps(
                {
                    "masked_contact": data.get("masked_contact"),
                    "sent_at": store._now(),
                    "password_login_available": True,
                    "flow": data.get("flow"),
                },
                ensure_ascii=False,
            ),
        )
        _log(item_id, "password login available → awaiting_password_login")
        return {
            "ok": True,
            "status": "awaiting_password_login",
            "needs": ["password"],
            "item": store.row_to_public(store.get_item(item_id)),  # type: ignore[arg-type]
        }
    except MuseProtocolError as exc:
        msg = _format_protocol_error(exc)
        store.update_item(item_id, status="failed", last_error=msg[:800], message=f"失败: {msg[:400]}")
        _log(item_id, f"password-login 失败: {msg}", "error")
        return {"ok": False, "error": exc.message, "status_code": exc.status_code, "detail": exc.detail}


async def apply_password_login(
    item_id: str, password: str, *, remember_device: bool = True
) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    st = item.get("status")
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    email = item.get("email") or ""
    into_pool = bool(item.get("into_pool"))
    try:
        if st != "awaiting_password_login" or not sess.csrf_token:
            # bootstrap the password-login waterfall
            boot = await start_password_login(item_id)
            if not boot.get("ok"):
                return boot
            meta = store.get_session_meta(item_id)
            sess = _sess_from_meta(meta)
        data = await client.confirm_password(
            sess, password, remember_device=remember_device
        )
        if data.get("already_activated"):
            sess.already_activated = True
        sess.has_password = True
        _log(item_id, "confirm-password OK")
        store.save_session_meta(item_id, _meta_from_sess(sess))
        return await _after_session(
            item_id,
            sess,
            into_pool=into_pool,
            email=email,
            skip_payment_if_activated=True,
        )
    except MuseProtocolError as exc:
        store.update_item(item_id, last_error=exc.message, message=f"密码登录失败: {exc.message}")
        _log(item_id, f"confirm-password 失败: {exc.message}", "error")
        return {"ok": False, "error": exc.message, "status": "awaiting_password_login"}


async def apply_invite_redeem(item_id: str, code: str) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    if item.get("status") not in ("activated", "pooled", "awaiting_external_payment", "post_payment"):
        return {
            "ok": False,
            "error": "invalid_status",
            "message": "需已有会话后再兑码（建议激活后）",
        }
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _restore_sess_cookie(item, sess)
    if not sess.hatch_sess:
        return {"ok": False, "error": "no_session", "message": "无 hatch_sess，无法兑码"}
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    try:
        data = await client.invite_redeem(sess, code.strip())
        store.save_session_meta(item_id, _meta_from_sess(sess))
        store.update_item(item_id, message=f"邀请码已兑换: {code.strip().upper()}")
        _log(item_id, f"invite redeem OK {code.strip().upper()}")
        return {"ok": True, "result": data, "item": store.row_to_public(store.get_item(item_id))}  # type: ignore[arg-type]
    except MuseProtocolError as exc:
        return {"ok": False, "error": exc.message}


async def invite_panel(item_id: str, *, create: bool = False) -> dict[str, Any]:
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    meta = store.get_session_meta(item_id)
    sess = _sess_from_meta(meta)
    _restore_sess_cookie(item, sess)
    if not sess.hatch_sess:
        return {"ok": False, "error": "no_session", "message": "无会话"}
    _bind_session_proxy(item, sess)
    client = get_muse_client(proxy=sess.proxy_url or None)
    try:
        if create:
            data = await client.invite_create(sess)
        else:
            data = await client.invite_get(sess)
        store.save_session_meta(item_id, _meta_from_sess(sess))
        return {"ok": True, "invite": data}
    except MuseProtocolError as exc:
        return {"ok": False, "error": exc.message}


async def apply_cookie(
    item_id: str, cookie_header: str, *, into_pool: bool | None = None
) -> dict[str, Any]:
    """Escape hatch: paste cookie → activated / pooled. Clear blob after."""
    item = store.get_item(item_id)
    if not item:
        return {"ok": False, "error": "not_found"}
    into = bool(item.get("into_pool") if into_pool is None else into_pool)
    preview = _preview(cookie_header)
    store.update_item(
        item_id,
        status="activated",
        cookie_preview=preview,
        cookie_blob=cookie_header if into else None,
        message="手工 Cookie 已写入",
        activated_at=store._now(),
        last_error=None,
    )
    _log(item_id, "cookie 逃生舱写入")
    if into and cookie_header:
        aid = await _push_pool(item_id, item.get("email") or "", cookie_header)
        store.clear_cookie_blob(item_id)
        return {
            "ok": True,
            "status": "pooled" if aid else "activated",
            "upstream_account_id": aid,
            "item": store.row_to_public(store.get_item(item_id)),  # type: ignore[arg-type]
        }
    store.clear_cookie_blob(item_id)
    return {"ok": True, "status": "activated", "item": store.row_to_public(store.get_item(item_id))}  # type: ignore[arg-type]
