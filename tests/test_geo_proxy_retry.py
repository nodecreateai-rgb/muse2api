"""Verify automatic proxy rotation and retry on geo_gated / 403 signup responses."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class GeoProxyRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_file = Path(self._tmp.name) / "test-geo-retry.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
        os.environ["SIGNUP_CLIENT"] = "mock"
        os.environ["CHECKOUT_ENABLED"] = "true"
        os.environ["SIGNUP_AUTO_PASSWORD"] = "false"

        from app.config import reload_settings
        from app.db import init_db
        from app.runtime_settings import set_overrides
        from app.signup import protocol as signup_protocol

        reload_settings()
        init_db()
        signup_protocol._mock_singleton = None
        set_overrides(
            {
                "proxies": (
                    "http://u1:p1@proxy-a.example:10000\n"
                    "http://u2:p2@proxy-b.example:10000\n"
                    "http://u3:p3@proxy-c.example:10000"
                )
            }
        )

    def tearDown(self) -> None:
        from app.config import reload_settings

        self._tmp.cleanup()
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("SIGNUP_CLIENT", None)
        os.environ.pop("CHECKOUT_ENABLED", None)
        os.environ.pop("SIGNUP_AUTO_PASSWORD", None)
        reload_settings()

    def test_fresh_sticky_proxy_rotates_multiline_pool_without_sid_placeholder(self) -> None:
        from app.runtime_settings import set_overrides
        from app.signup.protocol import fresh_sticky_proxy, pick_proxy

        p1 = pick_proxy(
            "http://u1:p1@proxy-a.example:10000\nhttp://u2:p2@proxy-b.example:10000"
        )
        self.assertIsNotNone(p1)
        p2 = fresh_sticky_proxy(exclude={p1 or ""})
        self.assertTrue(p2)
        self.assertNotEqual(p1, p2)

        set_overrides({"proxies": "http://u1:p1@proxy-a.example:10000"})
        only_one = pick_proxy("http://u1:p1@proxy-a.example:10000")
        self.assertEqual(fresh_sticky_proxy(exclude={only_one or ""}), "")

    def test_after_session_rotates_proxy_on_403_geo_gate_and_succeeds(self) -> None:
        from app.cards import store as cards_store
        from app.signup import machine, store
        from app.signup.import_parse import MailCreds
        from app.signup.protocol import MuseProtocolError, MuseSession, get_muse_client

        cards_store.import_cards("4111111111111111|09/30|123|10001|US")
        iid, _ = store.upsert_from_creds(
            MailCreds(email="geo1@example.com", password="p", client_id="c", refresh_token="r"),
            region="US",
            into_pool=True,
        )
        bad_proxy = "http://u1:p1@proxy-a.example:10000"
        sess = MuseSession(
            email="geo1@example.com",
            hatch_sess="sess_geo1",
            csrf_token="csrf",
            proxy_url=bad_proxy,
        )
        store.save_session_meta(iid, machine._meta_from_sess(sess))

        seen_proxies: list[str] = []
        orig_client = get_muse_client()
        orig_checkout_init = orig_client.checkout_init
        orig_accept_tos = orig_client.accept_tos

        def fake_get_client(*, proxy=None):
            orig_client._current_proxy = proxy or ""
            return orig_client

        async def gated_checkout_init(s: MuseSession, **kw):
            cur = s.proxy_url or getattr(orig_client, "_current_proxy", "")
            seen_proxies.append(cur)
            if cur == bad_proxy:
                raise MuseProtocolError(403, "muse_http_403", {"status_code": 403})
            return await orig_checkout_init(s, **kw)

        async def gated_accept_tos(s: MuseSession):
            cur = s.proxy_url or getattr(orig_client, "_current_proxy", "")
            if cur == bad_proxy:
                raise MuseProtocolError(403, "muse_http_403", {"status_code": 403})
            return await orig_accept_tos(s)

        with (
            patch("app.signup.machine.get_muse_client", side_effect=fake_get_client),
            patch.object(orig_client, "checkout_init", side_effect=gated_checkout_init),
            patch.object(orig_client, "accept_tos", side_effect=gated_accept_tos),
            patch("app.signup.machine.asyncio.sleep", return_value=None),
        ):
            out = asyncio.run(
                machine._after_session(
                    iid,
                    sess,
                    into_pool=True,
                    email="geo1@example.com",
                )
            )

        self.assertTrue(out.get("ok"), out)
        self.assertEqual(out.get("status"), "pooled", out)
        self.assertIn(bad_proxy, seen_proxies)
        self.assertTrue(any(p and p != bad_proxy for p in seen_proxies), seen_proxies)

    def test_retrying_existing_geo_gated_item_rotates_away_from_bad_proxy(self) -> None:
        from app.cards import store as cards_store
        from app.signup import machine, store
        from app.signup.import_parse import MailCreds
        from app.signup.protocol import MuseProtocolError, MuseSession, get_muse_client

        cards_store.import_cards("4111111111111111|09/30|123|10001|US")
        iid, _ = store.upsert_from_creds(
            MailCreds(email="geo2@example.com", password="p", client_id="c", refresh_token="r"),
            region="US",
            into_pool=True,
        )
        bad_proxy = "http://u1:p1@proxy-a.example:10000"
        sess = MuseSession(
            email="geo2@example.com",
            hatch_sess="sess_geo2",
            csrf_token="csrf",
            proxy_url=bad_proxy,
        )
        store.save_session_meta(iid, machine._meta_from_sess(sess))
        store.update_item(iid, status="geo_gated", last_error="403 Forbidden")

        seen_proxies: list[str] = []
        orig_client = get_muse_client()
        orig_checkout_init = orig_client.checkout_init

        async def track_checkout_init(s: MuseSession, **kw):
            seen_proxies.append(s.proxy_url)
            if s.proxy_url == bad_proxy:
                raise MuseProtocolError(403, "muse_http_403", {"status_code": 403})
            return await orig_checkout_init(s, **kw)

        with (
            patch.object(orig_client, "checkout_init", side_effect=track_checkout_init),
            patch("app.signup.machine.asyncio.sleep", return_value=None),
        ):
            out = asyncio.run(machine.run_until_blocked(iid))

        self.assertTrue(out.get("ok"), out)
        self.assertEqual(out.get("status"), "pooled", out)
        self.assertTrue(seen_proxies)
        self.assertNotIn(bad_proxy, seen_proxies)

    def test_pre_otp_restart_rotates_proxy_on_403(self) -> None:
        from app.signup import machine, store
        from app.signup.import_parse import MailCreds
        from app.signup.protocol import MuseProtocolError, MuseSession, get_muse_client

        iid, _ = store.upsert_from_creds(
            MailCreds(email="geo3@example.com", password="p", client_id="c", refresh_token="r"),
            region="US",
            into_pool=True,
        )
        bad_proxy = "http://u1:p1@proxy-a.example:10000"
        good_proxy = "http://u2:p2@proxy-b.example:10000"
        orig_client = get_muse_client()
        orig_restart = orig_client.restart
        restart_proxies: list[str] = []

        async def gated_restart(s: MuseSession, *, resume: bool = False):
            restart_proxies.append(s.proxy_url)
            if s.proxy_url == bad_proxy:
                raise MuseProtocolError(403, "muse_http_403", {"status_code": 403})
            return await orig_restart(s, resume=resume)

        with (
            patch("app.signup.machine.pick_proxy", return_value=bad_proxy),
            patch("app.signup.machine.fresh_sticky_proxy", return_value=good_proxy),
            patch.object(orig_client, "restart", side_effect=gated_restart),
            patch.object(orig_client, "fetch_otp_from_mail", return_value=None),
            patch("app.signup.machine.asyncio.sleep", return_value=None),
        ):
            out = asyncio.run(machine.run_until_blocked(iid))

        self.assertEqual(restart_proxies, [bad_proxy, good_proxy])
        self.assertEqual(out.get("status"), "awaiting_otp", out)


if __name__ == "__main__":
    unittest.main()
