"""Verify single-card occupancy queueing and card reuse across concurrent signups."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path


class CardQueueReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_file = Path(self._tmp.name) / "test-card-queue.db"
        os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
        os.environ["SIGNUP_CLIENT"] = "mock"
        os.environ["CHECKOUT_ENABLED"] = "true"
        os.environ["SIGNUP_AUTO_PASSWORD"] = "false"

        from app.config import reload_settings
        from app.db import init_db
        from app.signup import protocol as signup_protocol

        reload_settings()
        init_db()
        signup_protocol._mock_singleton = None

    def tearDown(self) -> None:
        from app.config import reload_settings

        self._tmp.cleanup()
        os.environ.pop("DATABASE_URL", None)
        os.environ.pop("SIGNUP_CLIENT", None)
        os.environ.pop("CHECKOUT_ENABLED", None)
        os.environ.pop("SIGNUP_AUTO_PASSWORD", None)
        reload_settings()

    def test_single_card_reused_sequentially_under_concurrent_checkout(self) -> None:
        from app.cards import store as cards_store
        from app.signup import machine, store
        from app.signup.import_parse import MailCreds
        from app.signup.protocol import MuseSession, get_muse_client

        res = cards_store.import_cards("4111111111111111|09/30|123|10001|US")
        self.assertEqual(res["added"], 1)

        client = get_muse_client()
        orig_run_checkout = client.run_checkout
        active_checkouts = 0
        max_concurrent_checkouts = 0

        async def slow_run_checkout(sess: MuseSession, card: dict) -> dict:
            nonlocal active_checkouts, max_concurrent_checkouts
            active_checkouts += 1
            max_concurrent_checkouts = max(max_concurrent_checkouts, active_checkouts)
            try:
                await asyncio.sleep(0.08)
                return await orig_run_checkout(sess, card)
            finally:
                active_checkouts -= 1

        client.run_checkout = slow_run_checkout  # type: ignore[method-assign]

        async def run_three() -> list[dict]:
            ids: list[str] = []
            for idx in range(3):
                email = f"qtest-{idx}@example.com"
                iid, _ = store.upsert_from_creds(
                    MailCreds(email=email, password="p", client_id="c", refresh_token="r"),
                    region="US",
                    into_pool=True,
                )
                sess = MuseSession(email=email, hatch_sess=f"sess_{idx}", csrf_token="csrf")
                store.save_session_meta(iid, machine._meta_from_sess(sess))
                store.update_item(
                    iid,
                    status="awaiting_card",
                    cookie_blob=f"hatch_sess=sess_{idx}",
                )
                ids.append(iid)
            return await asyncio.gather(*(machine.run_until_blocked(i) for i in ids))

        results = asyncio.run(run_three())
        self.assertEqual(max_concurrent_checkouts, 1)
        for out in results:
            self.assertTrue(out.get("ok"), out)
            self.assertEqual(out.get("status"), "pooled", out)

        card_items = cards_store.list_cards(page=1, page_size=10)["items"]
        self.assertEqual(len(card_items), 1)
        self.assertEqual(card_items[0]["status"], "available")
        self.assertEqual(card_items[0]["attempts"], 3)
        self.assertEqual(card_items[0]["fail_count"], 0)

    def test_recover_stale_in_use_unblocks_cards(self) -> None:
        from app.cards import store as cards_store

        cards_store.import_cards("4111111111111111|09/30|123|10001|US")
        claimed = cards_store.claim_card("item_stuck")
        self.assertIsNotNone(claimed)
        self.assertEqual(cards_store.count_in_use(), 1)
        self.assertIsNone(cards_store.claim_card("item_other"))

        recovered = cards_store.recover_stale_in_use(max_age_sec=0)
        self.assertEqual(recovered, 1)
        self.assertEqual(cards_store.count_in_use(), 0)
        reclaimed = cards_store.claim_card("item_other")
        self.assertIsNotNone(reclaimed)


if __name__ == "__main__":
    unittest.main()
