from __future__ import annotations

import unittest

import numpy as np

from generation.weighted_entropy import (
    AskState,
    AttributeTable,
    HARD_REFUSAL_LIMIT,
    ask_state_from_profile,
    rank_attributes,
)


class HardRefusalLimitTests(unittest.TestCase):
    def test_attribute_stays_open_below_the_limit(self) -> None:
        state = ask_state_from_profile(
            {}, previous_ask_attribute="material", refusals={"material": HARD_REFUSAL_LIMIT - 2}
        )
        self.assertNotIn("material", state.banned)

    def test_attribute_becomes_banned_at_the_limit(self) -> None:
        state = ask_state_from_profile(
            {}, previous_ask_attribute="material", refusals={"material": HARD_REFUSAL_LIMIT - 1}
        )
        self.assertEqual(state.refusals["material"], HARD_REFUSAL_LIMIT)
        self.assertIn("material", state.banned)

    def test_confirmed_attribute_is_never_banned_by_refusal_count(self) -> None:
        # A stale refusal count sitting next to a now-filled slot must not
        # ban an attribute the customer already answered.
        state = ask_state_from_profile(
            {"material": ["cotton"]},
            previous_ask_attribute="material",
            refusals={"material": HARD_REFUSAL_LIMIT + 5},
        )
        self.assertIn("material", state.confirmed)
        # Banned is fine to also contain it (is_blocked() checks either set),
        # but confirmed must be true regardless of the stale count.

    def test_banned_attribute_is_excluded_from_ranking(self) -> None:
        # rank_attributes() must actually drop a banned attribute from its
        # output entirely (default include_blocked=False), not just score it
        # low -- this is what lets search.py's select() move on to something
        # else, or to None, once nothing else is left.
        codes = {"material": np.array([0, 1, 0, 1], dtype=np.int32)}
        vocabs = {"material": ["cotton", "wool"]}
        table = AttributeTable(codes=codes, vocabs=vocabs, n_products=4)
        pool = [{"catalog_index": i} for i in range(4)]

        open_state = AskState()
        ranked_open = rank_attributes(pool, table, open_state)
        self.assertTrue(any(item.attribute == "material" for item in ranked_open))

        fatigued_state = AskState(banned={"material"})
        ranked_banned = rank_attributes(pool, table, fatigued_state)
        self.assertFalse(any(item.attribute == "material" for item in ranked_banned))


if __name__ == "__main__":
    unittest.main()
