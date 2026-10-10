import math
from types import SimpleNamespace
import unittest

import torch

from seesaw_policy.task import make_task_recipe as make_recipe
from seesaw_policy.env.board.rewards import (
    SETTLING_REWARD_CONFIG,
    RELATIVE_TRACKING_REWARD_CONFIG,
    balance_terms,
    settling_terms,
)


class SettlingTests(unittest.TestCase):
    def config(self):
        return dict(make_recipe().env_config["seesaw"], **SETTLING_REWARD_CONFIG)

    def boarding(self, n):
        return SimpleNamespace(
            mounted=torch.ones(n, dtype=torch.bool),
            supported=torch.ones(n, dtype=torch.bool),
            failure=torch.zeros(n, dtype=torch.bool),
        )

    def test_braking_strength_increases_near_horizontal(self):
        b, c = self.boarding(3), self.config()
        angle = torch.tensor([0.0, math.radians(5), math.radians(30)])
        terms = settling_terms(b, angle, torch.ones(3), torch.zeros(3), c, 0.02)
        self.assertAlmostEqual(float(terms["seesaw_rate"][0]), 5.0)
        self.assertGreater(float(terms["seesaw_rate"][1]), 1.0)
        self.assertAlmostEqual(float(terms["seesaw_rate"][2]), 1.0)
        self.assertFalse(bool(terms["seesaw_hold"].any()))

    def test_hold_builds_only_while_supported_and_steady(self):
        b, c = self.boarding(4), self.config()
        angle = torch.tensor([0.0, math.radians(1.5), 0.0, 0.0])
        rate = torch.tensor([0.0, 0.0, 0.12, 0.0])
        b.supported[3] = False
        timer = torch.zeros(4)
        for _ in range(110):
            terms = settling_terms(b, angle, rate, timer, c, 0.02)
        torch.testing.assert_close(timer, torch.tensor([2.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(
            terms["seesaw_hold"], torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        b.failure[0] = True
        terms = settling_terms(b, angle, rate, timer, c, 0.02)
        self.assertFalse(bool(timer.any()))
        self.assertEqual(float(terms["seesaw_rate"][0]), 0.0)

    def test_fast_crossing_and_three_degree_tilt_are_not_stable(self):
        b, c = self.boarding(3), self.config()
        terms = balance_terms(
            b,
            torch.zeros(3, 3),
            torch.tensor([0.0, 0.0, math.radians(3)]),
            torch.tensor([0.0, 0.35, 0.0]),
            c,
        )
        self.assertEqual(float(terms["seesaw_stable"][0]), 1.0)
        self.assertLess(float(terms["seesaw_stable"][1]), 0.001)
        self.assertLess(float(terms["seesaw_stable"][2]), 0.11)

    def test_tracking_uses_relative_angle_and_angular_speed(self):
        b = self.boarding(3)
        c = dict(self.config(), **RELATIVE_TRACKING_REWARD_CONFIG)
        maximum = torch.tensor([10.0, 20.0, 30.0]) * math.pi / 180
        angle, rate = maximum * 0.05, maximum * 0.1 / c["tracking_response_seconds"]
        terms = balance_terms(b, torch.zeros(3, 3), angle, rate, c, max_angle=maximum)
        for value in terms.values():
            torch.testing.assert_close(value, value[0].expand_as(value))
        expected = math.exp(
            -((0.05 / c["stable_angle_relative_sigma"]) ** 2)
            - (0.1 / c["stable_rate_relative_sigma"]) ** 2
        )
        torch.testing.assert_close(terms["seesaw_stable"], torch.full((3,), expected))
        at_limit = balance_terms(
            b, torch.zeros(3, 3), maximum, torch.zeros(3), c, max_angle=maximum
        )
        self.assertTrue(bool((at_limit["seesaw_level"] < 1e-6).all()))

    def test_relative_tracking_preserves_absolute_hold_threshold(self):
        b = self.boarding(4)
        c = dict(self.config(), **RELATIVE_TRACKING_REWARD_CONFIG)
        maximum = torch.tensor([10.0, 30.0, 10.0, 30.0]) * math.pi / 180
        angle = torch.tensor([1.0, 1.0, 2.0, 2.0]) * math.pi / 180
        timer = torch.zeros(4)
        terms = settling_terms(
            b, angle, torch.zeros(4), timer, c, 0.02, max_angle=maximum
        )
        torch.testing.assert_close(timer, torch.tensor([0.02, 0.02, 0.0, 0.0]))
        # The same absolute error is more significant on the lower-tilt board.
        self.assertLess(float(terms["seesaw_hold"][0]), float(terms["seesaw_hold"][1]))
        self.assertEqual(c["success_angle"], math.radians(1.5))


if __name__ == "__main__":
    unittest.main()
