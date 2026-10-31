"""
Unit and acceptance tests for forecaster.py.
Validates Acceptance Tests 2, 3, 4, 6, and 9 from APP_BUILD_SPEC.md.
"""

import unittest
import numpy as np
import forecaster


class TestForecasterAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = forecaster.load_model()
        cls.default_inputs = {
            "micro_market": "Hinjewadi",
            "configuration": "2BHK",
            "carpet_area_sqft": 750,
            "price_per_sqft": 10000,
            "developer_tier": 1,
            "total_units": 150,
            "months_to_possession": 42,
            "dist_metro_km": 2.2,
            "dist_it_hub_km": 2.2,
            "market_sales_trend_pct": -9.0,
        }

    def test_acceptance_test_2_prediction_and_range(self):
        """Acceptance Test 2: Default inputs produce a prediction between 0 and 100 with a range."""
        pred, low, high = forecaster.predict(self.default_inputs, self.bundle)
        self.assertIsInstance(pred, float)
        self.assertIsInstance(low, float)
        self.assertIsInstance(high, float)
        self.assertGreaterEqual(pred, 0.0)
        self.assertLessEqual(pred, 100.0)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 100.0)
        self.assertLessEqual(low, pred)
        self.assertGreaterEqual(high, pred)

    def test_acceptance_test_3_monotonic_price(self):
        """Acceptance Test 3: Raising price (all else equal) never increases the prediction (10 price steps)."""
        base_inputs = dict(self.default_inputs)
        # Test 10 price points spanning discount (-20%) to premium (+25%)
        pct_steps = np.linspace(-20.0, 25.0, 10)
        predictions = []

        for pct in pct_steps:
            test_inp = dict(base_inputs)
            test_inp["price_vs_market_pct"] = pct
            test_inp.pop("price_per_sqft", None)
            pred, _, _ = forecaster.predict(test_inp, self.bundle)
            predictions.append(pred)

        # As price increases, prediction must either decrease or stay the same (never increase)
        for i in range(len(predictions) - 1):
            self.assertGreaterEqual(
                predictions[i] + 1e-5,
                predictions[i + 1],
                f"Monotonic price check failed: price step {pct_steps[i]}% ({predictions[i]}%) "
                f"was followed by {pct_steps[i+1]}% ({predictions[i+1]}%)",
            )

    def test_acceptance_test_4_monotonic_developer_tier(self):
        """Acceptance Test 4: Changing tier from 1 -> 3 never increases the prediction."""
        tier_predictions = []
        for tier in [1, 2, 3]:
            test_inp = dict(self.default_inputs)
            test_inp["developer_tier"] = tier
            pred, _, _ = forecaster.predict(test_inp, self.bundle)
            tier_predictions.append(pred)

        # Tier 1 should have highest or equal absorption, Tier 3 lowest or equal
        self.assertGreaterEqual(
            tier_predictions[0],
            tier_predictions[1],
            f"Tier 1 ({tier_predictions[0]}%) should be >= Tier 2 ({tier_predictions[1]}%)",
        )
        self.assertGreaterEqual(
            tier_predictions[1],
            tier_predictions[2],
            f"Tier 2 ({tier_predictions[1]}%) should be >= Tier 3 ({tier_predictions[2]}%)",
        )

    def test_acceptance_test_6_target_finder(self):
        """Acceptance Test 6: Target finder returns a price or 'not achievable'."""
        # 1. Feasible target (e.g. 35%)
        res_35 = forecaster.max_price_for_target(self.default_inputs, 35.0, self.bundle)
        self.assertTrue(res_35["achievable"])
        self.assertIsNotNone(res_35["max_price_psf"])
        self.assertGreater(res_35["max_price_psf"], 0)
        self.assertIn("Highest price that still hits", res_35["message"])

        # 2. Impossible target (e.g. 99% for a modest project)
        res_impossible = forecaster.max_price_for_target(self.default_inputs, 99.0, self.bundle)
        self.assertFalse(res_impossible["achievable"])
        self.assertIsNone(res_impossible["max_price_psf"])
        self.assertIn("Not achievable", res_impossible["message"])

    def test_acceptance_test_9_sanity_reference_case(self):
        """Acceptance Test 9: Hinjewadi, 2BHK, 750 sq ft, ₹10,000/sq ft, Tier 1, 150 units, 42 months -> ~41.5%."""
        pred, low, high = forecaster.predict(self.default_inputs, self.bundle)
        self.assertAlmostEqual(pred, 41.5, delta=1.5)
        self.assertAlmostEqual(low, 30.1, delta=1.5)
        self.assertAlmostEqual(high, 53.0, delta=1.5)

    def test_unit_mix_and_explain(self):
        """Bonus Acceptance Test: Unit mix shows 4 rows and explain provides 3 valid bullets."""
        mix_df, takeaway = forecaster.unit_mix(self.default_inputs, self.bundle)
        self.assertEqual(len(mix_df), 4)
        self.assertListEqual(list(mix_df["configuration"]), ["1BHK", "2BHK", "3BHK", "4BHK"])
        self.assertIn("fastest", takeaway)

        exp = forecaster.explain(self.default_inputs, self.bundle)
        self.assertEqual(len(exp["drivers"]), 10)
        self.assertEqual(len(exp["bullets"]), 3)

    def test_project_mix_weighted_forecast(self):
        """Test Project Mix: 1BHK:0, 2BHK:150, 3BHK:100, 4BHK:0 yields 2 active rows and weighted average."""
        mix_input = {"1BHK": 0, "2BHK": 150, "3BHK": 100, "4BHK": 0}
        df, summary = forecaster.project_mix_forecast(self.default_inputs, mix_input, self.bundle)
        self.assertEqual(len(df), 2)  # only 2BHK and 3BHK active
        self.assertEqual(summary["total_units"], 250)
        self.assertGreater(summary["total_units_sold"], 0)
        self.assertLessEqual(summary["total_units_sold"], 250)
        self.assertGreaterEqual(summary["weighted_pct"], 0.0)
        self.assertLessEqual(summary["weighted_pct"], 100.0)
        # Verify units-weighted average formula
        expected_pct = round(
            (df.iloc[0]["units"] * df.iloc[0]["predicted_pct"] + df.iloc[1]["units"] * df.iloc[1]["predicted_pct"]) / 250,
            1
        )
        self.assertAlmostEqual(summary["weighted_pct"], expected_pct, places=1)



    def test_mixed_tower_acceptance_spec(self):
        """
        Acceptance Test for Mixed Tower Mode:
        Shared: Hinjewadi, Tier 1, 42 mos, default Advanced.
        2BHK: 150 units @ 750 sq ft, ₹10,000/sq ft -> 41.5%, 62 units sold, ₹46.50 cr.
        3BHK: 100 units @ 1100 sq ft, ₹10,000/sq ft -> 36.2%, 36 units sold, ₹39.60 cr.
        Whole tower: 39.4%, 98 of 250 units, ₹86.10 cr.
        """
        shared = dict(self.default_inputs)
        inp_2bhk = {**shared, "configuration": "2BHK", "carpet_area_sqft": 750, "price_per_sqft": 10000, "total_units": 150}
        inp_3bhk = {**shared, "configuration": "3BHK", "carpet_area_sqft": 1100, "price_per_sqft": 10000, "total_units": 100}

        p2, l2, h2 = forecaster.predict(inp_2bhk, self.bundle)
        p3, l3, h3 = forecaster.predict(inp_3bhk, self.bundle)

        self.assertAlmostEqual(p2, 41.5, delta=0.5)
        self.assertAlmostEqual(p3, 36.2, delta=0.5)

        u2 = int(round(150 * p2 / 100.0))
        u3 = int(round(100 * p3 / 100.0))
        self.assertEqual(u2, 62)
        self.assertEqual(u3, 36)

        weighted_p = round((150 * p2 + 100 * p3) / 250.0, 1)
        self.assertAlmostEqual(weighted_p, 39.4, delta=0.5)
        self.assertEqual(u2 + u3, 98)

        # Sales value in ₹ cr
        cr2 = round((u2 * 750 * 10000) / 1e7, 2)
        cr3 = round((u3 * 1100 * 10000) / 1e7, 2)
        total_cr = round(cr2 + cr3, 2)
        self.assertEqual(cr2, 46.50)
        self.assertEqual(cr3, 39.60)
        self.assertEqual(total_cr, 86.10)


if __name__ == "__main__":
    unittest.main()
