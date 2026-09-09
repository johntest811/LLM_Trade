import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from core.research_validation import ValidationCosts, validate_observations


def inputs():
    frame = pd.DataFrame({"time": pd.date_range("2032-03-01T12:00:00Z", periods=90, freq="min"),
                          "open": 100., "high": 100.3, "low": 99.9, "close": 100.1})
    watch = {"candidate": True, "live_eligible": False, "strategy_version": "test-v1", "direction": "BUY",
             "candle_time": "2032-03-01T12:00:00Z", "signal_time_utc": "2032-03-01T12:05:00Z",
             "reference_price": 100, "stop_loss": 99, "take_profit": 102, "atr": 1}
    event = {"symbol": "TEST", "observed_at_utc": "2032-03-01T12:05:10Z", "config_fingerprint": "A", "account_scope_id": "account-A",
             "snapshot": {"research_watch": watch}}
    return event, frame


def run(events, frame, **kwargs):
    return validate_observations(events, frame, "TEST", kwargs.pop("costs", ValidationCosts(.1, .05, .02)),
                                 as_of=kwargs.pop("as_of", "2032-03-01T15:00:00Z"), **kwargs)


class ResearchValidationTests(unittest.TestCase):
    def test_cli_runs_read_only_and_preserves_input_fingerprints(self):
        event, frame = inputs()
        frame.loc[7, "high"] = 103
        frame.time = frame.time.map(lambda stamp: stamp.replace(year=2020))
        event["observed_at_utc"] = event["observed_at_utc"].replace("2032", "2020")
        for key in ("candle_time", "signal_time_utc"):
            event["snapshot"]["research_watch"][key] = event["snapshot"]["research_watch"][key].replace("2032", "2020")
        with tempfile.TemporaryDirectory() as directory:
            scans, prices = Path(directory) / "scans.json", Path(directory) / "bars.csv"
            scans.write_text(json.dumps({"observations": [event]}), encoding="utf-8")
            frame.to_csv(prices, index=False)
            before = scans.read_bytes(), prices.read_bytes()
            result = subprocess.run([sys.executable, "-m", "core.research_validation",
                                     "--scans", str(scans), "--bars", str(prices), "--symbol", "TEST",
                                     "--spread-price", "0.1", "--slippage-price", "0.05", "--commission-r", "0.02"],
                                    cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertFalse(report["baseline"]["automatic_live_promotion"])
            self.assertEqual(report["baseline"]["groups"][0]["overall"]["trades"], 1)
            self.assertIn("double_cost_stress", report)
            self.assertEqual(before, (scans.read_bytes(), prices.read_bytes()))
            self.assertEqual(len(list(Path(directory).iterdir())), 2)

    def test_next_minute_entry_and_costs_not_hindsight_signal_close(self):
        event, frame = inputs()
        frame.loc[5, "high"] = 105  # Before the first eligible fill: must be ignored.
        frame.loc[7, "high"] = 102.5
        report = run([event], frame)
        result = report["groups"][0]["results"][0]
        self.assertEqual(result["entry_time"], "2032-03-01T12:06:00+00:00")
        self.assertEqual(result["exit_time"], "2032-03-01T12:08:00+00:00")
        self.assertAlmostEqual(result["entry_price"], 100.15)
        self.assertAlmostEqual(result["net_r"], (101.95 - 100.15) / 1.15 - .02)
        self.assertFalse(report["automatic_live_promotion"])
        self.assertEqual(report["groups"][0]["evidence"], "INSUFFICIENT DATA")
        json.dumps(report, allow_nan=False)

    def test_both_touched_is_counted_as_loss(self):
        event, frame = inputs()
        frame.loc[6, ["low", "high"]] = [98, 103]
        result = run([event], frame)["groups"][0]["results"][0]
        self.assertEqual(result["status"], "BOTH_TOUCHED_STOP_FIRST")
        self.assertLess(result["net_r"], -1)

    def test_gap_stop_can_lose_more_than_one_r(self):
        event, frame = inputs()
        frame.loc[7, ["open", "high", "low", "close"]] = [98.5, 98.8, 98.3, 98.6]
        result = run([event], frame)["groups"][0]["results"][0]
        self.assertAlmostEqual(result["exit_price"], 98.45)
        self.assertLess(result["net_r"], -1.4)

    def test_sell_stop_uses_ask_not_bid(self):
        event, frame = inputs()
        event["snapshot"]["research_watch"].update(direction="SELL", stop_loss=101, take_profit=98)
        frame.loc[7, "high"] = 100.95
        result = run([event], frame)["groups"][0]["results"][0]
        self.assertEqual(result["status"], "STOP")
        self.assertAlmostEqual(result["exit_price"], 101.05)

    def test_missing_minute_cannot_hide_a_loss_before_a_later_target(self):
        event, frame = inputs()
        frame.loc[8, "high"] = 103
        report = run([event], frame.drop(index=7))
        self.assertEqual(report["groups"][0]["results"][0]["status"], "DATA_GAP")
        self.assertEqual(report["groups"][0]["overall"]["trades"], 0)

    def test_unfinished_and_future_bars_cannot_resolve_outcomes(self):
        event, frame = inputs()
        frame.loc[7, "high"] = 103
        report = run([event], frame, as_of="2032-03-01T12:07:30Z")
        self.assertEqual(report["groups"][0]["results"][0]["status"], "INCOMPLETE_DATA")
        future = copy.deepcopy(event)
        future["observed_at_utc"] = "2032-03-01T12:08:00Z"
        self.assertEqual(run([future], frame, as_of="2032-03-01T12:07:30Z")["groups"], [])

    def test_duplicate_scans_use_first_observation_and_versions_do_not_mix(self):
        event, frame = inputs()
        duplicate = copy.deepcopy(event)
        duplicate["observed_at_utc"] = "2032-03-01T12:07:50Z"
        other_version = copy.deepcopy(event)
        other_version["config_fingerprint"] = "B"
        report = run([duplicate, other_version, event], frame)
        self.assertEqual(len(report["groups"]), 2)
        for group in report["groups"]:
            self.assertEqual(group["observations"], 1)
            self.assertEqual(group["results"][0]["observed_at_utc"], "2032-03-01T12:05:10+00:00")

    def test_overlapping_signals_do_not_inflate_sample_size(self):
        event, frame = inputs()
        second = copy.deepcopy(event)
        second["observed_at_utc"] = "2032-03-01T12:10:10Z"
        second["snapshot"]["research_watch"].update(candle_time="2032-03-01T12:05:00Z", signal_time_utc="2032-03-01T12:10:00Z")
        report = run([event, second], frame)
        self.assertEqual(report["groups"][0]["status_counts"]["OVERLAP_SKIPPED"], 1)
        self.assertEqual(report["groups"][0]["overall"]["trades"], 1)

    def test_crossing_fold_boundary_is_purged(self):
        event, frame = inputs()
        group = run([event], frame)["groups"][0]
        self.assertEqual(group["boundary_purged"], 1)
        self.assertEqual(sum(fold["trades"] for fold in group["folds"]), 0)

    def test_bad_costs_and_bad_bars_fail_explicitly(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): ValidationCosts(value, 0, 0)
        event, frame = inputs()
        for malformed in (frame.iloc[::-1], pd.concat([frame, frame.iloc[-1:]]), frame.assign(low=101), frame.assign(high=float("nan"))):
            with self.assertRaises(ValueError): run([event], malformed)

    def test_stale_signal_and_execution_gap_are_not_simulated_fills(self):
        event, frame = inputs()
        event["observed_at_utc"] = "2032-03-01T12:07:01Z"
        self.assertEqual(run([event], frame)["groups"][0]["results"][0]["status"], "STALE_SIGNAL")
        event["observed_at_utc"] = "2032-03-01T12:05:01Z"
        frame.loc[6, ["open", "high", "low", "close"]] = [101, 101.2, 100.9, 101.1]
        self.assertEqual(run([event], frame)["groups"][0]["results"][0]["status"], "UNFILLABLE_OR_DRIFT")

    def test_signal_before_close_and_invalid_geometry_are_rejected(self):
        for change in ({"signal_time_utc": "2032-03-01T12:04:00Z"}, {"stop_loss": 101}, {"atr": float("nan")}):
            event, frame = inputs()
            event["snapshot"]["research_watch"].update(change)
            self.assertEqual(run([event], frame)["groups"][0]["results"][0]["status"], "INVALID_OBSERVATION")

    def test_buy_sell_symmetry_without_costs(self):
        event, frame = inputs()
        frame.loc[7, "high"] = 102.5
        buy = run([event], frame, costs=ValidationCosts(0, 0, 0))["groups"][0]["overall"]["net_r"]
        event["snapshot"]["research_watch"].update(direction="SELL", stop_loss=101, take_profit=98)
        frame[["open", "close"]] = 200 - frame[["open", "close"]]
        high, low = frame.high.copy(), frame.low.copy()
        frame["high"], frame["low"] = 200 - low, 200 - high
        sell = run([event], frame, costs=ValidationCosts(0, 0, 0))["groups"][0]["overall"]["net_r"]
        self.assertEqual(buy, sell)

    def test_no_observations_cannot_report_validation_success(self):
        _, frame = inputs()
        report = run([], frame)
        self.assertEqual(report["groups"], [])
        self.assertFalse(report["automatic_live_promotion"])

    def test_accounts_are_isolated_and_wrong_symbol_input_fails(self):
        event, frame = inputs()
        second = copy.deepcopy(event)
        second["account_scope_id"] = "account-B"
        self.assertEqual(len(run([event, second], frame)["groups"]), 2)
        with self.assertRaises(ValueError): run([event], frame.assign(symbol="OTHER"))

    def test_malformed_observations_are_counted_not_promoted(self):
        event, frame = inputs()
        for snapshot in (1, {"research_watch": 2}):
            report = run([{**event, "snapshot": snapshot}], frame)
            self.assertEqual(report["invalid_observations"], 1)
            self.assertEqual(report["groups"], [])
        with self.assertRaises(ValueError): run([1], frame)

    def test_cost_stress_reduces_the_same_fill_outcome(self):
        event, frame = inputs()
        frame.loc[7, "high"] = 103
        baseline = run([event], frame)["groups"][0]["overall"]["net_r"]
        stressed = run([event], frame, costs=ValidationCosts(.2, .1, .04))["groups"][0]["overall"]["net_r"]
        self.assertLess(stressed, baseline)

    def test_calendar_and_price_scale_do_not_change_r_results(self):
        reference = None
        for year in (2020, 2026, 2032, 2040):
            for scale in (.00001, .01, 1, 1000):
                with self.subTest(year=year, scale=scale):
                    event, frame = inputs()
                    frame.loc[7, "high"] = 103
                    frame.time = frame.time.map(lambda stamp: stamp.replace(year=year))
                    frame[["open", "high", "low", "close"]] *= scale
                    watch = event["snapshot"]["research_watch"]
                    for key in ("reference_price", "stop_loss", "take_profit", "atr"): watch[key] *= scale
                    for key in ("candle_time", "signal_time_utc"): watch[key] = watch[key].replace("2032", str(year))
                    event["observed_at_utc"] = event["observed_at_utc"].replace("2032", str(year))
                    result = run([event], frame, as_of=f"{year}-03-01T15:00:00Z",
                                 costs=ValidationCosts(.1 * scale, .05 * scale, .02))["groups"][0]["overall"]["net_r"]
                    reference = result if reference is None else reference
                    self.assertAlmostEqual(result, reference, places=5)


if __name__ == "__main__":
    unittest.main()
