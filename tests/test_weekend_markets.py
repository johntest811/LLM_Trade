import unittest
from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from core.engine import TradingEngine


class WeekendMarketTests(unittest.TestCase):
    def test_weekend_scope_contains_only_configured_crypto_markets(self):
        isolated = replace(
            settings,
            weekend_trading_enabled=True,
            crypto_only_on_weekend=True,
            weekend_symbols=["ETHUSD", "LTCUSD", "XRPUSD"],
            trading_symbols=["USDJPY", "USDCAD"],
            market_candidate_symbols=["USDJPY", "USDCAD", "EURUSD"],
        )
        with (
            patch("core.engine.settings", isolated),
            patch("core.engine.is_weekend", return_value=True),
        ):
            symbols = TradingEngine._market_candidates_for_current_market()

        self.assertEqual(symbols, ["ETHUSD", "LTCUSD", "XRPUSD"])

    def test_weekend_scope_is_empty_when_weekend_trading_is_disabled(self):
        isolated = replace(
            settings,
            weekend_trading_enabled=False,
            crypto_only_on_weekend=False,
            weekend_symbols=["ETHUSD"],
        )
        with (
            patch("core.engine.settings", isolated),
            patch("core.engine.is_weekend", return_value=True),
        ):
            symbols = TradingEngine._market_candidates_for_current_market()

        self.assertEqual(symbols, [])

    def test_weekday_scope_does_not_include_weekend_only_crypto(self):
        isolated = replace(
            settings,
            dynamic_market_selection_enabled=True,
            trading_symbols=["USDJPY"],
            market_candidate_symbols=["USDJPY", "USDCAD"],
            weekend_symbols=["ETHUSD"],
        )
        with (
            patch("core.engine.settings", isolated),
            patch("core.engine.is_weekend", return_value=False),
        ):
            symbols = TradingEngine._market_candidates_for_current_market()

        self.assertEqual(symbols, ["USDJPY", "USDCAD"])


if __name__ == "__main__":
    unittest.main()
