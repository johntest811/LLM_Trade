from dataclasses import replace
from unittest.mock import patch

from app_config.settings import settings
from risk.manager import RiskManager


def _analysis(
    timeframe,
    *,
    direction="BEARISH",
    adx=30.0,
    adx_delta=1.0,
    timestamp="2026-08-11T06:40:00+00:00",
    breakout="BEARISH BREAKOUT",
    patterns=None,
    events=None,
):
    return {
        "timeframe": timeframe,
        "timestamp": timestamp,
        "indicators": {
            "current_price": 100.0,
            "atr_14": 1.0,
            "atr_14_pips": 10.0,
            "adx_14": adx,
            "adx_delta": adx_delta,
            "rsi_14": 45.0,
            "candle_range_atr": 0.8,
            "candle_return_atr": -0.4,
            "candle_body_atr_signed": -0.4,
            "stochastic": {"k": 45.0, "d": 45.0},
            "bollinger_bands": {"upper": 102.0, "lower": 98.0},
        },
        "market_structure": {
            "trend": direction,
            "trend_state_direction": direction,
            "breakout_status": breakout,
            "candlestick_patterns": (
                ["Bearish Engulfing (Bearish Reversal)"]
                if patterns is None
                else patterns
            ),
            "structure_events": (
                [
                    {
                        "type": "BOS",
                        "direction": direction,
                        "time": timestamp,
                    }
                ]
                if events is None
                else events
            ),
            "support": 98.0,
            "resistance": 102.0,
            "demand_zones": [],
            "supply_zones": [],
        },
    }


def _failed_buy_history(close_time="2026-08-11T05:21:00+00:00"):
    return [
        {
            "symbol": "NZDUSD",
            "direction": "BUY",
            "close_time": close_time,
            "net_profit": -0.42,
            "magic": settings.strategy_magic,
        }
    ]


def test_bounded_failed_thesis_reversal_qualifies_broker_loss_and_fresh_evidence():
    configured = replace(
        settings,
        confidence_threshold=0.70,
        failed_thesis_reversal_enabled=True,
        failed_thesis_reversal_min_confidence=0.60,
        failed_thesis_reversal_max_age_bars=18,
        failed_thesis_reversal_min_m5_adx=25.0,
        failed_thesis_reversal_min_m15_adx=19.1,
    )
    with patch("risk.manager.settings", configured):
        ok, detail = RiskManager.qualify_failed_thesis_reversal(
            "SELL",
            0.60,
            _failed_buy_history(),
            _analysis("M5"),
            _analysis("M15", adx=22.0),
        )

    assert ok, detail
    assert "latest BUY thesis lost" in detail


def test_failed_thesis_reversal_rejects_unconfirmed_m15_direction():
    with patch(
        "risk.manager.settings",
        replace(
            settings,
            confidence_threshold=0.70,
            failed_thesis_reversal_enabled=True,
            failed_thesis_reversal_min_confidence=0.60,
        ),
    ):
        ok, detail = RiskManager.qualify_failed_thesis_reversal(
            "SELL",
            0.60,
            _failed_buy_history(),
            _analysis("M5"),
            _analysis("M15", direction="BULLISH", adx=22.0),
        )

    assert not ok
    assert "M15 direction" in detail


def test_failed_thesis_reversal_rejects_confidence_below_sixty_percent():
    configured = replace(
        settings,
        confidence_threshold=0.70,
        failed_thesis_reversal_enabled=True,
        failed_thesis_reversal_min_confidence=0.60,
    )
    with patch("risk.manager.settings", configured):
        ok, detail = RiskManager.qualify_failed_thesis_reversal(
            "SELL",
            0.599,
            _failed_buy_history(),
            _analysis("M5"),
            _analysis("M15", adx=22.0),
        )

    assert not ok
    assert "bounded reversal window" in detail


def test_failed_thesis_reversal_rejects_stale_loss_and_missing_pattern():
    configured = replace(
        settings,
        confidence_threshold=0.70,
        failed_thesis_reversal_enabled=True,
        failed_thesis_reversal_min_confidence=0.60,
        failed_thesis_reversal_max_age_bars=6,
    )
    with patch("risk.manager.settings", configured):
        stale_ok, stale_detail = RiskManager.qualify_failed_thesis_reversal(
            "SELL",
            0.60,
            _failed_buy_history("2026-08-11T05:00:00+00:00"),
            _analysis("M5"),
            _analysis("M15", adx=22.0),
        )
        pattern_ok, pattern_detail = RiskManager.qualify_failed_thesis_reversal(
            "SELL",
            0.60,
            _failed_buy_history("2026-08-11T06:20:00+00:00"),
            _analysis("M5", patterns=[]),
            _analysis("M15", adx=22.0),
        )

    assert not stale_ok
    assert "outside" in stale_detail
    assert not pattern_ok
    assert "pattern" in pattern_detail


def test_failed_thesis_mode_requires_engine_source_and_can_bridge_h1_lag():
    m5 = _analysis("M5")
    m15 = _analysis("M15", adx=22.0)
    h1 = _analysis("H1", direction="BULLISH", adx=23.0)
    h4 = _analysis("H4", direction="BULLISH", adx=25.0)
    spoofed = {
        "_strategy": {
            "mode": "FAILED_THESIS_REVERSAL",
            "source": "MODEL",
        }
    }
    verified = {
        "_strategy": {
            "mode": "FAILED_THESIS_REVERSAL",
            "source": "DETERMINISTIC_FAILED_THESIS_REVERSAL",
        }
    }

    assert RiskManager._resolve_strategy_mode(
        "SELL", spoofed, m5, m15, h1, h4
    ) == ""
    assert RiskManager._resolve_strategy_mode(
        "SELL", verified, m5, m15, h1, h4
    ) == "FAILED_THESIS_REVERSAL"
    ok, detail = RiskManager._check_entry_structure(
        "SELL",
        m5,
        m15,
        h1,
        h4,
        strategy_mode="FAILED_THESIS_REVERSAL",
    )
    assert ok, detail


def test_failed_thesis_mode_rechecks_deterministic_pattern_inside_risk_gate():
    ok, detail = RiskManager._check_entry_structure(
        "SELL",
        _analysis("M5", patterns=[]),
        _analysis("M15", adx=22.0),
        _analysis("H1", direction="BULLISH"),
        _analysis("H4", direction="BULLISH"),
        strategy_mode="FAILED_THESIS_REVERSAL",
    )

    assert not ok
    assert "Failed-Thesis Reversal" in detail
