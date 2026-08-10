from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.shadow_outcomes import evaluate_shadow_candidate
from database.replay_logger import TradeReplayLogger


def _candidate(action="BUY", *, created=None):
    created = created or datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)
    return {
        "created_at_utc": created.isoformat(),
        "action": action,
        "entry": 1.1000,
        "stop_loss": 1.0990 if action == "BUY" else 1.1010,
        "take_profit": 1.1020 if action == "BUY" else 1.0980,
        "horizon_minutes": 60,
    }


def test_buy_shadow_resolves_target_and_tracks_excursion():
    bars = pd.DataFrame(
        [
            {"time": "2026-08-03T10:01:00Z", "high": 1.1006, "low": 1.0998, "close": 1.1004},
            {"time": "2026-08-03T10:02:00Z", "high": 1.1021, "low": 1.1002, "close": 1.1019},
        ]
    )

    result = evaluate_shadow_candidate(_candidate(), bars)

    assert result is not None
    assert result.status == "TP"
    assert result.outcome_r == pytest.approx(2.0)
    assert result.mfe_r >= 2.0


def test_same_bar_stop_and_target_is_ambiguous_not_a_win():
    bars = [
        {
            "time": "2026-08-03T10:01:00Z",
            "high": 1.1021,
            "low": 1.0989,
            "close": 1.1005,
        }
    ]

    result = evaluate_shadow_candidate(_candidate(), bars)

    assert result is not None
    assert result.status == "AMBIGUOUS"
    assert result.outcome_r is None


def test_timeout_uses_directional_close_return():
    created = datetime(2026, 8, 3, 10, 0, tzinfo=timezone.utc)
    bars = [
        {
            "time": "2026-08-03T10:59:00Z",
            "high": 1.1007,
            "low": 1.0998,
            "close": 1.1005,
        }
    ]

    result = evaluate_shadow_candidate(
        _candidate(created=created),
        bars,
        now_utc=created + timedelta(minutes=61),
    )

    assert result is not None
    assert result.status == "TIMEOUT_WIN"
    assert round(result.outcome_r or 0.0, 3) == 0.5


def test_replay_logger_deduplicates_and_summarizes_shadow_rows(tmp_path):
    logger = TradeReplayLogger(str(tmp_path / "shadow.db"))
    values = {
        "symbol": "EURUSD",
        "action": "BUY",
        "candle_time": "2026-08-03T10:00:00Z",
        "entry": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "rejection_stage": "RISK REJECTED",
        "rejection_reason": "test gate",
        "horizon_minutes": 60,
        "created_at_utc": "2026-08-03T10:00:01Z",
    }
    first = logger.log_shadow_candidate(**values)
    second = logger.log_shadow_candidate(**values)

    assert first == second
    pending = logger.get_pending_shadow_candidates()
    assert len(pending) == 1
    assert logger.resolve_shadow_candidate(
        first,
        status="TP",
        resolved_at_utc="2026-08-03T10:20:00Z",
        exit_price=1.1020,
        outcome_r=2.0,
        mfe_r=2.1,
        mae_r=0.2,
    )
    summary = logger.shadow_summary()
    assert summary["pending"] == 0
    assert summary["resolved"] == 1
    assert summary["wins"] == 1
    assert summary["expectancy_r"] == 2.0
