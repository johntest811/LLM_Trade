from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.shadow_outcomes import (
    evaluate_exit_counterfactual,
    evaluate_shadow_candidate,
)
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


def test_exit_counterfactual_uses_only_wholly_post_exit_bars():
    exited = datetime(2026, 8, 3, 10, 0, 30, tzinfo=timezone.utc)
    candidate = {
        "action": "BUY",
        "entry": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "actual_exit_price": 1.1005,
        "realized_r": 0.40,
        "exit_time_utc": exited.isoformat(),
        "horizon_minutes": 15,
    }
    bars = [
        # This bar contains pre-exit price path and must be ignored.
        {"time": "2026-08-03T10:00:00Z", "high": 1.1021, "low": 1.0998, "close": 1.1019},
        {"time": "2026-08-03T10:15:00Z", "high": 1.1012, "low": 1.1002, "close": 1.1010},
    ]

    result = evaluate_exit_counterfactual(
        candidate, bars, now_utc=exited + timedelta(minutes=16)
    )

    assert result is not None
    assert result.status == "HORIZON"
    assert result.outcome_r == pytest.approx(1.0)
    assert result.delta_vs_realized_r == pytest.approx(0.6)


def test_exit_counterfactual_records_target_reached_after_exit():
    candidate = {
        "action": "SELL",
        "entry": 181.200,
        "stop_loss": 181.500,
        "take_profit": 180.660,
        "actual_exit_price": 181.100,
        "realized_r": 0.30,
        "exit_time_utc": "2026-08-03T10:00:30Z",
        "horizon_minutes": 30,
    }
    bars = [
        {"time": "2026-08-03T10:01:00Z", "high": 181.15, "low": 180.65, "close": 180.70},
    ]

    result = evaluate_exit_counterfactual(candidate, bars)

    assert result is not None
    assert result.status == "TARGET_AFTER_EXIT"
    assert result.outcome_r == pytest.approx(1.8)
    assert result.delta_vs_realized_r == pytest.approx(1.5)


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
        "rejection_reason": "REJECTED [Structure Gate]: test gate",
        "horizon_minutes": 60,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
    assert summary["gate_breakdown"][0] == {
        "gate": "Structure Gate",
        "resolved": 1,
        "wins": 1,
        "losses": 0,
        "expectancy_r": 2.0,
    }


def test_replay_logger_deduplicates_exit_counterfactuals(tmp_path):
    logger = TradeReplayLogger(str(tmp_path / "exit-shadow.db"))
    plan = {
        "entry": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "confidence": 0.80,
    }
    logger.log_replay_attempt(
        "EURUSD", "BUY", "", plan, {}, {}, 80.0, 70.0,
        "OPEN", ticket=202,
    )
    closed_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    trade = {
        "position_id": 202,
        "symbol": "EURUSD",
        "direction": "BUY",
        "open_price": 1.1000,
        "close_price": 1.1005,
        "close_time": closed_at,
        "close_reason": "PROFIT_GIVEBACK",
        "rr_achieved": 0.40,
    }

    assert logger.log_exit_candidates(
        account_login=1001, trades=[trade], horizons_minutes=[15, 30]
    ) == 2
    assert logger.log_exit_candidates(
        account_login=1001, trades=[trade], horizons_minutes=[15, 30]
    ) == 0
    pending = logger.get_pending_exit_candidates()
    assert len(pending) == 2
    assert logger.resolve_exit_candidate(
        pending[0]["id"],
        status="HORIZON",
        resolved_at_utc=datetime.now(timezone.utc).isoformat(),
        exit_price=1.1010,
        outcome_r=1.0,
        delta_vs_realized_r=0.6,
        post_exit_mfe_r=1.2,
        post_exit_mae_r=0.3,
    )
    summary = logger.exit_counterfactual_summary()
    assert summary["exit_pending"] == 1
    assert summary["exit_resolved"] == 1
    assert summary["exit_held_better"] == 1
    assert summary["exit_average_delta_r"] == 0.6


def test_shadow_summary_exposes_symmetric_direction_funnel(tmp_path):
    logger = TradeReplayLogger(str(tmp_path / "directions.db"))
    base_decision = {
        "entry": 1.1000,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "confidence": 0.80,
    }
    logger.log_replay_attempt(
        "EURUSD", "BUY", "", base_decision, {}, {}, 80.0, 70.0,
        "OPEN", ticket=101,
    )
    assert logger.update_replay_outcome(101, 0.50)
    logger.log_replay_attempt(
        "GBPUSD", "SELL", "", base_decision, {}, {}, 75.0, 65.0,
        "REJECTED: REJECTED [Structure Gate]: test",
    )

    created = datetime.now(timezone.utc).isoformat()
    buy_shadow = logger.log_shadow_candidate(
        symbol="AUDUSD",
        action="BUY",
        candle_time="2026-08-11T10:00:00Z",
        entry=1.1000,
        stop_loss=1.0990,
        take_profit=1.1020,
        rejection_stage="RISK REJECTED",
        rejection_reason="REJECTED [Structure Gate]: buy test",
        horizon_minutes=60,
        created_at_utc=created,
    )
    sell_shadow = logger.log_shadow_candidate(
        symbol="USDCHF",
        action="SELL",
        candle_time="2026-08-11T10:00:00Z",
        entry=1.1000,
        stop_loss=1.1010,
        take_profit=1.0980,
        rejection_stage="RISK REJECTED",
        rejection_reason="REJECTED [Structure Gate]: sell test",
        horizon_minutes=60,
        created_at_utc=created,
    )
    assert logger.resolve_shadow_candidate(
        buy_shadow,
        status="TP",
        resolved_at_utc="2026-08-11T10:20:00Z",
        exit_price=1.1020,
        outcome_r=2.0,
        mfe_r=2.0,
        mae_r=0.1,
    )
    assert logger.resolve_shadow_candidate(
        sell_shadow,
        status="SL",
        resolved_at_utc="2026-08-11T10:20:00Z",
        exit_price=1.1010,
        outcome_r=-1.0,
        mfe_r=0.2,
        mae_r=1.0,
    )

    summary = logger.shadow_summary()
    directions = {
        row["action"]: row for row in summary["direction_breakdown"]
    }

    assert summary["evidence_window_hours"] == 48
    assert directions["BUY"]["candidates"] == 1
    assert directions["BUY"]["executed"] == 1
    assert directions["BUY"]["closed_net_usd"] == 0.50
    assert directions["BUY"]["shadow_expectancy_r"] == 2.0
    assert directions["BUY"]["top_rejection_gates"] == [
        {"gate": "Structure Gate", "count": 1}
    ]
    assert directions["SELL"]["candidates"] == 1
    assert directions["SELL"]["rejected"] == 1
    assert directions["SELL"]["executed"] == 0
    assert directions["SELL"]["shadow_expectancy_r"] == -1.0
    assert directions["SELL"]["top_rejection_gates"] == [
        {"gate": "Structure Gate", "count": 1}
    ]
