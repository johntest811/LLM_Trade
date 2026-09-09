"""Shared account-currency budgets for discovery and final entry validation."""

import math


def risk_capital(account):
    """Never compound floating gains or size through an equity drawdown.

    Both fields must be explicit and valid. A missing equity reading is not
    evidence that an account is flat or that its balance is available to risk.
    """
    try:
        values = (float(account["balance"]), float(account["equity"]))
        return min(values) if all(math.isfinite(v) and v > 0 for v in values) else 0.0
    except (KeyError, TypeError, ValueError, OverflowError):
        return 0.0


def daily_loss_limit(account, config):
    capital = risk_capital(account)
    if capital <= 0:
        return 0.0
    try:
        dollar, percent = float(config.max_daily_loss_usd), float(config.max_daily_loss_pct)
        if not all(math.isfinite(v) and v >= 0 for v in (dollar, percent)):
            return 0.0
        limits = [value for value in (dollar, capital * percent / 100) if value > 0]
        return min(limits) if limits else math.inf
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0


def entry_risk_budget(account, config, *, risk_percent=None, daily_loss=0.0):
    """Combine percentage, dollar and remaining daily limits; never round up."""
    capital = risk_capital(account)
    try:
        percent = float(config.risk_percent if risk_percent is None else risk_percent)
        loss = float(daily_loss)
        if capital <= 0 or not math.isfinite(percent) or percent <= 0 or not math.isfinite(loss) or loss < 0:
            return 0.0
        budget = min(capital * percent / 100, max(0.0, daily_loss_limit(account, config) - loss))
        if config.auto_close_loss_enabled:
            dollar = float(config.auto_close_loss_usd)
            if not math.isfinite(dollar) or dollar < 0:
                return 0.0
            if dollar > 0:
                budget = min(budget, dollar)
        return budget if math.isfinite(budget) and budget > 0 else 0.0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0
