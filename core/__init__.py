"""Core package exports without importing the complete runtime eagerly.

Eagerly importing ``TradingEngine`` here created an order-dependent circular
import when the prompt builder imported ``core.evidence``.  Keep the public
exports, but resolve them only when a caller explicitly requests them.
"""

from typing import Any

__all__ = ["TradingEngine", "MarketAnalysisEngine"]


def __getattr__(name: str) -> Any:
    if name == "TradingEngine":
        from core.engine import TradingEngine

        return TradingEngine
    if name == "MarketAnalysisEngine":
        from core.analysis_engine import MarketAnalysisEngine

        return MarketAnalysisEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
