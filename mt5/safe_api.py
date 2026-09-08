"""Process-wide serialization for the MetaTrader 5 Python session.

The MetaTrader5 package exposes one process-global terminal connection.
Initialization, shutdown, order submission, and background reads must not race
each other across worker threads. The proxy serializes every native call, while
``serialized_mt5`` lets an order/reconnect hold the same re-entrant gate across
its full multi-call transaction.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

import MetaTrader5 as _native_mt5


_NATIVE_GATE = threading.RLock()
_T = TypeVar("_T")
_SYMBOL_CALLS = {
    "symbol_info", "symbol_info_tick", "symbol_select", "copy_rates_from",
    "copy_rates_from_pos", "copy_rates_range", "copy_ticks_from", "copy_ticks_range",
    "order_calc_profit", "order_calc_margin",
}


class _SerializedMT5Proxy:
    def __init__(self, module: Any) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_wrappers", {})
        object.__setattr__(self, "_symbol_names", {})

    def register_symbols(self, symbols: Any) -> None:
        """Map normalized app IDs to exact broker names; never guess on ties."""
        with _NATIVE_GATE:
            grouped = {}
            for item in symbols:
                name = str(getattr(item, "name", "") or "")
                if name:
                    grouped.setdefault(name.upper(), set()).add(name)
            self._symbol_names.clear()
            self._symbol_names.update({key: next(iter(names)) for key, names in grouped.items() if len(names) == 1})

    def broker_symbol_name(self, symbol: str) -> str:
        """Resolve app IDs for exact-match broker history queries as well."""
        with _NATIVE_GATE:
            return self._symbol_names.get(symbol.upper(), symbol)

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._module, name)
        if not callable(attribute):
            return attribute
        wrappers = self._wrappers
        if name not in wrappers:
            @functools.wraps(attribute)
            def guarded(*args: Any, **kwargs: Any) -> Any:
                with _NATIVE_GATE:
                    if name in ("initialize", "login", "shutdown"):
                        self._symbol_names.clear()
                    if name in ("initialize", "login"):
                        initialized = getattr(self._module, name)(*args, **kwargs)
                        if initialized:
                            self.register_symbols(self._module.symbols_get() or ())
                        return initialized
                    # The dashboard uses uppercase IDs; brokers can expose
                    # case-sensitive contract names such as SpotBrent or .a.
                    symbol_index = 1 if name in ("order_calc_profit", "order_calc_margin") else 0
                    if name in _SYMBOL_CALLS and len(args) > symbol_index:
                        positional = list(args)
                        value = positional[symbol_index]
                        if isinstance(value, str):
                            positional[symbol_index] = self._symbol_names.get(value.upper(), value)
                        args = tuple(positional)
                    if "symbol" in kwargs:
                        kwargs = dict(kwargs)
                        value = kwargs["symbol"]
                        kwargs["symbol"] = self._symbol_names.get(str(value).upper(), value)
                    if name in ("order_check", "order_send") and len(args) == 1 and isinstance(args[0], dict):
                        request = dict(args[0])
                        value = request.get("symbol")
                        if value is not None:
                            request["symbol"] = self._symbol_names.get(str(value).upper(), value)
                        args = (request,)
                    # MetaTrader5 5.0.5735 rejects request dictionaries that
                    # reach order_check/order_send through ``*args`` with
                    # ``(-2, 'Unnamed arguments not allowed')``.  The native
                    # extension requires a literal one-positional-argument
                    # call for these two endpoints.  Keep every other native
                    # function on the generic serialized path.
                    if len(args) == 1 and not kwargs:
                        if name == "order_check":
                            return self._module.order_check(args[0])
                        if name == "order_send":
                            return self._module.order_send(args[0])
                    return getattr(self._module, name)(*args, **kwargs)

            wrappers[name] = guarded
        return wrappers[name]


mt5 = _SerializedMT5Proxy(_native_mt5)


@contextmanager
def mt5_session() -> Iterator[None]:
    """Hold exclusive ownership across a multi-call native transaction."""
    with _NATIVE_GATE:
        yield


def serialized_mt5(function: Callable[..., _T]) -> Callable[..., _T]:
    """Serialize an entire synchronous MT5 worker transaction."""
    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> _T:
        with _NATIVE_GATE:
            return function(*args, **kwargs)

    return guarded
