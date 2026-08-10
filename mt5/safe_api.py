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


class _SerializedMT5Proxy:
    def __init__(self, module: Any) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_wrappers", {})

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._module, name)
        if not callable(attribute):
            return attribute
        wrappers = self._wrappers
        if name not in wrappers:
            @functools.wraps(attribute)
            def guarded(*args: Any, **kwargs: Any) -> Any:
                with _NATIVE_GATE:
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
