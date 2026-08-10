"""Cross-process leader guard for the single-terminal trading engine."""

from __future__ import annotations

import socket
import zlib
from pathlib import Path
from typing import Optional


class InstanceLock:
    def __init__(self, workspace: Path) -> None:
        fingerprint = zlib.crc32(str(workspace.resolve()).lower().encode("utf-8"))
        self.port = 45000 + fingerprint % 4000
        self._socket: Optional[socket.socket] = None

    def acquire(self) -> None:
        guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            guard.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            guard.bind(("127.0.0.1", self.port))
            guard.listen(1)
        except OSError as exc:
            guard.close()
            raise RuntimeError(
                "Another LLM Trading Terminal instance already owns this workspace. "
                "Stop it before starting a second engine."
            ) from exc
        self._socket = guard

    def release(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
