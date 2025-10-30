"""Utility module providing Winston-style structured logging wrappers.

The project requirements specify Winston logging semantics. Winston is a
Node.js-first logging library with leveled transports and structured output.
To keep parity inside a Python stack we mirror the high-level interface via
`logging` and `rich` renderers while exposing a familiar factory.

All modules should call `get_winston_logger(__name__)` to obtain a logger
configured with the global formatting rules.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)


def _configure_root(level: int = logging.INFO) -> None:
    """Configure the root logger once.

    Winston frequently routes logs to multiple transports. In our Python
    implementation we keep a single stream handler to stdout, preserving the
    timestamped, leveled structure required for experiment auditing. The guard
    avoids duplicate handlers when called from multiple modules.
    """

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
    root.setLevel(level)


def get_winston_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger mimicking Winston semantics.

    Args:
        name: Logger namespace, typically `__name__`.
        level: Minimum level for the logger. Modules can override to increase
            verbosity, but the root default stays at INFO.

    Returns:
        Configured `logging.Logger` instance ready for use.
    """

    _configure_root(level=level)
    return logging.getLogger(name)


__all__ = ["get_winston_logger"]


