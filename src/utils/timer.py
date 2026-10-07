from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
import logging


logger = logging.getLogger(__name__)


@contextmanager
def time_block(name: str):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000.0
        logger.info(f"{name}: {elapsed_ms:.2f} ms")
