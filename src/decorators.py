import time
import asyncio
import functools
from contextlib import contextmanager

from src.logger import DicLogger, LOGGING_CONFIG

log = DicLogger(LOGGING_CONFIG).log


def duration(func):
    """Tracks duration of ansyc function"""

    @contextmanager
    def wrapping_logic():
        start_ts = time.time()
        yield
        log.info(f'Function {func.__name__} took {(time.time() - start_ts):.2f} seconds')

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not asyncio.iscoroutinefunction(func):
            with wrapping_logic():
                return func(*args, **kwargs)

        async def tmp():
            with wrapping_logic():
                return await func(*args, **kwargs)
        return tmp()

    return wrapper
