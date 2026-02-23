import atexit
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import functools
import sys

# 全局线程池复用，避免频繁创建
EXECUTOR = ThreadPoolExecutor(max_workers=4)

def setup_logging(level: str = "INFO"):
    levelno = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=levelno,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

def call_with_timeout(func, args=(), kwargs=None, timeout: int = 90):
    if kwargs is None:
        kwargs = {}
    fut = EXECUTOR.submit(func, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except FuturesTimeoutError:
        try:
            fut.cancel()
        except Exception:
            pass
        raise TimeoutError(f"调用超时（>{timeout}s）")
    except Exception:
        raise

def shutdown_executor():
    try:
        EXECUTOR.shutdown(wait=False)
    except Exception:
        pass

atexit.register(shutdown_executor)