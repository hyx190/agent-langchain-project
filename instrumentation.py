import logging
import time
from functools import wraps
from .context_request_id import get_request_id

logger = logging.getLogger("agent.instrumentation")

def log_tool_call(tool_name: str):
    """
    装饰器/包装器：记录工具调用的开始、结束、耗时与异常（供工具函数包装或直接在调用处使用）。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rid = get_request_id()
            start = time.time()
            logger.debug("tool_call.start", extra={"request_id": rid, "tool": tool_name, "args_summary": str(args)[:500]})
            try:
                res = func(*args, **kwargs)
                duration_ms = int((time.time() - start) * 1000)
                logger.info("tool_call.success", extra={
                    "request_id": rid, "tool": tool_name, "duration_ms": duration_ms,
                    "result_summary": str(res)[:2000]
                })
                return res
            except Exception as e:
                duration_ms = int((time.time() - start) * 1000)
                logger.exception("tool_call.error", extra={
                    "request_id": rid, "tool": tool_name, "duration_ms": duration_ms, "error": str(e)
                })
                raise
        return wrapper
    return decorator

def audit_memory_write(key_name: str):
    """
    装饰 memory.write/add/persist 等方法以记录 old/new 摘要与谁发起的 request_id。
    用法:
      @audit_memory_write("mykey")
      def add(...): ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            rid = get_request_id()
            try:
                old = None
                # 尝试从第一个参数（self 或 memory object）读取可能存在的状态以做摘要（若有）
            except Exception:
                old = None
            logger.debug("memory.write.attempt", extra={"request_id": rid, "key": key_name})
            res = func(*args, **kwargs)
            logger.info("memory.write.done", extra={"request_id": rid, "key": key_name, "result_summary": str(res)[:500]})
            return res
        return wrapper
    return decorator