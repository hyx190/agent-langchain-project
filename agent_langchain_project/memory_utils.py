# （完整文件）自动保存对话 turn 的工具
from typing import Optional
import time
import os
import hashlib
import logging
import re

from .memory import get_long_memory, init_long_memory
from .context_request_id import get_request_id

logger = logging.getLogger("agent.memory_utils")

# 配置（可通过环境变量覆盖）
AUTO_SAVE_MODE = os.getenv("AGENT_AUTO_SAVE", "always")  # "always" / "on_change" / "off"
AUTO_SAVE_SUMMARIZE = os.getenv("AGENT_MEMORY_SUMMARIZE", "true").lower() in ("1", "true", "yes")
MAX_MEMORY_ITEMS = int(os.getenv("AGENT_MAX_MEMORY_ITEMS", "10000"))
MIN_LENGTH_TO_SUMMARIZE = int(os.getenv("AGENT_MIN_LENGTH_TO_SUMMARIZE", "200"))  # chars

# 简单敏感信息检测正则（可扩展）
_SENSITIVE_PATTERNS = [
    re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9\-\._]{8,}", re.I),
    re.compile(r"secret\s*[:=]\s*[A-Za-z0-9\-\._]{8,}", re.I),
    re.compile(r"password\s*[:=]\s*\S{4,}", re.I),
    re.compile(r"-----BEGIN (RSA|PRIVATE) KEY-----", re.I),
    re.compile(r"ssh-rsa\s+[A-Za-z0-9+/=]{100,}", re.I),
]

def _contains_sensitive(text: str) -> bool:
    if not text:
        return False
    for p in _SENSITIVE_PATTERNS:
        if p.search(text):
            return True
    return False

def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]

def _should_save(user_text: str, assistant_text: str) -> bool:
    mode = AUTO_SAVE_MODE
    if mode == "off":
        return False
    if _contains_sensitive(user_text) or _contains_sensitive(assistant_text):
        logger.info("memory_utils.skip_sensitive", extra={"request_id": get_request_id()})
        return False
    if mode == "on_change":
        mem = get_long_memory()
        if mem is None:
            return True
        last_hash = _short_hash(user_text + "\n" + assistant_text)
        n = min(10, len(mem.texts))
        for i in range(1, n+1):
            idx = len(mem.texts) - i
            try:
                if _short_hash(mem.texts[idx]) == last_hash:
                    logger.debug("memory_utils.duplicate_detected", extra={"request_id": get_request_id(), "idx": idx})
                    return False
            except Exception:
                continue
        return True
    return True

def _maybe_summarize(text: str) -> str:
    if not text:
        return ""
    if not AUTO_SAVE_SUMMARIZE:
        return text[:3000]
    if len(text) < MIN_LENGTH_TO_SUMMARIZE:
        return text[:1000]
    head = text[:300]
    tail = text[-200:]
    head = " ".join(head.split())
    tail = " ".join(tail.split())
    summary = head + " ... " + tail
    return summary[:1200]

def trim_memory_if_needed(mem) -> None:
    try:
        if not mem:
            return
        while len(mem.texts) > MAX_MEMORY_ITEMS:
            mem.texts.pop(0)
            mem.metadatas.pop(0)
    except Exception as e:
        logger.exception("memory_utils.trim_fail", extra={"error": str(e), "request_id": get_request_id()})

def auto_save_turn(user_text: str,
                   assistant_text: Optional[str],
                   request_id: Optional[str] = None,
                   mem = None,
                   persist: bool = True) -> Optional[int]:
    try:
        if request_id is None:
            request_id = get_request_id()
        if not _should_save(user_text, assistant_text or ""):
            return None

        if mem is None:
            mem = get_long_memory()
        if mem is None:
            logger.debug("memory_utils.no_mem", extra={"request_id": request_id})
            return None

        combined = f"User: {user_text}\nAssistant: {assistant_text or ''}"
        summary = _maybe_summarize(combined)
        meta = {
            "type": "turn",
            "source": "auto",
            "request_id": request_id,
            "ts": time.time(),
            "user_len": len(user_text or ""),
            "assistant_len": len(assistant_text or ""),
        }
        idx = mem.add(summary, meta=meta)
        logger.info("memory_utils.saved_turn", extra={"request_id": request_id, "idx": idx, "summary_len": len(summary)})
        try:
            trim_memory_if_needed(mem)
        except Exception:
            pass
        if persist:
            try:
                mem.persist()
            except Exception:
                logger.exception("memory_utils.persist_fail", extra={"request_id": request_id, "idx": idx})
        return idx
    except Exception as e:
        logger.exception("memory_utils.save_error", extra={"error": str(e), "request_id": request_id})
        return None