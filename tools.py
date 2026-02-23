"""
tongyi_agent/tools.py

工具注册与默认实现（包含 SavePersona 的修复版本）。
提供 build_tools() 返回一个 Tool 列表，每个 Tool 至少包含:
  - name: str
  - func: Callable[[str], Any]
  - description: str

注意：
- logging.extra 中避免使用与 LogRecord 冲突的键（例如 'name', 'args' 等）。
- SavePersona 会把 meta.type='persona' 写入长期记忆并立即 persist。
"""

from dataclasses import dataclass
from typing import Callable, Any, Dict, List, Optional
import logging
import os
import time
# 导入整个模块（避免按名字导入引起的 ImportError）
from . import tools_trading
from .memory import init_long_memory, get_long_memory
from .context_request_id import get_request_id
from .tools_trading import (
    tool_read_trading_csv,
    tool_analyze,
    tool_suggest,
    tool_simulate,
)
logger = logging.getLogger("agent.tools")


@dataclass
class Tool:
    name: str
    func: Callable[[str], Any]
    description: str = ""


# ---- Persona / Memory tools ----

def _get_mem():
    path = os.getenv("AGENT_VECTOR_STORE_PATH", os.path.expanduser("~/.agent_vector_store"))
    model_path = os.getenv("AGENT_EMBEDDING_MODEL_PATH", "")
    try:
        mem = init_long_memory(index_path=path, model_path=model_path)
        return mem
    except Exception as e:
        logger.exception("memory.init_failed", extra={"request_id": get_request_id(), "error": str(e)})
        return None


def save_persona(text: str, name: str = "") -> Dict[str, Any]:
    """
    将 persona 保存到长期记忆，meta 包含 type="persona"，并立即持久化。
    返回 dict: { saved_index, persisted, meta } 或 { error: ... }
    """
    try:
        mem = _get_mem()
        if mem is None:
            return {"error": "long_memory_unavailable"}
    except Exception as e_init:
        return {"error": f"init_long_memory failed: {e_init}"}

    meta = {
        "type": "persona",
        "persona_name": name or (text.splitlines()[0] if text else ""),
        "ts": time.time(),
        "request_id": get_request_id(),
        "source": "manual_save"
    }

    try:
        idx = mem.add(text, meta=meta)
    except Exception as e_add:
        logger.exception("save_persona.add_failed", extra={"request_id": get_request_id(), "error": str(e_add)})
        return {"error": f"mem.add failed: {e_add}"}

    # Log safely (avoid reserved keys)
    try:
        logger.info("save_persona.saved", extra={
            "request_id": get_request_id(),
            "persona_idx": idx,
            "persona_name_safe": meta["persona_name"]
        })
    except Exception:
        # swallow logging error
        pass

    try:
        persisted = mem.persist()
    except Exception as e_persist:
        persisted = False
        try:
            logger.exception("save_persona.persist_failed", extra={
                "request_id": get_request_id(),
                "persona_idx": idx,
                "error": str(e_persist)
            })
        except Exception:
            pass

    try:
        logger.info("save_persona.complete", extra={
            "request_id": get_request_id(),
            "persona_idx": idx,
            "persona_name_safe": meta["persona_name"],
            "persisted": bool(persisted)
        })
    except Exception:
        pass

    return {"saved_index": idx, "persisted": bool(persisted), "meta": meta}


def save_persona_tool(text: str) -> Dict[str, Any]:
    """
    Wrapper expected by CLI/tools registration.
    Supports two input forms:
      - raw persona text
      - prefixed "persona_name:XXX\n..." (used by autosave)
    """
    # try parse persona_name: prefix
    name = ""
    body = text
    try:
        if isinstance(text, str) and text.startswith("persona_name:"):
            # very simple parse: persona_name:<name>\nrest...
            parts = text.splitlines()
            first = parts[0]
            _, maybe_name = first.split(":", 1)
            name = maybe_name.strip()
            body = "\n".join(parts[1:]).strip() or name
    except Exception:
        body = text

    return save_persona(body, name=name)


def save_memory_tool(text: str) -> Dict[str, Any]:
    """
    Save an arbitrary memory entry (type 'note').
    """
    mem = _get_mem()
    if mem is None:
        return {"error": "long_memory_unavailable"}
    meta = {
        "type": "note",
        "ts": time.time(),
        "request_id": get_request_id(),
        "source": "manual_save"
    }
    try:
        idx = mem.add(text, meta=meta)
        persisted = mem.persist()
        try:
            logger.info("save_memory", extra={"request_id": get_request_id(), "memory_idx": idx})
        except Exception:
            pass
        return {"saved_index": idx, "persisted": bool(persisted)}
    except Exception as e:
        logger.exception("save_memory.error", extra={"request_id": get_request_id(), "error": str(e)})
        return {"error": str(e)}


def recall_memory_tool(query: str) -> List[Dict[str, Any]]:
    """
    Simple search wrapper that returns search hits.
    """
    mem = get_long_memory()
    if mem is None:
        mem = _get_mem()
    if mem is None:
        return []
    try:
        hits = mem.search(query, k=5)
        return hits
    except Exception as e:
        logger.exception("recall_memory.error", extra={"request_id": get_request_id(), "error": str(e)})
        return []


def list_personas_tool(_: str = "") -> List[Dict[str, Any]]:
    """
    Return list of persona metas (no arguments).
    """
    mem = get_long_memory()
    if mem is None:
        mem = _get_mem()
    if mem is None:
        return []
    res = []
    try:
        # iterate through metadatas and texts
        for t, m in zip(mem.texts, mem.metadatas):
            if isinstance(m, dict) and m.get("type") == "persona":
                res.append({"meta": m, "text": t})
    except Exception as e:
        logger.exception("list_personas.error", extra={"request_id": get_request_id(), "error": str(e)})
    return res


def persist_memory_tool(_: str = "") -> Dict[str, Any]:
    mem = get_long_memory()
    if mem is None:
        mem = _get_mem()
    if mem is None:
        return {"error": "long_memory_unavailable"}
    try:
        ok = mem.persist()
        return {"persisted": bool(ok)}
    except Exception as e:
        logger.exception("persist_memory.error", extra={"request_id": get_request_id(), "error": str(e)})
        return {"error": str(e)}


def load_memory_tool(_: str = "") -> Dict[str, Any]:
    """
    No-op for many backends; return info.
    """
    mem = get_long_memory()
    if mem is None:
        mem = _get_mem()
    if mem is None:
        return {"error": "long_memory_unavailable"}
    try:
        info = mem.info() if hasattr(mem, "info") else {}
        return {"info": info}
    except Exception as e:
        logger.exception("load_memory.error", extra={"request_id": get_request_id(), "error": str(e)})
        return {"error": str(e)}


# ---- FetchRenderedPage tool wrapper ----
def fetch_rendered_page_tool(arg: str) -> Dict[str, Any]:
    """
    Wrapper around tongyi_agent.tools_fetch.fetch_rendered_page.
    Expects arg to be URL (and optionally other params).
    Returns the dict as produced by the underlying function, but ensures we return a stringable summary
    if upper layers expect a string.
    """
    try:
        from .tools_fetch import fetch_rendered_page  # type: ignore
    except Exception as e:
        logger.exception("fetch_tool.import_fail", extra={"request_id": get_request_id(), "error": str(e)})
        return {"source": "error", "text": "", "error": f"fetch_impl_missing: {e}"}

    url = (arg or "").strip()
    if not url:
        return {"source": "error", "text": "", "error": "no_url_provided"}
    try:
        res = fetch_rendered_page(url)
        # normalize: ensure keys present
        if isinstance(res, dict):
            return res
        else:
            return {"source": "unknown", "text": str(res), "error": None}
    except Exception as e:
        logger.exception("fetch_tool.exec_error", extra={"request_id": get_request_id(), "error": str(e)})
        return {"source": "error", "text": "", "error": str(e)}


# ---- Build tool list ----
def build_tools() -> List[Tool]:
    """
    Return default tool list used by CLI/runner.
    Adjust order/contents as needed.
    """
    tools: List[Tool] = [
        Tool(name="SavePersona", func=save_persona_tool, description="Save a persona to long-term memory"),
        Tool(name="SaveMemory", func=save_memory_tool, description="Save arbitrary text to long-term memory"),
        Tool(name="RecallMemory", func=recall_memory_tool, description="Recall related memories"),
        Tool(name="ListPersonas", func=list_personas_tool, description="List stored personas"),
        Tool(name="PersistMemory", func=persist_memory_tool, description="Persist memory to disk"),
        Tool(name="LoadMemory", func=load_memory_tool, description="Load memory metadata/info"),
        Tool(name="FetchRenderedPage", func=fetch_rendered_page_tool, description="Fetch rendered page (Playwright or requests fallback)"),
        Tool(name="ReadTradingCSV", func=tool_read_trading_csv, description="Read trading CSV or 'clipboard'"),
        Tool(name="AnalyzePortfolio", func=tool_analyze, description="Analyze portfolio snapshot or file"),
        Tool(name="SuggestAction", func=tool_suggest, description="Suggest trading actions (read-only)"),
        Tool(name="SimulateTrade", func=tool_simulate, description="Simulate a trade given action and snapshot"),
    ]
    return tools


# Backwards-compatible alias for older code paths that expect a dict-like tool
def build_langchain_tools():
    """
    Try to build a list consumable by LangChain if available.
    Returns None if langchain is not present.
    """
    try:
        import langchain  # noqa: F401
    except Exception:
        return None

    lc_tools = []
    for t in build_tools():
        # Create a simple callable for LangChain tool wrapper
        def _make_func(func):
            return lambda x: func(x)
        lc_tools.append(Tool(name=t.name, func=_make_func(t.func), description=t.description))
    return lc_tools