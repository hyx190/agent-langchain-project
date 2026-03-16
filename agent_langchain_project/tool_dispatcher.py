# ✅ 结论已经确定：LLM 根本没有输出 CALL_TOOL/CALL_TOOL_JSON，而是在“装作调用成功”。
# parsed_tool_call=None 证明 dispatcher 没执行任何工具，所以你当然看不到 TOOL_RESULT_JSON。
#
# 解决办法：在 dispatcher 里加一个“硬门禁”：
# - 如果用户输入本身就是 CALL_TOOL 指令，则直接执行工具（不让 LLM 决定）
# - 或者：如果用户输入包含 CALL_TOOL，但 LLM 没产生可解析的 CALL_TOOL，则判为失败并提示重试
#
# 我给你最稳的方案：优先执行“用户的 CALL_TOOL”，LLM 只负责总结工具结果。
# 这样用户手动输入 CALL_TOOL 时，永远是真调用，不会被模型假装。

from typing import List, Tuple, Optional
import re
import json
import time
import logging
from datetime import datetime

from .context_request_id import get_request_id

try:
    from .prompt_inject import inject_persona_to_system
except Exception:
    def inject_persona_to_system(x):  # type: ignore
        return x or ""

try:
    from .tools import Tool
except Exception:
    class Tool:
        def __init__(self, name, func, description=""):
            self.name = name
            self.func = func
            self.description = description

# ✅ 支持全角冒号
_TOOL_CALL_RE = re.compile(r"^\s*CALL_TOOL[:：]\s*([A-Za-z0-9_]+)(?:\s+(.*))?\s*$", re.IGNORECASE)
_TOOL_CALL_JSON_RE = re.compile(r"CALL_TOOL_JSON[:：]\s*(\{.*\})", re.IGNORECASE | re.DOTALL)

logger = logging.getLogger("agent.dispatcher")


def _parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
    if not text:
        return None

    mjson = _TOOL_CALL_JSON_RE.search(text)
    if mjson:
        try:
            payload = json.loads(mjson.group(1))
            name = payload.get("tool") or payload.get("name")
            args = payload.get("args") or payload.get("arguments") or ""
            if name:
                return name.strip(), str(args).strip()
        except Exception:
            pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _TOOL_CALL_RE.match(line)
        if m:
            name = m.group(1).strip()
            args = (m.group(2) or "").strip()
            return name, args

    return None


_TIME_QUERY_RE = re.compile(r"(今年是哪年|今年是哪一年|现在是(哪年|什么年)|当前年份|现在是哪一年|现在是什么年)", re.I)


def _answer_time_shortcircuit(user_prompt: str):
    if _TIME_QUERY_RE.search(user_prompt or ""):
        now = datetime.now()
        return f"今年是 {now.year} 年。"
    return None


def _tool_list_text(tools: List[Tool]) -> str:
    return "可用工具：" + ", ".join([t.name for t in tools])


def run_with_tools(
    llm_callable,
    user_prompt: str,
    tools: List[Tool],
    system_prompt: Optional[str] = None,
    max_tool_calls: int = 3,
    call_timeout_sec: Optional[int] = None,
    debug: bool = True,
    echo_tool_result: bool = True,
) -> str:
    """
    ✅ 强化版调度器（关键变化）：
    1) 如果“用户输入本身”就是 CALL_TOOL 指令：直接执行工具（绕过 LLM，杜绝假调用）
    2) 其他情况：仍允许 LLM 产生 CALL_TOOL，但必须解析成功才会执行
    """
    import sys as _sys

    short = _answer_time_shortcircuit(user_prompt)
    if short is not None:
        return short

    tool_map = {t.name: t for t in tools}

    # ---------- NEW: user-direct tool call gate ----------
    user_direct = _parse_tool_call(user_prompt or "")
    if user_direct:
        tool_name, tool_args = user_direct
        tool = tool_map.get(tool_name)
        if tool is None:
            available = ", ".join(tool_map.keys())
            return f"[ToolError] unknown tool '{tool_name}'. Available tools: {available}"

        logger.info("dispatcher.user_direct_tool.call", extra={"request_id": get_request_id(), "tool": tool_name})
        try:
            result = tool.func(tool_args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.exception("dispatcher.user_direct_tool.fail", extra={"request_id": get_request_id(), "tool": tool_name, "error": str(e)})
            return f"[ToolError] tool '{tool_name}' failed: {e}"

        # ✅ 强制回显 raw
        raw = result
        if len(raw) > 8000:
            raw = raw[:8000] + "\n... (truncated)"
        if echo_tool_result:
            print("=== TOOL_RESULT_JSON (raw) ===", file=_sys.stderr, flush=True)
            print(raw, file=_sys.stderr, flush=True)
            print("=== END_TOOL_RESULT_JSON ===", file=_sys.stderr, flush=True)

        # 让 LLM 只做总结（并强制引用 raw）
        system_text = inject_persona_to_system(system_prompt or "")
        anti = (
            "\n\n【重要】你必须严格基于 TOOL_RESULT_JSON 总结，不得新增不存在的漏洞/CVE/包名/版本。\n"
            "如果 TOOL_RESULT_JSON 里 vulns 为空，就明确说明“工具未命中漏洞”。\n"
        )
        context = (
            system_text
            + anti
            + "\n\nSystem:\n"
            f"TOOL_CALLED: {tool_name}\n"
            f"TOOL_ARGS: {tool_args}\n"
            "TOOL_RESULT_JSON:\n"
            + result
            + "\nEND_TOOL_RESULT_JSON\n\n"
            "User:\n请根据 TOOL_RESULT_JSON 生成简短摘要。"
        )
        llm_out = llm_callable(context)
        return str(llm_out)
    # ---------- END NEW gate ----------

    # Normal flow: LLM decides whether to call tools
    system_text = inject_persona_to_system(system_prompt or "")
    anti_hallucination = (
        "\n\n【重要】工具调用规则：\n"
        "1) 如需调用工具，必须输出一行 CALL_TOOL 或 CALL_TOOL_JSON。\n"
        "2) 未出现 CALL_TOOL/JSON 就不得声称“工具调用成功”。\n"
        "3) 后续总结必须引用 TOOL_RESULT_JSON。\n"
    )

    context = system_text + anti_hallucination + "\n\n" + _tool_list_text(tools) + "\n\nUser:\n" + (user_prompt or "")
    last_tool_result: Optional[str] = None

    for step in range(max_tool_calls + 1):
        llm_out = llm_callable(context)

        if debug:
            print("\n[debug] llm_out(raw) =====", file=_sys.stderr, flush=True)
            print(str(llm_out), file=_sys.stderr, flush=True)
            print("[debug] llm_out(end) =====\n", file=_sys.stderr, flush=True)

        parsed = _parse_tool_call(llm_out)

        if debug:
            print("[debug] parsed_tool_call = " + repr(parsed), file=_sys.stderr, flush=True)

        if not parsed:
            if last_tool_result:
                raw = last_tool_result
                if len(raw) > 8000:
                    raw = raw[:8000] + "\n... (truncated)"
                return (
                    "=== TOOL_RESULT_JSON (raw) ===\n"
                    + raw
                    + "\n=== END_TOOL_RESULT_JSON ===\n\n"
                    + str(llm_out)
                )
            return str(llm_out)

        tool_name, tool_args = parsed
        tool = tool_map.get(tool_name)
        if tool is None:
            available = ", ".join(tool_map.keys())
            context += "\n\nSystem:\n" + f"[ToolError] unknown tool '{tool_name}'. Available tools: {available}"
            continue

        try:
            result = tool.func(tool_args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            last_tool_result = result
        except Exception as e:
            context += "\n\nSystem:\n" + f"[ToolError] tool '{tool_name}' failed: {e}"
            continue

        context += (
            "\n\nSystem:\n"
            f"TOOL_CALLED: {tool_name}\n"
            f"TOOL_ARGS: {tool_args}\n"
            "TOOL_RESULT_JSON:\n"
            + last_tool_result
            + "\nEND_TOOL_RESULT_JSON\n"
            "请基于 TOOL_RESULT_JSON 继续回答（不得编造）。"
        )

    if last_tool_result:
        raw = last_tool_result if len(last_tool_result) <= 8000 else last_tool_result[:8000] + "\n... (truncated)"
        return (
            "=== TOOL_RESULT_JSON (raw) ===\n"
            + raw
            + "\n=== END_TOOL_RESULT_JSON ===\n\n"
            + "已达到最大工具调用次数，请根据以上工具原始输出自行解读。"
        )
    return "已达到最大工具调用次数，且未获得工具结果。"