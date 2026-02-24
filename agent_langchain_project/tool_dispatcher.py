# 调度器：run_with_tools 的增强版
from typing import List, Tuple, Optional
import re
import json
import time
import logging
from datetime import datetime
from .context_request_id import get_request_id

# 引入 persona 注入（若缺失则退化为空函数）
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

_TOOL_CALL_RE = re.compile(r"CALL_TOOL:\s*([A-Za-z0-9_]+)(?:\s+(.*))?$", re.IGNORECASE)
_TOOL_CALL_JSON_RE = re.compile(r"CALL_TOOL_JSON:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)

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

# 简单判断用户在询问“今年/当前年份/当前时间”类问题
_TIME_QUERY_RE = re.compile(r"(今年是哪年|今年是哪一年|现在是(哪年|什么年)|当前年份|现在是哪一年|现在是什么年)", re.I)

def _answer_time_shortcircuit(user_prompt: str):
    """
    若匹配到时间/年份询问，优先用本机系统时间返回。
    可扩展为调用 worldtimeapi 获取更精确的 UTC/时区时间。
    返回 None 表示不命中短路规则。
    """
    if _TIME_QUERY_RE.search(user_prompt or ""):
        now = datetime.now()
        year = now.year
        # 返回一个简短答复（保持中文）
        return f"今年是 {year} 年。"
    return None

def run_with_tools(llm_callable, user_prompt: str, tools: List[Tool],
                   system_prompt: Optional[str] = None,
                   max_tool_calls: int = 3,
                   call_timeout_sec: Optional[int] = None) -> str:
    """
    主循环（增强）：
      - 在调用 LLM 之前对年份/时间问题短路（直接返回系统时间）。
      - 在构造 context 后记录 context preview 到日志，便于调试 persona 注入顺序。
      - 其余行为与原有工具调度一致。
    """
    name2tool = {t.name: t for t in tools}

    # 1) 时间类短路：立即返回系统时间，避免模型使用旧知识
    time_answer = _answer_time_shortcircuit(user_prompt)
    if time_answer is not None:
        logger.info("shortcircuit.time_query", extra={"request_id": get_request_id()})
        return time_answer

    # 2) 构造初始上下文，注入 persona 到 system prompt
    system_with_persona = inject_persona_to_system(system_prompt)
    context_parts = []
    if system_with_persona:
        context_parts.append(system_with_persona.strip())
    context_parts.append(user_prompt.strip())
    context = "\n\n".join(context_parts)

    # 记录 context preview（前 1200 字）以便验证 persona 注入是否生效
    try:
        logger.debug("context_preview", extra={
            "request_id": get_request_id(),
            "context_preview": context[:1200]
        })
    except Exception:
        # 不要让日志失败影响流程
        pass

    for call_idx in range(max_tool_calls + 1):
        try:
            response = llm_callable(context)
            logger.debug("llm.response", extra={"request_id": get_request_id(), "response_summary": str(response)[:1000]})
        except Exception as e:
            logger.exception("llm.call.error", extra={"request_id": get_request_id(), "error": str(e)})
            return f"LLM 调用失败：{e}"

        tc = _parse_tool_call(response)
        if not tc:
            return response

        tool_name, tool_args = tc
        tool = name2tool.get(tool_name)
        if not tool:
            logger.warning("tool.not_found", extra={"request_id": get_request_id(), "tool": tool_name})
            context += f"\n\n[ToolError] 未找到工具: {tool_name}\n模型原文:\n{response}\n请不要再尝试调用不存在的工具。"
            continue

        start = time.time()
        try:
            logger.info("tool.invoke.start", extra={
                "request_id": get_request_id(),
                "tool": tool_name,
                "tool_args": str(tool_args)[:1000]
            })
            tool_result = tool.func(tool_args)
            elapsed = time.time() - start
            logger.info("tool.invoke.success", extra={
                "request_id": get_request_id(),
                "tool": tool_name,
                "elapsed_s": elapsed,
                "result_summary": str(tool_result)[:1000]
            })
        except Exception as e:
            elapsed = time.time() - start
            logger.exception("tool.invoke.error", extra={
                "request_id": get_request_id(),
                "tool": tool_name,
                "elapsed_s": elapsed,
                "error": str(e)
            })
            tool_result = f"工具执行异常：{e}"

        observation = (
            f"\n\n[ToolInvocation]\nTool: {tool_name}\nArgs: {tool_args}\nResult (truncated):\n{tool_result}\n"
            f"(执行耗时: {elapsed:.2f}s)\n"
        )
        context += "\n\n" + response.strip() + "\n" + observation
        context += "\n请基于上面的工具结果继续并生成最终回答（如果需要可再次发出 CALL_TOOL: 指令）。"

    try:
        final = llm_callable(context + "\n\n已达到最大工具调用次数，请基于现有信息给出最终回答。")
    except Exception as e:
        logger.exception("llm.call.final.error", extra={"request_id": get_request_id(), "error": str(e)})
        return f"LLM 调用失败（终结轮）：{e}"
    return final