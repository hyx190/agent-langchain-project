# tongyi_agent/runner.py (with context propagation for background saves)
import sys
import traceback
from typing import List, Callable, Optional, Any
from .utils import call_with_timeout, EXECUTOR
from .tools import Tool, build_tools
import os
import contextvars
from .context_request_id import new_request_id, get_request_id
import logging
from .memory_utils import auto_save_turn

# compatibility imports for langchain agent creation
try:
    from langchain.agents import create_agent as _create_agent  # type: ignore
    create_agent = _create_agent
except Exception:
    try:
        from langchain.agents.factory import create_agent as _create_agent  # type: ignore
        create_agent = _create_agent
    except Exception:
        create_agent = None

try:
    from langchain.agents import initialize_agent, AgentType  # type: ignore
except Exception:
    initialize_agent = None
    AgentType = None

logger = logging.getLogger("agent.runner")

def _prepare_tools_for_factory(tools: List[Tool]):
    prepared = []
    try:
        from langchain.tools import Tool as LCTool  # type: ignore
        for t in tools:
            try:
                prepared.append(LCTool(name=t.name, func=t.func, description=t.description))
            except Exception:
                prepared.append({"name": t.name, "func": t.func, "description": t.description})
    except Exception:
        for t in tools:
            prepared.append({"name": t.name, "func": t.func, "description": t.description})
    return prepared

def _make_agent_runner(agent_obj: Any) -> Optional[Callable[[str], str]]:
    if agent_obj is None:
        return None
    if hasattr(agent_obj, "run") and callable(getattr(agent_obj, "run")):
        return lambda u: agent_obj.run(u)
    if callable(agent_obj):
        return lambda u: agent_obj(u)
    if hasattr(agent_obj, "agent") and hasattr(agent_obj.agent, "run"):
        return lambda u: agent_obj.agent.run(u)
    return None

def make_system_prompt(persona_text: Optional[str] = None) -> str:
    base = (
        "你是一个助理。回答务实、简洁、并尽可能参考已��的用户偏好。"
        "当用户明确表示“以我的想法去思考”时，请把用户保存到长期记忆的偏好和笔记当作首要参考。"
    )
    if persona_text:
        base += "\n\n" + persona_text
    return base

def run_agent_with_memory(llm, tools, long_memory, call_timeout: int = 90):
    persona_text = ""
    if long_memory is not None:
        try:
            for t, m in zip(long_memory.texts, long_memory.metadatas):
                if isinstance(m, dict) and m.get("type") == "persona":
                    persona_text += f"- ({m.get('name')}) {t}\n"
        except Exception:
            pass
    system_prompt = make_system_prompt(persona_text if persona_text else None)

    agent_obj = None
    agent_runner = None

    if create_agent is not None:
        try:
            prepared_tools = _prepare_tools_for_factory(tools)
            try:
                agent_obj = create_agent(model=llm, tools=prepared_tools, system_prompt=system_prompt, debug=True)
            except Exception:
                try:
                    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or "qwen-plus"
                    agent_obj = create_agent(model=model_name, tools=prepared_tools, system_prompt=system_prompt, debug=True)
                except Exception:
                    agent_obj = None
        except Exception:
            traceback.print_exc(file=sys.stderr)
            agent_obj = None

    if agent_obj is None and initialize_agent is not None:
        try:
            if AgentType is not None and hasattr(AgentType, "ZERO_SHOT_REACT_DESCRIPTION"):
                agent_obj = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            else:
                agent_obj = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            agent_obj = None

    if agent_obj is not None:
        agent_runner = _make_agent_runner(agent_obj)

    print("启动完成。命令：/save_persona 文本, /list_personas, /save 文本, /recall 查询, /persist, /load, exit/quit。")
    short_history = []

    while True:
        try:
            q = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出交互。")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # create request id for this interaction and log
        rid = new_request_id()
        logger.debug("user.input", extra={"request_id": rid, "text_summary": q[:500]})

        if q.startswith("/save_persona "):
            txt = q[len("/save_persona "):].strip()
            print([t.func(txt) for t in tools if t.name == "SavePersona"][0])
            continue
        if q.startswith("/list_personas"):
            print([t.func("") for t in tools if t.name == "ListPersonas"][0])
            continue
        if q.startswith("/save "):
            txt = q[len("/save "):].strip()
            print([t.func(txt) for t in tools if t.name == "SaveMemory"][0])
            continue
        if q.startswith("/recall "):
            qquery = q[len("/recall "):].strip()
            print([t.func(qquery) for t in tools if t.name == "RecallMemory"][0])
            continue
        if q.startswith("/persist"):
            print([t.func("") for t in tools if t.name == "PersistMemory"][0])
            continue
        if q.startswith("/load"):
            print([t.func("") for t in tools if t.name == "LoadMemory"][0])
            continue

        short_history.append(("user", q))
        if len(short_history) > 200:
            short_history = short_history[-200:]

        recall_snippet = ""
        if long_memory is not None:
            try:
                hits = long_memory.search(q, k=3)
                if hits:
                    recall_snippet = "相关长期记忆：\n" + "\n".join([f"(score={h.get('score',0):.4f}) {h.get('text')}" for h in hits]) + "\n\n"
            except Exception:
                pass

        prompt = system_prompt + "\n\n" + "\n".join([f"{r}: {t}" for r, t in short_history[-6:]]) + "\n\n" + recall_snippet + "User: " + q

        if agent_runner is not None:
            try:
                out = call_with_timeout(agent_runner, args=(q,), timeout=call_timeout)
            except Exception as e:
                logger.exception("agent_runner.call.error", extra={"request_id": rid, "error": str(e)})
                print("Agent 调用失败，降级到直接调用模型：", e, file=sys.stderr)
                try:
                    out = call_with_timeout(llm._call, args=(prompt,), timeout=call_timeout)
                except Exception as e2:
                    logger.exception("llm.direct.call.error", extra={"request_id": rid, "error": str(e2)})
                    print("LLM 调用失败：", e2, file=sys.stderr)
                    continue
        else:
            try:
                out = call_with_timeout(llm._call, args=(prompt,), timeout=call_timeout)
            except Exception as e:
                logger.exception("llm.direct.call.error", extra={"request_id": rid, "error": str(e)})
                print("LLM 调用失败：", e, file=sys.stderr)
                continue

        print("\nAssistant> ", out)
        short_history.append(("assistant", out))

        # 非阻塞保存对话 turn 到长期记忆（确保 contextvars 传播）
        try:
            ctx = contextvars.copy_context()
            EXECUTOR.submit(ctx.run, auto_save_turn, q, out, get_request_id(), None, True)
        except Exception:
            logger.exception("auto_save.submit_failed", extra={"request_id": get_request_id()})