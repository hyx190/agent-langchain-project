# tongyi_agent/cli.py (with context propagation for background saves)
import os
import logging
import contextvars
from .utils import setup_logging as utils_setup_logging  # keep original available if needed
from .logging_setup import setup_logging
from .context_request_id import new_request_id, get_request_id
from .llm import DashScopeLLM
from .tools import build_tools, build_langchain_tools
from .memory import init_long_memory, get_long_memory
from .runner import run_agent_with_memory
from .utils import call_with_timeout, EXECUTOR

# dispatcher used when langchain is unavailable
from .tool_dispatcher import run_with_tools
from .memory_utils import auto_save_turn
# 在 tongyi_agent/cli.py 顶部适当位置加入（确保在任何 logger 使用前）
from .cli_logging_patch import register_request_id_filter
register_request_id_filter()
import logging

class RequestIdFilter(logging.Filter):
    """确保每个 LogRecord 都有 request_id 属性，避免格式化失败。"""
    def filter(self, record):
        if not hasattr(record, "request_id"):
            # 使用短横作为缺省值；可改为 None 或空串
            record.request_id = "-"
        return True

def register_request_id_filter():
    root = logging.getLogger()
    root.addFilter(RequestIdFilter())
    # 对现有 handler 也加一遍（有时 root filter 就够）
    for h in root.handlers:
        h.addFilter(RequestIdFilter())
VECTOR_STORE_PATH = os.getenv("AGENT_VECTOR_STORE_PATH", "~/.agent_vector_store")
MODEL_PATH = os.getenv("AGENT_EMBEDDING_MODEL_PATH", "")

logger = logging.getLogger("agent.cli")

def main():
    # initialize structured logging (DEBUG useful for dev; change to INFO in prod)
    setup_logging("INFO", log_file="agent-debug.log")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("未检测到 DASHSCOPE_API_KEY 环境变量。请先设置。")

    try:
        llm = DashScopeLLM(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.2,
            extra_body={},
        )
    except Exception as e:
        logger.exception("无法实例化 DashScopeLLM", extra={"error": str(e)})
        print("无法实例化 DashScopeLLM：", e)
        return

    # init memory backend (Faiss or Json fallback)
    long_memory = init_long_memory(index_path=VECTOR_STORE_PATH, model_path=MODEL_PATH)

    # build tools
    tools = build_tools()

    # try to build langchain tools wrapper
    lc_tools = build_langchain_tools()

    # system prompt with tool call specification
    system_prompt = (
        "你是一个可以调用外部工具的助手。当你需要从网页获取最新事实/渲染后正文或调用外部功能时，"
        "请在单独一行严格使用下列格式之一来指示调用工具（工具名必须与可用工具列表中的 name 完全一致）：\n\n"
        "CALL_TOOL: ToolName <arg string>\n"
        "或\n"
        "CALL_TOOL_JSON: {\"tool\":\"ToolName\",\"args\":\"...\"}\n\n"
        "例如：\nCALL_TOOL: FetchRenderedPage https://example.com\n\n"
        "当工具调用返回结果后，继续生成最终回答并引用工具返回的内容与来源。最多允许多次工具调用（系统会限制次数）。"
    )

    if lc_tools:
        try:
            from langchain.agents import initialize_agent, AgentType
            agent = initialize_agent(lc_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            print("使用 LangChain Agent（工具已注入）。输入 'exit' 结束会话。")
            while True:
                try:
                    user = input("You> ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n退出。")
                    break
                if not user:
                    continue
                if user.lower() in ("exit", "quit"):
                    print("退出。")
                    break

                # create request_id for this interaction
                rid = new_request_id()
                logger.debug("session.start", extra={"request_id": rid, "user_input_summary": user[:200]})

                try:
                    resp = agent.run(system_prompt + "\n\n" + user)
                except Exception as e:
                    logger.exception("agent.run.error", extra={"request_id": rid, "error": str(e)})
                    resp = f"Agent 调用失败：{e}"

                print("Agent> ", resp)
                # 非阻塞保存对话 turn 到长期记忆，确保 contextvars 在 worker 中传播
                try:
                    ctx = contextvars.copy_context()
                    EXECUTOR.submit(ctx.run, auto_save_turn, user, resp, get_request_id(), None, True)
                except Exception:
                    logger.exception("auto_save.submit_failed", extra={"request_id": get_request_id()})
            return
        except Exception as e:
            logger.exception("初始化 LangChain agent 时出错，回退到 dispatcher 模式", extra={"error": str(e)})
            print("初始化 LangChain agent 时出错，回退到 dispatcher 模式。错误：", e)

    # fallback: dispatcher loop
    print("使用内置工具调度器（无需 langchain）。输入 'exit' 结束会话。")

    tools = build_tools()
    save_persona_tool = next((t for t in tools if t.name == "SavePersona"), None)
    recall_memory_tool = next((t for t in tools if t.name == "RecallMemory"), None)

    while True:
        try:
            user = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出。")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("退出。")
            break

        # 每次用户交互生成 request_id
        rid = new_request_id()
        logger.debug("session.start", extra={"request_id": rid, "user_input_summary": user[:200]})

        # 自动检测用户声明身份并保存（保持原有逻辑）
        try:
            import re
            m = re.search(r"(?:我叫|我是)\s*([^\s,。.]{1,30})", user)
            if m and save_persona_tool:
                detected_name = m.group(1).strip()
                save_text = f"persona_name:{detected_name}\nsource:autosave\n原话:{user}"
                try:
                    save_persona_tool.func(save_text)
                    print(f"[系统] 已自动保存 persona: {detected_name}")
                except Exception as e:
                    logger.exception("autosave.persona.error", extra={"request_id": rid, "error": str(e)})
                    print(f"[系统] 自动保存 persona 失败: {e}")
        except Exception:
            logger.exception("autosave.persona.unexpected", extra={"request_id": rid})

        # llm_callable: inject long-memory snippet before calling LLM
        def llm_callable_with_memory(prompt_text: str) -> str:
            try:
                mem = get_long_memory()
            except Exception:
                mem = None

            memory_snippet = ""
            try:
                if mem is not None:
                    hits = mem.search(user, k=5)
                    if hits:
                        lines = ["相关长期记忆（供参考）："]
                        for h in hits:
                            score = h.get("score", 0.0)
                            meta = h.get("meta", {})
                            text = h.get("text", "")[:300]
                            lines.append(f"- score={score:.4f}, meta={meta}, text={text}")
                        memory_snippet = "\n".join(lines) + "\n\n"
            except Exception:
                memory_snippet = ""

            full_prompt = memory_snippet + prompt_text
            try:
                logger.debug("llm.call.start", extra={"request_id": rid, "prompt_summary": full_prompt[:500]})
                return llm(full_prompt)
            except Exception as e:
                logger.exception("llm.call.error", extra={"request_id": rid, "error": str(e)})
                return f"LLM 调用异常：{e}"

        result = run_with_tools(llm_callable_with_memory, user_prompt=user, tools=tools, system_prompt=system_prompt, max_tool_calls=3)
        logger.debug("session.end", extra={"request_id": rid, "result_summary": str(result)[:500]})
        print("Agent> ", result)

        # 非阻塞保存对话 turn 到长期记忆（确保 contextvars 传播）
        try:
            ctx = contextvars.copy_context()
            EXECUTOR.submit(ctx.run, auto_save_turn, user, result, get_request_id(), None, True)
        except Exception:
            logger.exception("auto_save.submit_failed", extra={"request_id": get_request_id()})

if __name__ == "__main__":
    main()