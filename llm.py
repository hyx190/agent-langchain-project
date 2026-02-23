# tongyi_agent/llm.py
from __future__ import annotations
import os
import asyncio
from typing import Optional, List, Mapping, Any

# Prefer the OpenAI-compatible client (used here as a DashScope-compatible wrapper)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

class DashScopeLLM:
    """
    A simple wrapper (not inheriting from langchain/pydantic LLM base)
    that exposes the usual methods/langchain expects: _call, _acall, __call__,
    model/model_name, temperature, _identifying_params, _llm_type.

    Uses object.__setattr__ in __init__ to avoid triggering any overridden
    __setattr__ (e.g. from pydantic) which may expect pydantic internals.
    """
    # class defaults
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    temperature: float = 0.7
    timeout: int = 30
    extra_body: Optional[dict] = None
    client: Any = None

    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, temperature: Optional[float] = None,
                 timeout: Optional[int] = None, extra_body: Optional[dict] = None):
        # Resolve API key
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或通过 api_key 参数传入。")

        if OpenAI is None:
            raise RuntimeError("openai 库未安装或不可用，请 pip install openai")

        # Assign attributes using object.__setattr__ to avoid pydantic __setattr__ interception
        if base_url:
            object.__setattr__(self, "base_url", base_url)
        else:
            object.__setattr__(self, "base_url", type(self).base_url)

        if model:
            object.__setattr__(self, "model", model)
        else:
            object.__setattr__(self, "model", type(self).model)

        if temperature is not None:
            object.__setattr__(self, "temperature", float(temperature))
        else:
            object.__setattr__(self, "temperature", float(type(self).temperature))

        if timeout is not None:
            object.__setattr__(self, "timeout", int(timeout))
        else:
            object.__setattr__(self, "timeout", int(type(self).timeout))

        object.__setattr__(self, "api_key", key)
        object.__setattr__(self, "extra_body", extra_body or {})
        # create OpenAI-compatible client (DashScope 提供兼容 API)
        try:
            client = OpenAI(api_key=key, base_url=getattr(self, "base_url"))
            object.__setattr__(self, "client", client)
        except Exception as e:
            raise RuntimeError(f"无法创建 OpenAI client: {e}")

    @property
    def _llm_type(self) -> str:
        return "dashscope-tongyi"

    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": getattr(self, "model"), "temperature": getattr(self, "temperature")}

    def __call__(self, prompt: str, **kwargs) -> str:
        # Some callers may call the LLM object directly
        return self._call(prompt, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Synchronous call wrapper: build a chat-style request and parse defensively.
        """
        messages = [{"role": "user", "content": prompt}]
        body = {
            "model": getattr(self, "model"),
            "messages": messages,
            "temperature": float(getattr(self, "temperature")),
            **(getattr(self, "extra_body") or {}),
        }
        # Allow certain overrides from kwargs defensively
        for k in ("max_tokens", "temperature", "top_p", "n"):
            if k in kwargs:
                body[k] = kwargs[k]

        try:
            resp = getattr(self, "client").chat.completions.create(**body)
        except Exception as e:
            raise RuntimeError(f"模型调用失败: {e}")

        # Defensive parsing of response to extract text/content
        content = None
        try:
            choice0 = None
            try:
                choice0 = resp.choices[0]
            except Exception:
                if isinstance(resp, dict) and "choices" in resp and isinstance(resp["choices"], (list, tuple)):
                    choice0 = resp["choices"][0]
            if choice0 is None:
                raise RuntimeError("响应不包含 choices 字段")

            msg = getattr(choice0, "message", None) or (choice0.get("message") if isinstance(choice0, dict) else None)
            if isinstance(msg, dict):
                content = msg.get("content")
            else:
                content = getattr(msg, "content", None)

            if content is None:
                content = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)
        except Exception as e:
            raise RuntimeError(f"无法解析模型返回内容: {e}\n完整响应: {resp}")

        if content is None:
            raise RuntimeError(f"无法从响应中提取文本内容。完整响应: {resp}")

        text = content
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Non-blocking wrapper using asyncio.to_thread
        return await asyncio.to_thread(self._call, prompt, stop)

    @property
    def model_name(self) -> str:
        return getattr(self, "model")
