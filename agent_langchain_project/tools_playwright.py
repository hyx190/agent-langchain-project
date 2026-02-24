# tongyi_agent/tools_playwright.py
# Playwright 抓取工具（同步接口）
# 依赖: playwright
# 安装: pip install playwright
# 下载浏览器: python -m playwright install

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Optional
import html
import re

def _text_from_html(html_content: str, max_chars: int = 3000) -> str:
    # 简单从 HTML 提取可读文本：移除 script/style，去掉标签并解码 HTML 实体，然后截断
    text = re.sub(r'(?is)<(script|style).*?>.*?</\1>', '', html_content)
    text = re.sub(r'(?s)<[^>]+>', ' ', text)  # strip tags
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]

def fetch_rendered_summary(url: str,
                           wait_for_selector: Optional[str] = None,
                           timeout: int = 15000,
                           headless: bool = True,
                           max_chars: int = 3000) -> str:
    """
    使用 Playwright 抓取渲染后页面并返回文本摘要（去掉脚本/样式并截断）。
    - url: 页面地址
    - wait_for_selector: 可选 CSS selector（若指定，将等待该元素出现）
    - timeout: 毫秒超时（默认 15000）
    - headless: 是否无头运行
    - max_chars: 返回文本最大长度
    返回:
      - 成功: 页面正文纯文本摘要
      - 失败: 以 "抓取失败：" 开头的错误信息
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            page.goto(url, timeout=timeout)
            if wait_for_selector:
                try:
                    page.wait_for_selector(wait_for_selector, timeout=timeout)
                except PlaywrightTimeoutError:
                    # 等待超时则继续抓取当前页面内容
                    pass

            # 优先使用 inner_text 获取可读文本
            summary = ""
            try:
                body_text = page.inner_text("body")
                summary = body_text.strip()
            except Exception:
                content = page.content()
                summary = _text_from_html(content, max_chars=max_chars)

            browser.close()
            summary = summary[:max_chars]
            if not summary:
                return "未能提取到页面文本内容。"
            return summary
    except Exception as e:
        return f"抓取失败：{e}"
