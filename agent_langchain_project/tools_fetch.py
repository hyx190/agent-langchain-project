"""
FetchRenderedPage 工具实现（Playwright 优先，失败则回退到 requests）。
返回 dict 格式，包含 source/text/error/attempts 等字段，便于上层记录与调试。
"""

from typing import Dict, Any


def fetch_rendered_page(url: str, timeout: int = 15, max_retries: int = 2) -> Dict[str, Any]:
    """
    尝试使用 Playwright 渲染页面（可执行 JS）。
    - 重试机制：第一次使用 timeout 秒，重试时加倍 timeout。
    - 回退机制：若 Playwright 不可用或多次超时，使用 requests 抓取静态 HTML。
    返回 dict:
      {
        "source": "playwright"/"requests"/"error",
        "text": "...",
        "error": optional_error_string,
        "attempts": n_attempts
      }
    """
    attempts = 0
    last_err = None

    # 1) Playwright 优先（重试）
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout  # type: ignore

        for attempt in range(max_retries + 1):
            attempts += 1
            try:
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True)
                    page = browser.new_page()
                    cur_timeout = timeout * (2 ** attempt)
                    # 使用 domcontentloaded 可减少等待，必要时改为 "networkidle"
                    page.goto(url, timeout=int(cur_timeout * 1000), wait_until="domcontentloaded")
                    text = page.content()
                    try:
                        browser.close()
                    except Exception:
                        pass
                    return {"source": "playwright", "text": text, "error": None, "attempts": attempts}
            except PWTimeout as te:
                last_err = f"playwright timeout (attempt {attempt + 1}): {te}"
                # 重试循环继续
            except Exception as e:
                last_err = f"playwright error (attempt {attempt + 1}): {e}"
                # 对非超时错误也记录并尝试继续/回退
        # 如果所有重试都失败，last_err 会保存最后一次异常信息
    except Exception as e_play:
        last_err = f"playwright import/launch error: {e_play}"

    # 2) 回退到 requests（静态抓取）
    try:
        import requests  # type: ignore
        headers = {"User-Agent": "agent/1.0 (+https://example.com)"}
        r = requests.get(url, timeout=max(timeout * 2, 10), headers=headers)
        return {
            "source": "requests",
            "text": r.text,
            "error": f"playwright_failed: {last_err}",
            "attempts": attempts + 1
        }
    except Exception as e_req:
        return {
            "source": "error",
            "text": "",
            "error": f"playwright_failed: {last_err}; requests_failed: {e_req}",
            "attempts": attempts + 1
        }