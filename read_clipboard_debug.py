#!/usr/bin/env python3
"""
读取并诊断剪贴板内容，尝试多种方式解析成表格。
用法:
  python tongyi_agent/read_clipboard_debug.py
输出:
  - 打印原始剪贴板前 2000 字符的 repr 预览
  - 尝试多种解析方法并打印每次尝试的结果摘要
  - 若解析成功，保存为 clipboard_parsed.csv 并打印保存路径
"""
import sys
import io
import re
import csv
import traceback
from typing import Optional
try:
    import pandas as pd
except Exception:
    print("请先在虚拟环境中安装 pandas: pip install pandas")
    sys.exit(1)

def get_clipboard_text() -> Optional[str]:
    # First try pyperclip (easy), fallback to win32clipboard
    try:
        import pyperclip
        txt = pyperclip.paste()
        if txt:
            return txt
    except Exception:
        pass
    # fallback to win32clipboard (Windows)
    try:
        import win32clipboard
        win32clipboard.OpenClipboard()
        try:
            data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()
        return data
    except Exception:
        pass
    return None

def try_parse_with_sep(text: str, sep: str, engine: str = "python"):
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(text), sep=sep, engine=engine, dtype=str)
        return df
    except Exception as e:
        # try with python engine if not already
        if engine != "python":
            try:
                df = pd.read_csv(StringIO(text), sep=sep, engine="python", dtype=str)
                return df
            except Exception:
                return None
        return None

def try_sniffer(text: str):
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return None

def try_read_html(text: str):
    try:
        dfs = pd.read_html(text)
        if dfs:
            return dfs[0]
    except Exception:
        return None
    return None

def parse_by_fallbacks(text: str):
    # 1) if contains literal '\t' sequences (escaped), unescape
    if "\\t" in text:
        t2 = text.replace("\\t", "\t")
        df = try_parse_with_sep(t2, sep="\t")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return "unescaped_backslash_tabs", df

    # 2) If contains real tabs
    if "\t" in text:
        df = try_parse_with_sep(text, sep="\t")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return "tabs", df

    # 3) try csv.Sniffer
    delim = try_sniffer(text)
    if delim:
        df = try_parse_with_sep(text, sep=delim)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return f"sniffer_delim:{delim}", df

    # 4) If many lines but columns seem separated by multiple spaces, try regex sep (2+ spaces)
    if re.search(r" {2,}", text):
        df = try_parse_with_sep(text, sep=r"\s{2,}")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return "multi_space", df

    # 5) try comma
    if "," in text:
        df = try_parse_with_sep(text, sep=",")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return "comma", df

    # 6) try read_html (if clipboard contains HTML table)
    df = try_read_html(text)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return "html_table", df

    # 7) last resort: try splitting lines and splitting by tab or multiple spaces manually
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return "no_lines", None
    # try splitting first line by any of these
    for candidate in ("\t", "|", ","):
        cols = lines[0].split(candidate)
        if len(cols) > 1:
            rows = [l.split(candidate) for l in lines]
            try:
                df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
                return f"manual_split_by_{candidate}", df
            except Exception:
                pass
    # try splitting by multiple spaces
    try:
        rows = [re.split(r"\s{2,}", l.strip()) for l in lines]
        if all(len(r) == len(rows[0]) for r in rows[:5]):
            df = pd.DataFrame(rows[1:], columns=[c.strip() for c in rows[0]])
            return "manual_split_multi_space", df
    except Exception:
        pass

    return None, None

def main():
    text = get_clipboard_text()
    if not text:
        print("剪贴板为空或无法读取剪贴板文本。请先在同花顺中选中表格并复制 (Ctrl+C)，然后再运行本脚本。")
        return
    print("==== 剪贴板原始预览（repr，最多2000字符） ====")
    print(repr(text[:2000]))
    print("==== 原始文本前 5 行（用于快速人工观察） ====")
    for i, line in enumerate(text.splitlines()[:5]):
        print(f"{i+1}: {line}")
    print("==== 开始尝试解析 ====")
    method, df = parse_by_fallbacks(text)
    if method and isinstance(df, pd.DataFrame):
        print(f"[OK] 解析成功，方法: {method}")
        print("DataFrame shape:", df.shape)
        # show columns and head (10 rows)
        print("Columns:", list(df.columns))
        print("Preview (first 10 rows):")
        print(df.head(10).to_string(index=False))
        out = "clipboard_parsed.csv"
        try:
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print("已保存为:", out)
        except Exception as e:
            print("保存 CSV 失败:", e)
    else:
        print("[FAIL] 所有自动解析尝试失败。建议：")
        print("  1) 在同花顺中使用“复制为表格”或右键导出为 Excel/CSV；")
        print("  2) 将表格粘到 Excel 中，检查列是否正常，然后另存为 CSV UTF-8，再用 inspect 脚本读取；")
        print("  3) 把本脚本打印的原始预览（上方）贴给我，我会基于该原始文本编写定制解析器。")
        if method:
            print("最后一次尝试返回:", method)
        else:
            print("未检测到可用解析方法。")

if __name__ == "__main__":
    main()