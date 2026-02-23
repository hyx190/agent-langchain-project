#!/usr/bin/env python3
"""
诊断脚本���针对给定股票代码（如 SH600000）从 clipboard_parsed.csv 或剪贴板中查找行并做字段级校验。
用法:
  python tongyi_agent/diagnose_symbols.py SH600000 SH600029
输出:
  - 每只股票的原始行（字段原样）
  - 关键字段解析：code, name, price, chg_pct, PE_TTM, PB, EPS, listing_date
  - 对每个字段的诊断（missing / non-numeric / extreme / ok）
"""
import sys
import re
import json
from typing import List
import pandas as pd
import numpy as np
import os

KEY_FIELD_CANDIDATES = {
    "code": ["代码", "证券代码", "code"],
    "name": ["名称", "证券名称", "name"],
    "price": ["现价", "最新价", "price"],
    "chg": ["涨幅", "change", "chg", "涨跌幅"],
    "pe": ["TTM市盈率", "TTM", "市盈率", "pe", "PE"],
    "pb": ["市净率", "PB"],
    "eps": ["每股盈利", "每股收益", "EPS"],
    "listing_date": ["上市日期", "上市时间", "上市"]
}

def try_read_table():
    # 1) try clipboard_parsed.csv
    f = "clipboard_parsed.csv"
    if os.path.exists(f) and os.path.getsize(f) > 0:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
            return df, f
        except Exception:
            try:
                df = pd.read_csv(f, encoding="gb18030", dtype=str)
                return df, f
            except Exception:
                pass
    # 2) try pandas.read_clipboard()
    try:
        df = pd.read_clipboard(dtype=str)
        # if it's single-column and contains '\t', split
        if len(df.columns) == 1:
            col = df.columns[0]
            sample = df[col].dropna().astype(str).iloc[0] if len(df) > 0 else ""
            if "\t" in sample:
                df = df[col].str.split("\t", expand=True)
        return df, "clipboard"
    except Exception as e:
        print("无法读取 clipboard_parsed.csv，也无法从剪贴板解析表格。错误：", e)
        return None, None

def find_best_col(df: pd.DataFrame, candidates: List[str]):
    cols = list(df.columns)
    lc = [c.strip() for c in cols]
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand.lower() == str(c).strip().lower():
                return cols[i]
    # fuzzy: contains substring
    for cand in candidates:
        for i, c in enumerate(cols):
            if cand.lower() in str(c).strip().lower():
                return cols[i]
    return None

def parse_number(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace(",", "").replace("%", "")
    if s == "" or s in ("--","-"):
        return None
    # percent?
    if re.match(r'^[\+\-]?\d+(\.\d+)?%$', str(x)):
        try:
            return float(s)/100.0
        except:
            return None
    # handle scientific or weird
    try:
        return float(s)
    except:
        return None

def diag_value(field, val):
    if val is None:
        return "MISSING"
    if isinstance(val, (int, float, np.floating, np.integer)):
        v = float(val)
    else:
        try:
            v = float(val)
        except:
            return "NON_NUMERIC"
    # ranges:
    if field == "price":
        if v <= 0:
            return f"UNUSUAL({v})"
        return "OK"
    if field == "chg":
        # expecting fraction e.g. 0.01 for +1%
        if v < -5 or v > 5:
            return f"EXTREME({v})"
        return "OK"
    if field == "pe":
        if abs(v) > 1000:
            return f"EXTREME({v})"
        return "OK"
    if field == "pb":
        if v < -10 or v > 100:
            return f"EXTREME({v})"
        return "OK"
    if field == "eps":
        if abs(v) > 1e6:
            return f"EXTREME({v})"
        return "OK"
    return "OK"

def extract_and_diag(df, symbol):
    # find rows matching symbol (allow variants)
    sym_variants = {symbol, symbol.replace("SH","").replace("SZ",""), symbol.replace(".SS",".SH").replace(".SZ",".SZ")}
    # match any cell in row contains the symbol string
    symbol_norm = symbol.upper()
    matches = []
    for idx, row in df.iterrows():
        row_text = " ".join([str(x) for x in row.values if pd.notna(x)])
        if symbol_norm in row_text.upper() or re.search(r'\b' + re.escape(symbol_norm.replace("SH","").replace("SZ","")) + r'\b', row_text):
            matches.append((idx, row))
    results = []
    for idx, row in matches:
        rowd = {str(c).strip(): (row[c] if pd.notna(row[c]) else "") for c in df.columns}
        # locate columns
        found = {}
        for key, cands in KEY_FIELD_CANDIDATES.items():
            col = find_best_col(df, cands)
            if col:
                found[key] = row.get(col, "")
            else:
                # try heuristics: code in any col
                found[key] = None
                for c in df.columns:
                    v = str(row[c]) if pd.notna(row[c]) else ""
                    if key == "code" and symbol_norm in v.upper():
                        found[key] = v
                        break
        # additional attempts: if not found, try scanning row by header index positions
        # parse numeric fields
        parsed = {}
        parsed_diag = {}
        # price
        price_raw = found.get("price") if found.get("price") is not None else None
        parsed["price"] = parse_number(price_raw) if price_raw is not None else None
        parsed_diag["price"] = diag_value("price", parsed["price"])
        # change percent
        chg_raw = found.get("chg")
        parsed["chg"] = None
        if isinstance(chg_raw, str) and "%" in chg_raw:
            parsed["chg"] = parse_number(chg_raw)
        else:
            parsed["chg"] = parse_number(chg_raw)
        parsed_diag["chg"] = diag_value("chg", parsed["chg"])
        # PE
        pe_raw = found.get("pe")
        parsed["pe"] = parse_number(pe_raw)
        parsed_diag["pe"] = diag_value("pe", parsed["pe"])
        # PB
        pb_raw = found.get("pb")
        parsed["pb"] = parse_number(pb_raw)
        parsed_diag["pb"] = diag_value("pb", parsed["pb"])
        # EPS
        eps_raw = found.get("eps")
        parsed["eps"] = parse_number(eps_raw)
        parsed_diag["eps"] = diag_value("eps", parsed["eps"])
        # listing date - try to parse as int YYYYMMDD or string
        ld_raw = found.get("listing_date")
        ld_flag = "MISSING"
        if ld_raw is None or str(ld_raw).strip() == "":
            ld_flag = "MISSING"
        else:
            s = str(ld_raw).strip()
            if re.match(r'^\d{8}$', s):
                ld_flag = "OK"
            elif re.match(r'^\d{4}\.\d+$', s) or re.match(r'^\d{4}-\d{2}-\d{2}$', s):
                ld_flag = "OK"
            else:
                ld_flag = "UNUSUAL"
        parsed_diag["listing_date"] = ld_flag

        results.append({
            "index": int(idx),
            "row_snippet": {k: rowd[k] for k in list(rowd.keys())[:8]},
            "full_row": rowd,
            "found_fields_raw": found,
            "parsed": parsed,
            "diag": parsed_diag
        })
    return results

def main():
    if len(sys.argv) < 2:
        print("用法: python diagnose_symbols.py SH600000 SH600029")
        sys.exit(1)
    symbols = [s.strip().upper() for s in sys.argv[1:]]
    df, source = try_read_table()
    if df is None:
        print("无法读取表格，请先在同花顺中复制表格或确保 clipboard_parsed.csv 存在。")
        sys.exit(2)
    print(f"读取表格: rows={len(df)}, cols={len(df.columns)}, source={source}")
    # normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]
    out = {}
    for sym in symbols:
        res = extract_and_diag(df, sym)
        out[sym] = res
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()