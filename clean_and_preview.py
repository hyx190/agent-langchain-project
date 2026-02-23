#!/usr/bin/env python3
"""
尝试性清洗 / 预览脚本（对 clipboard_parsed.csv 或剪贴板表格进行列对齐）
用法:
  python clean_and_preview.py SH600000 SH600029
输出:
  - 对每个指定代码，打印原始 full_row（部分）& 清洗后 key->value 以及字段诊断
注意:
  - 这是 heuristics 式修复，能修复“表头/数据交错（header / unnamed）”多数情况，但不是万无一失。
  - 请人工核验清洗结果，若多数行仍异常，建议重新从同花顺导出为 CSV/Excel 再处理。
"""
import sys, os, re, json
from typing import List
import pandas as pd
import numpy as np

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
    try:
        df = pd.read_clipboard(dtype=str)
        if len(df.columns) == 1:
            col = df.columns[0]
            sample = df[col].dropna().astype(str).iloc[0] if len(df) > 0 else ""
            if "\t" in sample:
                df = df[col].str.split("\t", expand=True)
        return df, "clipboard"
    except Exception as e:
        raise RuntimeError(f"无法读取 clipboard_parsed.csv 也无法从剪贴板解析表格：{e}")

def is_likely_header(s: str):
    if s is None:
        return False
    s = str(s).strip()
    if s == "":
        return False
    # header often contains Chinese chars or spaces, not pure numbers or percent
    if re.search(r'[\u4e00-\u9fff]', s):
        return True
    if any(k in s.lower() for k in ("代码","名称","涨","价","市","pe","pb","ttm","每股","上市")):
        return True
    return False

def parse_number(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s in ("", "--", "-"):
        return None
    # percent handling like "+0.12%" or "-1.2%"
    if s.endswith("%"):
        try:
            return float(s.replace("%",""))
        except:
            return None
    try:
        return float(s)
    except:
        return None

def pairwise_normalize_row(df, row_idx):
    """
    Heuristic: columns often alternate header / value or header / unnamed value.
    We'll iterate columns in steps of 2 and produce key->value mapping:
     - key = column name at i (stripped)
     - candidate values: row[i] and row[i+1] (if exists)
     - choose the one that looks like a value (not header) or matches expected pattern for code/name
    Additionally, try sliding window if pairs don't match well.
    """
    cols = list(df.columns)
    row = df.iloc[row_idx]
    normalized = {}
    n = len(cols)
    i = 0
    while i < n:
        key_raw = str(cols[i]).strip()
        val = None
        # try get right neighbor if exists
        right = cols[i+1] if i+1 < n else None
        left_val = row[cols[i]] if cols[i] in row.index else None
        right_val = row[right] if right is not None and right in row.index else None

        # If right_val looks like a header (string that contains Chinese header words), prefer left_val
        # If left_val looks like header and right_val non-empty, prefer right_val
        if right is not None:
            if (is_likely_header(left_val) and not is_likely_header(right_val)):
                val = right_val
            elif (not is_likely_header(left_val) and (str(left_val).strip() != "" and (str(right_val).strip()=="" or is_likely_header(right_val)))):
                val = left_val
            else:
                # heuristics by content type: if key_raw contains code-like word, look for code pattern in left/right
                if re.search(r'代码|code', key_raw, re.I):
                    if isinstance(left_val,str) and re.search(r'\b(SH|SZ)?\d{6}\b', left_val, re.I):
                        val = left_val
                    elif isinstance(right_val,str) and re.search(r'\b(SH|SZ)?\d{6}\b', right_val, re.I):
                        val = right_val
                    else:
                        val = left_val if str(left_val).strip() else right_val
                else:
                    # default: choose right_val if it's non-empty, else left_val
                    val = right_val if (right_val is not None and str(right_val).strip()!='') else left_val
        else:
            val = left_val

        normalized[key_raw if key_raw!='' else f"col_{i}"] = val if (val is not None and str(val)!='nan') else ""
        i += 2
    # Postprocess: sometimes some headers are in Unnamed columns; also attempt to salvage common numeric fields by searching row
    # Build reverse mapping of potential keys to values by searching entire row for code / price etc if missing
    # If '代码' missing in normalized, search across all columns
    if not any(k for k in normalized.keys() if '代码' in str(k) or k.lower()=='code'):
        # search for first code-like cell in row
        for c in cols:
            v = row[c]
            if isinstance(v,str) and re.search(r'\b(SH|SZ)?\d{6}\b', v, re.I):
                normalized['代码?'] = v
                break
    return normalized

def extract_key_fields(normalized):
    # try to map normalized keys to canonical keys by fuzzy matching on header text
    out = {}
    # build mapping from normalized keys to values
    for k, v in normalized.items():
        k_low = k.lower()
        # match by keywords
        if any(x in k_low for x in ["代码", "code"]):
            out.setdefault("code", v)
        elif any(x in k_low for x in ["名", "名称", "证券名称"]):
            out.setdefault("name", v)
        elif any(x in k_low for x in ["现价", "最新价", "price"]):
            out.setdefault("price", v)
        elif any(x in k_low for x in ["涨幅", "涨跌", "change", "%"]):
            # could be multiple; prefer ones containing '%' or '涨'
            if "chg" not in out:
                out["chg"] = v
        elif any(x in k_low for x in ["ttm", "市盈", "pe"]):
            out.setdefault("pe", v)
        elif any(x in k_low for x in ["市净", "pb"]):
            out.setdefault("pb", v)
        elif any(x in k_low for x in ["每股", "eps"]):
            out.setdefault("eps", v)
        elif any(x in k_low for x in ["上市"]):
            out.setdefault("listing_date", v)
        elif any(x in k_low for x in ["行业", "所属行业", "细分行业"]):
            out.setdefault("industry", v)
        elif any(x in k_low for x in ["总市值", "总股本"]):
            out.setdefault("total_value", v)
    return out

def diag_parsed_fields(parsed):
    diag = {}
    # price
    price = parse_number(parsed.get("price"))
    if price is None:
        diag["price"] = "MISSING_OR_NONNUM"
    elif price <= 0:
        diag["price"] = f"UNUSUAL({price})"
    else:
        diag["price"] = "OK"
    # chg: percent or number
    chg = parsed.get("chg")
    chg_parsed = None
    if chg is not None:
        try:
            s = str(chg).strip()
            if s.endswith("%"):
                chg_parsed = float(s.replace("%",""))
            else:
                chg_parsed = float(s)
        except:
            chg_parsed = None
    if chg_parsed is None:
        diag["chg"] = "MISSING_OR_NONNUM"
    elif abs(chg_parsed) > 20:
        diag["chg"] = f"EXTREME({chg_parsed})"
    else:
        diag["chg"] = "OK"
    # pe
    pe = parse_number(parsed.get("pe"))
    if pe is None:
        diag["pe"] = "MISSING"
    elif abs(pe) > 1000:
        diag["pe"] = f"EXTREME({pe})"
    else:
        diag["pe"] = "OK"
    # pb
    pb = parse_number(parsed.get("pb"))
    if pb is None:
        diag["pb"] = "MISSING"
    elif pb < -10 or pb > 100:
        diag["pb"] = f"EXTREME({pb})"
    else:
        diag["pb"] = "OK"
    return diag

def preview_for_symbols(df, symbols: List[str]):
    cols = list(df.columns)
    results = {}
    for sym in symbols:
        found = []
        # find rows where any cell contains symbol (allow without prefix)
        sym_core = sym.upper().replace("SH","").replace("SZ","")
        for idx, row in df.iterrows():
            combined = " ".join([str(x) for x in row.values if pd.notna(x)])
            if sym_core in combined.upper() or sym.upper() in combined.upper():
                found.append(idx)
        if not found:
            results[sym] = {"error": "not found"}
            continue
        rows_out = []
        for idx in found:
            normalized = pairwise_normalize_row(df, idx)
            extracted = extract_key_fields(normalized)
            diag = diag_parsed_fields(extracted)
            rows_out.append({
                "index": int(idx),
                "original_partial": {k: df.iloc[idx].get(k) for k in list(df.columns)[:12]},
                "normalized_sample": {k: normalized[k] for k in list(normalized.keys())[:20]},
                "extracted": extracted,
                "diag": diag
            })
        results[sym] = rows_out
    return results

def main():
    if len(sys.argv) < 2:
        print("用法: python clean_and_preview.py SH600000 [SH600029 ...]")
        sys.exit(1)
    syms = [s.strip().upper() for s in sys.argv[1:]]
    df, src = try_read_table()
    print(f"读取表格 rows={len(df)}, cols={len(df.columns)}, source={src}")
    df.columns = [str(c).strip() for c in df.columns]
    res = preview_for_symbols(df, syms)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()