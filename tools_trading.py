"""
读取同花顺导出/剪贴板数据 + 简单分析与模拟下单（MVP，仅读取/模拟）
工具（将由 build_tools 注册）:
 - ReadTradingCSV(path_or_clipboard)
 - AnalyzePortfolio(path_or_snapshot_json)
 - SuggestAction(path_or_snapshot_json)
 - SimulateTrade(payload_json)
备注：
 - 若使用 yfinance 作为行情回退，代码会把同花顺的 'SH600000'/'SZ000001' 转为 '600000.SS'/'000001.SZ'。
 - 真实下单需额外接券商 API（不在此文件中实现）。
"""
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
import datetime
import json
import logging

logger = logging.getLogger("agent.tools_trading")

DEFAULT_PARAMS = {
    "ma_short": 5,
    "ma_long": 20,
    "atr_period": 14,
    "rsi_period": 14,
    "stop_loss_pct": 0.10,
    "take_profit_pct": 0.05,
    "slippage": 0.0005,
    "fee_rate": 0.0003,
    "max_position_pct": 0.2
}

# ---------------- util: symbol mapping ----------------
def normalize_symbol_for_yfinance(sym: str) -> str:
    """
    同花顺可能给出 'SH600000' 或 'SZ000001' 或 '600000'。
    yfinance expects '600000.SS' for Shanghai and '000001.SZ' for Shenzhen.
    """
    if not isinstance(sym, str):
        return sym
    s = sym.strip()
    if s == "":
        return s
    if s.upper().startswith("SH"):
        core = s[2:]
        return core + ".SS"
    if s.upper().startswith("SZ"):
        core = s[2:]
        return core + ".SZ"
    if s.endswith(".SS") or s.endswith(".SZ") or "." in s:
        return s
    if len(s) == 6 and s[0] == "6":
        return s + ".SS"
    if len(s) == 6:
        return s + ".SZ"
    return s

# ---------------- Data ingest ----------------
def _read_any(path_or_clipboard: str) -> Dict[str, Any]:
    """读取 CSV/Excel 或 clipboard，返回 snapshot dict"""
    # clipboard
    if isinstance(path_or_clipboard, str) and path_or_clipboard.lower() == "clipboard":
        try:
            df = pd.read_clipboard()
            src = "clipboard"
        except Exception as e:
            return {"error": f"clipboard_read_error: {e}"}
    else:
        if not os.path.exists(path_or_clipboard):
            return {"error": f"file_not_found: {path_or_clipboard}"}
        ext = os.path.splitext(path_or_clipboard)[1].lower()
        try:
            if ext in (".xls", ".xlsx", ".xlsm", ".xlsb"):
                df = pd.read_excel(path_or_clipboard, dtype=str, engine="openpyxl")
                src = "excel"
            else:
                df = None
                for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"):
                    try:
                        df = pd.read_csv(path_or_clipboard, encoding=enc, dtype=str)
                        src = "csv"
                        break
                    except Exception:
                        df = None
                if df is None:
                    return {"error": "failed_read_csv_with_common_encodings"}
        except Exception as e:
            return {"error": f"read_file_error: {e}"}

    # normalize column names mapping (cover common ths names)
    col_map = {
        '代码': 'code', '证券代码': 'code', '代码 ': 'code', 'code': 'code',
        '名称': 'name', '证券名称': 'name', 'name': 'name',
        '方向': 'side', '买卖方向': 'side', 'side': 'side',
        '数量': 'qty', '成交数量': 'qty', 'qty': 'qty',
        '成交价': 'price', '价格': 'price', 'price': 'price',
        '现价': 'price', '最新价': 'price',
        '成交时间': 'time', '时间': 'time', 'timestamp': 'time'
    }
    df_cols = list(df.columns)
    rename = {}
    for c in df_cols:
        c_norm = str(c).strip()
        for k, v in col_map.items():
            if c_norm.lower() == k.strip().lower():
                rename[c] = v
                break
    df = df.rename(columns=rename)

    # If single-column with tabs, split it
    if len(df.columns) == 1:
        single_col = df.columns[0]
        try:
            sample = df[single_col].dropna().astype(str).iloc[0]
        except Exception:
            sample = ""
        if "\t" in sample:
            df = df[single_col].str.split("\t", expand=True)

    # coerce qty/price to numeric when present
    for col in ['qty', 'price']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # try to detect code column heuristically
    code_col = None
    for c in df.columns:
        cc = str(c).strip()
        if cc.lower() in ("code", "代码", "证券代码"):
            code_col = c
            break
    if not code_col:
        for c in df.columns:
            sample_vals = []
            try:
                sample_vals = df[c].astype(str).dropna().head(10).tolist()
            except Exception:
                sample_vals = []
            if any(isinstance(v, str) and (v.upper().startswith("SH") or v.upper().startswith("SZ")) for v in sample_vals):
                code_col = c
                break
    if code_col and 'code' not in df.columns:
        df = df.rename(columns={code_col: 'code'})

    # build holdings by aggregating trades or snapshot if possible
    holdings = {}
    if 'code' in df.columns and 'qty' in df.columns:
        if 'side' in df.columns:
            def signed_qty(r):
                s = str(r.get('side','')).lower()
                try:
                    q = float(r.get('qty') or 0)
                except Exception:
                    q = 0.0
                if any(x in s for x in ('卖', 'sell', 's')):
                    return -abs(q)
                return abs(q)
            df['qty_signed'] = df.apply(lambda r: signed_qty(r), axis=1)
        else:
            df['qty_signed'] = pd.to_numeric(df['qty'], errors='coerce').fillna(0)

        grouped = df.groupby('code')
        for sym, g in grouped:
            total_qty = int(g['qty_signed'].sum()) if not g['qty_signed'].isnull().all() else 0
            qty_abs = g['qty_signed'].abs().sum()
            if qty_abs > 0 and 'price' in g.columns:
                avg_price = (g['qty_signed'].abs() * g['price'].fillna(0)).sum() / qty_abs
            else:
                avg_price = 0.0
            holdings[str(sym).strip()] = {"qty": int(total_qty), "avg_price": float(avg_price) if avg_price else 0.0}

    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": src,
        "path": path_or_clipboard if src != "clipboard" else None,
        "trades_count": len(df),
        "holdings": holdings,
        "raw": df.to_dict(orient="records")
    }
    return snapshot

# ---------------- Market data ----------------
def fetch_market_price(symbol: str) -> Dict[str, Any]:
    """
    使用 yfinance 作为回退行情来源（示例）。
    symbol 可以是 'SH600000'，会自动映射为 '600000.SS'。
    """
    try:
        import yfinance as yf
    except Exception as e:
        return {"error": f"yfinance_missing: {e}"}
    try:
        yf_sym = normalize_symbol_for_yfinance(symbol)
        t = yf.Ticker(yf_sym)
        hist = t.history(period="1d", interval="1m")
        if hist is None or hist.empty:
            info = {}
            try:
                info = t.info if hasattr(t, "info") else {}
            except Exception:
                info = {}
            last = info.get("regularMarketPrice")
            if last:
                return {"symbol": symbol, "price": float(last)}
            return {"error": "no_data"}
        last = hist['Close'].iloc[-1]
        return {"symbol": symbol, "price": float(last)}
    except Exception as e:
        return {"error": str(e)}

# ---------------- Analysis ----------------
def analyze_portfolio(snapshot: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or DEFAULT_PARAMS
    holdings = snapshot.get("holdings", {}) or {}
    total_value = 0.0
    details = {}
    for sym, info in holdings.items():
        qty = int(info.get("qty", 0))
        avg = float(info.get("avg_price", 0.0) or 0.0)
        if qty == 0:
            continue
        mp = fetch_market_price(sym)
        cur_price = mp.get("price", avg) if isinstance(mp, dict) else avg
        value = qty * cur_price
        pnl = (cur_price - avg) * qty
        details[sym] = {"qty": qty, "avg_price": avg, "cur_price": float(cur_price), "value": float(value), "pnl": float(pnl)}
        total_value += value
    return {"timestamp": datetime.datetime.now().isoformat(), "total_value": float(total_value), "details": details}

# ---------------- Strategy ----------------
def suggest_actions_from_summary(summary: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    params = params or DEFAULT_PARAMS
    out = []
    total = summary.get("total_value", 0) or 1.0
    for sym, d in summary.get("details", {}).items():
        qty = d.get("qty", 0)
        avg = d.get("avg_price", 1.0)
        cur = d.get("cur_price", avg)
        value = d.get("value", 0)
        pnl = d.get("pnl", 0)
        pnl_pct = pnl / (abs(avg) * qty) if (abs(avg) * qty) != 0 else 0
        pct_of_portfolio = value / total if total > 0 else 0
        if pnl_pct <= -params["stop_loss_pct"] and pct_of_portfolio > 0.05:
            out.append({"symbol": sym, "action": "reduce", "reason": f"浮亏{pnl_pct:.2%}超过止损阈值", "suggest_qty": int(abs(qty) * 0.3)})
        elif pnl_pct >= params["take_profit_pct"]:
            out.append({"symbol": sym, "action": "take_profit_partial", "reason": f"浮盈{pnl_pct:.2%}达到止盈阈值", "suggest_qty": int(abs(qty) * 0.2)})
        elif pct_of_portfolio > params.get("max_position_pct", 0.2):
            out.append({"symbol": sym, "action": "reduce", "reason": f"仓位占比过高({pct_of_portfolio:.2%})", "suggest_qty": int(abs(qty) * 0.2)})
    return out

# ---------------- Simulation ----------------
def simulate_trade(action: Dict[str, Any], snapshot: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = params or DEFAULT_PARAMS
    sym = action.get("symbol")
    act = action.get("action") or action.get("side") or ""
    qty = int(action.get("suggest_qty", action.get("qty", 0)) or 0)
    cur_qty = snapshot.get("holdings", {}).get(sym, {}).get("qty", 0)
    avg_price = snapshot.get("holdings", {}).get(sym, {}).get("avg_price", 0.0)
    mp = fetch_market_price(sym)
    if "price" not in mp:
        return {"error": "price_unavailable", "details": mp}
    price = mp["price"]
    slippage = params.get("slippage", 0.0005)
    fee_rate = params.get("fee_rate", 0.0003)
    side = "SELL" if str(act).lower().startswith("r") or str(act).lower().startswith("take") or str(act).lower().startswith("s") else "BUY"
    exec_price = price * (1 - slippage if side == "SELL" else 1 + slippage)
    cost = exec_price * qty
    fee = abs(cost) * fee_rate
    if side == "SELL":
        new_qty = cur_qty - qty
        new_avg = avg_price
    else:
        new_qty = cur_qty + qty
        if cur_qty == 0:
            new_avg = exec_price
        else:
            new_avg = (cur_qty * avg_price + qty * exec_price) / (cur_qty + qty)
    new_snapshot = dict(snapshot)
    new_holdings = dict(snapshot.get("holdings", {}))
    new_holdings[sym] = {"qty": int(new_qty), "avg_price": float(new_avg)}
    new_snapshot["holdings"] = new_holdings
    return {
        "symbol": sym, "side": side, "qty": int(qty), "exec_price": float(exec_price),
        "fee": float(fee), "timestamp": datetime.datetime.now().isoformat(),
        "new_snapshot": new_snapshot
    }

# --------------- Tool wrappers ---------------
def tool_read_trading_csv(arg: str) -> Dict[str, Any]:
    return _read_any(arg)

def tool_analyze(arg: str) -> Dict[str, Any]:
    try:
        maybe = json.loads(arg)
        if isinstance(maybe, dict) and "holdings" in maybe:
            snapshot = maybe
        else:
            snapshot = _read_any(arg)
    except Exception:
        snapshot = _read_any(arg)
    return analyze_portfolio(snapshot)

def tool_suggest(arg: str) -> Dict[str, Any]:
    try:
        maybe = json.loads(arg)
        snapshot = maybe if isinstance(maybe, dict) else _read_any(arg)
    except Exception:
        snapshot = _read_any(arg)
    summary = analyze_portfolio(snapshot)
    suggestions = suggest_actions_from_summary(summary)
    return {"summary": summary, "suggestions": suggestions}

def tool_simulate(arg: str) -> Dict[str, Any]:
    try:
        payload = json.loads(arg)
        action = payload.get("action")
        snapshot = payload.get("snapshot")
        params = payload.get("params", {})
        if not action or not snapshot:
            return {"error": "invalid_payload"}
    except Exception:
        try:
            a, s = arg.split("||", 1)
            action = json.loads(a)
            snapshot = json.loads(s)
        except Exception:
            return {"error": "invalid_argument_format"}
        params = {}
    return simulate_trade(action, snapshot, params)
