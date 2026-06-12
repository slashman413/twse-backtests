"""Bulk chart generator — loads adjusted_all.parquet once, exports all signal stocks."""
from __future__ import annotations
import json, os, sys, math, time
import pandas as pd
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from indicators import macd, dmi, wr, rsi, n2

ADJUSTED_PATH = os.environ.get("ADJ_ALL", "D:/TWSE-Data/Adjusted/adjusted_all.parquet")
_DOCS = os.path.join(_HERE, "..", "docs")
JSON_PATH = os.environ.get("BACKTEST_JSON", os.path.join(_DOCS, "data", "summary.json"))
CHART_DIR = os.environ.get("CHART_DIR",    os.path.join(_DOCS, "charts"))
os.makedirs(CHART_DIR, exist_ok=True)

# Load signal stocks
with open(JSON_PATH) as f:
    data = json.load(f)

signal_stocks = [s for s in data.get("stocks",[]) if (s.get("buy_count",0) or s.get("sell_count",0)) > 0]
existing = set(f.replace(".json","") for f in os.listdir(CHART_DIR) if f.endswith(".json"))
missing = [s for s in signal_stocks if s["ticker"] not in existing]
print(f"Total signal: {len(signal_stocks)}, Existing: {len(existing)}, Missing: {len(missing)}")
if not missing:
    print("All done!")
    sys.exit(0)

# Build signal lookup
buy_map, sell_map = {}, {}
for b in data.get("buy_signals", []):
    buy_map.setdefault(b["ticker"], []).append(b["date"])
for s in data.get("sell_signals", []):
    sell_map.setdefault(s["ticker"], []).append(s["date"])

missing_tickers = [s["ticker"] for s in missing]
print(f"Loading parquet for {len(missing_tickers)} tickers...")
df = pd.read_parquet(
    ADJUSTED_PATH,
    filters=[("Ticker", "in", missing_tickers)],
    columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume",
             "Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume"],
)
print(f"Loaded {len(df)} rows for {df['Ticker'].nunique()} tickers")
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

def _r(val, nd=2):
    try:
        v = float(val)
        return round(v, nd) if not (np.isnan(v) or math.isnan(v)) else None
    except: return None

def _round(val, nd=2):
    try: return round(float(val), nd)
    except: return 0.0

t0 = time.time()
for idx, ticker in enumerate(missing_tickers):
    t1 = time.time()
    sub = df[df["Ticker"] == ticker].sort_values("Date").copy()
    if sub.empty:
        print(f"[{idx+1}/{len(missing_tickers)}] {ticker}: no data, skip")
        continue

    close = sub["Adj_Close"]
    high = sub["Adj_High"]
    low = sub["Adj_Low"]

    m = macd(close, 12, 26, 9)
    sub["dif"] = m["macd"]
    sub["macds"] = m["signal"]
    sub["macdh"] = m["histogram"]
    d = dmi(high, low, close, 14)
    sub["adx"] = d["adx"]
    sub["wr"] = wr(high, low, close, 14)
    sub["rsi"] = rsi(close, 14)
    sub["n2"] = n2(high, low, 42)
    sub["ma60"] = close.rolling(60).mean()

    stock_rows = []
    for _, row in sub.iterrows():
        stock_rows.append({
            "d": row["Date"], "o": _round(row["Adj_Open"]),
            "h": _round(row["Adj_High"]), "l": _round(row["Adj_Low"]),
            "c": _round(row["Adj_Close"]), "v": int(row["Volume"]),
            "ma60": _r(row.get("ma60")), "wr": _r(row.get("wr"),1),
            "rsi": _r(row.get("rsi"),1), "dif": _r(row.get("dif"),3),
            "macds": _r(row.get("macds"),3), "macdh": _r(row.get("macdh"),3),
            "adx": _r(row.get("adx"),1), "n2": _r(row.get("n2")),
        })

    # 0050
    taiex_rows = []
    tx = df[df["Ticker"] == "0050"].sort_values("Date")
    if not tx.empty:
        for _, row in tx.iterrows():
            taiex_rows.append({
                "d": row["Date"], "o": _round(row["Adj_Open"]),
                "h": _round(row["Adj_High"]), "l": _round(row["Adj_Low"]),
                "c": _round(row["Adj_Close"]), "v": int(row["Volume"]),
            })

    # P&L
    buys = buy_map.get(ticker, [])
    sells = sell_map.get(ticker, [])
    pm = {r["d"]: r["c"] for r in stock_rows}
    trades, holding, ep, ed = [], False, 0.0, ""
    for d_str in sorted(set(buys) | set(sells)):
        if d_str in pm:
            p = pm[d_str]
            if d_str in buys and not holding:
                holding, ep, ed = True, p, d_str
            elif d_str in sells and holding:
                trades.append({"buy_date":ed, "buy_price":round(ep,2),
                    "sell_date":d_str, "sell_price":round(p,2),
                    "pl":round(p-ep,2), "return_pct":round((p-ep)/ep*100,2)})
                holding = False
    total_pl = sum(t["pl"] for t in trades)
    total_cost = sum(t["buy_price"] for t in trades)
    return_rate = round(total_pl/total_cost*100,2) if total_cost > 0 else 0.0

    chart_data = {
        "ticker": ticker, "period": f"{stock_rows[0]['date'][:4]}-{stock_rows[-1]['date'][:4]}" if stock_rows else "",
        "taiex": taiex_rows, "stock": stock_rows,
        "signals": {"buy": sorted(buys), "sell": sorted(sells)},
        "simulation": {"trades":trades, "total_pl":round(total_pl,2),
            "return_rate":return_rate, "trade_count":len(trades)},
    }
    out = os.path.join(CHART_DIR, f"{ticker}.json")
    with open(out, "w") as f:
        json.dump(chart_data, f, ensure_ascii=False, separators=(",",":"))
    elapsed = time.time() - t1
    print(f"[{idx+1}/{len(missing_tickers)}] {ticker} ({len(stock_rows)}d) {elapsed:.1f}s - {len(trades)} trades")

total = time.time() - t0
print(f"\nDone! {len(missing_tickers)} charts in {total:.0f}s ({total/len(missing_tickers):.1f}s avg)")
