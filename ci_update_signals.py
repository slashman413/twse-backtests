#!/usr/bin/env python3
"""
CI daily update ? twse-backtests
- Fetches today's TWSE prices
- Updates docs/data/charts/*.json (180-day OHLCV window)
- Updates docs/data/signals.json scan_date and entry prices
"""
import json, sys, math
from datetime import date, datetime
from pathlib import Path

from slashman_finance import stock_day_all

CHARTS_DIR  = Path("docs/data/charts")
SIGNALS_FILE = Path("docs/data/signals.json")
TODAY = date.today().isoformat()

def fetch_prices():
    """{code: {close, open, high, low, volume}} — OpenAPI first, RWD fallback (shared lib)."""
    return stock_day_all()[1]

def update_chart(path, ticker, p):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except:
        data = {"ticker": ticker, "scan_date": TODAY, "ohlcv": [], "high20": [], "macd": [], "adx": [], "wr": [], "signal_marks": []}

    ohlcv = data.get("ohlcv", [])
    if ohlcv and ohlcv[-1].get("time") == TODAY:
        return  # already updated today

    ohlcv.append({"time": TODAY, "open": p["open"], "high": p["high"],
                  "low": p["low"], "close": p["close"], "volume": int(p["volume"])})
    # Keep last 180 rows
    data["ohlcv"] = ohlcv[-180:]
    data["scan_date"] = TODAY

    # Recompute high20 from ohlcv
    highs = [r["high"] for r in data["ohlcv"]]
    h20 = []
    for i in range(len(highs)):
        if i >= 19:
            h20.append({"time": data["ohlcv"][i]["time"], "value": max(highs[i-19:i+1])})
    data["high20"] = h20

    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Fetching prices for {TODAY}...")
    prices = fetch_prices()
    if not prices:
        print("No price data ? non-trading day?"); sys.exit(0)
    print(f"  Got {len(prices)} stocks")

    # Update chart files for signal stocks
    for chart_path in sorted(CHARTS_DIR.glob("*.json")):
        ticker = chart_path.stem
        if ticker in prices:
            update_chart(chart_path, ticker, prices[ticker])
            print(f"  Chart updated: {ticker} -> {prices[ticker]['close']}")

    # Update signals.json
    signals_data = json.loads(SIGNALS_FILE.read_text(encoding="utf-8"))
    entry_sigs = signals_data.get("entry_signals", [])
    updated = 0
    for sig in entry_sigs:
        ticker = sig.get("ticker","")
        if ticker in prices:
            sig["price"] = prices[ticker]["close"]
            updated += 1
    signals_data["scan_date"] = TODAY
    SIGNALS_FILE.write_text(json.dumps(signals_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  signals.json: scan_date={TODAY}, updated {updated}/{len(entry_sigs)} prices")

if __name__ == "__main__":
    main()
