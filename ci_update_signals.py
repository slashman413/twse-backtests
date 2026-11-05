#!/usr/bin/env python3
"""
CI daily update ? twse-backtests
- Fetches the latest session TWSE has published, labelled with TWSE's own date
- Updates docs/data/charts/*.json (180-day OHLCV window)
- Updates docs/data/signals.json scan_date and entry prices
"""
import json, sys, math
from datetime import datetime
from pathlib import Path

from slashman_finance import fetch_text, roc_to_ad, stock_day_all
from slashman_finance.twse import RWD, parse_stock_day_rwd

CHARTS_DIR  = Path("docs/data/charts")
SIGNALS_FILE = Path("docs/data/signals.json")

def fetch_prices():
    """(session ISO date, {code: {close, open, high, low, volume}}) for the latest published session.

    Bars must carry this date, never the clock: the data can be older than today. The RWD file
    rolls soon after the close; the OpenAPI file only rolls at about 05:20 Taipei the next
    morning, so a run on the cron would read the previous session from it.
    """
    try:
        roc, prices, _ = parse_stock_day_rwd(fetch_text(f"{RWD}/afterTrading/STOCK_DAY_ALL?response=json"))
    except Exception as e:
        print(f"  RWD STOCK_DAY_ALL failed: {e}, trying OpenAPI...")
        roc, prices = "", {}
    if not prices:
        roc, prices = stock_day_all(fallback=False)
    return roc_to_ad(roc).isoformat(), prices

def high20_series(ohlcv):
    """Rolling 20-bar high of the bar highs, from the 20th bar on."""
    highs = [r["high"] for r in ohlcv]
    return [{"time": ohlcv[i]["time"], "value": max(highs[i-19:i+1])}
            for i in range(19, len(highs))]

def update_chart(path, ticker, p, session):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except:
        data = {"ticker": ticker, "scan_date": session, "ohlcv": [], "high20": [], "macd": [], "adx": [], "wr": [], "signal_marks": []}

    ohlcv = data.get("ohlcv", [])
    if ohlcv and ohlcv[-1].get("time") > session:
        return  # older data than the chart already has
    if ohlcv and ohlcv[-1].get("time") == session:
        ohlcv.pop()  # re-read: volumes are provisional until about 14:35 Taipei

    ohlcv.append({"time": session, "open": p["open"], "high": p["high"],
                  "low": p["low"], "close": p["close"], "volume": int(p["volume"])})
    # Keep last 180 rows
    data["ohlcv"] = ohlcv[-180:]
    data["scan_date"] = session
    data["high20"] = high20_series(data["ohlcv"])

    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def main():
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] Fetching latest TWSE session...")
    try:
        session, prices = fetch_prices()
    except Exception as e:
        print(f"No price data: {e}"); sys.exit(0)
    if not prices:
        print("No price data ? non-trading day?"); sys.exit(0)
    print(f"  Got {len(prices)} stocks for session {session}")

    # Update chart files for signal stocks
    for chart_path in sorted(CHARTS_DIR.glob("*.json")):
        ticker = chart_path.stem
        if ticker in prices:
            update_chart(chart_path, ticker, prices[ticker], session)
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
    signals_data["scan_date"] = session
    SIGNALS_FILE.write_text(json.dumps(signals_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  signals.json: scan_date={session}, updated {updated}/{len(entry_sigs)} prices")

if __name__ == "__main__":
    main()
