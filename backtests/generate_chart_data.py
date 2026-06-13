"""
Generate per-stock chart data JSON for the signals dashboard.
Reads current signals.json, outputs docs/data/charts/{ticker}.json
with 180 days of OHLCV + MACD/ADX/WR indicators.
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import date

_HERE   = os.path.dirname(os.path.abspath(__file__))
ADJ_TEMP_DIR = os.environ.get("ADJ_TEMP_DIR", os.path.join("D:/TWSE-Data/Adjusted", "_temp"))
OUT_DIR = os.environ.get("BACKTEST_OUT_DIR", os.path.join(_HERE, "..", "docs", "data"))
CORE_DIR = os.path.join(_HERE, "..", "core")
sys.path.insert(0, CORE_DIR)

from indicators import macd as macd_fn, dmi, wr as wr_fn

CHART_DIR  = os.path.join(OUT_DIR, "charts")
CHART_DAYS = 180   # days of history to include in chart


def round2(v):
    return round(float(v), 2) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None


def generate_chart(ticker: str, df_all: pd.DataFrame, scan_date: pd.Timestamp) -> dict:
    grp = df_all[df_all["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    if grp.empty:
        return {}

    # Limit to CHART_DAYS before scan_date; drop duplicate dates (keep last)
    grp = grp[grp["Date"] <= scan_date]
    grp = grp.drop_duplicates(subset=["Date"], keep="last")
    grp = grp.tail(CHART_DAYS).reset_index(drop=True)
    if len(grp) < 30:
        return {}

    close = grp["Adj_Close"].astype(float)
    high  = grp["Adj_High"].astype(float)
    low   = grp["Adj_Low"].astype(float)
    vol   = grp["Adj_Volume"].astype(float)
    n     = len(grp)

    cs = pd.Series(close.values, index=range(n))
    hs = pd.Series(high.values,  index=range(n))
    ls = pd.Series(low.values,   index=range(n))

    macd_r  = macd_fn(cs, 12, 26, 9)
    dmi_r   = dmi(hs, ls, cs, 14)
    wr_r    = wr_fn(hs, ls, cs, 50)

    # 20-day high (no look-ahead for signal, but show raw for chart)
    high20 = close.rolling(20).max().shift(1)
    avg_vol = (vol / 1000.0).rolling(20).mean().shift(1)
    rvol    = (vol / 1000.0) / (avg_vol + 1e-9)

    dates = grp["Date"].dt.strftime("%Y-%m-%d").tolist()

    ohlcv = [
        {
            "time":   dates[i],
            "open":   round2(grp["Adj_Open"].iloc[i]),
            "high":   round2(high.iloc[i]),
            "low":    round2(low.iloc[i]),
            "close":  round2(close.iloc[i]),
            "volume": int(vol.iloc[i] / 1000),  # in lots (張)
        }
        for i in range(n)
    ]

    macd_data = [
        {
            "time":  dates[i],
            "macd":  round2(macd_r["macd"].iloc[i]),
            "signal":round2(macd_r["signal"].iloc[i]),
            "hist":  round2(macd_r["histogram"].iloc[i]),
        }
        for i in range(n)
    ]

    adx_data = [
        {
            "time":     dates[i],
            "adx":      round2(dmi_r["adx"].iloc[i]),
            "plus_di":  round2(dmi_r["plus_di"].iloc[i]),
            "minus_di": round2(dmi_r["minus_di"].iloc[i]),
        }
        for i in range(n)
    ]

    wr_data = [
        {"time": dates[i], "wr": round2(wr_r.iloc[i])}
        for i in range(n)
    ]

    high20_data = [
        {"time": dates[i], "value": round2(high20.iloc[i])}
        for i in range(n) if not np.isnan(high20.iloc[i])
    ]

    # Mark signal date
    signal_marks = []
    if scan_date.strftime("%Y-%m-%d") in dates:
        idx = dates.index(scan_date.strftime("%Y-%m-%d"))
        signal_marks.append({
            "time":  dates[idx],
            "price": round2(close.iloc[idx]),
            "rvol":  round2(rvol.iloc[idx]),
        })

    return {
        "ticker":      ticker,
        "scan_date":   scan_date.strftime("%Y-%m-%d"),
        "ohlcv":       ohlcv,
        "high20":      high20_data,
        "macd":        macd_data,
        "adx":         adx_data,
        "wr":          wr_data,
        "signal_marks": signal_marks,
    }


def main():
    sig_path = os.path.join(OUT_DIR, "signals.json")
    if not os.path.exists(sig_path):
        print("signals.json not found — run scan_signals.py first")
        return

    with open(sig_path) as f:
        signals = json.load(f)

    tickers    = [s["ticker"] for s in signals.get("entry_signals", [])]
    scan_date  = pd.Timestamp(signals["scan_date"])
    curr_year  = scan_date.year

    if not tickers:
        print("No signals to chart.")
        return

    print(f"Loading {curr_year} data for {len(tickers)} tickers…")
    f_curr = os.path.join(ADJ_TEMP_DIR, f"{curr_year}.parquet")
    f_prev = os.path.join(ADJ_TEMP_DIR, f"{curr_year-1}.parquet")

    dfs = []
    if os.path.exists(f_prev):
        dp = pd.read_parquet(f_prev)
        dp["Date"] = pd.to_datetime(dp["Date"])
        dfs.append(dp)
    dc = pd.read_parquet(f_curr)
    dc["Date"] = pd.to_datetime(dc["Date"])
    dfs.append(dc)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["Ticker"] = df_all["Ticker"].astype(str).str.zfill(4)
    df_all = df_all[df_all["Ticker"].str.len() <= 4]
    df_all = df_all[df_all["Adj_Close"].notna()]

    os.makedirs(CHART_DIR, exist_ok=True)

    for ticker in tickers:
        print(f"  {ticker}…", end=" ", flush=True)
        data = generate_chart(ticker, df_all, scan_date)
        if data:
            path = os.path.join(CHART_DIR, f"{ticker}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            print(f"{len(data['ohlcv'])} days")
        else:
            print("no data")

    print(f"✅ Charts saved to {CHART_DIR}")


if __name__ == "__main__":
    main()
