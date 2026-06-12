"""
TWSE 還原權值驗證器 — 與 yfinance 比對確認資料正確性
=====================================================

使用方式：
    python twse_validator.py --tickers 2330,2454,2317,2412,2308
    python twse_validator.py --tickers 2330 --full  # 完整檢驗

策略：
    1. 用 yfinance 抓取指定台股的原始 OHLCV + 還原收盤價 (Adj Close)
    2. 用我們 adjuster.py 的 Backward Adjustment 計算還原價
    3. 比對兩者的 Adj Close，計算 MAPE (平均絕對百分比誤差)
    4. 若有偏差，分析原因（股利缺失、配股率差異等）

驗證通過條件：MAPE < 0.5% 且最大單日誤差 < 2%
"""

from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── 路徑 ──────────────────────────────────────────────────────
DATA_DIR = os.environ.get("TWSE_DATA_DIR", "D:/TWSE-Data")
RAW_DIR = os.path.join(DATA_DIR, "Raw")
ADJ_DIR = os.path.join(DATA_DIR, "Adjusted")

# ── 還原權值計算 (獨立版, 與 adjuster.py 邏輯一致) ──────────


def backward_adjust(
    prices: pd.DataFrame,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """向前還原 (Backward Adjustment) ─ 用於驗證.

    與 twse_adjuster.py 的 _compute_cumulative_factors 邏輯相同，
    但獨立於 adjuster.py 模組。

    Args:
        prices: 原始日線，須含 ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        dividends: 除權息資料，須含 ['Date', 'Cash_Dividend', 'Stock_Dividend']
                   若 None 則不回補（僅用 yfinance 的 dividend 欄位）

    Returns:
        含 Adj_Open, Adj_High, Adj_Low, Adj_Close, CumFactor 的 DataFrame
    """
    result = prices.sort_values("Date").copy()
    n = len(result)

    closes = result["Close"].values.astype(np.float64)
    opens = result["Open"].values.astype(np.float64)
    highs = result["High"].values.astype(np.float64)
    lows = result["Low"].values.astype(np.float64)
    vols = result["Volume"].values.astype(np.float64)
    dates = result["Date"].values

    # 建立事件查詢
    event_map: dict[pd.Timestamp, tuple[float, float]] = {}
    if dividends is not None:
        for _, row in dividends.iterrows():
            event_map[row["Date"]] = (
                float(row["Cash_Dividend"]),
                float(row["Stock_Dividend"]),
            )

    # 遞迴計算累積因子（從最新日往舊日）
    cum_factors = np.ones(n, dtype=np.float64)
    cum_factor = 1.0

    for i in range(n - 1, -1, -1):
        cum_factors[i] = cum_factor

        evt = event_map.get(dates[i])
        if evt is not None and i > 0:
            d_cash, d_stock = evt
            prev_close = float(closes[i - 1])
            if np.isnan(prev_close) or prev_close <= 0:
                continue

            denom = 1.0 + d_stock / 1000.0
            ref_price = (prev_close - d_cash) / denom
            event_factor = ref_price / prev_close
            event_factor = min(event_factor, 1.0)
            cum_factor *= event_factor

    result["Adj_Open"] = opens * cum_factors
    result["Adj_High"] = highs * cum_factors
    result["Adj_Low"] = lows * cum_factors
    result["Adj_Close"] = closes * cum_factors
    result["Adj_Volume"] = (vols * cum_factors).round(0).astype("int64")
    result["CumFactor"] = cum_factors

    return result


# ═══════════════════════════════════════════════════════════════
# yfinance 資料擷取
# ═══════════════════════════════════════════════════════════════

def fetch_yf_data(
    ticker: str,
    period: str = "5y",
) -> dict[str, pd.DataFrame]:
    """從 yfinance 抓取台股資料.

    Args:
        ticker: 台股代號, e.g. "2330"
        period: "5y" 或 "max" 等

    Returns:
        {
            "raw": 原始 OHLCV (含 Dividend, Stock Split),
            "adj": yfinance 已還原的價格 (Adj Close),
        }
    """
    yf_ticker = f"{ticker}.TW"
    tk = yf.Ticker(yf_ticker)

    # 抓歷史資料 (auto_adjust=False 取得原始 + Adj Close)
    hist = tk.history(period=period, auto_adjust=False)
    if hist.empty:
        # 試 OTC
        yf_ticker = f"{ticker}.TWO"
        tk = yf.Ticker(yf_ticker)
        hist = tk.history(period=period, auto_adjust=False)

    if hist.empty:
        raise ValueError(f"yfinance 無資料: {ticker}")

    # 重新命名欄位
    hist = hist.rename(columns={
        "Open": "yf_Open",
        "High": "yf_High",
        "Low": "yf_Low",
        "Close": "yf_Close",
        "Adj Close": "yf_Adj_Close",
        "Volume": "yf_Volume",
        "Dividends": "yf_Dividend",
        "Stock Splits": "yf_Split",
    })
    hist.index.name = "Date"
    hist.reset_index(inplace=True)
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_convert(None).dt.normalize()

    # 股利資料 (從 tk.dividends 取得，更可靠)
    div = tk.dividends
    splits = tk.splits

    div_df = pd.DataFrame()
    if not div.empty:
        div_df = div.reset_index()
        div_df.columns = ["Date", "yf_Dividend_Amount"]
        div_df["Date"] = pd.to_datetime(div_df["Date"]).dt.tz_convert(None).dt.normalize()

    split_df = pd.DataFrame()
    if not splits.empty:
        split_df = splits.reset_index()
        split_df.columns = ["Date", "yf_Split_Ratio"]
        split_df["Date"] = pd.to_datetime(split_df["Date"]).dt.tz_convert(None).dt.normalize()

    return {
        "raw": hist,
        "dividend": div_df,
        "split": split_df,
        "info": tk.info if hasattr(tk, "info") else {},
    }


# ═══════════════════════════════════════════════════════════════
# 驗證比對
# ═══════════════════════════════════════════════════════════════

def validate_ticker(
    ticker: str,
    period: str = "10y",
    plot: bool = False,
) -> dict[str, any]:
    """對單一股票執行還原權值驗證.

    Args:
        ticker: 台股代號
        period: 回測期間
        plot: 是否輸出 HTML 圖表

    Returns:
        驗證報告 dict
    """
    print(f"\n{'='*60}")
    print(f"驗證: {ticker}")
    print(f"{'='*60}")

    # 1. 從 yfinance 抓資料
    print(f"📥 從 yfinance 抓取 {ticker}...")
    yf_data = fetch_yf_data(ticker, period=period)
    hist = yf_data["raw"]

    if hist.empty:
        return {"ticker": ticker, "status": "NO_DATA", "error": "yfinance 無回傳資料"}

    print(f"   期間: {hist['Date'].min().date()} ~ {hist['Date'].max().date()}")
    print(f"   交易日: {len(hist)}")

    # 2. 建立 dividends 資料 (從 yfinance)
    div_events = []
    if not yf_data["dividend"].empty:
        div_df = yf_data["dividend"]
        # 合併到 hist 上，找出有配息的日期
        hist_with_div = hist.merge(div_df, on="Date", how="left")
        for _, row in hist_with_div.iterrows():
            if pd.notna(row.get("yf_Dividend_Amount", np.nan)) and row["yf_Dividend_Amount"] > 0:
                div_events.append({
                    "Date": row["Date"],
                    "Cash_Dividend": float(row["yf_Dividend_Amount"]),
                    "Stock_Dividend": 0.0,  # 由 split 另行處理
                })

    # 3. 處理股票分割
    split_events = []
    if not yf_data["split"].empty:
        split_df = yf_data["split"]
        hist_with_split = hist.merge(split_df, on="Date", how="left")
        for _, row in hist_with_split.iterrows():
            if pd.notna(row.get("yf_Split_Ratio", np.nan)) and row["yf_Split_Ratio"] != 1.0:
                ratio = float(row["yf_Split_Ratio"])
                # yfinance split ratio: 2.0 = 2:1 split
                # For 配股: Stock_Dividend = (ratio - 1) * 1000
                if ratio > 0:
                    stock_div = max(0, (ratio - 1.0) * 1000.0)
                    split_events.append({
                        "Date": row["Date"],
                        "Cash_Dividend": 0.0,
                        "Stock_Dividend": stock_div,
                    })

    # 合併所有事件 (按日期排序)
    all_events = pd.DataFrame(div_events + split_events)
    if not all_events.empty:
        all_events = all_events.sort_values("Date").drop_duplicates(subset=["Date"])
        print(f"   除權息事件: {len(all_events)} 次")

        # 顯示前幾筆
        for _, evt in all_events.head(10).iterrows():
            parts = []
            if evt["Cash_Dividend"] > 0:
                parts.append(f"配息 ${evt['Cash_Dividend']:.2f}")
            if evt["Stock_Dividend"] > 0:
                parts.append(f"配股 {evt['Stock_Dividend']:.0f}/千股")
            print(f"     {evt['Date'].date()}: {', '.join(parts)}")

        if len(all_events) > 10:
            print(f"     ... 尚有 {len(all_events) - 10} 筆")
    else:
        print(f"   無除權息事件")

    # 4. 用我們的 Backward Adjustment 計算
    prices_for_adj = hist.rename(columns={
        "yf_Open": "Open", "yf_High": "High",
        "yf_Low": "Low", "yf_Close": "Close", "yf_Volume": "Volume",
    })[["Date", "Open", "High", "Low", "Close", "Volume"]]

    adj_result = backward_adjust(prices_for_adj, all_events if not all_events.empty else None)

    # 5. 比對：我們的 Adj_Close vs yfinance 的 Adj Close
    # yfinance auto_adjust=False 已經有 'Adj Close' 欄位
    comp = adj_result.copy()

    # 對齊日期
    yf_adj = hist[["Date", "yf_Adj_Close"]].copy()
    comp = comp.merge(yf_adj, on="Date", how="left")

    # 6. 計算誤差指標
    valid = comp.dropna(subset=["yf_Adj_Close", "Adj_Close"]).copy()
    if valid.empty:
        return {
            "ticker": ticker,
            "status": "NO_MATCH",
            "error": "無法對齊 yfinance Adj Close",
        }

    valid["Error_Pct"] = (
        (valid["Adj_Close"] - valid["yf_Adj_Close"]).abs()
        / valid["yf_Adj_Close"]
        * 100
    )

    mape = valid["Error_Pct"].mean()
    max_err = valid["Error_Pct"].max()
    std_err = valid["Error_Pct"].std()
    p95_err = valid["Error_Pct"].quantile(0.95)

    recent = valid.tail(min(252, len(valid)))
    recent_mape = recent["Error_Pct"].mean()

    print(f"\n📊 誤差統計:")
    print(f"   MAPE (全期):          {mape:.4f}%")
    print(f"   MAPE (近1年):         {recent_mape:.4f}%")
    print(f"   最大單日誤差:         {max_err:.4f}%")
    print(f"   標準差:               {std_err:.4f}%")
    print(f"   P95 誤差:             {p95_err:.4f}%")
    print(f"   可比對交易日:         {len(valid)}")

    # 找出最大誤差日
    worst = valid.loc[valid["Error_Pct"].idxmax()]
    print(f"\n⚠️  最大誤差日: {worst['Date'].date()}")
    print(f"   我們的 Adj_Close: {worst['Adj_Close']:.4f}")
    print(f"   yfinance Adj_Close: {worst['yf_Adj_Close']:.4f}")
    print(f"   原始 Close: {worst['Close']:.4f}")

    # 7. 判斷
    passed = mape < 0.5 and max_err < 2.0
    status = "✅ PASS" if passed else "⚠️ FAIL"
    print(f"\n   → {status} (MAPE={mape:.4f}%, MaxErr={max_err:.4f}%)")

    # 8. 找出疑似缺失的股利事件
    # 檢查 CumFactor 在股利日前後是否有變化
    missing_divs = []
    if not all_events.empty and len(valid) > 20:
        for _, evt in all_events.iterrows():
            evt_date = evt["Date"]
            # 找事件日前後的 CumFactor 變化
            before = valid[valid["Date"] < evt_date].tail(1)
            after = valid[valid["Date"] >= evt_date].head(1)
            if not before.empty and not after.empty:
                cf_before = before["CumFactor"].values[0]
                cf_after = after["CumFactor"].values[0]
                if abs(cf_before - cf_after) < 1e-8:
                    missing_divs.append(evt_date)

    if missing_divs:
        print(f"\n⚠️  可能缺失股利事件的日期 ({len(missing_divs)} 筆):")
        for d in missing_divs[:5]:
            print(f"     {d.date()}")

    return {
        "ticker": ticker,
        "status": status,
        "mape_pct": round(mape, 4),
        "recent_mape_pct": round(recent_mape, 4),
        "max_error_pct": round(max_err, 4),
        "std_pct": round(std_err, 4),
        "p95_pct": round(p95_err, 4),
        "n_days": len(valid),
        "n_events": len(all_events),
        "passed": passed,
        "worst_date": str(worst["Date"].date()),
        "worst_error_pct": round(valid["Error_Pct"].max(), 4),
    }


# ═══════════════════════════════════════════════════════════════
# 大量驗證 (多檔股票)
# ═══════════════════════════════════════════════════════════════

def validate_batch(
    tickers: list[str],
    period: str = "10y",
) -> pd.DataFrame:
    """批量驗證多檔股票.

    Args:
        tickers: 股票代號列表
        period: 回測期間

    Returns:
        驗證摘要 DataFrame
    """
    results = []
    for i, ticker in enumerate(tickers):
        print(f"\n[{i+1}/{len(tickers)}] ", end="")
        try:
            r = validate_ticker(ticker, period=period)
            results.append(r)
        except Exception as e:
            print(f"  ❌ {e}")
            results.append({"ticker": ticker, "status": "ERROR", "error": str(e)})

    summary = pd.DataFrame(results)

    # 摘要
    passed = summary["passed"].sum() if "passed" in summary.columns else 0
    total = len(results)
    print(f"\n{'='*60}")
    print(f"📋 驗證摘要: {passed}/{total} 通過")

    if "mape_pct" in summary.columns:
        print(f"   平均 MAPE: {summary['mape_pct'].mean():.4f}%")
        print(f"   最大 MAPE: {summary['mape_pct'].max():.4f}%")

    return summary


# ═══════════════════════════════════════════════════════════════
# 股利資料匯出 (給 adjuster.py 用)
# ═══════════════════════════════════════════════════════════════

def export_dividends_from_yf(
    tickers: list[str],
    output_path: str | None = None,
    period: str = "max",
) -> pd.DataFrame:
    """從 yfinance 匯出股利資料，供 adjuster.py 使用。

    對每檔股票抓取 dividend history + stock splits，
    轉換成 adjuster.py 接受的格式:
        Date, Ticker, Cash_Dividend, Stock_Dividend

    Args:
        tickers: 股票代號列表
        output_path: 匯出 CSV 路徑 (None=不回檔)
        period: 往回抓取期間

    Returns:
        合併的股利 DataFrame
    """
    all_divs = []

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
        try:
            # 試上市
            tk = yf.Ticker(f"{ticker}.TW")
            hist = tk.history(period="1mo")

            dividends = tk.dividends
            splits = tk.splits

            if dividends.empty and splits.empty:
                # 試上櫃
                tk = yf.Ticker(f"{ticker}.TWO")
                dividends = tk.dividends
                splits = tk.splits

            events = []

            # 現金股利
            if not dividends.empty:
                for dt, amt in dividends.items():
                    dt_naive = pd.Timestamp(dt).tz_localize(None)
                    events.append({
                        "Date": dt_naive,
                        "Ticker": ticker,
                        "Cash_Dividend": float(amt),
                        "Stock_Dividend": 0.0,
                    })

            # 股票股利 (從 split ratio 換算)
            if not splits.empty:
                for dt, ratio in splits.items():
                    dt_naive = pd.Timestamp(dt).tz_localize(None)
                    if ratio != 1.0 and ratio > 0:
                        stock_div = max(0, (ratio - 1.0) * 1000.0)
                        events.append({
                            "Date": dt_naive,
                            "Ticker": ticker,
                            "Cash_Dividend": 0.0,
                            "Stock_Dividend": round(stock_div, 2),
                        })

            if events:
                df = pd.DataFrame(events).sort_values("Date")
                all_divs.append(df)
                print(f"{len(events)} 筆事件")
            else:
                print("無資料")

        except Exception as e:
            print(f"❌ {e}")

    if not all_divs:
        print("⚠️ 無任何股利資料")
        return pd.DataFrame()

    merged = pd.concat(all_divs, ignore_index=True).sort_values(["Ticker", "Date"])
    merged = merged.drop_duplicates(subset=["Date", "Ticker"])

    if output_path:
        merged.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n📦 匯出 {len(merged)} 筆 → {output_path}")

    return merged


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TWSE 還原權值驗證器")
    parser.add_argument("--tickers", default="2330,2454,2317,2412,2308",
                        help="股票代號，逗號分隔")
    parser.add_argument("--period", default="10y",
                        help="回測期間 (e.g. 5y, 10y, max)")
    parser.add_argument("--full", action="store_true",
                        help="單檔完整檢驗")
    parser.add_argument("--export-dividends",
                        help="從 yfinance 匯出股利資料到指定 CSV")

    args = parser.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]

    if args.export_dividends:
        print(f"📥 從 yfinance 匯出股利資料...")
        export_dividends_from_yf(tickers, output_path=args.export_dividends,
                                 period="max" if args.period == "max" else args.period)
    else:
        validate_batch(tickers, period=args.period)
