"""歷史回測報表產生器 — TWSE 量化交易策略回測.

使用 data_loader + strategy v2 對指定年份區間進行回溯，
輸出買賣訊號統計與績效摘要。

用法：
    python backtest.py --ticker 2330 --start 2004 --end 2026
    python backtest.py --all --start 2004 --end 2004
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import math

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import TWSEStockLoader
from strategy import (
    BigStockBuySignalV2,
    BigStockSellSignalV2,
    MarketSignalV2,
    SectorSignalV2,
    StockSurgeSignal,
    EntrySignal,
    CrashExitSignal,
    check_macd_surge,
    TradeSignal,
    MarketState,
    _safe_last,
)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# 大盤加權指數快取 (避免每次回測重複下載)
_TAIEX_CACHE: pd.DataFrame | None = None


def _get_taiex(start_str: str, end_str: str) -> pd.DataFrame:
    """取得加權指數資料 (快取)."""
    global _TAIEX_CACHE
    if _TAIEX_CACHE is not None:
        return _TAIEX_CACHE
    try:
        import yfinance as yf
        twii = yf.Ticker("^TWII")
        raw = twii.history(start=start_str, end=end_str)
        if not raw.empty:
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = df.index.tz_localize(None)
            _TAIEX_CACHE = df
            return df
    except Exception:
        pass
    return pd.DataFrame()


def run_backtest(
    ticker: str,
    start_year: int = 2004,
    end_year: int = 2026,
    progress: bool = True,
) -> dict[str, Any]:
    """對單一股票執行歷史回測.

    Args:
        ticker: 股票代號
        start_year: 起始年份
        end_year: 結束年份
        progress: 顯示進度

    Returns:
        {
            "ticker": 股票代號,
            "period": f"{start}-{end}",
            "total_days": N,
            "buy_signals": [...],
            "sell_signals": [...],
            "market_states": {...},
            "signal_stats": {...},
        }
    """
    loader = TWSEStockLoader()
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"

    if progress:
        print(f"📥 載入 {ticker} {start_str}~{end_str} ...")

    tf = loader.load_multi_timeframe(ticker, start=start_str, end=end_str, adjusted=True)
    daily = tf["daily"]
    weekly = tf["weekly"]
    monthly = tf["monthly"]

    if daily.empty:
        return {"ticker": ticker, "error": "無資料"}

    if progress:
        print(f"   日線 {len(daily)} 行, 週線 {len(weekly)} 行, 月線 {len(monthly)} 行")

    # ── 逐日掃描 ──
    buy_signals: list[dict] = []
    sell_signals: list[dict] = []
    market_states: Counter = Counter()
    surge_days = 0
    total_days = len(daily)

    # 大盤判定 (使用整段資料)
    ms = MarketSignalV2()
    market = ms.assess(daily)
    market_final = {
        "state": market.state.value,
        "score": market.score,
        "reasons": market.reasons,
    }

    # 大盤加權指數 (^TWII) — 用於進場閘門判斷
    taiex_df = _get_taiex(start_str, end_str)
    has_market_data = not taiex_df.empty

    # 每 30 天取樣一次 (比原來密, 因為條件更緊)
    sample_interval = 30

    for i in range(0, total_days, sample_interval):
        chunk = daily.iloc[:i+1]
        chunk_close = chunk["Adj_Close"]
        chunk_high = chunk["Adj_High"]
        chunk_low = chunk["Adj_Low"]

        if len(chunk) < 60:
            continue

        date_str = str(chunk["Date"].iloc[-1].date())
        chunk_date = pd.Timestamp(date_str)

        # 週線/月線同步裁切到當前掃描日 (避免使用未來資料造成前視偏差)
        weekly_chunk = weekly[weekly["Date"] <= chunk_date]
        monthly_chunk = monthly[monthly["Date"] <= chunk_date]

        # ── 大盤進場閘門 ──
        market_entry_zone = False
        market_entry_reason = ""
        if has_market_data and len(taiex_df) > 42:
            tx_sub = taiex_df[taiex_df.index <= chunk_date].copy()
            if len(tx_sub) >= 42:
                tx_close = tx_sub["Close"]
                tx_high = tx_sub["High"]
                tx_low = tx_sub["Low"]

                # N2
                hh = tx_high.rolling(42).max()
                ll = tx_low.rolling(42).min()
                n2_val = (hh + ll) / 2
                last_n2 = float(n2_val.iloc[-1])
                last_close = float(tx_close.iloc[-1])

                # 條件1: 備戰區 (N2-100)
                if last_close < last_n2 - 100:
                    market_entry_zone = True
                    market_entry_reason = f"備戰區: 加權{last_close:.0f} < N2{last_n2:.0f}-100"

                # 條件2: DIF210 創60日新低 (既有)
                if not market_entry_zone and len(tx_sub) >= 210:
                    from indicators import macd
                    m210 = macd(tx_close, fast=200, slow=209, signal=210)
                    dif = m210["macd"]
                    if len(dif) >= 60:
                        dif_now = float(dif.iloc[-1])
                        dif_low = float(dif.rolling(60).min().iloc[-1])
                        if not pd.isna(dif_now) and not pd.isna(dif_low) and dif_now <= dif_low:
                            market_entry_zone = True
                            market_entry_reason = f"DIF210觸底: {dif_now:.2f} 創60日新低"

                # 條件3: 崩盤買點 A — DIF210 觸碰下緣 (DIF<0 + Hist轉正)
                if not market_entry_zone and len(tx_sub) >= 210:
                    m210 = macd(tx_close, fast=200, slow=209, signal=210)
                    dif = m210["macd"]
                    hist_diff = m210["hist_diff"]
                    dif_now = float(dif.iloc[-1])
                    hist_diff_now = float(hist_diff.iloc[-1]) if len(hist_diff) > 0 else 0.0
                    if dif_now < 0 and hist_diff_now > 0:
                        market_entry_zone = True
                        market_entry_reason = f"崩盤買點A: DIF210={dif_now:.2f}<0, Hist轉正"

                # 條件4: 崩盤買點 B — 月線黑 6K (需月線大盤資料)
                if not market_entry_zone and len(tx_sub) >= 60:
                    # 從日線取月線
                    monthly_tx = tx_sub.resample("ME").agg({
                        "Open": "first", "High": "max", "Low": "min", "Close": "last",
                    }).dropna()
                    if len(monthly_tx) >= 10:
                        from indicators import k6k9
                        mk = k6k9(monthly_tx["High"], monthly_tx["Low"], monthly_tx["Close"],
                                  monthly_tx.get("Open", None))
                        k6k9_sig = _safe_last(mk["signal"], False)
                        k6k9_type = _safe_last(mk["signal_type"], None)
                        if k6k9_sig and k6k9_type == "DOWN_6K":
                            market_entry_zone = True
                            market_entry_reason = f"崩盤買點B: 月線黑6K成立"

                # 條件5: 空頭乖離搶反彈 — 日RSI60 < 34 (近似60分RSI)
                if not market_entry_zone and len(tx_sub) >= 60:
                    from indicators import rsi as rsi_fn
                    tx_rsi60 = rsi_fn(tx_close, period=60)
                    rsi60_last = _safe_last(tx_rsi60, 50.0)
                    if not pd.isna(rsi60_last) and rsi60_last < 34:
                        market_entry_zone = True
                        market_entry_reason = f"空頭乖離: 日RSI60={rsi60_last:.0f} < 34"

        # 買進訊號 (僅在大盤進場區時評估)
        if market_entry_zone:
            bbs = BigStockBuySignalV2()
            buy = bbs.evaluate(ticker, daily=chunk, weekly=weekly_chunk, monthly=monthly_chunk)
            if buy.signal in (TradeSignal.STRONG_BUY, TradeSignal.BUY):
                # 資金配置：依進場條件決定倉位大小
                reason_lower = market_entry_reason.lower()
                if "備戰區" in reason_lower:
                    position_size = 0.2  # 備戰區進場: 20%
                elif "空頭乖離" in reason_lower:
                    position_size = 0.2  # 空頭乖離搶反彈: 20%, 嚴禁攤平
                else:
                    position_size = 1.0  # 崩盤買點: 全倉

                buy_signals.append({
                    "date": date_str,
                    "signal": buy.signal.value,
                    "confidence": round(buy.confidence, 2),
                    "met": buy.conditions_met + [market_entry_reason],
                    "position_size": position_size,
                })
        elif progress:
            pass  # 大盤不在進場區, 不做買進檢查

        # 賣出訊號
        bss = BigStockSellSignalV2()
        sell = bss.evaluate(ticker, daily=chunk, weekly=weekly_chunk, monthly=monthly_chunk)
        if sell.signal in (TradeSignal.STRONG_SELL, TradeSignal.SELL):
            sell_signals.append({
                "date": date_str,
                "signal": sell.signal.value,
                "confidence": round(sell.confidence, 2),
                "met": sell.conditions_met,
            })

        # 飆漲訊號
        surge = StockSurgeSignal()
        s = surge.evaluate(chunk)
        if s["both_surge"]:
            surge_days += 1

    if progress:
        print(f"   買進訊號: {len(buy_signals)} 次")
        print(f"   賣出訊號: {len(sell_signals)} 次")
        print(f"   飆漲訊號: {surge_days} 次")

    # 模擬買賣操作
    sim = simulate_trades(daily, buy_signals, sell_signals)
    if progress and sim["trade_count"] > 0:
        print(f"   模擬交易: {sim['trade_count']} 次, 總損益 {sim['total_pl']:+.2f}, 報酬率 {sim['return_rate']:+.2f}%")

    return {
        "ticker": ticker,
        "period": f"{start_year}-{end_year}",
        "total_days": total_days,
        "market": market_final,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "surge_days": surge_days,
        "total_samples": total_days // sample_interval,
        "simulation": sim,
        "_daily": daily,  # 供圖表匯出使用
    }


def generate_report(results: list[dict[str, Any]]) -> str:
    """產生回測報表文字.

    Args:
        results: run_backtest 結果列表

    Returns:
        報表文字
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  TWSE 量化策略歷史回測報告")
    lines.append(f"  產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    for r in results:
        if "error" in r:
            lines.append(f"\n❌ {r['ticker']}: {r['error']}")
            continue

        lines.append(f"\n{'─' * 40}")
        lines.append(f"  📈 {r['ticker']} ({r['period']})")
        lines.append(f"{'─' * 40}")
        lines.append(f"  總樣本數: {r['total_days']} 日")
        lines.append(f"  掃描頻率: 每 60 日取樣一次")

        # 大盤狀態
        m = r.get("market", {})
        if m:
            lines.append(f"\n  大盤狀態: {m.get('state', 'N/A')} (score={m.get('score', 0)})")
            for reason in m.get("reasons", [])[:5]:
                lines.append(f"    • {reason}")

        # 買進訊號統計
        buys = r.get("buy_signals", [])
        lines.append(f"\n  🟢 買進訊號: {len(buys)} 次")
        high_conf = [b for b in buys if b["confidence"] >= 0.75]
        if high_conf:
            lines.append(f"    高信心度 (≥75%): {len(high_conf)} 次")
        for b in buys[:10]:
            lines.append(f"    + {b['date']} | {b['signal']} ({b['confidence']*100:.0f}%)")
            for cond in b["met"][:3]:
                lines.append(f"      └ {cond}")
        if len(buys) > 10:
            lines.append(f"    ... 尚有 {len(buys)-10} 筆")

        # 賣出訊號統計
        sells = r.get("sell_signals", [])
        lines.append(f"\n  🔴 賣出訊號: {len(sells)} 次")
        high_conf_s = [s for s in sells if s["confidence"] >= 0.6]
        if high_conf_s:
            lines.append(f"    高信心度 (≥60%): {len(high_conf_s)} 次")
        for s in sells[:10]:
            lines.append(f"    - {s['date']} | {s['signal']} ({s['confidence']*100:.0f}%)")
            for cond in s["met"][:3]:
                lines.append(f"      └ {cond}")
        if len(sells) > 10:
            lines.append(f"    ... 尚有 {len(sells)-10} 筆")

        # 飆漲訊號
        lines.append(f"\n  ⚡ 飆漲訊號 (DIF210+ADX300雙螺旋): {r.get('surge_days', 0)} 次")

        # 總結
        lines.append(f"\n  📊 摘要")
        lines.append(f"     取樣次數: {r.get('total_samples', 0)}")
        lines.append(f"     買進率: {len(buys)/max(r.get('total_samples',1),1)*100:.1f}%")
        lines.append(f"     賣出率: {len(sells)/max(r.get('total_samples',1),1)*100:.1f}%")

    lines.append(f"\n{'=' * 60}")
    lines.append("  報告完畢")
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TWSE 歷史回測報表")
    parser.add_argument("--ticker", default=None, help="股票代號")
    parser.add_argument("--all", action="store_true", help="回測所有可用股票")
    parser.add_argument("--watchlist", default="2330,2454,2317,2308,2301", help="逗號分隔清單")
    parser.add_argument("--start", type=int, default=2004, help="起始年份")
    parser.add_argument("--end", type=int, default=2026, help="結束年份")
    parser.add_argument("--output", default=None, help="輸出檔案路徑")
    args = parser.parse_args()

    if args.all:
        loader = TWSEStockLoader()
        tickers = loader.list_available_tickers()
        # 過濾掉非數字代號 (權證、認購等)
        tickers = [t for t in tickers if t.isdigit() and len(t) == 4]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [t.strip() for t in args.watchlist.split(",")]

    results = []
    for t in tickers:
        r = run_backtest(t, args.start, args.end)
        results.append(r)

    report = generate_report(results)

    output_path = args.output or os.path.join(
        REPORT_DIR, f"backtest_{date.today().isoformat()}.txt",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── 輸出 JSON (供 GitHub Pages dashboard 使用) ──
    json_path = None
    for candidate in [
        "D:/twse-surge-stocks-dna/docs/backtest_data.json",
        os.path.join(os.path.dirname(__file__), "..", "docs", "backtest_data.json"),
    ]:
        if os.path.exists(os.path.dirname(candidate)):
            json_path = candidate
            break

    if json_path and os.path.exists(os.path.dirname(json_path)):
        import json as json_mod
        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_until": f"{args.end}",
            "period": f"{args.start}-{args.end}",
            "tickers_tested": len(results),
            "ticker": args.ticker or "multiple",
            "market": results[0].get("market") if results else None,
            "stocks": [],
            "buy_signals": [],
            "sell_signals": [],
        }
        for r in results:
            ticker = r.get("ticker", "")
            buys = r.get("buy_signals", [])
            sells = r.get("sell_signals", [])
            samples = r.get("total_samples", 1)
            # per-stock summary
            sim = r.get("simulation", {})
            summary["stocks"].append({
                "ticker": ticker,
                "total_days": r.get("total_days", 0),
                "samples": samples,
                "market": r.get("market", {}),
                "buy_count": len(buys),
                "sell_count": len(sells),
                "buy_rate": round(len(buys) / max(samples, 1) * 100, 1),
                "sell_rate": round(len(sells) / max(samples, 1) * 100, 1),
                "surge_days": r.get("surge_days", 0),
                "trade_count": sim.get("trade_count", 0),
                "total_pl": sim.get("total_pl", 0),
                "return_rate": sim.get("return_rate", 0),
            })
            for b in buys:
                entry = dict(b)
                entry["ticker"] = ticker
                summary["buy_signals"].append(entry)
            for s in sells:
                entry = dict(s)
                entry["ticker"] = ticker
                summary["sell_signals"].append(entry)

        with open(json_path, "w", encoding="utf-8") as f:
            json_mod.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📊 Dashboard JSON: {json_path}")

        # ── 匯出圖表資料 (僅有訊號的股票) ──
        chart_count = 0
        for r in results:
            ticker = r.get("ticker", "")
            buys = r.get("buy_signals", [])
            sells = r.get("sell_signals", [])
            if not buys and not sells:
                continue
            daily = r.get("_daily")
            if daily is None or daily.empty:
                continue
            buy_dates = [b["date"] for b in buys]
            sell_dates = [s["date"] for s in sells]
            out = export_chart_data(ticker, daily, args.start, args.end, buy_dates, sell_dates)
            if out:
                chart_count += 1
        if chart_count > 0:
            print(f"📈 Chart JSON 匯出完成: {chart_count} 檔股票")
        # 從結果中移除 _daily 以節省記憶體
        for r in results:
            r.pop("_daily", None)

        # ── 自動推送到 GitHub Pages ──
        git_dir = os.path.dirname(json_path)
        try:
            import subprocess
            subprocess.run(
                ["git", "add", "."],
                check=False, cwd=git_dir,
                capture_output=True, timeout=30,
            )
            subprocess.run(
                ["git", "commit", "-m", f"auto: 更新回測報告 {args.start}-{args.end}"],
                check=False, cwd=git_dir,
                capture_output=True, timeout=30,
            )
            r = subprocess.run(
                ["git", "push"],
                check=False, cwd=git_dir,
                capture_output=True, timeout=60,
            )
            if r.returncode == 0:
                print(f"🚀 GitHub Pages 已更新 (push OK)")
            else:
                print(f"⚠️  Git push 略過 ({r.stderr.decode()[:80]})")
        except Exception as e:
            print(f"⚠️  Git push 略過: {e}")

    print(report)
    print(f"\n📁 報表已儲存: {output_path}")


# ═══════════════════════════════════════════════════════════════
# P&L Simulation — 買賣操作模擬
# ═══════════════════════════════════════════════════════════════

def simulate_trades(
    daily: pd.DataFrame,
    buy_signals: list[dict],
    sell_signals: list[dict],
) -> dict:
    """模擬股票買賣操作，計算盈虧與報酬率.

    規則：
    - 買進訊號依 position_size (預設 1.0) 建立倉位，累計不超過 1.0
    - 賣出訊號一律平掉全部累計倉位
    - 資金配置：備戰區/空頭乖離 20%，崩盤買點 100%

    Args:
        daily: 日線 DataFrame
        buy_signals: 買進訊號列表 (含 position_size)
        sell_signals: 賣出訊號列表

    Returns:
        {"trades": [...], "total_pl": float, "return_rate": float, "trade_count": int}
    """
    close_col = "Adj_Close" if "Adj_Close" in daily.columns else "Close"
    price_map = {}
    for _, row in daily.iterrows():
        d = row["Date"]
        if isinstance(d, pd.Timestamp):
            d = d.strftime("%Y-%m-%d")
        price_map[d] = float(row[close_col])

    # 合併訊號時間軸
    events: list[tuple[str, str, float, float]] = []  # (date, type, price, size)
    for b in buy_signals:
        d = b["date"]
        size = b.get("position_size", 1.0)
        if d in price_map:
            events.append((d, "buy", price_map[d], size))
    for s in sell_signals:
        d = s["date"]
        if d in price_map:
            events.append((d, "sell", price_map[d], 0.0))

    events.sort(key=lambda x: x[0])

    trades = []
    accumulated_pos = 0.0  # 0.0~1.0 累計倉位
    entry_price = 0.0
    entry_date = ""

    for date_str, typ, price, size in events:
        if typ == "buy" and accumulated_pos < 1.0:
            # 加倉 (最多累計到 1.0)
            add = min(size, 1.0 - accumulated_pos)
            if accumulated_pos == 0.0:
                entry_price = price
                entry_date = date_str
            else:
                # 加倉：平均成本
                entry_price = (entry_price * accumulated_pos + price * add) / (accumulated_pos + add)
            accumulated_pos += add

        elif typ == "sell" and accumulated_pos > 0:
            pl = (price - entry_price) * accumulated_pos
            ret = (price - entry_price) / entry_price * accumulated_pos * 100
            trades.append({
                "buy_date": entry_date,
                "buy_price": round(entry_price, 2),
                "sell_date": date_str,
                "sell_price": round(price, 2),
                "pl": round(pl, 2),
                "return_pct": round(ret, 2),
                "position_size": round(accumulated_pos, 2),
            })
            accumulated_pos = 0.0

    total_pl = sum(t["pl"] for t in trades)
    total_cost = sum(t["position_size"] * t["buy_price"] for t in trades)
    return_rate = round(total_pl / total_cost * 100, 2) if total_cost > 0 else 0.0

    return {
        "trades": trades,
        "total_pl": round(total_pl, 2),
        "return_rate": return_rate,
        "trade_count": len(trades),
    }

CHART_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "twse-surge-stocks-dna", "docs", "charts",
) if os.path.exists("D:/twse-surge-stocks-dna") else os.path.join(
    os.path.dirname(__file__), "..", "docs", "charts",
)
os.makedirs(CHART_DIR, exist_ok=True)


def compute_indicators_for_chart(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """在完整 daily 資料上計算圖表所需的技術指標."""
    from indicators import macd, dmi, wr, rsi, n2

    df = daily.copy()
    close = df["Adj_Close"] if "Adj_Close" in df.columns else df["Close"]
    high = df["Adj_High"] if "Adj_High" in df.columns else df["High"]
    low = df["Adj_Low"] if "Adj_Low" in df.columns else df["Low"]
    volume = df["Volume"]

    m = macd(close, 12, 26, 9)
    df["MACD_DIF"] = m["macd"]
    df["MACD_Signal"] = m["signal"]
    df["MACD_Hist"] = m["histogram"]

    d = dmi(high, low, close, 14)
    df["ADX"] = d["adx"]
    df["PDI"] = d["plus_di"]
    df["MDI"] = d["minus_di"]

    df["WR"] = wr(high, low, close, 14)
    df["RSI"] = rsi(close, 14)
    df["N2"] = n2(high, low, 42)
    df["MA60"] = close.rolling(window=60).mean()
    df["VMA20"] = volume.rolling(window=20).mean()
    return df


def export_chart_data(
    ticker: str,
    daily: pd.DataFrame,
    start_year: int,
    end_year: int,
    buy_dates: list[str],
    sell_dates: list[str],
) -> str | None:
    """匯出個股 + 大盤 K 線資料供 Dashboard 繪圖."""
    if not buy_dates and not sell_dates:
        return None

    stock_df = compute_indicators_for_chart(daily)
    stock_rows: list[dict] = []
    for _, row in stock_df.iterrows():
        d = row["Date"]
        if isinstance(d, pd.Timestamp):
            d = d.strftime("%Y-%m-%d")
        stock_rows.append({
            "d": d, "o": _round(row.get("Adj_Open", row["Open"])),
            "h": _round(row.get("Adj_High", row["High"])),
            "l": _round(row.get("Adj_Low", row["Low"])),
            "c": _round(row.get("Adj_Close", row["Close"])),
            "v": int(row["Volume"]),
            "ma60": _r(row.get("MA60")), "wr": _r(row.get("WR"),1),
            "rsi": _r(row.get("RSI"),1), "dif": _r(row.get("MACD_DIF"),3),
            "macds": _r(row.get("MACD_Signal"),3), "macdh": _r(row.get("MACD_Hist"),3),
            "adx": _r(row.get("ADX"),1),
            "n2": _r(row.get("N2")),
        })

    # 大盤加權指數 ^TWII (yfinance)
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"
    taiex_rows: list[dict] = []
    try:
        import yfinance as yf
        twii = yf.Ticker("^TWII")
        tx = twii.history(start=start_str, end=end_str)
        for dt, row in tx.iterrows():
            d = dt.strftime("%Y-%m-%d")
            taiex_rows.append({
                "d": d, "o": _round(row.get("Open")),
                "h": _round(row.get("High")),
                "l": _round(row.get("Low")),
                "c": _round(row.get("Close")),
                "v": int(row.get("Volume", 0)),
            })
    except Exception:
        pass

    chart_data = {
        "ticker": ticker,
        "period": f"{start_year}-{end_year}",
        "taiex": taiex_rows,
        "stock": stock_rows,
        "signals": {"buy": sorted(buy_dates), "sell": sorted(sell_dates)},
        "simulation": simulate_trades(daily, [{"date": d} for d in buy_dates], [{"date": d} for d in sell_dates]),
    }
    out_path = os.path.join(CHART_DIR, f"{ticker}.json")
    import json as json_mod
    with open(out_path, "w", encoding="utf-8") as f:
        json_mod.dump(chart_data, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    print(f"📈 Chart data: {out_path} ({os.path.getsize(out_path)/1024:.1f}KB)")
    return out_path


def _round(val, ndigits: int = 2) -> float:
    try:
        v = float(val)
        return round(v, ndigits) if not (pd.isna(v) or math.isnan(v)) else 0.0
    except (ValueError, TypeError):
        return 0.0

def _r(val, ndigits: int = 2) -> float | None:
    try:
        v = float(val)
        return round(v, ndigits) if not (pd.isna(v) or math.isnan(v)) else None
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
