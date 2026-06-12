"""Yearly backtest — capital simulation with position sizing (10% per trade).

Fixes applied vs run_yearly_backtest_v7.py:
- No look-ahead bias: w_vr, m_vr, m_rsi4, m_pdi1 are precomputed as time-series
  and filtered to only past data at each sample point.
- Positive/negative year classification uses > 0 / <= 0 (was broken > -99.99).
- Paths read from env vars / config, no hardcoded machine paths.
- Volume backward adjustment uses / CumFactor (see adjuster.py fix).
"""
import os, sys, time, json, gc, math
import pandas as pd
import numpy as np

# ── Path config ──────────────────────────────────────────────────────────────
ADJ_TEMP_DIR = os.environ.get("ADJ_TEMP_DIR", os.path.join("D:/TWSE-Data/Adjusted", "_temp"))
OUT_DIR      = os.environ.get("BACKTEST_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "docs", "data"))
CORE_DIR     = os.path.join(os.path.dirname(__file__), "..", "core")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, CORE_DIR)

from indicators import macd_4arrows, dmi, wr, rsi
from strategy import _safe_last

INITIAL_CAPITAL  = 1_000_000
FIXED_ALLOCATION = 100_000   # 每筆固定 10 萬，不隨跨股現金變動
SAMPLE_STEP      = 30        # 每 30 日取樣一次


def _lookup_series(series: pd.Series, cutoff: pd.Timestamp, default: float) -> float:
    """Return the last value in series whose index <= cutoff, else default."""
    if series.empty:
        return default
    past = series[series.index <= cutoff]
    return float(past.iloc[-1]) if not past.empty else default


def process_year(year: int):
    t0 = time.time()
    f = os.path.join(ADJ_TEMP_DIR, f"{year}.parquet")
    if not os.path.exists(f):
        return None

    df = pd.read_parquet(f)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(4)

    for col in ["Adj_Close", "Adj_High", "Adj_Low"]:
        df = df[df[col].notna()]

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    n_total = len(df)

    trades = []
    year_pl = 0.0
    equity_curve = []
    n_tickers_simulated = 0

    for ticker, grp in df.groupby("Ticker", sort=False):
        grp = grp.reset_index(drop=True)
        close  = np.nan_to_num(grp["Adj_Close"].values.astype(np.float64),  nan=0.0)
        high   = np.nan_to_num(grp["Adj_High"].values.astype(np.float64),   nan=0.0)
        low    = np.nan_to_num(grp["Adj_Low"].values.astype(np.float64),    nan=0.0)
        volume = grp["Adj_Volume"].values.astype(np.float64)
        dates  = grp["Date"].values
        n = len(close)

        if n < 100:
            continue
        n_tickers_simulated += 1

        close_s = pd.Series(close, index=range(n))
        high_s  = pd.Series(high,  index=range(n))
        low_s   = pd.Series(low,   index=range(n))

        m4       = macd_4arrows(close_s, fast=200, slow=209, signal=210)
        d4       = np.nan_to_num(m4["arrows_count"].values, nan=0)
        dm       = dmi(high_s, low_s, close_s, period=300)
        adx_arr  = np.nan_to_num(dm["adx"].values,           nan=0)
        wr_arr   = np.nan_to_num(wr(high_s, low_s, close_s, 50).values, nan=0)
        rsi60_arr = np.nan_to_num(rsi(close_s, 60).values,   nan=50)

        # ── Precompute time-indexed indicator series (no look-ahead) ──────
        daily_df = pd.DataFrame({
            "Date": pd.to_datetime(dates), "Close": close,
            "High": high, "Low": low, "Volume": volume
        }).set_index("Date")

        weekly  = daily_df.resample("W").agg({"Close":"last","High":"max","Low":"min","Volume":"sum"}).dropna()
        monthly = daily_df.resample("ME").agg({"Close":"last","High":"max","Low":"min","Volume":"sum"}).dropna()

        # Weekly VR series (index = week-end date)
        if len(weekly) > 2:
            w_up    = weekly["Close"].diff() > 0
            w_down  = weekly["Close"].diff() < 0
            w_avs   = (weekly["Volume"] * w_up).rolling(2).sum()
            w_bvs   = (weekly["Volume"] * w_down).rolling(2).sum()
            w_vr_s  = (100.0 * w_avs / w_bvs.replace(0, np.nan)).fillna(0.0)
        else:
            w_vr_s = pd.Series(dtype=float)

        # Monthly VR series
        if len(monthly) > 2:
            m_up    = monthly["Close"].diff() > 0
            m_down  = monthly["Close"].diff() < 0
            m_avs   = (monthly["Volume"] * m_up).rolling(2).sum()
            m_bvs   = (monthly["Volume"] * m_down).rolling(2).sum()
            m_vr_s  = (100.0 * m_avs / m_bvs.replace(0, np.nan)).fillna(0.0)
        else:
            m_vr_s = pd.Series(dtype=float)

        # Monthly RSI4 and +DI1 series
        if len(monthly) > 14:
            _mc = pd.Series(monthly["Close"].values, index=range(len(monthly)))
            _m_rsi4_raw = rsi(_mc, 4)
            _m_rsi4_raw.index = monthly.index
            m_rsi4_s = _m_rsi4_raw

            _m_dmi = dmi(
                pd.Series(monthly["High"].values,  index=range(len(monthly))),
                pd.Series(monthly["Low"].values,   index=range(len(monthly))),
                _mc, period=1
            )
            _m_pdi1_raw = _m_dmi["plus_di"]
            _m_pdi1_raw.index = monthly.index
            m_pdi1_s = _m_pdi1_raw
        else:
            m_rsi4_s = pd.Series(dtype=float)
            m_pdi1_s = pd.Series(dtype=float)

        # ── Per-ticker simulation ─────────────────────────────────────────
        ticker_cash = FIXED_ALLOCATION
        position_shares    = 0
        position_buy_price = None
        position_buy_date  = None

        for i in range(min(100, n), n, SAMPLE_STEP):
            if close[i] == 0.0:
                continue

            current_date = pd.Timestamp(dates[i])

            # Look up indicators up to current_date only (no look-ahead)
            w_vr   = _lookup_series(w_vr_s,   current_date, 0.0)
            m_vr   = _lookup_series(m_vr_s,   current_date, 0.0)
            m_rsi4 = _lookup_series(m_rsi4_s, current_date, 50.0)
            m_pdi1 = _lookup_series(m_pdi1_s, current_date, 0.0)

            d4_val   = float(d4[i])
            adx_val  = float(adx_arr[i])
            wr_val   = float(wr_arr[i])
            rsi60_val = float(rsi60_arr[i])

            # Sell check
            if position_shares > 0:
                sell_flag   = False
                sell_reason = ""
                if m_rsi4 < 77:
                    sell_flag   = True
                    sell_reason = f"月RSI4={m_rsi4:.0f}<77"

                if sell_flag:
                    sell_price = close[i]
                    proceeds   = position_shares * sell_price
                    pl         = proceeds - (position_shares * position_buy_price)
                    ticker_cash += proceeds
                    trades.append({
                        "ticker":         ticker,
                        "buy_date":       position_buy_date,
                        "buy_price":      round(position_buy_price, 2),
                        "sell_date":      current_date.strftime("%Y-%m-%d"),
                        "sell_price":     round(sell_price, 2),
                        "shares":         position_shares,
                        "pl":             round(pl, 2),
                        "pl_pct":         round((sell_price - position_buy_price) / position_buy_price * 100, 2),
                        "sell_reason":    sell_reason,
                    })
                    equity_curve.append({
                        "date":   current_date.strftime("%Y-%m-%d"),
                        "equity": round(ticker_cash, 2),
                        "type":   "sell",
                        "ticker": ticker,
                    })
                    # Do NOT add to year_pl here — ticker_pl at end accumulates all P&L
                    position_shares    = 0
                    position_buy_price = None
                    position_buy_date  = None

            # Buy check
            if position_shares == 0 and d4_val >= 3 and adx_val > 20 and wr_val < -20:
                bonus = 0
                if rsi60_val > 57:                        bonus += 1
                if abs(w_vr - 150) < 50:                 bonus += 1
                if abs(m_vr - 150) < 50:                 bonus += 1
                if m_pdi1 > 50 and m_rsi4 > 77:          bonus += 1

                if bonus >= 1:
                    allocation = min(FIXED_ALLOCATION, ticker_cash)
                    if allocation > 0:
                        pos_shares = math.floor(allocation / close[i])
                        if pos_shares > 0:
                            cost = pos_shares * close[i]
                            ticker_cash -= cost
                            position_shares    = pos_shares
                            position_buy_price = close[i]
                            position_buy_date  = current_date.strftime("%Y-%m-%d")
                            equity_curve.append({
                                "date":   position_buy_date,
                                "equity": round(ticker_cash + pos_shares * close[i], 2),
                                "type":   "buy",
                                "ticker": ticker,
                            })

        # Year-end close (mark-to-market; ticker_cash absorbs proceeds)
        if position_shares > 0 and close[-1] != 0.0:
            sell_price = close[-1]
            proceeds   = position_shares * sell_price
            pl         = proceeds - (position_shares * position_buy_price)
            ticker_cash += proceeds
            trades.append({
                "ticker":      ticker,
                "buy_date":    position_buy_date,
                "buy_price":   round(position_buy_price, 2),
                "sell_date":   pd.Timestamp(dates[-1]).strftime("%Y-%m-%d") + " (年終)",
                "sell_price":  round(sell_price, 2),
                "shares":      position_shares,
                "pl":          round(pl, 2),
                "pl_pct":      round((sell_price - position_buy_price) / position_buy_price * 100, 2),
                "sell_reason": "年終結算",
            })

        ticker_pl = ticker_cash - FIXED_ALLOCATION
        year_pl += ticker_pl

    equity_curve.append({
        "date":   f"{year}-12-31",
        "equity": round(INITIAL_CAPITAL + year_pl, 2),
        "type":   "final",
        "ticker": "",
    })

    elapsed     = time.time() - t0
    total_deployed = n_tickers_simulated * FIXED_ALLOCATION
    # avg_return: average per-ticker return (year_pl / total capital deployed across all tickers)
    avg_return_pct = round(year_pl / total_deployed * 100, 2) if total_deployed > 0 else 0.0
    final_equity = INITIAL_CAPITAL + year_pl

    summary = {
        "year":               year,
        "stocks_simulated":   n_tickers_simulated,
        "stocks_with_signals": len(set(t["ticker"] for t in trades)),
        "trade_count":        len(trades),
        "total_pl":           round(year_pl, 2),
        "total_deployed":     total_deployed,
        "avg_return_pct":     avg_return_pct,
        "total_return_pct":   avg_return_pct,  # alias for dashboard compatibility
        "trade_win_rate":     round(len([t for t in trades if t["pl"] > 0]) / len(trades) * 100, 1) if trades else 0,
        "elapsed_s":          round(elapsed),
        "n_rows":             n_total,
    }

    if trades:
        output = {"trades": trades, "equity_curve": equity_curve}
        with open(os.path.join(OUT_DIR, f"{year}_trades.json"), "w", encoding="utf-8") as fout:
            json.dump(output, fout, indent=2, ensure_ascii=False)

    print(f"📅 {year}: {n_tickers_simulated} tickers, {len(trades)} trades, "
          f"PL={year_pl:+.0f}, AvgReturn={avg_return_pct:+.2f}% ({elapsed:.0f}s)",
          flush=True)

    del df
    gc.collect()
    return summary


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    years = list(range(2004, 2027))
    all_summary = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_year, y): y for y in years}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_summary.append(result)
                print(f"  ✓ {result['year']} done", flush=True)

    all_summary.sort(key=lambda s: s["year"])

    out_path = os.path.join(OUT_DIR, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("\n📊 年度資金模擬回測摘要")
    print(f"{'Year':>6} {'Tickers':>8} {'Trades':>8} {'PL':>14} {'AvgRet%':>8} {'WinRate':>8}")
    for s in all_summary:
        print(f"{s['year']:>6} {s['stocks_simulated']:>8} {s['trade_count']:>8} "
              f"{s['total_pl']:>+13.0f} {s['avg_return_pct']:>+7.2f}% {s['trade_win_rate']:>6.1f}%")

    if not all_summary:
        print("No results."); return

    total_pl        = sum(s["total_pl"] for s in all_summary)
    positive_years  = len([s for s in all_summary if s["total_return_pct"] > 0])
    negative_years  = len([s for s in all_summary if s["total_return_pct"] <= 0])
    best_year       = max(all_summary, key=lambda s: s["total_return_pct"])
    worst_year      = min(all_summary, key=lambda s: s["total_return_pct"])
    avg_annual_return = sum(s["total_return_pct"] for s in all_summary) / len(all_summary)
    avg_win_rate    = sum(s["trade_win_rate"]    for s in all_summary) / len(all_summary)

    print(f"\n{'='*60}")
    print(f"初始資金: {INITIAL_CAPITAL:>12,.0f}")
    print(f"各年起始資金: 1,000,000（各年獨立計算）")
    print(f"{'='*60}")
    print(f"總損益(加總):   {total_pl:>+12,.0f}")
    print(f"年均報酬率:     {avg_annual_return:>+10.2f}%")
    print(f"平均勝率:       {avg_win_rate:>10.1f}%")
    print(f"最佳年份:       {best_year['year']:>4} ({best_year['total_return_pct']:>+.2f}%)")
    print(f"最差年份:       {worst_year['year']:>4} ({worst_year['total_return_pct']:>+.2f}%)")
    print(f"正報酬年數:     {positive_years}/{len(all_summary)}")
    print(f"負報酬年數:     {negative_years}/{len(all_summary)}")
    print(f"{'='*60}")
    print(f"✅ 結果已儲存至 {out_path}")


if __name__ == "__main__":
    main()
