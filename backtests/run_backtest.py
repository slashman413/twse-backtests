"""
Backtest v8 — research-driven quality filters.

Changes vs v7 (based on Deep Research recommendations):
1. Volatility gate: if 0050 20-day realized vol > 25% (annualised), suppress new entries
   (targets 2016 Jan-Feb panic, 2011 EU-debt crisis — high-vol regimes kill breakouts)
2. RVOL ≥ 1.2: breakout day volume must be ≥ 1.2x its own 20-day average
   (dynamic filter — catches genuine breakout energy without static lot threshold)
3. ATR Ratio ≥ 1.3: breakout bar range must be ≥ 1.3x recent ATR baseline
   (filters low-energy fake breakouts)
4. RVOL bonus scoring: +1 if ≥1.5x, +2 if ≥2.5x (on top of existing score)

Unchanged from v7:
- 300-lot static floor, BULL = 0050 MACD≥3 AND ADX>20
- Hard stop-loss: -10% BULL, -7% ALERT/BEAR, MIN_SCORE ≥ 1
"""
import os, sys, time, json, gc, math
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Path config ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ADJ_TEMP_DIR = os.environ.get("ADJ_TEMP_DIR",
               os.path.join("D:/TWSE-Data/Adjusted", "_temp"))
OUT_DIR      = os.environ.get("BACKTEST_OUT_DIR",
               os.path.join(_HERE, "..", "docs", "data"))
CORE_DIR     = os.path.join(_HERE, "..", "core")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, CORE_DIR)

from indicators import macd_4arrows, dmi, wr, rsi

INITIAL_CAPITAL = 1_000_000
POSITION_SIZE   = 50_000      # Fixed capital per position
TOP_N_PER_DAY   = 5           # Max new buys per trading day
MARKET_PROXY    = "0050"      # Used for bull/crash detection

# Risk management thresholds
STOP_LOSS_BULL  = 0.90        # Hard stop -10% in BULL market
STOP_LOSS_WEAK  = 0.93        # Hard stop -7% in ALERT/BEAR market
MIN_SCORE       = 1           # Minimum bonus conditions met to enter
MKT_ADX_MIN     = 20          # 0050 ADX must exceed this for BULL classification
MIN_AVG_VOL_LOTS = 300        # Min 20-day avg daily volume in lots (張); Adj_Volume in shares ÷ 1000
MKT_VOL_GATE    = 0.25        # 0050 20-day realized vol > this → suppress new entries
RVOL_MIN        = 1.2         # Breakout day volume must be ≥ 1.2× its 20-day average
ATR_RATIO_MIN   = 1.0         # v9b: relaxed from 1.3 to reduce over-filtering in trend years


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _forward_fill_to_daily(ts: pd.Series, daily_index: pd.DatetimeIndex,
                            default: float) -> np.ndarray:
    """Reindex a weekly/monthly series onto daily dates with forward-fill."""
    if ts.empty:
        return np.full(len(daily_index), default, dtype=np.float32)
    return ts.reindex(daily_index, method="ffill").fillna(default).values.astype(np.float32)


def _compute_indicators(grp: pd.DataFrame):
    """Return per-day indicator arrays for one ticker. Returns None if too short."""
    grp = grp.sort_values("Date").reset_index(drop=True)
    n = len(grp)
    if n < 120:
        return None

    close = np.nan_to_num(grp["Adj_Close"].values.astype(np.float64), nan=0.0)
    high  = np.nan_to_num(grp["Adj_High"].values.astype(np.float64),  nan=0.0)
    low   = np.nan_to_num(grp["Adj_Low"].values.astype(np.float64),   nan=0.0)
    vol   = grp["Adj_Volume"].values.astype(np.float64)
    dates = pd.to_datetime(grp["Date"].values)

    cs = pd.Series(close, index=range(n))
    hs = pd.Series(high,  index=range(n))
    ls = pd.Series(low,   index=range(n))

    # Daily indicators
    d4      = np.nan_to_num(macd_4arrows(cs, 200, 209, 210)["arrows_count"].values, nan=0).astype(np.float32)
    adx_arr = np.nan_to_num(dmi(hs, ls, cs, 300)["adx"].values, nan=0).astype(np.float32)
    wr_arr  = np.nan_to_num(wr(hs, ls, cs, 50).values, nan=0).astype(np.float32)
    rsi60   = np.nan_to_num(rsi(cs, 60).values, nan=50).astype(np.float32)

    # 20-day breakout threshold: yesterday's 20-day high (shift=1 → no look-ahead)
    high20 = pd.Series(close).rolling(20).max().shift(1).values.astype(np.float32)

    # 20-day avg volume in lots (張): Adj_Volume in shares ÷ 1000, shifted to avoid look-ahead
    avg_vol_lots = (pd.Series(vol).rolling(20).mean().shift(1) / 1000.0).values.astype(np.float32)

    # ATR ratio: today's ATR(14) ÷ 20-day avg of ATR(14), shifted to avoid look-ahead
    prev_cl = pd.Series(close).shift(1)
    tr = pd.concat([
        pd.Series(high) - pd.Series(low),
        (pd.Series(high) - prev_cl).abs(),
        (pd.Series(low)  - prev_cl).abs(),
    ], axis=1).max(axis=1)
    atr14     = tr.rolling(14).mean()
    atr_ratio = (atr14 / atr14.rolling(20).mean().shift(1)).fillna(0).values.astype(np.float32)

    # Multi-timeframe — compute on a date-indexed DataFrame
    daily_df = pd.DataFrame(
        {"Close": close, "High": high, "Low": low, "Volume": vol},
        index=dates
    )
    dti = pd.DatetimeIndex(dates)

    weekly  = daily_df.resample("W").agg({"Close":"last","High":"max","Low":"min","Volume":"sum"}).dropna()
    monthly = daily_df.resample("ME").agg({"Close":"last","High":"max","Low":"min","Volume":"sum"}).dropna()

    # Weekly VR → forward-fill to daily
    if len(weekly) > 3:
        wu  = weekly["Close"].diff() > 0
        wd  = weekly["Close"].diff() < 0
        wvr = (100.0 * (weekly["Volume"]*wu).rolling(2).sum()
               / (weekly["Volume"]*wd).rolling(2).sum().replace(0, np.nan)).fillna(0.0)
    else:
        wvr = pd.Series(dtype=float)
    w_vr_d = _forward_fill_to_daily(wvr, dti, 0.0)

    # Monthly VR → forward-fill
    if len(monthly) > 3:
        mu  = monthly["Close"].diff() > 0
        md  = monthly["Close"].diff() < 0
        mvr = (100.0 * (monthly["Volume"]*mu).rolling(2).sum()
               / (monthly["Volume"]*md).rolling(2).sum().replace(0, np.nan)).fillna(0.0)
    else:
        mvr = pd.Series(dtype=float)
    m_vr_d = _forward_fill_to_daily(mvr, dti, 0.0)

    # Monthly RSI4 + DMI+DI1 → forward-fill
    if len(monthly) > 14:
        mc       = pd.Series(monthly["Close"].values, index=range(len(monthly)))
        m_rsi4_s = rsi(mc, 4)
        m_rsi4_s.index = monthly.index
        _dm      = dmi(
            pd.Series(monthly["High"].values, index=range(len(monthly))),
            pd.Series(monthly["Low"].values,  index=range(len(monthly))),
            mc, period=1
        )
        m_pdi1_s = _dm["plus_di"]; m_pdi1_s.index = monthly.index
    else:
        m_rsi4_s = pd.Series(dtype=float)
        m_pdi1_s = pd.Series(dtype=float)

    m_rsi4_d = _forward_fill_to_daily(m_rsi4_s, dti, 50.0)
    m_pdi1_d = _forward_fill_to_daily(m_pdi1_s, dti, 0.0)

    return {
        "dates":        dates,
        "close":        close.astype(np.float32),
        "vol":          vol.astype(np.float32),   # raw daily volume (shares)
        "d4":           d4,
        "adx":          adx_arr,
        "wr":           wr_arr,
        "rsi60":        rsi60,
        "high20":       high20,
        "avg_vol_lots": avg_vol_lots,
        "atr_ratio":    atr_ratio,
        "w_vr_d":       w_vr_d,
        "m_vr_d":       m_vr_d,
        "m_rsi4_d":     m_rsi4_d,
        "m_pdi1_d":     m_pdi1_d,
    }


def _market_signal_series(mkt_info: dict) -> dict:
    """Build date → signal mapping from market proxy indicators.

    BULL   : d4 >= 3  AND  ADX > MKT_ADX_MIN  (confirmed trend strength)
    ALERT  : d4 in {1, 2}  OR  (d4 >= 3 but ADX ≤ MKT_ADX_MIN)
    BEAR   : d4 == 0
    CRASH  : d4 == 0  AND  close dropped ≥ 3% vs 3 days ago
    """
    dates = mkt_info["dates"]
    close = mkt_info["close"]
    d4    = mkt_info["d4"]
    adx   = mkt_info["adx"]
    sig   = {}
    n = len(dates)
    for i in range(n):
        arrows = int(d4[i])
        adx_v  = float(adx[i])
        crash = False
        if arrows == 0 and i >= 3:
            drop = (close[i] - close[i-3]) / max(close[i-3], 1e-9)
            crash = drop <= -0.03
        if crash:
            s = "CRASH"
        elif arrows >= 3 and adx_v > MKT_ADX_MIN:
            s = "BULL"
        elif arrows >= 1:
            s = "ALERT"
        else:
            s = "BEAR"
        sig[dates[i]] = s
    return sig


# ── Main year processor ──────────────────────────────────────────────────────

def process_year(year: int):
    t0 = time.time()
    f = os.path.join(ADJ_TEMP_DIR, f"{year}.parquet")
    if not os.path.exists(f):
        return None

    df = pd.read_parquet(f)
    df["Date"]   = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.zfill(4)
    for col in ["Adj_Close", "Adj_High", "Adj_Low"]:
        df = df[df[col].notna()]
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # ── Market proxy ──────────────────────────────────────────────────────────
    mkt_grp = df[df["Ticker"] == MARKET_PROXY]
    if len(mkt_grp) >= 120:
        mkt_info = _compute_indicators(mkt_grp)
        mkt_signals = _market_signal_series(mkt_info) if mkt_info else {}
        # Volatility gate: 20-day realized vol > MKT_VOL_GATE → suppress new entries
        mkt_close_s  = pd.Series(mkt_info["close"], index=pd.DatetimeIndex(mkt_info["dates"]))
        mkt_rvol_20d = mkt_close_s.pct_change().rolling(20).std() * np.sqrt(252)
        mkt_high_vol = {d: (not np.isnan(v) and v > MKT_VOL_GATE)
                        for d, v in mkt_rvol_20d.items()}
    else:
        mkt_signals  = {}
        mkt_high_vol = {}

    # ── Pre-compute indicators per ticker ─────────────────────────────────────
    print(f"  {year}: computing indicators for {df['Ticker'].nunique()} tickers…", flush=True)
    ticker_info = {}
    for ticker, grp in df.groupby("Ticker", sort=False):
        info = _compute_indicators(grp)
        if info is not None:
            info["date_to_idx"] = {d: i for i, d in enumerate(info["dates"])}
            ticker_info[ticker] = info

    # ── Pre-build candidates per trading date ─────────────────────────────────
    # For each (ticker, day) that passes B1+B2+B3+breakout, register as candidate.
    # This avoids an inner loop over all tickers in the simulation.
    print(f"  {year}: building candidate index…", flush=True)
    candidates_by_date: dict[pd.Timestamp, list] = {}

    for ticker, info in ticker_info.items():
        n = len(info["dates"])
        cl   = info["close"]
        d4   = info["d4"]
        adx  = info["adx"]
        wr_  = info["wr"]
        h20  = info["high20"]
        avl  = info["avg_vol_lots"]
        atr_ = info["atr_ratio"]
        vol_ = info["vol"]

        # Vectorised pre-filter (numpy) to avoid pure Python per-row loop
        warmup   = 120
        rvol_arr = (vol_ / 1000.0) / (avl + 1e-9)
        valid_mask = (
            (d4[warmup:]          >= 3)           &
            (adx[warmup:]         >  20)           &
            (wr_[warmup:]         < -20)           &
            (~np.isnan(h20[warmup:]))              &
            (cl[warmup:]          > h20[warmup:])  &
            (cl[warmup:]          > 0)             &
            (avl[warmup:]         >= MIN_AVG_VOL_LOTS) &
            (rvol_arr[warmup:]    >= RVOL_MIN)     &
            (atr_[warmup:]        >= ATR_RATIO_MIN)
        )
        for rel_i in np.where(valid_mask)[0]:
            i = rel_i + warmup
            day = info["dates"][i]

            rsi60_v = float(info["rsi60"][i])
            w_vr_v  = float(info["w_vr_d"][i])
            m_vr_v  = float(info["m_vr_d"][i])
            m_rsi4_v = float(info["m_rsi4_d"][i])
            m_pdi1_v = float(info["m_pdi1_d"][i])

            score = int(rsi60_v > 57) + int(abs(w_vr_v - 150) < 50) + \
                    int(abs(m_vr_v - 150) < 50) + int(m_pdi1_v > 50 and m_rsi4_v > 77)

            rvol_v = float(vol_[i]) / 1000.0 / max(float(avl[i]), 1e-9)
            score += int(rvol_v >= 1.5) + int(rvol_v >= 2.5)

            if score < MIN_SCORE:
                continue  # Require at least 1 bonus condition

            breakout_pct = float((cl[i] - h20[i]) / h20[i] * 100)

            if day not in candidates_by_date:
                candidates_by_date[day] = []
            candidates_by_date[day].append({
                "ticker":       ticker,
                "close":        float(cl[i]),
                "score":        score,
                "breakout_pct": breakout_pct,
                "m_rsi4":       m_rsi4_v,
            })

    # Sort each day's list once (score DESC, breakout_pct DESC)
    for day_list in candidates_by_date.values():
        day_list.sort(key=lambda x: (-x["score"], -x["breakout_pct"]))

    # ── Simulation ────────────────────────────────────────────────────────────
    all_dates = sorted(df["Date"].unique())

    cash       = float(INITIAL_CAPITAL)
    positions  = {}   # ticker → {shares, buy_price, buy_date, cost}
    trades     = []
    missed     = []
    equity_curve = []

    def portfolio_value(day: pd.Timestamp) -> float:
        mval = 0.0
        for tkr, pos in positions.items():
            info = ticker_info.get(tkr)
            if info is None:
                mval += pos["cost"]
                continue
            idx = info["date_to_idx"].get(day)
            px  = float(info["close"][idx]) if idx is not None else pos["buy_price"]
            mval += pos["shares"] * (px if px > 0 else pos["buy_price"])
        return cash + mval

    for raw_day in all_dates:
        day = pd.Timestamp(raw_day)
        mkt = mkt_signals.get(day, "BULL")

        # ── CRASH: liquidate everything ───────────────────────────────────────
        if mkt == "CRASH":
            for tkr, pos in list(positions.items()):
                info = ticker_info.get(tkr)
                idx  = info["date_to_idx"].get(day) if info else None
                sell_price = float(info["close"][idx]) if (idx is not None and info["close"][idx] > 0) \
                             else pos["buy_price"]
                proceeds = pos["shares"] * sell_price
                pl = proceeds - pos["cost"]
                cash += proceeds
                trades.append({
                    "ticker":      tkr,
                    "buy_date":    pos["buy_date"],
                    "buy_price":   round(pos["buy_price"], 2),
                    "sell_date":   day.strftime("%Y-%m-%d"),
                    "sell_price":  round(sell_price, 2),
                    "shares":      pos["shares"],
                    "pl":          round(pl, 2),
                    "pl_pct":      round((sell_price - pos["buy_price"]) / pos["buy_price"] * 100, 2),
                    "sell_reason": "大盤CRASH出清",
                })
            positions.clear()
            equity_curve.append({"date": day.strftime("%Y-%m-%d"),
                                  "equity": round(cash, 2), "type": "crash"})
            continue

        # ── Regular sell: hard stop-loss / monthly RSI4 ──────────────────────
        sl = STOP_LOSS_BULL if mkt == "BULL" else STOP_LOSS_WEAK

        for tkr in list(positions.keys()):
            info = ticker_info.get(tkr)
            if info is None:
                continue
            idx = info["date_to_idx"].get(day)
            if idx is None:
                continue
            sell_price = float(info["close"][idx])
            if sell_price == 0:
                continue

            pos    = positions[tkr]
            buy_px = pos["buy_price"]
            m_rsi4 = float(info["m_rsi4_d"][idx])

            sell_flag   = False
            sell_reason = ""

            if sell_price <= buy_px * sl:
                sell_flag   = True
                sell_reason = f"止損{(sell_price/buy_px-1)*100:.1f}%"
            elif m_rsi4 < 77:
                sell_flag   = True
                sell_reason = f"月RSI4={m_rsi4:.0f}<77"

            if not sell_flag:
                continue

            pos = positions.pop(tkr)
            proceeds = pos["shares"] * sell_price
            pl = proceeds - pos["cost"]
            cash += proceeds
            trades.append({
                "ticker":      tkr,
                "buy_date":    pos["buy_date"],
                "buy_price":   round(buy_px, 2),
                "sell_date":   day.strftime("%Y-%m-%d"),
                "sell_price":  round(sell_price, 2),
                "shares":      pos["shares"],
                "pl":          round(pl, 2),
                "pl_pct":      round((sell_price - buy_px) / buy_px * 100, 2),
                "sell_reason": sell_reason,
            })

        # ── Buy: BULL market, no high-vol regime ────────────────────────────
        if mkt == "BULL" and not mkt_high_vol.get(day, False):
            today_candidates = candidates_by_date.get(day, [])
            buys_today = 0
            for c in today_candidates:
                if buys_today >= TOP_N_PER_DAY:
                    break
                tkr = c["ticker"]
                if tkr in positions:
                    continue  # already holding
                if cash < POSITION_SIZE:
                    missed.append({
                        "ticker":       tkr,
                        "date":         day.strftime("%Y-%m-%d"),
                        "price":        round(c["close"], 2),
                        "score":        c["score"],
                        "breakout_pct": round(c["breakout_pct"], 2),
                        "reason":       "資金不足",
                    })
                    continue
                alloc  = min(POSITION_SIZE, cash)
                shares = math.floor(alloc / c["close"])
                if shares <= 0:
                    continue
                cost = shares * c["close"]
                cash -= cost
                positions[tkr] = {
                    "shares":    shares,
                    "buy_price": c["close"],
                    "buy_date":  day.strftime("%Y-%m-%d"),
                    "cost":      cost,
                }
                buys_today += 1

        # Daily equity snapshot (every 5 days to keep JSON size manageable)
        if raw_day in all_dates[::5]:
            equity_curve.append({
                "date":   day.strftime("%Y-%m-%d"),
                "equity": round(portfolio_value(day), 2),
                "type":   "daily",
            })

    # ── Year-end mark-to-market ───────────────────────────────────────────────
    last_day = pd.Timestamp(all_dates[-1])
    for tkr, pos in list(positions.items()):
        info = ticker_info.get(tkr)
        last_close = float(info["close"][-1]) if info else 0.0
        if last_close == 0:
            last_close = pos["buy_price"]
        proceeds = pos["shares"] * last_close
        pl = proceeds - pos["cost"]
        cash += proceeds
        trades.append({
            "ticker":      tkr,
            "buy_date":    pos["buy_date"],
            "buy_price":   round(pos["buy_price"], 2),
            "sell_date":   last_day.strftime("%Y-%m-%d") + " (年終)",
            "sell_price":  round(last_close, 2),
            "shares":      pos["shares"],
            "pl":          round(pl, 2),
            "pl_pct":      round((last_close - pos["buy_price"]) / pos["buy_price"] * 100, 2),
            "sell_reason": "年終結算",
        })
    positions.clear()

    equity_curve.append({
        "date":   f"{year}-12-31",
        "equity": round(cash, 2),
        "type":   "final",
    })

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    year_pl = cash - INITIAL_CAPITAL
    return_pct = round(year_pl / INITIAL_CAPITAL * 100, 2)

    summary = {
        "year":                 year,
        "stocks_simulated":     len(ticker_info),
        "stocks_with_signals":  len(set(t["ticker"] for t in trades)),
        "trade_count":          len(trades),
        "missed_count":         len(missed),
        "total_pl":             round(year_pl, 2),
        "total_deployed":       INITIAL_CAPITAL,
        "avg_return_pct":       return_pct,
        "total_return_pct":     return_pct,
        "trade_win_rate":       round(
            len([t for t in trades if t["pl"] > 0]) / len(trades) * 100, 1
        ) if trades else 0.0,
        "elapsed_s":            round(elapsed),
    }

    if trades or missed:
        out = {"trades": trades, "equity_curve": equity_curve, "missed_trades": missed}
        with open(os.path.join(OUT_DIR, f"{year}_trades.json"), "w", encoding="utf-8") as fout:
            json.dump(out, fout, indent=2, ensure_ascii=False)

    print(
        f"📅 {year}: {len(ticker_info)} tickers | {len(trades)} trades | "
        f"{len(missed)} missed | PL={year_pl:+.0f} | Return={return_pct:+.2f}% ({elapsed:.0f}s)",
        flush=True,
    )

    del df, ticker_info, candidates_by_date
    gc.collect()
    return summary


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    years = list(range(2004, 2027))
    all_summary = []

    with ProcessPoolExecutor(max_workers=3) as executor:
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

    if not all_summary:
        print("No results."); return

    print(f"\n{'='*70}")
    print(f"{'Year':>6} {'Tickers':>8} {'Trades':>8} {'Missed':>7} "
          f"{'PL':>14} {'Return%':>8} {'WinRate':>8}")
    print(f"{'─'*70}")
    for s in all_summary:
        cls = "+" if s["total_return_pct"] > 0 else ""
        print(f"{s['year']:>6} {s['stocks_simulated']:>8} {s['trade_count']:>8} "
              f"{s['missed_count']:>7} {s['total_pl']:>+13.0f} "
              f"{s['total_return_pct']:>+7.2f}% {s['trade_win_rate']:>6.1f}%")

    pos_years = [s for s in all_summary if s["total_return_pct"] > 0]
    neg_years = [s for s in all_summary if s["total_return_pct"] <= 0]
    best  = max(all_summary, key=lambda s: s["total_return_pct"])
    worst = min(all_summary, key=lambda s: s["total_return_pct"])
    avg_ret  = sum(s["total_return_pct"] for s in all_summary) / len(all_summary)
    avg_win  = sum(s["trade_win_rate"]   for s in all_summary) / len(all_summary)

    print(f"{'='*70}")
    print(f"起始資金(各年獨立):  {INITIAL_CAPITAL:>12,.0f}")
    print(f"每筆進場資金:        {POSITION_SIZE:>12,.0f}")
    print(f"每日最多買進:        {TOP_N_PER_DAY:>12} 檔")
    print(f"{'─'*70}")
    print(f"年均報酬率:         {avg_ret:>+10.2f}%")
    print(f"平均勝率:           {avg_win:>10.1f}%")
    print(f"最佳年份:           {best['year']:>4}  ({best['total_return_pct']:>+.2f}%)")
    print(f"最差年份:           {worst['year']:>4}  ({worst['total_return_pct']:>+.2f}%)")
    print(f"正報酬年數:         {len(pos_years):>4}/{len(all_summary)}")
    print(f"負報酬年數:         {len(neg_years):>4}/{len(all_summary)}")
    print(f"{'='*70}")
    print(f"✅ 結果已儲存至 {out_path}")


if __name__ == "__main__":
    main()
