"""
Signal scanner — find stocks with active entry signals on the latest trading date.

Outputs docs/data/signals.json with:
  {
    "scan_date": "2026-06-09",
    "market_regime": "BULL",
    "market_vol_gate": false,
    "entry_signals": [...],
    "exit_signals": []
  }
"""
import os, sys, json
import pandas as pd
import numpy as np
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
ADJ_TEMP_DIR = os.environ.get("ADJ_TEMP_DIR",
               os.path.join("D:/TWSE-Data/Adjusted", "_temp"))
ADJ_ALL_FILE = os.environ.get("ADJ_FILE",
               os.path.join("D:/TWSE-Data/Adjusted", "adjusted_all.parquet"))
OUT_DIR      = os.environ.get("BACKTEST_OUT_DIR",
               os.path.join(_HERE, "..", "docs", "data"))
CORE_DIR     = os.path.join(_HERE, "..", "core")
NAMES_FILE   = os.path.join(OUT_DIR, "stock_names.json")
sys.path.insert(0, CORE_DIR)

from indicators import macd_4arrows, dmi, wr, rsi

MARKET_PROXY     = "0050"
MKT_ADX_MIN      = 20
MKT_VOL_GATE     = 0.25
MIN_AVG_VOL_LOTS = 300
RVOL_MIN         = 1.2
ATR_RATIO_MIN    = 1.0
STOP_LOSS_BULL   = 0.90
STOP_LOSS_WEAK   = 0.93


def _forward_fill_to_daily(ts: pd.Series, daily_index: pd.DatetimeIndex, default: float) -> np.ndarray:
    if ts.empty:
        return np.full(len(daily_index), default, dtype=np.float32)
    return ts.reindex(daily_index, method="ffill").fillna(default).values.astype(np.float32)


def _compute_indicators(grp: pd.DataFrame):
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

    d4      = np.nan_to_num(macd_4arrows(cs, 12, 26, 9)["arrows_count"].values, nan=0).astype(np.float32)
    adx_arr = np.nan_to_num(dmi(hs, ls, cs, 14)["adx"].values, nan=0).astype(np.float32)
    wr_arr  = np.nan_to_num(wr(hs, ls, cs, 50).values, nan=0).astype(np.float32)
    rsi60   = np.nan_to_num(rsi(cs, 60).values, nan=50).astype(np.float32)

    high20       = pd.Series(close).rolling(20).max().shift(1).values.astype(np.float32)
    avg_vol_lots = (pd.Series(vol).rolling(20).mean().shift(1) / 1000.0).values.astype(np.float32)

    prev_cl  = pd.Series(close).shift(1)
    tr = pd.concat([
        pd.Series(high) - pd.Series(low),
        (pd.Series(high) - prev_cl).abs(),
        (pd.Series(low)  - prev_cl).abs(),
    ], axis=1).max(axis=1)
    atr14     = tr.rolling(14).mean()
    atr_ratio = (atr14 / atr14.rolling(20).mean().shift(1)).fillna(0).values.astype(np.float32)

    daily_df = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": vol}, index=dates)
    dti = pd.DatetimeIndex(dates)

    monthly = daily_df.resample("ME").agg({"Close":"last","High":"max","Low":"min","Volume":"sum"}).dropna()
    if len(monthly) > 14:
        mc       = pd.Series(monthly["Close"].values, index=range(len(monthly)))
        m_rsi4_s = rsi(mc, 4)
        m_rsi4_s.index = monthly.index
    else:
        m_rsi4_s = pd.Series(dtype=float)
    m_rsi4_d = _forward_fill_to_daily(m_rsi4_s, dti, 50.0)

    return {
        "dates":        dates,
        "close":        close.astype(np.float32),
        "vol":          vol.astype(np.float32),
        "d4":           d4,
        "adx":          adx_arr,
        "wr":           wr_arr,
        "rsi60":        rsi60,
        "high20":       high20,
        "avg_vol_lots": avg_vol_lots,
        "atr_ratio":    atr_ratio,
        "m_rsi4_d":     m_rsi4_d,
    }


def _load_adj_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load last 2 years of adjusted data.

    Primary source: _temp/{year}.parquet (fast, written by adjuster).
    Fallback: adjusted_all.parquet filtered to last 2 years.
    Returns (df, scan_date).
    """
    COLS = ["Date", "Ticker", "Adj_Close", "Adj_High", "Adj_Low", "Adj_Volume"]

    curr_year = date.today().year
    f_curr = os.path.join(ADJ_TEMP_DIR, f"{curr_year}.parquet")
    f_prev = os.path.join(ADJ_TEMP_DIR, f"{curr_year-1}.parquet")
    use_temp = os.path.exists(f_curr) or os.path.exists(f_prev)

    if use_temp:
        # Fast path: read per-year slices from _temp
        if not os.path.exists(f_curr):
            curr_year -= 1
            f_curr = os.path.join(ADJ_TEMP_DIR, f"{curr_year}.parquet")
            f_prev = os.path.join(ADJ_TEMP_DIR, f"{curr_year-1}.parquet")
        print(f"Loading {curr_year} data from _temp…")
        df_curr = pd.read_parquet(f_curr, columns=COLS)
        df_curr["Date"]   = pd.to_datetime(df_curr["Date"])
        df_curr["Ticker"] = df_curr["Ticker"].astype(str).str.zfill(4)
        df_curr = df_curr[df_curr["Ticker"].str.len() <= 4]
        scan_date = df_curr["Date"].max()
        if os.path.exists(f_prev):
            df_prev = pd.read_parquet(f_prev, columns=COLS)
            df_prev["Date"]   = pd.to_datetime(df_prev["Date"])
            df_prev["Ticker"] = df_prev["Ticker"].astype(str).str.zfill(4)
            df_prev = df_prev[df_prev["Ticker"].str.len() <= 4]
            df = pd.concat([df_prev, df_curr], ignore_index=True)
        else:
            df = df_curr
    else:
        # Fallback: read adjusted_all.parquet (2-year window)
        print(f"_temp not found — falling back to {ADJ_ALL_FILE}…")
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=2)
        df = pd.read_parquet(ADJ_ALL_FILE, columns=COLS)
        df["Date"]   = pd.to_datetime(df["Date"])
        df["Ticker"] = df["Ticker"].astype(str).str.zfill(4)
        df = df[(df["Date"] >= cutoff) & (df["Ticker"].str.len() <= 4)]
        scan_date = df["Date"].max()

    # Deduplicate (raw data may have duplicate rows from crawler re-runs)
    df.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
    for col in ["Adj_Close", "Adj_High", "Adj_Low"]:
        df = df[df[col].notna()]
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df, scan_date


def main():
    df, scan_date = _load_adj_data()
    print(f"Scan date: {scan_date.date()}")

    # Market proxy
    mkt_grp  = df[df["Ticker"] == MARKET_PROXY]
    mkt_info = _compute_indicators(mkt_grp) if len(mkt_grp) >= 120 else None

    market_regime   = "UNKNOWN"
    market_vol_gate = False
    if mkt_info:
        dates_arr = list(mkt_info["dates"])
        close_arr = mkt_info["close"]
        d4_arr    = mkt_info["d4"]
        adx_arr   = mkt_info["adx"]
        try:
            idx = dates_arr.index(scan_date)
        except ValueError:
            idx = len(dates_arr) - 1

        arrows = int(d4_arr[idx])
        adx_v  = float(adx_arr[idx])
        crash  = False
        if arrows == 0 and idx >= 3:
            drop = (close_arr[idx] - close_arr[idx-3]) / max(close_arr[idx-3], 1e-9)
            crash = drop <= -0.03
        if crash:
            market_regime = "CRASH"
        elif arrows >= 3 and adx_v > MKT_ADX_MIN:
            market_regime = "BULL"
        elif arrows >= 1:
            market_regime = "ALERT"
        else:
            market_regime = "BEAR"

        mkt_close_s  = pd.Series(mkt_info["close"], index=pd.DatetimeIndex(mkt_info["dates"]))
        rv = mkt_close_s.pct_change().rolling(20).std() * np.sqrt(252)
        rv_tail = rv[~rv.index.duplicated(keep="last")]
        if scan_date in rv_tail.index:
            v = float(rv_tail.loc[scan_date])
            market_vol_gate = bool(not np.isnan(v) and v > MKT_VOL_GATE)

    # Load stock names
    stock_names = {}
    if os.path.exists(NAMES_FILE):
        with open(NAMES_FILE) as f:
            stock_names = json.load(f)

    # Compute indicators and scan
    print(f"Computing indicators for {df['Ticker'].nunique()} tickers…")
    entry_signals = []

    for ticker, grp in df.groupby("Ticker", sort=False):
        if ticker == MARKET_PROXY:
            continue
        info = _compute_indicators(grp)
        if info is None:
            continue

        try:
            dates_list = list(info["dates"])
            idx = dates_list.index(scan_date)
        except ValueError:
            continue  # ticker has no data on scan_date

        cl   = float(info["close"][idx])
        d4   = float(info["d4"][idx])
        adx  = float(info["adx"][idx])
        wr_v = float(info["wr"][idx])
        h20  = float(info["high20"][idx]) if not np.isnan(info["high20"][idx]) else None
        avl  = float(info["avg_vol_lots"][idx])
        atr  = float(info["atr_ratio"][idx])
        vol  = float(info["vol"][idx])
        rsi4 = float(info["m_rsi4_d"][idx])

        rvol = (vol / 1000.0) / (avl + 1e-9)

        entry_ok = (
            d4 >= 3 and adx > 20 and wr_v < -20 and
            h20 is not None and cl > h20 and cl > 0 and
            avl >= MIN_AVG_VOL_LOTS and rvol >= RVOL_MIN and atr >= ATR_RATIO_MIN
        )

        if entry_ok:
            breakout_pct = (cl / h20 - 1) * 100 if h20 and h20 > 0 else 0
            score = 0
            if rvol >= 2.0:   score += 1
            if rsi4 < 50:     score += 1
            if adx > 30:      score += 1

            entry_signals.append({
                "ticker":       ticker,
                "name":         stock_names.get(ticker, ""),
                "price":        round(cl, 2),
                "d4":           int(d4),
                "adx":          round(adx, 1),
                "wr":           round(wr_v, 1),
                "rvol":         round(rvol, 2),
                "atr_ratio":    round(atr, 2),
                "avg_vol_lots": int(avl),
                "breakout_pct": round(breakout_pct, 2),
                "m_rsi4":       round(rsi4, 1),
                "score":        score,
            })

    entry_signals.sort(key=lambda x: (-x["score"], -x["breakout_pct"]))
    print(f"Found {len(entry_signals)} entry signals.")

    out = {
        "scan_date":        scan_date.strftime("%Y-%m-%d"),
        "market_regime":    market_regime,
        "market_vol_gate":  market_vol_gate,
        "entry_signals":    entry_signals,
    }
    out_path = os.path.join(OUT_DIR, "signals.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved to {out_path}")

    # Auto-generate chart data for signal stocks
    import subprocess
    chart_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_chart_data.py")
    subprocess.run([sys.executable, chart_script])


if __name__ == "__main__":
    main()
