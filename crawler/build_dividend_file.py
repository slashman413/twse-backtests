#!/usr/bin/env python3
"""Build combined dividend CSV from per-day exdiv files + price data.

Per-day files: Raw/YYYY/YYYYMMDD_exdiv.csv  (Ticker, Close_ExDate)

Estimates Cash_Dividend from prev_close - Close_ExDate.

⚠️  KNOWN LIMITATION: This estimation is imprecise because market price
    movements on ex-date can mask or inflate the true dividend amount.
    Stock_Dividend (配股) is NOT captured here; for accurate dividends
    use twse_validator.py (yfinance) or import Goodinfo data.

Output: D:/TWSE-Data/Raw/_combined_dividends.csv
"""
import os
import numpy as np
import pandas as pd

RAW_DIR = os.environ.get("TWSE_RAW_DIR", "D:/TWSE-Data/Raw")
OUTPUT  = os.path.join(RAW_DIR, "_combined_dividends.csv")

# ── Load daily price data ────────────────────────────────────────────────────
print("📥 Loading daily price data...", flush=True)
year_dirs = sorted(d for d in os.listdir(RAW_DIR) if d.isdigit())
dfs = []
for year in year_dirs:
    merged = os.path.join(RAW_DIR, year, f"_{year}_daily.parquet")
    if os.path.exists(merged):
        dfs.append(pd.read_parquet(merged, columns=["Date", "Ticker", "Close"]))
daily = pd.concat(dfs, ignore_index=True)
daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
daily["Ticker"] = daily["Ticker"].astype(str).str.zfill(4)
daily.sort_values(["Ticker", "Date"], inplace=True)
print(f"   {len(daily):,} rows", flush=True)

# ── Build vectorised prev-close lookup ──────────────────────────────────────
# ticker_map: ticker → (dates_array, closes_array) sorted by date
ticker_groups = {t: grp[["Date","Close"]].reset_index(drop=True)
                 for t, grp in daily.groupby("Ticker", sort=False)}
print(f"   {len(ticker_groups)} tickers loaded", flush=True)

# ── Process per-day exdiv files ─────────────────────────────────────────────
dividend_rows = []
no_price = 0
skipped_up = 0
matched = 0

for year in year_dirs:
    year_dir = os.path.join(RAW_DIR, year)
    if not os.path.isdir(year_dir):
        continue
    for fname in sorted(os.listdir(year_dir)):
        if not fname.endswith("_exdiv.csv") or fname.startswith("_"):
            continue
        date_str = fname[:8]
        try:
            dt = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
        except Exception:
            continue
        edf = pd.read_csv(os.path.join(year_dir, fname))
        for _, row in edf.iterrows():
            t = str(row["Ticker"]).zfill(4)
            ex_close = float(row["Close_ExDate"])
            grp = ticker_groups.get(t)
            if grp is None:
                no_price += 1
                continue
            mask = grp["Date"] < dt
            if not mask.any():
                no_price += 1
                continue
            prev_close = float(grp.loc[mask, "Close"].iloc[-1])
            cash_div = prev_close - ex_close
            if cash_div <= 0:
                # Price rose on ex-date due to market movement — dividend underestimated.
                # Record a zero-dividend event so the adjuster skips this ex-date cleanly.
                skipped_up += 1
                continue
            dividend_rows.append({
                "Date":           dt,
                "Ticker":         t,
                "Cash_Dividend":  round(cash_div, 4),
                "Stock_Dividend": 0.0,
            })
            matched += 1

print(f"   Matched: {matched}, Skipped (price up on ex-date): {skipped_up}, "
      f"No prev price: {no_price}", flush=True)
if skipped_up > 0:
    print(f"⚠️  {skipped_up} ex-date events had price appreciation on ex-date — "
          f"Cash_Dividend underestimated. Consider using yfinance (twse_validator.py) "
          f"for accurate dividend data.", flush=True)

if dividend_rows:
    div_df = (pd.DataFrame(dividend_rows)
              .drop_duplicates(subset=["Ticker", "Date"])
              .sort_values(["Ticker", "Date"]))
    div_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"✅ Written {len(div_df)} dividend events → {OUTPUT}", flush=True)
else:
    print("❌ No events!", flush=True)
