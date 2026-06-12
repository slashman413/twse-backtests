#!/usr/bin/env python3
"""Fast adjuster v5 — same core as v4 but writes per-year from numpy arrays (avoids OOM)."""
import os, time
import numpy as np
import pandas as pd

RAW_DIR = "D:/TWSE-Data/Raw"
ADJ_DIR = "D:/TWSE-Data/Adjusted"
OUTPUT = os.path.join(ADJ_DIR, "adjusted_all.parquet")
TEMP_DIR = os.path.join(ADJ_DIR, "_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

t0 = time.time()

# ── Step 1: Load all daily data ──
print("📥 Loading daily data...", flush=True)
year_dirs = sorted(d for d in os.listdir(RAW_DIR) if d.isdigit())
dfs = []
for year in year_dirs:
    p = os.path.join(RAW_DIR, year, f"_{year}_daily.parquet")
    if os.path.exists(p):
        dfs.append(pd.read_parquet(p, columns=["Date","Ticker","Open","High","Low","Close","Volume"]))
daily = pd.concat(dfs, ignore_index=True)
daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
daily.sort_values(["Ticker","Date"], inplace=True)
daily.reset_index(drop=True, inplace=True)
n = len(daily)
print(f"   {n:,} rows, {daily['Ticker'].nunique()} tickers ({time.time()-t0:.0f}s)", flush=True)

# ── Step 2: Build exdiv dict ──
print("📥 Loading exdiv events...", flush=True)
exdiv = {}
for year in year_dirs:
    yd = os.path.join(RAW_DIR, year)
    for fn in os.listdir(yd):
        if not fn.endswith("_exdiv.csv") or fn.startswith("_"):
            continue
        ds = fn[:8]
        dt = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
        edf = pd.read_csv(os.path.join(yd, fn))
        for _, r in edf.iterrows():
            exdiv.setdefault(str(r["Ticker"]).zfill(4), {})[dt] = float(r["Close_ExDate"])
print(f"   {sum(len(v) for v in exdiv.values()):,} events ({time.time()-t0:.0f}s)", flush=True)

# ── Step 3: Extract arrays ──
tickers = daily["Ticker"].astype(str).str.zfill(4).values
dates = daily["Date"].values.astype("datetime64[D]")
opens = daily["Open"].values.astype(np.float64)
highs = daily["High"].values.astype(np.float64)
lows = daily["Low"].values.astype(np.float64)
closes = daily["Close"].values.astype(np.float64)
vols = daily["Volume"].values.astype(np.float64)

# ── Step 4: Process per ticker ──
print("📐 Processing tickers...", flush=True)
adj_o = np.empty(n, dtype=np.float64)
adj_h = np.empty(n, dtype=np.float64)
adj_l = np.empty(n, dtype=np.float64)
adj_c = np.empty(n, dtype=np.float64)
adj_v = np.empty(n, dtype=np.float64)
cf    = np.empty(n, dtype=np.float64)

change_pts = np.where(tickers[1:] != tickers[:-1])[0] + 1
bounds = np.concatenate([[0], change_pts, [n]])
total = len(bounds) - 1

for gi in range(total):
    s, e = bounds[gi], bounds[gi + 1]
    evts = exdiv.get(tickers[s])
    m = e - s
    grp_cf = np.ones(m, dtype=np.float64)
    grp_c = closes[s:e]
    cum = 1.0
    for j in range(m - 1, -1, -1):
        grp_cf[j] = cum
        if evts and j > 0:
            ts_str = pd.Timestamp(dates[s + j]).strftime("%Y-%m-%d")
            ex_c = evts.get(ts_str)
            if ex_c is not None:
                prev_c = float(grp_c[j - 1])
                if prev_c > 0:
                    cum *= min(ex_c / prev_c, 1.0)
    adj_o[s:e] = opens[s:e] * grp_cf
    adj_h[s:e] = highs[s:e] * grp_cf
    adj_l[s:e] = lows[s:e] * grp_cf
    adj_c[s:e] = closes[s:e] * grp_cf
    # Volume is divided by CumFactor: after a split/stock-dividend the share count
    # increases, so historical volume must scale up (opposite direction to price).
    safe_cf = np.where(grp_cf > 0, grp_cf, 1.0)
    adj_v[s:e] = (vols[s:e] / safe_cf).round(0)
    cf[s:e] = grp_cf

    if (gi + 1) % 2000 == 0 or gi == total - 1:
        print(f"   {gi+1}/{total} tickers ({time.time()-t0:.0f}s)", flush=True)

# ── Step 5: Write per-year temp files from arrays ──
print("📝 Writing per-year temp files...", flush=True)
# Use daily's Date column to find year boundaries
year_arr = daily["Date"].dt.year.values
unique_years = np.unique(year_arr)
for yr in unique_years:
    mask = year_arr == yr
    idx = np.where(mask)[0]
    out = pd.DataFrame({
        "Date": daily["Date"].iloc[idx].values,
        "Ticker": daily["Ticker"].iloc[idx].values,
        "Open": opens[idx],
        "High": highs[idx],
        "Low": lows[idx],
        "Close": closes[idx],
        "Volume": vols[idx],
        "Adj_Open": adj_o[idx],
        "Adj_High": adj_h[idx],
        "Adj_Low": adj_l[idx],
        "Adj_Close": adj_c[idx],
        "Adj_Volume": adj_v[idx].astype("int64"),
        "CumFactor": cf[idx],
    })
    out.to_parquet(os.path.join(TEMP_DIR, f"{yr}.parquet"), index=False, compression="snappy")
    print(f"   {yr}: {len(out):,} rows", flush=True)

# Cleanup big data
del daily, opens, highs, lows, closes, vols, adj_o, adj_h, adj_l, adj_c, adj_v, cf

# ── Step 6: Merge ──
print("📐 Merging...", flush=True)
temp_files = sorted(os.listdir(TEMP_DIR))
parts = [pd.read_parquet(os.path.join(TEMP_DIR, f)) for f in temp_files]
final = pd.concat(parts, ignore_index=True)
del parts
print(f"💾 Saving {len(final):,} rows...", flush=True)
final.to_parquet(OUTPUT, compression="snappy", index=False)

for f in temp_files:
    os.remove(os.path.join(TEMP_DIR, f))
os.rmdir(TEMP_DIR)

elapsed = time.time() - t0
print(f"✅ Done! {elapsed:.0f}s → {OUTPUT} ({os.path.getsize(OUTPUT)/1024/1024:.0f}MB)", flush=True)
