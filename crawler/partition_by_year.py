"""Partition adjusted_all.parquet into year files for fast backtesting"""
import os, time, gc
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = "D:/TWSE-Data/Adjusted/adjusted_all.parquet"
DST = "D:/TWSE-Data/Adjusted/_temp_v5"
os.makedirs(DST, exist_ok=True)

from datetime import datetime

for year in range(2004, 2027):
    t0 = time.time()
    out = os.path.join(DST, f"{year}.parquet")
    if os.path.exists(out):
        sz = os.path.getsize(out) / (1024*1024)
        print(f"  {year}.parquet already exists ({sz:.0f}MB) — skipped", flush=True)
        continue

    # Use pq.read_table with row group filtering — pass actual timestamps
    start_ts = datetime(year, 1, 1)
    end_ts = datetime(year + 1, 1, 1) if year < 2026 else datetime(2027, 1, 1)

    table = pq.read_table(
        SRC,
        filters=[
            ("Date", ">=", start_ts),
            ("Date", "<", end_ts),
        ]
    )
    n = len(table)
    if n == 0:
        print(f"  {year}: no data", flush=True)
        continue

    pq.write_table(table, out, compression="zstd")
    sz = os.path.getsize(out) / (1024*1024)
    elapsed = time.time() - t0
    print(f"  ✅ {year}: {n:,} rows → {sz:.0f}MB ({elapsed:.0f}s)", flush=True)

    del table
    gc.collect()

print(f"✅ All done! Files in {DST}")
