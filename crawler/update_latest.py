"""
One-shot updater: crawl missing days → merge → adjust → scan → push.
Run from the project root or backtests/ directory.
"""
import os, sys, subprocess, time
from datetime import date, timedelta

_HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)

from twse_crawler import fetch_one_day, merge_year, _is_trading_day, _fmt_date, _sleep, log, RAW_DIR

YEAR = date.today().year


def missing_days(year: int) -> list[date]:
    import json
    prog_file = os.path.join(RAW_DIR, f"_progress_{year}.json")
    completed: set[str] = set()
    if os.path.exists(prog_file):
        with open(prog_file) as f:
            completed = set(json.load(f).get("completed_dates", []))

    start = date(year, 1, 1)
    end   = date.today()
    days  = []
    d = start
    while d <= end:
        if _is_trading_day(d) and _fmt_date(d) not in completed:
            days.append(d)
        d += timedelta(days=1)
    return days


def update_progress(year: int, ds: str):
    import json
    prog_file = os.path.join(RAW_DIR, f"_progress_{year}.json")
    completed: set[str] = set()
    data = {}
    if os.path.exists(prog_file):
        with open(prog_file) as f:
            data = json.load(f)
        completed = set(data.get("completed_dates", []))
    completed.add(ds)
    data["completed_dates"] = sorted(completed)
    data["last_date"] = ds
    with open(prog_file, "w") as f:
        import json as j
        j.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # Step 1: crawl missing days
    missing = missing_days(YEAR)
    if not missing:
        log.info("✅ 已是最新，無需爬取")
    else:
        log.info(f"📥 需爬取 {len(missing)} 天: {[d.isoformat() for d in missing]}")
        for i, d in enumerate(missing):
            result = fetch_one_day(d, save=True)
            if result["daily_rows"] > 0:
                update_progress(YEAR, _fmt_date(d))
                log.info(f"  ✅ {d}: {result['daily_rows']:,} 筆")
            else:
                log.warning(f"  ⚠️ {d}: 無資料（假日/停市）")
            if i < len(missing) - 1:
                _sleep()   # 2~5 min between requests

    # Step 2: merge year
    log.info(f"\n📦 合併 {YEAR} 年度檔...")
    merge_year(YEAR)

    # Step 3: adjust
    log.info("\n📐 執行調整程式...")
    adj_script = os.path.join(_HERE, "adjuster.py")
    r = subprocess.run([sys.executable, adj_script], cwd=_HERE)
    if r.returncode != 0:
        log.error("❌ adjuster.py 失敗")
        sys.exit(1)

    # Step 3b: re-partition adjusted_all.parquet → _temp (adjuster deletes _temp after merge)
    log.info("\n📂 重建 _temp 分年檔...")
    import gc
    import pyarrow.parquet as pq
    from datetime import datetime as dt2
    src = os.path.join("D:/TWSE-Data/Adjusted", "adjusted_all.parquet")
    dst = os.path.join("D:/TWSE-Data/Adjusted", "_temp")
    os.makedirs(dst, exist_ok=True)
    curr_yr = date.today().year
    for yr in [curr_yr - 1, curr_yr]:
        out = os.path.join(dst, f"{yr}.parquet")
        tbl = pq.read_table(src, filters=[("Date",">=",dt2(yr,1,1)),("Date","<",dt2(yr+1,1,1))])
        pq.write_table(tbl, out, compression="snappy")
        log.info(f"  {yr}: {len(tbl):,} rows")
        del tbl; gc.collect()

    # Step 4: scan signals
    log.info("\n🔍 掃描訊號...")
    scan_script = os.path.join(ROOT, "backtests", "scan_signals.py")
    r = subprocess.run([sys.executable, scan_script], cwd=ROOT)
    if r.returncode != 0:
        log.error("❌ scan_signals.py 失敗")
        sys.exit(1)

    # Step 5: git push
    log.info("\n🚀 推送至 GitHub Pages...")
    for cmd in [
        ["git", "add", "docs/data/signals.json"],
        ["git", "commit", "-m", f"update signals to {date.today().isoformat()}"],
        ["git", "push", "origin", "main"],
    ]:
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout or ""):
            log.warning(f"  ⚠️ {' '.join(cmd)} 回傳 {r.returncode}")

    log.info("\n✅ 全部完成！")


if __name__ == "__main__":
    main()
