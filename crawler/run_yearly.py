"""TWSE 逐年爬蟲啟動腳本 — 背景執行用"""
import sys, os, json, time, random
from datetime import date, timedelta
sys.path.insert(0, "D:/TWSE-Data/Code")

from twse_crawler import (
    fetch_one_day, _sleep, _is_trading_day, _fmt_date,
    RAW_DIR, YEAR_SLEEP_MIN, YEAR_SLEEP_MAX,
    log,
)

YEAR_SLEEP = (YEAR_SLEEP_MIN, YEAR_SLEEP_MAX)  # 4~10 分

def run_all():
    start_year = 2004
    end_year = date.today().year  # 2026

    for year in range(start_year, end_year + 1):
        log.info(f"\n{'='*60}")
        log.info(f"📅  年度: {year}")
        log.info(f"{'='*60}")

        progress_file = os.path.join(RAW_DIR, f"_progress_{year}.json")

        # 載入進度
        completed: set[str] = set()
        if os.path.exists(progress_file):
            with open(progress_file) as f:
                prog = json.load(f)
            completed = set(prog.get("completed_dates", []))
            log.info(f"📋 已有 {len(completed)} 天完成")

        start = date(year, 1, 1)
        end = date(year, 12, 31)
        if end > date.today():
            end = date.today()

        # 檢查是否已完整
        total_trading = sum(1 for i in range((end - start).days + 1)
                            if (start + timedelta(days=i)).weekday() < 5)
        if len(completed) >= total_trading:
            log.info(f"✅ {year} 已完整 ({total_trading} 天)，跳過")
            if year < end_year:
                _sleep(YEAR_SLEEP)
            continue

        year_dir = os.path.join(RAW_DIR, str(year))
        os.makedirs(year_dir, exist_ok=True)

        current = start
        day_count = 0
        total_rows = 0
        exdiv_count = 0

        while current <= end:
            ds = _fmt_date(current)
            if ds in completed:
                current += timedelta(days=1)
                continue
            if not _is_trading_day(current):
                current += timedelta(days=1)
                continue

            result = fetch_one_day(current, save=True)
            if result["daily_rows"] > 0:
                day_count += 1
                total_rows += result["daily_rows"]
                exdiv_count += result["exdiv_count"]
                completed.add(ds)

                # 進度存檔
                with open(progress_file, "w") as f:
                    json.dump({
                        "year": year, "completed_dates": sorted(completed),
                        "total_days": day_count, "total_rows": total_rows,
                        "exdiv_events": exdiv_count, "last_date": ds,
                    }, f, ensure_ascii=False, indent=2)

            # 跨日 sleep 2~5 分
            _sleep()
            current += timedelta(days=1)

        # 年度完成
        log.info(f"✅ {year} DONE: {day_count}天, {total_rows:,}行, {exdiv_count}次除權息")

        # 合併該年
        log.info(f"📦 合併 {year} ...")
        try:
            import subprocess
            subprocess.run([
                sys.executable, "D:/TWSE-Data/Code/twse_crawler.py",
                "--merge", str(year),
            ], check=True, cwd="D:/TWSE-Data/Code")
        except Exception as e:
            log.error(f"合併失敗: {e}")

        # ── 年度後處理：還原權值 + 回測 ──
        adjuster_py = "D:/TWSE-Data/Code/twse_adjuster.py"
        backtest_py = "D:/Hermes-Agent/大飆股DNA台股篩選/backtest.py"

        # 還原權值 (掃描 Raw/ 輸出到 Adjusted/)
        if os.path.exists(adjuster_py):
            log.info(f"📐 計算 {year} 還原權值 ...")
            try:
                subprocess.run([
                    sys.executable, adjuster_py,
                ], check=True, cwd="D:/TWSE-Data/Code", timeout=1800)
                log.info(f"✅ {year} 還原權值完成")
            except Exception as e:
                log.error(f"還原權值失敗: {e}")

        # 歷史回測 (全量上市股票)
        if os.path.exists(backtest_py):
            log.info(f"📊 執行歷史回測 (2004~{year}) ...")
            try:
                report_path = (
                    f"D:/Hermes-Agent/大飆股DNA台股篩選/reports/"
                    f"backtest_until_{year}.txt"
                )
                subprocess.run([
                    sys.executable, backtest_py,
                    "--all",
                    "--start", "2004",
                    "--end", str(year),
                    "--output", report_path,
                ], check=True, cwd="D:/TWSE-Data/Code", timeout=7200)
                log.info(f"✅ 回測完成 (2004~{year}) → {report_path}")
            except Exception as e:
                log.error(f"回測失敗: {e}")

        # 跨年 sleep 4~10 分
        if year < end_year:
            mins = random.randint(YEAR_SLEEP[0] // 60, YEAR_SLEEP[1] // 60)
            log.info(f"\n⏳ 跨年休息 {mins} 分鐘 ...\n")
            _sleep(YEAR_SLEEP)

    # 全量合併
    log.info(f"\n{'='*60}")
    log.info(f"📦📦 全量合併 ...")
    try:
        import subprocess
        subprocess.run([
            sys.executable, "D:/TWSE-Data/Code/twse_crawler.py",
            "--merge-all",
        ], check=True, cwd="D:/TWSE-Data/Code")
    except Exception as e:
        log.error(f"全量合併失敗: {e}")

    log.info(f"\n{'='*60}")
    log.info(f"🎉 全部完成！")

if __name__ == "__main__":
    run_all()
