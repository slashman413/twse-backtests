"""
TWSE 資料爬蟲 — 台灣證券交易所資料抓取模組
============================================

策略：
  ⚠️ 每次請求後隨機 sleep 2~5 分鐘，避免 IP 被封鎖
  ⚠️ 支援 retry + exponential backoff
  ⚠️ 僅交易日才發請求（skip 週末）

API 格式 (2025+)：
  MI_INDEX 資料在 response['tables'][8]（每日收盤行情全部）
  除權息標記在 漲跌(+/-) 欄位 = '<p>X</p>'

儲存路徑：
  Raw/{year}/                    — 每年分目錄
    {YYYYMMDD}_daily.parquet     — 該日全市場 OHLCV
    {YYYYMMDD}_exdiv.csv         — 該日除權息標記清單
  Raw/_progress.json             — 進度檔 (支援中斷續爬)

使用方法：
    python twse_crawler.py --year 2026
    python twse_crawler.py --year-range 2004 2026
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import logging
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

# ── 設定 ──────────────────────────────────────────────────────
DATA_DIR = os.environ.get("TWSE_DATA_DIR", "D:/TWSE-Data")
RAW_DIR = os.path.join(DATA_DIR, "Raw")
LOG_FILE = os.path.join(DATA_DIR, "logs", "crawler.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

TWSE_DAILY_URL = (
    "https://www.twse.com.tw/exchangeReport/MI_INDEX"
    "?response=json&date={date_str}&type=ALL"
)

# 請求間隔（秒）— TWSE 會封鎖過快的 IP，最少 2 分鐘
REQ_SLEEP_MIN = 120   # 2 分鐘
REQ_SLEEP_MAX = 300   # 5 分鐘

# Retry
MAX_RETRIES = 3
BACKOFF_BASE = 30

# User-Agent 池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Edge/131.0.0.0",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("twse_crawler")


# ── 工具函式 ──────────────────────────────────────────────────

def _random_ua() -> dict[str, str]:
    return {"User-Agent": random.choice(USER_AGENTS)}


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5


def _sleep(secs: int | tuple[int, int] | None = None):
    """sleep 指定或隨機秒數。"""
    if secs is None:
        secs = random.randint(REQ_SLEEP_MIN, REQ_SLEEP_MAX)
    elif isinstance(secs, tuple):
        secs = random.randint(*secs)
    if secs > 0:
        _secs = secs
        mins, sec_remain = divmod(_secs, 60)
        log.info(f"⏳ Sleep {mins}分{sec_remain}秒 ...")
        time.sleep(_secs)


def _fmt_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _request_with_retry(url: str) -> dict[str, Any] | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_random_ua(), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            stat = data.get("stat", "")
            if "查詢" in stat or "沒有符合" in stat:
                log.warning(f"  ⚠️ TWSE: {stat[:60]}")
                return None
            return data
        except Exception as e:
            log.warning(f"  ⚠️ 第 {attempt}/{MAX_RETRIES} 次失敗: {e}")
            if attempt < MAX_RETRIES:
                backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                log.info(f"  ⏳ Retry 等 {backoff} 秒 ...")
                time.sleep(backoff)
            else:
                log.error(f"  ❌ 放棄")
                return None
    return None


# ── 解析 MI_INDEX ────────────────────────────────────────────
#
# 新格式 (2025+)：資料在 response['tables'][8]
#   fields: [證券代號, 證券名稱, 成交股數, 成交筆數, 成交金額,
#            開盤價, 最高價, 最低價, 收盤價, 漲跌(+/-), 漲跌價差, ...]
#   漲跌(+/-) = '<p>X</p>' 代表當日為除權息日

DAILY_FIELDS = {
    "證券代號": "Ticker",
    "成交股數": "Volume",
    "開盤價": "Open",
    "最高價": "High",
    "最低價": "Low",
    "收盤價": "Close",
}


def _parse_daily_table(table: dict[str, Any], d: date) -> pd.DataFrame:
    """解析 MI_INDEX tables[8] → DataFrame."""
    fields = table.get("fields", [])
    rows = table.get("data", [])

    if not rows:
        return pd.DataFrame()

    # 建立欄位 index
    col_idx: dict[str, int] = {}
    for i, f in enumerate(fields):
        col_idx[f.strip()] = i

    needed = list(DAILY_FIELDS.keys())
    if any(c not in col_idx for c in needed):
        missing = [c for c in needed if c not in col_idx]
        log.warning(f"  缺少欄位: {missing}, 跳過")
        return pd.DataFrame()

    records = []
    for row in rows:
        rec: dict[str, Any] = {"Date": d}
        for cn, en in DAILY_FIELDS.items():
            val = row[col_idx[cn]]
            rec[en] = val
        # 除權息標記
        chg_sign = str(row[col_idx.get("漲跌(+/-)", -1)]) if "漲跌(+/-)" in col_idx else ""
        rec["IsExDiv"] = "X" in chg_sign.replace(" ", "")
        records.append(rec)

    df = pd.DataFrame(records)

    # 數值清洗
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace("--", np.nan)
            .astype("float32")
        )
    df["Volume"] = (
        df["Volume"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("--", "0")
        .astype("int64")
    )

    return df


def _extract_exdiv_list(df: pd.DataFrame, d: date) -> pd.DataFrame | None:
    """從每日資料中取出有除權息標記的股票清單。"""
    exdiv = df[df["IsExDiv"]].copy()
    if exdiv.empty:
        return None
    return exdiv[["Ticker", "Close"]].rename(columns={"Close": "Close_ExDate"})


# ── 單日爬取 ──────────────────────────────────────────────────

def fetch_one_day(d: date, *, save: bool = True) -> dict[str, Any]:
    """爬取指定日期的全市場日成交資料。

    每次請求後隨機 sleep 2~5 分鐘。

    Args:
        d: 日期
        save: 是否存檔

    Returns:
        {"daily_rows": N, "exdiv_count": N, "daily_path": str or None, "exdiv_path": str or None}
    """
    result: dict[str, Any] = {"daily_rows": 0, "exdiv_count": 0,
                               "daily_path": None, "exdiv_path": None}

    date_str = _fmt_date(d)
    url = TWSE_DAILY_URL.format(date_str=date_str)

    log.info(f"🌐 {d}")
    data = _request_with_retry(url)
    if data is None:
        return result

    tables = data.get("tables", [])
    if len(tables) < 9:
        log.warning(f"  ⚠️ 無 tables[8] 資料")
        return result

    df = _parse_daily_table(tables[8], d)
    if df.empty:
        log.warning(f"  ⚠️ 無資料")
        return result

    result["daily_rows"] = len(df)

    # 儲存日線
    if save:
        year_dir = os.path.join(RAW_DIR, str(d.year))
        os.makedirs(year_dir, exist_ok=True)

        daily_path = os.path.join(year_dir, f"{date_str}_daily.parquet")
        df.drop(columns=["IsExDiv"]).to_parquet(daily_path, index=False)
        result["daily_path"] = daily_path
        log.info(f"  ✅ {len(df):,} 檔 → {os.path.basename(daily_path)}")

    # 除權息清單
    exdiv = _extract_exdiv_list(df, d)
    if exdiv is not None and not exdiv.empty:
        result["exdiv_count"] = len(exdiv)
        if save:
            exdiv_path = os.path.join(year_dir, f"{date_str}_exdiv.csv")
            exdiv.to_csv(exdiv_path, index=False, encoding="utf-8-sig")
            result["exdiv_path"] = exdiv_path
            log.info(f"  🔖 {len(exdiv)} 檔除權息")
    else:
        log.info(f"  (無除權息)")

    return result


# ── 年度爬取 ──────────────────────────────────────────────────

def fetch_year(
    year: int,
    *,
    progress_file: str | None = None,
) -> dict[str, Any]:
    """爬取指定年份的所有交易日資料。

    Args:
        year: 西元年份 (2004~2026)
        progress_file: 進度 JSON 路徑 (支援續爬)

    Returns:
        統計摘要
    """
    # 載入進度
    completed_dates: set[str] = set()
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file) as f:
                prog = json.load(f)
            completed_dates = set(prog.get("completed_dates", []))
            log.info(f"📋 載入進度: {len(completed_dates)} 天已完成")
        except Exception:
            pass

    start = date(year, 1, 1)
    end = date(year, 12, 31)
    if end > date.today():
        end = date.today()

    total_days = 0
    total_rows = 0
    exdiv_events = 0
    year_dir = os.path.join(RAW_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)

    current = start
    while current <= end:
        date_str = _fmt_date(current)

        # 跳過已完成的
        if date_str in completed_dates:
            current += timedelta(days=1)
            continue

        # 跳過週末
        if not _is_trading_day(current):
            current += timedelta(days=1)
            continue

        result = fetch_one_day(current)
        total_rows += result["daily_rows"]
        exdiv_events += result["exdiv_count"]

        if result["daily_rows"] > 0:
            total_days += 1
            completed_dates.add(date_str)

            # 更新進度
            if progress_file:
                try:
                    with open(progress_file, "w") as f:
                        json.dump({
                            "year": year,
                            "completed_dates": sorted(completed_dates),
                            "total_days": total_days,
                            "total_rows": total_rows,
                            "last_date": date_str,
                        }, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        # 跨日 sleep (每次請求後)
        _sleep()

        current += timedelta(days=1)

    return {
        "year": year,
        "trading_days": total_days,
        "total_rows": total_rows,
        "exdiv_events": exdiv_events,
    }


def year_is_complete(year: int, progress_file: str) -> bool:
    """檢查某一年是否已完整爬完。"""
    if not os.path.exists(progress_file):
        return False
    try:
        with open(progress_file) as f:
            prog = json.load(f)
    except Exception:
        return False
    if prog.get("year") != year:
        return False

    # 計算該年交易日數（到今天的粗略值）
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    if end > date.today():
        end = date.today()
    trading_days = sum(1 for i in range((end - start).days + 1)
                       if (start + timedelta(days=i)).weekday() < 5)
    return len(prog.get("completed_dates", [])) >= trading_days


# ── 年度區間爬取（含跨年 sleep）───────────────────────────────

YEAR_SLEEP_MIN = 10   # 10 秒
YEAR_SLEEP_MAX = 30  # 30 秒


def fetch_year_range(
    start_year: int,
    end_year: int,
    *,
    sleep_between: tuple[int, int] = (YEAR_SLEEP_MIN, YEAR_SLEEP_MAX),
) -> list[dict[str, Any]]:
    """逐年爬取，每年之間隨機休息 4~10 分鐘。

    Args:
        start_year: 起始年 (e.g. 2004)
        end_year: 截止年 (e.g. 2026)
        sleep_between: 年度之間的 sleep 範圍 (秒)

    Returns:
        各年統計摘要列表
    """
    results = []

    for year in range(start_year, end_year + 1):
        log.info(f"\n{'='*60}")
        log.info(f"📅  年度: {year}")
        log.info(f"{'='*60}")

        progress_file = os.path.join(RAW_DIR, f"_progress_{year}.json")

        # 檢查是否已完整
        if year_is_complete(year, progress_file):
            log.info(f"✅ {year} 已完成，跳過")
            # Load previous result
            try:
                with open(progress_file) as f:
                    prog = json.load(f)
                results.append({
                    "year": year,
                    "trading_days": prog.get("total_days", 0),
                    "total_rows": prog.get("total_rows", 0),
                    "exdiv_events": prog.get("exdiv_events", 0),
                    "status": "skipped",
                })
            except Exception:
                results.append({"year": year, "status": "skipped"})

            # 仍要跨年 sleep
            if year < end_year:
                _sleep(sleep_between)
            continue

        result = fetch_year(year, progress_file=progress_file)
        result["status"] = "completed"
        results.append(result)

        log.info(f"✅ {year} 完成: {result['trading_days']} 天, "
                 f"{result['total_rows']:,} 行, "
                 f"{result['exdiv_events']} 次除權息")

        # 跨年 sleep（4~10 分鐘）
        if year < end_year:
            log.info(f"{'='*60}")
            _sleep(sleep_between)

    return results


# ── 合併年度資料 ──────────────────────────────────────────────

def merge_year(year: int) -> dict[str, str | int]:
    """將單年度所有日 parquet 合併成一個年度檔。

    Returns:
        {"year": year, "daily_rows": N, "exdiv_rows": N,
         "daily_path": str, "exdiv_path": str or None}
    """
    year_dir = os.path.join(RAW_DIR, str(year))
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"{year_dir} 不存在")

    result: dict[str, str | int] = {"year": year}

    # 合併日線
    daily_files = sorted(f for f in os.listdir(year_dir)
                         if f.endswith("_daily.parquet"))
    if daily_files:
        dfs = []
        for f in daily_files:
            dfs.append(pd.read_parquet(os.path.join(year_dir, f)))
        all_df = pd.concat(dfs, ignore_index=True)
        all_df.sort_values(["Ticker", "Date"], inplace=True)
        merged_path = os.path.join(year_dir, f"_{year}_daily.parquet")
        all_df.to_parquet(merged_path, index=False)
        result["daily_rows"] = len(all_df)
        result["daily_path"] = merged_path
        log.info(f"📦 {year} 日線: {len(all_df):,} 行 → {merged_path}")

    # 合併除權息
    exdiv_files = sorted(f for f in os.listdir(year_dir)
                         if f.endswith("_exdiv.csv"))
    if exdiv_files:
        dfs = []
        for f in exdiv_files:
            dfs.append(pd.read_csv(os.path.join(year_dir, f)))
        all_exdiv = pd.concat(dfs, ignore_index=True)
        merged_path = os.path.join(year_dir, f"_{year}_exdiv.csv")
        all_exdiv.to_csv(merged_path, index=False, encoding="utf-8-sig")
        result["exdiv_rows"] = len(all_exdiv)
        result["exdiv_path"] = merged_path
        log.info(f"📦 {year} 除權息: {len(all_exdiv)} 筆 → {merged_path}")

    return result


def merge_all_years() -> dict[str, str | int]:
    """將所有年度合併成一個全量檔。"""
    daily_parts = []
    exdiv_parts = []

    for y in sorted(os.listdir(RAW_DIR)):
        if not y.isdigit():
            continue
        year_dir = os.path.join(RAW_DIR, y)
        daily_path = os.path.join(year_dir, f"_{y}_daily.parquet")
        exdiv_path = os.path.join(year_dir, f"_{y}_exdiv.csv")
        if os.path.exists(daily_path):
            daily_parts.append(pd.read_parquet(daily_path))
        if os.path.exists(exdiv_path):
            exdiv_parts.append(pd.read_csv(exdiv_path))

    result: dict[str, str | int] = {}

    if daily_parts:
        full = pd.concat(daily_parts, ignore_index=True)
        full.sort_values(["Ticker", "Date"], inplace=True)
        path = os.path.join(RAW_DIR, "_all_daily.parquet")
        full.to_parquet(path, index=False)
        result["all_daily_rows"] = len(full)
        result["all_daily_path"] = path
        log.info(f"📦📦 全量日線: {len(full):,} 行 → {path}")

    if exdiv_parts:
        full = pd.concat(exdiv_parts, ignore_index=True)
        path = os.path.join(RAW_DIR, "_all_exdiv.csv")
        full.to_csv(path, index=False, encoding="utf-8-sig")
        result["all_exdiv_rows"] = len(full)
        result["all_exdiv_path"] = path
        log.info(f"📦📦 全量除權息: {len(full)} 筆 → {path}")

    return result


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TWSE 資料爬蟲")
    parser.add_argument("--year", type=int, help="單一年份")
    parser.add_argument("--year-range", nargs=2, type=int, metavar=("START", "END"),
                        help="年份區間 (e.g. 2004 2026)")
    parser.add_argument("--merge", type=int, nargs="?", const=-1,
                        help="合併指定年份 (無參數=全部)")
    parser.add_argument("--merge-all", action="store_true",
                        help="合併所有年份為全量檔")
    args = parser.parse_args()

    if args.merge is not None:
        if args.merge == -1:
            # merge all years individually
            for y in sorted(os.listdir(RAW_DIR)):
                if y.isdigit():
                    merge_year(int(y))
        else:
            merge_year(args.merge)
    elif args.merge_all:
        merge_all_years()
    elif args.year:
        fetch_year(args.year,
                    progress_file=os.path.join(RAW_DIR, f"_progress_{args.year}.json"))
        merge_year(args.year)
    elif args.year_range:
        results = fetch_year_range(*args.year_range)
        # 合併各年
        for r in results:
            try:
                merge_year(r["year"])
            except Exception as e:
                log.warning(f"合併 {r['year']} 失敗: {e}")
        merge_all_years()
    else:
        parser.print_help()
