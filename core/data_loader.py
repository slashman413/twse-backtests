"""
TWSE 量化資料載入層 — Data Access Layer
=========================================

功能：
    1. 從爬蟲 Raw/ 目錄讀取股票日線 (parquet)
    2. 套用 Backward Adjustment 還原權值
    3. Resample 日線 → 週線 / 月線
    4. yfinance 補償（爬蟲尚未爬到的資料）

依賴：
    pip install pandas numpy pyarrow yfinance

使用方式：
    from data_loader import TWSEStockLoader

    loader = TWSEStockLoader()
    daily = loader.load_daily("2330", adjusted=True)
    weekly = loader.resample_weekly(daily)
    monthly = loader.resample_monthly(daily)
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ── 路徑 ──────────────────────────────────────────────────────
DATA_DIR = os.environ.get("TWSE_DATA_DIR", "D:/TWSE-Data")
RAW_DIR = os.path.join(DATA_DIR, "Raw")
ADJ_DIR = os.path.join(DATA_DIR, "Adjusted")
CODE_DIR = os.path.join(DATA_DIR, "Code")
DIVIDEND_FILE = os.path.join(RAW_DIR, "_yf_dividends.csv")

# ── Backward Adjustment (輕量版, 從 adjuster.py 提取) ─────────


def _calc_cumulative_factors(
    dates: np.ndarray,
    closes: np.ndarray,
    event_map: dict[pd.Timestamp, tuple[float, float]],
) -> np.ndarray:
    """計算單一股票的累積還原因子 (Backward Adjustment).

    與 twse_adjuster.py 邏輯一致：
    從最新日往舊日遞迴，遇除權息則更新累積因子。

    Args:
        dates: 日期陣列 (ascending, datetime64[D])
        closes: 收盤價陣列
        event_map: {日期: (現金股利, 股票股利)}

    Returns:
        cum_factors: 與 dates 同長度, dates[-1]=1.0
    """
    n = len(dates)
    factors = np.ones(n, dtype=np.float64)
    cum = 1.0

    for i in range(n - 1, -1, -1):
        factors[i] = cum
        ts = pd.Timestamp(dates[i])
        evt = event_map.get(ts)
        if evt is not None and i > 0:
            d_cash, d_stock = evt
            prev_close = float(closes[i - 1])
            if np.isnan(prev_close) or prev_close <= 0:
                continue
            denom = 1.0 + d_stock / 1000.0
            ref = (prev_close - d_cash) / denom
            ef = min(ref / prev_close, 1.0)
            cum *= ef
    return factors


def apply_backward_adjustment(
    df: pd.DataFrame,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """對 DataFrame 套用 Backward Adjustment.

    Args:
        df: 須含 Date(ascending), Ticker, Open, High, Low, Close, Volume
        events: 除權息事件 (Date, Cash_Dividend, Stock_Dividend)
                None 則不回補，CumFactor=1.0

    Returns:
        新增 Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume, CumFactor 欄位
    """
    result = df.copy()
    n = len(result)
    opens = result["Open"].values.astype(np.float64)
    highs = result["High"].values.astype(np.float64)
    lows = result["Low"].values.astype(np.float64)
    closes = result["Close"].values.astype(np.float64)
    vols = result["Volume"].values.astype(np.float64)
    dates = result["Date"].values

    # 建立事件查詢
    event_map: dict[pd.Timestamp, tuple[float, float]] = {}
    if events is not None and not events.empty:
        for _, row in events.iterrows():
            event_map[row["Date"]] = (
                float(row["Cash_Dividend"]),
                float(row["Stock_Dividend"]),
            )

    cf = _calc_cumulative_factors(dates, closes, event_map)

    result["Adj_Open"] = opens * cf
    result["Adj_High"] = highs * cf
    result["Adj_Low"] = lows * cf
    result["Adj_Close"] = closes * cf
    safe_cf = np.where(cf > 0, cf, 1.0)
    result["Adj_Volume"] = (vols / safe_cf).round(0).astype("int64")
    result["CumFactor"] = cf

    return result


# ═══════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════

class TWSEStockLoader:
    """TWSE 股票資料載入器.

    支援從爬蟲 parquet 讀取日線 + yfinance dividend 還原權值
    + resample 為週線/月線。

    Usage:
        loader = TWSEStockLoader()
        df = loader.load_daily("2330", start="2020-01-01", adjusted=True)
        wk = loader.resample_weekly(df)
        mo = loader.resample_monthly(df)
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        dividend_path: str | None = DIVIDEND_FILE,
        use_yfinance_fallback: bool = True,
    ):
        """
        Args:
            data_dir: TWSE-Data 根目錄
            dividend_path: 股利 CSV 路徑 (None=不載入)
            use_yfinance_fallback: 爬蟲無資料時是否用 yfinance 補償
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "Raw")
        self.dividend_path = dividend_path
        self.use_yfinance = use_yfinance_fallback

        # 載入股利資料
        self._dividends: pd.DataFrame | None = None
        if dividend_path and os.path.exists(dividend_path):
            self._dividends = pd.read_csv(dividend_path)
            self._dividends["Date"] = (
                pd.to_datetime(self._dividends["Date"])
                .dt.tz_localize(None)
                .dt.normalize()
            )

        # 爬蟲年度目錄列表
        self._year_dirs: list[int] = sorted(
            int(d) for d in os.listdir(self.raw_dir)
            if d.isdigit() and os.path.isdir(os.path.join(self.raw_dir, d))
        )

    # ── 主要 API ──────────────────────────────────────────────

    def load_daily(
        self,
        ticker: str,
        start: str | date | None = None,
        end: str | date | None = None,
        *,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """載入指定股票的日線資料.

        Args:
            ticker: 股票代號 (e.g. "2330")
            start: 起始日期 "YYYY-MM-DD" 或 date
            end: 截止日期
            adjusted: 是否回補還原權值

        Returns:
            DataFrame 含 Date, Ticker, Open, High, Low, Close, Volume
            (adjusted=True 時額外含 Adj_Open~Close, CumFactor)
        """
        df = self._read_from_parquet(ticker)

        # 若爬蟲資料不足（日期區間不在爬蟲範圍內），用 yfinance 補償
        _use_yf = False
        if df.empty and self.use_yfinance:
            _use_yf = True
        elif self.use_yfinance and not df.empty:
            # 檢查爬蟲資料是否涵蓋近期（最近 30 天內）
            pq_max = df["Date"].max()
            days_gap = (pd.Timestamp.now() - pq_max).days
            if days_gap > 30:
                _use_yf = True

        if _use_yf:
            yf_df = self._fetch_yfinance(ticker)
            if not yf_df.empty:
                # 合併爬蟲資料 + yfinance 資料 (去重)
                combined = pd.concat([df, yf_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["Date"])
                df = combined

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("Date").reset_index(drop=True)

        # 先對完整資料做還原權值（確保所有股利事件都計入累積因子）
        if adjusted and not df.empty:
            events = self._get_dividend_events(ticker)
            df = apply_backward_adjustment(df, events)

        # 最後才裁切日期範圍
        if start:
            df = df[df["Date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["Date"] <= pd.Timestamp(end)]

        return df

    def load_batch(
        self,
        tickers: list[str],
        start: str | date | None = None,
        end: str | date | None = None,
        *,
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """批量載入多檔股票.

        Returns:
            {ticker: DataFrame}
        """
        return {
            t: self.load_daily(t, start, end, adjusted=adjusted)
            for t in tickers
        }

    # ── Resample ──────────────────────────────────────────────

    @staticmethod
    def resample_weekly(
        df: pd.DataFrame,
        *,
        price_col: str = "Adj_Close",
    ) -> pd.DataFrame:
        """日線 → 週線.

        週線規則：
            Open  = 該週首日開盤
            High  = 該週最高
            Low   = 該週最低
            Close = 該週最後收盤
            Volume = 該週總成交量

        Args:
            df: 日線 DataFrame (須含 Date, Open, High, Low, Close, Volume)
            price_col: 使用的價格欄位 (Adj_Close 或 Close)

        Returns:
            週線 DataFrame, index=週一日期
        """
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"缺少必要欄位: {col}")

        d = df.set_index("Date").copy()
        d = d.sort_index()

        # 使用指定的價格欄位
        close_col = price_col if price_col in d.columns else "Close"
        open_col = close_col.replace("Close", "Open")
        high_col = close_col.replace("Close", "High")
        low_col = close_col.replace("Close", "Low")
        vol_col = "Adj_Volume" if "Adj_Volume" in d.columns else "Volume"

        # 確保欄位存在
        o = d[open_col] if open_col in d.columns else d["Open"]
        h = d[high_col] if high_col in d.columns else d["High"]
        l = d[low_col] if low_col in d.columns else d["Low"]
        c = d[close_col] if close_col in d.columns else d["Close"]
        v = d[vol_col] if vol_col in d.columns else d["Volume"]

        # 週 resample
        weekly = pd.DataFrame({
            "Open": o.resample("W").first(),
            "High": h.resample("W").max(),
            "Low": l.resample("W").min(),
            "Close": c.resample("W").last(),
            "Volume": v.resample("W").sum(),
        }).dropna(subset=["Close"])

        weekly.index.name = "Date"
        return weekly.reset_index()

    @staticmethod
    def resample_monthly(
        df: pd.DataFrame,
        *,
        price_col: str = "Adj_Close",
    ) -> pd.DataFrame:
        """日線 → 月線.

        規則同週線，但以月為單位。

        Args:
            df: 日線 DataFrame
            price_col: 使用的價格欄位

        Returns:
            月線 DataFrame
        """
        d = df.set_index("Date").copy().sort_index()

        close_col = price_col if price_col in d.columns else "Close"
        open_col = close_col.replace("Close", "Open")
        high_col = close_col.replace("Close", "High")
        low_col = close_col.replace("Close", "Low")
        vol_col = "Adj_Volume" if "Adj_Volume" in d.columns else "Volume"

        o = d[open_col] if open_col in d.columns else d["Open"]
        h = d[high_col] if high_col in d.columns else d["High"]
        l = d[low_col] if low_col in d.columns else d["Low"]
        c = d[close_col] if close_col in d.columns else d["Close"]
        v = d[vol_col] if vol_col in d.columns else d["Volume"]

        monthly = pd.DataFrame({
            "Open": o.resample("ME").first(),
            "High": h.resample("ME").max(),
            "Low": l.resample("ME").min(),
            "Close": c.resample("ME").last(),
            "Volume": v.resample("ME").sum(),
        }).dropna(subset=["Close"])

        monthly.index.name = "Date"
        return monthly.reset_index()

    # ── 多時間框架載入 ─────────────────────────────────────────

    def load_multi_timeframe(
        self,
        ticker: str,
        start: str | date | None = None,
        end: str | date | None = None,
        *,
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """一次載入日/週/月三種週期.

        Returns:
            {"daily": df, "weekly": df, "monthly": df}
        """
        daily = self.load_daily(ticker, start, end, adjusted=adjusted)
        if daily.empty:
            return {"daily": daily, "weekly": pd.DataFrame(), "monthly": pd.DataFrame()}

        weekly = self.resample_weekly(daily)
        monthly = self.resample_monthly(daily)

        return {"daily": daily, "weekly": weekly, "monthly": monthly}

    # ── 內部方法 ──────────────────────────────────────────────

    def _read_from_parquet(self, ticker: str) -> pd.DataFrame:
        """從爬蟲的 Raw/{year}/{yyyymmdd}_daily.parquet 讀取指定股票.

        跨年度掃描，只取出該 ticker 的資料。
        """
        chunks: list[pd.DataFrame] = []

        for year in self._year_dirs:
            year_dir = os.path.join(self.raw_dir, str(year))
            if not os.path.isdir(year_dir):
                continue

            # 掃描該年所有日 parquet
            for fname in sorted(os.listdir(year_dir)):
                if not fname.endswith("_daily.parquet"):
                    continue
                path = os.path.join(year_dir, fname)
                try:
                    # 用 pyarrow 的 filter 功能只讀需要的 ticker
                    # 加速大量重複讀取
                    df = pd.read_parquet(
                        path,
                        filters=[("Ticker", "==", ticker)],
                        columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                    if not df.empty:
                        chunks.append(df)
                except Exception:
                    # 降級：讀取整個檔案再過濾
                    try:
                        full = pd.read_parquet(path)
                        sub = full[full["Ticker"] == ticker]
                        if not sub.empty:
                            chunks.append(sub[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]])
                    except Exception:
                        continue

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True)
        result["Date"] = pd.to_datetime(result["Date"])
        return result.sort_values("Date").reset_index(drop=True)

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame:
        """用 yfinance 補償爬蟲尚未爬到的資料."""
        try:
            import yfinance as yf
        except ImportError:
            return pd.DataFrame()

        # 試上市
        for suffix in [".TW", ".TWO"]:
            try:
                tk = yf.Ticker(f"{ticker}{suffix}")
                hist = tk.history(period="max", auto_adjust=False)
                if hist.empty:
                    continue

                result = pd.DataFrame({
                    "Date": pd.to_datetime(hist.index).tz_localize(None).normalize(),
                    "Ticker": ticker,
                    "Open": hist["Open"].values.astype(np.float32),
                    "High": hist["High"].values.astype(np.float32),
                    "Low": hist["Low"].values.astype(np.float32),
                    "Close": hist["Close"].values.astype(np.float32),
                    "Volume": hist["Volume"].values.astype(np.int64),
                })
                return result.sort_values("Date").reset_index(drop=True)
            except Exception:
                continue

        return pd.DataFrame()

    def _get_dividend_events(self, ticker: str) -> pd.DataFrame:
        """取得指定股票的除權息事件."""
        if self._dividends is None:
            return pd.DataFrame()

        # Ticker 欄位可能是 int (CSV 純數字)
        ticker_val: str | int = ticker
        if self._dividends["Ticker"].dtype in (np.int64, np.int32, np.int_):
            try:
                ticker_val = int(ticker)
            except ValueError:
                pass

        mask = self._dividends["Ticker"] == ticker_val
        events = self._dividends[mask][
            ["Date", "Cash_Dividend", "Stock_Dividend"]
        ].copy()
        return events.sort_values("Date").reset_index(drop=True)

    def list_available_tickers(self) -> list[str]:
        """列出爬蟲已下載的所有股票代號 (從最新一年取樣)."""
        if not self._year_dirs:
            return []

        # 從最新的有資料年份取樣
        for year in reversed(self._year_dirs):
            year_dir = os.path.join(self.raw_dir, str(year))
            parquets = sorted(
                f for f in os.listdir(year_dir)
                if f.endswith("_daily.parquet")
            )
            if parquets:
                try:
                    df = pd.read_parquet(
                        os.path.join(year_dir, parquets[0]),
                        columns=["Ticker"],
                    )
                    return sorted(df["Ticker"].unique().tolist())
                except Exception:
                    continue
        return []

    def get_date_range(self, ticker: str) -> tuple[date | None, date | None]:
        """查詢某股票的資料日期範圍."""
        df = self._read_from_parquet(ticker)
        if df.empty:
            return None, None
        return df["Date"].min().date(), df["Date"].max().date()


# ═══════════════════════════════════════════════════════════════
# 快速測試
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TWSE Data Loader 測試")
    parser.add_argument("--ticker", default="2330", help="股票代號")
    parser.add_argument("--start", default="2024-01-01", help="開始日期")
    parser.add_argument("--end", default=None, help="截止日期")
    parser.add_argument("--list", action="store_true", help="列出可用股票")
    args = parser.parse_args()

    loader = TWSEStockLoader()

    if args.list:
        tickers = loader.list_available_tickers()
        print(f"可用股票: {len(tickers)} 檔")
        print(f"範例: {tickers[:20]}")
        sys.exit(0)

    print(f"載入 {args.ticker} {args.start} ~ {args.end or '今天'} ...")

    # 多時間框架
    tf = loader.load_multi_timeframe(args.ticker, start=args.start, end=args.end)

    for period, df in tf.items():
        print(f"\n{'='*50}")
        print(f"{period}: {len(df)} 行")
        print(f"  日期: {df['Date'].min().date() if not df.empty else 'N/A'} ~ "
              f"{df['Date'].max().date() if not df.empty else 'N/A'}")
        if not df.empty:
            cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
                    if c in df.columns]
            print(f"  最新 3 筆:\n{df.tail(3)[cols].to_string(index=False)}")

    # 日期範圍
    start_d, end_d = loader.get_date_range(args.ticker)
    if start_d:
        print(f"\n爬蟲資料範圍: {start_d} ~ {end_d}")
