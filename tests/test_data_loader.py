"""Data Loader 單元測試.

測試涵蓋：
    1. TWSEStockLoader 初始化與目錄掃描
    2. 還原權值計算 (Backward Adjustment)
    3. Resample: 日線 → 週線 / 月線
    4. yfinance fallback
    5. 多時間框架載入
    6. 邊界條件: 空資料、資料不足、無除權息事件
"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import (
    TWSEStockLoader,
    _calc_cumulative_factors,
    apply_backward_adjustment,
    DATA_DIR,
    RAW_DIR,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """標準 OHLCV 測試資料 (300 筆日線)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.012, len(dates)))).astype(np.float32)
    open_p = close - close * np.random.normal(0, 0.003, len(dates)).astype(np.float32)
    high = close * (1 + np.abs(np.random.normal(0.003, 0.003, len(dates))).astype(np.float32))
    low = close * (1 - np.abs(np.random.normal(0.003, 0.003, len(dates))).astype(np.float32))
    vol = np.random.randint(50_000, 500_000, len(dates), dtype=np.int64)

    df = pd.DataFrame({
        "Date": dates,
        "Ticker": "2330",
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    })
    return df.sort_values("Date").reset_index(drop=True)


@pytest.fixture
def adj_data() -> pd.DataFrame:
    """含除權息事件的測試資料."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    # 股票價格逐步上漲，並在特定日期發生除權息
    raw_close = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.015, len(dates))))
    df = pd.DataFrame({
        "Date": dates,
        "Ticker": "2330",
        "Open": raw_close - raw_close * 0.002,
        "High": raw_close * 1.015,
        "Low": raw_close * 0.985,
        "Close": raw_close.astype(np.float32),
        "Volume": np.random.randint(100_000, 500_000, len(dates)),
    })

    # 在 index 50 加入除權息事件 (現金 10 元 + 股票股利 0.5 元)
    event_date = dates[50]
    event_close_before = float(df.loc[df["Date"] == event_date, "Close"].iloc[0])
    event_open_next = df.loc[df["Date"] == event_date, "Open"].iloc[0]

    events = pd.DataFrame([
        {"Date": event_date, "Cash_Dividend": 10.0, "Stock_Dividend": 0.5},
    ])
    return df, events


# ═══════════════════════════════════════════════════════════════
# 1. 還原權值計算
# ═══════════════════════════════════════════════════════════════


class TestBackwardAdjustment:
    def test_calc_cumulative_factors_no_events(self):
        """無除權息事件 → 因子全為 1.0."""
        dates = pd.date_range("2024-01-01", periods=10)
        closes = np.ones(10) * 100
        factors = _calc_cumulative_factors(
            dates.values, closes, {},
        )
        assert np.allclose(factors, 1.0)

    def test_calc_cumulative_factors_cash_only(self):
        """僅現金股利."""
        dates = pd.date_range("2024-01-01", periods=20)
        closes = np.ones(20) * 100
        event_date = dates[5]
        event_map = {dates[5]: (5.0, 0.0)}  # 5 元現金
        dates_arr = dates.values.astype("datetime64[D]")

        factors = _calc_cumulative_factors(
            dates_arr, closes.astype(np.float64), event_map,
        )
        # 事件前因子 < 1.0 (還原權值使舊價格降低)
        assert np.all(factors[:5] < 1.0), f"Factors before event: {factors[:5]}"
        # 事件當天及之後因子 = 1.0 (最新價格不變)
        assert np.allclose(factors[5:], 1.0), f"Factors after event: {factors[5:]}"
        # 因子遞減 (越遠的事件累積越多，但平盤時前段皆相等)
        assert factors[0] < factors[5], f"factor[0]={factors[0]} should be < factor[5]={factors[5]}"

    def test_calc_cumulative_factors_stock_only(self):
        """僅股票股利."""
        dates = pd.date_range("2024-01-01", periods=20)
        closes = np.ones(20, dtype=np.float64) * 100
        event_date = dates[5]
        stock_event = pd.Timestamp(event_date)
        event_map = {stock_event: (0.0, 500.0)}  # 50% 股票股利 (500/1000=0.5)
        dates_arr = dates.values.astype("datetime64[D]")

        factors = _calc_cumulative_factors(
            dates_arr, closes, event_map,
        )
        # 事件前因子 < 1.0
        assert np.all(factors[:5] < 1.0), f"Factors before event: {factors[:5]}"
        # 事件當天及之後因子 = 1.0
        assert np.allclose(factors[5:], 1.0), f"Factors after event: {factors[5:]}"
        assert factors[0] < 0.67  # 50% 股票股利應造成較大調整

    def test_apply_adjustment(self, adj_data):
        """套用還原權值後，舊價格應被調低."""
        df, events = adj_data
        event_date = events.iloc[0]["Date"]

        adjusted = apply_backward_adjustment(df, events)

        assert "Adj_Close" in adjusted.columns
        assert "Adj_Open" in adjusted.columns
        assert "Adj_High" in adjusted.columns
        assert "Adj_Low" in adjusted.columns
        assert "Adj_Volume" in adjusted.columns

        # 除權息後的資料應與原始相同 (最新價格不變)
        post_mask = adjusted["Date"] >= event_date
        pre_mask = adjusted["Date"] < event_date

        assert np.allclose(
            adjusted.loc[post_mask, "Adj_Close"].values,
            adjusted.loc[post_mask, "Close"].values,
        )
        # 除權息前的資料應被調低
        assert np.all(
            adjusted.loc[pre_mask, "Adj_Close"].values
            < adjusted.loc[pre_mask, "Close"].values
        )

    def test_apply_empty_events(self, sample_ohlcv: pd.DataFrame):
        """無除權息事件 → Adj_* = 原始值."""
        adjusted = apply_backward_adjustment(sample_ohlcv, None)
        for col in ["Adj_Close", "Adj_Open", "Adj_High", "Adj_Low"]:
            assert np.allclose(
                adjusted[col].values, adjusted[col.replace("Adj_", "")].values,
            )
        assert "Adj_Volume" in adjusted.columns


# ═══════════════════════════════════════════════════════════════
# 2. Resample 測試
# ═══════════════════════════════════════════════════════════════


class TestResample:
    @pytest.fixture
    def daily_wide(self) -> pd.DataFrame:
        """含 full Adj_* 欄位的日線."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.012, len(dates)))).astype(np.float32)
        open_p = close - close * np.random.normal(0, 0.003, len(dates))
        high = close * (1 + np.abs(np.random.normal(0.003, 0.003, len(dates))))
        low = close * (1 - np.abs(np.random.normal(0.003, 0.003, len(dates))))
        vol = np.random.randint(50_000, 500_000, len(dates), dtype=np.int64)

        df = pd.DataFrame({
            "Date": dates,
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": vol,
            "Adj_Open": open_p,
            "Adj_High": high,
            "Adj_Low": low,
            "Adj_Close": close,
            "Adj_Volume": vol,
        })
        return df.sort_values("Date").reset_index(drop=True)

    def test_resample_weekly_shape(self, daily_wide: pd.DataFrame):
        """週線 resample 結構正確."""
        loader = TWSEStockLoader()
        weekly = loader.resample_weekly(daily_wide)

        assert "Date" in weekly.columns
        assert "Open" in weekly.columns
        assert "High" in weekly.columns
        assert "Low" in weekly.columns
        assert "Close" in weekly.columns
        assert "Volume" in weekly.columns
        assert len(weekly) < len(daily_wide)

    def test_resample_weekly_ohlc_logic(self, daily_wide: pd.DataFrame):
        """週線 OHLC 邏輯正確:
        - Open = 週一開盤
        - Close = 週五收盤
        - High = 該週最高
        - Low = 該週最低
        - Volume = 該週總量
        """
        loader = TWSEStockLoader()
        weekly = loader.resample_weekly(daily_wide)

        # 比對最後一週
        daily_last_week = daily_wide.tail(5)
        last_open = daily_last_week.iloc[0]["Open"]
        last_close = daily_last_week.iloc[-1]["Close"]
        last_high = daily_last_week["High"].max()
        last_low = daily_last_week["Low"].min()
        last_vol = daily_last_week["Volume"].sum()

        last_weekly = weekly.iloc[-1]
        assert abs(last_weekly["Open"] - last_open) < 0.01, (
            f"Weekly Open {last_weekly['Open']} != {last_open}"
        )
        assert abs(last_weekly["Close"] - last_close) < 0.01
        assert last_weekly["High"] >= last_high - 0.01
        assert last_weekly["Low"] <= last_low + 0.01

    def test_resample_monthly_shape(self, daily_wide: pd.DataFrame):
        """月線 resample 結構正確."""
        loader = TWSEStockLoader()
        monthly = loader.resample_monthly(daily_wide)

        assert "Date" in monthly.columns
        assert len(monthly) < len(daily_wide)
        # 3 個月資料約有 3 個月
        assert len(monthly) >= 2
        assert len(monthly) <= 15  # 上限 (含 partial months)

    def test_resample_empty(self):
        """空 DataFrame resample → 空 DataFrame 或 ValueError."""
        loader = TWSEStockLoader()
        empty = pd.DataFrame()
        try:
            weekly = loader.resample_weekly(empty)
            assert weekly.empty
        except (ValueError, KeyError):
            pass  # 驗證欄位檢查合理
        try:
            monthly = loader.resample_monthly(empty)
            assert monthly.empty
        except (ValueError, KeyError):
            pass

    def test_resample_weekly_no_duplicates(self, daily_wide: pd.DataFrame):
        """週線無重複日期."""
        loader = TWSEStockLoader()
        weekly = loader.resample_weekly(daily_wide)
        assert weekly["Date"].is_unique

    def test_resample_monthly_no_duplicates(self, daily_wide: pd.DataFrame):
        """月線無重複日期."""
        loader = TWSEStockLoader()
        monthly = loader.resample_monthly(daily_wide)
        assert monthly["Date"].is_unique

    def test_resample_weekly_volume_positive(self, daily_wide: pd.DataFrame):
        """週線成交量為正."""
        loader = TWSEStockLoader()
        weekly = loader.resample_weekly(daily_wide)
        assert (weekly["Volume"] > 0).all()


# ═══════════════════════════════════════════════════════════════
# 3. 多時間框架載入
# ═══════════════════════════════════════════════════════════════


class TestMultiTimeframe:
    def test_load_multi_timeframe_with_yfinance(self, sample_ohlcv: pd.DataFrame):
        """load_multi_timeframe 回傳 3 個週期."""
        loader = TWSEStockLoader()
        # 用 mock 的方式：直接傳入 sample OHLCV
        tf = loader.load_multi_timeframe("2330", adjusted=True)
        # 因為沒有真實資料，會 fallback 到 yfinance 或回傳空的 DataFrame
        assert isinstance(tf, dict)
        assert "daily" in tf
        assert "weekly" in tf
        assert "monthly" in tf

    def test_multi_timeframe_structure(self, sample_ohlcv: pd.DataFrame):
        """多時間框架的 key 與順序正確."""
        loader = TWSEStockLoader()
        tf = loader.load_multi_timeframe("2330")
        expected_keys = ["daily", "weekly", "monthly"]
        assert list(tf.keys()) == expected_keys

    def test_multi_timeframe_no_duplicate_dates(self, sample_ohlcv: pd.DataFrame):
        """各時間框架內無重複日期."""
        loader = TWSEStockLoader()
        tf = loader.load_multi_timeframe("2330", adjusted=True)
        for period, df in tf.items():
            if not df.empty:
                assert df["Date"].is_unique, f"{period} has duplicate dates"

    def test_multi_timeframe_date_order(self, sample_ohlcv: pd.DataFrame):
        """各時間框架日期遞增."""
        loader = TWSEStockLoader()
        tf = loader.load_multi_timeframe("2330", adjusted=True)
        for period, df in tf.items():
            if len(df) > 1:
                assert df["Date"].is_monotonic_increasing, f"{period} not sorted"


# ═══════════════════════════════════════════════════════════════
# 4. 初始化與邊界測試
# ═══════════════════════════════════════════════════════════════


class TestLoaderInit:
    def test_loader_init(self):
        """初始化不拋錯."""
        loader = TWSEStockLoader()
        assert loader.raw_dir == RAW_DIR

    def test_loader_list_available(self):
        """list_available_tickers 不回拋錯."""
        loader = TWSEStockLoader()
        tickers = loader.list_available_tickers()
        # 正常情況下可能空清單 (視資料)
        assert isinstance(tickers, list)

    def test_get_date_range_no_data(self):
        """無資料的 ticker → (None, None)."""
        loader = TWSEStockLoader()
        start, end = loader.get_date_range("9999")
        assert start is None
        assert end is None

    def test_loader_empty_dividends_file(self, tmp_path: Path):
        """無股利檔案不拋錯."""
        loader = TWSEStockLoader()
        assert loader._dividends is None or isinstance(loader._dividends, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════
# 5. 資料完整性測試
# ═══════════════════════════════════════════════════════════════


class TestDataIntegrity:
    def test_no_infinite_values(self, sample_ohlcv: pd.DataFrame):
        """資料中無 infinite 值."""
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert not np.any(np.isinf(sample_ohlcv[col])), f"{col} has inf"

    def test_no_nan_in_core_fields(self, sample_ohlcv: pd.DataFrame):
        """核心欄位無 NaN."""
        for col in ["Open", "High", "Low", "Close"]:
            assert not sample_ohlcv[col].isna().any(), f"{col} has NaN"

    def test_ohlcv_logical_constraints(self, sample_ohlcv: pd.DataFrame):
        """OHLC 邏輯約束:
        - High >= Low
        - High >= Close (Open 不強制，因合成資料可能微幅偏差)
        - Low <= Close
        """
        assert (sample_ohlcv["High"] >= sample_ohlcv["Low"]).all()
        assert (sample_ohlcv["High"] >= sample_ohlcv["Close"]).all()
        assert (sample_ohlcv["Low"] <= sample_ohlcv["Close"]).all()

    def test_volume_positive(self, sample_ohlcv: pd.DataFrame):
        """成交量為正."""
        assert (sample_ohlcv["Volume"] > 0).all()

    def test_no_duplicate_daily(self, sample_ohlcv: pd.DataFrame):
        """日線無重複日期."""
        assert sample_ohlcv["Date"].is_unique
