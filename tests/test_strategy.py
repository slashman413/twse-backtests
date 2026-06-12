"""Strategy Engine 單元測試 (v2).

測試涵蓋重新實作後的所有類別與函式：
    1. MarketSignalV2 — 大盤多空 + 備戰區 + 低轉折
    2. SectorSignalV2 — 類股燈號 (MACD 4 箭頭)
    3. BigStockBuySignalV2 — 大飆股買進 8 條件
    4. BigStockSellSignalV2 — 大飆股賣出 5 條件
    5. EntrySignal — 切入訊號 (60分RSI60, 月黑6K, DIF210觸底)
    6. CrashExitSignal — 日頂天 + 月6K/9K賣出
    7. StockSurgeSignal — 股票飆漲 (DIF210 + ADX300 螺旋)
    8. CapitalAllocator — 資金配置
    9. SignalAligner — 跨週期對齊 (向後相容)
    10. check_macd_surge — MACD 四箭頭
    11. 資料結構 (列舉, dataclass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy import (
    BigStockBuySignal,
    BigStockBuySignalV2,
    BigStockSellSignal,
    BigStockSellSignalV2,
    CapitalAllocator,
    CrashExitSignal,
    EntrySignal,
    MarketAssessment,
    MarketSignal,
    MarketSignalV2,
    MarketState,
    ScanReport,
    SectorLight,
    SectorSignal,
    SectorSignalV2,
    SignalAligner,
    SignalResult,
    StockSurgeSignal,
    TradeSignal,
    check_macd_surge,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def daily_bull() -> pd.DataFrame:
    """強勢多頭日線."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.015, len(dates)))).astype(np.float32)
    high = close * (1 + np.abs(np.random.normal(0.005, 0.005, len(dates)))).astype(np.float32)
    low = close * (1 - np.abs(np.random.normal(0.005, 0.005, len(dates)))).astype(np.float32)
    vol = np.random.randint(100_000, 500_000, len(dates), dtype=np.int64)
    open_p = close - (close * np.random.normal(0, 0.005, len(dates))).astype(np.float32)
    df = pd.DataFrame({
        "Date": dates, "Ticker": "2330",
        "Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol,
    })
    df["Adj_Open"] = df["Open"]; df["Adj_High"] = df["High"]
    df["Adj_Low"] = df["Low"]; df["Adj_Close"] = df["Close"]; df["Adj_Volume"] = df["Volume"]
    return df


@pytest.fixture
def weekly_bull() -> pd.DataFrame:
    """多頭週線."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="W-FRI")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.003, 0.02, len(dates)))).astype(np.float32)
    df = pd.DataFrame({
        "Date": dates, "Open": close * 0.99, "High": close * 1.02,
        "Low": close * 0.98, "Close": close.astype(np.float32),
        "Volume": np.random.randint(500_000, 2_000_000, len(dates)),
    })
    return df


@pytest.fixture
def monthly_bull() -> pd.DataFrame:
    """多頭月線."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=24, freq="ME")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.01, 0.04, len(dates)))).astype(np.float32)
    df = pd.DataFrame({
        "Date": dates, "Open": close * 0.985, "High": close * 1.03,
        "Low": close * 0.97, "Close": close.astype(np.float32),
        "Volume": np.random.randint(1_000_000, 5_000_000, len(dates)),
    })
    return df


@pytest.fixture
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# 1. MarketSignalV2 測試
# ═══════════════════════════════════════════════════════════════


class TestMarketSignalV2:
    def test_bull(self, daily_bull: pd.DataFrame):
        ms = MarketSignalV2()
        r = ms.assess(daily_bull)
        assert isinstance(r, MarketAssessment)
        assert r.score >= 0
        assert len(r.reasons) > 0

    def test_empty(self, empty_df: pd.DataFrame):
        ms = MarketSignalV2()
        r = ms.assess(empty_df)
        assert r.state == MarketState.ALERT

    def test_crash_detection(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        close = np.ones(100, dtype=np.float32) * 100
        for i in range(-5, 0):
            close[i] = close[i - 1] * 0.95
        df = pd.DataFrame({
            "Date": dates, "Ticker": "0050", "Open": close * 0.99,
            "High": close, "Low": close * 0.98, "Close": close,
            "Volume": np.ones(100) * 100_000,
        })
        for c in ["Adj_Open", "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume"]:
            df[c] = df[c.replace("Adj_", "")]
        ms = MarketSignalV2()
        r = ms.assess(df)
        assert "崩盤" in str(r.state) or "崩盤" in str(r.reasons)

    def test_low_turn_point(self, daily_bull: pd.DataFrame):
        """低轉折檢查 (合成資料應為非觸發)."""
        ms = MarketSignalV2()
        # 多頭資料不應觸發低轉折
        assert not ms.is_low_turn_point(daily_bull)


# ═══════════════════════════════════════════════════════════════
# 2. SectorSignalV2 測試
# ═══════════════════════════════════════════════════════════════


class TestSectorSignalV2:
    def test_assess_all(self, daily_bull: pd.DataFrame):
        ss = SectorSignalV2()
        results = ss.assess_all(daily_bull)
        assert isinstance(results, dict)
        assert len(results) == len(ss.DEFAULT_SECTORS)
        for name, light in results.items():
            assert isinstance(light, SectorLight)

    def test_yellow_on_empty(self, empty_df: pd.DataFrame):
        ss = SectorSignalV2()
        r = ss.assess("2330", empty_df)
        assert r == SectorLight.YELLOW


# ═══════════════════════════════════════════════════════════════
# 3. BigStockBuySignalV2 測試
# ═══════════════════════════════════════════════════════════════


class TestBigStockBuySignalV2:
    def test_buy_result_structure(self, daily_bull: pd.DataFrame):
        bbs = BigStockBuySignalV2()
        r = bbs.evaluate("2330", daily=daily_bull, weekly=daily_bull, monthly=daily_bull)
        assert isinstance(r, SignalResult)
        assert r.ticker == "2330"
        assert isinstance(r.conditions_met, list)
        assert isinstance(r.conditions_failed, list)
        assert 0 <= r.confidence <= 1.0

    def test_buy_empty_daily(self, empty_df: pd.DataFrame):
        bbs = BigStockBuySignalV2()
        r = bbs.evaluate("2330", daily=empty_df, weekly=empty_df, monthly=empty_df)
        assert r.signal == TradeSignal.NEUTRAL
        assert "無日線資料" in str(r.detail.get("error", ""))

    def test_buy_different_tickers(self, daily_bull: pd.DataFrame):
        bbs = BigStockBuySignalV2()
        r1 = bbs.evaluate("2330", daily=daily_bull, weekly=daily_bull, monthly=daily_bull)
        r2 = bbs.evaluate("2454", daily=daily_bull, weekly=daily_bull, monthly=daily_bull)
        assert r1.ticker == "2330"
        assert r2.ticker == "2454"


# ═══════════════════════════════════════════════════════════════
# 4. BigStockSellSignalV2 測試
# ═══════════════════════════════════════════════════════════════


class TestBigStockSellSignalV2:
    def test_sell_result_structure(self, daily_bull: pd.DataFrame):
        bss = BigStockSellSignalV2()
        r = bss.evaluate("2330", daily=daily_bull, weekly=daily_bull, monthly=daily_bull)
        assert isinstance(r, SignalResult)
        assert isinstance(r.conditions_met, list)
        assert isinstance(r.conditions_failed, list)
        assert 0 <= r.confidence <= 1.0

    def test_sell_empty_data(self, empty_df: pd.DataFrame):
        bss = BigStockSellSignalV2()
        r = bss.evaluate("2330", daily=empty_df, weekly=empty_df, monthly=empty_df)
        assert r.signal == TradeSignal.NEUTRAL
        assert "無日線資料" in str(r.detail.get("error", ""))

    def test_sell_empty_monthly(self, daily_bull: pd.DataFrame):
        """無月線時仍可產出."""
        bss = BigStockSellSignalV2()
        empty = pd.DataFrame()
        r = bss.evaluate("2330", daily=daily_bull, weekly=daily_bull, monthly=empty)
        assert isinstance(r, SignalResult)
        assert r.signal in (TradeSignal.HOLD, TradeSignal.NEUTRAL)


# ═══════════════════════════════════════════════════════════════
# 5. EntrySignal 測試
# ═══════════════════════════════════════════════════════════════


class TestEntrySignal:
    def test_low_entry_60m_buy(self):
        es = EntrySignal()
        r = es.check_low_entry_60m("2330", 30.0)
        assert r.signal == TradeSignal.BUY

    def test_low_entry_60m_neutral(self):
        es = EntrySignal()
        r = es.check_low_entry_60m("2330", 50.0)
        assert r.signal == TradeSignal.NEUTRAL

    def test_black_6k_no_monthly(self):
        es = EntrySignal()
        r = es.check_black_6k_buy("2330", pd.DataFrame())
        assert r.signal == TradeSignal.NEUTRAL

    def test_dif210_touch_floor_neutral(self, daily_bull: pd.DataFrame):
        es = EntrySignal()
        r = es.check_dif210_touch_floor("2330", daily_bull)
        assert isinstance(r, SignalResult)

    def test_dif210_touch_empty(self, empty_df: pd.DataFrame):
        es = EntrySignal()
        r = es.check_dif210_touch_floor("2330", empty_df)
        assert r.signal == TradeSignal.NEUTRAL


# ═══════════════════════════════════════════════════════════════
# 6. CrashExitSignal 測試
# ═══════════════════════════════════════════════════════════════


class TestCrashExitSignal:
    def test_daily_top_empty(self, empty_df: pd.DataFrame):
        ce = CrashExitSignal()
        r = ce.check_daily_top("2330", empty_df)
        assert r.signal == TradeSignal.NEUTRAL

    def test_monthly_k6k9_no_monthly(self):
        ce = CrashExitSignal()
        r = ce.check_monthly_k6k9_sell("2330", pd.DataFrame())
        assert r.signal == TradeSignal.NEUTRAL

    def test_monthly_k6k9_empty(self):
        ce = CrashExitSignal()
        r = ce.check_monthly_k6k9_sell("2330", pd.DataFrame())
        assert "月線資料不足" in str(r.conditions_failed)


# ═══════════════════════════════════════════════════════════════
# 7. StockSurgeSignal 測試
# ═══════════════════════════════════════════════════════════════


class TestStockSurgeSignal:
    def test_surge_empty(self, empty_df: pd.DataFrame):
        ss = StockSurgeSignal()
        r = ss.evaluate(empty_df)
        assert r["both_surge"] is False

    def test_surge_structure(self, daily_bull: pd.DataFrame):
        ss = StockSurgeSignal()
        r = ss.evaluate(daily_bull)
        for k in ("dif210_spiral", "adx300_spiral", "both_surge"):
            assert k in r
            assert isinstance(r[k], bool)


# ═══════════════════════════════════════════════════════════════
# 8. check_macd_surge 測試
# ═══════════════════════════════════════════════════════════════


class TestCheckMacdSurge:
    def test_surge_structure(self, daily_bull: pd.DataFrame):
        r = check_macd_surge(daily_bull["Adj_Close"])
        for k in ("short_all_up", "short_arrows", "long_dif_up", "long_arrows", "surge"):
            assert k in r


# ═══════════════════════════════════════════════════════════════
# 9. SignalAligner 測試 (向後相容)
# ═══════════════════════════════════════════════════════════════


class TestSignalAlignerCompat:
    def test_align_bull(self, daily_bull: pd.DataFrame, weekly_bull: pd.DataFrame,
                        monthly_bull: pd.DataFrame):
        aligner = SignalAligner()
        signal = SignalResult(ticker="2330", signal=TradeSignal.BUY)
        r = aligner.align("2330", signal, daily=daily_bull, weekly=weekly_bull, monthly=monthly_bull)
        assert isinstance(r, SignalResult)
        assert r.ticker == "2330"

    def test_align_empty(self, empty_df: pd.DataFrame):
        aligner = SignalAligner()
        signal = SignalResult(ticker="2330", signal=TradeSignal.BUY)
        r = aligner.align("2330", signal, daily=empty_df)
        assert isinstance(r, SignalResult)


# ═══════════════════════════════════════════════════════════════
# 10. CapitalAllocator 測試
# ═══════════════════════════════════════════════════════════════


class TestCapitalAllocator:
    def test_assess_index_strength(self, monthly_bull: pd.DataFrame):
        ca = CapitalAllocator()
        s = ca.assess_index_strength("0050", monthly_bull)
        assert 0 <= s <= 100


# ═══════════════════════════════════════════════════════════════
# 11. 資料結構測試
# ═══════════════════════════════════════════════════════════════


class TestDataStructures:
    def test_market_state_enum(self):
        assert MarketState.BULL == "多頭"
        assert MarketState.BEAR == "空頭"
        assert MarketState.ALERT == "備戰"
        assert MarketState.CRASH == "崩盤"

    def test_sector_light_enum(self):
        assert SectorLight.GREEN == "🟢 強勢"
        assert SectorLight.YELLOW == "🟡 中性"
        assert SectorLight.RED == "🔴 弱勢"

    def test_trade_signal_enum(self):
        assert TradeSignal.STRONG_BUY == "強烈買進"
        assert TradeSignal.NEUTRAL == "中立觀望"

    def test_signal_result_defaults(self):
        sr = SignalResult(ticker="2330")
        assert sr.signal == TradeSignal.NEUTRAL
        assert sr.confidence == 0.0
        assert sr.detail == {}

    def test_market_assessment_defaults(self):
        ma = MarketAssessment()
        assert ma.state == MarketState.ALERT
        assert ma.score == 0

    def test_scan_report_defaults(self):
        sr = ScanReport()
        assert sr.date == ""
        assert sr.summary == ""


# ═══════════════════════════════════════════════════════════════
# 12. 向後相容別名測試
# ═══════════════════════════════════════════════════════════════


class TestBackwardCompat:
    def test_market_signal_alias(self):
        assert MarketSignal is MarketSignalV2

    def test_sector_signal_alias(self):
        assert SectorSignal is SectorSignalV2

    def test_big_stock_buy_alias(self):
        assert BigStockBuySignal is BigStockBuySignalV2

    def test_big_stock_sell_alias(self):
        assert BigStockSellSignal is BigStockSellSignalV2

    def test_v2_importable(self):
        ms = MarketSignalV2()
        assert isinstance(ms, MarketSignalV2)
        bbs = BigStockBuySignalV2()
        assert isinstance(bbs, BigStockBuySignalV2)
