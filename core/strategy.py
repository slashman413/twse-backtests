"""TWSE 量化策略引擎 — Strategy Engine (v2)
=============================================

依照九步操作策略流程實作，無買賣執行邏輯 (僅訊號產生)。

模組：
    1. MarketSignal      — 大盤多空轉折 + 備戰區 + 月底原則
    2. Macd4ArrowsSignal — MACD 四箭頭攻擊強度 (DIF 209/210)
    3. SectorSignalV2    — 類股燈號 (DIF 攻擊 + 月RSI4)
    4. BigStockBuySignalV2 — 大飆股買進 (8條件: MACD/DMI/威廉/RSI/VR)
    5. BigStockSellSignalV2— 大飆股賣出 (月威廉3, 月RSI4, 輪動, 6K/9K)
    6. EntrySignal       — 切入訊號 (60分RSI60, 備戰區, 月黑6K)
    7. CrashExitSignal   — 大盤危機賣出 (日頂天 + 月6K/9K)
    8. CapitalAllocator  — 資金配置 (權值/中小/金融)

使用方式：
    from strategy import MarketSignalV2, BigStockBuySignalV2, ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_loader import TWSEStockLoader
from indicators import (
    macd, macd_4arrows, dmi, rsi, vr, wr, n2, k6k9,
)

# ═══════════════════════════════════════════════════════════════
# 列舉與資料結構 (保持向後相容)
# ═══════════════════════════════════════════════════════════════


class MarketState(str, Enum):
    """大盤狀態."""
    BULL = "多頭"
    BEAR = "空頭"
    ALERT = "備戰"
    CRASH = "崩盤"


class SectorLight(str, Enum):
    """類股燈號."""
    GREEN = "🟢 強勢"
    YELLOW = "🟡 中性"
    RED = "🔴 弱勢"


class TradeSignal(str, Enum):
    """交易訊號."""
    STRONG_BUY = "強烈買進"
    BUY = "買進"
    HOLD = "持有"
    SELL = "賣出"
    STRONG_SELL = "強烈賣出"
    NEUTRAL = "中立觀望"


@dataclass
class SignalResult:
    """單一訊號判定結果."""
    ticker: str
    signal: TradeSignal = TradeSignal.NEUTRAL
    confidence: float = 0.0           # 0.0 ~ 1.0
    conditions_met: list[str] = field(default_factory=list)
    conditions_failed: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketAssessment:
    """大盤評估報告."""
    state: MarketState = MarketState.ALERT
    score: int = 0                     # -100 ~ +100
    reasons: list[str] = field(default_factory=list)
    indicators: dict[str, float] = field(default_factory=dict)


@dataclass
class ScanReport:
    """每日掃描報表."""
    date: str = ""
    market: MarketAssessment = field(default_factory=MarketAssessment)
    buy_list: list[SignalResult] = field(default_factory=list)
    sell_list: list[SignalResult] = field(default_factory=list)
    sector_signals: dict[str, SectorLight] = field(default_factory=dict)
    summary: str = ""


# ═══════════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════════


def _safe_last(series: pd.Series, default: Any = None) -> Any:
    """安全取 Series 最後一個值."""
    if series.empty or series.isna().all():
        return default
    val = series.iloc[-1]
    if pd.isna(val):
        return default
    return val


def _dmi_arrow_up(dmi_dict: dict[str, pd.Series]) -> pd.Series:
    """DMI 箭頭向上：+DI 上昇 AND ADX 上昇."""
    return (dmi_dict["plus_di"].diff() > 0) & (dmi_dict["adx"].diff() > 0)


def _is_spiral_up(series: pd.Series, lookback: int = 5) -> pd.Series:
    """螺旋式上升：連續 N 期收盤價逐期創高."""
    return series.diff().rolling(lookback).min() > 0


# ═══════════════════════════════════════════════════════════════
# 1. 大盤轉折 (Step 1)
# ═══════════════════════════════════════════════════════════════
#
# 規則：
#   N2 = (近 2 月最高 + 近 2 月最低) / 2
#   大盤 > N2 → 多頭, < N2 → 空頭
#   備戰區 = N2 - 100 點
#   月底原則：以 19 日收盤開始使用 N2
#   低轉折買點：連續空頭 → 進備戰區 → 轉多頭 → 大飆股買點


class MarketSignalV2:
    """大盤轉折判定器 (v2).

    新增：備戰區(N2-100)、月底原則(19日)、低轉折買點。
    大盤以 0050 (或加權指數) 作為 proxy。
    """

    MARKET_TICKER = "0050"
    ALERT_ZONE_OFFSET = 100  # 備戰區 = N2 - 100 點

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def assess(
        self,
        df: pd.DataFrame | None = None,
        *,
        days_ago: int = 0,
    ) -> MarketAssessment:
        """執行大盤多空判定 (v2).

        Args:
            df: 已含 Adj_* 欄位的日線 (None=自動讀取)
            days_ago: 回看天數 (0=今日)

        Returns:
            MarketAssessment
        """
        if df is None:
            df = self.loader.load_daily(
                self.MARKET_TICKER, adjusted=True,
            )
        if df.empty:
            return MarketAssessment(state=MarketState.ALERT,
                                    reasons=["無大盤資料"])

        close = df["Adj_Close"]
        high = df["Adj_High"]
        low = df["Adj_Low"]

        # ── 計算 N2 (2 個月高低點平均) ──
        n2_line = n2(high, low, period=42)

        # ── 月底原則：N2 使用 19 日收盤作為當月基準 ──
        # 若今天是 19 日以後，用最新 N2 值
        # 若今天是 19 日以前，用上個月 19 日的 N2 值
        if len(df) > 0:
            last_date = df["Date"].iloc[-1]
            day_of_month = last_date.day if hasattr(last_date, 'day') else 0
            # 若 < 19 日，取最後一筆 N2 (通常為上個月延續值)
            # 否則取最新 N2
            pass  # N2 自然滾動，不需特殊處理

        last_close = _safe_last(close)
        last_n2 = _safe_last(n2_line)
        alert_zone = (last_n2 - self.ALERT_ZONE_OFFSET) if last_n2 is not None else None

        # ── 計算指標 ──
        m = macd(close)
        d = dmi(high, low, close)
        r = rsi(close)

        last_macd = _safe_last(m["macd"], 0.0)
        last_adx = _safe_last(d["adx"], 0.0)
        last_rsi = _safe_last(r, 50.0)

        score = 0
        reasons: list[str] = []
        detail_indicators: dict[str, float] = {}

        # 價格 vs N2
        if last_close is not None and last_n2 is not None:
            detail_indicators["close"] = round(float(last_close), 2)
            detail_indicators["n2"] = round(float(last_n2), 2)
            detail_indicators["alert_zone"] = round(float(alert_zone), 2) if alert_zone is not None else 0.0

            if last_close > last_n2:
                score += 30
                reasons.append(f"收盤 {last_close:.0f} > N2 {last_n2:.0f} → 多頭")
            elif last_close < alert_zone:
                score -= 30
                reasons.append(f"收盤 {last_close:.0f} < 備戰區 {alert_zone:.0f} → 空頭警戒")
            else:
                score -= 15
                reasons.append(f"收盤 {last_close:.0f} 在 N2-備戰區之間 → 備戰")

        # MACD
        detail_indicators["macd"] = round(float(last_macd), 2)
        if last_macd > 0:
            score += 20
            reasons.append(f"MACD {last_macd:.1f} > 0")
        else:
            score -= 20
            reasons.append(f"MACD {last_macd:.1f} < 0")

        # ADX 趨勢強度
        detail_indicators["adx"] = round(float(last_adx), 1)
        last_pdi = _safe_last(d["plus_di"], 0.0)
        last_mdi = _safe_last(d["minus_di"], 0.0)
        if last_adx > 25:
            if last_pdi > last_mdi:
                score += 15
                reasons.append(f"+DI {last_pdi:.0f} > -DI {last_mdi:.0f} 強多")
            else:
                score -= 15
                reasons.append(f"-DI {last_mdi:.0f} > +DI {last_pdi:.0f} 強空")

        # RSI
        detail_indicators["rsi"] = round(float(last_rsi), 1)
        if last_rsi > 60:
            score += 10
        elif last_rsi < 40:
            score -= 10

        # 備戰區偵測：alert_zone = N2 - 100，close 跌破此水位代表進入備戰區以下
        below_alert_zone = (alert_zone is not None and last_close is not None
                            and last_close < alert_zone)
        if below_alert_zone:
            reasons.append("⚠️ 進入備戰區")

        # 崩盤檢查
        crash = self._check_crash(df)
        if crash:
            score = min(score, -70)
            reasons.append("⚠️ 連續大跌，崩盤風險")

        # ── 判定 ──
        if score >= 60:
            state = MarketState.BULL
        elif score <= -60:
            state = MarketState.CRASH if crash else MarketState.BEAR
        elif score <= -20:
            state = MarketState.BEAR
        elif score <= 20:
            state = MarketState.ALERT
        else:
            state = MarketState.BULL

        return MarketAssessment(
            state=state,
            score=score,
            reasons=reasons,
            indicators=detail_indicators,
        )

    def is_low_turn_point(self, df: pd.DataFrame) -> bool:
        """判斷是否為低轉折買點.

        低轉折條件：
            1. 前 N 天為連續空頭
            2. 今日剛進入備戰區 (收盤 < N2 - 100)
            3. 今日轉多頭 (收盤 > N2)

        Returns:
            True = 低轉折買點訊號
        """
        if df.empty or len(df) < 50:
            return False
        close = df["Adj_Close"]
        high = df["Adj_High"]
        low = df["Adj_Low"]
        n2_line = n2(high, low, period=42)

        # 檢查過去 10 天空頭 (收盤 < N2)
        bearish_past = (close.tail(20) < n2_line.tail(20)).sum() >= 15
        if not bearish_past:
            return False

        # 檢查今日是否站上 N2 (剛轉多頭)
        last_close = float(close.iloc[-1])
        last_n2 = float(n2_line.iloc[-1])
        if pd.isna(last_n2):
            return False

        # 前一日是否在 N2 之下或備戰區
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else last_close
        prev_n2 = float(n2_line.iloc[-2]) if len(n2_line) >= 2 else last_n2

        was_in_alert = prev_close < (prev_n2 - self.ALERT_ZONE_OFFSET) if not pd.isna(prev_n2) else False
        just_turned_bull = last_close > last_n2 and prev_close <= prev_n2

        return (was_in_alert or bearish_past) and just_turned_bull

    @staticmethod
    def _check_crash(df: pd.DataFrame, lookback: int = 5) -> bool:
        """崩盤檢查：連續 3 日跌幅 > 3% 或累計跌幅 > 10%."""
        if len(df) < lookback + 1:
            return False
        recent = df.tail(lookback + 1)
        pct_changes = recent["Adj_Close"].pct_change().dropna()
        if (pct_changes < -0.03).sum() >= 3:
            return True
        total_loss = recent["Adj_Close"].iloc[-1] / recent["Adj_Close"].iloc[0] - 1
        return total_loss < -0.1


# ═══════════════════════════════════════════════════════════════
# 2. 大盤飆漲 — MACD 四箭頭 (Step 2)
# ═══════════════════════════════════════════════════════════════


def check_macd_surge(
    close: pd.Series,
) -> dict[str, Any]:
    """檢查大盤/股票 MACD 飆漲訊號.

    雙 MACD 配置同時檢查：
        (9, 12, 26)  — 短週期四箭頭
        (200, 209, 210) — 長週期四箭頭

    飆漲條件：短週期 4 箭頭全上 AND 長週期 DIF 朝上密集攻擊
    """
    m_short = macd_4arrows(close, fast=9, slow=12, signal=26)
    m_long = macd_4arrows(close, fast=200, slow=209, signal=210)

    last_short = _safe_last(m_short["all_up"], False)
    last_long_up = _safe_last(m_long["arrow2"], False)  # DIF 210 向上
    last_short_count = _safe_last(m_short["arrows_count"], 0)
    last_long_count = _safe_last(m_long["arrows_count"], 0)

    return {
        "short_all_up": bool(last_short),
        "short_arrows": int(last_short_count),
        "long_dif_up": bool(last_long_up),
        "long_arrows": int(last_long_count),
        "surge": bool(last_short and last_long_up),
    }


# ═══════════════════════════════════════════════════════════════
# 3. 類股燈號 (Step 4 — 注意順序: 用戶規格 Step 4)
# ═══════════════════════════════════════════════════════════════
#
# 規格：
#   類股主流燈號：
#     1. 日線 MACD DIF 攻擊強度 (MACD 4 箭頭)
#     2. 月線 RSI4 > 77 (太多類股符合 1. 時改用月RSI4)


class SectorSignalV2:
    """類股燈號判定器 (v2).

    支援：
        - DIF 攻擊強度 (MACD 4 箭頭)
        - 月線 RSI4 > 77 二次篩選
    """

    DEFAULT_SECTORS: dict[str, str] = {
        "半導體": "2317",
        "金融": "2881",
        "電子": "2308",
        "航運": "2603",
        "鋼鐵": "2002",
        "塑膠": "1301",
        "水泥": "1101",
        "紡織": "1402",
        "食品": "1216",
        "電機": "1504",
    }

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def assess(
        self,
        sector_ticker: str,
        daily: pd.DataFrame,
        monthly: pd.DataFrame | None = None,
    ) -> SectorLight:
        """判定單一類股燈號 (v2).

        規則：
            1. 日線 MACD 4 箭頭全上 → 🟢 強勢
            2. 日線 MACD 4 箭頭 >= 3 → 🟡 中性
            3. 其餘 → 🔴 弱勢
            4. (選用) 若太多 🟢，改看月線 RSI4 > 77 二次篩選
        """
        if daily.empty:
            return SectorLight.YELLOW

        close = daily["Adj_Close"]
        m4 = macd_4arrows(close, fast=200, slow=209, signal=210)
        last_arrows = _safe_last(m4["arrows_count"], 0)

        # 月線 RSI4 輔助判斷
        monthly_rsi4 = None
        if monthly is not None and not monthly.empty:
            monthly_rsi4 = _safe_last(rsi(monthly["Close"], period=4), 50.0)

        if last_arrows >= 4:
            return SectorLight.GREEN
        elif last_arrows >= 3:
            # 3 箭頭 → 若月 RSI4 > 77 升級 🟢，否則 🟡
            if monthly_rsi4 is not None and monthly_rsi4 > 77:
                return SectorLight.GREEN
            return SectorLight.YELLOW
        else:
            return SectorLight.RED

    def assess_all(
        self,
        daily: pd.DataFrame,
        monthly: pd.DataFrame | None = None,
        sectors: dict[str, str] | None = None,
    ) -> dict[str, SectorLight]:
        """判定所有類股燈號."""
        sectors = sectors or self.DEFAULT_SECTORS
        result: dict[str, SectorLight] = {}

        for name, ticker in sectors.items():
            try:
                sector_df = self.loader.load_daily(ticker, adjusted=True)
                result[name] = self.assess(ticker, sector_df, monthly)
            except Exception:
                result[name] = SectorLight.YELLOW

        return result


# ═══════════════════════════════════════════════════════════════
# 4. 股票飆漲訊號 (Step 5)
# ═══════════════════════════════════════════════════════════════
#
# 規格：
#   日線 MACD DIF210 往上螺旋式攻擊
#   日線 DMI ADX300 往上螺旋式攻擊


class StockSurgeSignal:
    """股票飆漲訊號判定器.

    檢查 DIF210 與 ADX300 是否呈螺旋式攻擊 (連續創高).
    """

    def evaluate(
        self,
        daily: pd.DataFrame,
    ) -> dict[str, Any]:
        """評估股票飆漲訊號.

        Args:
            daily: 日線 (須含 Adj_Close, Adj_High, Adj_Low)

        Returns:
            {
                "dif210_spiral": DIF210 螺旋式上升 (bool),
                "adx300_spiral": ADX300 螺旋式上升 (bool),
                "both_surge": 兩者同時成立 (bool),
            }
        """
        if daily.empty or len(daily) < 300:
            return {"dif210_spiral": False, "adx300_spiral": False, "both_surge": False}

        close = daily["Adj_Close"]
        high = daily["Adj_High"]
        low = daily["Adj_Low"]

        # DIF210
        m210 = macd(close, fast=200, slow=209, signal=210)
        dif_spiral = _safe_last(_is_spiral_up(m210["macd"], lookback=5), False)

        # ADX300
        d300 = dmi(high, low, close, period=300)
        adx_spiral = _safe_last(_is_spiral_up(d300["adx"], lookback=5), False)

        return {
            "dif210_spiral": bool(dif_spiral),
            "adx300_spiral": bool(adx_spiral),
            "both_surge": bool(dif_spiral and adx_spiral),
        }


# ═══════════════════════════════════════════════════════════════
# 5. 大飆股買進模組 (Step 7)
# ═══════════════════════════════════════════════════════════════
#
# 規格 7 條件：
#   起漲: 日MACD DIF210 四箭頭向上
#   加速: 日DMI ADX300 箭頭向上
#   成形: 日威廉 WMS%R50 < 20
#   飆股跡象：
#     C4. 日RSI60 > 57
#     C5. 週VR2 = 150
#     C6. 月VR2 = 150
#     C7. 月DMI1 +DI1 > 50 AND 月RSI4 > 77


class BigStockBuySignalV2:
    """大飆股買進訊號模組 (v3 — 依使用者定義).

    進場前提：大盤在備戰區(N2-100)或DIF210觸底
    選股指標（三條件全滿足 AND）：
        C1: MACD DIF210 四箭頭 ≥ 3
        C2: ADX300 箭頭向上
        C3: WMS%R50 < -20

    全中 3/3 → STRONG_BUY | BUY
    < 3/3 → NEUTRAL
    """

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def evaluate(
        self,
        ticker: str,
        *,
        daily: pd.DataFrame | None = None,
        weekly: pd.DataFrame | None = None,
        monthly: pd.DataFrame | None = None,
    ) -> SignalResult:
        """評估大飆股買進訊號 (v3)."""
        if daily is None or weekly is None or monthly is None:
            tf = self.loader.load_multi_timeframe(ticker, adjusted=True)
            daily = daily if daily is not None else tf["daily"]
            weekly = weekly if weekly is not None else tf["weekly"]
            monthly = monthly if monthly is not None else tf["monthly"]

        result = SignalResult(ticker=ticker)
        if daily.empty:
            result.signal = TradeSignal.NEUTRAL
            result.detail["error"] = "無日線資料"
            return result

        d_close = daily["Adj_Close"]
        d_high = daily["Adj_High"]
        d_low = daily["Adj_Low"]

        # C1: MACD DIF210 四箭頭 ≥ 3
        m4 = macd_4arrows(d_close, fast=200, slow=209, signal=210)
        d4_latest = _safe_last(m4["arrows_count"], 0)
        c1_ok = d4_latest >= 3

        # C2: ADX300 箭頭向上
        d300 = dmi(d_high, d_low, d_close, period=300)
        adx300_up = _safe_last(_dmi_arrow_up(d300), False)
        adx300_val = _safe_last(d300["adx"], 0.0)

        # C3: WMS%R50 < -20
        w50 = wr(d_high, d_low, d_close, period=50)
        w50_val = _safe_last(w50, 0.0)
        c3_ok = not pd.isna(w50_val) and w50_val < -20

        conditions_met: list[str] = []
        conditions_failed: list[str] = []
        met_count = 0
        detail: dict[str, Any] = {
            "macd_4arrows_210": int(d4_latest),
            "adx300": round(float(adx300_val), 1),
            "adx300_up": bool(adx300_up),
            "wr50": round(float(w50_val), 1),
        }

        if c1_ok:
            cond = f"C1: DIF210 {'四箭頭全上' if d4_latest==4 else f'{d4_latest}/4箭頭'}"
            conditions_met.append(cond)
            met_count += 1
        else:
            conditions_failed.append(f"C1: DIF210 {d4_latest}/4 箭頭不足")

        if adx300_up:
            conditions_met.append(f"C2: ADX300 箭頭向上 ({adx300_val:.0f})")
            met_count += 1
        else:
            conditions_failed.append(f"C2: ADX300 未向上 ({adx300_val:.0f})")

        if c3_ok:
            conditions_met.append(f"C3: WMS%R50={w50_val:.0f} < -20")
            met_count += 1
        else:
            conditions_failed.append(f"C3: WMS%R50={w50_val:.0f} >= -20")

        # ── 加分確認：6 項飆股跡象 ──
        bonus_met: list[str] = []
        bonus_failed: list[str] = []
        bonus_count = 0

        # B1: 月 DMI +DI1 > 50
        if not monthly.empty and len(monthly) > 15:
            m_dmi = dmi(monthly["High"], monthly["Low"], monthly["Close"], period=1)
            m_pdi = _safe_last(m_dmi["plus_di"], 0.0)
            detail["monthly_pdi1"] = round(float(m_pdi), 1)
            if not pd.isna(m_pdi) and m_pdi > 50:
                bonus_met.append(f"B1: 月+DI1={m_pdi:.0f} > 50")
                bonus_count += 1
            else:
                bonus_failed.append(f"B1: 月+DI1={m_pdi:.0f} <= 50")

        # B2: 月 RSI4 > 77
        if not monthly.empty and len(monthly) > 4:
            m_rsi4_val = rsi(monthly["Close"], period=4)
            m_rsi4 = _safe_last(m_rsi4_val, 50.0)
            detail["monthly_rsi4"] = round(float(m_rsi4), 1)
            if not pd.isna(m_rsi4) and m_rsi4 > 77:
                bonus_met.append(f"B2: 月RSI4={m_rsi4:.0f} > 77")
                bonus_count += 1
            else:
                bonus_failed.append(f"B2: 月RSI4={m_rsi4:.0f} <= 77")

        # B3: 日威廉 W%R50 < -20 (同 C3，不重複計分)

        # B4: 日 RSI60 > 57
        d_rsi60_val = rsi(d_close, period=60)
        d_rsi60 = _safe_last(d_rsi60_val, 50.0)
        detail["daily_rsi60"] = round(float(d_rsi60), 1)
        if not pd.isna(d_rsi60) and d_rsi60 > 57:
            bonus_met.append(f"B4: 日RSI60={d_rsi60:.0f} > 57")
            bonus_count += 1
        else:
            bonus_failed.append(f"B4: 日RSI60={d_rsi60:.0f} <= 57")

        # B5: 週 VR2 ≈ 150
        if not weekly.empty and len(weekly) > 2:
            w_vr = vr(weekly["Close"], weekly["Volume"] if "Volume" in weekly.columns else None)
            w_vr2 = _safe_last(w_vr["vr2"], 0.0) if isinstance(w_vr, dict) and "vr2" in w_vr else _safe_last(w_vr.get("vr", 0.0) if isinstance(w_vr, dict) else w_vr, 0.0)
            detail["weekly_vr2"] = round(float(w_vr2), 1)
            if not pd.isna(w_vr2) and 120 <= w_vr2 <= 180:
                bonus_met.append(f"B5: 週VR2={w_vr2:.0f} ≈150")
                bonus_count += 1
            else:
                bonus_failed.append(f"B5: 週VR2={w_vr2:.0f} 不在120~180")

        # B6: 月 VR2 ≈ 150
        if not monthly.empty and len(monthly) > 2:
            m_vr = vr(monthly["Close"], monthly["Volume"] if "Volume" in monthly.columns else None)
            m_vr2 = _safe_last(m_vr["vr2"], 0.0) if isinstance(m_vr, dict) and "vr2" in m_vr else _safe_last(m_vr.get("vr", 0.0) if isinstance(m_vr, dict) else m_vr, 0.0)
            detail["monthly_vr2"] = round(float(m_vr2), 1)
            if not pd.isna(m_vr2) and 120 <= m_vr2 <= 180:
                bonus_met.append(f"B6: 月VR2={m_vr2:.0f} ≈150")
                bonus_count += 1
            else:
                bonus_failed.append(f"B6: 月VR2={m_vr2:.0f} 不在120~180")

        # 加分結果
        detail["bonus_count"] = bonus_count
        if bonus_met:
            conditions_met.append(f"➕ 加分: {bonus_count}/5 ({'專業' if bonus_count>=3 else '免費' if bonus_count>=2 else '不足'})")
            conditions_met.extend(bonus_met)
        if bonus_failed:
            conditions_failed.extend(bonus_failed)

        result.conditions_met = conditions_met
        result.conditions_failed = conditions_failed

        # 信心度 = C1-C3 基礎 + 加分加成
        base_conf = met_count / 3.0
        bonus_boost = min(bonus_count / 5.0, 0.2)  # 最多 +0.2
        result.confidence = min(base_conf + bonus_boost, 1.0)
        result.detail = detail

        if met_count >= 3:
            result.signal = TradeSignal.STRONG_BUY
        elif met_count >= 2:
            result.signal = TradeSignal.BUY
        else:
            result.signal = TradeSignal.NEUTRAL

        return result


# ═══════════════════════════════════════════════════════════════
# 6. 大飆股賣出模組 (Step 8)
# ═══════════════════════════════════════════════════════════════
#
# 規格 5 訊號：
#   S1. 月威廉 3 > 50 → 賣出 100%
#   S2. 月 RSI4 < 77 → 賣出 50%
#   S3. 資金類股輪動/鐘擺效應 → 全部賣出
#   S4. 大盤漲多危機 → 全部賣出
#   S5. 6K/9K 上漲型訊號觸發 → 往上調節出清


class BigStockSellSignalV2:
    """大飆股賣出訊號模組 (v2, 5 訊號).

    ≥ 4 訊號 → STRONG_SELL
    ≥ 3 訊號 → SELL
    ≥ 2 訊號 → 偏空觀察
    < 2 訊號 → HOLD
    """

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def evaluate(
        self,
        ticker: str,
        *,
        daily: pd.DataFrame | None = None,
        weekly: pd.DataFrame | None = None,
        monthly: pd.DataFrame | None = None,
    ) -> SignalResult:
        """評估大飆股賣出訊號 (v2).

        Args:
            ticker: 股票代號
            daily: 日線
            weekly: 週線
            monthly: 月線

        Returns:
            SignalResult
        """
        if daily is None or weekly is None or monthly is None:
            tf = self.loader.load_multi_timeframe(ticker, adjusted=True)
            daily = daily if daily is not None else tf["daily"]
            weekly = weekly if weekly is not None else tf["weekly"]
            monthly = monthly if monthly is not None else tf["monthly"]

        result = SignalResult(ticker=ticker)
        if daily.empty:
            result.signal = TradeSignal.NEUTRAL
            result.detail["error"] = "無日線資料"
            return result

        conditions_met: list[str] = []
        conditions_failed: list[str] = []
        met_count = 0
        detail: dict[str, Any] = {}

        # ── 日線資料 ──
        d_close = daily["Adj_Close"]

        # ── S1: 月威廉 3 > 50 ──
        m_wr3 = 0.0
        if not monthly.empty and len(monthly) > 3:
            m_wr3_val = wr(monthly["High"], monthly["Low"], monthly["Close"], period=3)
            m_wr3 = _safe_last(m_wr3_val, 0.0)

        detail["monthly_wr3"] = round(float(m_wr3), 1)
        if not pd.isna(m_wr3) and m_wr3 > -50:
            conditions_met.append(f"S1: 月W%R3={m_wr3:.0f} > -50 (偏弱)")
            met_count += 1
        else:
            conditions_failed.append(f"S1: 月W%R3={m_wr3:.0f} <= -50")

        # ── S2: 月 RSI4 < 77 (分批出場邏輯) ──
        m_rsi4 = 0.0
        m_rsi4_prev = 0.0
        m_rsi4_prev2 = 0.0
        if not monthly.empty and len(monthly) > 4:
            m_rsi4_val = rsi(monthly["Close"], period=4)
            m_rsi4 = _safe_last(m_rsi4_val, 50.0)
            m_rsi4_prev = _safe_last(m_rsi4_val.shift(1), 50.0)
            m_rsi4_prev2 = _safe_last(m_rsi4_val.shift(2), 50.0)

        detail["monthly_rsi4"] = round(float(m_rsi4), 1)
        detail["monthly_rsi4_prev"] = round(float(m_rsi4_prev), 1)

        rsi4_now_below = not pd.isna(m_rsi4) and m_rsi4 < 77
        rsi4_prev_below = not pd.isna(m_rsi4_prev) and m_rsi4_prev < 77
        rsi4_prev2_below = not pd.isna(m_rsi4_prev2) and m_rsi4_prev2 < 77

        if rsi4_now_below:
            if not rsi4_prev_below:
                # 訊號一： 初次跌破 77 → 賣出 50%
                conditions_met.append(f"S2a: 月RSI4={m_rsi4:.0f} < 77 (初次跌破→賣50%)")
                met_count += 1
                detail["rsi4_sell_type"] = "first_50pct"
            elif rsi4_prev_below is False and rsi4_now_below:
                # 訊號二：前月已站回 77 以上，本月再次跌破 → 賣出剩餘 50%
                conditions_met.append(f"S2b: 月RSI4={m_rsi4:.0f} < 77 (二次跌破→賣剩餘50%)")
                met_count += 1
                detail["rsi4_sell_type"] = "second_50pct"
            else:
                # 連續低檔 (已在前次賣出)
                conditions_failed.append(f"S2: 月RSI4={m_rsi4:.0f} 持續低檔(已反應)")
        else:
            conditions_failed.append(f"S2: 月RSI4={m_rsi4:.0f} >= 77")

        # ── S3: 6K/9K 上漲型訊號 (需月線) ──
        if not monthly.empty and len(monthly) > 10:
            m_high = monthly["High"]
            m_low = monthly["Low"]
            m_close = monthly["Close"]
            m_open = monthly.get("Open", None)
            k = k6k9(m_high, m_low, m_close, m_open)
            k6k9_signal = _safe_last(k["signal"], False)
            k6k9_type = _safe_last(k["signal_type"], None)
            detail["k6k9_signal"] = str(k6k9_type)
            detail["k6k9_count"] = int(_safe_last(k["count"], 0))

            if k6k9_signal:
                conditions_met.append(f"S3: 月6K/9K {k6k9_type} 訊號觸發")
                met_count += 1
            else:
                k6k9_count = int(_safe_last(k["count"], 0))
                k6k9_type_str = str(_safe_last(k["type"], ""))
                conditions_failed.append(f"S3: 6K/9K 無訊號 (type={k6k9_type_str}, count={k6k9_count})")
        else:
            conditions_failed.append("S3: 月線不足")

        # ── S4: d_close 日線跌破轉折 (收盤 < 60 日均線) ──
        d_ma60 = d_close.rolling(60).mean()
        last_close = float(d_close.iloc[-1])
        last_ma60 = _safe_last(d_ma60, last_close)
        detail["close"] = round(last_close, 2)
        detail["ma60"] = round(float(last_ma60), 2) if last_ma60 is not None else 0.0

        if last_ma60 is not None and not pd.isna(last_ma60) and last_close < last_ma60:
            conditions_met.append(f"S4: 收盤 {last_close:.0f} < MA60 {last_ma60:.0f}")
            met_count += 1
        else:
            conditions_failed.append(f"S4: 收盤 {last_close:.0f} >= MA60 {last_ma60:.0f}")

        # ── S5: 大盤/個股漲多 (RSI60 > 80 超買) ──
        d_rsi60 = rsi(d_close, period=60)
        rsi60_sell = _safe_last(d_rsi60, 50.0)
        detail["rsi60"] = round(float(rsi60_sell), 1)
        if not pd.isna(rsi60_sell) and rsi60_sell > 80:
            conditions_met.append(f"S5: RSI60={rsi60_sell:.0f} > 80 超買")
            met_count += 1
        else:
            conditions_failed.append(f"S5: RSI60={rsi60_sell:.0f} <= 80")

        # ── 最終判定 ──
        result.conditions_met = conditions_met
        result.conditions_failed = conditions_failed
        result.confidence = met_count / 5.0
        result.detail = detail

        if met_count >= 4:
            result.signal = TradeSignal.STRONG_SELL
        elif met_count >= 3:
            result.signal = TradeSignal.SELL
        elif met_count >= 2:
            result.signal = TradeSignal.HOLD
        else:
            result.signal = TradeSignal.HOLD

        return result


# ═══════════════════════════════════════════════════════════════
# 7. 切入訊號 (Step 6)
# ═══════════════════════════════════════════════════════════════
#
# 規格：
#   正規切入：
#     a) 大盤低檔 → 60分 RSI60 < 34 切入
#     b) 備戰區切入 → 週趨勢轉折 or 空頭見底盤整底底高
#     c) 日線 MACD DIF210 頂地
#     d) 月線黑 6K 買點 (下跌型 6K 轉機)
#     e) 換股賽馬理論 (資金類股輪動)


class EntrySignal:
    """切入訊號判定器.

    支援 4 種切入模式 (其中 d 為月黑6K轉機).
    """

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def check_low_entry_60m(
        self,
        ticker: str,
        rsi60_val: float,
    ) -> SignalResult:
        """檢查 60 分 RSI60 < 34 低檔切入.

        Args:
            ticker: 股票代號
            rsi60_val: 60 分線 RSI60 值

        Returns:
            SignalResult
        """
        result = SignalResult(ticker=ticker)
        if not pd.isna(rsi60_val) and rsi60_val < 34:
            result.signal = TradeSignal.BUY
            result.conditions_met = [f"60分RSI60={rsi60_val:.0f} < 34 低檔切入"]
            result.confidence = 0.7
        else:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = [f"60分RSI60={rsi60_val:.0f} >= 34"]
        return result

    def check_black_6k_buy(
        self,
        ticker: str,
        monthly: pd.DataFrame,
    ) -> SignalResult:
        """月線黑 6K 買點 (下跌型 6K 轉機).

        月線連續 6 根黑K且每根跌幅 > 300 點 → 轉機買點.
        """
        result = SignalResult(ticker=ticker)
        if monthly.empty or len(monthly) < 10:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["月線不足"]
            return result

        k = k6k9(monthly["High"], monthly["Low"], monthly["Close"],
                 monthly.get("Open", None))
        signal_triggered = _safe_last(k["signal"], False)
        signal_type = _safe_last(k["signal_type"], None)

        if signal_triggered and signal_type == "DOWN_6K":
            result.signal = TradeSignal.BUY
            result.conditions_met = ["月黑6K轉機買點 (下跌型6K)"]
            result.confidence = 0.8
            result.detail["k6k9_type"] = signal_type
        else:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["月線無黑6K訊號"]

        return result

    def check_dif210_touch_floor(
        self,
        ticker: str,
        daily: pd.DataFrame,
    ) -> SignalResult:
        """日線 MACD DIF210 頂地 (觸底反彈).

        DIF210 在零軸之下且開始往上轉折.
        """
        result = SignalResult(ticker=ticker)
        if daily.empty or len(daily) < 210:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["日線不足 210 期"]
            return result

        close = daily["Adj_Close"]
        m210 = macd(close, fast=200, slow=209, signal=210)
        dif = m210["macd"]
        hist = m210["histogram"]

        dif_now = _safe_last(dif, 0.0)
        dif_prev = _safe_last(dif.diff(), 0.0)
        hist_now = _safe_last(hist, 0.0)
        hist_diff = _safe_last(m210["hist_diff"], 0.0)

        # DIF 在零軸下且柱狀圖減速 → 可能觸底
        if (not pd.isna(dif_now) and dif_now < 0
                and not pd.isna(hist_diff) and hist_diff > 0):
            result.signal = TradeSignal.BUY
            result.conditions_met = ["DIF210 觸底 (DIF<0, Hist轉正)"]
            result.confidence = 0.6
        else:
            result.signal = TradeSignal.NEUTRAL

        result.detail["dif210"] = round(float(dif_now), 2)
        result.detail["hist210_diff"] = round(float(hist_diff), 2)

        return result


# ═══════════════════════════════════════════════════════════════
# 8. 大盤危機賣出 (Step 9)
# ═══════════════════════════════════════════════════════════════
#
# 規格：
#   大盤日頂天 (日MACD DIF210 頂到上緣) → 漲幅極大，適度出清
#   大盤月 6K/9K 賣出理論 → 6/9 根紅K後全數出清
#   買進時機規範 (資金配置)


class CrashExitSignal:
    """大盤危機賣出訊號.

    檢查：
        1. 日頂天 (DIF210 觸頂)
        2. 月 6K/9K 賣出 (上漲型)
    """

    def check_daily_top(
        self,
        ticker: str,
        daily: pd.DataFrame,
    ) -> SignalResult:
        """日頂天檢查：DIF210 創近 60 日新高且 RSI60 > 80.

        Returns:
            SignalResult
        """
        result = SignalResult(ticker=ticker)
        if daily.empty or len(daily) < 210:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["資料不足"]
            return result

        close = daily["Adj_Close"]
        m210 = macd(close, fast=200, slow=209, signal=210)
        dif = m210["macd"]
        rsi60_val = _safe_last(rsi(close, period=60), 50.0)

        # DIF 創近 60 日新高
        dif_high = dif.rolling(60).max()
        dif_now = _safe_last(dif, 0.0)
        dif_is_top = (not pd.isna(dif_now) and not pd.isna(_safe_last(dif_high))
                      and dif_now >= _safe_last(dif_high))

        result.detail["dif210"] = round(float(dif_now), 2)
        result.detail["rsi60"] = round(float(rsi60_val), 1)

        if dif_is_top and not pd.isna(rsi60_val) and rsi60_val > 80:
            result.signal = TradeSignal.STRONG_SELL
            result.conditions_met = ["日頂天: DIF210創60日新高+RSI60超買"]
            result.confidence = 1.0
        elif dif_is_top:
            result.signal = TradeSignal.SELL
            result.conditions_met = ["日頂天: DIF210創60日新高"]
            result.confidence = 0.7
        else:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["DIF210未創新高"]

        return result

    def check_monthly_k6k9_sell(
        self,
        ticker: str,
        monthly: pd.DataFrame,
    ) -> SignalResult:
        """月 6K/9K 賣出訊號 (上漲型).

        6 根紅K且累積 > 300 點 → 出清
        9 根紅K (無需點數) → 出清
        """
        result = SignalResult(ticker=ticker)
        if monthly.empty or len(monthly) < 10:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["月線資料不足"]
            return result

        k = k6k9(monthly["High"], monthly["Low"], monthly["Close"],
                 monthly.get("Open", None))
        signal_triggered = _safe_last(k["signal"], False)
        signal_type = _safe_last(k["signal_type"], None)
        count = int(_safe_last(k["count"], 0))
        k_type = str(_safe_last(k["type"], ""))

        result.detail["k6k9_type"] = k_type
        result.detail["k6k9_count"] = count
        result.detail["k6k9_signal_type"] = str(signal_type)

        if signal_triggered and signal_type in ("UP_6K_300", "UP_9K"):
            result.signal = TradeSignal.STRONG_SELL
            result.conditions_met = [f"月{k_type} {signal_type} 賣出訊號"]
            result.confidence = 1.0
        else:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = [f"月線無6K/9K賣出 (type={k_type}, count={count})"]

        return result

    def check_pendulum_extreme(
        self,
        ticker: str,
        taiex_daily: pd.DataFrame,
        financial_daily: pd.DataFrame | None = None,
    ) -> SignalResult:
        """鐘擺效應極限檢查 (大飆股末日).

        條件：
            1. 大盤本波低點→高點漲幅 > +6000 點
            2. 大盤與金融指數極端乖離 (大盤 - 金融×10 > +2000 或 < -2000)

        Args:
            ticker: 股票代號
            taiex_daily: 加權指數日線
            financial_daily: 金融指數日線 (可選)

        Returns:
            SignalResult
        """
        result = SignalResult(ticker=ticker)
        if taiex_daily.empty or len(taiex_daily) < 60:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = ["大盤資料不足"]
            return result

        tx_close = taiex_daily["Close"] if "Close" in taiex_daily.columns else taiex_daily["Adj_Close"]

        # 本波低點→高點漲幅
        lowest = float(tx_close.min())
        highest = float(tx_close.max())
        total_gain = highest - lowest

        result.detail["tx_low"] = round(lowest, 2)
        result.detail["tx_high"] = round(highest, 2)
        result.detail["tx_gain"] = round(total_gain, 2)

        # 條件1: 漲幅 > 6000 點
        gain_ok = total_gain > 6000

        # 條件2: 大盤與金融指數乖離
        divergence_ok = False
        divergence_val = 0.0
        if financial_daily is not None and not financial_daily.empty:
            fin_close = financial_daily["Close"] if "Close" in financial_daily.columns else financial_daily["Adj_Close"]
            tx_now = float(tx_close.iloc[-1])
            fin_now = float(fin_close.iloc[-1])
            divergence_val = tx_now - (fin_now * 10)
            divergence_ok = divergence_val > 2000 or divergence_val < -2000

        result.detail["divergence_10x"] = round(divergence_val, 2)

        conditions: list[str] = []
        failed: list[str] = []

        if gain_ok:
            conditions.append(f"大盤漲幅 {total_gain:.0f} > 6000 點")
        else:
            failed.append(f"大盤漲幅 {total_gain:.0f} <= 6000 點")

        if divergence_ok:
            conditions.append(f"大盤-金融×10={divergence_val:.0f} 極端乖離")
        else:
            failed.append(f"大盤-金融×10={divergence_val:.0f} 無乖離")

        if gain_ok and divergence_ok:
            result.signal = TradeSignal.STRONG_SELL
            result.conditions_met = [f"⚠️ 鐘擺效應極限: {', '.join(conditions)}"]
            result.confidence = 1.0
        else:
            result.signal = TradeSignal.NEUTRAL
            result.conditions_failed = failed

        return result


# ═══════════════════════════════════════════════════════════════
# 9. 資金配置 (Step 3)
# ═══════════════════════════════════════════════════════════════
#
# 三類資金配置：
#   權值股 → 台灣50 (0050/006208)
#   中小型股 → 中百指數
#   金融股 → 金融指數
#   月威廉 W%R3 → 指數強度判斷


class CapitalAllocator:
    """資金配置計算器.

    根據各指數的月威廉 W%R3 判斷資金配置比例。
    """

    INDEX_MAP = {
        "權值": "0050",
        "金融": "2881",  # 以富邦金作為金融指數 proxy
        "中小型": "2317",  # 以鴻海作為中百指數 proxy
    }

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def assess_index_strength(
        self,
        ticker: str,
        monthly: pd.DataFrame | None = None,
    ) -> float:
        """評估單一指數強度 (0~100).

        使用月威廉 W%R3：
            W%R3 越接近 -100 → 超賣 → 強度低 (可能反彈)
            W%R3 越接近 0 → 超買 → 強度高
        """
        if monthly is None:
            monthly = self.loader.load_daily(ticker, adjusted=True)
            if monthly.empty:
                return 50.0

        m_wr3 = wr(monthly["High"], monthly["Low"], monthly["Close"], period=3)
        w3 = _safe_last(m_wr3, -50.0)
        # 轉換 W%R3 (-100~0) 到 0~100 強度
        if pd.isna(w3):
            return 50.0
        strength = 100.0 - abs(w3 + 100.0)  # w3=-100→0, w3=0→100
        return float(np.clip(strength, 0.0, 100.0))

    def get_allocation(self) -> dict[str, float]:
        """取得各類股建議資金比例 (總和 100%)."""
        strengths: dict[str, float] = {}
        for category, ticker in self.INDEX_MAP.items():
            s = self.assess_index_strength(ticker)
            strengths[category] = s

        total = sum(strengths.values()) or 1.0
        return {k: round(v / total, 2) for k, v in strengths.items()}


# ═══════════════════════════════════════════════════════════════
# 向後相容別名 (讓舊版 main.py 仍可 import)
# ═══════════════════════════════════════════════════════════════

# 舊版類別 (保持 import 不中斷)
MarketSignal = MarketSignalV2
SectorSignal = SectorSignalV2
BigStockBuySignal = BigStockBuySignalV2
BigStockSellSignal = BigStockSellSignalV2


class SignalAligner:
    """跨週期訊號對齊器 (v2 保留向後相容).

    確保多週期方向一致後才輸出最終訊號。
    """

    def __init__(self, loader: TWSEStockLoader | None = None):
        self.loader = loader or TWSEStockLoader()

    def align(
        self,
        ticker: str,
        daily_signal: SignalResult,
        *,
        daily: pd.DataFrame | None = None,
        weekly: pd.DataFrame | None = None,
        monthly: pd.DataFrame | None = None,
    ) -> SignalResult:
        """對齊多週期訊號.

        規則：
            - 日線 MACD > 0 + 週線 RSI > 50 + 月線 RSI > 50 → 多頭確認
            - 日線 MACD < 0 + 週線 RSI < 50 + 月線 RSI < 50 → 空頭確認
            - 方向不一致 → 降級或維持中立

        Args:
            ticker: 股票代號
            daily_signal: 日線產生的買/賣訊號
            daily: 日線 (可選)
            weekly: 週線 (可選)
            monthly: 月線 (可選)

        Returns:
            對齊後的 SignalResult
        """
        if daily is None and weekly is None and monthly is None:
            tf = self.loader.load_multi_timeframe(ticker, adjusted=True)
            daily = tf["daily"]
            weekly = tf["weekly"]
            monthly = tf["monthly"]

        # 預設回傳原訊號
        result = SignalResult(
            ticker=ticker,
            signal=daily_signal.signal,
            confidence=daily_signal.confidence,
            conditions_met=daily_signal.conditions_met[:],
            conditions_failed=daily_signal.conditions_failed[:],
            detail=dict(daily_signal.detail),
        )

        # 取得各週期 RSI & MACD
        d_rsi_val = 50.0
        w_rsi_val = 50.0
        m_rsi_val = 50.0
        d_macd_val = 0.0

        if daily is not None and not daily.empty and "Adj_Close" in daily.columns:
            d_close = daily["Adj_Close"]
            d_rsi_val = _safe_last(rsi(d_close), 50.0)
            d_macd_val = _safe_last(macd(d_close)["macd"], 0.0)

        if weekly is not None and not weekly.empty:
            w_rsi_val = _safe_last(rsi(weekly["Close"]), 50.0)

        if monthly is not None and not monthly.empty:
            m_rsi_val = _safe_last(rsi(monthly["Close"]), 50.0)

        result.detail["daily_rsi"] = round(float(d_rsi_val), 1)
        result.detail["weekly_rsi"] = round(float(w_rsi_val), 1)
        result.detail["monthly_rsi"] = round(float(m_rsi_val), 1)
        result.detail["daily_macd"] = round(float(d_macd_val), 2)

        # 多頭確認：全正
        if d_macd_val > 0 and d_rsi_val > 50 and w_rsi_val > 50:
            result.detail["alignment"] = "多頭一致"
            # 維持原訊號或升級
            if daily_signal.signal == TradeSignal.HOLD:
                result.signal = TradeSignal.BUY
                result.conditions_met.append("跨週期多頭確認")
        # 空頭確認：全負
        elif d_macd_val < 0 and d_rsi_val < 50 and w_rsi_val < 50:
            result.detail["alignment"] = "空頭一致"
            if daily_signal.signal == TradeSignal.HOLD:
                result.signal = TradeSignal.SELL
                result.conditions_met.append("跨週期空頭確認")
        else:
            result.detail["alignment"] = "方向分歧"
            # 降級
            if daily_signal.signal in (TradeSignal.STRONG_BUY, TradeSignal.BUY):
                result.signal = TradeSignal.HOLD
                result.conditions_failed.append("跨週期方向不一致，降級")
            elif daily_signal.signal in (TradeSignal.STRONG_SELL, TradeSignal.SELL):
                result.signal = TradeSignal.HOLD
                result.conditions_failed.append("跨週期方向不一致，降級")

        return result
