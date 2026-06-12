"""
TWSE 技術指標庫 — pytest 單元測試
==================================

重點測試：
    1. 各標準指標的數學正確性 (MACD, DMI, WMS%R, RSI, VR, N2)
    2. 6K/9K 演算法 (月線版本)：內含K、十字K、破前低/前高、累積300點
    3. 邊界條件：空序列、NaN、極端值

執行：
    pytest tests/test_indicators.py -v
"""

from __future__ import annotations

import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from indicators import (
    macd, dmi, wr, rsi, vr, n2, k6k9,
    _is_doji, _is_inside_bar_v2,
)


# ═══════════════════════════════════════════════════════════════
# 測試資料產生器
# ═══════════════════════════════════════════════════════════════

def _series(values: list[float], name: str = "val") -> pd.Series:
    return pd.Series(values, name=name)


def _ohlcv(
    close: list[float],
    high: list[float] | None = None,
    low: list[float] | None = None,
    open_p: list[float] | None = None,
    volume: list[float] | None = None,
) -> dict[str, pd.Series]:
    n = len(close)
    O = open_p or [c * 0.99 for c in close]
    H = high or [c * 1.02 for c in close]
    L = low or [c * 0.98 for c in close]
    V = volume or [1000000.0] * n
    return {"close": _series(close), "high": _series(H),
            "low": _series(L), "open": _series(O), "volume": _series(V)}


# ═══════════════════════════════════════════════════════════════
# 1. MACD
# ═══════════════════════════════════════════════════════════════

class TestMACD:
    def test_basic(self):
        close = _series([100 + i * 0.5 for i in range(100)])
        r = macd(close)
        assert "macd" in r and len(r["macd"]) == 100
        assert r["macd"].iloc[-1] > 0

    def test_custom_periods(self):
        close = _series([100 + math.sin(i * 0.1) * 10 for i in range(300)])
        r = macd(close, fast=200, slow=209, signal=210)
        assert not r["macd"].isna().all()

    def test_flat(self):
        close = _series([100.0] * 50)
        r = macd(close)
        assert abs(r["macd"].iloc[-1]) < 1e-6

    def test_empty(self):
        assert macd(_series([]))["macd"].empty


# ═══════════════════════════════════════════════════════════════
# 2. DMI / ADX
# ═══════════════════════════════════════════════════════════════

class TestDMI:
    def test_basic(self):
        o = _ohlcv([100 + i for i in range(100)], high=[102 + i for i in range(100)],
                    low=[98 + i for i in range(100)])
        r = dmi(o["high"], o["low"], o["close"])
        assert r["plus_di"].iloc[-1] > r["minus_di"].iloc[-1]

    def test_adx_300(self):
        n = 500
        c = [100 + math.sin(i * 0.05) * 20 for i in range(n)]
        h, l = [x * 1.01 for x in c], [x * 0.99 for x in c]
        r = dmi(_series(h), _series(l), _series(c), period=300)
        assert r["adx"].iloc[-1] >= 0

    def test_strong_trend(self):
        o = _ohlcv([100 + i * 2 for i in range(100)],
                    high=[105 + i * 2 for i in range(100)],
                    low=[95 + i * 2 for i in range(100)])
        assert dmi(o["high"], o["low"], o["close"])["adx"].iloc[-1] > 25


# ═══════════════════════════════════════════════════════════════
# 3. WMS%R
# ═══════════════════════════════════════════════════════════════

class TestWR:
    def test_basic(self):
        o = _ohlcv([100 + i for i in range(30)])
        w = wr(o["high"], o["low"], o["close"])
        assert -100 <= w.iloc[-1] <= 0

    def test_oversold(self):
        c = [100 - i * 2 for i in range(30)]
        o = _ohlcv(c, high=[x + 3 for x in c], low=[x - 3 for x in c])
        assert wr(o["high"], o["low"], o["close"]).iloc[-1] <= -80

    def test_overbought(self):
        c = [100 + i * 2 for i in range(30)]
        o = _ohlcv(c, high=[x + 3 for x in c], low=[x - 3 for x in c])
        assert wr(o["high"], o["low"], o["close"]).iloc[-1] >= -20


# ═══════════════════════════════════════════════════════════════
# 4. RSI
# ═══════════════════════════════════════════════════════════════

class TestRSI:
    def test_basic(self):
        r = rsi(_series([100 + i for i in range(30)]))
        assert 0 <= r.iloc[-1] <= 100

    def test_overbought(self):
        assert rsi(_series([100 + i * 5 for i in range(30)])).iloc[-1] > 70

    def test_oversold(self):
        assert rsi(_series([100 - i * 5 for i in range(30)])).iloc[-1] < 30

    def test_flat(self):
        r = rsi(_series([100 + (i % 3 - 1) for i in range(50)]))
        assert 30 <= r.iloc[-1] <= 70


# ═══════════════════════════════════════════════════════════════
# 5. VR
# ═══════════════════════════════════════════════════════════════

class TestVR:
    def test_basic(self):
        o = _ohlcv([100 + (i % 5 - 2) for i in range(50)])
        assert vr(o["close"], o["volume"]).iloc[-1] > 0

    def test_up_trend(self):
        c = _series([100 + i for i in range(50)])
        v = _series([2e6 if i > 25 else 5e5 for i in range(50)])
        assert vr(c, v).iloc[-1] > 100

    def test_down_trend(self):
        c = _series([200 - i for i in range(50)])
        v = _series([2e6] * 50)
        assert vr(c, v).iloc[-1] < 100


# ═══════════════════════════════════════════════════════════════
# 6. N2
# ═══════════════════════════════════════════════════════════════

class TestN2:
    def test_basic(self):
        n = n2(_series([110, 120, 130, 125, 115]), _series([90, 95, 100, 98, 92]), period=3)
        assert abs(n.iloc[-1] - 111.0) < 1e-6

    def test_price_above(self):
        h = _series([100 + i * 2 for i in range(60)])
        l = _series([90 + i * 2 for i in range(60)])
        c = _series([95 + i * 2 for i in range(60)])
        assert c.iloc[-1] > n2(h, l).iloc[-1]


# ═══════════════════════════════════════════════════════════════
# 7. 6K/9K (月線版本 — 核心測試)
# ═══════════════════════════════════════════════════════════════

class TestK6K9:

    # ── 輔助函式 ──

    def _run(self, close, high, low, open_p=None):
        return k6k9(
            _series(high), _series(low), _series(close),
            open_p=_series(open_p) if open_p else None,
        )

    # ── 子功能：內含K、十字K ──

    def test_inside_bar_v2(self):
        assert _is_inside_bar_v2(105, 95, 110, 90) is True
        assert _is_inside_bar_v2(115, 95, 110, 90) is False
        assert _is_inside_bar_v2(105, 85, 110, 90) is False

    def test_doji_detection(self):
        assert _is_doji(100.0, 100.0) is True       # 完全十字
        assert _is_doji(100.0, 100.05, 0.1) is True  # 極小實體
        assert _is_doji(100.0, 101.0, 0.1) is False  # 明顯實體

    # ── 基本行為 ──

    def test_no_signal_flat(self):
        k = self._run([100]*20, [101]*20, [99]*20)
        assert k["signal"].sum() == 0

    def test_empty(self):
        k = self._run([], [], [])
        assert k["signal"].empty

    def test_short(self):
        k = self._run([100, 101], [102, 103], [99, 100])
        assert len(k["count"]) == 2

    def test_nan(self):
        k = self._run([100, np.nan, 104], [102, np.nan, 106], [98, np.nan, 102])
        assert len(k["count"]) == 3

    # ── 上漲型 6K 訊號 (count=6 + accum_pts>300) ──

    def test_up_6k_300_signal(self):
        """6 根紅K連續上漲，累積 > 300 點 → UP_6K_300 訊號."""
        # 每根實體漲幅 > 50, 6根後累積 > 300
        close = [100, 160, 230, 310, 400, 500, 610]
        high  = [105, 165, 235, 315, 405, 505, 615]
        low   = [95, 155, 225, 305, 395, 495, 605]
        open_p = [99, 105, 165, 235, 315, 405, 505]
        k = self._run(close, high, low, open_p)
        # accum_pts = (160-105)+(230-165)+(310-235)+(400-315)+(500-405)+(610-505)
        # = 55+65+75+85+95+105 = 480 > 300
        sig = k["signal_type"]
        has_up6 = (sig == "UP_6K_300").sum()
        assert has_up6 >= 1, f"應觸發 UP_6K_300, signal_types={sig.tolist()}"

    def test_up_9k_signal(self):
        """9 根連續上漲 (無點數限制) → UP_9K 訊號."""
        close = [100 + i*10 for i in range(12)]
        high  = [105 + i*10 for i in range(12)]
        low   = [95 + i*10 for i in range(12)]
        open_p= [99 + i*10 for i in range(12)]
        k = self._run(close, high, low, open_p)
        assert (k["signal_type"] == "UP_9K").sum() >= 1

    def test_down_6k_signal(self):
        """6 根黑K每根跌 > 300 點 → DOWN_6K 訊號."""
        close = [1500, 1200, 880, 520, 150, -200, -600]
        high  = [1550, 1250, 930, 570, 200, -150, -550]
        low   = [1450, 1150, 830, 470, 100, -250, -650]
        open_p= [1490, 1550, 1260, 940, 580, 210, -140]
        k = self._run(close, high, low, open_p)
        assert (k["signal_type"] == "DOWN_6K").sum() >= 1, (
            f"應觸發 DOWN_6K, types={k['signal_type'].tolist()}"
        )

    # ── 破前低重算 ──

    def test_break_low_resets_up(self):
        """上漲中若 L < prev_valid_low → 重算."""
        close = [100, 110, 105]  # 第3根收盤下跌, 低點跌破
        high  = [105, 115, 110]
        low   = [95, 105, 90]    # 第3根 L=90 < prev_valid_low=95
        open_p= [99, 105, 108]
        k = self._run(close, high, low, open_p)
        # 第1根: up count=1
        # 第2根: up count=2, prev_low=105
        # 第3根: L=90 < 105 → reset → count=0
        assert k["count"].iloc[-1] == 0, f"應重算為0, 實際={k['count'].iloc[-1]}"

    # ── 破前高重算 ──

    def test_break_high_resets_down(self):
        """下跌中若 H > prev_valid_high → 重算."""
        close = [200, 180, 210]
        high  = [205, 185, 215]
        low   = [195, 175, 205]
        open_p= [198, 190, 200]
        k = self._run(close, high, low, open_p)
        # 第1根: Up count=1, prev_high=205
        # 第2根: L=175 < 195 → reset Up. Down count=1, prev_high=185
        # 第3根: H=215 > 185 → reset Down → type=None, count=0
        assert k["count"].iloc[-1] == 0, f"count={k['count'].iloc[-1]}"
        assert str(k["type"].iloc[-1]) != "Down", f"type={k['type'].iloc[-1]}"

    # ── 內含K不影響計數 ──

    def test_inside_bar_preserves_count(self):
        """內含K繼承前一期計數，不中斷."""
        close = [100, 110, 112, 120, 130]
        high  = [105, 115, 114, 125, 135]
        low   = [95, 105, 106, 115, 125]  # 第3根 L=106 >= prev_low=105 → 內含K
        open_p= [99, 108, 110, 115, 128]
        k = self._run(close, high, low, open_p)
        # count[2] 應繼承 count[1]
        assert k["inside_bar"].iloc[2] == True, f"第3根應為內含K, 但 inside_bar={k['inside_bar'].iloc[2]}"
        assert k["count"].iloc[2] == k["count"].iloc[1], (
            f"內含K應繼承 count, count[1]={k['count'].iloc[1]}, count[2]={k['count'].iloc[2]}"
        )

    # ── 十字K兩根算一根 ──

    def test_doji_two_as_one(self):
        """兩根十字K後接紅K，十字K視為一根暫存，紅K確認計入上漲型 count=1."""
        # doji1(O=C=100) → doji2(O=C=100) → 紅K(O=100,C=110)
        # 合併後：open=100, close=110 → 紅K → 上漲型 count 應 >= 1
        close = [100, 100, 110, 115, 120]
        high  = [105, 104, 115, 120, 125]
        low   = [95,  96,  105, 110, 115]
        open_p= [100, 100, 100, 114, 119]
        k = self._run(close, high, low, open_p)
        counts = k["count"].tolist()
        # 確認第3根（index=2，紅K）後計數 >= 1
        self.assertGreaterEqual(counts[2], 1, "十字K合併後紅K應啟動上漲型計數")

    def test_doji_then_valid(self):
        """十字K暫存後，下一根有效K正常計數."""
        close = [100, 100, 110, 115]
        high  = [105, 104, 115, 120]
        low   = [95, 96, 105, 110]
        open_p= [100, 100, 108, 114]
        k = self._run(close, high, low, open_p)
        # 第1根: doji, pending
        # 第2根: not doji, not inside → 正常處理
        # doji_pending 應在非十字時重置
        assert k["count"].iloc[2] >= 1
