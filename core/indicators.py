"""
TWSE 量化技術指標庫 — Indicator Engine
========================================

數學運算專用，不含任何買賣邏輯。

標準指標：
    MACD, DMI (ADX), WMS%R (威廉), RSI, VR (成交量變異)

獨創指標：
    N2 (大盤轉折, 2個月高低點平均)
    6K/9K 理論判定（含內含K無效處理、不破前低/前高、累積300點）

使用方式：
    from indicators import macd, dmi, wr, rsi, vr, n2, k6k9
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# 1. MACD
# ═══════════════════════════════════════════════════════════════

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """計算 MACD 指標。

    支援自訂週期，例如 macd(close, 200, 209, 210)。

    Args:
        close: 收盤價序列
        fast: 快線 EMA 週期 (default 12)
        slow: 慢線 EMA 週期 (default 26)
        signal: 訊號線 EMA 週期 (default 9)

    Returns:
        {"macd": MACD線, "signal": 訊號線, "histogram": 柱狀圖, "hist_diff": 柱狀圖增減}
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_diff = histogram.diff()

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
        "hist_diff": hist_diff,
    }


def macd_4arrows(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, Any]:
    """計算 MACD 四箭頭攻擊強度。

    四箭頭定義 (全數向上 = 攻擊訊號)：
        箭頭1: MACD(DIF) > 0 (零軸之上)
        箭頭2: MACD(DIF) 往上 (diff > 0)
        箭頭3: 訊號線往上 (signal_line.diff() > 0)
        箭頭4: 柱狀圖往上 (histogram.diff() > 0)

    Args:
        close: 收盤價序列
        fast: 快線週期 (default 12, 可設 200)
        slow: 慢線週期 (default 26, 可設 209)
        signal: 訊號線週期 (default 9, 可設 210)

    Returns:
        {
            "arrow1": MACD > 0 (bool Series),
            "arrow2": MACD 向上 (bool Series),
            "arrow3": 訊號線向上 (bool Series),
            "arrow4": 柱狀圖向上 (bool Series),
            "arrows_count": 四箭頭計數 (0-4),
            "all_up": 全數向上 (bool Series),
        }
    """
    m = macd(close, fast, slow, signal)
    arrow1 = m["macd"] > 0
    arrow2 = m["macd"].diff() > 0
    arrow3 = m["signal"].diff() > 0
    arrow4 = m["hist_diff"] > 0

    count = arrow1.astype(int) + arrow2.astype(int) + arrow3.astype(int) + arrow4.astype(int)

    return {
        "arrow1": arrow1,
        "arrow2": arrow2,
        "arrow3": arrow3,
        "arrow4": arrow4,
        "arrows_count": count,
        "all_up": count == 4,
    }


# ═══════════════════════════════════════════════════════════════
# 2. DMI / ADX
# ═══════════════════════════════════════════════════════════════

def dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """計算 DMI (Directional Movement Index) 與 ADX。

    ADX 值域 0~100，>25 代表趨勢明顯。
    支援 ADX 300 (period=300) 等長週期。

    Args:
        high: 最高價
        low: 最低價
        close: 收盤價
        period: 週期 (default 14, 可設 300 等長週期)

    Returns:
        {"plus_di": +DI, "minus_di": -DI, "adx": ADX, "adxr": ADXR}
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=close.index, dtype="float64")
    minus_dm = pd.Series(0.0, index=close.index, dtype="float64")

    # +DM when up > down and up > 0
    cond_up = (up_move > down_move) & (up_move > 0)
    plus_dm[cond_up] = up_move[cond_up]

    # -DM when down > up and down > 0
    cond_down = (down_move > up_move) & (down_move > 0)
    minus_dm[cond_down] = down_move[cond_down]

    # Smoothed by EMA
    tr_smooth = tr.ewm(span=period, adjust=False).mean()
    plus_smooth = plus_dm.ewm(span=period, adjust=False).mean()
    minus_smooth = minus_dm.ewm(span=period, adjust=False).mean()

    # +DI, -DI
    plus_di = 100.0 * plus_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_smooth / tr_smooth.replace(0, np.nan)

    # DX, ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()

    # ADXR = (ADX + ADX.shift(period)) / 2
    adxr = (adx + adx.shift(period)) / 2.0

    return {
        "plus_di": plus_di,
        "minus_di": minus_di,
        "adx": adx,
        "adxr": adxr,
    }


# ═══════════════════════════════════════════════════════════════
# 3. WMS%R (威廉指標)
# ═══════════════════════════════════════════════════════════════

def wr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """計算 Williams %R (威廉指標).

    公式: %R = (HH_n - Close) / (HH_n - LL_n) × (-100)
    值域 -100 ~ 0, < -80 超賣, > -20 超買。

    Args:
        high: 最高價
        low: 最低價
        close: 收盤價
        period: 週期 (default 14)

    Returns:
        WMS%R Series, 值域 [-100, 0]
    """
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    denom = hh - ll
    w = -100.0 * (hh - close) / denom.replace(0, np.nan)
    # 固定值域 [-100, 0]
    return w.clip(-100.0, 0.0)


# ═══════════════════════════════════════════════════════════════
# 4. RSI
# ═══════════════════════════════════════════════════════════════

def rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """計算 RSI (Relative Strength Index).

    使用 Wilder's Smoothing Method。

    Args:
        close: 收盤價
        period: 週期 (default 14)

    Returns:
        RSI Series, 值域 [0, 100]
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - 100.0 / (1.0 + rs)

    # 處理全漲/全跌的邊界情形
    # avg_loss=0 → RS=∞ → RSI=100
    zero_loss = avg_loss < 1e-10
    rsi_val = rsi_val.mask(zero_loss, 100.0)

    # avg_gain=0 → RS=0 → RSI=0
    zero_gain = avg_gain < 1e-10
    rsi_val = rsi_val.mask(zero_gain, 0.0)

    return rsi_val.clip(0.0, 100.0)


# ═══════════════════════════════════════════════════════════════
# 5. VR (成交量變異指標, Volume Ratio)
# ═══════════════════════════════════════════════════════════════

def vr(
    close: pd.Series,
    volume: pd.Series,
    period: int = 26,
) -> pd.Series:
    """計算 VR (Volume Ratio / 成交量變異指標).

    公式:
        VR = (AVS + 0.5 × CVS) / (BVS + 0.5 × CVS) × 100

        AVS = 上漲日成交量總和
        BVS = 下跌日成交量總和
        CVS = 平盤日成交量總和

    VR > 280 超買, VR < 70 超賣

    Args:
        close: 收盤價
        volume: 成交量
        period: 週期 (default 26)

    Returns:
        VR Series
    """
    # 判斷漲跌
    up = close.diff() > 0
    down = close.diff() < 0
    flat = close.diff() == 0

    # Rolling sum by type
    avs = (volume * up).rolling(window=period).sum()
    bvs = (volume * down).rolling(window=period).sum()
    cvs = (volume * flat).rolling(window=period).sum()

    denom = bvs + 0.5 * cvs
    vr_val = 100.0 * (avs + 0.5 * cvs) / denom.replace(0, np.nan)

    # 全漲無跌日 → VR > 100 (理論無窮大)
    # 全跌無漲日 → VR < 100 (理論趨近 0)
    zero_denom = denom < 1e-10
    all_up = (zero_denom) & (avs > 0)
    all_down = (zero_denom) & (bvs > 0)
    vr_val = vr_val.mask(all_up, 500.0)
    vr_val = vr_val.mask(all_down, 0.0)

    return vr_val.clip(0.0, 1000.0)


# ═══════════════════════════════════════════════════════════════
# 6. 大盤轉折 N2 (2個月內高低點平均)
# ═══════════════════════════════════════════════════════════════

def n2(
    high: pd.Series,
    low: pd.Series,
    period: int = 42,
) -> pd.Series:
    """計算大盤轉折 N2。

    N2 = (近 period 日最高價 + 近 period 日最低價) / 2

    約 2 個月 = 42 個交易日。
    當收盤價站上 N2 視為轉折向上訊號。

    Args:
        high: 最高價
        low: 最低價
        period: 週期 (default 42, 約 2 個月)

    Returns:
        N2 Series (中價線)
    """
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    return (hh + ll) / 2.0


# ═══════════════════════════════════════════════════════════════
# 7. 6K/9K 理論判定演算法 (核心獨創指標 — 月線版本)
# ═══════════════════════════════════════════════════════════════
#
# 理論基礎 (依用戶提供之虛擬碼實作)：
#   ┌─────────────────────────────────────────────────────────┐
#   │ 6K/9K = 連續 6 根 / 9 根有效 K 線的強弱判定。          │
#   │                                                        │
#   │ 核心規則：                                              │
#   │   1. 內含K (Inside Bar)：K線高低點完全落在              │
#   │      「前一根有效 K 線」之內 → 無效，繼承前一期計數。   │
#   │   2. 上漲型 (Up)：                                     │
#   │      a. 若跌向前期有效低點 (L < prev_valid_low) →      │
#   │         重算 (count=0, pts=0, type=None)。             │
#   │      b. 首根紅K必須突破前波下降的最高點。              │
#   │      c. 每根有效紅K收盤須高於前一根有效K的最高點。     │
#   │   3. 下跌型 (Down)：                                   │
#   │      a. 若漲破前期有效高點 (H > prev_valid_high) →     │
#   │         重算。                                          │
#   │      b. 每根有效黑K收盤須低於前一根有效K的最低點。     │
#   │   4. 訊號觸發：                                        │
#   │      上漲型:  count=6 且 累積漲幅 > 300點 → 警示     │
#   │               count=9 → 強烈警示                      │
#   │      下跌型:  count=6 (每根黑K跌幅須 > 300點) → 轉機 │
#   │   5. 十字K (Doji / 十字線)：                           │
#   │      符合標準十字K (收=開 or 實體極小)，兩根算一根。  │
#   └─────────────────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════

def _is_doji(open_p: float, close_p: float, threshold_pct: float = 0.1) -> bool:
    """判斷是否為十字K (Doji)。

    十字K定義：收盤與開盤價差 ≤ 實體範圍的 threshold_pct%。
    兩根十字K合併視為一根有效K線。

    Args:
        open_p: 開盤價
        close_p: 收盤價
        threshold_pct: 十字K容忍閾值 (實體的百分比)

    Returns:
        True = 十字K
    """
    body = abs(close_p - open_p)
    total_range = max(abs(close_p), abs(open_p))
    if total_range < 1e-10:
        return True  # 完全無波動視為十字
    return body / total_range < threshold_pct / 100.0


def _is_inside_bar_v2(
    high: float, low: float,
    prev_high: float, prev_low: float,
) -> bool:
    """判斷是否為內含K (Inside Bar).

    內含K：當根 K 線的高低點完全落在「前一根有效 K 線」範圍內。

    Args:
        high: 當根最高
        low: 當根最低
        prev_high: 前一根有效 K 線最高
        prev_low: 前一根有效 K 線最低

    Returns:
        True = 內含K (無效)
    """
    if np.isnan(high) or np.isnan(low) or np.isnan(prev_high) or np.isnan(prev_low):
        return False
    return high <= prev_high and low >= prev_low


def k6k9(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_p: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """6K/9K 理論判定演算法 (月線版本).

    依用戶提供之虛擬碼實作，適用於月線資料。

    Args:
        high: 最高價序列
        low: 最低價序列
        close: 收盤價序列
        open_p: 開盤價序列 (None 時用 close 替代)

    Returns:
        {
            "count":       累積有效K線計數 (int)
            "type":        'Up' / 'Down' / None
            "accum_pts":   累積漲跌幅點數
            "signal":      訊號觸發 (bool)
            "signal_type": 'UP_6K_300' / 'UP_9K' / 'DOWN_6K' / None
            "inside_bar":  該K線為內含K (bool)
            "doji":        該K線為十字K (bool)
            "per_bar_drop": 單根黑K跌幅 (僅 Down 型有效)
        }
    """
    if open_p is None:
        open_p = close  # fallback

    n = len(close)
    opens_arr = open_p.values.astype(np.float64)
    highs_arr = high.values.astype(np.float64)
    lows_arr = low.values.astype(np.float64)
    closes_arr = close.values.astype(np.float64)

    count_arr = np.zeros(n, dtype=np.int32)
    type_arr = np.full(n, None, dtype=object)
    accum_arr = np.zeros(n, dtype=np.float64)
    signal_arr = np.zeros(n, dtype=bool)
    signal_type_arr = np.full(n, None, dtype=object)
    inside_arr = np.zeros(n, dtype=bool)
    doji_arr = np.zeros(n, dtype=bool)
    per_bar_drop_arr = np.zeros(n, dtype=np.float64)

    # ── 狀態變數 ──
    current_count = 0
    current_type: str | None = None  # 'Up' or 'Down'
    accumulated_pts = 0.0

    # 前一根有效 K 線
    prev_valid_high = np.nan
    prev_valid_low = np.nan
    prev_valid_close = np.nan
    prev_valid_open = np.nan
    prev_valid_direction: str | None = None  # 前一根有效 K 線的漲跌方向 ('Up'/'Down')

    # 十字K暫存：兩根十字算一根 → 記錄是否已有第一根十字
    doji_pending = False
    doji_high_buffer = np.nan
    doji_low_buffer = np.nan

    for i in range(n):
        O = opens_arr[i]
        H = highs_arr[i]
        L = lows_arr[i]
        C = closes_arr[i]

        is_red = C > O
        is_black = C < O
        is_doji_k = _is_doji(O, C)

        doji_arr[i] = is_doji_k

        # ── 十字K處理：兩根算一根 ──
        if is_doji_k:
            if not doji_pending:
                # 第一根十字K：暫存，跳過
                doji_pending = True
                doji_high_buffer = H
                doji_low_buffer = L
                # 繼承前一期狀態（不改變計數）
                count_arr[i] = current_count
                type_arr[i] = current_type
                accum_arr[i] = accumulated_pts
                continue
            else:
                # 第二根十字K：與第一根合併為一根有效K線
                # 取兩根的最高/最低為合併後的範圍
                merged_H = max(doji_high_buffer, H)
                merged_L = min(doji_low_buffer, L)
                merged_O = O  # 以第二根的開盤為準
                merged_C = C  # 以第二根的收盤為準

                # 用合併後的 K 線繼續邏輯
                H = merged_H
                L = merged_L
                is_red = merged_C > merged_O
                is_black = merged_C < merged_O

                doji_pending = False  # 消耗完畢
                # 繼續執行下方正常判斷（不 continue）

        # ── 1. 內含K判定 ──
        if not np.isnan(prev_valid_high) and not np.isnan(prev_valid_low):
            if _is_inside_bar_v2(H, L, prev_valid_high, prev_valid_low):
                inside_arr[i] = True
                # 內含K：繼承前一期狀態，不更新 prev_valid
                count_arr[i] = current_count
                type_arr[i] = current_type
                accum_arr[i] = accumulated_pts
                continue

        # ── 2. 趨勢方向判定 ──
        is_first_valid = np.isnan(prev_valid_high)  # 尚無有效 K 線
        just_started = (current_count == 0)

        if current_type == 'Up' or current_type is None:
            # ── 破前低重算 ──
            if not is_first_valid and not np.isnan(prev_valid_low) and L < prev_valid_low:
                current_count = 0
                accumulated_pts = 0.0
                current_type = None

            # ── 有效紅K判定 (上漲型計數) ──
            if is_red:
                # 首根紅K條件：突破前波下降最高點
                can_start_up = True
                if just_started and prev_valid_direction == 'Down':
                    # 必須突破前波下降的最高點
                    if not np.isnan(prev_valid_high) and C <= prev_valid_high:
                        can_start_up = False

                if can_start_up:
                    if just_started:
                        current_type = 'Up'
                    if just_started or C > prev_valid_high:
                        current_count += 1
                        bar_gain = C - O  # 當根 K 線實體漲幅
                        accumulated_pts += bar_gain
                        prev_valid_high = H
                        prev_valid_low = L
                        prev_valid_close = C
                        prev_valid_open = O
                        prev_valid_direction = 'Up'

        if current_type == 'Down' or current_type is None:
            # ── 破前高重算 ──
            if not is_first_valid and not np.isnan(prev_valid_high) and H > prev_valid_high:
                current_count = 0
                accumulated_pts = 0.0
                current_type = None

            # ── 有效黑K判定 (下跌型計數) ──
            if is_black:
                current_type = 'Down'  # 無論是否 just_started，設定方向
                if just_started or C < prev_valid_low:
                    current_count += 1
                    bar_loss = O - C  # 單根跌幅
                    per_bar_drop_arr[i] = bar_loss
                    accumulated_pts += bar_loss
                    prev_valid_high = H
                    prev_valid_low = L
                    prev_valid_close = C
                    prev_valid_open = O
                    prev_valid_direction = 'Down'

        # ── 3. 訊號觸發判定 ──
        signal_triggered = False
        signal_type: str | None = None

        if current_type == 'Up':
            if current_count == 6 and accumulated_pts > 300:
                signal_triggered = True
                signal_type = 'UP_6K_300'
            elif current_count == 9:
                signal_triggered = True
                signal_type = 'UP_9K'
        elif current_type == 'Down':
            if current_count == 6:
                # 檢查最近 6 根每根黑K跌幅 > 300
                # 從當前往前看最多 6 根
                lookback_start = max(0, i - 5)
                recent_drops = per_bar_drop_arr[lookback_start:i+1]
                if np.all(recent_drops[recent_drops > 0] > 300):
                    signal_triggered = True
                    signal_type = 'DOWN_6K'

        # ── 4. 記錄狀態 ──
        count_arr[i] = current_count
        type_arr[i] = current_type
        accum_arr[i] = accumulated_pts
        signal_arr[i] = signal_triggered
        signal_type_arr[i] = signal_type

        # ── 5. 訊號觸發後重置 ──
        if signal_triggered:
            current_count = 0
            accumulated_pts = 0.0
            current_type = None

    return {
        "count": pd.Series(count_arr, index=close.index),
        "type": pd.Series(type_arr, index=close.index),
        "accum_pts": pd.Series(accum_arr, index=close.index),
        "signal": pd.Series(signal_arr, index=close.index),
        "signal_type": pd.Series(signal_type_arr, index=close.index),
        "inside_bar": pd.Series(inside_arr, index=close.index),
        "doji": pd.Series(doji_arr, index=close.index),
        "per_bar_drop": pd.Series(per_bar_drop_arr, index=close.index),
    }


# ═══════════════════════════════════════════════════════════════
# 8. 模組測試 (python indicators.py 直接執行)
# ═══════════════════════════════════════════════════════════════

def _demo() -> None:
    """使用 data_loader 載入真實資料並計算所有指標."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import TWSEStockLoader

    loader = TWSEStockLoader()
    tf = loader.load_multi_timeframe("2330", start="2024-01-01", end="2026-06-09")
    daily = tf["daily"]
    weekly = tf["weekly"]
    monthly = tf["monthly"]

    if daily.empty:
        print("❌ 無資料")
        return

    close = daily["Adj_Close"]
    high = daily["Adj_High"]
    low = daily["Adj_Low"]
    vol = daily["Adj_Volume"]

    print(f"📊 2330 日線指標計算 ({len(daily)} 行)")

    # MACD
    m = macd(close)
    print(f"  MACD:  最後值={m['macd'].iloc[-1]:.2f}, 訊號={m['signal'].iloc[-1]:.2f}")
    print(f"  Hist:  最後值={m['histogram'].iloc[-1]:.2f}, 增減={m['hist_diff'].iloc[-1]:.2f}")

    # MACD 自訂週期
    m200 = macd(close, 200, 209, 210)
    print(f"  MACD(200,209,210):  最後值={m200['macd'].iloc[-1]:.2f}")

    # DMI
    d = dmi(high, low, close, period=14)
    print(f"  DMI:   +DI={d['plus_di'].iloc[-1]:.1f}, -DI={d['minus_di'].iloc[-1]:.1f}")
    print(f"  ADX:   {d['adx'].iloc[-1]:.1f}")

    # ADX 300
    d300 = dmi(high, low, close, period=300)
    print(f"  ADX(300): {d300['adx'].iloc[-1]:.1f}")

    # WMS%R
    w = wr(high, low, close)
    print(f"  WMS%R: {w.iloc[-1]:.1f}")

    # RSI
    r = rsi(close)
    print(f"  RSI:   {r.iloc[-1]:.1f}")

    # VR
    v = vr(close, vol)
    print(f"  VR:    {v.iloc[-1]:.1f}")

    # N2
    n = n2(high, low)
    print(f"  N2:    {n.iloc[-1]:.2f}")

    # 6K/9K
    k = k6k9(high, low, close, open_p=daily.get("Open"))
    bull_6 = (k["signal"] & (k["signal_type"] == "UP_6K_300")).sum()
    bull_9 = (k["signal"] & (k["signal_type"] == "UP_9K")).sum()
    bear_6 = (k["signal"] & (k["signal_type"] == "DOWN_6K")).sum()
    inside = k["inside_bar"].sum()
    doji = k["doji"].sum()
    print(f"  6K/9K: UP_6K_300={bull_6}次, UP_9K={bull_9}次, DOWN_6K={bear_6}次")
    print(f"  內含K: {inside}根 ({inside/len(daily)*100:.1f}%)")
    print(f"  十字K: {doji}根")

    # 週線 / 月線
    if not weekly.empty:
        w_close = weekly["Close"]
        print(f"\n📊 週線: MACD={macd(w_close)['macd'].iloc[-1]:.2f}, RSI={rsi(w_close).iloc[-1]:.1f}")

    if not monthly.empty:
        m_close = monthly["Close"]
        print(f"📊 月線: MACD={macd(m_close)['macd'].iloc[-1]:.2f}, RSI={rsi(m_close).iloc[-1]:.1f}")

    print("\n✅ 所有指標計算完成")


import os

if __name__ == "__main__":
    _demo()
