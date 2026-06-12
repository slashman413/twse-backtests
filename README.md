# 台股量化回測系統 — TWSE Backtest

> **大飆股 DNA 策略**的量化實作與歷史回測，涵蓋 2004–2026 年，支援全市場逐日掃描與 GitHub Pages 儀表板。

🌐 **儀表板**：[slashman413.github.io/twse-backtests](https://slashman413.github.io/twse-backtests/)

---

## 資料夾結構

```
twse-backtests/
│
├── core/                      # 策略核心（指標、策略、回測、掃描）
│   ├── indicators.py          # 技術指標計算庫
│   ├── strategy.py            # 買賣訊號邏輯（9步操作法）
│   ├── data_loader.py         # 資料載入與還原權值
│   ├── backtest.py            # 單股/全市場歷史回測報表
│   ├── main.py                # 每日全市場掃描器 + HTML/CSV 報表
│   └── bulk_charts.py         # 批量輸出訊號股 K線圖 JSON
│
├── crawler/                   # 資料爬蟲與調整器
│   ├── twse_crawler.py        # TWSE 每日 OHLCV 爬蟲（防封 IP）
│   ├── adjuster.py            # 還原權值調整器（backward adjustment）
│   ├── run_backtest.py        # 全市場年度資金模擬回測
│   ├── build_dividend_file.py # 從除權息日資料估算股利
│   ├── twse_validator.py      # 用 yfinance 驗證還原價格 MAPE < 0.5%
│   ├── run_yearly.py          # 按年份批量執行爬蟲
│   ├── check_year_progress.py # 檢查各年爬蟲完成進度
│   └── partition_by_year.py   # 將全量 parquet 依年份切割
│
├── tests/                     # 單元測試
│   ├── test_indicators.py     # 指標計算正確性驗證
│   ├── test_data_loader.py    # OHLCV 資料約束測試
│   └── test_strategy.py       # 買賣訊號邏輯測試
│
├── docs/                      # GitHub Pages 網站
│   ├── index.html             # 回測儀表板（無需框架，純 HTML/JS）
│   └── data/
│       ├── summary.json       # 各年度彙總（報酬率、勝率、交易次數）
│       └── YYYY_trades.json   # 每年逐筆交易明細 + 資產曲線
│
└── .gitignore
```

---

## 技術指標（`core/indicators.py`）

### 標準指標

| 指標 | 函數 | 說明 |
|------|------|------|
| **MACD** | `macd(close, fast, slow, signal)` | 標準 MACD；支援長週期（如 200/209/210） |
| **MACD 四箭頭** | `macd_4arrows(close, fast, slow, signal)` | 箭頭1=DIF>0，箭頭2=DIF上升，箭頭3=訊號線上升，箭頭4=柱狀圖上升；`arrows_count` 為 0–4 |
| **DMI / ADX** | `dmi(high, low, close, period)` | 方向動量指標；支援 ADX300（超長週期確認主趨勢）；輸出 +DI、−DI、ADX、ADXR |
| **威廉指標 WMS%R** | `wr(high, low, close, period)` | 值域 −100 ~ 0；< −80 超賣，> −20 超買 |
| **RSI** | `rsi(close, period)` | 標準相對強弱指標；策略使用 RSI60（日）、RSI4（月）等多週期 |
| **VR 成交量變異率** | `vr(close, volume, period)` | `(上漲日成交量 + 0.5×平盤) / (下跌日成交量 + 0.5×平盤) × 100`；中值≈100，>150 量能偏多 |

### 獨創指標

#### N2 大盤轉折值
```
N2 = (前2個月最高點 + 前2個月最低點) / 2
備戰區 = N2 − 100
```
- N2 是大盤多空分水嶺；站上為多頭，跌破備戰區開始觀望
- 每月底以新的 2 個月高低點更新，避免訊號過早切換

#### 6K/9K 上漲/下跌型理論
基於日 K 線計算累積攻擊/回調波段，是策略買賣時機的核心輔助指標：
- **上漲型 6K**：連續收紅 K（不破前高），每根計 1 點，累積滿 6 點 = 短期過熱
- **上漲型 9K**：累積 9 點 = 強力買進確認
- **下跌型 6K/9K**：對應空頭波段計數
- **特殊處理**：
  - 內含 K（High/Low 完全包含）：視為無效，不計入
  - 十字 K（Open ≈ Close）：暫存合併為下一根計算
  - 破前低/前高：計數重置

---

## 策略邏輯（`core/strategy.py`）

實作「九步操作策略法」，分為 8 個獨立訊號模組：

### 1. MarketSignalV2 — 大盤多空判定
**輸入**：大盤日 K + MACD（200/209/210）
**輸出**：`BULL / BEAR / ALERT / CRASH`

| 條件 | 加分 |
|------|------|
| MACD 四箭頭全上 | +30 |
| DIF > 0 | +20 |
| 收盤 > N2 | +25 |
| 月 DMI +DI1 > 50 | +15 |
| 連續大跌 3 根 | 強制 CRASH |

### 2. BigStockBuySignalV2 — 大飆股買進（8條件）
| 條件 | 說明 |
|------|------|
| B1 MACD 四箭頭 ≥ 3 | 日線 MACD(200/209/210) 至少 3 箭頭向上 |
| B2 ADX > 20 | 日 DMI300 趨勢強度 > 20（主趨勢確認） |
| B3 威廉 < −20 | 日 WMS%R(50) < −20（未超買） |
| B4 RSI60 > 57 | 日 RSI60 > 57（中長期動能偏強） |
| B5 週 VR ≈ 150 | 週 VR 在 100–200（量能健康） |
| B6 月 VR ≈ 150 | 月 VR 在 100–200（月線量能確認） |
| B7 月 +DI1 > 50 | 月 DMI +DI1 > 50（月線方向向上） |
| B8 月 RSI4 > 77 | 月 RSI4 > 77（月線動能強勁） |

- **STRONG_BUY**：B1+B2+B3 全中，加分條件 ≥ 3
- **BUY**：B1+B2+B3 全中，加分條件 ≥ 1

### 3. BigStockSellSignalV2 — 大飆股賣出
| 條件 | 說明 |
|------|------|
| S1 月威廉 < −5 | 月 WMS%R < −5（月線超買） |
| S2a 月 RSI4 初次跌破 77 | 賣出 50% |
| S2b 月 RSI4 站回後再跌破 77 | 賣出剩餘 50% |
| S3 月 6K/9K | 月下跌型 9K 達標 |
| S4 類股輪動 | 同類股出現更強訊號 |

### 4. EntrySignal — 切入時機
- RSI60 分鐘線（60 分 K）反彈確認
- 收盤進入備戰區（N2 − 100 附近）
- 月線出現黑色 6K（短期回調進場）

### 5. CrashExitSignal — 危機出場
- 日線出現「頂天」型態
- 月線同時出現 6K/9K 下跌型
- 大盤跌破 N2 且 MACD 多箭頭消失

### 6. CapitalAllocator — 資金配置
- 權值股（0050 成份）：可動用 30%
- 中小型飆股：可動用 20%
- 金融股：視月 RSI4 動態調整

---

## 資料流程（`core/data_loader.py`）

```
D:/TWSE-Data/Raw/YYYY/_YYYY_daily.parquet
    ↓ 還原權值（Backward Adjustment）
D:/TWSE-Data/Adjusted/_temp/YYYY.parquet
    ↓ TWSEStockLoader.load_daily(ticker)
pandas.DataFrame [Date, Adj_Open, Adj_High, Adj_Low, Adj_Close, Adj_Volume, CumFactor]
```

**還原公式（Backward Adjustment）**：

從最新交易日往前，每遇到除權息日：
```
CumFactor[t] *= (Close_ExDate / Close_PrevDay)
Adj_Price[t] = Raw_Price[t] × CumFactor[t]
Adj_Volume[t] = Raw_Volume[t] / CumFactor[t]   ← 配股後股數增加，歷史量需放大
```

---

## 爬蟲（`crawler/twse_crawler.py`）

- **資料來源**：TWSE MI_INDEX API（全市場每日 OHLCV）
- **防封 IP**：每次請求間隔 **120–300 秒**（2–5 分鐘）
- **支援中斷續爬**：進度寫入 `_progress_YYYY.json`，任意時間點重啟可接續
- **Retry + Backoff**：失敗最多重試 3 次，每次等待 ×2
- **跳過非交易日**：自動判斷週末 + 台灣國定假日

---

## 回測方法（`crawler/run_backtest.py`）

### 模擬架構
- **資本單位**：每筆固定投入 **10 萬元**（`FIXED_ALLOCATION = 100,000`）
- **各年獨立**：每年各股各自從 10 萬元啟動，年底結算（不複利跨年）
- **報酬率計算**：`avg_return_pct = ΣP&L / (股票數 × 10萬) × 100%`
- **採樣頻率**：每 **30 個交易日**評估一次訊號（非每日，避免過度交易）

### 進場條件（同 `BigStockBuySignalV2`）
```
MACD 四箭頭 ≥ 3  AND  ADX > 20  AND  WMS%R < −20  AND  加分條件 ≥ 1
```
加分條件（4項，達1項可進場）：
- RSI60 > 57
- 週 VR 在 100–200
- 月 VR 在 100–200
- 月 +DI1 > 50 且月 RSI4 > 77

### 出場條件
```
月 RSI4 < 77 → 賣出
```

### 無 Look-ahead Bias 設計
月 RSI4、週/月 VR、月 +DI1 在每個採樣點都只使用**截至當日**的歷史資料：
```python
# 每個採樣點 i：
current_date = dates[i]
m_rsi4 = m_rsi4_series[m_rsi4_series.index <= current_date].iloc[-1]
```

---

## 資料依賴

| 資料 | 路徑 | 說明 |
|------|------|------|
| 原始日線 | `D:/TWSE-Data/Raw/YYYY/` | TWSE 爬蟲下載 |
| 還原後年度 | `D:/TWSE-Data/Adjusted/_temp/YYYY.parquet` | `adjuster.py` 產生 |
| 全量還原 | `D:/TWSE-Data/Adjusted/adjusted_all.parquet` | 完整歷史合併檔 |

> 若更換機器，透過環境變數覆蓋路徑：
> ```bash
> set ADJ_TEMP_DIR=E:/data/Adjusted/_temp
> set BACKTEST_OUT_DIR=docs/data
> python crawler/run_backtest.py
> ```

---

## 執行回測

```bash
# 執行全市場回測（2004–2026，需先有 Adjusted/_temp/YYYY.parquet）
set ADJ_TEMP_DIR=D:/TWSE-Data/Adjusted/_temp
set BACKTEST_OUT_DIR=docs/data
python crawler/run_backtest.py

# 執行全市場掃描（今日訊號）
python core/main.py --all

# 單股回測報表
python core/backtest.py --ticker 2330 --start 2020 --end 2026

# 執行測試
python -m pytest tests/ -v
```

---

## Bug 修復紀錄（vs 原始版本）

| # | 問題 | 修復 |
|---|------|------|
| 1 | **Look-ahead bias** — 月RSI4/週VR 用年末值判斷全年 | 改為逐點計算，每點只用截至當日資料 |
| 2 | **P&L 雙重計算** — 年中賣出與年末結算重複加計 | 統一由 `ticker_cash` 計算，只做一次差額 |
| 3 | **報酬率分母錯誤** — 除以固定 100 萬而非實際部署資本 | 改為 `ΣP&L / (n_tickers × 100k)` |
| 4 | **Volume 還原方向反向** — `× CumFactor`（配股後量減少）| 改為 `÷ CumFactor`（配股後歷史量應放大）|
| 5 | **爬蟲 sleep 過短** — 2–5 秒會觸發 TWSE IP 封鎖 | 改為 120–300 秒 |
| 6 | **S2 二次跌破條件錯誤** — 邏輯判斷月份偏移 | 修正為「前月站回 77 → 本月再跌破」 |
| 7 | **正/負報酬年計算錯誤** — 門檻為 −99.99% | 改為 > 0 / ≤ 0 |
| 8 | **硬碼機器路徑** | 改為環境變數 + 相對路徑 |
| 9 | **HTML XSS** — ticker/signal 未 escape | 加 `html.escape()` |
| 10 | **空測試** — `test_doji_two_as_one` 無 assertion | 補充實際驗證邏輯 |

---

## 授權

本專案僅供個人學習與研究用途，不構成任何投資建議。
